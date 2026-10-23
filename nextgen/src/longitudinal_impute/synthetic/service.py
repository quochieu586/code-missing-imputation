"""Strict specifications and shared Python mask application/export service."""
from dataclasses import dataclass
from datetime import date
from enum import Enum
import csv
import hashlib
import json
import numpy as np
from .backend import validate_seed
from .capabilities import NotReadyError, capabilities
from .masking import mask_counts, mask_fixed_wave


class Scenario(str, Enum):
    DECLARED_PATTERNS = 'declared_patterns'
    FIXED_WAVE = 'fixed_wave'


@dataclass(frozen=True)
class MaskSpec:
    scenario: Scenario
    mechanism: str
    case_probability: float
    seed: int
    n_weight: float = 1.
    trait_weight: float = 1.
    patterns: tuple = ()
    frequencies: tuple = ()
    wave_time: int | None = None

    @classmethod
    def parse(cls, value, *, components=None):
        if isinstance(value, cls):
            value = value.to_dict()
        if not isinstance(value, dict):
            raise ValueError('Mask spec must be an object')
        if value.get('mechanism') == 'MNAR':
            raise NotReadyError('NOT-READY: latent_p_mnar')
        blocked = {'beta', 'cell_rate', 'exact_rate', 'block_length'}
        if blocked & value.keys():
            raise NotReadyError(f'NOT-READY spec fields: {sorted(blocked & value.keys())}')
        scenario = value.get('scenario')
        if scenario in ('latent_p_mnar', 'random_temporal_blocks', 'whole_component_within_entity'):
            raise NotReadyError(f'NOT-READY: {scenario}')
        try:
            scenario = Scenario(scenario)
        except (ValueError, TypeError) as exc:
            raise ValueError(f'Unsupported mask scenario: {scenario}') from exc
        required = {'scenario', 'mechanism', 'case_probability', 'seed'}
        required |= {'patterns', 'frequencies'} if scenario == Scenario.DECLARED_PATTERNS else {'wave_time'}
        allowed = required | {'n_weight', 'trait_weight'}
        if required - value.keys() or value.keys() - allowed:
            raise ValueError(f'Invalid spec fields; missing={sorted(required-value.keys())}, unknown={sorted(value.keys()-allowed)}')
        if value['mechanism'] not in ('MCAR', 'MAR'):
            raise ValueError('mechanism must be MCAR or MAR')
        validate_seed(value['seed'])
        for key in ('case_probability', 'n_weight', 'trait_weight'):
            number = value.get(key, 1.)
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not np.isfinite(number):
                raise ValueError(f'{key} must be a finite number')
        probability = value['case_probability']
        nw, tw = value.get('n_weight', 1.), value.get('trait_weight', 1.)
        if not 0 < probability < 1 or min(nw, tw) < 0:
            raise ValueError('Invalid probability or covariate weights')
        if value['mechanism'] == 'MAR' and nw == tw == 0:
            raise ValueError('MAR needs a nonzero observed-covariate weight')
        patterns, frequencies, wave = (), (), None
        if scenario == Scenario.DECLARED_PATTERNS:
            pat = np.asarray(value['patterns'])
            freq = np.asarray(value['frequencies'], dtype=float)
            if (pat.ndim != 2 or not all(pat.shape) or
                (components is not None and pat.shape[1] != components) or
                not np.isin(pat, [0, 1]).all() or not (pat == 0).any(axis=1).all()):
                raise ValueError('patterns must be K x D binary rows, each hiding a target')
            if freq.shape != (len(pat),) or not np.isfinite(freq).all() or (freq <= 0).any() or not np.isclose(freq.sum(), 1):
                raise ValueError('Positive frequencies must match patterns and sum to one')
            patterns = tuple(tuple(row) for row in pat.tolist())
            frequencies = tuple(freq.tolist())
        else:
            wave = value['wave_time']
            if isinstance(wave, str):
                wave = date.fromisoformat(wave).toordinal()
            if isinstance(wave, bool) or not isinstance(wave, int):
                raise ValueError('wave_time must be an ISO date or ordinal integer')
            date.fromordinal(wave)
        return cls(scenario, value['mechanism'], probability, value['seed'], nw, tw, patterns, frequencies, wave)

    def to_dict(self):
        result = dict(scenario=self.scenario.value, mechanism=self.mechanism,
                      case_probability=self.case_probability, seed=self.seed,
                      n_weight=self.n_weight, trait_weight=self.trait_weight)
        if self.scenario == Scenario.DECLARED_PATTERNS:
            result.update(patterns=self.patterns, frequencies=self.frequencies)
        else:
            result['wave_time'] = self.wave_time
        return result


def apply_mask(raw, spec, *, rscript=None):
    spec = MaskSpec.parse(spec, components=raw.counts.shape[1])
    kwargs = {key: getattr(spec, key) for key in ('mechanism', 'case_probability', 'seed', 'n_weight', 'trait_weight')}
    if spec.scenario == Scenario.DECLARED_PATTERNS:
        return mask_counts(raw, patterns=spec.patterns, frequencies=spec.frequencies, rscript=rscript, **kwargs)
    if spec.scenario == Scenario.FIXED_WAVE:
        return mask_fixed_wave(raw, wave_time=spec.wave_time, rscript=rscript, **kwargs)
    raise ValueError('Unsupported scenario')


def write_mask(output, raw, result, spec, *, input_sha256, schema_bytes, source_rows=None):
    """Write inside caller-owned staging; transport the boolean mask in memory."""
    schema = raw.schema
    fields = [schema.entity_column, schema.time_column, schema.total_column, *schema.target_columns]
    if 'entity_trait_01' in raw.metadata:
        fields.append('entity_trait_01')
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f'Mask staging is not empty: {output}')
    with (output/'conditioning.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(fields)
        for i in range(len(raw.row_ids)):
            row = [str(raw.entity_ids[i]), date.fromordinal(int(raw.timestamps[i])).isoformat(), format(raw.totals[i], '.17g')]
            row += ['' if result.hidden[i,j] else format(raw.counts[i,j], '.17g') for j in range(raw.counts.shape[1])]
            if 'entity_trait_01' in fields:
                row.append(str(raw.metadata['entity_trait_01'][i]))
            if source_rows is not None:
                row = [source_rows[i][field] if field not in schema.target_columns or not result.hidden[i,schema.target_columns.index(field)] else '' for field in fields]
            writer.writerow(row)
    np.savetxt(output/'hidden_mask.csv', result.hidden.astype(int), delimiter=',', fmt='%d')
    manifest = dict(status='COMPLETED', spec=spec.to_dict(), diagnostics=result.diagnostics,
                    capabilities=capabilities(), row_ids=list(raw.row_ids), input_sha256=input_sha256,
                    schema_sha256=hashlib.sha256(schema_bytes).hexdigest(), conditioning_columns=fields,
                    private_truth_exported=False)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
