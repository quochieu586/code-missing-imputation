"""Domain-neutral generator; explicit benchmark assumptions, not a paper model."""
from dataclasses import dataclass, asdict
from datetime import date, timedelta
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from .backend import validate_seed
from .capabilities import capabilities

PROFILES = {
    'easy': dict(offset_sd=1., dominance_sd=.10, noise_sd=.05, mean_n=2000., cv_n=.30),
    'medium': dict(offset_sd=4., dominance_sd=.35, noise_sd=.15, mean_n=500., cv_n=.60),
    'hard': dict(offset_sd=8., dominance_sd=.70, noise_sd=.35, mean_n=100., cv_n=1.),
}
SCALES = {'small': 30, 'standard': 60, 'large': 120}

@dataclass(frozen=True)
class GeneratorConfig:
    profile: str = 'easy'
    scale: str = 'standard'
    seed: int = 2026
    timepoints: int = 120
    components: int = 12
    temporal_correlation: float = .8
    wave_width: float = 8.
    onset_lead: float = 16.
    tail_weight: float = .0001
    residual_mean: float = .10
    residual_sd: float = .03
    train_fraction: float = .6
    validation_fraction: float = .2

    def validate(self):
        validate_seed(self.seed)
        if self.profile not in PROFILES or self.scale not in SCALES:
            raise ValueError('Unknown profile or scale')
        if self.timepoints != 120 or self.components != 12:
            raise ValueError('Approved scales keep 120 times and 12 components fixed')
        if not 0 <= self.temporal_correlation < 1 or self.wave_width <= 0 or self.onset_lead <= 0:
            raise ValueError('Invalid temporal/wave parameters')
        if not np.isfinite([self.temporal_correlation, self.wave_width, self.onset_lead,
                            self.tail_weight, self.residual_mean, self.residual_sd,
                            self.train_fraction, self.validation_fraction]).all():
            raise ValueError('Parameters must be finite')
        if not 0 < self.tail_weight < 1 or not 0 < self.residual_mean < 1:
            raise ValueError('Tail/residual parameters out of range')
        if not self.residual_sd > 0 or not 0 < self.residual_sd**2 < self.residual_mean*(1-self.residual_mean):
            raise ValueError('Residual moments do not define a beta distribution')
        if min(self.train_fraction, self.validation_fraction) <= 0 or self.train_fraction+self.validation_fraction >= 1:
            raise ValueError('All entity split fractions must be positive')


def _ar1(rng, shape, correlation):
    z = rng.standard_normal(shape)
    for t in range(1, shape[1]):
        z[:, t] = correlation*z[:, t-1] + np.sqrt(1-correlation**2)*z[:, t]
    return z


def generate(config=GeneratorConfig()):
    """Return evaluator-owned arrays. Never pass this truth dict to a model."""
    config.validate()
    e, t, d = SCALES[config.scale], config.timepoints, config.components
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(config.seed).spawn(9)]
    settings = PROFILES[config.profile]
    shifts = streams[0].standard_normal(e)*settings['offset_sd']
    dominance = streams[1].standard_normal((e, d))*settings['dominance_sd']
    noise = _ar1(streams[2], (e, t, d), config.temporal_correlation)*settings['noise_sd']
    peaks = np.linspace(0., t-1., d)[None, :] + shifts[:, None]
    onsets = peaks-config.onset_lead
    onsets[:, 0] = np.minimum(onsets[:, 0], 0.)  # initial component ensures valid composition
    clock = np.arange(t)[None, :, None]
    delta = clock-peaks[:, None, :]
    active = clock >= onsets[:, None, :]
    weights = np.where(active, (np.exp(-.5*(delta/config.wave_width)**2)
                       + config.tail_weight*(delta >= 0))*np.exp(dominance[:, None, :]+noise), 0.)
    concentration = config.residual_mean*(1-config.residual_mean)/config.residual_sd**2-1
    residual = streams[3].beta(config.residual_mean*concentration,
                              (1-config.residual_mean)*concentration, (e,t))
    p = weights/weights.sum(axis=2, keepdims=True)*(1-residual[:, :, None])
    sigma = np.sqrt(np.log1p(settings['cv_n']**2))
    z_entity = streams[4].standard_normal((e,1))
    z_time = _ar1(streams[5], (e,t), config.temporal_correlation)
    totals = np.maximum(1, np.rint(settings['mean_n']*np.exp(sigma*(z_entity+z_time)/np.sqrt(2)-sigma**2/2))).astype(np.int64)
    all_p = np.concatenate([p, residual[:, :, None]], axis=2)
    counts = np.array([streams[6].multinomial(int(n), probs) for n, probs in
                       zip(totals.ravel(), all_p.reshape(-1,d+1))]).reshape(e,t,d+1)
    trait = streams[7].standard_normal(e)
    order = streams[8].permutation(e)
    ntrain, nval = int(e*config.train_fraction), int(e*config.validation_fraction)
    if min(ntrain,nval,e-ntrain-nval) < 1: raise ValueError('Empty entity split')
    split = np.full(e, 'test', dtype='<U10'); split[order[:ntrain]]='train'; split[order[ntrain:ntrain+nval]]='validation'
    strata = np.where(p == 0, 0, np.where(counts[:, :, :d] == 0, 1, 2)).astype(np.uint8)
    return dict(p=p, residual_p=residual, counts=counts[:, :, :d], residual_counts=counts[:,:,d],
                N=totals, entity_trait_01=trait, split=split, strata=strata,
                onsets=onsets, peaks=peaks, entity_shifts=shifts)


def _write_csv(path, fields, rows):
    with path.open('x', encoding='utf-8', newline='') as f:
        w=csv.writer(f); w.writerow(fields); w.writerows(rows)


def _strata_summary(strata, hidden=None):
    select=np.ones(strata.shape, dtype=bool) if hidden is None else hidden
    return {name: int(((strata == k)&select).sum()) for k,name in
            enumerate(('true_zero','sampling_zero','positive_count'))}


def export_dataset(output, config=GeneratorConfig(), *, mask_specs=(), rscript=None):
    """Versioned bundle compatible with csv_io, RawPanel and ConditioningPanel."""
    from .service import MaskSpec
    from .publication import staged_directory
    config.validate()
    specs = [MaskSpec.parse(s, components=config.components) for s in mask_specs]
    with staged_directory(output) as staging:
        return _write_dataset(staging, config, specs, rscript)


def _write_dataset(out, config, mask_specs, rscript):
    from .service import apply_mask, write_mask
    from ..contracts import RawPanel
    from ..baselines.csv_io import schema_from_config
    import tomllib
    truth = generate(config)
    private=out/'private'; private.mkdir()
    e,t,d=truth['counts'].shape
    entities=[f'entity_{i+1:03d}' for i in range(e)]
    dates=[(date(2000,1,1)+timedelta(days=i)).isoformat() for i in range(t)]
    targets=[f'component_{j+1:02d}' for j in range(d)]
    fields=['entity_id','date','N',*targets,'entity_trait_01']
    _write_csv(private/'complete_counts.csv', fields,
               ([entities[i], dates[k], int(truth['N'][i,k]),
                 *truth['counts'][i,k].tolist(), format(truth['entity_trait_01'][i],'.17g')]
                for i in range(e) for k in range(t)))
    row_ids = tuple(json.dumps([entity, day], separators=(',', ':')) for entity in entities for day in dates)
    np.savez_compressed(private/'ground_truth.npz', row_ids=np.asarray(row_ids).reshape(e,t), **truth)
    _write_csv(out/'entities.csv', ['entity_id','split'], zip(entities,truth['split']))
    schema='[dataset]\nentity = "entity_id"\ntime = "date"\ntotal = "N"\ntargets = '+json.dumps(targets)+'\nmetadata_candidates = ["entity_trait_01"]\n'
    (out/'schema.toml').write_text(schema,encoding='utf-8')
    schema_bytes = (out/'schema.toml').read_bytes()
    panel_schema = schema_from_config(tomllib.loads(schema))
    raw = RawPanel(row_ids, truth['counts'].reshape(-1,d).astype(float), np.ones((e*t,d),dtype=bool),
                   truth['N'].ravel().astype(float), np.repeat(entities,t),
                   np.tile([date.fromisoformat(day).toordinal() for day in dates],e),
                   {'entity_trait_01': np.repeat(truth['entity_trait_01'],t)}, panel_schema)
    mask_results=[]
    for i,spec in enumerate(mask_specs):
        name=f'{i+1:02d}_{spec.mechanism.lower()}_{spec.scenario.value}'
        specs=out/'mask_specs'; specs.mkdir(exist_ok=True)
        (specs/f'{name}.json').write_text(json.dumps(spec.to_dict(),indent=2),encoding='utf-8')
        result = apply_mask(raw, spec, rscript=rscript)
        write_mask(out/'conditioning'/name, raw, result, spec,
                   input_sha256=hashlib.sha256((private/'complete_counts.csv').read_bytes()).hexdigest(),
                   schema_bytes=schema_bytes)
        hidden=result.hidden.reshape(e,t,d)
        mask_results.append(dict(name=name,hidden_strata=_strata_summary(truth['strata'],hidden),
                                 by_split={s:_strata_summary(truth['strata'],hidden & (truth['split']==s)[:,None,None])
                                           for s in ('train','validation','test')}))
    manifest=dict(format_version='synthetic-panel-v2',status='COMPLETE',config=asdict(config),
                  profile_parameters=PROFILES[config.profile], dimensions=dict(entities=e,timepoints=t,components=d),
                  row_order='entity-major then ascending date; npz axes entity,time,component',
                  truth_access='private/ is evaluator-only; models load conditioning/ only',
                  strata_codes={'0':'true_zero','1':'sampling_zero','2':'positive_count'},
                  diagnostics=dict(mean_n=float(truth['N'].mean()),cv_n=float(truth['N'].std()/truth['N'].mean()),
                                   residual_mean=float(truth['residual_p'].mean()),residual_sd=float(truth['residual_p'].std()),
                                   strata=_strata_summary(truth['strata'])), masks=mask_results,capabilities=capabilities(),
                  numpy_version=np.__version__, generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  assumptions='Benchmark generator choices; not equations or validated difficulty ordering from Schouten2018.')
    manifest['files_sha256']={str(p.relative_to(out)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(out.rglob('*')) if p.is_file()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest
