"""Tier 3: what are the tier 2 diagnostics good for? Tested on masked cells where the truth is known.

Use cases U1-U4, thresholds and sensitivities were frozen in step 6 before any output was scored (full
text at tag evaluation-v1-results, src/longitudinal_impute/tier3_calibration.py). A case is one
(mask, seed): E = hidden cells of the model group, V = O minus H (tier2.calibration_cells). The hidden
truth enters only the "truth" row and the tier 1 errors, never a reference.

U1  D1 warns about a zero shift (under-fill / over-fill), U2 D2 as a proxy ranking (+ regret),
U3  the D3 flag, U4 decoys put strictly worse than every real model (reported, no threshold).
Pass: U1 / U2 mean match rate >= PASS_RATE within EACH mask and no case below MIN_CASE_RATE (U2 also
regret <= REGRET_MAX in >= GOOD_CASES cases); U3 gap >= U3_GAP in >= GOOD_CASES cases.
load_mask() checks a set of masked run directories (data, schema, group map, clean tree, model tables,
saved masks = make_mask) before any case is scored.
"""
import hashlib
import itertools
import json

import numpy as np

from .report import ms, na, sha256, signed
from .metrics import cell_errors, read_predictions
from .tier0 import validate_pair
from .tier2 import CONFIGS, DATA, GROUP_MAP, MAIN, OTHER, RESIDUAL, SCHEMA, ZERO_BELOW, calibration_cells, check_cells, diagnose, local_parts, local_support, recorded_table, temporal_support
from ..data import make_mask, read_schema
from ..run import read_models
from . import tier1
from .tier1 import load_run, table_runs
from .report import find_run, shown


REAL = ('hron', 'hron_cap', 'tsagris', 'micoda', 'micoda_other', 'same_day_mean')
DECOYS = ('mean_fill', 'zero_fill', 'marginal_sampler')
TIE = 1e-3
PASS_RATE = .80
MIN_CASE_RATE = .5
REGRET_MAX = .005
U3_GAP = .2
GOOD_CASES = 9
MIN_CASES = 4
PROP_ZERO_BELOW = 1e-3


def case_table(counts, hidden, pred):
    """Values of a model in one case: observed counts, and its predictions on H (pred read on H only)."""
    values = np.array(counts, float, copy=True)
    values[hidden] = np.asarray(pred, float)[hidden]
    return values


def truth_table(counts, eval_cells, ref_cells):
    """The truth row: the true counts on E (and the observed ones on V), nothing else."""
    return np.where(eval_cells | ref_cells, counts, np.nan)


def used(entry, ref):
    return entry['supported'] if ref == 'local' else entry['eval_cells']


def common_targets(diags, ref, key, models):
    """Targets where every listed diagnostic result (truth row and real models) has a value."""
    first = diags[models[0]]['targets']
    return [t for t in first if all(diags[m]['targets'][t][ref][key] is not None for m in models)]


def scalar(diag, ref, key, targets, truth=None, absolute=False):
    """Cell-weighted mean over targets of the value, or of (value - truth's value), or of its |.|."""
    if not targets:
        return None
    x = np.array([diag['targets'][t][ref][key] - (truth['targets'][t][ref][key] if truth else 0.)
                  for t in targets], float)
    w = np.array([used(diag['targets'][t], ref) for t in targets], float)
    if w.sum() <= 0:
        return None
    return float(np.sum(w * (np.abs(x) if absolute else x)) / w.sum())


def scored_cells(support, eval_cells, ref, columns):
    """N x D cells a diagnostic scored: supported E cells (R-local) or E cells (R-pooled) of the columns."""
    cells = np.zeros(eval_cells.shape, bool)
    for j in columns:
        if ref == 'local':
            s = support[j]
            cells[s['rows'][s['supported']], j] = True
        else:
            cells[:, j] = eval_cells[:, j]
    return cells


def errors_on(counts, values, totals, cells):
    """Tier 1 errors (evaluate.cell_errors) on the given cells: groups 'all', 'truth_zero', 'truth_positive'."""
    rows, cols = np.nonzero(cells)
    return cell_errors(counts[rows, cols], values[rows, cols], np.asarray(totals)[rows], np.zeros(len(rows), bool))


def pairwise(diag, error, models, direction=1, tie=TIE):
    """Within one case: does sign(direction * delta diag) match sign(delta error), per pair of models?"""
    pairs = []
    for a, b in itertools.combinations(models, 2):
        if any(v.get(m) is None for v in (diag, error) for m in (a, b)):
            status = 'na'
        else:
            de, dd = error[a] - error[b], direction * (diag[a] - diag[b])
            status = 'tie' if abs(de) < tie else ('match' if dd != 0 and (dd > 0) == (de > 0) else 'mismatch')
        pairs.append((a, b, status))
    count = {s: sum(p[2] == s for p in pairs) for s in ('match', 'mismatch', 'tie', 'na')}
    decided = count['match'] + count['mismatch']
    return {'pairs': pairs, **count, 'rate': count['match'] / decided if decided else None}


def regret(diag, error, models):
    """(model the diagnostic picks: smallest value, ties by order; its error minus the smallest error)."""
    ok = [m for m in models if diag.get(m) is not None and error.get(m) is not None]
    errors = [error[m] for m in models if error.get(m) is not None]
    if not ok:
        return None, None
    chosen = min(ok, key=lambda m: (diag[m], models.index(m)))
    return chosen, float(error[chosen] - min(errors))


def flags(values, facts, eval_cells, ref_cells):
    """(positive, flagged): E cells filled >= 0.5, and those at a day outside T_j (T_j from V only)."""
    positive = np.zeros(eval_cells.shape, bool)
    positive[eval_cells] = values[eval_cells] >= ZERO_BELOW
    flagged = np.zeros_like(positive)
    for j in range(values.shape[1]):
        span = temporal_support(values, facts, ref_cells, j)
        outside = np.ones(len(facts.days), bool) if span is None else (facts.days < span[0]) | (facts.days > span[1])
        flagged[:, j] = positive[:, j] & outside
    return positive, flagged


def flag_precision(truth_zero, flagged):
    """U3 on positive-filled cells: share of truth = 0 among flagged and among not flagged cells."""
    truth_zero, flagged = np.asarray(truth_zero, bool), np.asarray(flagged, bool)
    p_flag = float(truth_zero[flagged].mean()) if flagged.any() else None
    p_rest = float(truth_zero[~flagged].mean()) if (~flagged).any() else None
    return {'flagged': int(flagged.sum()), 'unflagged': int((~flagged).sum()), 'p_flagged': p_flag,
            'p_unflagged': p_rest, 'gap': None if p_flag is None or p_rest is None else p_flag - p_rest}


def decoy_worse(scores, decoy, real=REAL, higher_is_worse=True):
    """True when the decoy's score is strictly worse than every real model's; None when any is missing."""
    if scores.get(decoy) is None or any(scores.get(m) is None for m in real):
        return None
    sign = 1 if higher_is_worse else -1
    return all(sign * (scores[decoy] - scores[m]) > 0 for m in real)


def prop_zero_gap(eval_props, ref_props, weights=None):
    """D1 with zero = c / N < PROP_ZERO_BELOW (sensitivity), same weighting as tier2 zero_gap."""
    eval_props, ref_props = np.asarray(eval_props, float), np.asarray(ref_props, float)
    w = np.ones(len(ref_props)) if weights is None else np.asarray(weights, float)
    if not len(eval_props) or not len(ref_props) or w.sum() <= 0:
        return None
    return float(np.mean(eval_props < PROP_ZERO_BELOW) - np.sum(w * (ref_props < PROP_ZERO_BELOW)) / w.sum())


def d1_proportion(values, facts, eval_cells, ref_cells, support, targets):
    """Per target D1 (R-local and R-pooled) with the proportion threshold; shaped like diagnose()."""
    values = check_cells(values, eval_cells, ref_cells)
    props = values / facts.totals[:, None]
    out = {}
    for j, name in enumerate(targets):
        e, r = eval_cells[:, j], ref_cells[:, j]
        parts = local_parts(values, props, j, support[j])
        out[name] = {'supported': parts[5], 'eval_cells': int(e.sum()),
                     'local': {'d1': prop_zero_gap(parts[1], parts[3], parts[4])},
                     'pooled': {'d1': prop_zero_gap(props[e, j], props[r, j])}}
    return {'targets': out}


def rate_verdict(rates):
    """U1 / U2 rates {mask: [rate of each case or None]} -> per mask summary and 'pass' / 'fail' / 'insufficient'."""
    per = {}
    for mask, values in rates.items():
        got = [v for v in values if v is not None]
        per[mask] = {'cases': len(got), 'mean': float(np.mean(got)) if got else None, 'min': min(got) if got else None}
    if any(v['cases'] < MIN_CASES for v in per.values()):
        return per, 'insufficient'
    ok = all(v['mean'] >= PASS_RATE and v['min'] >= MIN_CASE_RATE for v in per.values())
    return per, 'pass' if ok else 'fail'


def regret_ok(regrets):
    """U2 extra: regret <= REGRET_MAX in at least GOOD_CASES cases (a missing regret does not count)."""
    good = sum(r is not None and r <= REGRET_MAX for r in regrets)
    return good, good >= GOOD_CASES


def u3_verdict(gaps):
    got = [g for g in gaps if g is not None]
    good = sum(g >= U3_GAP for g in got)
    if len(got) < GOOD_CASES:
        return good, 'insufficient'
    return good, 'pass' if good >= GOOD_CASES else 'fail'


# ---------- cases and tables ----------

MASKS, SEEDS, RATE = ('whole_feature', 'random_cell'), ('0', '1', '2', '3', '4'), .05
SHORT = {'whole_feature': 'wf', 'random_cell': 'rc', 'whole_feature_matched': 'wfm'}
MODELS = REAL + DECOYS
RUNS = {MAIN: ({}, 'models_default_a17.toml', ('hron', 'tsagris', 'micoda', 'same_day_mean', *DECOYS)),
        OTHER: ({'micoda': 'micoda_other'}, 'models_micoda_other.toml', ('micoda',)),
        RESIDUAL: ({}, 'models_residual_rule.toml', ('hron_cap',))}
DROPPED = {'tsagris_cap': 'trùng tsagris (`cap` ≡ `prop_other`, step 4–5)',
           'micoda_other_cap': 'gần trùng micoda_other (step 4–5)'}
LABEL = {'pass': 'đạt', 'fail': 'không đạt', 'insufficient': 'không đủ support'}


def load_mask(data, mask, runs_of=None, pair_root=None, seeds=None, target=None, roots=None):
    """Hidden masks, prediction tables {(model, seed): N x D}, tier 1 runs and provenance of one mask.

    roots: directories holding <prefix>_<mask> (find_run). runs_of / pair_root / seeds / target default
    to the exploratory passes main, micoda_other, residual on seeds 0-4."""
    hidden, preds, runs, records, checked = {}, {}, {}, [], 0
    use = tuple(str(s) for s in seeds) if seeds else SEEDS       # case seeds
    validate_pair(pair_root or find_run(roots, f'main_{mask}').parent, mask, seeds or tier1.SEEDS[mask])   # F2
    col = {t: j for j, t in enumerate(data.targets)}
    for prefix, (rename, models_file, tables) in (runs_of or RUNS).items():
        folder = find_run(roots, f'{prefix}_{mask}')
        config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
        expected = read_models(CONFIGS / models_file, list(tables))
        problems = [text for bad, text in (
            (config['data_sha256'] != sha256(DATA), 'data sha256'),
            (config['schema'] != read_schema(SCHEMA), 'schema'),
            ((config.get('group_map') or {}).get('sha256') != sha256(GROUP_MAP), 'group map'),
            ((config.get('git') or {}).get('uncommitted_changes') is not False, 'unclean tree'),
            (any(recorded_table(config, t) != expected[t] for t in tables), f'model tables vs {models_file}'),
            (config['mask'] != mask or config['mask_rate'] != RATE, 'mask kind or rate')) if bad]
        for seed in use:
            saved = np.load(folder / f'mask_{mask}_seed{seed}.npy')
            hidden.setdefault(seed, saved)
            if not np.array_equal(saved, hidden[seed]) or not np.array_equal(saved, make_mask(data, mask, RATE,
                                                                                             int(seed), target)[0]):
                problems.append(f'mask seed {seed}')
        if problems:
            raise ValueError(f'{shown(folder)}: ' + ', '.join(problems) + ' differ')
        status, scores, record, (values, _) = load_run(folder, rename, mask)
        checked += values
        for model, by_seed in table_runs(status, scores).items():
            if model in MODELS:
                runs[model] = {s: by_seed[s] for s in use}
        for row in read_predictions(folder / f'predictions_{mask}.csv.gz'):
            name, seed = rename.get(row['model'], row['model']), str(row['seed'])
            if name in MODELS and seed in use:
                table = preds.setdefault((name, seed), np.full(data.counts.shape, np.nan))
                table[row['row_id'], col[row['target']]] = row['pred_count']
        records.append({'dir': shown(folder), 'config_sha256': record['config_sha256'],
                        'predictions_sha256': record['predictions_sha256'], 'git': (config.get('git') or {}).get(
                            'commit')})
    for (name, seed), table in preds.items():
        if not np.array_equal(np.isfinite(table), hidden[seed]):
            raise ValueError(f'{mask} {name} seed {seed}: long file does not cover exactly the hidden cells')
    masks = {s: hashlib.sha256(np.ascontiguousarray(h).tobytes()).hexdigest() for s, h in hidden.items()}
    return hidden, preds, runs, {'runs': records, 'mask_sha256': masks, 'regression_values_checked': checked}


def scalars(diags, ref, key, models, data, support, cells, values, truth_key='truth'):
    """Common targets, scored cells, per-model distance / raw values and tier 1 errors on those cells."""
    # Real models that failed in this case are absent: their pairs become NA (step 6 rule), not an error.
    common = common_targets(diags, ref, key, [truth_key, *(m for m in REAL if m in diags)])
    scored = scored_cells(support, cells['eval'], ref, [data.targets.index(t) for t in common])
    out = {'targets': len(common), 'cells': int(scored.sum()), 'dist': {}, 'raw': {}, 'abs_raw': {}, 'err': {}}
    for m in models:
        ok = all(diags[m]['targets'][t][ref][key] is not None for t in common) and common
        out['dist'][m] = scalar(diags[m], ref, key, common, diags[truth_key]) if ok else None
        out['raw'][m] = scalar(diags[m], ref, key, common) if ok else None
        out['abs_raw'][m] = scalar(diags[m], ref, key, common, absolute=True) if ok else None
        out['err'][m] = errors_on(data.counts, values[m], data.totals, scored) if common else None
    return out


def mae(err, group='all'):
    return {m: (None if e is None else e[group]['mae_proportion']) for m, e in err.items()}


def u_pair(d, direction, group, use='dist', absolute=False):
    diag = {m: (None if v is None else abs(v) if absolute else v) for m, v in d[use].items()}
    return pairwise(diag, mae(d['err'], group), list(REAL), direction)


def one_case(data, facts, hidden, preds, runs, seed):
    cells = calibration_cells(data, hidden)
    support = local_support(facts, cells['eval'], cells['ref'])
    targets = list(data.targets)
    ok = [m for m in MODELS if runs.get(m, {}).get(seed, {}).get('eligible') and (m, seed) in preds]
    values = {m: case_table(data.counts, hidden, preds[m, seed]) for m in ok}
    values['truth'] = truth_table(data.counts, cells['eval'], cells['ref'])
    diags = {m: diagnose(v, facts, cells['eval'], cells['ref'], support, targets) for m, v in values.items()}
    out = {'seed': seed, 'models_ok': ok, 'E': int(cells['eval'].sum()), 'H': int(hidden.sum()),
           'supported': int(sum(s['supported'].sum() for s in support.values())), 'diag': {}, 'u': {}}
    for m in ok + ['truth']:
        s = diags[m]['summary']
        out['diag'][m] = {f'{r}_{d}': s[f'{r}_{d}']['weighted'] for r in ('local', 'pooled') for d in ('d1', 'd2')}
        out['diag'][m]['d3'] = diags[m]['d3_cells']
    for ref in ('local', 'pooled'):
        d1 = scalars(diags, ref, 'd1', ok, data, support, cells, values)
        d2 = scalars(diags, ref, 'd2', ok, data, support, cells, values)
        full = {m: runs[m][seed]['mae_prop'] for m in ok}
        u = {'d1': d1, 'd2': d2, 'u1_under': u_pair(d1, 1, 'truth_positive'),
             'u1_over': u_pair(d1, -1, 'truth_zero'), 'u2': u_pair(d2, 1, 'all', absolute=True),
             'u2_raw': u_pair(d2, 1, 'all', 'raw'), 'u2_full': pairwise(
                 {m: abs(v) if v is not None else None for m, v in d2['dist'].items()}, full, list(REAL))}
        u['regret'] = regret({m: abs(v) for m, v in d2['dist'].items() if v is not None}, mae(d2['err']),
                                list(REAL))
        u['regret_raw'] = regret({m: v for m, v in d2['raw'].items() if v is not None}, mae(d2['err']),
                                    list(REAL))
        u['regret_full'] = regret({m: abs(v) for m, v in d2['dist'].items() if v is not None}, full,
                                     list(REAL))
        out['u'][ref] = u
    prop = {m: d1_proportion(values[m], facts, cells['eval'], cells['ref'], support, targets)
            for m in [*REAL, 'truth'] if m in values}
    if any(m in prop for m in REAL):
        present = [m for m in REAL if m in prop]
        out['u']['prop'] = {'d1': (p := scalars(prop, 'local', 'd1', present, data, support, cells, values))}
        out['u']['prop'].update(u1_under=u_pair(p, 1, 'truth_positive'), u1_over=u_pair(p, -1, 'truth_zero'))
    flagged = {m: flags(values[m], facts, cells['eval'], cells['ref']) for m in ok}
    real = [m for m in REAL if m in flagged]
    zero = data.counts == 0
    out['u3'] = flag_precision(np.concatenate([zero[flagged[m][0]] for m in real]),
                                  np.concatenate([flagged[m][1][flagged[m][0]] for m in real]))
    out['u3_models'] = {m: flag_precision(zero[p], f[p]) for m, (p, f) in flagged.items()}
    local = out['u']['local']
    scores = {'d1_dist': {m: None if v is None else abs(v) for m, v in local['d1']['dist'].items()},
              'd1_abs_raw': local['d1']['abs_raw'], 'd2_dist': {
                  m: None if v is None else abs(v) for m, v in local['d2']['dist'].items()},
              'd2_raw': local['d2']['raw'], 'd3_rate': {m: f[1].sum() / cells['eval'].sum()
                                                        for m, f in flagged.items()}}
    scores['tier1_mae'] = {m: runs[m][seed]['mae_prop'] for m in ok}     # reference row, not a diagnostic
    out['u4'] = {name: {d: decoy_worse(s, d) for d in DECOYS} for name, s in scores.items()}
    out['u4_scores'] = {name: {m: None if v is None else float(v) for m, v in s.items()} for name, s in scores.items()}
    return out, cells, support


U4_LABELS = {'d1_dist': '\\|D1 − D1_truth\\|', 'd1_abs_raw': '\\|D1\\| thô', 'd2_dist': '\\|D2 − D2_truth\\|',
             'd2_raw': 'D2 thô', 'd3_rate': 'D3: ô cờ / ô E', 'tier1_mae': '*tham chiếu: MAE Tầng 1, toàn ô model*'}


def rate_cell(p):
    return 'NA' if p['rate'] is None else f"{p['rate']:.2f} ({p['match']}/{p['match'] + p['mismatch']}" + (
        f", hòa {p['tie']}" if p['tie'] else '') + (f", NA {p['na']}" if p['na'] else '') + ')'


def verdict_lines(cases, key, ref='local'):
    rates = {mask: [c['u'][ref][key]['rate'] if key in c['u'].get(ref, {}) else None for c in cases[mask]]
             for mask in cases}
    per, verdict = rate_verdict(rates)
    text = '; '.join(f"{mask}: TB {na(v['mean'], 2)}, min {na(v['min'], 2)} ({v['cases']} ca)" for mask, v in per.items())
    return text, verdict


def u1_table(cases, ref='local', block='u1'):
    keys = ('u1_under', 'u1_over')
    out = ['| Mask | Seed | thiếu: khớp (khớp/quyết định) | thừa: khớp |', '|---|---:|---|---|']
    for mask in cases:
        for c in cases[mask]:
            u = c['u'].get(ref, {})
            out.append(f"| {mask} | {c['seed']} | " + ' | '.join(rate_cell(u[k]) if k in u else 'NA' for k in keys) + ' |')
    verdicts = {}
    for key, label in zip(keys, ('thiếu (D1 cao ↔ MAE ô đáp án > 0)', 'thừa (D1 thấp ↔ MAE ô đáp án = 0)')):
        text, verdicts[key] = verdict_lines(cases, key, ref)
        out.append(f'- U1 {label}: {text} → **{LABEL[verdicts[key]]}**.')
    return out, verdicts


def u2_table(cases, ref='local'):
    out = ['| Mask | Seed | khớp \\|D2 − D2t\\| | chọn → regret | khớp D2 thô | regret thô | khớp, MAE toàn ô model | '
           'regret, MAE toàn ô model |', '|---|---:|---|---|---|---:|---|---:|']
    regrets = []
    for mask in cases:
        for c in cases[mask]:
            u = c['u'][ref]
            regrets.append(u['regret'][1])
            out.append(f"| {mask} | {c['seed']} | {rate_cell(u['u2'])} | {u['regret'][0]} → {na(u['regret'][1])} | "
                       f"{rate_cell(u['u2_raw'])} | {na(u['regret_raw'][1])} | {rate_cell(u['u2_full'])} | "
                       f"{na(u['regret_full'][1])} |")
    text, verdict = verdict_lines(cases, 'u2', ref)
    good, enough = regret_ok(regrets)
    final = verdict if verdict != 'pass' else ('pass' if enough else 'fail')
    out.append(f'- U2: {text}; regret ≤ {REGRET_MAX} ở {good}/10 ca → **{LABEL[final]}**.')
    return out, final


def u3_table(cases):
    out = ['| Mask | Seed | ô dương bị cờ | tỷ lệ đáp án 0 khi bị cờ | ô dương không cờ | khi không cờ | chênh |',
           '|---|---:|---:|---:|---:|---:|---:|']
    gaps = []
    for mask in cases:
        for c in cases[mask]:
            u = c['u3']
            gaps.append(u['gap'])
            out.append(f"| {mask} | {c['seed']} | {u['flagged']} | {na(u['p_flagged'], 3)} | {u['unflagged']} | "
                       f"{na(u['p_unflagged'], 3)} | {signed(u['gap'])} |")
    good, verdict = u3_verdict(gaps)
    out.append(f'- U3: chênh ≥ {U3_GAP} ở {good}/10 ca → **{LABEL[verdict]}**.')
    return out, verdict


def u4_table(cases):
    names = list(next(iter(cases.values()))[0]['u4'])
    out = ['Số ca diagnostic xếp đối thủ giả kém hơn **mọi** model thật (trên số ca có giá trị), R-local.', '',
           '| Đối thủ giả | Mask | ' + ' | '.join(U4_LABELS[n] for n in names) + ' |', '|---|---|' + ':---:|' * len(names)]
    for d in DECOYS:
        for mask in cases:
            cells = []
            for n in names:
                got = [c['u4'][n][d] for c in cases[mask] if c['u4'][n][d] is not None]
                cells.append(f'{sum(got)}/{len(got)}' if got else 'NA')
            out.append(f'| {d} | {mask} | ' + ' | '.join(cells) + ' |')
    out += ['', 'Giá trị trung bình qua 5 seed (R-local, trên target chung của model thật):', '',
            '| Model | Mask | ' + ' | '.join(U4_LABELS[n] for n in names) + ' |', '|---|---|' + '---:|' * len(names)]
    for m in MODELS:
        for mask in cases:
            out.append(f'| {m} | {mask} | ' + ' | '.join(
                ms([c['u4_scores'][n].get(m) for c in cases[mask]], 4) for n in names) + ' |')
    return out


def sensitivity_lines(cases):
    out, v = ['**D1 với ngưỡng zero c/N < 0.001 (R-local)**', ''], {}
    lines, v['prop'] = u1_table(cases, 'prop')
    out += lines + ['', '**D1 với R-pooled**', '']
    lines, v['pooled_u1'] = u1_table(cases, 'pooled')
    out += lines + ['', '**D2 với R-pooled**', '']
    lines, v['pooled_u2'] = u2_table(cases, 'pooled')
    return out + lines, v
