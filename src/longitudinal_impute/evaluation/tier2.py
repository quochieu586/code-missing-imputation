"""Tier 2: diagnostics on the real missing cells (D1, D2, D3) and their tables. Descriptive, NOT accuracy.

D1 zero-rate gap, D2 W1 on the positive part, D3 flag outside the observed temporal support, each
against an explicit reference: R-local (same continent, +-28 days, >= 20 cells over >= 3 locations)
and R-pooled. Every function takes two explicit cell masks and never reads NaN itself:

    E (eval_cells)  cells whose filled values are described. Native: M_m, the real missing
                    cells not forced to 0. Masked (calibration, tier 3): hidden cells H of the
                    model group.
    V (ref_cells)   cells whose values form the reference. Native: O, the observed cells.
                    Masked: O minus H. Hidden truth never enters V.

The definitions were frozen in step 5 before any output was scored (full text at tag
evaluation-v1-results, src/longitudinal_impute/tier2_diagnostics.py). summarize() writes the native
tables from the native completed CSVs of the exploratory runs (roots) and, with `rerun`, checks that
another run of the same models gives the same CSV bytes.
"""
import csv
from dataclasses import dataclass
from datetime import date, timedelta
import json
from pathlib import Path

import numpy as np

from .report import na, sha256, signed, update_report
from .tier0 import micoda_other
from ..data import check_completed, load_csv, read_schema
from ..run import read_models
from .metrics import forced_zero
from .report import ROOT, find_run, shown, write_csv


ZERO_BELOW = .5
WINDOW_DAYS = 28
MIN_CELLS = 20
MIN_LOCATIONS = 3
MIN_STRATUM_CELLS = 30
MATERIAL_CHANGE = .10       # tier 2 table 6: |D1 local - D1 pooled| at least this is "large"
N_LABELS = ('N1', 'N2', 'N3')


@dataclass(frozen=True)
class RowFacts:
    """Row facts known whatever the mask: no counts, so no NaN can be read through them."""
    totals: np.ndarray
    days: np.ndarray
    entities: np.ndarray
    groups: np.ndarray


def row_facts(data):
    if data.days is None or data.time_kind != 'date' or data.groups is None:
        raise ValueError('tier 2 diagnostics need a date column and a group column (continent)')
    return RowFacts(np.asarray(data.totals, float), np.asarray(data.days), np.asarray(data.entities),
                    np.asarray(data.groups))


def native_cells(data):
    """Native split of the real missing cells: M, M_f (forced to 0), M_m = M minus M_f, and O."""
    missing = ~np.isfinite(data.counts)
    forced = forced_zero(data, np.zeros_like(missing)) & missing
    return {'missing': missing, 'forced': forced, 'eval': missing & ~forced, 'ref': ~missing}


def calibration_cells(data, hidden):
    """Masked run: E = hidden cells of the model group, V = observed cells left visible."""
    observed = np.isfinite(data.counts)
    if hidden.shape != observed.shape or (hidden & ~observed).any():
        raise ValueError('hidden must be a subset of the observed cells')
    return {'eval': hidden & ~forced_zero(data, hidden), 'ref': observed & ~hidden}


def check_cells(values, eval_cells, ref_cells):
    values = np.asarray(values, float)
    if eval_cells.shape != values.shape or ref_cells.shape != values.shape:
        raise ValueError('values, eval_cells and ref_cells must have the same shape')
    if eval_cells.dtype != bool or ref_cells.dtype != bool or (eval_cells & ref_cells).any():
        raise ValueError('eval_cells and ref_cells must be disjoint boolean masks')
    used = eval_cells | ref_cells
    if not np.isfinite(values[used]).all() or (values[used] < 0).any():
        raise ValueError('values must be finite and nonnegative on eval_cells and ref_cells')
    return values


def weighted_wasserstein(a, b, wa=None, wb=None):
    """W1 between two weighted samples (equal weights when None); None when a side is empty."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    wa = np.ones(len(a)) if wa is None else np.asarray(wa, float)
    wb = np.ones(len(b)) if wb is None else np.asarray(wb, float)
    if not len(a) or not len(b) or wa.sum() <= 0 or wb.sum() <= 0:
        return None
    edges = np.unique(np.r_[a, b])

    def cdf(x, w):
        order = np.argsort(x, kind='stable')
        cumulative = np.r_[0., np.cumsum(w[order])] / w.sum()
        return cumulative[np.searchsorted(x[order], edges[:-1], side='right')]
    return float(np.sum(np.abs(cdf(a, wa) - cdf(b, wb)) * np.diff(edges)))


def zero_gap(eval_counts, ref_counts, ref_weights=None):
    """D1: share of zero (count < ZERO_BELOW) in E minus the (weighted) share in the reference."""
    eval_counts, ref_counts = np.asarray(eval_counts, float), np.asarray(ref_counts, float)
    weights = np.ones(len(ref_counts)) if ref_weights is None else np.asarray(ref_weights, float)
    if not len(eval_counts) or not len(ref_counts) or weights.sum() <= 0:
        return None, None, None
    hat = float(np.mean(eval_counts < ZERO_BELOW))
    ref = float(np.sum(weights * (ref_counts < ZERO_BELOW)) / weights.sum())
    return hat - ref, hat, ref


def positive_w1(eval_counts, eval_props, ref_counts, ref_props, ref_weights=None):
    """D2: W1 of p between the positive E cells and the positive (weighted) reference cells."""
    keep_e, keep_r = np.asarray(eval_counts) >= ZERO_BELOW, np.asarray(ref_counts) >= ZERO_BELOW
    weights = None if ref_weights is None else np.asarray(ref_weights, float)[keep_r]
    return weighted_wasserstein(np.asarray(eval_props)[keep_e], np.asarray(ref_props)[keep_r], None, weights)


def local_support(facts, eval_cells, ref_cells, window=WINDOW_DAYS, min_cells=MIN_CELLS,
                  min_locations=MIN_LOCATIONS):
    """R-local structure per target; depends on the masks and row facts only, never on values.

    {j: {'rows': rows of the E cells of j, 'n_ref', 'n_locations', 'supported' (per E cell),
         'owner' (index into rows), 'ref_rows', 'weight' (1 / |S_e|), one entry per (e, r in S_e),
         supported e only}}
    """
    _, entity = np.unique(facts.entities, return_inverse=True)
    days = np.asarray(facts.days)
    out = {}
    for j in range(eval_cells.shape[1]):
        rows = np.nonzero(eval_cells[:, j])[0]
        n_ref, n_loc = np.zeros(len(rows), int), np.zeros(len(rows), int)
        owners, refs = [], []
        pools = {}
        for g in np.unique(facts.groups[rows]):
            pool = np.nonzero(ref_cells[:, j] & (facts.groups == g))[0]
            pool = pool[np.argsort(days[pool], kind='stable')]
            pools[g] = (pool, days[pool])
        for k, row in enumerate(rows):
            pool, pool_days = pools[facts.groups[row]]
            lo = np.searchsorted(pool_days, days[row] - window, side='left')
            hi = np.searchsorted(pool_days, days[row] + window, side='right')
            near = pool[lo:hi]
            near = near[entity[near] != entity[row]]
            n_ref[k], n_loc[k] = len(near), len(np.unique(entity[near]))
            if n_ref[k] >= min_cells and n_loc[k] >= min_locations:
                owners.append(np.full(len(near), k))
                refs.append(np.sort(near))
        supported = (n_ref >= min_cells) & (n_loc >= min_locations)
        owner = np.concatenate(owners) if owners else np.zeros(0, int)
        ref_rows = np.concatenate(refs) if refs else np.zeros(0, int)
        out[j] = {'rows': rows, 'n_ref': n_ref, 'n_locations': n_loc, 'supported': supported,
                  'owner': owner, 'ref_rows': ref_rows, 'weight': 1. / n_ref[owner] if len(owner) else
                  np.zeros(0)}
    return out


def n_strata(facts, eval_cells):
    """N tertile label of every row, cut on the rows holding an E cell; and the cut points."""
    rows = eval_cells.any(axis=1)
    if not rows.any():
        return np.full(len(facts.totals), ''), (None, None)
    q1, q2 = (float(q) for q in np.quantile(facts.totals[rows], [1 / 3, 2 / 3]))
    label = np.where(facts.totals <= q1, N_LABELS[0], np.where(facts.totals <= q2, N_LABELS[1], N_LABELS[2]))
    return label, (q1, q2)


def temporal_support(values, facts, ref_cells, j):
    """T_j: (first, last) day with a positive count of j in V, or None when there is none."""
    positive = ref_cells[:, j] & (values[:, j] >= ZERO_BELOW)
    if not positive.any():
        return None
    return int(facts.days[positive].min()), int(facts.days[positive].max())


def local_parts(values, props, j, support, select=None):
    """(eval counts, eval props, ref counts, ref props, ref weights, n cells) of R-local, optionally
    restricted to the supported E cells whose position k in support['rows'] has select[k]."""
    keep = support['supported'] if select is None else support['supported'] & select
    entries = keep[support['owner']]
    rows, ref_rows = support['rows'][keep], support['ref_rows'][entries]
    return (values[rows, j], props[rows, j], values[ref_rows, j], props[ref_rows, j],
            support['weight'][entries], int(keep.sum()))


def d1_d2(eval_counts, eval_props, ref_counts, ref_props, weights=None):
    gap, hat, ref = zero_gap(eval_counts, ref_counts, weights)
    return {'d1': gap, 'pr_zero_hat': hat, 'pr_zero_ref': ref,
            'd2': positive_w1(eval_counts, eval_props, ref_counts, ref_props, weights),
            'eval_positive': int(np.sum(np.asarray(eval_counts) >= ZERO_BELOW)),
            'ref_positive': int(np.sum(np.asarray(ref_counts) >= ZERO_BELOW))}


def aggregate(per_target, key, cells_key):
    """Equal mean and cell-weighted mean of per_target[j][key] over targets where it is not None."""
    pairs = [(v[key], v[cells_key]) for v in per_target.values() if v.get(key) is not None]
    if not pairs:
        return {'equal': None, 'weighted': None, 'targets': 0, 'cells': 0}
    x, w = np.array(pairs, float).T
    return {'equal': float(x.mean()), 'weighted': float(np.sum(x * w) / w.sum()) if w.sum() > 0 else None,
            'targets': len(pairs), 'cells': int(w.sum())}


def reference_shift(local, pooled):
    """Table 6: D1 R-local vs R-pooled of one target. Sign differs when the product is < 0 (an
    exact 0 has no sign); large when |local - pooled| >= MATERIAL_CHANGE. None when a side is NA."""
    if local is None or pooled is None:
        return {'sign_differs': None, 'change': None, 'large': None}
    return {'sign_differs': bool(local * pooled < 0), 'change': float(local - pooled),
            'large': bool(abs(local - pooled) >= MATERIAL_CHANGE)}


def diagnose(values, facts, eval_cells, ref_cells, support=None, targets=None):
    """D1, D2 (R-pooled and R-local) and D3 per target, then strata and aggregates.

    support: the result of local_support for the same masks and facts (computed when None).
    """
    values = check_cells(values, eval_cells, ref_cells)
    support = local_support(facts, eval_cells, ref_cells) if support is None else support
    props = values / facts.totals[:, None]
    targets = targets or [str(j) for j in range(values.shape[1])]
    n_label, cuts = n_strata(facts, eval_cells)
    strata = {**{('group', g): facts.groups == g for g in sorted(set(facts.groups[eval_cells.any(axis=1)]))},
              **{('n', n): n_label == n for n in N_LABELS}}
    per_target, stratum_rows = {}, {s: {} for s in strata}
    for j, name in enumerate(targets):
        e, r, s = eval_cells[:, j], ref_cells[:, j], support[j]
        pooled = d1_d2(values[e, j], props[e, j], values[r, j], props[r, j])
        local = d1_d2(*local_parts(values, props, j, s)[:5])
        span = temporal_support(values, facts, ref_cells, j)
        outside = e & ((facts.days < span[0]) | (facts.days > span[1]) if span else np.ones(len(e), bool))
        flagged = outside & (values[:, j] >= ZERO_BELOW)
        per_target[name] = {
            'eval_cells': int(e.sum()), 'ref_cells': int(r.sum()), 'supported': int(s['supported'].sum()),
            'coverage': float(s['supported'].mean()) if len(s['rows']) else None,
            'mean_ref_cells': float(s['n_ref'][s['supported']].mean()) if s['supported'].any() else None,
            'mean_ref_locations': float(s['n_locations'][s['supported']].mean()) if s['supported'].any() else None,
            'pooled': pooled, 'local': local, 'span': span, 'outside_cells': int(outside.sum()),
            'd3_cells': int(flagged.sum()), 'd3_mass': float(props[flagged, j].sum())}
        for key, rows in strata.items():
            select = rows[s['rows']]
            parts = local_parts(values, props, j, s, select)
            entry = {'eval_cells': int(select.sum()), 'supported': parts[5],
                     'd3_cells': int((flagged & rows).sum()), 'd3_mass': float(props[flagged & rows, j].sum())}
            entry.update(d1_d2(*parts[:5]) if parts[5] >= MIN_STRATUM_CELLS else {'d1': None, 'd2': None})
            stratum_rows[key][name] = entry
    summary = {f'{ref}_{d}': aggregate({t: {**v[ref], 'cells': v['supported'] if ref == 'local' else
                                            v['eval_cells']} for t, v in per_target.items()}, d, 'cells')
               for ref in ('pooled', 'local') for d in ('d1', 'd2')}
    stratified = []
    for (kind, label), entries in stratum_rows.items():
        cells = sum(v['eval_cells'] for v in entries.values())
        d3 = cells >= MIN_STRATUM_CELLS
        stratified.append({
            'kind': kind, 'stratum': label, 'eval_cells': cells,
            'supported': sum(v['supported'] for v in entries.values()),
            'd1': aggregate(entries, 'd1', 'supported'), 'd2': aggregate(entries, 'd2', 'supported'),
            'd3_cells': sum(v['d3_cells'] for v in entries.values()) if d3 else None,
            'd3_mass': float(sum(v['d3_mass'] for v in entries.values())) if d3 else None})
    return {'targets': per_target, 'summary': summary, 'strata': stratified, 'n_cuts': cuts,
            'eval_cells': int(eval_cells.sum()), 'supported': int(sum(v['supported'] for v in per_target.values())),
            'd3_cells': sum(v['d3_cells'] for v in per_target.values()),
            'd3_mass': float(sum(v['d3_mass'] for v in per_target.values()))}


# ---------- native tables ----------

DATA, SCHEMA = ROOT / 'data/covariants.csv', ROOT / 'configs/baselines_covariants.toml'
GROUP_MAP, CONFIGS = ROOT / 'configs/covid_location_continent.csv', ROOT / 'configs'
MAIN, OTHER, RESIDUAL = 'main', 'micoda_other', 'residual'      # run directory prefixes (pass names)
EXPLORATORY = ROOT / 'configs/evaluation/exploratory'
TIER1 = {'tier01_runs.csv': ('model', ('hron', 'tsagris', 'micoda', 'micoda_other', 'same_day_mean',
                                       'mean_fill', 'zero_fill', 'marginal_sampler')),
         'residual_runs.csv': ('variant', ('hron_cap', 'tsagris_cap', 'micoda_other_cap'))}
SOURCES = {
    'hron': (MAIN, 'hron', 'models_default_a17.toml', 'main'),
    'tsagris': (MAIN, 'tsagris', 'models_default_a17.toml', 'main'),
    'micoda': (MAIN, 'micoda', 'models_default_a17.toml', 'main'),
    'micoda_other': (OTHER, 'micoda', 'models_micoda_other.toml', 'other'),
    'same_day_mean': (MAIN, 'same_day_mean', 'models_default_a17.toml', 'main'),
    'mean_fill': (MAIN, 'mean_fill', 'models_default_a17.toml', 'main'),
    'zero_fill': (MAIN, 'zero_fill', 'models_default_a17.toml', 'main'),
    'marginal_sampler': (MAIN, 'marginal_sampler', 'models_default_a17.toml', 'main'),
    'hron_cap': (RESIDUAL, 'hron_cap', 'models_residual_rule.toml', 'residual'),
    'tsagris_cap': (RESIDUAL, 'tsagris_cap', 'models_residual_rule.toml', 'residual'),
    'micoda_other_cap': (RESIDUAL, 'micoda_other_cap', 'models_residual_rule.toml', 'residual'),
}
MASKS = ('whole_feature', 'random_cell')     # every run directory holds the native run; read the first
TIER1_SEEDS = ('0', '1', '2', '3', '4')


def recorded_table(config, table):
    """[table] of a run's config.json, with MICoDa's `other` written out when the run left it to the
    runner's default (micoda_other: other = false before PR #5), so that a run made before
    PR #5 compares equal to configs/models_default_a17.toml, where A17 is explicit."""
    params = config['models'].get(table)
    if params is None:
        return None
    params = dict(params)
    if params.get('model', table) == 'micoda' and 'other' not in params:
        params['other'] = micoda_other(config, table)
    return params


def load_source(name, data, rerun=None, roots=None):
    """Checked native completed counts of one model, and its provenance record."""
    prefix, table, models_file, rerun_dir = SOURCES[name]
    folder = find_run(roots, f'{prefix}_{MASKS[0]}')
    config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    path, twin = folder / f'{table}_completed.csv', find_run(roots, f'{prefix}_{MASKS[1]}') / f'{table}_completed.csv'
    group_map = config.get('group_map') or {}
    problems = [text for bad, text in (
        (config['data_sha256'] != sha256(DATA), 'data sha256 differs from data/covariants.csv'),
        (config['schema'] != read_schema(SCHEMA), 'schema differs from the current schema'),
        (group_map.get('sha256') != sha256(GROUP_MAP) or group_map.get('rows_resolved') != len(data.counts),
         'group map differs'),
        (recorded_table(config, table) != read_models(CONFIGS / models_file, [table])[table],
         f'[{table}] differs from {models_file}'),
        ((config.get('git') or {}).get('uncommitted_changes') is not False, 'run made from an unclean tree'),
        ((summary['native'].get(table) or {}).get('status') != 'ok', 'native status is not ok'),
        (not path.exists() or not twin.exists() or sha256(path) != sha256(twin),
         f'native CSV missing or not identical in the {MASKS[1]} run'),
    ) if bad]
    if problems:
        raise ValueError(f'{name} ({shown(folder)}): ' + '; '.join(problems))
    counts = read_completed(path, data)
    check_completed(data, counts, np.zeros(data.counts.shape, bool))
    record = {'dir': shown(folder), 'table': table, 'models_file': models_file,
              'config_sha256': sha256(folder / 'config.json'), 'csv': path.name, 'csv_sha256': sha256(path),
              'git_commit': (config.get('git') or {}).get('commit'), f'same_csv_in_{MASKS[1]}': True}
    if rerun is not None:
        again = (rerun[rerun_dir] if isinstance(rerun, dict) else rerun / rerun_dir) / path.name
        record['rerun_identical'] = again.exists() and sha256(again) == sha256(path)
    return counts, record


def read_completed(path, data):
    """Target columns of a completed CSV, row by row equal to the data apart from the filled targets.

    Not load_csv: it refuses a row whose filled sum exceeds N by a rounding error, which
    check_completed accepts (1e-10 relative), so the checks are done by check_completed."""
    with path.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(data.rows):
        raise ValueError(f'{path.name}: {len(rows)} rows, the data has {len(data.rows)}')
    fixed = [c for c in data.fields if c not in data.targets]
    if any(row[c] != source[c] for row, source in zip(rows, data.rows) for c in fixed):
        raise ValueError(f'{path.name}: non-target columns differ from the data')
    return np.array([[float(row[t]) for t in data.targets] for row in rows])


def day_text(first, day):
    return 'NA' if day is None else (first + timedelta(days=int(day))).isoformat()


def support_lines(data, cells, results, first):
    one = next(iter(results.values()))
    m, f, mm = (int(cells[k].sum()) for k in ('missing', 'forced', 'eval'))
    q1, q2 = one['n_cuts']
    out = [f"Ô thiếu thật M = {m}; ép về 0 M_f = {f} ({f / m:.2%}); model quyết định M_m = {mm} ({mm / m:.2%}); "
           f"ô quan sát O = {int(cells['ref'].sum())}. R-local có support: {one['supported']} / {mm} ô M_m "
           f"(coverage {one['supported'] / mm:.2%}). Ngưỡng nhóm N (tam phân vị trên dòng có ô M_m): "
           f"N1 ≤ {q1:g} < N2 ≤ {q2:g} < N3.", '',
           '| Target | ô O | ô M_m | có support | coverage | TB ô O trong reference | TB location | '
           'T_j (ngày đầu – cuối có count dương) | ô M_m ngoài T_j |', '|---|' + '---:|' * 6 + '---|---:|']
    rows = []
    for name, v in one['targets'].items():
        span = v['span'] or (None, None)
        out.append(f"| {name} | {v['ref_cells']} | {v['eval_cells']} | {v['supported']} | {na(v['coverage'], 3)} | "
                   f"{na(v['mean_ref_cells'], 1)} | {na(v['mean_ref_locations'], 1)} | "
                   f"{day_text(first, span[0])} – {day_text(first, span[1])} | {v['outside_cells']} |")
        rows.append({'kind': 'target', 'stratum': name, **{k: v[k] for k in (
            'ref_cells', 'eval_cells', 'supported', 'coverage', 'mean_ref_cells', 'mean_ref_locations',
            'outside_cells')}, 'span_first': day_text(first, span[0]), 'span_last': day_text(first, span[1])})
    out += ['', '| Tầng | ô M_m | có support | coverage | số target có ≥ 30 ô có support |', '|---|---:|---:|---:|---:|']
    for s in one['strata']:
        enough = s['d1']['targets']
        out.append(f"| {s['stratum']} | {s['eval_cells']} | {s['supported']} | "
                   f"{na(s['supported'] / s['eval_cells'] if s['eval_cells'] else None, 3)} | {enough} |")
        rows.append({'kind': s['kind'], 'stratum': s['stratum'], 'eval_cells': s['eval_cells'],
                     'supported': s['supported'], 'targets_scored': enough})
    return out, rows


def target_table(results, targets, ref, key, digits, fmt):
    """Model x target table of results[m]['targets'][t][ref][key], with the two aggregates."""
    head = ['| Model | ' + ' | '.join(targets) + ' | TB đều | TB trọng số (số ô) |',
            '|---|' + '---:|' * (len(targets) + 2)]
    for m, r in results.items():
        agg = r['summary'][f'{ref}_{key}']
        head.append(f'| {m} | ' + ' | '.join(fmt(r['targets'][t][ref][key], digits) for t in targets)
                    + f" | {fmt(agg['equal'], digits)} | {fmt(agg['weighted'], digits)} ({agg['targets']} target) |")
    return head


def d1_lines(results, targets):
    out = []
    for ref, label in (('local', 'R-local (chính)'), ('pooled', 'R-pooled')):
        one = next(iter(results.values()))['targets']
        out += [f'**D1 {label}** = Pr̂(zero, ô M_m) − Pr_ref(zero). Dòng đầu: Pr_ref(zero) của reference.', '',
                *target_table(results, targets, ref, 'd1', 3, signed)[:2],
                '| *Pr_ref(zero)* | ' + ' | '.join(na(one[t][ref]['pr_zero_ref'], 3) for t in targets) + ' | | |',
                *target_table(results, targets, ref, 'd1', 3, signed)[2:], '']
    return out


def d2_lines(results, targets):
    out = []
    for ref, label in (('local', 'R-local (chính)'), ('pooled', 'R-pooled')):
        out += [f'**D2 {label}** = W1 trên tỷ lệ c/N, phần dương (count ≥ 0.5). NA = một phía không có ô dương.',
                '', *target_table(results, targets, ref, 'd2', 4, na), '']
    return out


def d3_lines(results, targets):
    one = next(iter(results.values()))['targets']
    out = ['Số ô M_m được điền dương (≥ 0.5 count) ở ngày ngoài T_j; dòng đầu: số ô M_m ngoài T_j (mọi model).', '',
           '| Model | ' + ' | '.join(targets) + ' | tổng ô | tổng khối lượng tỷ lệ |',
           '|---|' + '---:|' * (len(targets) + 2),
           '| *ô M_m ngoài T_j* | ' + ' | '.join(str(one[t]['outside_cells']) for t in targets)
           + f" | {sum(one[t]['outside_cells'] for t in targets)} | |"]
    for m, r in results.items():
        out.append(f'| {m} | ' + ' | '.join(str(r['targets'][t]['d3_cells']) for t in targets)
                   + f" | {r['d3_cells']} | {r['d3_mass']:.3f} |")
    return out


def strata_lines(results):
    out = []
    one = next(iter(results.values()))['strata']
    for kind, title in (('group', 'châu lục'), ('n', 'nhóm N')):
        labels = [s['stratum'] for s in one if s['kind'] == kind]
        for key, label in (('d1', 'D1 R-local'), ('d2', 'D2 R-local')):
            out += [f'**{label} theo {title}**: TB đều / TB trọng số qua các target có ≥ 30 ô có support '
                    '(số target trong ngoặc).', '', '| Model | ' + ' | '.join(labels) + ' |',
                    '|---|' + '---:|' * len(labels)]
            for m, r in results.items():
                cells = {s['stratum']: s[key] for s in r['strata'] if s['kind'] == kind}
                fmt = signed if key == 'd1' else na
                out.append(f'| {m} | ' + ' | '.join(
                    f"{fmt(cells[x]['equal'], 3)} / {fmt(cells[x]['weighted'], 3)} ({cells[x]['targets']})"
                    for x in labels) + ' |')
            out.append('')
        out += [f'**D3 theo {title}**: số ô cờ / khối lượng tỷ lệ (NA khi tầng có < 30 ô M_m).', '',
                '| Model | ' + ' | '.join(labels) + ' |', '|---|' + '---:|' * len(labels)]
        for m, r in results.items():
            cells = {s['stratum']: s for s in r['strata'] if s['kind'] == kind}
            out.append(f'| {m} | ' + ' | '.join('NA' if cells[x]['d3_cells'] is None else
                                                 f"{cells[x]['d3_cells']} / {cells[x]['d3_mass']:.3f}"
                                                 for x in labels) + ' |')
        out.append('')
    return out


def shift_lines(results, targets):
    """Table 6: per target and model, does D1 change sign or by >= MATERIAL_CHANGE between references?"""
    one = next(iter(results.values()))['targets']
    out = [f'Ký hiệu: **±** đổi dấu, **L** |D1 local − D1 pooled| ≥ {MATERIAL_CHANGE:g}, · không có cả hai. '
           'Cột Δref = Pr_ref(zero) local − pooled (chung mọi model).', '',
           '| Target | Δref | ' + ' | '.join(results) + ' |', '|---|---:|' + ':---:|' * len(results)]
    counts = {m: [0, 0, 0] for m in results}
    for t in targets:
        ref = one[t]
        delta = None if ref['local']['pr_zero_ref'] is None else ref['local']['pr_zero_ref'] - ref['pooled'][
            'pr_zero_ref']
        marks = []
        for m, r in results.items():
            s = reference_shift(r['targets'][t]['local']['d1'], r['targets'][t]['pooled']['d1'])
            if s['change'] is None:
                marks.append('NA')
                continue
            counts[m][0] += s['sign_differs']
            counts[m][1] += s['large']
            counts[m][2] += 1
            marks.append(('±' if s['sign_differs'] else '') + ('L' if s['large'] else '') or '·')
        out.append(f'| {t} | {signed(delta)} | ' + ' | '.join(marks) + ' |')
    out += ['', 'Số target đổi dấu / thay đổi lớn (trên số target có cả hai giá trị): ' + '; '.join(
        f'{m} {c[0]} / {c[1]} (/{c[2]})' for m, c in counts.items()) + '.']
    return out


def decoy_lines(results, targets):
    """Table 7: behaviour of the diagnostics on the two decoys. Not a gate."""
    out = []
    if 'zero_fill' in results:
        z = results['zero_fill']['targets']
        has = {ref: [t for t in targets if z[t][ref]['d1'] is not None] for ref in ('local', 'pooled')}
        exact = sum(z[t]['pooled']['d1'] == 1 - z[t]['pooled']['pr_zero_ref'] for t in has['pooled'])
        top = {ref: sum(all(r['targets'][t][ref]['d1'] is None or r['targets'][t][ref]['d1'] <= z[t][ref]['d1']
                            for r in results.values()) for t in has[ref]) for ref in has}
        d2na = {ref: sum(z[t][ref]['d2'] is None for t in has[ref]) for ref in has}
        s = results['zero_fill']['summary']
        n = {ref: len(v) for ref, v in has.items()}
        out += [f"- zero_fill: D1 pooled = 1 − Pr_ref(zero) ở {exact}/{n['pooled']} target có D1 (mọi ô M_m "
                f"điền 0). D1 TB đều R-local {signed(s['local_d1']['equal'])}, R-pooled "
                f"{signed(s['pooled_d1']['equal'])}. D1 lớn nhất (hoặc bằng) trong các model ở "
                f"{top['local']}/{n['local']} target (R-local), {top['pooled']}/{n['pooled']} (R-pooled). D2 = NA ở "
                f"{d2na['local']}/{n['local']} (R-local), {d2na['pooled']}/{n['pooled']} (R-pooled) target có D1. "
                f"Target không có D1 (không có ô M_m, hoặc không ô nào có support): "
                f"{', '.join(t for t in targets if t not in has['local']) or 'không'} (R-local); "
                f"{', '.join(t for t in targets if t not in has['pooled']) or 'không'} (R-pooled)."]
    if 'marginal_sampler' in results:
        r = results['marginal_sampler']
        s = r['summary']
        has = [t for t in targets if r['targets'][t]['pooled']['d1'] is not None]
        small = sum(abs(r['targets'][t]['pooled']['d1']) <= .05 for t in has)
        out += [f"- marginal_sampler: D1 R-pooled TB đều {signed(s['pooled_d1']['equal'])}, trọng số "
                f"{signed(s['pooled_d1']['weighted'])}; |D1 pooled| ≤ 0.05 ở {small}/{len(has)} target có D1. "
                f"D2 R-pooled TB đều {na(s['pooled_d2']['equal'])}, trọng số {na(s['pooled_d2']['weighted'])}. "
                f"R-local: D1 {signed(s['local_d1']['equal'])} / {signed(s['local_d1']['weighted'])}, "
                f"D2 {na(s['local_d2']['equal'])} / {na(s['local_d2']['weighted'])} (đều / trọng số)."]
    return out


def overview_lines(results):
    out = ['| Model | D1 local (đều / trọng số) | D2 local (đều / trọng số) | D1 pooled (đều / trọng số) | '
           'D2 pooled (đều / trọng số) | D3 ô | D3 khối lượng |', '|---|---|---|---|---|---:|---:|']
    for m, r in results.items():
        s = r['summary']
        out.append(f"| {m} | {signed(s['local_d1']['equal'])} / {signed(s['local_d1']['weighted'])} | "
                   f"{na(s['local_d2']['equal'])} / {na(s['local_d2']['weighted'])} | "
                   f"{signed(s['pooled_d1']['equal'])} / {signed(s['pooled_d1']['weighted'])} | "
                   f"{na(s['pooled_d2']['equal'])} / {na(s['pooled_d2']['weighted'])} | {r['d3_cells']} | "
                   f"{r['d3_mass']:.3f} |")
    return out


def tier1_lines(results, tier1):
    seeds = {m: ', '.join(v['seeds']) for m, v in tier1.items()}
    out = [f"Tầng 1: trung bình trên seed chung của 11 model (whole_feature: {seeds.get('whole_feature')}; "
           f"random_cell: {seeds.get('random_cell')}), lấy từ tier01_runs.csv và residual_runs.csv đã commit. "
           'Nhóm ô model; z = đáp án 0, + = đáp án dương. Tầng 2: native, R-local, TB trọng số.', '',
           '| Model | T1 wf MAE | T1 wf MAE z | T1 wf MAE + | T1 rc MAE | T1 rc MAE z | T1 rc MAE + | '
           'T2 D1 | T2 D2 | T2 D3 ô |', '|---|' + '---:|' * 9]
    for m, r in results.items():
        cols = [na(tier1[mask]['means'][m][k]) for mask in ('whole_feature', 'random_cell')
                for k in ('mae_prop', 'mae_prop_model_truth_zero', 'mae_prop_truth_positive')]
        s = r['summary']
        out.append(f"| {m} | " + ' | '.join(cols) + f" | {signed(s['local_d1']['weighted'])} | "
                   f"{na(s['local_d2']['weighted'])} | {r['d3_cells']} |")
    return out


def tier1_rows():
    """Tier 1 means (committed step 2 / step 4 tables) over the seeds where all 11 models count."""
    runs = {}
    for path, (column, models) in TIER1.items():
        with (EXPLORATORY / path).open(encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                if row[column] in models and row['seed'] in TIER1_SEEDS:
                    runs.setdefault(row['mask'], {}).setdefault(row[column], {})[row['seed']] = row
    out = {}
    for mask, models in runs.items():
        seeds = [s for s in TIER1_SEEDS if all(models.get(m, {}).get(s, {}).get('eligible') == 'True'
                                               for m in SOURCES)]
        out[mask] = {'seeds': seeds, 'means': {m: {k: float(np.mean([float(models[m][s][k]) for s in seeds]))
                                                   for k in ('mae_prop', 'mae_prop_model_truth_zero',
                                                             'mae_prop_truth_positive')} for m in SOURCES}}
    return out


def summarize(out_dir, rerun=None, roots=None):
    data = load_csv(DATA, SCHEMA)
    first = date.fromisoformat(data.rows[int(np.argmin(data.days))]['date'])
    loaded = {name: load_source(name, data, rerun, roots) for name in SOURCES}
    facts, cells = row_facts(data), native_cells(data)
    support = local_support(facts, cells['eval'], cells['ref'])
    targets = list(data.targets)
    results = {name: diagnose(values, facts, cells['eval'], cells['ref'], support, targets)
               for name, (values, _) in loaded.items()}
    blocks = {}
    blocks['support'], support_rows = support_lines(data, cells, results, first)
    blocks['d1'], blocks['d2'] = d1_lines(results, targets), d2_lines(results, targets)
    blocks['d3'], blocks['strata'] = d3_lines(results, targets), strata_lines(results)
    blocks['shift'], blocks['decoys'] = shift_lines(results, targets), decoy_lines(results, targets)
    blocks['overview'], blocks['tier1'] = overview_lines(results), tier1_lines(results, tier1_rows())
    target_rows = [{'model': m, 'ref': ref, 'target': t, 'eval_cells': v['eval_cells'] if ref == 'pooled' else
                    v['supported'], **v[ref]} for m, r in results.items() for t, v in r['targets'].items()
                   for ref in ('local', 'pooled')]
    target_rows += [{'model': m, 'ref': 'd3', 'target': t, 'eval_cells': v['eval_cells'], 'outside_cells':
                     v['outside_cells'], 'd3_cells': v['d3_cells'], 'd3_mass': v['d3_mass']}
                    for m, r in results.items() for t, v in r['targets'].items()]
    strata_rows = [{'model': m, 'kind': s['kind'], 'stratum': s['stratum'], 'eval_cells': s['eval_cells'],
                    'supported': s['supported'], **{f'{d}_{k}': s[d][k] for d in ('d1', 'd2')
                                                    for k in ('equal', 'weighted', 'targets', 'cells')},
                    'd3_cells': s['d3_cells'], 'd3_mass': s['d3_mass']} for m, r in results.items() for s in r['strata']]
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / 'tier2_targets.csv', target_rows)
    write_csv(out_dir / 'tier2_strata.csv', strata_rows)
    write_csv(out_dir / 'tier2_support.csv', support_rows)
    md = ['# Step 5: Tầng 2 native (mô tả, không xếp hạng) — **exploratory**', '',
          'Sinh bởi `tier2.py`; không sửa tay. Không có đáp án trên ô thiếu thật: mọi số là mô tả so với '
          'reference từ ô quan sát, không phải độ chính xác.', '']
    for name, title in (('support', '1. Support'), ('d1', '2. D1'), ('d2', '3. D2'), ('d3', '4. D3'),
                        ('strata', '5. Phân tầng'), ('shift', '6. R-local vs R-pooled'),
                        ('decoys', '7. Đối thủ giả'), ('overview', 'Tổng hợp'), ('tier1', 'So với Tầng 1')):
        md += [f'## {title}', '', *blocks[name], '']
    (out_dir / 'tier2_results.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    manifest = {'data_sha256': sha256(DATA), 'schema_sha256': sha256(SCHEMA), 'group_map_sha256': sha256(GROUP_MAP),
                'diagnostics_sha256': sha256(Path(__file__)),
                'constants': {k: globals()[k] for k in ('ZERO_BELOW', 'WINDOW_DAYS', 'MIN_CELLS', 'MIN_LOCATIONS',
                                                          'MIN_STRATUM_CELLS', 'MATERIAL_CHANGE')},
                'cells': {k: int(v.sum()) for k, v in cells.items()}, 'models': {m: rec for m, (_, rec) in loaded.items()},
                'results_sha256': {m: __import__('hashlib').sha256(json.dumps(r, sort_keys=True).encode()).hexdigest()
                                   for m, r in results.items()}}
    (out_dir / 'tier2_manifest.json').write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n',
                                                 encoding='utf-8')
    blocks['check'] = check_lines(manifest, rerun is not None)
    update_report(out_dir / 'REPORT_vi.md', blocks)
    return results, manifest


def check_lines(manifest, rerun):
    models = manifest['models']
    out = [f"data sha256 `{manifest['data_sha256'][:12]}`, schema `{manifest['schema_sha256'][:12]}`, group map "
           f"`{manifest['group_map_sha256'][:12]}`, evaluation/tier2.py `{manifest['diagnostics_sha256'][:12]}`.", '',
           '| Model | Thư mục run | commit của run | CSV native sha256 | cùng byte ở random_cell | chạy lại = cùng byte |',
           '|---|---|---|---|:---:|:---:|']
    for m, rec in models.items():
        again = rec.get('rerun_identical', 'không chạy') if rerun else 'không chạy'
        out.append(f"| {m} | `{rec['dir']}` | `{(rec['git_commit'] or '')[:7]}` | `{rec['csv_sha256'][:12]}` | "
                   f"{rec[f'same_csv_in_{MASKS[1]}']} | {again} |")
    return out
