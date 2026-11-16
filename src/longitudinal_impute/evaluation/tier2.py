"""Tier 2 diagnostics (step 5): describe filled values where no truth exists. NOT accuracy.

Frozen before any output was scored; change a definition only in a later step, never after
seeing numbers. Every function takes two explicit cell masks and never reads NaN itself:

    E (eval_cells)  cells whose filled values are described. Native: M_m, the real missing
                    cells not forced to 0. Masked (calibration, tier 3): hidden cells H of the
                    model group.
    V (ref_cells)   cells whose values form the reference. Native: O, the observed cells.
                    Masked: O minus H. Hidden truth never enters V.

`values` is one count table (N x D): the filled values are read on E, the reference on V,
nothing else. Unit: the proportion p = c / N of each cell, per target j.

D1  zero-rate gap_j = Pr_hat(zero | E of j) - Pr_ref(zero | reference of j), signed.
    A cell is zero when its count < ZERO_BELOW (strictly below 0.5), positive otherwise.
D2  W1 on the positive part: W1(p of positive E cells of j, p of positive reference cells of j).
    None (NA) when a side has no positive cell. D1 and D2 are never added.
D3  out-of-observed-temporal-support flag. T_j = [first, last] day with a positive count of j
    in V (any location; ends included). Counted: E cells of j filled positive at a day outside
    T_j, and their summed proportion. When j has no positive cell in V, T_j is empty and every
    positive fill is outside. A flag, not an error: it means "outside the support observed in
    this table", not "certainly wrong" nor "variant absent".

Two references for D1 / D2:
    R-pooled  every V cell of j, equal weights.
    R-local   (main) for each E cell e = (l, t, j): S_e = V cells of j at a location l' != l of the
              same group (continent), with |day(t') - day(t)| <= WINDOW_DAYS. e has support when
              |S_e| >= MIN_CELLS and S_e spans >= MIN_LOCATIONS locations; no widening, no
              global fallback. The reference of j is the mixture over supported cells
                  R_j = (1 / n_s) * sum_{e supported} (1 / |S_e|) * sum_{r in S_e} delta(p_r),
              i.e. each reference cell r of S_e has weight 1 / |S_e|, so every supported E cell
              carries total weight 1 (a V cell may appear in several S_e). D1 / D2 R-local use the
              supported E cells only; D2 compares them with R_j conditioned on positive cells.
Strata (D1, D2 R-local, D3): group (continent), and N tertile (cut points: quantiles 1/3, 2/3
of N over the rows holding an E cell; N <= q1, q1 < N <= q2, N > q2). A stratum x target value
with fewer than MIN_STRATUM_CELLS cells is None (NA).
Across targets: equal mean over targets and mean weighted by the cells each value used; targets
with NA are left out and counted. No composite score, no ranking.
"""
from dataclasses import dataclass

import numpy as np

from .evaluate import forced_zero

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
