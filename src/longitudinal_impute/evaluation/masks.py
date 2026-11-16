"""whole_feature_matched (step 7): hide whole series, drawn so that the hidden model cells follow the
strata distribution pi of the native model-decided missing cells (M_m). A sensitivity mask, not
native validation: the hidden cells are still observed cells.

Frozen before any mask was drawn. Called by data.make_mask (kind 'whole_feature_matched').
- Strata: group (continent) x N group of the row. N1: N <= q1, N2: q1 < N <= q2, N3: N > q2, with
  the step 5 cut points, read from the target file, never recomputed here.
- Target: pi_k = share of the native M_m cells in stratum k, computed once (native_target) and
  frozen in a JSON file {'n_cuts': [q1, q2], 'pi': {stratum: share}}. M_m uses the forced-zero
  split (observed row sum = N), i.e. observed sums only, never a hidden value.
- Candidates: every (entity, target) pair with observed cells; its whole observed series is hidden
  (as whole_feature). n_{s,k} = cells of candidate s in stratum k; p_k = share of all candidate
  cells in k; w_s = sum_k n_{s,k} pi_k / p_k over the strata of pi with p_k > 0. Strata of pi with
  p_k = 0 are dropped and their mass is reported as unreachable.
- Draw: without replacement, probability proportional to w_s: Efraimidis-Spirakis keys
  log(u) / w_s with u = 1 - uniform(default_rng(seed)), largest key first; w_s = 0 is never drawn.
  Then the whole_feature rules: skip a series that would leave a row with no visible target (the
  guard, counted), stop once rate x observed cells are hidden.
- Firewall: the selection reads entities, groups, N and which cells are observed, never a count.

step5_cuts reads the N cut points q1, q2 from the step 5 tier 2 table (configs/evaluation/step5_tier2).

step5_cuts reads the N cut points q1, q2 from the step 5 tier 2 table (configs/evaluation/step5_tier2).

step5_cuts reads the N cut points q1, q2 from the step 5 tier 2 table (configs/evaluation/step5_tier2).

step5_cuts reads the N cut points q1, q2 from the step 5 tier 2 table (configs/evaluation/step5_tier2).
"""
import json
from pathlib import Path
import re

import numpy as np

from .metrics import forced_zero


N_LABELS = ('N1', 'N2', 'N3')


def strata(groups, totals, n_cuts):
    """Stratum label 'group|N?' of every row, from its group and N only."""
    if groups is None:
        raise ValueError('whole_feature_matched needs a group column (schema group_col)')
    q1, q2 = n_cuts
    totals = np.asarray(totals, float)
    n = np.where(totals <= q1, N_LABELS[0], np.where(totals <= q2, N_LABELS[1], N_LABELS[2]))
    return np.array([f'{g}|{k}' for g, k in zip(groups, n)])


def shares(labels, cells):
    """{stratum: share of the selected cells}, cells an N x D mask and labels per row."""
    per_row = np.asarray(cells).sum(axis=1)
    total = per_row.sum()
    if total == 0:
        return {}
    return {k: float(per_row[labels == k].sum() / total) for k in sorted(set(labels)) if per_row[labels == k].any()}


def total_variation(a, b):
    """0.5 * sum |a_k - b_k| over the union of strata (a missing stratum has share 0)."""
    return float(.5 * sum(abs(a.get(k, 0.) - b.get(k, 0.)) for k in set(a) | set(b)))


def native_target(data, n_cuts):
    """pi of the native model-decided missing cells (missing, row residual > 0), per stratum."""
    missing = ~np.isfinite(data.counts)
    model = missing & ~forced_zero(data, np.zeros_like(missing))
    return {'n_cuts': [float(q) for q in n_cuts], 'pi': shares(strata(data.groups, data.totals, n_cuts), model)}


def check_target(target):
    cuts, pi = target.get('n_cuts'), target.get('pi')
    if (not isinstance(cuts, list) or len(cuts) != 2 or not cuts[0] < cuts[1] or not isinstance(pi, dict)
            or not pi or any(v < 0 for v in pi.values()) or abs(sum(pi.values()) - 1) > 1e-9):
        raise ValueError('mask target needs n_cuts [q1 < q2] and pi {stratum: share} summing to 1')
    return target


def read_target(path):
    return check_target(json.loads(Path(path).read_text(encoding='utf-8')))


def weights(data, entities, observed, pairs, target):
    """w_s of every candidate pair and a record (p_k, unreachable strata and mass)."""
    target = check_target(target)
    labels = strata(data.groups, data.totals, target['n_cuts'])
    keys = sorted(target['pi'])
    column = {k: i for i, k in enumerate(keys)}
    pair_id = {p: i for i, p in enumerate(pairs)}
    n = np.zeros((len(pairs), len(keys)))
    rows, cols = np.nonzero(observed)
    candidate_cells = len(rows)
    for r, c in zip(rows, cols):
        k = column.get(labels[r])
        if k is not None:
            n[pair_id[(entities[r], c)], k] += 1
    p = n.sum(axis=0) / max(candidate_cells, 1)
    pi = np.array([target['pi'][k] for k in keys])
    reach = p > 0
    w = n[:, reach] @ (pi[reach] / p[reach])
    unreachable = [k for k, ok in zip(keys, reach) if not ok]
    return w, {'candidate_pairs': len(pairs), 'candidate_cells': int(candidate_cells),
               'zero_weight_pairs': int((w == 0).sum()), 'unreachable_strata': unreachable,
               'unreachable_mass': float(pi[~reach].sum()), 'p': {k: float(v) for k, v in zip(keys, p)}}


def draw_order(rng, w):
    """Indices of the candidates with w > 0, in a weighted draw without replacement (largest key first)."""
    u = 1. - rng.random(len(w))
    keys = np.full(len(w), -np.inf)
    keys[w > 0] = np.log(u[w > 0]) / w[w > 0]
    order = np.argsort(-keys, kind='stable')
    return order[w[order] > 0]



def step5_cuts(path):
    """N cut points as printed by step 5 (table 1): 'N1 ≤ q1 < N2 ≤ q2 < N3'."""
    found = re.findall(r'N1 ≤ ([0-9.e+-]+) < N2 ≤ ([0-9.e+-]+) < N3', Path(path).read_text(encoding='utf-8'))
    if len(found) != 1:
        raise ValueError(f'{path}: expected one line with the N cut points, found {len(found)}')
    return [float(found[0][0]), float(found[0][1])]
