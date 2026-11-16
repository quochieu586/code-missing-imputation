"""Tier 3 calibration (step 6): what are the tier 2 diagnostics good for? Tested on masked cells.

Frozen (use cases, thresholds, sensitivities) before any output was scored; never changed after.
A case is one (mask, seed). In a case: E = hidden cells of the model group, V = O minus H
(tier2_diagnostics.calibration_cells); D1 / D2 / D3 keep their step 5 definitions; T_j of D3 is
read from V only. The hidden truth enters only the "truth" row (its own D1 / D2 on E) and the
tier 1 errors, never a reference.

Scalars of a (case, model): the per-target value (or its distance to the truth row's value)
averaged over targets with weights = cells the value used (R-local: supported E cells; R-pooled:
E cells), over the COMMON targets where the truth row and every real model have a value. The
tier 1 error of a diagnostic is evaluate.cell_errors on the very cells that diagnostic scored
(same cells, same function as tier 1). Note: for D1 the reference cancels in D1 - D1_truth on
the same cells, so U1 does not depend on the reference except through which cells are scored.

U1  D1 warns about a zero shift. Per case, over the pairs of real models:
    under-fill: sign of delta(D1 - D1_truth) vs sign of delta(MAE of p on truth > 0 cells);
    over-fill:  sign of -delta(D1 - D1_truth) vs sign of delta(MAE of p on truth = 0 cells).
U2  D2 as a proxy ranking: sign of delta|D2 - D2_truth| vs sign of delta(MAE of p). Regret = MAE of
    the real model with the smallest |D2 - D2_truth| minus the smallest MAE of the case.
U3  D3 flag: among E cells filled positive (count >= 0.5) by the real models (pooled over models
    in a case), share of truth = 0 when flagged minus when not flagged.
U4  decoys: does a diagnostic put zero_fill / mean_fill / marginal_sampler strictly worse than
    EVERY real model in the case? Reported, no threshold.
Pairs: |delta error| < TIE is a tie (reported, not in the rate); a delta of the diagnostic equal
to 0 counts as a mismatch; a pair with a missing value is NA (reported, not in the rate).
Pass (pragmatic, not universal):
    U1 (each direction), U2: mean match rate >= PASS_RATE within EACH mask, and no case below
        MIN_CASE_RATE; U2 also regret <= REGRET_MAX in >= GOOD_CASES of the 10 cases.
    U3: gap >= U3_GAP in >= GOOD_CASES of the 10 cases.
    Not enough support: fewer than MIN_CASES cases with a rate in a mask (U1, U2), or fewer than
    GOOD_CASES cases with a gap (U3).
Sensitivities (beside, never instead of, the main result): D1 with zero = c / N < PROP_ZERO_BELOW
(U1); R-pooled instead of R-local (U1, U2).
"""
import itertools

import numpy as np

from .evaluate import cell_errors
from . import tier2_diagnostics as t2

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
    positive[eval_cells] = values[eval_cells] >= t2.ZERO_BELOW
    flagged = np.zeros_like(positive)
    for j in range(values.shape[1]):
        span = t2.temporal_support(values, facts, ref_cells, j)
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
    values = t2.check_cells(values, eval_cells, ref_cells)
    props = values / facts.totals[:, None]
    out = {}
    for j, name in enumerate(targets):
        e, r = eval_cells[:, j], ref_cells[:, j]
        parts = t2.local_parts(values, props, j, support[j])
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
