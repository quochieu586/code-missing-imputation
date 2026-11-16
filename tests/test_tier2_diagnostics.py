"""Tier 2 diagnostics (step 5): D1, D2, D3, the R-local reference and the calibration helper."""
import json

import numpy as np
import pytest

from longitudinal_impute.data import Dataset
from longitudinal_impute.evaluate import wasserstein
from longitudinal_impute import tier2_diagnostics as t2


def facts(entities, days, groups, totals=None):
    n = len(entities)
    return t2.RowFacts(np.ones(n) * 10 if totals is None else np.asarray(totals, float), np.asarray(days),
                       np.asarray(entities), np.asarray(groups))


def panel(seed=0, locations=8, dates=30, parts=3):
    """Two groups of `locations` locations, weekly dates, random counts; about 30% missing."""
    rng = np.random.default_rng(seed)
    entities, days, groups = [], [], []
    for g in ('G1', 'G2'):
        for loc in range(locations):
            for t in range(dates):
                entities.append(f'{g}L{loc}')
                days.append(7 * t)
                groups.append(g)
    n = len(entities)
    totals = rng.integers(5, 200, n).astype(float)
    counts = np.floor(rng.dirichlet(np.ones(parts + 1) * .3, n)[:, :parts] * totals[:, None])
    counts[rng.random(counts.shape) < .3] = np.nan
    return Dataset(counts, totals, tuple(f't{j}' for j in range(parts)), tuple(entities), [], [], [],
                   np.array(days), None, 'date', tuple(groups))


def filled(data, rng_seed=1):
    values = data.counts.copy()
    missing = ~np.isfinite(values)
    residual = data.totals - np.nansum(data.counts, axis=1)
    rng = np.random.default_rng(rng_seed)
    share = rng.random(values.shape) * missing
    share = share / np.maximum(share.sum(axis=1, keepdims=True), 1e-12)
    values[missing] = np.floor(share * residual[:, None])[missing]
    return values


# ---------- W1, D1, D2 ----------

def test_weighted_w1_equals_the_existing_w1_with_equal_and_replicated_weights():
    rng = np.random.default_rng(3)
    a, b = rng.random(7), rng.random(5)
    assert t2.weighted_wasserstein(a, b) == pytest.approx(wasserstein(a, b), abs=1e-15)
    w = np.array([3, 1, 2, 1, 4])
    assert t2.weighted_wasserstein(a, b, None, w) == pytest.approx(wasserstein(a, np.repeat(b, w)), abs=1e-14)
    assert t2.weighted_wasserstein([], b) is None and t2.weighted_wasserstein(a, b, None, np.zeros(5)) is None
    assert t2.weighted_wasserstein([.2, .2], [.2]) == 0.


def test_d1_and_d2_match_hand_values_and_zero_is_strictly_below_half():
    gap, hat, ref = t2.zero_gap([0, .4, .5, 3], [0, 0, 0, 5])
    assert (hat, ref) == (.5, .75) and gap == -.25                  # 0.4 is zero, 0.5 is not
    gap, hat, ref = t2.zero_gap([0, 2], [0, 4], [3, 1])               # weighted reference: 3/4 zero
    assert (gap, ref) == (.5 - .75, .75)
    assert t2.positive_w1([0, 1, 3], [0, .1, .3], [0, 2], [0, .2]) == pytest.approx(.1)
    assert t2.positive_w1([1, 1], [.1, .3], [2, 4, 0], [.2, .4, 0], [3, 1, 5]) == pytest.approx(
        wasserstein([.1, .3], [.2, .2, .2, .4]))


def test_d2_is_na_when_a_side_has_no_positive_cell_and_d1_is_na_when_empty():
    assert t2.positive_w1([0, 0], [0, 0], [1, 2], [.1, .2]) is None
    assert t2.positive_w1([1, 2], [.1, .2], [0, .4], [0, .04]) is None
    assert t2.zero_gap([], [0, 1]) == (None, None, None)
    assert t2.aggregate({'a': {'d': None, 'n': 5}}, 'd', 'n') == {'equal': None, 'weighted': None,
                                                                  'targets': 0, 'cells': 0}
    out = t2.aggregate({'a': {'d': .2, 'n': 1}, 'b': {'d': .5, 'n': 3}, 'c': {'d': None, 'n': 9}}, 'd', 'n')
    assert out['equal'] == pytest.approx(.35) and out['weighted'] == pytest.approx(1.7 / 4) and out['targets'] == 2


# ---------- R-local ----------

def local_toy():
    """Row 0 = the E cell (location A, group G1, day 100); every other row is a V candidate."""
    rows = [('A', 100, 'G1'),                                         # 0: E
            ('A', 100, 'G1'), ('A', 90, 'G1'),                        # own location: never used
            ('B', 72, 'G1'), ('B', 128, 'G1'),                        # |dt| = 28: inside
            ('C', 71, 'G1'), ('C', 129, 'G1'),                        # |dt| = 29: outside
            ('D', 100, 'G1'), ('E', 100, 'G2'), ('F', 99, 'G1')]      # E is another group
    entities, days, groups = zip(*rows)
    eval_cells = np.zeros((len(rows), 1), bool)
    eval_cells[0] = True
    return facts(entities, days, groups), eval_cells, ~eval_cells


def test_local_reference_excludes_own_location_other_groups_and_cells_beyond_28_days():
    f, e, v = local_toy()
    s = t2.local_support(f, e, v, min_cells=1, min_locations=1)[0]
    assert s['ref_rows'].tolist() == [3, 4, 7, 9]
    assert s['n_ref'].tolist() == [4] and s['n_locations'].tolist() == [3]
    assert s['weight'].tolist() == [.25] * 4


def test_local_thresholds_need_both_cells_and_locations():
    f, e, v = local_toy()
    assert t2.local_support(f, e, v, min_cells=4, min_locations=3)[0]['supported'].tolist() == [True]
    assert t2.local_support(f, e, v, min_cells=5, min_locations=3)[0]['supported'].tolist() == [False]
    assert t2.local_support(f, e, v, min_cells=4, min_locations=4)[0]['supported'].tolist() == [False]
    unsupported = t2.local_support(f, e, v, min_cells=5)[0]
    assert len(unsupported['owner']) == 0 and len(unsupported['weight']) == 0
    rows = 21 * [('B', 100, 'G1')] + [('A', 100, 'G1'), ('C', 100, 'G1')]
    entities, days, groups = zip(*rows)
    e = np.zeros((len(rows), 1), bool)
    e[-2] = True
    s = t2.local_support(facts(entities, days, groups), e, ~e)[0]    # defaults: 20 cells, 3 locations
    assert s['n_ref'].tolist() == [22] and s['n_locations'].tolist() == [2] and not s['supported'][0]


def test_local_d1_is_the_mean_of_per_cell_local_zero_shares():
    data = panel()
    cells = t2.native_cells(data)
    f, values = t2.row_facts(data), filled(data)
    support = t2.local_support(f, cells['eval'], cells['ref'])
    out = t2.diagnose(values, f, cells['eval'], cells['ref'], support, list(data.targets))
    for j, name in enumerate(data.targets):
        s = support[j]
        assert out['targets'][name]['supported'] == s['supported'].sum() > 0
        shares = [np.mean(values[s['ref_rows'][s['owner'] == k], j] < .5) for k in np.nonzero(s['supported'])[0]]
        hat = np.mean(values[s['rows'][s['supported']], j] < .5)
        assert out['targets'][name]['local']['d1'] == pytest.approx(hat - np.mean(shares), abs=1e-12)
        for k in np.nonzero(s['supported'])[0]:
            assert np.sum(s['weight'][s['owner'] == k]) == pytest.approx(1.)


# ---------- cell sets ----------

def test_forced_zero_cells_are_not_in_the_model_cells():
    counts = np.array([[4., np.nan, np.nan], [2., np.nan, 1.], [np.nan, 3., 0.]])
    data = Dataset(counts, np.array([4., 5., 3.]), ('a', 'b', 'c'), ('x', 'y', 'z'), [], [], [])
    cells = t2.native_cells(data)
    assert cells['forced'].tolist() == [[False, True, True], [False] * 3, [True, False, False]]
    assert cells['eval'].tolist() == [[False] * 3, [False, True, False], [False] * 3]
    assert not (cells['eval'] & cells['forced']).any() and (cells['eval'] | cells['forced']).sum() == 4
    hidden = np.zeros_like(cells['ref'])
    hidden[1, 2] = True                         # row 1 visible sum 2 < 5: model cell
    hidden[0, 0] = True                         # row 0 has no visible target left: residual 4, model cell
    cal = t2.calibration_cells(data, hidden)
    assert cal['eval'].tolist() == [[True, False, False], [False, False, True], [False] * 3]
    assert not (cal['ref'] & hidden).any() and cal['ref'].sum() == 5 - 2


def test_calibration_helper_reads_only_eval_and_ref_cells():
    data = panel(seed=2)
    rng = np.random.default_rng(5)
    hidden = np.isfinite(data.counts) & (rng.random(data.counts.shape) < .1)
    cells = t2.calibration_cells(data, hidden)
    f = t2.row_facts(data)
    values = filled(data)
    values[hidden] = rng.integers(0, 3, hidden.sum())               # a model's predictions on H
    base = t2.diagnose(values, f, cells['eval'], cells['ref'], targets=list(data.targets))
    other = values.copy()
    other[cells['eval']] = rng.integers(0, 50, cells['eval'].sum()).astype(float)
    changed = t2.diagnose(other, f, cells['eval'], cells['ref'], targets=list(data.targets))
    for name in data.targets:                     # new E values: the reference side does not move
        for ref in ('pooled', 'local'):
            assert changed['targets'][name][ref]['pr_zero_ref'] == base['targets'][name][ref]['pr_zero_ref']
            assert changed['targets'][name][ref]['ref_positive'] == base['targets'][name][ref]['ref_positive']
        assert changed['targets'][name]['span'] == base['targets'][name]['span']
    outside = ~(cells['eval'] | cells['ref'])
    junk = values.copy()
    junk[outside] = np.where(rng.random(outside.sum()) < .5, np.nan, 1e6)   # truth / forced cells: unused
    assert json.dumps(t2.diagnose(junk, f, cells['eval'], cells['ref'], targets=list(data.targets))) == json.dumps(base)
    for cell_set in ('eval', 'ref'):
        bad = values.copy()
        bad[np.argwhere(cells[cell_set])[0][0], np.argwhere(cells[cell_set])[0][1]] = np.nan
        with pytest.raises(ValueError, match='finite'):
            t2.diagnose(bad, f, cells['eval'], cells['ref'])
    with pytest.raises(ValueError, match='disjoint'):
        t2.diagnose(values, f, cells['eval'], cells['ref'] | cells['eval'])


# ---------- D3, strata, determinism ----------

def test_d3_span_and_flags_include_both_ends():
    entities = ['A', 'B', 'C', 'D', 'E', 'F']
    days = [10, 20, 30, 10, 30, 31]
    f = facts(entities, days, ['G'] * 6)
    values = np.array([[0.], [3.], [2.], [1.], [1.], [1.]])
    ref = np.array([[True], [True], [True], [False], [False], [False]])
    assert t2.temporal_support(values, f, ref, 0) == (20, 30)          # day 10 has count 0
    out = t2.diagnose(values, f, ~ref, ref, targets=['t'])['targets']['t']
    assert out['span'] == (20, 30) and out['outside_cells'] == 2           # days 10 and 31
    assert out['d3_cells'] == 2 and out['d3_mass'] == pytest.approx(.2)  # day 30 (= end) is inside
    values[1:3] = 0.                                                       # no positive in V at all
    out = t2.diagnose(values, f, ~ref, ref, targets=['t'])['targets']['t']
    assert out['span'] is None and out['d3_cells'] == 3


def test_strata_below_thirty_cells_are_na():
    data = panel(locations=4, dates=12)
    cells = t2.native_cells(data)
    out = t2.diagnose(filled(data), t2.row_facts(data), cells['eval'], cells['ref'], targets=list(data.targets))
    for stratum in out['strata']:
        assert stratum['eval_cells'] < 30 * len(data.targets)
        assert stratum['d1']['targets'] == 0 and stratum['d1']['equal'] is None
        if stratum['eval_cells'] < 30:
            assert stratum['d3_cells'] is None
    q1, q2 = out['n_cuts']
    rows = cells['eval'].any(axis=1)
    assert (q1, q2) == tuple(np.quantile(data.totals[rows], [1 / 3, 2 / 3]))
    assert sum(s['eval_cells'] for s in out['strata'] if s['kind'] == 'n') == cells['eval'].sum()
    assert sum(s['eval_cells'] for s in out['strata'] if s['kind'] == 'group') == cells['eval'].sum()


def test_large_strata_are_scored_and_results_are_byte_identical():
    data = panel(locations=10, dates=40)
    cells = t2.native_cells(data)
    f = t2.row_facts(data)
    first = json.dumps(t2.diagnose(filled(data), f, cells['eval'], cells['ref'], targets=list(data.targets)))
    again = json.dumps(t2.diagnose(filled(data), f, cells['eval'], cells['ref'], targets=list(data.targets)))
    assert first == again
    scored = [s for s in json.loads(first)['strata'] if s['d1']['targets']]
    assert scored and all(s['d1']['cells'] >= 30 for s in scored)


def test_row_facts_need_dates_and_groups():
    data = panel()
    data.groups = None
    with pytest.raises(ValueError, match='group'):
        t2.row_facts(data)


def test_reference_shift_flags_sign_and_large_changes():
    assert t2.reference_shift(.05, -.02) == {'sign_differs': True, 'change': pytest.approx(.07), 'large': False}
    assert t2.reference_shift(.3, .1)['large'] and not t2.reference_shift(.3, .1)['sign_differs']
    assert t2.reference_shift(0., -.2)['sign_differs'] is False
    assert t2.reference_shift(None, .1) == {'sign_differs': None, 'change': None, 'large': None}
