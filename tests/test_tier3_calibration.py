"""Tier 3 calibration (step 6): reference firewall, truth row, pairwise agreement, regret, U3, verdicts."""
import json

import numpy as np
import pytest

from longitudinal_impute import tier2_diagnostics as t2
from longitudinal_impute import tier3_calibration as t3
from test_tier2_diagnostics import panel


def masked_case(seed=0):
    data = panel(seed=seed, locations=8, dates=30)
    rng = np.random.default_rng(seed + 10)
    hidden = np.isfinite(data.counts) & (rng.random(data.counts.shape) < .15)
    cells = t2.calibration_cells(data, hidden)
    pred = np.where(hidden, rng.integers(0, 4, data.counts.shape), np.nan)
    return data, hidden, cells, pred


# ---------- reference firewall and truth row ----------

def test_changing_the_truth_of_hidden_cells_never_changes_a_reference():
    data, hidden, cells, pred = masked_case()
    f, targets = t2.row_facts(data), list(data.targets)
    before = t2.diagnose(t3.case_table(data.counts, hidden, pred), f, cells['eval'], cells['ref'], targets=targets)
    other = data.counts.copy()
    other[hidden] = other[hidden] + 7                 # new truth on H, V unchanged
    assert not (cells['ref'] & hidden).any()
    after = t2.diagnose(t3.case_table(other, hidden, pred), f, cells['eval'], cells['ref'], targets=targets)
    assert json.dumps(after) == json.dumps(before)    # reference (local, pooled) and T_j of D3 unchanged
    support = t2.local_support(f, cells['eval'], cells['ref'])
    assert all(not hidden[s['ref_rows'], j].any() for j, s in support.items())


def test_truth_row_reads_the_truth_of_e_only():
    data, hidden, cells, pred = masked_case(1)
    f, targets = t2.row_facts(data), list(data.targets)
    truth = t3.truth_table(data.counts, cells['eval'], cells['ref'])
    assert np.isnan(truth[hidden & ~cells['eval']]).all() and np.isnan(truth[~np.isfinite(data.counts)]).all()
    assert np.array_equal(truth[cells['eval']], data.counts[cells['eval']])
    junk = data.counts.copy()
    junk[hidden & ~cells['eval']] = 1e6               # forced cells of H: never read
    a = t2.diagnose(truth, f, cells['eval'], cells['ref'], targets=targets)
    b = t2.diagnose(np.where(cells['eval'] | cells['ref'], junk, 5.), f, cells['eval'], cells['ref'], targets=targets)
    assert json.dumps(a) == json.dumps(b)
    model = t2.diagnose(t3.case_table(data.counts, hidden, data.counts), f, cells['eval'], cells['ref'],
                        targets=targets)
    assert json.dumps(model['targets']) == json.dumps(a['targets'])      # predicting the truth = truth row


# ---------- scalars, pairwise, regret ----------

def diag(values, cells=None):
    cells = cells or [1] * len(values)
    return {'targets': {f't{k}': {'supported': c, 'eval_cells': c, 'local': {'d1': v}, 'pooled': {'d1': v}}
                        for k, (v, c) in enumerate(zip(values, cells))}}


def test_scalar_is_a_cell_weighted_mean_of_distances_on_common_targets():
    diags = {'truth': diag([.1, .2, None]), 'a': diag([.3, -.2, .5], [1, 3, 1]), 'b': diag([.1, .1, .1])}
    common = t3.common_targets(diags, 'local', 'd1', ['truth', 'a', 'b'])
    assert common == ['t0', 't1']
    assert t3.scalar(diags['a'], 'local', 'd1', common, diags['truth']) == pytest.approx((.2 - 1.2) / 4)
    assert t3.scalar(diags['a'], 'local', 'd1', common, diags['truth'], absolute=True) == pytest.approx(1.4 / 4)
    assert t3.scalar(diags['a'], 'local', 'd1', []) is None


def test_pairwise_signs_ties_and_missing_pairs():
    models = ['a', 'b', 'c', 'd']
    d = {'a': .1, 'b': .3, 'c': .3, 'd': None}
    e = {'a': .02, 'b': .05, 'c': .0505, 'd': .04}
    out = t3.pairwise(d, e, models)
    status = {(a, b): s for a, b, s in out['pairs']}
    assert status[('a', 'b')] == 'match' and status[('a', 'c')] == 'match'
    assert status[('b', 'c')] == 'tie'                          # |delta error| = 0.0005 < 0.001
    assert status[('a', 'd')] == 'na' and out['na'] == 3
    assert out['rate'] == 1.
    assert t3.pairwise({'a': .3, 'b': .1}, e, ['a', 'b'])['rate'] == 0.
    assert t3.pairwise({'a': .1, 'b': .3}, e, ['a', 'b'], direction=-1)['rate'] == 0.
    assert t3.pairwise({'a': .2, 'b': .2}, e, ['a', 'b'])['pairs'][0][2] == 'mismatch'   # diag tie: not a match
    assert t3.pairwise({'a': .2, 'b': .2}, {'a': .1, 'b': .1005}, ['a', 'b'])['rate'] is None


def test_regret_on_a_hand_case():
    assert t3.regret({'a': .3, 'b': .1, 'c': .2}, {'a': .02, 'b': .05, 'c': .03}, ['a', 'b', 'c']) == ('b', pytest.approx(.03))
    assert t3.regret({'a': None, 'b': .1}, {'a': .01, 'b': .04}, ['a', 'b']) == ('b', pytest.approx(.03))
    assert t3.regret({'a': .1, 'b': .1}, {'a': .02, 'b': .01}, ['a', 'b']) == ('a', pytest.approx(.01))
    assert t3.regret({'a': None}, {'a': .1}, ['a']) == (None, None)


def test_no_pooling_across_cases_ex2_style():
    """Every case: A has the lower error but the diagnostic prefers B. Both grow with case difficulty."""
    rates, diag_all, err_all = [], [], []
    for hard in range(5):
        error = {'A': .01 + .02 * hard, 'B': .015 + .02 * hard}
        diagnostic = {'A': .11 + .1 * hard, 'B': .10 + .1 * hard}
        rates.append(t3.pairwise(diagnostic, error, ['A', 'B'])['rate'])
        diag_all += list(diagnostic.values())
        err_all += list(error.values())
    pooled = np.corrcoef(np.argsort(np.argsort(diag_all)), np.argsort(np.argsort(err_all)))[0, 1]
    assert pooled > .9 and rates == [0.] * 5
    assert t3.rate_verdict({'m': rates})[1] == 'fail'


# ---------- U3, U4, proportion threshold, verdicts ----------

def test_flag_precision_by_hand_and_na_without_flags():
    out = t3.flag_precision([1, 1, 0, 1, 0, 0], [1, 1, 1, 0, 0, 0])
    assert out['p_flagged'] == pytest.approx(2 / 3) and out['p_unflagged'] == pytest.approx(1 / 3)
    assert out['gap'] == pytest.approx(1 / 3) and (out['flagged'], out['unflagged']) == (3, 3)
    assert t3.flag_precision([1, 0], [0, 0])['gap'] is None


def test_flags_use_t_j_from_v_only():
    f = t2.RowFacts(np.ones(5) * 10, np.array([10, 20, 30, 40, 5]), np.array(list('ABCDE')), np.array(['G'] * 5))
    ref = np.array([[True], [True], [False], [False], [False]])
    ev = ~ref
    values = np.array([[1.], [2.], [9.], [0.], [1.]])                  # row 2: a hidden positive at day 30
    positive, flagged = t3.flags(values, f, ev, ref)
    assert positive[:, 0].tolist() == [False, False, True, False, True]
    assert flagged[:, 0].tolist() == [False, False, True, False, True]  # T_j = [10, 20] from V only


def test_decoy_must_be_worse_than_every_real_model():
    real = ('a', 'b')
    assert t3.decoy_worse({'a': .1, 'b': .2, 'z': .3}, 'z', real)
    assert not t3.decoy_worse({'a': .1, 'b': .3, 'z': .3}, 'z', real)        # equal is not worse
    assert t3.decoy_worse({'a': .1, 'b': .2, 'z': .0}, 'z', real, higher_is_worse=False)
    assert t3.decoy_worse({'a': .1, 'b': None, 'z': .3}, 'z', real) is None


def test_proportion_zero_threshold_is_strict():
    assert t3.prop_zero_gap([0, .0009999, .001, .5], [.001, .002]) == pytest.approx(.5)
    assert t3.prop_zero_gap([.001], [0., 1.], [3, 1]) == pytest.approx(-.75)
    assert t3.prop_zero_gap([], [0.]) is None
    data, hidden, cells, pred = masked_case(2)
    f = t2.row_facts(data)
    support = t2.local_support(f, cells['eval'], cells['ref'])
    out = t3.d1_proportion(t3.case_table(data.counts, hidden, pred), f, cells['eval'], cells['ref'], support,
                           list(data.targets))
    assert set(out['targets']) == set(data.targets) and all(
        v['supported'] == support[j]['supported'].sum() for j, v in enumerate(out['targets'].values()))


def test_verdicts():
    ok = {'wf': [.9, .85, .8, 1., .8], 'rc': [.9] * 5}
    assert t3.rate_verdict(ok)[1] == 'pass'
    assert t3.rate_verdict({**ok, 'rc': [.95, .95, .95, .95, .45]})[1] == 'fail'        # one case < 0.5
    assert t3.rate_verdict({**ok, 'rc': [.7] * 5})[1] == 'fail'
    assert t3.rate_verdict({**ok, 'rc': [.9, .9, .9, None, None]})[1] == 'insufficient'
    assert t3.regret_ok([0, .005, .001] * 3 + [.2]) == (9, True)
    assert t3.regret_ok([0] * 8 + [None, .2]) == (8, False)
    assert t3.u3_verdict([.3] * 9 + [.1]) == (9, 'pass')
    assert t3.u3_verdict([.3] * 8 + [None, None]) == (8, 'insufficient')
    assert t3.u3_verdict([.3] * 8 + [.1, .1]) == (8, 'fail')


def test_errors_on_scores_only_the_given_cells():
    counts = np.array([[0., 4.], [2., 1.]])
    values = np.array([[1., 4.], [0., 3.]])
    cells = np.array([[True, False], [True, True]])
    out = t3.errors_on(counts, values, np.array([10., 4.]), cells)
    assert out['all']['cells'] == 3 and out['all']['mae_proportion'] == pytest.approx((.1 + .5 + .5) / 3)
    assert out['truth_zero']['mae_proportion'] == pytest.approx(.1)


def test_same_input_same_bytes():
    data, hidden, cells, pred = masked_case(3)
    f, targets = t2.row_facts(data), list(data.targets)
    support = t2.local_support(f, cells['eval'], cells['ref'])
    runs = [json.dumps([t2.diagnose(t3.case_table(data.counts, hidden, pred), f, cells['eval'], cells['ref'],
                                    support, targets),
                        t3.d1_proportion(t3.case_table(data.counts, hidden, pred), f, cells['eval'], cells['ref'],
                                         support, targets)]) for _ in range(2)]
    assert runs[0] == runs[1]
