"""Regression contracts for diagnostics, effective policies and extreme allocations."""
import json
import numpy as np
import pytest
from test_general_baselines import panel, scope
from longitudinal_impute.baselines.adapters import CountBaselineAdapter
from longitudinal_impute.baselines.hron import HronIterative
from longitudinal_impute.baselines.tsagris import TsagrisAdaptive
from longitudinal_impute.baselines.registry import make_core, core_configuration
from longitudinal_impute.constraints import CountBudgetConstraint
from longitudinal_impute.errors import BaselineError


def test_hron_adapter_retains_prediction_convergence_for_each_row():
    data = np.random.default_rng(122).dirichlet(np.ones(4), 15)[:, :3] * 100
    train = panel(data)
    adapter = CountBaselineAdapter(HronIterative(max_iter=2)).fit(train, scope(train))
    result = adapter.predict_result(panel([[10., np.nan, 20.], [np.nan, 15., 25.]]))
    summaries = result.diagnostics['prediction_summary']
    assert [s['row'] for s in summaries] == [0, 1]
    assert all({'iterations', 'converged', 'stopping', 'deltas', 'output_underflow_parts'} <= s.keys()
               for s in summaries)
    assert result.diagnostics['fit_summary']['iterations'] >= 1
    json.dumps(result.diagnostics, allow_nan=False)


def test_adaptive_adapter_retains_train_and_unseen_pattern_cv_once():
    data = np.random.default_rng(12).dirichlet(np.ones(4), 9)[:, :3] * 100
    data[-1, 0] = np.nan
    train = panel(data)
    model = TsagrisAdaptive(ks=(1,), alphas=(1.,), repetitions=1, leave_n=1)
    adapter = CountBaselineAdapter(model).fit(train, scope(train))
    result = adapter.predict_result(panel([[10., np.nan, 20.], [10., np.nan, 20.]]))
    diag = result.diagnostics
    assert diag['fit_summary']['donors'] == 8
    assert len(diag['tuning']['cv']) == 2
    assert set(diag['tuning']['cv']) == {'(0, 3)', '(1, 3)'}
    assert all('cv' not in row for row in diag['prediction_summary'])
    json.dumps(diag, allow_nan=False)


@pytest.mark.parametrize('method', ['tsagris_basic', 'tsagris_adaptive'])
def test_effective_configuration_roundtrip_and_no_learned_state(method):
    core = make_core(method, degenerate_policies=True)
    config = core_configuration(core)
    assert config['zero_anchor'] == 'global' and config['zero_missing_mass'] == 'uniform'
    if method == 'tsagris_basic':
        core.fit([[.1, .2, .3, .4]])
        assert core_configuration(core) == config
    adapter = CountBaselineAdapter(make_core(method), degenerate_policies=True)
    assert core_configuration(adapter.core) == config


@pytest.mark.parametrize('weights,expected', [
    ([np.nextafter(0., 1.), 0., 0.], [90., 0.]),
    ([1e308, 5e307, 5e307], [45., 22.5]),
    ([1., 2., 3.], [15., 30.]),
])
def test_extreme_explicit_allocation_finite_ratio_and_json(weights, expected):
    p = panel([[10., np.nan, np.nan]])
    with np.errstate(all='raise'):
        result = CountBudgetConstraint().apply([[1., *weights]], p)
    np.testing.assert_allclose(result.counts, [[10., *expected]])
    assert result.counts[0, 0] == 10.
    assert result.counts.sum() <= 100.
    json.dumps(result.diagnostics, allow_nan=False)


def test_inequality_cap_and_zero_mass_do_not_create_mass():
    p = panel([[10., np.nan, np.nan]])
    json.dumps(CountBudgetConstraint().apply([[1., 2., 3.]], p).diagnostics, allow_nan=False)
    np.testing.assert_array_equal(CountBudgetConstraint().apply([[1., 2., 3.]], p).counts,
                                  [[10., 2., 3.]])
    np.testing.assert_allclose(CountBudgetConstraint().apply([[1., 1e308, 1e308]], p).counts,
                               [[10., 45., 45.]])
    with pytest.raises(BaselineError):
        CountBudgetConstraint().apply([[1., 0., 0., 0.]], p)
    np.testing.assert_array_equal(CountBudgetConstraint().apply([[1., 0., 0.]], p).counts,
                                  [[10., 0., 0.]])


def test_large_total_preserves_representable_tiny_target_and_other():
    tiny = np.nextafter(0., 1.)
    p = panel([[0., np.nan, np.nan]], totals=[1e308])
    with np.errstate(all='raise'):
        result = CountBudgetConstraint().apply([[0., 1e308, tiny, tiny]], p)
    assert result.counts[0, 2] == tiny
    assert result.diagnostics['other'] == [tiny]
