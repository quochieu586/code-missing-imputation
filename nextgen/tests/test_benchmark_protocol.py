"""Contract tests for full offline benchmark, including independent metric math."""
import copy
from dataclasses import replace
import numpy as np
import pytest
from longitudinal_impute.contracts import RawPanel, DatasetSchema
from longitudinal_impute.benchmark.protocol import make_folds
from longitudinal_impute.benchmark.masks import make_mask, SCENARIOS
from longitudinal_impute.benchmark.metrics import score, bootstrap_locations
from longitudinal_impute.baselines.panels import hide_targets, extended_proportions
from longitudinal_impute.baselines.common import distance_order
from longitudinal_impute.baselines.hron import HronKNN


def synthetic():
    entities = np.repeat([f'L{i}' for i in range(10)], 12)
    x = np.random.default_rng(3).integers(0, 15, size=(len(entities), 4)).astype(float)
    x[60:, 0] = np.nan
    return RawPanel(tuple(str(i) for i in range(len(x))), x, np.isfinite(x),
                    np.full(len(x), 100.), entities, np.tile(np.arange(12) * 14, 10), {},
                    DatasetSchema('location', 'date', 'N', ('a', 'b', 'c', 'd'), ()))


def test_group_split_every_row_test_once_and_complete_stratification():
    raw = synthetic()
    split = make_folds(raw)
    test_ids = []
    for fold in split['folds']:
        train, validation, test = map(set, [fold['train_locations'], fold['validation_locations'], fold['test_locations']])
        assert not train & validation and not train & test and not validation & test
        assert train | validation | test == set(raw.entity_ids)
        assert np.asarray(raw.original_observed)[fold['train_rows']].all(axis=1).any()
        assert sum(x in {f'L{i}' for i in range(5)} for x in test) == 1
        test_ids.extend(fold['test_rows'])
    assert sorted(test_ids) == list(range(len(raw.row_ids)))
    changed = replace(raw, counts=np.where(raw.original_observed, 999999., np.nan))
    assert make_folds(raw) == make_folds(changed)  # no target-value stratification


@pytest.mark.parametrize('scenario', SCENARIOS)
def test_masks_native_subset_reproducible_global_identity(scenario):
    raw = synthetic()
    hidden = make_mask(raw, scenario, 42)
    assert not (hidden & ~raw.original_observed).any()
    np.testing.assert_array_equal(hidden, make_mask(raw, scenario, 42))
    for entity in sorted(set(raw.entity_ids)):
        ids = np.flatnonzero(raw.entity_ids == entity)
        if scenario == 'whole_feature':
            assert (hidden[ids] == hidden[ids[0]]).all()
        if scenario in ('whole_row', 'time_block'):
            selected = np.flatnonzero(hidden[ids].any(axis=1))
            assert (hidden[ids[selected]] == raw.original_observed[ids[selected]]).all()
            if scenario == 'time_block': assert (np.diff(selected) == 1).all()
    panel, _ = hide_targets(raw, hidden, mask_id='test')
    assert np.isnan(extended_proportions(panel)[hidden.any(axis=1), -1]).all()
    changed = replace(raw, counts=np.where(hidden, 1e10, raw.counts))
    poisoned, _ = hide_targets(changed, hidden, mask_id='test')
    np.testing.assert_array_equal(panel.counts, poisoned.counts)


def test_sparse_metrics_no_native_truth_and_count_ratio_support():
    raw = synthetic()
    hidden = np.zeros_like(raw.original_observed)
    hidden[0, 0] = True; hidden[1, 1] = True
    pred = np.nan_to_num(raw.counts, nan=0.)
    pred[0, 0] += 2; pred[1, 1] += 4
    result = score(raw, np.arange(len(raw.row_ids)), pred, hidden)
    assert result['overall']['n'] == 2
    assert result['overall']['count_mae'] == 3.
    assert result['overall']['ratio_mae'] == pytest.approx(.03)
    assert result['overall']['count_rmse'] == pytest.approx(np.sqrt(10))
    assert result['validity'] == {'observed_changed': 0, 'negative': 0, 'budget_violations': 0}
    perfect = score(raw, np.arange(len(raw.row_ids)), np.nan_to_num(raw.counts), hidden)
    assert perfect['overall']['count_mae'] == 0
    assert perfect['temporal']['ratio_delta_mae'] == 0


def test_bootstrap_resamples_locations_not_individual_mask_repeats():
    stats = [{'location': 'a', 'n': 3, 'ratio_absolute_sum': .3},
             {'location': 'b', 'n': 9, 'ratio_absolute_sum': .9}]
    result = bootstrap_locations(stats, draws=100)
    np.testing.assert_allclose(result['ratio_mae_pooled_ci95'], [.1, .1])
    assert result['locations'] == 2


def test_fast_tie_order_identical_to_original_tolerance_rule():
    def original(values):
        order = np.argsort(values, kind='stable'); start = 0
        while start < len(order):
            stop = start + 1
            while stop < len(order) and np.isclose(values[order[stop]], values[order[start]], atol=1e-12, rtol=1e-12): stop += 1
            order[start:stop] = np.sort(order[start:stop]); start = stop
        return order
    rng = np.random.default_rng(4)
    for values in [rng.random(500), np.zeros(500), rng.integers(0, 10, 500).astype(float),
                   rng.permutation(np.arange(100) * .6e-12), np.array([1., 1. + .5e-12, 1. + 2e-12])]:
        np.testing.assert_array_equal(distance_order(values), original(values))
        for k in [1, 3, 5, 10, len(values)]:
            np.testing.assert_array_equal(distance_order(values, limit=k)[:k], original(values)[:k])


def test_hron_cache_does_not_change_predictions_or_survive_refit():
    donor = np.random.default_rng(3).uniform(.01, 1., (30, 5))
    q = donor[:4].copy(); q[:, 2:] = np.nan
    model = HronKNN(3).fit(donor)
    a = model.predict(q)
    assert model._donor_cache_
    b = model.predict(q)
    np.testing.assert_array_equal(a, b)
    model.fit(donor * 3)
    assert not model._donor_cache_
    np.testing.assert_allclose(model.predict(q), a)
