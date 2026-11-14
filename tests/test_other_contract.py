"""Common-budget contract; algorithm-specific preprocessing remains distinct."""
import numpy as np
import pytest

from longitudinal_impute.baselines.micoda import MICoDaCovid, prepare, restore
from longitudinal_impute.data import Dataset, model_input, restore_counts
from longitudinal_impute import run


def test_default_adapter_and_shipped_config_use_other():
    from pathlib import Path
    assert MICoDaCovid().other is True
    params = run.read_models(Path(__file__).resolve().parents[1] / 'configs/models_default.toml')
    assert MICoDaCovid(**params['micoda']).other is True
    assert run.space('micoda') == run.space('micoda', MICoDaCovid())
    assert MICoDaCovid(other=False).other is False


@pytest.mark.parametrize('residual', [0., 80.])
def test_same_proposals_share_the_same_budget(residual):
    counts = np.array([[100. - residual, 30., 10.], [20., 30., 10.]])
    hidden = np.array([[False, True, True], [False, False, False]])
    data = Dataset(counts, np.array([100., 100.]), ('a', 'b', 'c'), ('s1', 's2'), [], [], [1, 2])
    proposals = np.array([[.2, .3, .1, .4], [.2, .3, .1, .4]])
    prepared = prepare(counts, data.totals, hidden, other=True)
    adapter = restore(prepared, proposals)
    expected = restore_counts(data, proposals, hidden)
    assert adapter.status == 'completed'
    np.testing.assert_allclose(adapter.values, expected)
    np.testing.assert_array_equal(adapter.values[~hidden], counts[~hidden])
    assert adapter.unassigned[0] == pytest.approx(residual * .4 / .8)


def test_hidden_truth_cannot_determine_other_or_preprocessing():
    counts = np.array([[20., 30.], [10., 40.], [20., 30.]])
    hidden = np.array([[False, False], [False, False], [False, True]])
    changed = counts.copy()
    changed[hidden] = 70.
    totals = np.full(3, 100.)
    def dataset(x):
        return Dataset(x, totals, ('a', 'b'), ('s1', 's2', 's3'), [], [], [1, 2, 3])
    for positive in (False, True):
        np.testing.assert_array_equal(model_input(dataset(counts), hidden, positive=positive),
                                      model_input(dataset(changed), hidden, positive=positive))
    first, second = prepare(counts, totals, hidden, other=True), prepare(changed, totals, hidden, other=True)
    np.testing.assert_array_equal(first.values, second.values)
    assert np.isnan(first.values[-1, -1])
    a = MICoDaCovid().impute_counts(counts, totals, hidden)
    b = MICoDaCovid().impute_counts(changed, totals, hidden)
    assert a.status == b.status == 'completed'
    np.testing.assert_array_equal(a.values, b.values)


def test_paired_comparison_excludes_failures_and_mixed_budget():
    from longitudinal_impute.evaluate import paired_summary
    rows = [dict(model=m, **{'pass': 'masked'}, mask='random_cell', seed=s,
                 status='ok', space='18_with_other', hidden_cells=2, resolved_cells=2,
                 mse_count=float(s)) for m in ('hron', 'tsagris', 'micoda') for s in (1, 2)]
    rows[-1]['status'] = 'failed'
    result = paired_summary(rows)
    assert result['cases']['random_cell']['seeds_included'] == [1]
    assert result['cases']['random_cell']['seeds_excluded'] == [2]
    assert all(r['runs_ok'] == 1 and r['mse_count_mean'] == 1 for r in result['aggregate'])
    rows[-2]['space'] = rows[-1]['space'] = '17_targets'
    result = paired_summary(rows)
    assert result['aggregate'] == []
    assert result['cases']['random_cell']['reason'] == 'different_model_spaces'


def test_paired_comparison_refuses_partial_coverage_and_empty_masks():
    from longitudinal_impute.evaluate import paired_summary
    row = dict(model='micoda', **{'pass': 'masked'}, mask='random_cell', seed=1,
               status='ok', space='18_with_other', hidden_cells=2, resolved_cells=1)
    assert not paired_summary([row])['aggregate']
    row.update(hidden_cells=0, resolved_cells=0)
    assert not paired_summary([row])['aggregate']
