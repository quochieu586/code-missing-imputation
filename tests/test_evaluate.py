"""evaluate.py on examples small enough to check by hand."""
import numpy as np
import pytest

from longitudinal_impute.data import Dataset
from longitudinal_impute.evaluate import masked_errors, summarize, wasserstein, wasserstein_scores


def dataset(counts, totals):
    counts = np.array(counts, float)
    return Dataset(counts, np.array(totals, float), ('a', 'b'), ('e',) * len(counts), [], [], [])


def test_wasserstein_scores_split_forced_zero_and_model_cells():
    # Row 2: visible b = 4 = N, so a is forced to 0. Row 3: a is decided by the model.
    data = dataset([[1, 1], [3, 1], [np.nan, 4], [np.nan, 1]], [2, 4, 4, 5])
    completed = np.array([[1., 1], [3, 1], [0, 4], [4, 1]])
    a = wasserstein_scores(data, completed)['a']
    assert (a['observed_cells'], a['imputed_cells'], a['forced_zero_cells'], a['model_cells']) == (2, 2, 1, 1)
    assert a['count'] == pytest.approx({'observed_vs_completed': .5, 'observed_vs_imputed': 1.,
                                        'observed_vs_model_imputed': 2.})
    assert a['proportion'] == pytest.approx({'observed_vs_completed': .1375, 'observed_vs_imputed': .275,
                                             'observed_vs_model_imputed': .175})
    b = wasserstein_scores(data, completed)['b']
    assert b['imputed_cells'] == 0 and b['count']['observed_vs_imputed'] is None
    assert wasserstein([0., 2.], [1., 3.]) == 1.
    assert wasserstein([0.], [0., 2.]) == 1.


def test_masked_errors_by_hand():
    data = dataset([[10, 0], [5, 15], [0, 2]], [10, 20, 4])
    hidden = np.array([[False, True], [True, False], [True, False]])
    completed = np.array([[10., 0], [2, 15], [1, 2]])     # errors 0 (forced), -3, +1
    e = masked_errors(data, completed, hidden)
    expected = {
        'all': (3, 10 / 3, 4 / 3, .085 / 3, .4 / 3),
        'truth_zero': (2, .5, .5, .03125, .125),
        'truth_positive': (1, 9, 3, .0225, .15),
        'forced_zero': (1, 0, 0, 0, 0),
        'model': (2, 5, 2, .0425, .2),
        'model_truth_zero': (1, 1, 1, .0625, .25),
    }
    for group, (cells, mse, mae, pmse, pmae) in expected.items():
        assert e[group] == pytest.approx({'cells': cells, 'mse_count': mse, 'mae_count': mae,
                                          'mse_proportion': pmse, 'mae_proportion': pmae}), group
    empty = masked_errors(data, completed, np.array([[False, False], [True, False], [False, False]]))
    assert empty['forced_zero'] == {'cells': 0, 'mse_count': None, 'mae_count': None,
                                    'mse_proportion': None, 'mae_proportion': None}
    with pytest.raises(ValueError, match='subset'):
        masked_errors(dataset([[np.nan, 1]], [2]), np.ones((1, 2)), np.array([[True, False]]))


def test_summarize_gives_mean_sd_over_successful_seeds():
    rows = [{'model': 'm', 'pass': 'masked', 'mask': 'random_cell', 'seed': s, 'status': 'ok',
             'seconds': 1., 'mse_count': v, 'mse_count_forced_zero': None if s == 0 else 0., 'error': ''}
            for s, v in ((0, 1.), (1, 3.))]
    rows.append({'model': 'm', 'pass': 'masked', 'mask': 'random_cell', 'seed': 2, 'status': 'failed',
                 'seconds': .5, 'error': 'ValueError: x'})
    [summary] = summarize(rows)
    assert (summary['runs_ok'], summary['runs_failed']) == (2, 1)
    assert summary['mse_count_mean'] == 2. and summary['mse_count_sd'] == pytest.approx(np.sqrt(2))
    assert summary['seconds_mean'] == 1.
    assert summary['mse_count_forced_zero_mean'] == 0. and summary['mse_count_forced_zero_sd'] is None
    assert 'seed_mean' not in summary
