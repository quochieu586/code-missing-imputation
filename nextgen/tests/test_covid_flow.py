"""Observable behavior of full-table imputation and evaluation."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from longitudinal_impute.data import (CovidData, load_csv, evaluation_mask, model_input,
                                      restore_counts, write_completed_csv)
from longitudinal_impute.evaluation import distribution_scores, wasserstein
from longitudinal_impute.pipeline import impute, run
from longitudinal_impute.baselines.hron import HronKNN
from longitudinal_impute.baselines.micoda import MICoDa
from longitudinal_impute.baselines.tsagris import TsagrisKNN

NEXTGEN = Path(__file__).resolve().parents[1]


@pytest.fixture
def reference_data():
    record = json.loads((NEXTGEN / 'review_20261025/adapter_reference.json').read_text())
    counts = np.array(record['counts'])
    counts[counts == record['missing_sentinel']] = np.nan
    return CovidData(counts, np.array(record['totals']), ('a', 'b', 'c'), [], []), record['expected']


# The fixture has no Hron distance ties, so ties='group' (default) and the
# previous behaviour ties='index' must both reproduce the saved output.
@pytest.mark.parametrize('name,model', [('hron', HronKNN), ('hron', lambda: HronKNN(ties='index')),
                                        ('micoda', MICoDa), ('tsagris', TsagrisKNN)])
def test_direct_flow_matches_saved_adapter_output(reference_data, name, model):
    data, expected = reference_data
    result = impute(data, model(), np.zeros_like(data.counts, dtype=bool))
    np.testing.assert_allclose(result, expected[name], atol=1e-12, rtol=1e-12)


def test_hidden_truth_never_enters_model_input(reference_data):
    data, _ = reference_data
    hidden = np.zeros_like(data.counts, dtype=bool)
    hidden[0, 1] = True
    before = model_input(data, hidden)
    data.counts[0, 1] = 987654321.
    np.testing.assert_array_equal(before, model_input(data, hidden))
    assert np.isnan(before[0, -1])
    assert np.isnan(before[0, 1])


def test_mask_is_reproducible_and_keeps_an_observed_anchor():
    counts = np.array([[np.nan, 1., np.nan], [1., 2., 3.], [np.nan, np.nan, np.nan]])
    hidden = evaluation_mask(counts, .999, 7)
    np.testing.assert_array_equal(hidden, evaluation_mask(counts, .999, 7))
    assert not hidden[0].any() and not hidden[2].any()
    assert hidden[1].sum() == 2
    assert not (hidden & ~np.isfinite(counts)).any()


def test_other_uses_count_space_and_zero_replacement_is_input_only():
    data = CovidData(np.array([[1., 1., 1., 3., 0.]]), np.array([6.]), tuple('abcde'), [], [])
    hidden = np.zeros_like(data.counts, dtype=bool)
    assert model_input(data, hidden)[0, -1] == 0.
    assert (model_input(data, hidden, positive=True) > 0).all()
    assert data.counts[0, -1] == 0.


def test_restore_observed_zero_residual_and_all_missing():
    data = CovidData(np.array([[20., 30.], [np.nan, np.nan], [10., np.nan]]),
                     np.array([80., 100., 10.]), ('a', 'b'), [], [])
    hidden = np.zeros_like(data.counts, dtype=bool)
    result = restore_counts(data, np.array([[1., 2., 3.], [1., 2., 7.], [1., 0., 0.]]), hidden)
    np.testing.assert_allclose(result, [[20., 30.], [10., 20.], [10., 0.]])
    with pytest.raises(ValueError, match='no proposed mass'):
        restore_counts(data, np.zeros((3, 3)), hidden)


def test_restore_extreme_weights_preserves_representable_tiny_mass():
    data = CovidData(np.array([[np.nan, np.nan]]), np.array([1e308]), ('a', 'b'), [], [])
    result = restore_counts(data, np.array([[1e-300, 1e308, 1e-300]]), np.array([[False, False]]))
    assert np.isfinite(result).all()
    assert result[0, 0] == 1e-300
    assert result[0, 1] == 1e308


def test_distribution_compares_native_missing_and_reports_no_support(reference_data):
    data, expected = reference_data
    scores = distribution_scores(data, np.array(expected['hron']))['distribution_counts']
    assert scores['a']['imputed_cells'] == 1
    assert scores['a']['wasserstein_observed_imputed'] is not None
    assert scores['b']['imputed_cells'] == 0
    assert scores['b']['wasserstein_observed_imputed'] is None
    assert wasserstein([0., 2.], [1., 3.]) == 1.
    assert wasserstein([0.], [0., 2.]) == 1.


@pytest.mark.parametrize('invalid', [np.nan, np.inf, -1., 1000.])
def test_export_rejects_invalid_imputed_cells_before_writing(tmp_path, reference_data, invalid):
    data, _ = reference_data
    completed = np.nan_to_num(data.counts, nan=1.)
    completed[6, 0] = invalid
    target = tmp_path / 'invalid.csv'
    with pytest.raises(ValueError, match='incomplete or violates'):
        write_completed_csv(target, data, completed)
    assert not target.exists()


def test_full_runner_preserves_csv_and_runs_separate_mse(tmp_path, reference_data):
    data, _ = reference_data
    source = tmp_path / 'input.csv'
    with source.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['location', 'date', 'total', 'a', 'b', 'c', 'note'])
        for i, row in enumerate(data.counts):
            writer.writerow(['L', f'2020-01-{i + 1:02d}', '100',
                             *('' if np.isnan(v) else f'{v:.1f}' for v in row), 'keep'])
    schema = tmp_path / 'schema.toml'
    schema.write_text('[dataset]\nentity="location"\ntime="date"\ntotal="total"\ntargets=["a","b","c"]\n')
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / 'run'
    fitted = []

    class RecordingHron(HronKNN):
        def fit(self, values):
            fitted.append(values.copy())
            return super().fit(values)

    def failing_model():
        raise ValueError('explicit model failure')

    result = run(source, schema, output,
                 models={'failed': failing_model, 'hron': RecordingHron}, mask_rate=.1, seed=13)
    assert result['failed']['status'] == 'failed'
    assert result['hron']['status'] == 'completed'
    assert result['hron']['evaluation_status'] == 'completed'
    assert result['hron']['mse'] >= 0
    hidden = np.load(output / 'evaluation_mask.npy')
    assert hidden.any()
    assert not (hidden & ~np.isfinite(data.counts)).any()
    assert len(fitted) == 2 and all(len(x) == len(data.counts) for x in fitted)
    assert np.isfinite(fitted[0][:, :-1]).sum() == np.isfinite(data.counts).sum()
    assert np.isfinite(fitted[1][:, :-1]).sum() == np.isfinite(data.counts).sum() - hidden.sum()
    loaded = load_csv(output / 'hron.csv', schema)
    with source.open(newline='') as stream:
        original_rows = list(csv.DictReader(stream))
    for i, row in enumerate(loaded.rows):
        for j, target in enumerate(data.targets):
            if np.isfinite(data.counts[i, j]):
                assert row[target] == original_rows[i][target]
        assert row['note'] == 'keep'
    assert np.isfinite(loaded.counts).all()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
    with pytest.raises(FileExistsError):
        run(source, schema, output, models={'hron': HronKNN})
