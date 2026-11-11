"""Step 2 (tier 0-1): empty groups, series-macro MAE, the long prediction file, model input.

    PYTHONPATH=src python -W error::RuntimeWarning -m pytest tests/test_tier01_eval.py
"""
import copy
import json

import numpy as np
import pytest

from longitudinal_impute import run
from longitudinal_impute.baselines import micoda
from longitudinal_impute.cells import COLUMNS, read_predictions, score_predictions
from longitudinal_impute.data import load_csv, make_mask
from longitudinal_impute.evaluate import GROUPS, METRICS, cell_errors, masked_errors, series_macro_mae
from test_run import COVID_SCHEMA, FIXTURES, call, write


def test_empty_group_is_na_not_zero():
    truth, pred, totals = np.array([0., 3.]), np.array([0., 3.]), np.array([3., 3.])
    errors = cell_errors(truth, pred, totals, forced=np.array([True, True]))   # every cell forced to 0
    assert errors['model'] == {'cells': 0, **dict.fromkeys(METRICS)}
    assert errors['model_truth_zero']['mae_proportion'] is None
    assert errors['forced_zero']['mae_proportion'] == 0.                      # a real 0 stays 0
    assert series_macro_mae(truth, pred, totals, np.array(['a', 'b']), np.array([False, False])) is None


def test_cell_weighted_and_series_macro_differ_by_hand():
    # Series a: three cells with |error| 1/10; series b: one cell with |error| 5/10.
    truth = np.zeros(4)
    pred = np.array([1., 1., 1., 5.])
    totals = np.full(4, 10.)
    series = np.array(['a', 'a', 'a', 'b'])
    micro = cell_errors(truth, pred, totals, np.zeros(4, bool))['model']['mae_proportion']
    macro = series_macro_mae(truth, pred, totals, series, np.ones(4, bool))
    assert micro == pytest.approx((.1 + .1 + .1 + .5) / 4, abs=1e-15)          # 0.2
    assert macro == pytest.approx((.1 + .5) / 2, abs=1e-15)                    # 0.3
    # Only the selected cells count: dropping series b leaves 0.1 for both.
    keep = np.array([True, True, True, False])
    assert series_macro_mae(truth, pred, totals, series, keep) == pytest.approx(.1, abs=1e-15)


@pytest.mark.parametrize('mask', ['whole_feature', 'random_cell'])
def test_long_file_scores_match_masked_errors(tmp_path, mask):
    models = '[hron]\n[tsagris]\n[micoda]\n[same_day_mean]\nmin_obs = 2\n[mean_fill]\n[zero_fill]\n'
    code, out = call(tmp_path, FIXTURES / 'covid_small.csv', COVID_SCHEMA, models,
                     '--mask', mask, '--mask-rate', '.2', '--seeds', '1', '2', '--save-predictions')
    assert code == 0
    summary = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
    rows = read_predictions(out / f'predictions_{mask}.csv.gz')
    scores = score_predictions(rows)
    data = load_csv(FIXTURES / 'covid_small.csv', write(tmp_path, 'schema.toml', COVID_SCHEMA))
    assert {k[0] for k in scores} == {'hron', 'tsagris', 'micoda', 'same_day_mean', 'mean_fill', 'zero_fill'}
    for (model, mask_name, seed), score in scores.items():
        stored = summary['masked'][str(seed)]['models'][model]['errors']
        hidden = np.load(out / f'mask_{mask}_seed{seed}.npy')
        assert mask_name == mask and score['errors']['all']['cells'] == hidden.sum()
        stats = summary['masked'][str(seed)]['mask']
        assert (stats['forced_zero_cells'], stats['model_cells']) == (
            score['errors']['forced_zero']['cells'], score['errors']['model']['cells'])
        for group in GROUPS:
            for key in ('cells', *METRICS):
                a, b = score['errors'][group][key], stored[group][key]
                assert (a is None) == (b is None) and (a is None or abs(a - b) <= 1e-12), (model, seed, group, key)
    # The truth columns are the data; the long file keeps the row-major order of masked_errors.
    first = [r for r in rows if r['model'] == 'hron' and r['seed'] == 1]
    hidden = np.load(out / f'mask_{mask}_seed1.npy')
    assert [(r['row_id'], data.targets.index(r['target'])) for r in first] == list(zip(*np.nonzero(hidden)))
    assert all(r['truth_count'] == data.counts[r['row_id'], data.targets.index(r['target'])] for r in first)
    assert all(r['date'] and r['truth_prop'] == r['truth_count'] / r['N'] for r in first)


def test_predictions_are_opt_in_and_file_is_deterministic(tmp_path):
    args = ('--mask', 'random_cell', '--mask-rate', '.2', '--seeds', '3')
    _, plain = call(tmp_path, FIXTURES / 'covid_small.csv', COVID_SCHEMA, '[mean_fill]\n', *args)
    assert not list(plain.glob('predictions_*'))
    hashes = []
    for name in ('a', 'b'):
        (tmp_path / name).mkdir()
        _, out = call(tmp_path / name, FIXTURES / 'covid_small.csv', COVID_SCHEMA, '[mean_fill]\n', *args,
                      '--save-predictions')
        hashes.append((out / 'predictions_random_cell.csv.gz').read_bytes())
    assert hashes[0] == hashes[1]
    rows = read_predictions(tmp_path / 'a' / 'out' / 'predictions_random_cell.csv.gz')
    assert rows and tuple(rows[0]) == COLUMNS


def recorded_inputs(monkeypatch, data, hidden, factory):
    """Every array handed to fit/predict (and MICoDa's core fit) while run_model runs."""
    seen = []

    def spy(cls, method):
        original = getattr(cls, method)

        def wrapper(self, *args, **kwargs):
            seen.append(copy.deepcopy((args, kwargs)))
            return original(self, *args, **kwargs)
        monkeypatch.setattr(cls, method, wrapper)

    model = factory()
    if isinstance(model, micoda.MICoDaCovid):
        spy(micoda.MICoDa, 'fit')
    else:
        spy(type(model), 'fit')
        spy(type(model), 'predict')
    output = run.run_model(data, model, hidden)[0]
    monkeypatch.undo()
    return seen, output


def assert_same(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_same(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_same(x, y)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    else:
        assert a == b


FACTORIES = {name: cls for name, cls in run.MODELS.items()}
FACTORIES['micoda_other'] = lambda: micoda.MICoDaCovid(other=True)


@pytest.mark.parametrize('kind', ['whole_feature', 'random_cell'])
@pytest.mark.parametrize('name', sorted(FACTORIES))
def test_changing_hidden_values_leaves_model_input_unchanged(tmp_path, monkeypatch, name, kind):
    data = load_csv(FIXTURES / 'covid_small.csv', write(tmp_path, 'schema.toml', COVID_SCHEMA))
    hidden = make_mask(data, kind, .2, 2)[0]
    assert hidden.any() and (data.counts[hidden] > 0).any()
    changed = copy.deepcopy(data)
    changed.counts[hidden] = 0.                 # still within N: a valid table with other truths
    first, out_first = recorded_inputs(monkeypatch, data, hidden, FACTORIES[name])
    second, out_second = recorded_inputs(monkeypatch, changed, hidden, FACTORIES[name])
    assert first, f'{name}: no fit call was recorded'
    assert_same(first, second)
    np.testing.assert_array_equal(out_first[~hidden], out_second[~hidden])
    np.testing.assert_array_equal(out_first[hidden], out_second[hidden])


def test_failed_run_keeps_an_error_code(tmp_path):
    source = write(tmp_path, 'one_complete.csv',
                   'location,date,total_sequence,v1,v2,v3,v4\nA,1,100,10,20,30,40\nB,1,100,10,,,\n')
    code, out = call(tmp_path, source, COVID_SCHEMA, '[micoda]\n[tsagris]\n', '--save-predictions')
    assert code == 1
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['native']['micoda']['error_code'] == 'insufficient_fit'
    assert summary['native_cells']['missing_model_cells'] + summary['native_cells']['missing_forced_zero_cells'] == 3
