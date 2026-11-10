"""Floors (zero_fill, mean_fill, marginal_sampler) and same_day_mean: count contract, seeds,
hidden-truth firewall, the same-day reference by hand, and how run.py selects them."""
import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from longitudinal_impute import run
from longitudinal_impute.baselines.common import fill_within_residual
from longitudinal_impute.baselines.marginal_sampler import MarginalSampler
from longitudinal_impute.baselines.mean_fill import MeanFill
from longitudinal_impute.baselines.same_day_mean import SameDayMean
from longitudinal_impute.baselines.zero_fill import ZeroFill
from longitudinal_impute.data import Dataset, load_csv, make_mask, model_input

FIXTURES = Path(__file__).resolve().parent / 'fixtures'
NEXTGEN = Path(__file__).resolve().parents[1]
SCHEMA = ('[dataset]\nentity="location"\ntime="date"\ntotal="total_sequence"\ntargets=["v1","v2","v3","v4"]\n'
          'group_col="continent"\ngroup_map="map.csv"\n')
MAP = 'location,continent\nAvalon,North\nBrill,North\nCorin,South\n'
FACTORIES = {
    'zero_fill': ZeroFill,
    'mean_fill': MeanFill,
    'marginal_sampler': lambda: MarginalSampler(seed=3),
    'same_day_mean': lambda: SameDayMean(31, min_obs=2, max_widen=1),
    'same_day_mean_exclude': lambda: SameDayMean(31, min_obs=2, max_widen=1, exclude_same_location=True),
}


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


@pytest.fixture
def covid(tmp_path):
    write(tmp_path, 'map.csv', MAP)
    return load_csv(FIXTURES / 'covid_small.csv', write(tmp_path, 'schema.toml', SCHEMA))


def masks(data):
    yield np.zeros(data.counts.shape, bool)
    for kind in ('random_cell', 'whole_feature'):
        yield make_mask(data, kind, .3, 2)[0]


@pytest.mark.parametrize('name', FACTORIES)
def test_counts_contract_on_native_and_masked_passes(covid, name):
    for hidden in masks(covid):
        completed, status, info = run.run_model(covid, FACTORIES[name](), hidden)   # runs check_completed
        visible = np.isfinite(covid.counts) & ~hidden
        assert status == 'completed' and info['filled_cells'] == (~visible).sum()
        np.testing.assert_array_equal(completed[visible], covid.counts[visible])
        assert np.isfinite(completed).all() and (completed >= 0).all()
        assert (completed.sum(axis=1) <= covid.totals * (1 + 1e-10)).all()


@pytest.mark.parametrize('name', FACTORIES)
def test_same_seed_same_output_and_hidden_truth_is_never_read(covid, name):
    hidden = make_mask(covid, 'whole_feature', .3, 2)[0]
    first = run.run_model(covid, FACTORIES[name](), hidden)[0]
    np.testing.assert_array_equal(first, run.run_model(covid, FACTORIES[name](), hidden)[0])
    poisoned = copy.deepcopy(covid)
    poisoned.counts[hidden] = 1e100
    np.testing.assert_array_equal(first, run.run_model(poisoned, FACTORIES[name](), hidden)[0])
    random_hidden = make_mask(covid, 'random_cell', .3, 4)[0]
    first = run.run_model(covid, FACTORIES[name](), random_hidden)[0]
    poisoned = copy.deepcopy(covid)
    poisoned.counts[random_hidden] = 1e100
    np.testing.assert_array_equal(first, run.run_model(poisoned, FACTORIES[name](), random_hidden)[0])


def test_marginal_sampler_draws_from_the_observed_cells_of_the_same_target(covid):
    hidden = make_mask(covid, 'random_cell', .3, 1)[0]
    values = model_input(covid, hidden)
    model = MarginalSampler(seed=5).fit(values)
    out = model.predict(values)
    unscaled = np.isclose(np.nansum(values[:, :-1], axis=1) + out[:, -1] + np.where(
        np.isnan(values[:, :-1]), out[:, :-1], 0).sum(axis=1), 1) & (out[:, -1] > 0)
    for i, j in zip(*np.nonzero(np.isnan(values[:, :-1]))):
        if unscaled[i]:
            assert out[i, j] in model.pools_[j]
    other = MarginalSampler(seed=6).fit(values).predict(values)
    assert not np.array_equal(out, other)
    with pytest.raises(ValueError, match='seed'):
        MarginalSampler(seed=-1)


def test_zero_fill_and_mean_fill_values(covid):
    hidden = make_mask(covid, 'random_cell', .3, 1)[0]
    missing = ~(np.isfinite(covid.counts) & ~hidden)
    zero = run.run_model(covid, ZeroFill(), hidden)[0]
    assert (zero[missing] == 0).all()
    values = model_input(covid, hidden)
    means = np.nanmean(values[:, :-1], axis=0)
    mean = run.run_model(covid, MeanFill(), hidden)[0]
    residual = 1 - np.nansum(values[:, :-1], axis=1)
    fits = (np.where(missing, means, 0).sum(axis=1) <= residual)
    for i, j in zip(*np.nonzero(missing)):
        if fits[i]:
            assert mean[i, j] == pytest.approx(means[j] * covid.totals[i], rel=1e-12)
        else:      # scaled down to the residual: the row is then exactly full
            assert mean[i].sum() == pytest.approx(covid.totals[i], rel=1e-12)


def test_fill_within_residual_caps_at_the_residual_and_gives_the_rest_to_other():
    values = np.array([[.5, np.nan, np.nan, np.nan],     # R = .5, proposals .1 + .2 -> Other .2
                       [.5, np.nan, np.nan, np.nan],     # proposals .6 + .4 > .5 -> scaled to .3 + .2
                       [1., np.nan, np.nan, np.nan],     # R = 0
                       [.2, .3, .5, 0.]])                # complete row untouched
    proposal = np.array([[0, .1, .2], [0, .6, .4], [0, .3, .3], [9, 9, 9]], float)
    out, scaled = fill_within_residual(values, proposal)
    np.testing.assert_allclose(out[0], [.5, .1, .2, .2])
    np.testing.assert_allclose(out[1], [.5, .3, .2, 0])
    np.testing.assert_allclose(out[2], [1, 0, 0, 1])     # Other = 1 is a weight; restore_counts skips R = 0
    np.testing.assert_array_equal(out[3], values[3])
    assert scaled == 2


def toy(time_kind='date'):
    """Five rows, target a missing in row 0 only; target b observed everywhere."""
    counts = np.array([[np.nan, 10.], [20, 10], [30, 5], [9, 1], [10, 0]])
    totals = np.array([100., 100, 50, 10, 10])
    days = np.array([0, 0, 10, 0, 3])
    return Dataset(counts, totals, ('a', 'b'), ('A', 'B', 'C', 'D', 'A'), [], [], [],
                   days=days if time_kind else None, week=None, time_kind=time_kind,
                   groups=('g1', 'g1', 'g1', 'g2', 'g1'))


@pytest.mark.parametrize('params,expected,level', [
    # +-5 days in g1 without row 0: rows 1 and 4 -> (20 + 10) / (100 + 10)
    (dict(window_days=5, min_obs=2), 30 / 110, 'window'),
    # the same location (row 4) excluded: row 1 alone
    (dict(window_days=5, min_obs=1, exclude_same_location=True), 20 / 100, 'window'),
    # 1 cell left in +-5 -> doubled to +-10: rows 1 and 2 -> (20 + 30) / (100 + 50)
    (dict(window_days=5, min_obs=2, exclude_same_location=True), 50 / 150, 'widened_1'),
    # still short after one doubling -> global over every group: (20 + 30 + 9 + 10) / 170
    (dict(window_days=5, min_obs=4, max_widen=1), 69 / 170, 'global'),
    # global without the same location: (20 + 30 + 9) / 160
    (dict(window_days=5, min_obs=4, max_widen=1, exclude_same_location=True), 59 / 160, 'global'),
])
def test_same_day_reference_by_hand(params, expected, level):
    data = toy()
    model = SameDayMean(**params)
    completed = run.run_model(data, model, np.zeros(data.counts.shape, bool))[0]
    assert completed[0, 0] == pytest.approx(expected * 100, rel=1e-12)     # below R = 90, so not capped
    assert model.info_['cells_by_level'][level] == 1 and sum(model.info_['cells_by_level'].values()) == 1
    assert model.levels_[0, 0] == list(model.info_['cells_by_level']).index(level)
    assert (model.levels_[1:] == -1).all() and (model.levels_[0, 1] == -1)


def test_same_day_windows_follow_the_time_kind():
    model = SameDayMean(28, max_widen=2)
    assert model.windows('date') == [28, 56, 112]
    assert model.windows('numeric') == [0, 1, 2]           # same visit, then +-1, +-2
    assert model.windows('label') == [0]                   # same label only
    assert model.windows(None) == [np.inf]                 # no time column: the whole group
    # numeric visit index: same visit first. Rows 1 and 3 share visit 0 but row 3 is in g2.
    data = toy('numeric')
    completed = run.run_model(data, SameDayMean(min_obs=1), np.zeros(data.counts.shape, bool))[0]
    assert completed[0, 0] == pytest.approx(20.)
    # no time column: every other row of the group
    data = toy(None)
    completed = run.run_model(data, SameDayMean(min_obs=1), np.zeros(data.counts.shape, bool))[0]
    assert completed[0, 0] == pytest.approx(100 * 60 / 160)


def test_excluding_the_location_can_leave_nothing_and_says_so(covid):
    # whole_feature seed 4 hides v1 of Avalon and Corin; Brill row 10 misses v1 natively, so
    # without Brill itself no visible v1 remains anywhere: an explicit error, no silent fallback.
    hidden = make_mask(covid, 'whole_feature', .3, 4)[0]
    with pytest.raises(ValueError, match='row 10, feature 0: no visible cell'):
        run.run_model(covid, FACTORIES['same_day_mean_exclude'](), hidden)
    run.run_model(covid, FACTORIES['same_day_mean'](), hidden)       # Brill's own v1 cells are allowed


def test_same_day_mean_parameters_are_checked():
    for bad in (dict(window_days=-1), dict(min_obs=0), dict(max_widen=1.5), dict(exclude_same_location='yes')):
        with pytest.raises(ValueError):
            SameDayMean(**bad)
    with pytest.raises(ValueError, match='totals'):
        SameDayMean().fit(np.full((2, 3), np.nan))


# --- run.py: selection and end to end -------------------------------------------------------

def test_default_config_still_runs_only_the_three_paper_baselines():
    models = run.read_models(NEXTGEN / 'configs/models_default.toml')
    assert list(models) == ['hron', 'tsagris', 'micoda']
    picked = run.read_models(NEXTGEN / 'configs/models_default.toml', ['same_day_mean', 'hron'])
    assert list(picked) == ['same_day_mean', 'hron']
    assert picked['same_day_mean'] == {'window_days': 28, 'min_obs': 20, 'max_widen': 2,
                                       'exclude_same_location': False}


def test_model_selection_errors(tmp_path):
    with pytest.raises(ValueError, match='missing'):
        run.read_models(write(tmp_path, 'm.toml', '[hron]\n'), ['zero_fill'])
    with pytest.raises(ValueError, match='enabled must be true or false'):
        run.read_models(write(tmp_path, 'm.toml', '[hron]\nenabled = "no"\n'))
    with pytest.raises(ValueError, match='every model has enabled = false'):
        run.read_models(write(tmp_path, 'm.toml', '[zero_fill]\nenabled = false\n'))
    with pytest.raises(ValueError, match='window_days'):
        run.read_models(write(tmp_path, 'm.toml', '[same_day_mean]\nwindow_days = -3\n'))


def test_run_end_to_end_with_the_new_models(tmp_path):
    write(tmp_path, 'map.csv', MAP)
    models = ('[hron]\n[zero_fill]\nenabled = false\n[mean_fill]\nenabled = false\n'
              '[marginal_sampler]\nenabled = false\nseed = 1\n'
              '[same_day_mean]\nenabled = false\nmin_obs = 2\n')
    out = tmp_path / 'out'
    code = run.main(['--data', str(FIXTURES / 'covid_small.csv'), '--schema', str(write(tmp_path, 'schema.toml', SCHEMA)),
                     '--models', str(write(tmp_path, 'models.toml', models)), '--output', str(out),
                     '--mask', 'whole_feature', '--mask-rate', '.2', '--seeds', '1', '2',
                     '--only', 'zero_fill', 'mean_fill', 'marginal_sampler', 'same_day_mean'])
    assert code == 0 and not (out / 'errors.log').exists()
    names = ('zero_fill', 'mean_fill', 'marginal_sampler', 'same_day_mean')
    with (out / 'summary.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    assert [(r['model'], r['pass'], r['status'], r['space']) for r in rows] == [
        (m, p, 'ok', '18_with_other') for p, s in (('native', ''), ('masked', 1), ('masked', 2))
        for m in names]
    assert all(r['mse_proportion_model'] for r in rows if r['pass'] == 'masked')
    summary = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
    levels = summary['masked']['1']['models']['same_day_mean']['info']['cells_by_level']
    assert set(levels) == {'window', 'widened_1', 'widened_2', 'global'}
    config = json.loads((out / 'config.json').read_text(encoding='utf-8'))
    assert list(config['models']) == list(names) and config['only'] == list(names)
    assert {p.name for p in out.glob('*_completed.csv')} == {f'{m}_completed.csv' for m in names}
