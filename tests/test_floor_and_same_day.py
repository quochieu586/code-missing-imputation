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
ROOT = Path(__file__).resolve().parents[1]     # repository root
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

def test_default_config_runs_the_paper_baselines_and_same_day_mean_but_no_floor():
    models = run.read_models(ROOT / 'configs/models_default.toml')
    assert list(models) == ['hron', 'tsagris', 'micoda', 'same_day_mean']
    picked = run.read_models(ROOT / 'configs/models_default.toml', ['same_day_mean', 'hron'])
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


# --- F1 (PR #4 review): endpoint lookup, checked against a direct-donor oracle and for memory ----

def oracle_same_day(values, totals, days, groups, entities, model, kind='date'):
    """Loop over every missing cell and every donor row: the definition, without prefix sums."""
    targets = values[:, :-1]
    proposal = np.zeros(targets.shape)
    levels = np.full(targets.shape, -1)
    windows = model.windows(kind)
    for i, j in zip(*np.nonzero(np.isnan(targets))):
        donors = [k for k in range(len(values)) if k != i and np.isfinite(targets[k, j])
                  and not (model.exclude_same_location and entities[k] == entities[i])]
        for step, width in enumerate(windows):
            inside = [k for k in donors if groups[k] == groups[i] and abs(days[k] - days[i]) <= width]
            if len(inside) >= model.min_obs:
                break
        else:
            inside, step = donors, len(windows)
        proposal[i, j] = sum(targets[k, j] * totals[k] for k in inside) / sum(totals[k] for k in inside)
        levels[i, j] = step
    return fill_within_residual(values, proposal)[0], levels


@pytest.mark.parametrize('seed', range(6))
@pytest.mark.parametrize('exclude', [False, True])
def test_same_day_mean_matches_a_direct_donor_oracle(seed, exclude):
    rng = np.random.default_rng(seed)
    n, d = 60, 3
    parts = rng.dirichlet(np.ones(d + 1), n)
    values = parts.copy()
    values[:, :-1][rng.random((n, d)) < .3] = np.nan
    values[np.isnan(values[:, :-1]).any(axis=1), -1] = np.nan
    totals = rng.integers(1, 500, n).astype(float)
    days = rng.integers(0, 40, n) * 3
    groups = [f'G{k}' for k in rng.integers(0, 3, n)]
    entities = [f'E{k}' for k in rng.integers(0, 8, n)]
    model = SameDayMean(5, min_obs=3, max_widen=2, exclude_same_location=exclude)
    model.fit(values, {'totals': totals, 'days': days, 'time_kind': 'date',
                       'groups': groups, 'entities': entities})
    out = model.predict(values)
    expected, levels = oracle_same_day(values, totals, days, groups, entities, model)
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-15)
    np.testing.assert_array_equal(model.levels_, levels)
    assert len(set(levels[levels >= 0].tolist())) > 1        # several fallback levels exercised


def test_same_day_mean_predict_memory_does_not_grow_with_rows_times_days():
    import tracemalloc
    n = 4000                                    # 4000 rows, 4000 distinct days, one group
    rng = np.random.default_rng(0)
    a, b = rng.uniform(0, .5, n), rng.uniform(0, .5, n)
    values = np.column_stack((a, b, 1 - a - b))
    values[::2, 0] = values[::2, 2] = np.nan
    for exclude in (False, True):
        model = SameDayMean(28, min_obs=20, exclude_same_location=exclude).fit(values, {
            'totals': rng.integers(1, 1000, n).astype(float), 'days': np.arange(n), 'time_kind': 'date',
            'groups': None, 'entities': [f'E{k % 10}' for k in range(n)]})
        tracemalloc.start()
        try:
            model.predict(values)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        # Before F1: ~185 MiB (exclude=False) and ~370 MiB (True); after: ~0.9 and ~2.7 MiB.
        assert peak < 16 * 2 ** 20, f'peak {peak / 2 ** 20:.1f} MiB'


# --- P3 (PR #4 review): extreme totals never give a silent 'completed' ---------------------------

def extreme(totals, counts, days):
    counts = np.array(counts, float)
    return Dataset(counts, np.array(totals, float), ('a', 'b'), tuple(f'E{k}' for k in range(len(counts))),
                   [], [], [], days=np.array(days), week=None, time_kind='date', groups=None)


TV1_CASES = {
    # two donors N = 1e308, a = 5e307: sum N overflows; before P3 the query got a = 0, completed
    'overflow': extreme([100, 1e308, 1e308], [[np.nan, 0], [5e307, 5e307], [5e307, 5e307]], [0, 0, 0]),
    # donor outside the window with N = 1e20, in-window donor N = 100, a = 20: the prefix-sum
    # difference cancels the in-window weight to 0; before P3 this ended in 0/0
    'cancellation': extreme([100, 100, 1e20], [[np.nan, 0], [20, 80], [5e19, 5e19]], [1000, 1000, 0]),
}


@pytest.mark.parametrize('case', TV1_CASES)
@pytest.mark.parametrize('warnings_as', ['ignore', 'error'])
def test_extreme_totals_are_rejected_not_filled(case, warnings_as):
    import warnings
    data = TV1_CASES[case]
    with warnings.catch_warnings():
        warnings.simplefilter(warnings_as, RuntimeWarning)
        with pytest.raises(ValueError, match=r'at most 2\*\*53'):
            run.run_model(data, SameDayMean(min_obs=1), np.zeros(data.counts.shape, bool))


@pytest.mark.parametrize('case', TV1_CASES)
def test_window_sums_that_overflow_or_cancel_raise_numerical_error(case):
    # Bypass the fit() domain check to exercise the second guard inside predict().
    from longitudinal_impute.errors import NumericalError
    data = TV1_CASES[case]
    values = model_input(data, np.zeros(data.counts.shape, bool))       # proportions are valid
    model = SameDayMean(min_obs=1).fit(values, {'totals': np.ones(len(values)), 'days': data.days,
                                                'time_kind': 'date'})
    model.totals_ = data.totals.copy()
    with pytest.raises(NumericalError, match='sums are not usable'):
        model.predict(values)
