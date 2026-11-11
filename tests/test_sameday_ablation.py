"""Step 3: same_day_mean ablation switches (use_time, use_group, weighting, residual_rule).

    PYTHONPATH=src python -W error::RuntimeWarning -m pytest tests/test_sameday_ablation.py
"""
import copy
import hashlib

import numpy as np
import pytest

from longitudinal_impute import run
from longitudinal_impute.baselines.common import fill_within_residual
from longitudinal_impute.baselines.same_day_mean import SameDayMean
from longitudinal_impute.data import load_csv, make_mask
from test_run import COVID_SCHEMA, FIXTURES, MICROBIOME_SCHEMA, write

# sha256[:16] of run_model(...)[0].tobytes() computed with the code BEFORE the switches existed
# (commit c982e25), on the fixtures below. The defaults must reproduce them byte for byte.
GOLDEN = {
    'covid_small.csv:none:0': '71992e80698a288b', 'covid_small.csv:none:1': 'aceaa5057b6d947a',
    'covid_small.csv:none:2': '6fb6da6fb26959c7', 'covid_small.csv:wf:0': '239be9082a293960',
    'covid_small.csv:wf:1': 'a3812d5774482005', 'covid_small.csv:wf:2': 'ecb8b666928a4066',
    'covid_small.csv:rc:0': '3dcd8bb44a79d3ea', 'covid_small.csv:rc:1': 'da58783d22128120',
    'covid_small.csv:rc:2': 'e3b0617e7a474573', 'microbiome_small.csv:none:0': 'ee5fc82cc26d38a8',
    'microbiome_small.csv:none:1': 'ee5fc82cc26d38a8', 'microbiome_small.csv:none:2': 'ee5fc82cc26d38a8',
    'microbiome_small.csv:rc:0': 'ee485d72060c9e28', 'microbiome_small.csv:rc:1': '7a0b2bc79991f298',
    'microbiome_small.csv:rc:2': '7a0b2bc79991f298'}
PARAMS = [{}, {'window_days': 31, 'min_obs': 2, 'max_widen': 1},
          {'window_days': 31, 'min_obs': 2, 'max_widen': 1, 'exclude_same_location': True}]
DEFAULTS = {'use_time': True, 'use_group': True, 'weighting': 'n', 'residual_rule': 'cap'}
VARIANTS = {'sdm_full': {}, 'sdm_no_time': {'use_time': False}, 'sdm_no_group': {'use_group': False},
            'sdm_unweighted': {'weighting': 'none'}, 'sdm_rescale': {'residual_rule': 'rescale'},
            'sdm_none': {'use_time': False, 'use_group': False, 'weighting': 'none'}}


@pytest.mark.parametrize('explicit', [False, True])
def test_defaults_reproduce_the_previous_output_byte_for_byte(tmp_path, explicit):
    seen = {}
    for name, schema in (('covid_small.csv', COVID_SCHEMA), ('microbiome_small.csv', MICROBIOME_SCHEMA)):
        data = load_csv(FIXTURES / name, write(tmp_path, 'schema.toml', schema))
        masks = {'none': np.zeros(data.counts.shape, bool)}
        if name == 'covid_small.csv':
            masks['wf'] = make_mask(data, 'whole_feature', .2, 2)[0]
        masks['rc'] = make_mask(data, 'random_cell', .3, 4)[0]
        for kind, hidden in masks.items():
            for i, params in enumerate(PARAMS):
                model = SameDayMean(**params, **(DEFAULTS if explicit else {}))
                values = run.run_model(data, model, hidden)[0]
                seen[f'{name}:{kind}:{i}'] = hashlib.sha256(values.tobytes()).hexdigest()[:16]
    assert seen == GOLDEN


# Toy: target a is missing on row q (group G1, day 0, b = 0.1 visible so R = 0.9).
#   row  a    N   group day
#   r0  0.2  10   G1    0
#   r1  0.6  30   G1    5
#   r2  0.9  10   G1  100   (outside +-10 days)
#   r3  0.1  20   G2    0
TOY_A = [.2, .6, .9, .1, np.nan]
TOY_N = np.array([10., 30., 10., 20., 50.])
TOY_CONTEXT = {'totals': TOY_N, 'days': np.array([0., 5., 100., 0., 0.]), 'time_kind': 'date',
               'groups': ('G1', 'G1', 'G1', 'G2', 'G1'), 'entities': ('e0', 'e1', 'e2', 'e3', 'q')}


def toy_values():
    a = np.array(TOY_A)
    b = np.array([0., 0., 0., 0., .1])
    other = np.where(np.isfinite(a), 1 - a - b, np.nan)
    return np.column_stack((a, b, other))


@pytest.mark.parametrize('variant, expected_a, expected_other', [
    ('sdm_full', (2 + 18) / (10 + 30), .9 - .5),                       # G1 window: r0, r1, N-weighted
    ('sdm_no_time', (2 + 18 + 9) / (10 + 30 + 10), .9 - .58),          # all G1 days: r0, r1, r2
    ('sdm_no_group', (2 + 18 + 2) / (10 + 30 + 20), .9 - 22 / 60),     # window, both groups: r0, r1, r3
    ('sdm_unweighted', (.2 + .6) / 2, .9 - .4),                        # G1 window, plain mean of c/N
    ('sdm_none', (.2 + .6 + .9 + .1) / 4, .9 - .45),                   # every row, plain mean
    ('sdm_rescale', .9, 0.),                                           # 0.5 scaled up to R = 0.9
])
def test_each_switch_by_hand(variant, expected_a, expected_other):
    model = SameDayMean(10, min_obs=1, max_widen=0, **VARIANTS[variant])
    out = model.fit(toy_values(), TOY_CONTEXT).predict(toy_values())
    assert out[4, 0] == pytest.approx(expected_a, abs=1e-15)
    assert out[4, 1] == .1 and out[4, 2] == pytest.approx(expected_other, abs=1e-15)
    np.testing.assert_array_equal(out[:4], toy_values()[:4])
    assert model.info_['residual_rule'] == VARIANTS[variant].get('residual_rule', 'cap')


def test_rescale_zero_proposals_and_zero_residual_never_divide_by_zero():
    values = np.array([[.3, np.nan, np.nan, np.nan],      # proposals 0 -> Other keeps R = 0.7
                       [1., np.nan, np.nan, np.nan],      # R = 0, proposal > 0 -> fill 0
                       [.2, np.nan, np.nan, np.nan],      # proposals .1 + .3 -> scaled to .8, Other 0
                       [.5, .5, 0., 0.]])                 # complete row untouched
    proposal = np.array([[0., 0., 0.], [0., .4, .4], [0., .1, .3], [0., 0., 0.]])
    out, scaled = fill_within_residual(values, proposal, 'rescale')
    assert np.isfinite(out).all() and scaled == 2
    np.testing.assert_allclose(out[0], [.3, 0., 0., .7], atol=1e-15)
    np.testing.assert_allclose(out[1], [1., 0., 0., 1.], atol=1e-15)   # Other = 1 is a pure weight
    np.testing.assert_allclose(out[2], [.2, .2, .6, 0.], atol=1e-15)
    np.testing.assert_array_equal(out[3], values[3])
    cap, _ = fill_within_residual(values, proposal)
    np.testing.assert_allclose(cap[2], [.2, .1, .3, .4], atol=1e-15)   # cap leaves .4 to Other
    with pytest.raises(ValueError, match='residual rule'):
        fill_within_residual(values, proposal, 'other')


def test_rescale_fills_exactly_the_row_residual_in_counts(tmp_path):
    data = load_csv(FIXTURES / 'covid_small.csv', write(tmp_path, 'schema.toml', COVID_SCHEMA))
    hidden = make_mask(data, 'random_cell', .3, 4)[0]
    completed = run.run_model(data, SameDayMean(31, min_obs=2, max_widen=1, residual_rule='rescale'), hidden)[0]
    missing = hidden | ~np.isfinite(data.counts)
    filled = np.where(missing, completed, 0.).sum(axis=1)
    residual = data.totals - np.where(missing, 0., data.counts).sum(axis=1)
    rows = missing.any(axis=1) & (filled > 0)
    assert rows.sum() >= 5
    np.testing.assert_allclose(filled[rows], residual[rows], rtol=1e-12)


def grouped(data):
    data = copy.deepcopy(data)
    names = sorted(set(data.entities))
    data.groups = tuple(f'g{names.index(entity) % 2}' for entity in data.entities)
    return data


@pytest.mark.parametrize('kind', ['whole_feature', 'random_cell'])
@pytest.mark.parametrize('variant', sorted(VARIANTS))
def test_hidden_cells_never_change_the_output(tmp_path, variant, kind):
    data = grouped(load_csv(FIXTURES / 'covid_small.csv', write(tmp_path, 'schema.toml', COVID_SCHEMA)))
    assert len(set(data.groups)) == 2
    hidden = make_mask(data, kind, .2, 2)[0]
    poisoned = copy.deepcopy(data)
    poisoned.counts[hidden] = 1e100
    params = {'window_days': 31, 'min_obs': 2, 'max_widen': 1, **VARIANTS[variant]}
    first = run.run_model(data, SameDayMean(**params), hidden)[0]
    np.testing.assert_array_equal(first, run.run_model(poisoned, SameDayMean(**params), hidden)[0])


def test_switch_values_are_checked():
    for bad in ({'use_time': 'no'}, {'use_group': 1}, {'weighting': 'N'}, {'residual_rule': 'r1'}):
        with pytest.raises(ValueError):
            SameDayMean(**bad)


ALIASES = ('[same_day_mean]\nwindow_days = 31\nmin_obs = 2\nmax_widen = 1\n'
           '[sdm_no_time]\nmodel = "same_day_mean"\nwindow_days = 31\nmin_obs = 2\nmax_widen = 1\nuse_time = false\n'
           '[mic17]\nmodel = "micoda"\nmax_iter = 40\n')


def test_model_key_runs_the_registered_class_under_the_table_name(tmp_path):
    import json
    from test_run import call
    code, out = call(tmp_path, FIXTURES / 'covid_small.csv', COVID_SCHEMA, ALIASES,
                     '--mask', 'random_cell', '--mask-rate', '.2', '--seeds', '3', '--save-predictions')
    assert code == 0
    assert {p.name for p in out.glob('*_completed.csv')} == {
        'same_day_mean_completed.csv', 'sdm_no_time_completed.csv', 'mic17_completed.csv'}
    config = json.loads((out / 'config.json').read_text())
    assert config['models']['sdm_no_time']['model'] == 'same_day_mean'
    assert config['resolved_models']['sdm_no_time'] == {
        'model': 'same_day_mean', 'class': 'SameDayMean',
        'params': {'window_days': 31, 'min_obs': 2, 'max_widen': 1, 'use_time': False}}
    assert config['resolved_models']['mic17']['class'] == 'MICoDaCovid'
    summary = json.loads((out / 'summary.json').read_text())
    masked = summary['masked']['3']['models']
    assert set(masked) == {'same_day_mean', 'sdm_no_time', 'mic17'}
    assert masked['mic17']['space'] == '17_targets' and masked['sdm_no_time']['info']['use_time'] is False
    # The alias gives exactly what the class gives with the same parameters.
    data = load_csv(FIXTURES / 'covid_small.csv', write(tmp_path, 'schema.toml', COVID_SCHEMA))
    hidden = np.load(out / 'mask_random_cell_seed3.npy')
    direct = run.run_model(data, SameDayMean(31, min_obs=2, max_widen=1, use_time=False), hidden)[0]
    assert masked['sdm_no_time']['errors'] == run.masked_errors(data, direct, hidden)
    assert masked['sdm_no_time']['errors'] != masked['same_day_mean']['errors']


@pytest.mark.parametrize('models, message', [
    ('[sdm_x]\nmodel = "same_day_means"\n', r"\[sdm_x\] model = 'same_day_means' is not one of"),
    ('[sdm_x]\nmodel = 3\n', r'\[sdm_x\] model = 3 is not one of'),
    ('[sdm_x]\nuse_time = false\n', 'declare one or more'),
    ('[sdm_x]\nmodel = "same_day_mean"\nuse_time = "no"\n', 'use_time must be true or false'),
])
def test_bad_model_key_stops_before_any_output(tmp_path, models, message):
    from test_run import call
    with pytest.raises(ValueError, match=message):
        call(tmp_path, FIXTURES / 'covid_small.csv', COVID_SCHEMA, models, '--mask', 'random_cell')
    assert not (tmp_path / 'out').exists()
