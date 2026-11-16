"""Step 4: one residual rule for every model (prop_other, cap, rescale) at the runner's restore layer.

    PYTHONPATH=src python -W error::RuntimeWarning -m pytest tests/test_residual_rule.py
"""
import copy
import hashlib
import json

import numpy as np
import pytest

from longitudinal_impute import run
from longitudinal_impute.baselines.micoda import MICoDaCovid
from longitudinal_impute.data import Dataset, apply_residual_rule, check_completed, load_csv, make_mask
from longitudinal_impute.evaluation.metrics import implied_other_errors
from test_run import COVID_SCHEMA, FIXTURES, call, write

# sha256[:16] of run_model(...) values + status, computed with the code BEFORE step 4 (commit
# 6948635) on covid_small: no residual_rule key must keep every model byte for byte.
GOLDEN = {
    'hron:none': '12482432cdcf047f', 'hron:wf': '762f1445da35f474', 'hron:rc': '0b5175653f9767a9',
    'marginal_sampler:none': '98eca40aee4d8c36', 'marginal_sampler:wf': '1cc205d0d1840a0f',
    'marginal_sampler:rc': 'fa5d971bd36c0e82', 'mean_fill:none': '1efbf9402b473053',
    'mean_fill:wf': '6f334c3ee1e8db51', 'mean_fill:rc': 'ae0cf233f2a6ae66', 'micoda:none': 'c2cd40b2dbc0e03b',
    'micoda:wf': 'eaeb9f7efa425568', 'micoda:rc': '9c13775bbb3a0404', 'micoda_other:none': '5086a8e3b5c5d654',
    'micoda_other:wf': '7afa52be66e5e2f8', 'micoda_other:rc': 'c97a581d19390f53',
    'same_day_mean:none': 'fc403bddad9be60a', 'same_day_mean:wf': 'd60d4eb53382190d',
    'same_day_mean:rc': 'c010b792766a8e67', 'tsagris:none': 'dd4d2ae349cec143', 'tsagris:wf': '4eb826bb4e24f1b6',
    'tsagris:rc': '9b7689fea480b558', 'zero_fill:none': 'd21af58fd7d9abc5', 'zero_fill:wf': 'aa45737c20916e8e',
    'zero_fill:rc': '1ebd621c6f239e28'}
# The step's models, A17 written out: MICoDa defaults to A18 since PR #5, and 'main' (the main model,
# PR #5) is not a baseline of this step.
FACTORIES = {**{n: c for n, c in run.MODELS.items() if n not in ('same_day_mean', 'micoda', 'main')},
             'micoda': lambda: MICoDaCovid(other=False),
             'same_day_mean': lambda: run.MODELS['same_day_mean'](31, min_obs=2, max_widen=1),
             'micoda_other': lambda: MICoDaCovid(other=True)}
NEW = {('hron', 'cap'): run.MODELS['hron'], ('hron', 'rescale'): run.MODELS['hron'],
       ('tsagris', 'cap'): run.MODELS['tsagris'], ('tsagris', 'rescale'): run.MODELS['tsagris'],
       ('micoda_other', 'cap'): FACTORIES['micoda_other'], ('micoda_other', 'rescale'): FACTORIES['micoda_other']}


@pytest.fixture
def covid(tmp_path):
    return load_csv(FIXTURES / 'covid_small.csv', write(tmp_path, 'schema.toml', COVID_SCHEMA))


def masks(data):
    return {'none': np.zeros(data.counts.shape, bool), 'wf': make_mask(data, 'whole_feature', .2, 2)[0],
            'rc': make_mask(data, 'random_cell', .3, 4)[0]}


# MICoDa (micoda, micoda_other) iterates to convergence, so its floating-point result depends on the
# platform and the numpy/BLAS build (e.g. micoda_other:wf differs in the last bits on Linux with numpy
# 2.5.5). Its golden is therefore the array itself, compared to rtol = atol = 1e-9 with the status
# compared exactly. The arrays were written from the same pre-step-4 code and reproduce the GOLDEN
# hashes below byte for byte where they were made (macOS, numpy 2.1.3); every other model is
# compared byte for byte.
ITERATIVE = ('micoda', 'micoda_other')
GOLDEN_ARRAYS = json.loads((FIXTURES / 'residual_rule_micoda_golden.json').read_text(encoding='utf-8'))


def test_without_a_rule_every_model_is_unchanged_byte_for_byte(covid):
    seen, close = {}, set()
    for name, make in sorted(FACTORIES.items()):
        for kind, hidden in masks(covid).items():
            values, status, _ = run.run_model(covid, make(), hidden)
            key = f'{name}:{kind}'
            if name in ITERATIVE:
                golden = GOLDEN_ARRAYS[key]
                assert status == golden['status'], key
                expected = np.array([[np.nan if v is None else v for v in row] for row in golden['values']])
                np.testing.assert_allclose(values, expected, rtol=1e-9, atol=1e-9, err_msg=key)
                close.add(key)
            else:
                seen[key] = hashlib.sha256(values.tobytes() + status.encode()).hexdigest()[:16]
    assert seen == {k: v for k, v in GOLDEN.items() if k.split(':')[0] not in ITERATIVE}
    assert close == set(GOLDEN_ARRAYS) == {k for k in GOLDEN if k.split(':')[0] in ITERATIVE}
    assert all(GOLDEN_ARRAYS[k]['sha256_16'] == GOLDEN[k] for k in close)


@pytest.mark.parametrize('name', ['hron', 'tsagris'])
def test_prop_other_through_the_shared_function_is_the_current_output(covid, name):
    for hidden in masks(covid).values():
        np.testing.assert_array_equal(run.run_model(covid, FACTORIES[name](), hidden)[0],
                                      run.run_model(covid, FACTORIES[name](), hidden, 'prop_other')[0])


def test_micoda_other_prop_other_matches_the_adapter_r2_to_rounding(covid):
    for hidden in masks(covid).values():
        old = run.run_model(covid, FACTORIES['micoda_other'](), hidden)[0]
        new = run.run_model(covid, FACTORIES['micoda_other'](), hidden, 'prop_other')[0]
        np.testing.assert_allclose(new, old, rtol=1e-12, atol=1e-9)


def toy(counts, totals, hidden):
    counts = np.array(counts, float)
    return Dataset(counts, np.array(totals, float), tuple('abc'[:counts.shape[1]]), ('e',) * len(counts),
                   [], [], []), np.array(hidden)


# Row: N = 100, a = 40 visible, b hidden -> R = 60. Compositions over (a, b, Other).
@pytest.mark.parametrize('p, prop_other, cap, rescale', [
    ([.5, .2, .3], 60 * .2 / .5, 20., 60.),        # sum q_U = 20 < R
    ([.3, .6, .1], 60 * .6 / .7, 60., 60.),        # sum q_U = 60 = R
    ([.2, .7, .1], 60 * .7 / .8, 60., 60.),        # sum q_U = 70 > R: cap scales down to R
    ([.5, 0., .5], 0., 0., 0.),                    # sum q_U = 0: rescale keeps the cap outcome
    ([1., .4, .6], 60 * .4 / 1., 20., 60.),        # not closed (sum 2): closed to [.5, .2, .3] first
])
def test_three_rules_by_hand_one_missing(p, prop_other, cap, rescale):
    data, hidden = toy([[40., 10.]], [100.], [[False, True]])
    for rule, expected in (('prop_other', prop_other), ('cap', cap), ('rescale', rescale)):
        out, info = apply_residual_rule(data, np.array([p]), hidden, rule)
        assert out[0, 0] == 40. and out[0, 1] == pytest.approx(expected, abs=1e-12), rule
        check_completed(data, out, hidden)
        assert info['rule'] == rule


def test_three_rules_by_hand_two_missing_and_zero_residual():
    # N = 100, c = 50 visible (R = 50), a and b hidden; then a row with R = 0.
    data, hidden = toy([[20., 30., 50.], [0., 0., 100.]], [100., 100.], [[True, True, False], [True, True, False]])
    expected = {  # p over (a, b, c, Other) of row 0
        'over': ([.3, .3, .2, .2], {'prop_other': (18.75, 18.75), 'cap': (25., 25.), 'rescale': (25., 25.)}),
        'under': ([.1, .2, .5, .2], {'prop_other': (10., 20.), 'cap': (10., 20.), 'rescale': (50 / 3, 100 / 3)})}
    for p, results in expected.values():
        for rule, (a, b) in results.items():
            out, _ = apply_residual_rule(data, np.array([p, [.3, .3, .2, .2]]), hidden, rule)
            np.testing.assert_allclose(out[0], [a, b, 50.], atol=1e-12)
            np.testing.assert_array_equal(out[1], [0., 0., 100.])            # R = 0: nothing to share
            check_completed(data, out, hidden)


def test_shared_function_rejects_bad_input():
    data, hidden = toy([[40., 10.]], [100.], [[False, True]])
    with pytest.raises(ValueError, match='residual rule'):
        apply_residual_rule(data, np.array([[.5, .2, .3]]), hidden, 'r1')
    with pytest.raises(ValueError, match='finite'):
        apply_residual_rule(data, np.array([[.5, np.nan, .3]]), hidden, 'cap')


@pytest.mark.parametrize('kind', ['wf', 'rc', 'none'])
@pytest.mark.parametrize('variant', sorted(NEW))
def test_new_variants_keep_the_count_contract_and_ignore_hidden_values(covid, variant, kind):
    name, rule = variant
    hidden = masks(covid)[kind]
    first, status, info = run.run_model(covid, NEW[variant](), hidden, rule)
    assert status == 'completed' and info['residual_rule']['rule'] == rule
    visible = np.isfinite(covid.counts) & ~hidden
    np.testing.assert_array_equal(first[visible], covid.counts[visible])
    assert (first >= 0).all() and (first.sum(axis=1) <= covid.totals * (1 + 1e-12)).all()
    check_completed(covid, first, hidden)
    poisoned = copy.deepcopy(covid)
    poisoned.counts[hidden] = 1e100
    np.testing.assert_array_equal(first, run.run_model(poisoned, NEW[variant](), hidden, rule)[0])


def test_tsagris_cap_equals_prop_other_because_its_rows_are_closed(covid):
    for hidden in masks(covid).values():
        np.testing.assert_allclose(run.run_model(covid, FACTORIES['tsagris'](), hidden, 'cap')[0],
                                   run.run_model(covid, FACTORIES['tsagris'](), hidden)[0], rtol=1e-12, atol=1e-9)


MODELS_TOML = ('[hron_cap]\nmodel = "hron"\nresidual_rule = "cap"\n'
               '[mo_rescale]\nmodel = "micoda"\nother = true\nmax_iter = 40\nresidual_rule = "rescale"\n')


def test_rule_in_toml_runs_and_is_recorded(tmp_path):
    import json
    code, out = call(tmp_path, FIXTURES / 'covid_small.csv', COVID_SCHEMA, MODELS_TOML,
                     '--mask', 'random_cell', '--mask-rate', '.2', '--seeds', '3')
    assert code == 0
    config = json.loads((out / 'config.json').read_text())
    assert config['resolved_models']['hron_cap']['runner_residual_rule'] == 'cap'
    assert 'residual_rule' not in config['resolved_models']['hron_cap']['effective_params']
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['masked']['3']['models']['mo_rescale']['info']['residual_rule']['rule'] == 'rescale'
    assert summary['masked']['3']['models']['mo_rescale']['space'] == '18_with_other'


@pytest.mark.parametrize('models, message', [
    ('[same_day_mean]\nresidual_rule = "prop_other"\n', 'not applicable to same_day_mean: it predicts no Other'),
    ('[micoda]\nother = false\nresidual_rule = "rescale"\n', r'not applicable to micoda with other = false \(A17\)'),
    ('[micoda]\nother = false\nresidual_rule = "cap"\n', r'not applicable to micoda with other = false \(A17\)'),
    ('[mean_fill]\nresidual_rule = "cap"\n', 'residual_rule is not applicable to mean_fill'),
    ('[hron]\nresidual_rule = "r1"\n', 'residual_rule must be one of'),
])
def test_inapplicable_rule_stops_before_any_output(tmp_path, models, message):
    with pytest.raises(ValueError, match=message):
        call(tmp_path, FIXTURES / 'covid_small.csv', COVID_SCHEMA, models, '--mask', 'random_cell')
    assert not (tmp_path / 'out').exists()


def test_micoda_without_other_key_is_a18_and_takes_a_rule():
    """After PR #5 the MICoDaCovid default is other = true (A18), so a [micoda] table without the key
    takes a residual rule; only an explicit other = false (A17) is refused."""
    run.check_rule('micoda', {'residual_rule': 'cap'})
    with pytest.raises(ValueError, match=r'other = false \(A17\)'):
        run.check_rule('micoda', {'residual_rule': 'cap', 'other': False})


def test_implied_other_by_hand():
    # Rows 0 and 1 were complete before masking; row 2 was not (excluded).
    # Row 0 (N = 100): hidden truths 10, 20, predictions 15, 30 -> implied - true = (30 - 45) / 100.
    # Row 1 (N = 50): hidden truth 5, prediction 0 -> (5 - 0) / 50.
    rows = [0, 0, 1, 2]
    truth, pred = [10., 20., 5., 7.], [15., 30., 0., 1.]
    totals, entities = [100., 100., 50., 80.], ['A', 'A', 'B', 'C']
    out = implied_other_errors(rows, truth, pred, totals, entities, {0, 1})
    assert (out['rows'], out['entities']) == (2, 2)
    assert out['mae_proportion'] == pytest.approx((.15 + .1) / 2, abs=1e-15)
    assert out['mean_signed_proportion'] == pytest.approx((-.15 + .1) / 2, abs=1e-15)
    empty = implied_other_errors(rows, truth, pred, totals, entities, {2} - {2})
    assert empty == {'rows': 0, 'entities': 0, 'mae_proportion': None, 'mean_signed_proportion': None}


def test_implied_other_agrees_with_the_completed_table(covid):
    hidden = masks(covid)['rc']
    completed = run.run_model(covid, FACTORIES['hron'](), hidden, 'rescale')[0]
    complete = np.flatnonzero(np.isfinite(covid.counts).all(axis=1))
    rows, cols = np.nonzero(hidden)
    out = implied_other_errors(rows, covid.counts[hidden], completed[hidden], covid.totals[rows],
                               np.asarray(covid.entities)[rows], set(complete))
    use = np.intersect1d(complete, rows)
    implied = (covid.totals[use] - completed[use].sum(axis=1)) / covid.totals[use]
    true = (covid.totals[use] - covid.counts[use].sum(axis=1)) / covid.totals[use]
    assert out['rows'] == len(use) > 0
    assert out['mae_proportion'] == pytest.approx(np.mean(np.abs(implied - true)), abs=1e-12)


def test_rows_without_a_missing_target_are_not_read(tmp_path):
    # MICoDa leaves NaN on rows it excludes (complete and all zero); no rule may read them.
    data, hidden = toy([[40., 10.], [0., 0.]], [100., 100.], [[False, True], [False, False]])
    for rule in ('prop_other', 'cap', 'rescale'):
        out, _ = apply_residual_rule(data, np.array([[.5, .2, .3], [np.nan] * 3]), hidden, rule)
        np.testing.assert_array_equal(out[1], [0., 0.])
        check_completed(data, out, hidden)


@pytest.mark.parametrize('rule', ['prop_other', 'cap', 'rescale'])
def test_micoda_other_with_an_all_zero_complete_row(tmp_path, rule):
    text = (FIXTURES / 'covid_small.csv').read_text().rstrip('\n') + '\nZulu,2021-01-04,10,0,0,0,0,keep\n'
    data = load_csv(write(tmp_path, 'zero.csv', text), write(tmp_path, 'schema.toml', COVID_SCHEMA))
    hidden = make_mask(data, 'random_cell', .3, 4)[0]
    hidden[-1] = False
    model = FACTORIES['micoda_other']()
    completed, status, info = run.run_model(data, model, hidden, rule)
    assert status == 'completed' and np.isnan(model.proposals_[-1]).all()
    np.testing.assert_array_equal(completed[-1], 0.)
    if rule == 'prop_other':
        np.testing.assert_allclose(completed, run.run_model(data, FACTORIES['micoda_other'](), hidden)[0],
                                   rtol=1e-12, atol=1e-9)
