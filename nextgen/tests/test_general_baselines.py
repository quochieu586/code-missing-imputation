"""Numerical/domain/information-boundary tests, not mirrors of boilerplate."""
import itertools
import numpy as np
import pytest
from longitudinal_impute.baselines.common import BaselineError, clr, pivot_basis
from longitudinal_impute.baselines.hron import HronKNN, HronIterative
from longitudinal_impute.baselines.micoda import MICoDa, allocate_missing
from longitudinal_impute.baselines.tsagris import TsagrisKNN, TsagrisAdaptive, alpha_mean, jsd
from longitudinal_impute.baselines.regression import lts


def positive_data(n=30, d=4):
    rng = np.random.default_rng(7)
    return rng.dirichlet(np.ones(d) * 2, size=n)


def test_tsagris_paper_example_full_donor_mean():
    donors = np.array([[.1, .2, .4, .1, .2], [.2, .4, .2, .1, .1], [.1, .3, .3, .2, .1]])
    query = np.array([[.2, np.nan, .3, .1, np.nan]])
    result = TsagrisKNN(2).fit(donors).predict_result(query)
    np.testing.assert_allclose(result.values, [[.2, 4 / 15, .3, .1, 2 / 15]], atol=1e-12)
    assert result.diagnostics['records'][0]['donor_ids'] == [1, 0]
    # Normalize-then-average gives .26 rather than 4/15 and must fail parity.
    wrong = .4 * np.mean([.2 / .4, .4 / .5])
    assert not np.isclose(result.values[0, 1], wrong)


def test_jsd_zero_safe_factor_of_two_and_bounds():
    assert jsd([1., 0.], [0., 1.]) == pytest.approx(2 * np.log(2))
    assert jsd([.3, .7], [.3, .7]) == pytest.approx(0.)
    assert jsd([.3, .7], [.8, .2]) == pytest.approx(jsd([.8, .2], [.3, .7]))


def test_alpha_mean_endpoints_and_near_zero_limit():
    x = positive_data()
    np.testing.assert_allclose(alpha_mean(x, 1), x.mean(axis=0))
    geometric = np.exp(np.log(x).mean(axis=0)); geometric /= geometric.sum()
    np.testing.assert_allclose(alpha_mean(x, 0), geometric)
    np.testing.assert_allclose(alpha_mean(x, 1e-5), geometric, atol=2e-6)
    for a in [-1., -.5, .25, .5]:
        powers = x ** a; powers /= powers.sum(axis=1, keepdims=True)
        expected = powers.mean(axis=0) ** (1 / a); expected /= expected.sum()
        np.testing.assert_allclose(alpha_mean(x, a), expected, atol=1e-12)
    with pytest.raises(BaselineError): alpha_mean([[1., 0.]], -.5)
    np.testing.assert_allclose(alpha_mean([[1., 0.], [.5, .5]], 0), [1., 0.])
    with pytest.raises(BaselineError, match='common positive support'):
        alpha_mean([[1., 0.], [0., 1.]], 0)
    np.testing.assert_allclose(alpha_mean([[1., 0.], [.5, .5]], .5).sum(), 1.)


def test_hron_ratio_of_medians_is_not_median_of_ratios():
    donor = np.array([[100., 200., 1., 9.]])
    query = np.array([[10., 30., 60., np.nan]])
    value = HronKNN(1).fit(donor).predict(query)[0, 3]
    assert value == pytest.approx(9 * 30 / 100)
    assert value != pytest.approx(9 * np.median(query[0, :3] / donor[0, :3]))
    np.testing.assert_allclose(HronKNN(1, adjustment='sum').fit(donor).predict(query)[0, 3], 9 * 100 / 301)


def test_hron_donor_policies_ties_and_scale_invariance():
    donors = np.array([[1., 2., 3., np.nan], [2., 4., 10., 8.], [4., 8., 24., 16.]])
    query = np.array([[2., 4., np.nan, np.nan]])
    available = HronKNN(1, donor_policy='available').fit(donors).predict_result(query)
    complete = HronKNN(1, donor_policy='complete').fit(donors).predict_result(query)
    assert available.values[0, 2] == pytest.approx(6)
    assert complete.values[0, 2] == pytest.approx(10)
    np.testing.assert_allclose(HronKNN(1).fit(donors).predict(query * 5), available.values * 5)
    assert available.diagnostics['records'][0]['donor_ids'] == [0]


def test_pivot_isometry_and_inverse_relative():
    x = positive_data()
    basis = pivot_basis(x.shape[1])
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-12)
    z = clr(x) @ basis
    back = np.exp(z @ basis.T); back /= back.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(back, x, atol=1e-12)


@pytest.mark.parametrize('regression', ['ls', 'lts'])
def test_hron_iterative_observed_ratios_stopping_and_bounded_fit(regression):
    x = positive_data(n=14, d=3)
    original = x.copy(); x[:3, 0] = np.nan; x[3:5, 2] = np.nan
    result = HronIterative(regression=regression, max_iter=4, lts_starts=8, lts_exact_limit=20).fit(x)
    out = result.fit_result_.values
    assert np.isfinite(out).all() and (out > 0).all()
    for i in range(len(x)):
        observed = np.flatnonzero(np.isfinite(x[i]))
        np.testing.assert_allclose(out[i, observed] / out[i, observed[0]],
                                   original[i, observed] / original[i, observed[0]], atol=1e-10)
    assert result.fit_result_.diagnostics['iterations'] <= 4
    assert len(result.models_) == 3  # last part also has a fitted predictor
    assert not np.array_equal(out[np.isfinite(x)], original[np.isfinite(x)])


def test_lts_exact_small_problem_and_seeded_approximation():
    design = np.column_stack([np.ones(7), np.arange(7.)])
    y = 2 + 3 * np.arange(7.); y[-1] += 100
    beta, info = lts(design, y, h=5)
    np.testing.assert_allclose(beta, [2, 3], atol=1e-10)
    assert info['global_optimum_certified'] and info['objective'] < 1e-16
    # Independent exhaustive trimmed objective comparison.
    best = min(np.sort((y - design @ np.linalg.lstsq(design[list(s)], y[list(s)], rcond=None)[0]) ** 2)[:5].sum()
               for s in itertools.combinations(range(7), 5))
    assert info['objective'] == pytest.approx(best, abs=1e-15)
    a, ia = lts(design, y, h=5, exact_limit=1, starts=8)
    b, ib = lts(design, y, h=5, exact_limit=1, starts=8)
    np.testing.assert_array_equal(a, b)
    assert not ia['global_optimum_certified']


def test_micoda_fixed_coefficient_formula_reference_cases():
    logits = np.log([2., 3.])
    np.testing.assert_allclose(allocate_missing([.2, np.nan, np.nan], logits, 2, 1), [.2, .6, .2])
    np.testing.assert_allclose(allocate_missing([np.nan, np.nan, .4], logits, 2, 1), [.24, .36, .4])
    np.testing.assert_allclose(allocate_missing([np.nan, .3, .4], logits, 2, 1), [.3, .3, .4])
    np.testing.assert_allclose(allocate_missing([np.nan, np.nan, np.nan], logits, 2, 1), [1/3, .5, 1/6])
    with pytest.raises(BaselineError): allocate_missing([.8, .7, np.nan], logits, 2, 1)


@pytest.mark.parametrize('version', ['v1', 'v2'])
def test_micoda_iterative_W_and_reference_auto(version):
    x = positive_data(); x[:3, 0] = np.nan; x[3:5, 2:] = np.nan
    x[8, 0] = 0.; x[8, 1] += positive_data()[8, 0]
    result = MICoDa(version=version, W=.001, max_iter=5).fit(x)
    out = result.fit_result_.values
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out.sum(axis=1), 1.)
    assert result.reference_ == 1
    assert result.fit_result_.diagnostics['iterations'] <= 5


def test_micoda_paper_v2_vs_author_observed_reference():
    x = positive_data(n=25, d=3)
    query = np.array([[np.nan, .3, .4]])
    paper = MICoDa(reference=2, version='v2').fit(x).predict(query)
    author = MICoDa(reference=2, version='v2', semantics='author_code').fit(x).predict(query)
    assert paper[0, 2] == .4 and paper[0, 1] == .3
    assert author[0, 2] != pytest.approx(.4)
    np.testing.assert_allclose(author.sum(axis=1), 1)


def test_regression_rank_and_missing_covariates_are_errors():
    x = positive_data()
    with pytest.raises(BaselineError, match='rank'):
        MICoDa().fit(x, np.ones((len(x), 2)))
    bad = np.arange(len(x), dtype=float)[:, None]; bad[0] = np.nan
    with pytest.raises(BaselineError, match='Covariates'):
        MICoDa().fit(x, bad)
    with pytest.raises(BaselineError): MICoDa().fit([[.2, .8], [np.nan, .6]])


@pytest.mark.parametrize('adaptive', [False, True])
def test_tsagris_training_only_cv_reproducible_and_unseen_patterns(adaptive):
    train = positive_data(n=18)
    train[0, [0, 1]] = np.nan; train[1, [2, 3]] = np.nan
    model = TsagrisAdaptive(ks=(1, 2), alphas=(.5, 1.), leave_n=2, repetitions=2, adaptive=adaptive).fit(train)
    query = positive_data(n=3); query[:, [0, 2]] = np.nan
    a = model.predict(query)
    b = model.predict(query)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(a.sum(axis=1), 1.)
    if adaptive: assert (0, 2) in model.selected_
    with pytest.raises(BaselineError): TsagrisAdaptive(leave_n=100).fit(train)


def test_adaptive_default_cv_size_is_pattern_frequency():
    train = positive_data(n=15)
    train[:2, 0] = np.nan; train[2, 1] = np.nan
    model = TsagrisAdaptive(ks=(1,), alphas=(1,), repetitions=1).fit(train)
    assert model.cv_sizes_['(0,)'] == 2
    assert model.cv_sizes_['(1,)'] == 1


def test_micoda_invalid_matrix_and_initial_complete_vs_pairwise_support():
    with pytest.raises(BaselineError): MICoDa().fit([.2, .8])
    train = positive_data(n=20)
    train[0, 0] = np.nan; train[1, 1] = np.nan
    complete = MICoDa(reference=3, max_iter=2).fit(train)
    pairwise = MICoDa(reference=3, max_iter=2, fit_rows='pairwise').fit(train)
    assert complete.fit_result_.diagnostics['initial_regression_support'] == [18, 18, 18]
    assert pairwise.fit_result_.diagnostics['initial_regression_support'] == [19, 19, 20]


def test_reference_degenerate_errors_and_explicit_fallbacks():
    with pytest.raises(BaselineError): HronKNN().fit([[0., 1.]])
    with pytest.raises(BaselineError): HronKNN().fit([[1., np.nan]]).predict([[np.nan, 1.]])
    with pytest.raises(BaselineError): TsagrisKNN().fit([[.1, np.nan]])
    fit = [[.4, .6], [.6, .4]]
    with pytest.raises(BaselineError): TsagrisKNN().fit(fit).predict([[0., np.nan]])
    np.testing.assert_allclose(TsagrisKNN(zero_anchor='global').fit(fit).predict([[0., np.nan]]), [[0., 1.]])
    with pytest.raises(BaselineError): TsagrisKNN().fit([[1., 0.]]).predict([[.5, np.nan]])
    np.testing.assert_allclose(TsagrisKNN(zero_missing_mass='uniform').fit([[1., 0.]]).predict([[.5, np.nan]]), [[.5, .5]])
