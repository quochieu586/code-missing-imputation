"""Numerical/domain/information-boundary tests, not mirrors of boilerplate."""
import itertools
import numpy as np
import pytest
from longitudinal_impute.baselines.common import BaselineError, clr, pivot_basis
from longitudinal_impute.baselines.micoda import MICoDa, allocate_missing
from longitudinal_impute.baselines.regression import lts


def positive_data(n=30, d=4):
    rng = np.random.default_rng(7)
    return rng.dirichlet(np.ones(d) * 2, size=n)


def test_pivot_isometry_and_inverse_relative():
    x = positive_data()
    basis = pivot_basis(x.shape[1])
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-12)
    z = clr(x) @ basis
    back = np.exp(z @ basis.T); back /= back.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(back, x, atol=1e-12)


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


def test_micoda_invalid_matrix_and_initial_complete_vs_pairwise_support():
    with pytest.raises(BaselineError): MICoDa().fit([.2, .8])
    train = positive_data(n=20)
    train[0, 0] = np.nan; train[1, 1] = np.nan
    complete = MICoDa(reference=3, max_iter=2).fit(train)
    pairwise = MICoDa(reference=3, max_iter=2, fit_rows='pairwise').fit(train)
    assert complete.fit_result_.diagnostics['initial_regression_support'] == [18, 18, 18]
    assert pairwise.fit_result_.diagnostics['initial_regression_support'] == [19, 19, 20]
