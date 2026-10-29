"""TsagrisKNN against Tsagris et al. sections 2.3-2.4.1, plus its explicit fallbacks."""
import numpy as np
import pytest
from longitudinal_impute.baselines.tsagris import TsagrisKNN, frechet_mean, jsd
from longitudinal_impute.errors import BaselineError, NotFittedError


def positive_data(n=30, d=4, seed=7):
    return np.random.default_rng(seed).dirichlet(np.ones(d) * 2, size=n)


def test_paper_example_section_2_3_1():
    x = np.array([[.2, np.nan, .3, .1, np.nan], [.1, .2, .4, .1, .2],
                  [.2, .4, .2, .1, .1], [.1, .3, .3, .2, .1]])
    out = TsagrisKNN(2).fit(x).predict(x)
    np.testing.assert_array_equal(np.round(out[0], 2), [.2, .27, .3, .1, .13])
    np.testing.assert_allclose(out[0], [.2, 4 / 15, .3, .1, 2 / 15], atol=1e-12)
    np.testing.assert_array_equal(out[1:], x[1:])


def test_jsd_is_twice_the_divergence_symmetric_and_zero_safe():
    assert jsd([1., 0.], [0., 1.]) == pytest.approx(2 * np.log(2))
    assert jsd([.3, .7], [.3, .7]) == pytest.approx(0.)
    assert jsd([.3, .7], [.8, .2]) == pytest.approx(jsd([.8, .2], [.3, .7]))
    assert np.isfinite(jsd([.5, .5, 0.], [0., .5, .5]))
    # Paper Step 4 prints 0.039 / 0.010 / 0.052 from rounded subcompositions.
    np.testing.assert_allclose(jsd([.33, .5, .17], [[.17, .67, .17], [.4, .4, .2], [.17, .5, .33]]),
                               [.039, .010, .052], atol=1.5e-3)


def test_frechet_mean_endpoints():
    x = positive_data()
    np.testing.assert_allclose(frechet_mean(x, 1), x.mean(axis=0))
    geometric = np.exp(np.log(x).mean(axis=0)); geometric /= geometric.sum()
    np.testing.assert_allclose(frechet_mean(x, 0), geometric)
    np.testing.assert_allclose(frechet_mean(x, 1e-5), geometric, atol=2e-6)
    for a in [-1., -.5, .25, .5, .75]:
        powers = x ** a; powers /= powers.sum(axis=1, keepdims=True)
        expected = powers.mean(axis=0) ** (1 / a); expected /= expected.sum()
        np.testing.assert_allclose(frechet_mean(x, a), expected, atol=1e-12)
    zeros = np.array([[1., 0., 0.], [.5, .5, 0.]])
    np.testing.assert_allclose(frechet_mean(zeros, .5).sum(), 1.)
    assert frechet_mean(zeros, .5)[2] == 0.
    # alpha = 0 with zeros: geometric mean on the parts positive in every row.
    np.testing.assert_array_equal(frechet_mean(zeros, 0.), [1., 0., 0.])
    with pytest.raises(BaselineError, match='common positive support'):
        frechet_mean([[1., 0.], [0., 1.]], 0.)
    for a in (-1., -.5, -1e-20):
        with pytest.raises(BaselineError, match='strictly positive'):
            frechet_mean(zeros, a)


@pytest.mark.parametrize('alpha', [-1., -.5, -1e-20, 0., 1e-20, .25, .5, 1.])
def test_frechet_mean_of_identical_donors_is_that_donor(alpha):
    donor = [.1, .9]
    np.testing.assert_allclose(frechet_mean([donor, donor], alpha), donor, rtol=1e-12)
    np.testing.assert_allclose(frechet_mean([[.1, .3, .6]] * 3, alpha), [.1, .3, .6], rtol=1e-12)


def test_frechet_mean_tends_to_the_geometric_mean_near_alpha_zero():
    x = positive_data()
    geometric = np.exp(np.log(x).mean(axis=0)); geometric /= geometric.sum()
    for a in (-1e-6, -1e-9, -1e-20, 1e-20, 1e-9, 1e-6):
        np.testing.assert_allclose(frechet_mean(x, a), geometric, atol=1e-6)


def test_extreme_parts_stay_finite_or_raise_a_numerical_error():
    x = [[1e-320, 1.], [1e-320, 1.]]
    for alpha in (-1., -.5, 0., .5, 1.):
        mean = frechet_mean(x, alpha)
        assert np.isfinite(mean).all()
        np.testing.assert_allclose(mean, [1e-320, 1.], rtol=1e-3, atol=0)
    out = TsagrisKNN(alpha=-1.).fit(x).predict([[np.nan, np.nan]])
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, [[1e-320, 1.]], rtol=1e-3, atol=0)


def test_non_finite_prediction_is_a_numerical_error_not_nan(monkeypatch):
    from longitudinal_impute.baselines import tsagris
    from longitudinal_impute.errors import NumericalError
    model = TsagrisKNN(k=1, alpha=.5).fit(DONORS)
    monkeypatch.setattr(tsagris, 'frechet_mean', lambda x, alpha: np.full(x.shape[1], np.nan))
    with pytest.raises(NumericalError, match='row 0'):
        model.predict([[.25, np.nan, .25]])
    assert not hasattr(model, 'info_')


@pytest.mark.parametrize('alpha', [1e-20, -1e-20, 0.])
def test_near_zero_alpha_prediction_copies_identical_donors(alpha):
    out = TsagrisKNN(k=2, alpha=alpha).fit([[.1, .3, .6], [.1, .3, .6]]).predict([[np.nan, np.nan, .6]])
    np.testing.assert_allclose(out, [[.1, .3, .6]], rtol=1e-12)


DONORS = [[.5, .5, 0.], [.6, .4, 0.]]
QUERIES = np.array([[0., np.nan, np.nan],      # observed sum 0 -> global donor mean
                    [.3, .7, np.nan],          # T == 0 -> missing parts are 0
                    [.2, .3, np.nan],          # donor mean is 0 on the missing part -> uniform
                    [.25, np.nan, .25]])       # ordinary row


def test_fallbacks_are_applied_and_counted():
    model = TsagrisKNN(1).fit(DONORS)
    out = model.predict(QUERIES)
    np.testing.assert_allclose(out, [[0., 1., 0.], [.3, .7, 0.], [.2, .3, .5], [.25, .5, .25]])
    info = model.info_
    assert (info['filled_rows'], info['filled_cells']) == (4, 5)
    assert info['zero_anchor_global_rows'] == 1
    assert info['zero_residual_rows'] == 1
    assert info['zero_missing_mass_uniform_rows'] == 1


def test_error_modes_and_invalid_residual():
    with pytest.raises(BaselineError, match='sum to 0') as error:
        TsagrisKNN(1, zero_anchor='error').fit(DONORS).predict(QUERIES)
    assert error.value.context == {'row': 0, 'reason': 'zero_anchor'}
    with pytest.raises(BaselineError, match='zero mass on the missing') as error:
        TsagrisKNN(1, zero_missing_mass='error').fit(DONORS).predict(QUERIES[1:])
    assert error.value.context['row'] == 1
    with pytest.raises(BaselineError, match='> 1') as error:
        TsagrisKNN(1).fit(DONORS).predict([[.7, .6, np.nan]])
    assert error.value.context['row'] == 0
    with pytest.raises(BaselineError, match='complete donor'):
        TsagrisKNN().fit([[.1, np.nan]])
    with pytest.raises(BaselineError, match='nonnegative'):
        TsagrisKNN().fit([[-.1, 1.1]])
    with pytest.raises(NotFittedError):
        TsagrisKNN().predict([[.1, np.nan]])


def test_output_keeps_observed_cells_and_closes_every_row():
    rng = np.random.default_rng(5)
    x = rng.dirichlet(np.ones(6) * .3, size=60)
    x[x < 1e-3] = 0.
    x /= x.sum(axis=1, keepdims=True)
    x[10:][rng.random((50, 6)) < .3] = np.nan
    model = TsagrisKNN().fit(x)
    out = model.predict(x)
    observed = np.isfinite(x)
    np.testing.assert_array_equal(out[observed], x[observed])
    assert np.isfinite(out).all() and (out >= 0).all()
    np.testing.assert_allclose(out.sum(axis=1), 1., atol=1e-10)


def test_global_tuning_is_reproducible_and_reported():
    x = positive_data(n=40)
    x[:8, [0, 1]] = np.nan
    x[8:12, 3] = np.nan
    kwargs = dict(tune=True, ks=(1, 3), alphas=(.5, 1.), repetitions=2, seed=3)
    a, b = TsagrisKNN(**kwargs).fit(x), TsagrisKNN(**kwargs).fit(x)
    assert a.tuning_ == b.tuning_
    assert a.tuning_['leave_n'] == int(.2 * 28)
    assert len(a.tuning_['table']) == 4
    selected = a.tuning_['selected']
    assert (selected['k'], selected['alpha']) == (a.k_, a.alpha_)
    best = min(row['mean_jsd'] for row in a.tuning_['table'])
    assert [row['mean_jsd'] for row in a.tuning_['table']
            if (row['k'], row['alpha']) == (a.k_, a.alpha_)] == [best]
    a.predict(x)
    assert a.info_['tuning'] is a.tuning_ and a.info_['k'] == a.k_
    with pytest.raises(BaselineError, match='incomplete rows'):
        TsagrisKNN(tune=True).fit(positive_data())
    with pytest.raises(BaselineError, match='no training donors'):
        TsagrisKNN(tune=True, leave_n=28).fit(x)
