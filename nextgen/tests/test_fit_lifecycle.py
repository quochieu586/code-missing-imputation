"""Public estimators publish a complete fit or preserve the previous model."""
import pickle
from unittest.mock import patch
import numpy as np
import pytest
from longitudinal_impute.baselines import hron
from longitudinal_impute.baselines.hron import HronKNN, HronIterative
from longitudinal_impute.baselines.tsagris import TsagrisKNN, TsagrisAdaptive
from longitudinal_impute.errors import BaselineError, NotFittedError


FACTORIES = [lambda: HronKNN(1), lambda: HronIterative(max_iter=2),
             lambda: HronIterative(regression='lts', max_iter=2, lts_starts=2, lts_exact_limit=1),
             lambda: TsagrisKNN(1), lambda: TsagrisKNN(1, alpha=-.5),
             lambda: TsagrisAdaptive(ks=(1,), alphas=(1.,), leave_n=1, repetitions=1),
             lambda: TsagrisAdaptive(ks=(1,), alphas=(1.,), leave_n=1, repetitions=1, adaptive=False)]


def fit(model, values):
    if isinstance(model, TsagrisAdaptive):
        return model.fit(values, patterns=[(0,), (1,)])
    return model.fit(values)


def bad_data(model):
    if isinstance(model, HronIterative):
        return np.tile([.1, .2, .3, .4], (8, 1))
    if isinstance(model, HronKNN):
        return [[-.1, .3, .3, .5]]
    if isinstance(model, TsagrisAdaptive):
        return [[.6, .1, .2, .1]]
    return np.full((3, 4), np.nan)


@pytest.mark.parametrize('factory', FACTORIES)
def test_failed_first_fit_never_publishes_fitted_guard(factory):
    model = factory()
    with pytest.raises(BaselineError):
        fit(model, bad_data(model))
    with pytest.raises(NotFittedError):
        model.predict([[np.nan, np.nan, .2, .1]])


@pytest.mark.parametrize('factory', FACTORIES)
def test_failed_refit_preserves_entire_state_and_predictions(factory):
    model = factory()
    good = np.random.default_rng(3).dirichlet(np.ones(4), 12)
    fit(model, good)
    query = [[np.nan, np.nan, .2, .1]]
    before = model.predict(query)
    state = pickle.dumps(vars(model))
    with pytest.raises(BaselineError):
        fit(model, bad_data(model))
    assert pickle.dumps(vars(model)) == state
    np.testing.assert_array_equal(before, model.predict(query))


@pytest.mark.parametrize('factory', FACTORIES)
def test_successful_refit_replaces_schema_and_matches_a_fresh_fit(factory):
    model = factory()
    fit(model, np.random.default_rng(1).dirichlet(np.ones(4), 12))
    model.predict([[np.nan, np.nan, .2, .1]])
    values = np.random.default_rng(2).dirichlet(np.ones(5), 15)
    assert fit(model, values) is model
    assert not hasattr(model, 'last_prediction_')
    fresh = fit(factory(), values)
    query = [[np.nan, np.nan, .2, .1, .1]]
    np.testing.assert_array_equal(model.predict(query), fresh.predict(query))


def test_hron_failure_after_first_target_preserves_initializer_and_models():
    values = np.random.default_rng(9).dirichlet(np.ones(4), 18)
    model = HronIterative(max_iter=2).fit(values)
    query = [[np.nan, np.nan, .2, .1]]
    before = model.predict(query)
    state = pickle.dumps(vars(model))
    original = hron.ols
    calls = 0

    def fail_second(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise BaselineError('injected failure after first fitted target')
        return original(*args)

    with patch.object(hron, 'ols', side_effect=fail_second):
        with pytest.raises(BaselineError, match='injected'):
            model.fit(values[::-1])
    assert calls == 2
    assert pickle.dumps(vars(model)) == state
    np.testing.assert_array_equal(before, model.predict(query))


def test_adaptive_failure_after_first_tuned_pattern_preserves_previous_fit():
    values = np.random.default_rng(3).dirichlet(np.ones(4), 12)
    model = TsagrisAdaptive(ks=(1,), alphas=(1.,), leave_n=1, repetitions=1)
    fit(model, values)
    query = [[np.nan, np.nan, .2, .1]]
    before = model.predict(query)
    state = pickle.dumps(vars(model))
    original = TsagrisAdaptive._tune
    calls = 0

    def fail_second(self, *args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise BaselineError('injected failure after first tuned pattern')
        return original(self, *args)

    with patch.object(TsagrisAdaptive, '_tune', new=fail_second):
        with pytest.raises(BaselineError, match='injected'):
            fit(model, values[::-1])
    assert calls == 2
    assert pickle.dumps(vars(model)) == state
    np.testing.assert_array_equal(before, model.predict(query))


def test_negative_alpha_refit_rejects_zero_donors_without_publishing_them():
    values = np.random.default_rng(2).dirichlet(np.ones(4), 12)
    model = TsagrisKNN(1, alpha=-.5).fit(values)
    before = pickle.dumps(vars(model))
    with pytest.raises(BaselineError, match='positive complete donors'):
        model.fit([[0., .2, .3, .5]])
    assert pickle.dumps(vars(model)) == before
