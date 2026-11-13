"""fit is atomic for both baselines: a failed fit publishes nothing, a failed refit keeps the old state."""
import numpy as np
import pytest

from longitudinal_impute.baselines.hron import HronKNN
from longitudinal_impute.baselines.tsagris import TsagrisKNN
from longitudinal_impute.errors import BaselineError, NotFittedError

A, QUERY_A = [[.2, .8], [.4, .6], [.3, .7]], [[np.nan, .25]]
B, QUERY_B = [[.2, .3, .5], [.1, .6, .3], [.3, .3, .4]], [[np.nan, .2, .3]]
# Inputs each model must reject; the 3-column ones also change the schema on a refit.
BAD = {'tsagris': [[[.8, .8, .4]], [[.1, np.nan, .9]]], 'hron': [[[0., .5, .5]], [[-1., 1., 1.]]]}
MODELS = {'tsagris': lambda: TsagrisKNN(k=1), 'hron': lambda: HronKNN(k=1)}
FITTED = {'tsagris': ('donors_', 'missing_patterns_', 'tuning_', 'k_', 'alpha_'), 'hron': ('donors_',)}


@pytest.mark.parametrize('name', MODELS)
def test_failed_first_fit_leaves_the_model_unfitted(name):
    for bad in BAD[name]:
        model = MODELS[name]()
        with pytest.raises(BaselineError):
            model.fit(bad)
        assert not any(hasattr(model, attribute) for attribute in FITTED[name])
        with pytest.raises(NotFittedError):
            model.predict(QUERY_B)


def test_failed_tuning_leaves_tsagris_unfitted():
    model = TsagrisKNN(tune=True)
    with pytest.raises(BaselineError, match='incomplete rows'):
        model.fit(B)                       # complete rows only: no pattern to copy
    assert not hasattr(model, 'donors_')
    with pytest.raises(NotFittedError):
        model.predict(QUERY_B)


@pytest.mark.parametrize('name', MODELS)
def test_failed_refit_keeps_previous_state_and_predictions(name):
    model = MODELS[name]().fit(A)
    before = {attribute: getattr(model, attribute) for attribute in FITTED[name]}
    prediction = model.predict(QUERY_A)
    for bad in BAD[name]:
        with pytest.raises(BaselineError):
            model.fit(bad)
        for attribute, value in before.items():
            assert getattr(model, attribute) is value
        np.testing.assert_array_equal(model.predict(QUERY_A), prediction)
    with pytest.raises(BaselineError, match='dimension'):
        model.predict(QUERY_B)


@pytest.mark.parametrize('name', MODELS)
def test_successful_refit_equals_a_fresh_fit(name):
    refit = MODELS[name]().fit(A).fit(B)
    fresh = MODELS[name]().fit(B)
    np.testing.assert_array_equal(refit.predict(QUERY_B), fresh.predict(QUERY_B))
    for attribute in FITTED[name]:
        np.testing.assert_array_equal(getattr(refit, attribute), getattr(fresh, attribute))
    assert refit.info_ == fresh.info_
