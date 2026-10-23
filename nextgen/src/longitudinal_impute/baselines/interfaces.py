"""Numerical-core and fitted-count contracts at the adaptation boundary."""
from typing import Literal, Protocol, Self
from ..contracts import Array, ConditioningPanel, FitScope, ImputationResult
from .common import CoreResult


class NumericalCore(Protocol):
    """Paper-level numeric model; values are proportions, never count panels."""

    input_domain: Literal['positive', 'nonnegative']
    prediction_policy: Literal['rowwise_adaptation', 'population_adaptation']

    def fit(self, values: Array, covariates: Array | None = None) -> Self: ...

    def predict_result(self, values: Array, covariates: Array | None = None) -> CoreResult: ...


class FittedCountBaseline(Protocol):
    """Dataset adapter; owns FitScope, count constraints, and diagnostics."""

    def fit(self, panel: ConditioningPanel, scope: FitScope) -> Self: ...

    def predict_result(self, panel: ConditioningPanel) -> ImputationResult: ...
