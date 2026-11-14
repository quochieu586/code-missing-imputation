"""Domain-neutral error types shared by data, baseline, and benchmark layers."""


class BaselineError(ValueError):
    """Expected input/support failure with a stable machine-readable code."""

    code = 'baseline_error'

    def __init__(self, message, *, context=None):
        super().__init__(message)
        self.context = dict(context or {})

    def add_context(self, **context):
        self.context.update({key: value for key, value in context.items() if value is not None})
        return self


class BaselineInputError(BaselineError):
    code = 'invalid_input'


class InsufficientDataError(BaselineError):
    code = 'insufficient_data'


class RankDeficientError(InsufficientDataError):
    code = 'rank_deficient'


class NumericalError(BaselineError):
    code = 'numerical_failure'


class NotFittedError(BaselineError):
    code = 'not_fitted'


class PredictionUnavailableError(BaselineError):
    """A row has no supported prediction, and may use an explicit row fallback."""

    code = 'prediction_unavailable'
