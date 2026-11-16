"""Zero fill: a floor ("decoy"), NOT a serious imputation method.

Every missing target cell gets 0; the whole row residual R = N - sum(observed)
goes to Other. It shows the score of a model that learns nothing; any real
baseline must beat it on the model-decided cells.
"""
import numpy as np
from ..errors import NotFittedError
from .common import fill_within_residual, observed_columns


class ZeroFill:
    input_domain = 'nonnegative'

    def fit(self, values):
        self.width_ = observed_columns(values, 'ZeroFill').shape[1]
        return self

    def predict(self, values):
        if not hasattr(self, 'width_'):
            raise NotFittedError('Fit ZeroFill before prediction')
        x = observed_columns(values, 'ZeroFill')
        out, _ = fill_within_residual(x, np.zeros((len(x), x.shape[1] - 1)))
        self.info_ = {'method': 'zero_fill (floor)', 'filled_cells': int((~np.isfinite(x[:, :-1])).sum())}
        return out
