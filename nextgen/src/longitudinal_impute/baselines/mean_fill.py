"""Global mean fill: a floor ("decoy"), NOT a serious imputation method.

A missing cell of target j gets the unweighted mean of the observed proportions
c/N of target j over the whole table (visible cells only, so a hidden cell never
contributes). The proposals of a row are scaled down when they exceed the row
residual R = 1 - sum(observed); Other gets what is left (common.fill_within_residual).
It ignores time, place and the row's own observed parts.
"""
import numpy as np
from ..errors import InsufficientDataError, NotFittedError
from .common import fill_within_residual, observed_columns


class MeanFill:
    input_domain = 'nonnegative'

    def fit(self, values):
        x = observed_columns(values, 'MeanFill')[:, :-1]
        seen = np.isfinite(x)
        empty = np.flatnonzero(~seen.any(axis=0))
        if len(empty):
            raise InsufficientDataError(f'feature {int(empty[0])}: no observed cell to average',
                                        context={'feature': int(empty[0])})
        self.means_ = np.nanmean(x, axis=0)
        return self

    def predict(self, values):
        if not hasattr(self, 'means_'):
            raise NotFittedError('Fit MeanFill before prediction')
        x = observed_columns(values, 'MeanFill')
        out, scaled = fill_within_residual(x, np.broadcast_to(self.means_, (len(x), len(self.means_))))
        self.info_ = {'method': 'mean_fill (floor)', 'means': self.means_.tolist(),
                      'filled_cells': int((~np.isfinite(x[:, :-1])).sum()), 'rows_scaled_to_residual': scaled}
        return out
