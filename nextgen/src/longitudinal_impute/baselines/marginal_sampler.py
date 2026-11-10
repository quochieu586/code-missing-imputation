"""Marginal sampler: a floor ("decoy"), NOT a serious imputation method.

Each missing cell of target j gets ONE proportion c/N drawn with replacement from
the observed (visible) cells of target j, so the imputed values follow the observed
marginal distribution of j while being wrong cell by cell. It scores well on
distribution metrics (Wasserstein) and badly on per-cell errors; a real method must
beat it on the latter. Draws are scaled down to the row residual R when they exceed
it, Other gets the rest. Draws are made in row-major order of the missing cells with
numpy default_rng(seed): same seed and same input, same output.
"""
import numpy as np
from ..errors import InsufficientDataError, NotFittedError
from .common import fill_within_residual, observed_columns


class MarginalSampler:
    input_domain = 'nonnegative'

    def __init__(self, seed=0):
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError('seed must be a nonnegative integer')
        self.seed = seed

    def fit(self, values):
        x = observed_columns(values, 'MarginalSampler')[:, :-1]
        self.pools_ = [x[np.isfinite(x[:, j]), j] for j in range(x.shape[1])]
        for j, pool in enumerate(self.pools_):
            if not len(pool):
                raise InsufficientDataError(f'feature {j}: no observed cell to sample from',
                                            context={'feature': j})
        return self

    def predict(self, values):
        if not hasattr(self, 'pools_'):
            raise NotFittedError('Fit MarginalSampler before prediction')
        x = observed_columns(values, 'MarginalSampler')
        rng = np.random.default_rng(self.seed)
        proposal = np.zeros((len(x), x.shape[1] - 1))
        rows, cols = np.nonzero(~np.isfinite(x[:, :-1]))      # row-major order
        draws = rng.random(len(rows))                          # one uniform draw per missing cell
        for j, pool in enumerate(self.pools_):
            cells = cols == j
            pick = np.minimum((draws[cells] * len(pool)).astype(np.int64), len(pool) - 1)
            proposal[rows[cells], j] = pool[pick]
        out, scaled = fill_within_residual(x, proposal)
        self.info_ = {'method': 'marginal_sampler (floor)', 'seed': self.seed,
                      'filled_cells': len(rows), 'rows_scaled_to_residual': scaled}
        return out
