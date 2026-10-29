"""Tsagris et al., JSD k-NN (section 2.3) and JSD alpha-k-NN (section 2.4).

Donors are the complete rows. For an incomplete row the observed parts of
the query and of every donor are closed on the same columns, compared with
the JSD of Eq. 5, and the k nearest donors (ties broken by row index) are
averaged with the Frechet mean of Eq. 6 over their FULL vectors. The mean at
the missing positions is closed and multiplied by T = 1 - sum(observed).
Optional global (k, alpha) tuning follows section 2.4.1 with a fixed
leave-N-out size; the per-pattern adaptive variant (2.5) is not provided.
"""
import numpy as np
from ..errors import BaselineError, NotFittedError

RESIDUAL_TOLERANCE = 1e-12


def jsd(x, y):
    """Eq. 5: twice the usual Jensen-Shannon divergence, natural log, 0 log 0 = 0."""
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    if not (np.isfinite(x).all() and np.isfinite(y).all()) or (x < 0).any() or (y < 0).any():
        raise BaselineError('JSD requires finite nonnegative parts')
    middle = (x + y) / 2
    a, b = np.zeros_like(x), np.zeros_like(y)
    a[x > 0] = x[x > 0] * np.log(x[x > 0] / middle[x > 0])
    b[y > 0] = y[y > 0] * np.log(y[y > 0] / middle[y > 0])
    return np.maximum((a + b).sum(axis=-1), 0.)


def frechet_mean(x, alpha=1.):
    """Eq. 6 with row-normalised powers; alpha = 1 is the arithmetic mean, alpha -> 0 the
    closed geometric mean. alpha <= 0 needs strictly positive rows."""
    x = np.asarray(x, float)
    if not np.isfinite(alpha) or not -1 <= alpha <= 1:
        raise BaselineError('alpha must lie in [-1, 1]')
    if alpha <= 0 and (x <= 0).any():
        raise BaselineError('alpha <= 0 requires strictly positive compositions; use alpha in (0, 1]')
    if alpha == 1:
        return x.mean(axis=0)
    with np.errstate(divide='ignore'):     # a part that is zero in every row has log mean -inf
        if alpha == 0:
            z = np.log(x).mean(axis=0)
        else:
            powers = x ** alpha
            z = np.log(np.mean(powers / powers.sum(axis=1, keepdims=True), axis=0)) / alpha
    mean = np.exp(z - z.max())             # log scale keeps small alpha from underflowing
    return mean / mean.sum()


def _check(values, width=None):
    x = np.array(values, dtype=float)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] < 2:
        raise BaselineError('Tsagris expects a nonempty N x D matrix with D >= 2')
    if width is not None and x.shape[1] != width:
        raise BaselineError('Feature dimension differs from fitted data',
                            context={'expected': width, 'got': x.shape[1]})
    bad = np.argwhere(np.isinf(x) | (np.isfinite(x) & (x < 0)))
    if len(bad):
        i, j = bad[0]
        raise BaselineError(f'Tsagris requires finite nonnegative parts; row {i}, feature {j} is {x[i, j]!r}',
                            context={'row': int(i), 'feature': int(j)})
    return x


class TsagrisKNN:
    input_domain = 'nonnegative'

    def __init__(self, k=3, *, alpha=1., zero_anchor='global', zero_missing_mass='uniform',
                 tune=False, ks=(1, 3, 5, 10), alphas=(.25, .5, .75, 1.), leave_n=None,
                 repetitions=5, seed=0):
        for value in (k, *ks):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise BaselineError('k values must be positive integers')
        for value in (alpha, *alphas):
            if not np.isfinite(value) or not -1 <= value <= 1:
                raise BaselineError('alpha values must lie in [-1, 1]')
        if zero_anchor not in ('global', 'error') or zero_missing_mass not in ('uniform', 'error'):
            raise BaselineError("zero_anchor is 'global'|'error'; zero_missing_mass is 'uniform'|'error'")
        if not ks or not alphas or repetitions < 1 or (leave_n is not None and leave_n < 1):
            raise BaselineError('Invalid tuning configuration')
        self.k, self.alpha = k, float(alpha)
        self.zero_anchor, self.zero_missing_mass = zero_anchor, zero_missing_mass
        self.tune, self.ks, self.alphas = tune, tuple(ks), tuple(float(a) for a in alphas)
        self.leave_n, self.repetitions, self.seed = leave_n, repetitions, seed

    def fit(self, values):
        # Everything is computed locally and published only once it all succeeded,
        # so a failed fit leaves no state (first fit) or the previous state (refit).
        x = _check(values)
        complete = np.isfinite(x).all(axis=1)
        donors = x[complete]
        if not len(donors):
            raise BaselineError('Tsagris needs at least one complete donor row')
        sums = donors.sum(axis=1)
        if not np.allclose(sums, 1., atol=1e-10, rtol=0):
            i = int(np.flatnonzero(complete)[np.argmax(np.abs(sums - 1))])
            raise BaselineError(f'row {i}: complete rows must sum to 1', context={'row': i})
        patterns = ~np.isfinite(x[~complete])
        tuning = self._tune(donors, patterns) if self.tune else None
        k, alpha = (tuning['selected']['k'], tuning['selected']['alpha']) if self.tune else (self.k, self.alpha)
        self.donors_, self.missing_patterns_, self.tuning_, self.k_, self.alpha_ = donors, patterns, tuning, k, alpha
        return self

    def predict(self, values):
        if not hasattr(self, 'donors_'):
            raise NotFittedError('Fit TsagrisKNN before prediction')
        out, counts = self._impute(_check(values, self.donors_.shape[1]), self.donors_, self.k_, self.alpha_)
        self.info_ = {'method': 'tsagris_jsd_knn', 'k': self.k_, 'alpha': self.alpha_,
                      'donors': len(self.donors_), **counts, 'tuning': self.tuning_}
        return out

    def _impute(self, x, donors, k, alpha):
        out = x.copy()
        counts = dict.fromkeys(('filled_rows', 'filled_cells', 'zero_residual_rows', 'zero_anchor_global_rows',
                                'zero_missing_mass_uniform_rows', 'tied_rows'), 0)
        everyone = None
        for i in np.flatnonzero(~np.isfinite(x).all(axis=1)):
            observed = np.isfinite(x[i])
            missing = ~observed
            anchor = x[i, observed].sum()
            residual = 1. - anchor
            counts['filled_rows'] += 1
            counts['filled_cells'] += int(missing.sum())
            if residual < -RESIDUAL_TOLERANCE:
                raise BaselineError(f'row {i}: observed parts sum to {anchor!r} > 1',
                                    context={'row': int(i), 'observed_sum': float(anchor)})
            if residual <= RESIDUAL_TOLERANCE:
                out[i, missing] = 0.
                counts['zero_residual_rows'] += 1
                continue
            if anchor == 0:
                if self.zero_anchor == 'error':
                    raise BaselineError(f'row {i}: observed parts sum to 0, JSD is undefined',
                                        context={'row': int(i), 'reason': 'zero_anchor'})
                everyone = frechet_mean(donors, alpha) if everyone is None else everyone
                mean = everyone
                counts['zero_anchor_global_rows'] += 1
            else:
                masses = donors[:, observed].sum(axis=1)
                eligible = np.flatnonzero(masses > 0)
                if not len(eligible):
                    raise BaselineError(f'row {i}: every donor has zero mass on the observed parts',
                                        context={'row': int(i), 'reason': 'zero_donor_mass'})
                distances = jsd(x[i, observed] / anchor, donors[eligible][:, observed] / masses[eligible, None])
                order = np.argsort(distances, kind='stable')
                if len(order) > k and distances[order[k]] == distances[order[k - 1]]:
                    counts['tied_rows'] += 1
                mean = frechet_mean(donors[eligible[order[:k]]], alpha)
            weights = mean[missing]
            if weights.sum() == 0:
                if self.zero_missing_mass == 'error':
                    raise BaselineError(f'row {i}: donor mean has zero mass on the missing parts',
                                        context={'row': int(i), 'reason': 'zero_missing_mass'})
                weights = np.ones(missing.sum())
                counts['zero_missing_mass_uniform_rows'] += 1
            out[i, missing] = residual * weights / weights.sum()
        return out, counts

    def _tune(self, donors, patterns):
        """Section 2.4.1: repeated leave-N-out on complete rows, mean JSD (data contain zeros)."""
        leave_n = self.leave_n if self.leave_n is not None else max(1, int(.2 * len(donors)))
        if not len(patterns):
            raise BaselineError('tune=True needs incomplete rows to copy missingness patterns from')
        if leave_n >= len(donors):
            raise BaselineError(f'leave_n={leave_n} leaves no training donors',
                                context={'leave_n': leave_n, 'donors': len(donors)})
        rng = np.random.default_rng(self.seed)
        splits = []          # shared by every candidate
        for _ in range(self.repetitions):
            hold = rng.choice(len(donors), leave_n, replace=False)
            query = donors[hold].copy()
            query[patterns[rng.integers(len(patterns), size=leave_n)]] = np.nan
            splits.append((np.setdiff1d(np.arange(len(donors)), hold), hold, query))
        table = []
        for k in self.ks:
            for alpha in self.alphas:
                try:
                    score = np.mean([jsd(donors[hold], self._impute(query, donors[train], k, alpha)[0]).mean()
                                     for train, hold, query in splits])
                    table.append({'k': k, 'alpha': alpha, 'mean_jsd': float(score), 'error': None})
                except BaselineError as exc:
                    table.append({'k': k, 'alpha': alpha, 'mean_jsd': None, 'error': str(exc)})
        valid = [row for row in table if row['error'] is None]
        if not valid:
            raise BaselineError('No (k, alpha) candidate could be scored', context={'table': table})
        best = min(valid, key=lambda row: (row['mean_jsd'], row['k'], row['alpha']))
        return {'selected': {'k': best['k'], 'alpha': best['alpha']}, 'table': table,
                'leave_n': leave_n, 'repetitions': self.repetitions, 'seed': self.seed}
