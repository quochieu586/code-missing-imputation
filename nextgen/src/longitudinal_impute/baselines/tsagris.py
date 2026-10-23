"""Tsagris JSD-kNN, alpha mean, repeated leave-N-out and pattern adaptation.

JSD is Eq. (5), i.e. twice the usual divergence, with natural logarithms.
Eq. (6) is interpreted with ROW-normalized powers, consistent with alpha=1
arithmetic mean and alpha->0 geometric mean; PDF denominator indices have a
typographical inconsistency. No handwritten annotations define this code.
"""
import copy
import numpy as np
from .common import BaselineError, CoreResult, simplex, closure, clr, stable_softmax, distance_order
from ..errors import NotFittedError, PredictionUnavailableError


def jsd(x, y):
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    if not np.isfinite(x).all() or not np.isfinite(y).all() or (x < 0).any() or (y < 0).any():
        raise BaselineError('JSD requires finite nonnegative parts')
    if not np.allclose(x.sum(axis=-1), 1.) or not np.allclose(y.sum(axis=-1), 1.):
        raise BaselineError('JSD inputs must be closed compositions')
    middle = (x + y) / 2
    a, b = np.zeros_like(x), np.zeros_like(y)
    ax, by = x > 0, y > 0
    a[ax] = x[ax] * np.log(x[ax] / middle[ax])
    b[by] = y[by] * np.log(y[by] / middle[by])
    return np.maximum((a + b).sum(axis=-1), 0.)


def alpha_mean(values, alpha=1.):
    x = simplex(values)
    if not np.isfinite(alpha) or not -1 <= alpha <= 1:
        raise BaselineError('alpha must be finite in [-1,1]')
    if alpha < 0 and (x == 0).any():
        raise BaselineError('alpha < 0 requires strictly positive donors; no implicit zero smoothing')
    if alpha == 1:
        return x.mean(axis=0)
    if alpha == 0:
        shared_positive = (x > 0).all(axis=0)
        if not shared_positive.any():
            raise BaselineError('Geometric mean has no common positive support; closure undefined')
        answer = np.zeros(x.shape[1])
        answer[shared_positive] = stable_softmax(np.log(x[:, shared_positive]).mean(axis=0))
        return answer
    if abs(alpha) < 1e-8 and (x > 0).all():
        return stable_softmax(np.log(x).mean(axis=0))
    # Work in logs; all-zero feature has zero mass for alpha>0.
    with np.errstate(divide='ignore'):
        logs = np.log(x)
    powered = alpha * logs
    maxima = powered.max(axis=1, keepdims=True)
    lognorm = maxima + np.log(np.exp(powered - maxima).sum(axis=1, keepdims=True))
    logu = powered - lognorm
    colmax = logu.max(axis=0)
    active = np.isfinite(colmax)
    logmean = np.full(x.shape[1], -np.inf)
    logmean[active] = colmax[active] + np.log(np.exp(logu[:, active] - colmax[active]).mean(axis=0))
    loganswer = logmean / alpha
    answer = np.zeros(x.shape[1])
    answer[active] = np.exp(loganswer[active] - loganswer[active].max())
    return closure(answer)


class TsagrisKNN:
    input_domain = 'nonnegative'
    prediction_policy = 'rowwise_adaptation'

    def __init__(self, k=3, *, alpha=1., zero_anchor='error', zero_missing_mass='error'):
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise BaselineError('k must be a positive integer')
        if not np.isfinite(alpha) or not -1 <= alpha <= 1:
            raise BaselineError('alpha must lie in [-1,1]')
        if zero_anchor not in ('error', 'global') or zero_missing_mass not in ('error', 'uniform'):
            raise BaselineError('Unknown explicit degenerate-case policy')
        self.k, self.alpha = k, float(alpha)
        self.zero_anchor, self.zero_missing_mass = zero_anchor, zero_missing_mass

    def fit(self, values, covariates=None):
        candidate = copy.copy(self)
        candidate.__dict__.pop('last_prediction_', None)
        candidate._fit_in_place(values, covariates)
        self.__dict__ = candidate.__dict__
        return self

    def _fit_in_place(self, values, covariates=None):
        x = simplex(values, partial=True)
        self.width_ = x.shape[1]
        self.donor_ids_ = np.flatnonzero(np.isfinite(x).all(axis=1))
        self.donors_ = x[self.donor_ids_].copy()
        if len(self.donors_) == 0:
            raise BaselineError('Tsagris needs at least one complete donor')
        if self.alpha < 0 and (self.donors_ == 0).any():
            raise BaselineError('alpha < 0 requires positive complete donors')
        return self

    def with_degenerate_policies(self):
        """Return a configured copy for adapters with declared fallbacks."""
        from .registry import configure_core
        return configure_core(self, degenerate_policies=True)

    def predict_result(self, values, covariates=None):
        if not hasattr(self, 'donors_'):
            raise NotFittedError('Fit TsagrisKNN before prediction')
        x = simplex(values, partial=True)
        if x.shape[1] != self.width_:
            raise BaselineError('Query dimension differs from fit')
        out, records = x.copy(), []
        for i, row in enumerate(x):
            observed, missing = np.isfinite(row), ~np.isfinite(row)
            if not missing.any():
                continue
            residual = max(0., 1 - row[observed].sum())
            if residual == 0:
                out[i, missing] = 0.
                records.append({'row': i, 'policy': 'zero_residual'})
                continue
            anchor_mass = row[observed].sum()
            policy = 'subcomposition_jsd'
            if anchor_mass == 0:
                if self.zero_anchor == 'error':
                    raise PredictionUnavailableError(f'row {i}: zero-total observed subcomposition',
                                                     context={'row': i})
                # Explicit application fallback: full donor mean, not arbitrary
                # first-k donors tied on an undefined distance.
                selected = np.arange(len(self.donors_))
                policy = 'global_donor_mean'
            else:
                masses = self.donors_[:, observed].sum(axis=1)
                eligible = np.flatnonzero(masses > 0)
                if not len(eligible):
                    raise PredictionUnavailableError(f'row {i}: all donor anchor masses zero',
                                                     context={'row': i})
                distances = jsd(row[observed] / anchor_mass,
                                self.donors_[eligible][:, observed] / masses[eligible, None])
                selected = eligible[distance_order(distances, limit=self.k)[:self.k]]
            mean = alpha_mean(self.donors_[selected], self.alpha)
            weights = mean[missing]
            if weights.sum() == 0:
                if self.zero_missing_mass == 'error':
                    raise PredictionUnavailableError(f'row {i}: donor mean has zero missing-part mass',
                                                     context={'row': i})
                weights = np.ones(missing.sum())
                policy += '+uniform_missing'
            out[i, missing] = residual * weights / weights.sum()
            records.append({'row': i, 'donor_ids': self.donor_ids_[selected].tolist(),
                            'effective_k': len(selected), 'alpha': self.alpha,
                            'policy': policy, 'one_anchor': bool(observed.sum() == 1)})
        return CoreResult(out, {'method': 'tsagris_knn', 'records': records,
                               'jsd_scale': 'twice_standard_natural_log'})

    def predict(self, values, covariates=None):
        result = self.predict_result(values)
        self.last_prediction_ = result.diagnostics
        return result.values

    def impute(self, values, covariates=None):
        return self.fit(values).predict(values)


class TsagrisAdaptive:
    input_domain = 'nonnegative'
    prediction_policy = 'rowwise_adaptation'

    """Section 2.4.1/2.5 CV using only supplied fitting population.

    adaptive=False chooses one global pair; True chooses a pair per pattern.
    leave_n=None uses the number of incomplete fit rows (paper rule). If that
    leaves no donors, raise and ask caller to choose a documented CV size.
    Unseen query patterns must be tuned from fit donors, never from query truth.
    """
    def __init__(self, *, ks=(1, 3, 5), alphas=(.25, .5, 1.), repetitions=5,
                 leave_n=None, adaptive=True, seed=42, zero_anchor='error',
                 zero_missing_mass='error'):
        if not ks or not alphas or repetitions < 1 or (leave_n is not None and leave_n < 1):
            raise BaselineError('Invalid CV configuration')
        self.ks, self.alphas = tuple(ks), tuple(alphas)
        for k in ks:
            for a in alphas:
                TsagrisKNN(k, alpha=a)
        self.repetitions, self.leave_n, self.adaptive = repetitions, leave_n, adaptive
        self.seed, self.zero_anchor, self.zero_missing_mass = seed, zero_anchor, zero_missing_mass

    def _model(self, k, alpha):
        return TsagrisKNN(k, alpha=alpha, zero_anchor=self.zero_anchor,
                          zero_missing_mass=self.zero_missing_mass)

    def _tune(self, patterns, leave_n):
        donors = self.donors_
        if leave_n >= len(donors):
            raise BaselineError('CV leave-N-out leaves no complete training donors')
        rng = np.random.default_rng(self.seed)
        # Shared splits across all candidates; do not let model-specific
        # randomness change the task used to choose hyperparameters.
        splits = []
        for _ in range(self.repetitions):
            hold = rng.choice(len(donors), leave_n, replace=False)
            train = np.setdiff1d(np.arange(len(donors)), hold)
            query = donors[hold].copy()
            for i in range(leave_n):
                query[i, list(patterns[i % len(patterns)])] = np.nan
            splits.append((train, hold, query))
        table = []
        for k in self.ks:
            for alpha in self.alphas:
                scores, error = [], None
                if k > len(donors) - leave_n:
                    table.append({'k': k, 'alpha': alpha, 'score': None, 'error': 'k exceeds CV donor count'})
                    continue
                for train, hold, query in splits:
                    try:
                        pred = self._model(k, alpha).fit(donors[train]).predict(query)
                        if (donors == 0).any():
                            distances = jsd(donors[hold], pred)
                        else:
                            distances = np.linalg.norm(clr(donors[hold]) - clr(pred), axis=1)
                        scores.append(float(distances.mean()))
                    except BaselineError as exc:
                        error = str(exc)
                        break
                table.append({'k': k, 'alpha': alpha,
                              'score': float(np.mean(scores)) if error is None else None,
                              'error': error})
        valid = [entry for entry in table if entry['score'] is not None]
        if not valid:
            raise BaselineError('No valid CV candidate: ' + repr(table))
        best = min(valid, key=lambda entry: (entry['score'], entry['k'], entry['alpha']))
        return (best['k'], best['alpha']), table

    def fit(self, values, covariates=None, *, patterns=None):
        candidate = copy.copy(self)
        candidate.__dict__.pop('last_prediction_', None)
        candidate._fit_in_place(values, covariates, patterns=patterns)
        self.__dict__ = candidate.__dict__
        return self

    def _fit_in_place(self, values, covariates=None, *, patterns=None):
        x = simplex(values, partial=True)
        self.donors_ = x[np.isfinite(x).all(axis=1)].copy()
        if len(self.donors_) < 2:
            raise BaselineError('Adaptive tuning needs at least two complete donors')
        inferred = [tuple(int(j) for j in np.flatnonzero(~np.isfinite(row))) for row in x if np.isnan(row).any()]
        patterns = inferred if patterns is None else [tuple(int(j) for j in p) for p in patterns]
        if not patterns:
            raise BaselineError('Supply training missingness patterns when fit data are complete')
        for pattern in patterns:
            if not pattern or len(set(pattern)) != len(pattern) or min(pattern) < 0 or max(pattern) >= x.shape[1]:
                raise BaselineError('Invalid missingness pattern')
        self.patterns_ = patterns  # repeats retain empirical pattern frequencies
        self.leave_n_ = self.leave_n if self.leave_n is not None else len(patterns)
        self.selected_, self.cv_, self.cv_sizes_ = {}, {}, {}
        if self.adaptive:
            for pattern in sorted(set(patterns)):
                count = self.leave_n if self.leave_n is not None else patterns.count(pattern)
                self.cv_sizes_[str(pattern)] = count
                self.selected_[pattern], self.cv_[str(pattern)] = self._tune([pattern], count)
        else:
            self.global_, self.cv_['global'] = self._tune(patterns, self.leave_n_)
        self.fit_summary_ = {'method': 'tsagris_adaptive' if self.adaptive else 'tsagris_global_cv',
                             'donors': len(self.donors_), 'training_patterns': [list(p) for p in patterns],
                             'seed': self.seed, 'leave_n': self.leave_n_, 'repetitions': self.repetitions}
        return self

    def with_degenerate_policies(self):
        """Return a configured copy for adapters with declared fallbacks."""
        from .registry import configure_core
        return configure_core(self, degenerate_policies=True)

    def predict_result(self, values, covariates=None):
        if not hasattr(self, 'selected_'):
            raise NotFittedError('Fit TsagrisAdaptive before prediction')
        x = simplex(values, partial=True)
        if x.shape[1] != self.donors_.shape[1]:
            raise BaselineError('Query width differs from fit')
        out, records = x.copy(), []
        for pattern in sorted(set(tuple(int(j) for j in np.flatnonzero(np.isnan(row))) for row in x)):
            if not pattern:
                continue
            if self.adaptive and pattern not in self.selected_:
                # Unseen pattern: use the explicit leave_n if supplied, else
                # total TRAIN pattern count. This documented fitting-only
                # fallback does not tune on query truths or donor values.
                self.cv_sizes_[str(pattern)] = self.leave_n_
                self.selected_[pattern], self.cv_[str(pattern)] = self._tune([pattern], self.leave_n_)
            k, alpha = self.selected_[pattern] if self.adaptive else self.global_
            rows = [i for i, row in enumerate(x) if tuple(np.flatnonzero(np.isnan(row))) == pattern]
            result = self._model(k, alpha).fit(self.donors_).predict_result(x[rows])
            out[rows] = result.values
            for event in result.diagnostics.get('row_events', []):
                event = dict(event)
                local_row = int(event.get('row', 0))
                if 0 <= local_row < len(rows):
                    event['row'] = int(rows[local_row])
                event.update({'pattern': list(pattern), 'k': k, 'alpha': alpha})
                records.append(event)
        return CoreResult(out, {'method': 'tsagris_adaptive' if self.adaptive else 'tsagris_global_cv',
                               'seed': self.seed, 'leave_n': self.leave_n_, 'repetitions': self.repetitions,
                               'cv': self.cv_.copy(), 'pattern_cv_sizes': self.cv_sizes_.copy(),
                               'row_events': records})

    def predict(self, values, covariates=None):
        result = self.predict_result(values)
        self.last_prediction_ = result.diagnostics
        return result.values

    def impute(self, values, covariates=None):
        return self.fit(values).predict(values)
