"""Hron et al. 2010: Eq. (5)-(13), positive-domain paper reference.

kNN supports (1a) complete donors and (2a) available observed+target donors.
Iterative regression uses target-first pivot balances: the last raw part has
no own balance in a fixed D-1 basis, so each target is rotated to position 1.
This convention is explicit and is checked against an independent oracle.
"""
import copy
import numpy as np
from collections import OrderedDict
from .common import (BaselineError, CoreResult, matrix, clr, pivot_basis,
                     cov_matrix, stable_softmax, distance_order)
from ..errors import NotFittedError, PredictionUnavailableError, NumericalError
from .regression import ols, lts


class HronKNN:
    input_domain = 'positive'
    prediction_policy = 'rowwise_adaptation'

    def __init__(self, k=3, *, donor_policy='available', adjustment='median',
                 aggregation='median'):
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise BaselineError('k must be a positive integer')
        if donor_policy not in ('complete', 'available'):
            raise BaselineError('donor_policy must be complete or available')
        if adjustment not in ('median', 'sum') or aggregation not in ('median', 'mean'):
            raise BaselineError('Unknown adjustment/aggregation')
        self.k, self.donor_policy = k, donor_policy
        self.adjustment, self.aggregation = adjustment, aggregation

    def fit(self, values, covariates=None):
        donors = matrix(values, positive=True)
        state = dict(vars(self), donors_=donors, _donor_cache_=OrderedDict())
        state.pop('last_prediction_', None)
        self.__dict__ = state
        return self

    def _candidates(self, observed, target):
        key = (tuple(observed), int(target))
        if key in self._donor_cache_:
            value = self._donor_cache_.pop(key)
            self._donor_cache_[key] = value
            return value
        valid = np.isfinite(self.donors_)
        eligible = (valid.all(axis=1) if self.donor_policy == 'complete'
                    else valid[:, np.r_[observed, target]].all(axis=1))
        ids = np.flatnonzero(eligible)
        parts = self.donors_[ids][:, observed]
        reduce = np.median if self.adjustment == 'median' else np.sum
        value = (ids, clr(parts), reduce(parts, axis=1)) if len(ids) else (ids, None, None)
        self._donor_cache_[key] = value
        # Memory bound; all entries are fitted donor-only state, never queries.
        while len(self._donor_cache_) > 32:
            self._donor_cache_.popitem(last=False)
        return value

    def predict_result(self, values, covariates=None):
        if not hasattr(self, 'donors_'):
            raise NotFittedError('Fit HronKNN before prediction')
        x = matrix(values, positive=True, width=self.donors_.shape[1])
        out, records = x.copy(), []
        reduce = np.median if self.adjustment == 'median' else np.sum
        aggregate = np.median if self.aggregation == 'median' else np.mean
        for i, row in enumerate(x):
            observed = np.flatnonzero(np.isfinite(row))
            missing = np.flatnonzero(~np.isfinite(row))
            if not len(missing):
                continue
            if not len(observed):
                raise PredictionUnavailableError(f'row {i}: Hron distance has no observed anchor',
                                                  context={'row': i})
            for j in missing:
                ids, donor_clr, donor_scale = self._candidates(observed, j)
                if len(ids) == 0:
                    raise PredictionUnavailableError(f'row {i}, feature {j}: no eligible Hron donors',
                                                     context={'row': i, 'feature': int(j)})
                distances = np.linalg.norm(donor_clr - clr(row[observed]), axis=1)
                positions = distance_order(distances, limit=self.k)[:self.k]
                selected = ids[positions]
                scale = reduce(row[observed]) / donor_scale[positions]
                out[i, j] = aggregate(scale * self.donors_[selected, j])
                records.append({'row': i, 'feature': int(j), 'donor_ids': selected.tolist(),
                                'eligible_donors': len(ids), 'effective_k': len(selected),
                                'one_anchor': len(observed) == 1})
        return CoreResult(out, {'method': 'hron_knn', 'records': records,
                               'adjustment': self.adjustment, 'donor_policy': self.donor_policy})

    def predict(self, values, covariates=None):
        result = self.predict_result(values, covariates)
        self.last_prediction_ = result.diagnostics
        return result.values

    def impute(self, values, covariates=None):
        return self.fit(values).predict(values)


class HronIterative:
    input_domain = 'positive'
    prediction_policy = 'rowwise_adaptation'

    def __init__(self, *, regression='ls', k=3, donor_policy='available',
                 adjustment='median', max_iter=50, tol=1e-8, seed=42,
                 lts_starts=64, lts_exact_limit=5000):
        if regression not in ('ls', 'lts') or max_iter < 1 or tol < 0:
            raise BaselineError('Invalid iterative regression configuration')
        self.regression, self.max_iter, self.tol, self.seed = regression, max_iter, tol, seed
        self.lts_starts, self.lts_exact_limit = lts_starts, lts_exact_limit
        self.initializer_ = HronKNN(k, donor_policy=donor_policy, adjustment=adjustment)

    def _sweep(self, log_state, observed, *, fitting, iteration):
        reports = []
        for target in self.order_:
            missing = ~observed[:, target]
            # In inductive fitting a fully-observed target still needs a
            # predictor for missing targets in a new query population.
            if not fitting and not missing.any():
                continue
            permutation = np.r_[target, self.order_[self.order_ != target]]
            z = log_state[:, permutation] @ self.basis_
            design = np.column_stack((np.ones(len(log_state)), z[:, 1:]))
            if fitting:
                rows = observed[:, target]
                try:
                    if self.regression == 'ls':
                        beta = ols(design[rows], z[rows, 0])
                        info = {'solver': 'ols'}
                    else:
                        beta, info = lts(design[rows], z[rows, 0], seed=self.seed + int(target),
                                         starts=self.lts_starts, exact_limit=self.lts_exact_limit)
                except BaselineError as exc:
                    raise exc.add_context(stage='fit', iteration=iteration, target=int(target))
                self.models_[int(target)] = beta
                reports.append({'feature': int(target), **info})
            if missing.any():
                z[missing, 0] = design[missing] @ self.models_[int(target)]
                # Keep centered log balances between targets. Converting each
                # sweep to simplex can underflow small parts to exact zero.
                if not np.isfinite(z).all():
                    raise NumericalError('Hron balance update is nonfinite',
                                         context={'stage': 'fit' if fitting else 'prediction',
                                                  'iteration': iteration, 'target': int(target)})
                updated = z @ self.basis_.T
                if not np.isfinite(updated).all():
                    raise NumericalError('Hron centered-log state is nonfinite',
                                         context={'stage': 'fit' if fitting else 'prediction',
                                                  'iteration': iteration, 'target': int(target)})
                log_state[:, permutation] = updated
        return log_state, reports

    def _iterate(self, values, *, fitting):
        observed = np.isfinite(values)
        state = self.initializer_.predict(values)
        if not np.isfinite(state).all() or (state <= 0).any():
            raise NumericalError('Hron initializer did not produce strictly positive compositions')
        log_state = np.log(state)
        log_state -= log_state.mean(axis=1, keepdims=True)
        history, reports = [], []
        previous = cov_matrix(log_state[:, self.order_] @ self.basis_)
        for iteration in range(self.max_iter):
            log_state, report = self._sweep(log_state, observed, fitting=fitting, iteration=iteration)
            current = cov_matrix(log_state[:, self.order_] @ self.basis_)
            delta = float(np.linalg.norm(current - previous))
            history.append(delta)
            reports = report or reports
            previous = current
            if delta <= self.tol:
                break
        output = stable_softmax(log_state)
        return CoreResult(output, {'method': 'hron_iterative', 'regression': self.regression,
                                 'iterations': len(history), 'converged': history[-1] <= self.tol,
                                 'covariance_deltas': history, 'regression_diagnostics': reports,
                                 'pivot_policy': 'target_first', 'observed_counts_preserved': False,
                                 'internal_state': 'centered_log_balances',
                                 'output_underflow_parts': int((output == 0).sum())})

    def fit(self, values, covariates=None):
        # The initializer has its own fitted donors/cache; isolate it as well
        # as the regression models before publishing a successful refit.
        candidate = copy.copy(self)
        candidate.initializer_ = copy.copy(self.initializer_)
        candidate.__dict__.pop('last_prediction_', None)
        candidate._fit_in_place(values, covariates)
        self.__dict__ = candidate.__dict__
        return self

    def _fit_in_place(self, values, covariates=None):
        x = matrix(values, positive=True)
        self.order_ = np.argsort(-np.isnan(x).sum(axis=0), kind='stable')
        self.basis_, self.models_ = pivot_basis(x.shape[1]), {}
        self.initializer_.fit(x)
        self.fit_result_ = self._iterate(x, fitting=True)
        return self

    def predict_result(self, values, covariates=None):
        if not hasattr(self, 'models_'):
            raise NotFittedError('Fit HronIterative before prediction')
        x = matrix(values, positive=True, width=len(self.order_))
        if len(x) == 1:
            # Covariance stop is not identifiable for a one-row query. Use
            # update-norm stop, explicitly marked as inductive adaptation.
            state = self.initializer_.predict(x)
            if not np.isfinite(state).all() or (state <= 0).any():
                raise NumericalError('Hron initializer did not produce strictly positive compositions')
            log_state = np.log(state)
            log_state -= log_state.mean(axis=1, keepdims=True)
            deltas = []
            for iteration in range(self.max_iter):
                old = stable_softmax(log_state)
                log_state, _ = self._sweep(log_state, np.isfinite(x), fitting=False, iteration=iteration)
                state = stable_softmax(log_state)
                delta = float(np.linalg.norm(state - old))
                deltas.append(delta)
                if delta <= self.tol:
                    break
            return CoreResult(state, {'method': 'hron_iterative', 'iterations': len(deltas),
                                     'converged': deltas[-1] <= self.tol,
                                     'stopping': 'single_query_update_norm', 'deltas': deltas,
                                     'internal_state': 'centered_log_balances',
                                     'output_underflow_parts': int((state == 0).sum())})
        return self._iterate(x, fitting=False)

    def predict(self, values, covariates=None):
        result = self.predict_result(values, covariates)
        self.last_prediction_ = result.diagnostics
        return result.values

    def impute(self, values, covariates=None):
        return self.fit(values).fit_result_.values.copy()
