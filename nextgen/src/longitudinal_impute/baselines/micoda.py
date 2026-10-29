"""MICoDa chapter 8: ALR regression, Eq. 8.2/Lemma 1, Algorithms 1/2.

Semantics='paper' preserves an observed reference in V2. 'author_code' selects
R-style pairwise regression and reference overwrite. Pinned-R replay additionally
needs fixed_iterations=True, W=0, reference last and appropriately prepared input;
the mode name alone does not guarantee R parity or observed-count preservation.
The complete-case paper fit, rank checks and configurable final rescale are
explicit. Covariates must already be complete numeric encodings.
"""
import copy
import numpy as np
from .common import (BaselineError, CoreResult, matrix, design_matrix, stable_softmax)
from ..errors import NotFittedError, InsufficientDataError, NumericalError, BaselineInputError
from .regression import ols


def allocate_missing(row, predicted_alr, reference, total, *, force_reference=False):
    """Lemma 1 with observed-reference case explicit; no exp overflow."""
    row = np.array(row, float, copy=True)
    missing = ~np.isfinite(row)
    if force_reference:
        missing[reference] = True
    if not missing.any():
        return row
    logits = np.insert(np.asarray(predicted_alr, float), reference, 0.)
    residual = total - row[~missing].sum()
    if residual < -1e-10:
        raise BaselineError('Observed composition exceeds the declared fixed total')
    row[missing] = max(0., residual) * stable_softmax(logits[missing])
    return row


class MICoDa:
    input_domain = 'nonnegative'
    prediction_policy = 'rowwise_adaptation'

    def __init__(self, *, version='v2', reference=None, total=1., W=0.,
                 max_iter=50, tol=1e-8, semantics='paper', fit_rows=None,
                 final_rescale=True, fixed_iterations=False):
        if version not in ('v1', 'v2') or semantics not in ('paper', 'author_code'):
            raise BaselineInputError('Unknown MICoDa version/semantics')
        if (not np.isfinite(total) or total <= 0 or not np.isfinite(W) or W < 0
                or isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, (int, np.integer))
                or max_iter < 1 or not np.isfinite(tol) or tol < 0):
            raise BaselineInputError('Invalid MICoDa total/W/iteration configuration')
        self.version, self.reference, self.total, self.W = version, reference, float(total), float(W)
        self.max_iter, self.tol, self.semantics = max_iter, tol, semantics
        self.fit_rows = fit_rows or ('pairwise' if semantics == 'author_code' else 'complete')
        if self.fit_rows not in ('complete', 'pairwise'):
            raise BaselineError('fit_rows must be complete or pairwise')
        self.final_rescale, self.fixed_iterations = final_rescale, fixed_iterations
        if reference is not None and (isinstance(reference, (bool, np.bool_))
                or not isinstance(reference, (int, np.integer)) or reference < 0):
            raise BaselineInputError('Reference must be a nonnegative integer index')

    def _prepare(self, values):
        x = matrix(values, width=getattr(self, 'width_', None))
        full = np.isfinite(x).all(axis=1)
        if (np.nansum(x, axis=1) > self.total + 1e-10).any():
            raise BaselineError('Observed mass exceeds fixed MICoDa total')
        if not np.allclose(x[full].sum(axis=1), self.total, atol=1e-10, rtol=0):
            raise BaselineError('Complete reference compositions must sum to fixed total')
        shifted = x + self.W
        if (shifted[np.isfinite(shifted)] <= 0).any():
            raise BaselineError('MICoDa ALR needs positive parts: declare W > 0 for zeros')
        return x, shifted

    def _regress(self, state, design, *, initial=False):
        columns = [j for j in range(self.width_) if j != self.reference_]
        complete = np.isfinite(state).all(axis=1)
        eligible_masks = []
        for j in columns:
            eligible = complete.copy() if initial and self.fit_rows == 'complete' else (
                np.isfinite(state[:, j]) & np.isfinite(state[:, self.reference_]))
            eligible &= (state[:, j] > 0) & (state[:, self.reference_] > 0)
            eligible_masks.append(eligible)

        if all(np.array_equal(eligible_masks[0], eligible) for eligible in eligible_masks[1:]):
            eligible = eligible_masks[0]
            if eligible.sum() <= design.shape[1]:
                raise InsufficientDataError(f'features {columns}: insufficient ALR observations for parameters')
            response = (np.log(state[eligible][:, columns])
                        - np.log(state[eligible, self.reference_])[:, None])
            try:
                coefficients = ols(design[eligible], response)
            except BaselineError as exc:
                raise exc.add_context(stage='fit', features=columns)
            return coefficients, [int(mask.sum()) for mask in eligible_masks]

        coefficients, supports = [], []
        for j, eligible in zip(columns, eligible_masks):
            # Paper assumption A2 is strict n_complete > number of parameters.
            if eligible.sum() <= design.shape[1]:
                raise InsufficientDataError(f'feature {j}: insufficient ALR observations for parameters')
            response = np.log(state[eligible, j]) - np.log(state[eligible, self.reference_])
            try:
                coefficients.append(ols(design[eligible], response))
            except BaselineError as exc:
                raise exc.add_context(stage='fit', feature=j)
            supports.append(int(eligible.sum()))
        return np.column_stack(coefficients), supports

    def _update(self, source, state, design, coefficients):
        out = state.copy()
        partial = ~np.isfinite(source).all(axis=1)
        predicted = design @ coefficients
        for i in np.flatnonzero(partial):
            if self.version == 'v2':
                out[i] = allocate_missing(source[i], predicted[i], self.reference_, self.shifted_total_,
                                           force_reference=self.semantics == 'author_code')
            elif not np.isfinite(source[i]).any():
                out[i] = self.shifted_total_ * stable_softmax(np.insert(predicted[i], self.reference_, 0.))
            else:
                row = out[i].copy()
                for target in range(self.width_):
                    if target == self.reference_:
                        continue
                    tmp = row.copy()
                    tmp[target], tmp[self.reference_] = np.nan, np.nan
                    allocated = allocate_missing(tmp, predicted[i], self.reference_, self.shifted_total_)
                    # Other still-missing parts remain unknown until their turn.
                    row[target], row[self.reference_] = allocated[target], allocated[self.reference_]
                out[i] = row
        if not np.isfinite(out).all():
            raise NumericalError('MICoDa update left nonfinite compositions')
        return out

    def _output(self, state):
        # Chapter Remark 1: rescale shifted compositions, NOT subtract W.
        return state * (self.total / self.shifted_total_) if self.final_rescale else state.copy()

    def fit(self, values, covariates=None, *, require_positive_state=False):
        # Calculate fitted attributes off-object, then publish them together.
        # A failed refit therefore cannot mix new schema state with old weights.
        candidate = copy.copy(self)
        candidate.__dict__.pop('last_prediction_', None)
        candidate._fit_in_place(values, covariates, require_positive_state=require_positive_state)
        self.__dict__ = candidate.__dict__
        return self

    def _fit_in_place(self, values, covariates=None, *, require_positive_state=False):
        checked = matrix(values)
        self.width_ = checked.shape[1]
        raw, source = self._prepare(checked)
        self.reference_ = int(np.argmin(np.isnan(raw).sum(axis=0))) if self.reference is None else self.reference
        if not isinstance(self.reference_, (int, np.integer)) or not 0 <= self.reference_ < self.width_:
            raise BaselineInputError('Reference index outside composition')
        self.shifted_total_ = self.total + self.width_ * self.W
        design = design_matrix(covariates, len(raw))
        self.covariate_width_ = design.shape[1] - 1
        coefficients, support = self._regress(source, design, initial=True)
        state, history = source.copy(), []
        for iteration in range(self.max_iter):
            new = self._update(source, state, design, coefficients)
            if require_positive_state and (new <= 0).any():
                row, feature = np.argwhere(new <= 0)[0]
                raise NumericalError('Nonpositive model state before refit; possible allocation underflow',
                    context={'stage': 'pre_refit', 'iteration': iteration + 1,
                             'row': int(row), 'feature': int(feature)})
            delta = None if not np.isfinite(state).all() else float(
                np.linalg.norm(new - state) / max(np.linalg.norm(state), np.finfo(float).tiny))
            if delta is not None and not np.isfinite(delta):
                raise NumericalError('Nonfinite MICoDa stopping change')
            history.append(delta)
            state = new
            coefficients, _ = self._regress(state, design)
            if not self.fixed_iterations and delta is not None and delta <= self.tol:
                break
        self.coefficients_ = coefficients
        converged = history[-1] is not None and history[-1] <= self.tol
        self.fit_result_ = CoreResult(self._output(state), {
            'method': 'micoda', 'version': self.version, 'semantics': self.semantics,
            'reference': int(self.reference_), 'initial_regression_support': support,
            'fit_rows': self.fit_rows, 'iterations': len(history), 'relative_changes': history,
            'converged': converged,
            'termination_reason': 'fixed_iterations' if self.fixed_iterations else ('tolerance' if converged else 'max_iter'),
            'stopping_formula': 'frobenius_full_state_relative_change',
            'positive_state_required': require_positive_state,
            'W': self.W, 'final_rescale': self.final_rescale,
            'reference_overwrite_compatibility': self.semantics == 'author_code'})
        return self

    def predict_result(self, values, covariates=None):
        if not hasattr(self, 'coefficients_'):
            raise NotFittedError('Fit MICoDa before prediction')
        _, source = self._prepare(values)
        design = design_matrix(covariates, len(source))
        if design.shape[1] != self.covariate_width_ + 1:
            raise BaselineError('Covariate dimension differs from fitted model')
        # Fixed fitted coefficients: no transductive refitting on query data.
        state = self._update(source, source, design, self.coefficients_)
        return CoreResult(self._output(state), {
            'method': 'micoda', 'version': self.version, 'semantics': self.semantics,
            'reference': int(self.reference_), 'prediction_mode': 'fixed_coefficients',
            'fit_diagnostics': self.fit_result_.diagnostics})

    def predict(self, values, covariates=None):
        result = self.predict_result(values, covariates)
        self.last_prediction_ = result.diagnostics
        return result.values

    def impute(self, values, covariates=None):
        return self.fit(values, covariates).fit_result_.values.copy()
