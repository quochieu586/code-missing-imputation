"""MICoDa in one place: chapter 8 core plus the COVID counts adapter (A17/R1).

Core (`MICoDa`, dataset-agnostic): ALR regression, Eq. 8.2/Lemma 1, Algorithms 1/2.
Semantics='paper' preserves an observed reference in V2. 'author_code' selects
R-style pairwise regression and reference overwrite. Pinned-R replay additionally
needs fixed_iterations=True, W=0, reference last and appropriately prepared input;
the mode name alone does not guarantee R parity or observed-count preservation.
The complete-case paper fit, rank checks and configurable final rescale are
explicit. Covariates must already be complete numeric encodings.

Adapter (`MICoDaCovid`, COVID-specific policy): raw counts + totals + mask in,
CovidResult out. Complete all-zero rows are omitted from the entire transductive
fit, not just initial OLS; their raw output is unchanged. No Other column and no
fallback model. The core above never sees dataset policy.
"""
import copy
from collections import Counter
from dataclasses import dataclass

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


# ---------------------------------------------------------------------------
# COVID counts adapter: A17/R1 is the default contract; other=True selects the
# A18/R2 sensitivity variant (Other column, residual shared with Other).
# Dataset policy lives below this line only; the core above stays reusable for
# any composition and R-parity checks.
# ---------------------------------------------------------------------------
@dataclass
class Prepared:
    raw: np.ndarray                 # visible counts; hidden truth already removed
    totals: np.ndarray
    values: np.ndarray              # excluded rows remain zero, outside core domain
    core_rows: np.ndarray
    delta: float
    positive_source_cells: int
    events: list[dict]
    other: bool = False             # A18: values carry a trailing Other column


@dataclass
class CovidResult:
    values: np.ndarray | None       # None only when input cannot be validated
    status: str
    diagnostics: dict
    unassigned: np.ndarray | None


def _visible_input(counts, totals, hidden):
    try:
        raw = np.array(counts, dtype=float, copy=True)
        totals = np.array(totals, dtype=float, copy=True)
    except (ValueError, TypeError) as exc:
        raise BaselineInputError('Counts and totals must be numeric arrays') from exc
    if raw.ndim != 2 or not len(raw) or raw.shape[1] < 2 or totals.shape != (len(raw),):
        raise BaselineInputError('Expected nonempty N x D counts (D >= 2) and N totals')
    if hidden is not None:
        hidden = np.asarray(hidden)
        if hidden.shape != raw.shape or hidden.dtype != bool or (hidden & ~np.isfinite(raw)).any():
            raise BaselineInputError('Hidden mask must be a boolean subset of observed counts')
        raw[hidden] = np.nan
    if np.isinf(raw).any() or (raw[np.isfinite(raw)] < 0).any():
        raise BaselineInputError('Visible counts must be finite and nonnegative')
    if not np.isfinite(totals).all() or (totals <= 0).any():
        raise BaselineInputError('Totals must be finite and positive')
    with np.errstate(over='ignore'):
        mass = np.nansum(raw, axis=1)
    if not np.isfinite(mass).all() or (mass > totals).any():
        raise BaselineInputError('Visible counts exceed total_sequence')
    return raw, totals


def _prepare(raw, totals, *, other=False):
    with np.errstate(under='ignore'):
        values = raw / totals[:, None]
    if ((raw > 0) & (values == 0)).any():
        raise NumericalError('Positive proportion is not representable in float64')
    positives = values[values > 0]
    if not positives.size:
        raise InsufficientDataError('No positive visible target for min/2')
    delta = float(positives.min() / 2)
    if delta <= 0:
        raise NumericalError('min/2 is not representable in float64')
    known = np.isfinite(raw)
    complete = known.all(axis=1)
    excluded = complete & (raw == 0).all(axis=1)
    if other:
        # A18 parity with data.model_input: Other is derivable only when every
        # target is visible; masked or natively partial rows leave it unknown.
        other_values = np.full(len(raw), np.nan)
        other_values[complete] = (totals[complete] - raw[complete].sum(axis=1)) / totals[complete]
        values = np.column_stack((values, other_values))
        known = np.column_stack((known, complete))
        complete = known.all(axis=1)
    events = []
    known_t = known[:, :-1] if other else known      # raw has no Other column
    for i in range(len(raw)):
        if excluded[i]:
            events.append({'row': i, 'code': 'all_zero_complete_excluded'})
        if complete[i] and raw[i].sum() < totals[i] and not other:
            events.append({'row': i, 'code': 'complete_residual_positive'})
        if excluded[i]:
            continue
        values[i, known[i] & (values[i] == 0)] = delta
        if complete[i]:
            values[i] /= values[i].sum()
        else:
            if known_t[i].any() and (raw[i, known_t[i]] == 0).all():
                events.append({'row': i, 'code': 'partial_visible_all_zero'})
            headroom = max(1e-12, int((~known[i]).sum()) * delta)
            if not 0 < headroom < 1:
                raise BaselineInputError(f'row {i}: invalid model headroom {headroom}')
            mass = float(np.nansum(values[i]))
            if mass > 1 - headroom:
                values[i, known[i]] *= (1 - headroom) / mass
                events.append({'row': i, 'code': 'model_observed_scaled'})
    core_rows = ~excluded
    core = values[core_rows]
    if (core[np.isfinite(core)] <= 0).any() or not np.allclose(
            values[complete & core_rows].sum(axis=1), 1., rtol=0, atol=1e-10):
        raise NumericalError('Preparation left an invalid model composition')
    if (np.nansum(values[~complete], axis=1) >= 1).any():
        raise NumericalError('Partial model composition has no representable headroom')
    return Prepared(raw, totals, values, core_rows, delta, int(positives.size), events, other)


def prepare(counts, totals, hidden=None, *, other=False):
    """A on visible target proportions; no closure of partial observed mass."""
    return _prepare(*_visible_input(counts, totals, hidden), other=other)


def _diagnostics(status, events, **fields):
    return {'status': status, **fields, 'row_events': events,
            'warning_counts': dict(Counter(event['code'] for event in events))}


def _allocate_budget(weights, residual):
    """Scale by the largest weight, retaining representable tiny allocations."""
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.max() <= 0:
        return None
    scale = weights.max()
    with np.errstate(under='ignore', over='ignore', invalid='ignore'):
        scaled = weights / scale
        factor = (residual / scaled.sum()) / scale
        allocation = weights * factor if np.isfinite(factor) and factor > 0 else scaled / scaled.sum() * residual
    if not np.isfinite(allocation).all() or (allocation < 0).any() or not allocation.any():
        return None
    if abs(allocation.sum() - residual) > 1e-10 * max(1., residual):
        return None
    return allocation


def restore(prepared, proposals):
    """R1: raw observed locked; entire residual goes to missing targets only.

    R2 (other variant): the residual is shared between missing targets and Other
    by proposal weight, like data.restore_counts does for Hron/Tsagris; Other's
    share stays implicit (targets are the only output columns).
    """
    if prepared.other:
        if np.shape(proposals) != prepared.values.shape:
            raise BaselineInputError('Expected proposals for every target plus Other')
    elif np.shape(proposals) != prepared.raw.shape:
        raise BaselineInputError('Expected one proposal per target, no Other column')
    result = prepared.raw.copy()
    events = list(prepared.events)
    unassigned = prepared.totals - np.nansum(prepared.raw, axis=1)
    missing = np.isnan(prepared.raw)
    for i in np.flatnonzero(missing.any(axis=1)):
        residual = float(unassigned[i])
        if residual == 0:
            result[i, missing[i]] = 0.
            continue
        if prepared.other:
            # R2: Other carries proposal weight, so even a single missing target
            # no longer absorbs the whole residual (no composition shortcut).
            weights = np.r_[np.asarray(proposals)[i, :-1][missing[i]], proposals[i, -1]]
            allocation = _allocate_budget(weights, residual)
            if allocation is not None:
                allocation = allocation[:-1]    # Other's share stays implicit
        elif missing[i].sum() == 1:
            # With one unknown the composition constraint determines it exactly;
            # no regression weight (possibly underflowed to zero) is needed.
            allocation = np.array([residual])
        else:
            allocation = _allocate_budget(np.asarray(proposals)[i, missing[i]], residual)
        if allocation is None:
            events.append({'row': int(i), 'code': 'allocation_unavailable'})
            continue
        result[i, missing[i]] = allocation
        # A completed CSV must also pass the raw loader's strict sum <= N test.
        # Correct floating-point overshoot on an IMPUTED cell only, never a raw
        # observation. One nextafter step avoids rounding the subtraction back up.
        corrected = False
        pivot = np.flatnonzero(missing[i])[np.argmax(allocation)]
        # Pairwise summation can round to the same total after one correction.
        # Bound representational repairs by row width; otherwise report failure.
        for _ in range(result.shape[1]):
            excess = float(result[i].sum() - prepared.totals[i])
            if excess <= 0 or excess > 1e-10 * max(1., prepared.totals[i]):
                break
            value = np.nextafter(result[i, pivot] - excess, 0.)
            if value < 0:
                break
            result[i, pivot] = value
            corrected = True
        if result[i].sum() > prepared.totals[i]:
            result[i] = prepared.raw[i]
            events.append({'row': int(i), 'code': 'allocation_unavailable'})
            continue
        if corrected:
            events.append({'row': int(i), 'code': 'allocation_roundoff_corrected'})
        # R1: nothing left over. R2: what remains is Other's implicit share.
        unassigned[i] = 0. if not prepared.other else residual - float(result[i, missing[i]].sum())
    visible = ~missing
    if not np.array_equal(result[visible], prepared.raw[visible]):
        raise NumericalError('Restoration changed raw observed counts')
    if np.isinf(result).any() or (result[np.isfinite(result)] < 0).any() or not np.allclose(
            np.nansum(result, axis=1) + unassigned, prepared.totals,
            rtol=0, atol=1e-10 * np.maximum(1., prepared.totals)):
        raise NumericalError('Restoration violates raw count budget')
    needed = int(missing.sum())
    solved = int(np.isfinite(result[missing]).sum())
    status = 'completed' if solved == needed else ('partial' if solved else 'failed')
    return CovidResult(result, status, _diagnostics(status, events,
        missing_cells=needed, resolved_cells=solved, unresolved_cells=needed - solved), unassigned)


class MICoDaCovid:
    """Approved deterministic V2 adaptation, not a general-purpose policy engine."""
    def __init__(self, *, max_iter=120, tol=1e-8, reference=None, other=False):
        if not isinstance(other, (bool, np.bool_)):
            raise BaselineInputError('other must be a boolean')
        # Reuse core configuration validation; no fitted state held by this adapter.
        MICoDa(max_iter=max_iter, tol=tol, reference=reference)
        self.max_iter, self.tol, self.reference, self.other = max_iter, tol, reference, bool(other)

    def impute_counts(self, counts, totals, hidden=None):
        config = {'version': 'v2', 'semantics': 'paper', 'zero_policy': 'min2_closure',
                  'budget_policy': 'R2' if self.other else 'R1',
                  'all_zero_policy': 'exclude_initial_and_refit',
                  'max_iter': int(self.max_iter), 'tol': float(self.tol),
                  'reference_requested': None if self.reference is None else int(self.reference),
                  'covariates': 'intercept_only', 'W': 0., 'other_column': self.other,
                  'positive_state_required': True}
        raw = n = prepared = None
        try:
            raw, n = _visible_input(counts, totals, hidden)
            prepared = _prepare(raw, n, other=self.other)
            core_values = prepared.values[prepared.core_rows]
            initial_rows = int(np.isfinite(core_values).all(axis=1).sum())
            if initial_rows <= 1:
                raise InsufficientDataError('Complete-case initialization needs more than one row')
            core = MICoDa(max_iter=self.max_iter, tol=self.tol, reference=self.reference)
            core.fit(core_values, require_positive_state=True)
            proposals = np.full(prepared.values.shape, np.nan)
            proposals[prepared.core_rows] = core.fit_result_.values
            result = restore(prepared, proposals)
            fit = core.fit_result_.diagnostics
            for key in ('reference', 'iterations', 'relative_changes', 'converged',
                        'termination_reason', 'stopping_formula', 'initial_regression_support'):
                result.diagnostics[key] = fit[key]
            result.diagnostics.update(config=config, delta=prepared.delta,
                positive_source_cells=prepared.positive_source_cells,
                initial_complete_rows=initial_rows, core_rows=int(prepared.core_rows.sum()),
                excluded_all_zero_rows=int((~prepared.core_rows).sum()),
                coefficients=core.coefficients_.tolist(),
                coefficient_target_indices=[j for j in range(core.fit_result_.values.shape[1])
                                            if j != core.reference_])
            if not fit['converged']:
                result.diagnostics['row_events'].append({'code': 'non_converged'})
                result.diagnostics['warning_counts']['non_converged'] = 1
            return result
        except BaselineError as exc:
            code = 'insufficient_fit' if type(exc) is InsufficientDataError else exc.code
            context = dict(exc.context)
            if prepared is not None and context.get('stage') == 'pre_refit' and 'row' in context:
                context['core_row'] = context['row']
                context['row'] = int(np.flatnonzero(prepared.core_rows)[context['core_row']])
            events = list(prepared.events) if prepared is not None else []
            events.append({'code': code, 'message': str(exc), **context})
            missing_cells = int(np.isnan(raw).sum()) if raw is not None else None
            diagnostics = _diagnostics('failed', events, config=config,
                error={'code': code, 'message': str(exc), 'context': context},
                missing_cells=missing_cells, resolved_cells=0, unresolved_cells=missing_cells)
            if prepared is not None:
                diagnostics.update(delta=prepared.delta,
                    positive_source_cells=prepared.positive_source_cells,
                    initial_complete_rows=int(np.isfinite(prepared.values[prepared.core_rows]).all(axis=1).sum()),
                    excluded_all_zero_rows=int((~prepared.core_rows).sum()))
            return CovidResult(raw, 'failed', diagnostics,
                n - np.nansum(raw, axis=1) if raw is not None else None)
