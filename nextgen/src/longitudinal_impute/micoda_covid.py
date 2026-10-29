"""COVID A17/R1 adapter. Raw counts never enter the ALR core directly.

Complete all-zero rows are omitted from the entire transductive fit, not just
initial OLS. Their raw output is unchanged. No Other column or fallback model.
"""
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .baselines.micoda import MICoDa
from .errors import BaselineError, BaselineInputError, InsufficientDataError, NumericalError


@dataclass
class Prepared:
    raw: np.ndarray                 # visible counts; hidden truth already removed
    totals: np.ndarray
    values: np.ndarray              # excluded rows remain zero, outside core domain
    core_rows: np.ndarray
    delta: float
    positive_source_cells: int
    events: list[dict]


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


def _prepare(raw, totals):
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
    events = []
    for i in range(len(raw)):
        if excluded[i]:
            events.append({'row': i, 'code': 'all_zero_complete_excluded'})
        if complete[i] and raw[i].sum() < totals[i]:
            events.append({'row': i, 'code': 'complete_residual_positive'})
        if excluded[i]:
            continue
        values[i, known[i] & (raw[i] == 0)] = delta
        if complete[i]:
            values[i] /= values[i].sum()
        else:
            if known[i].any() and (raw[i, known[i]] == 0).all():
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
    return Prepared(raw, totals, values, core_rows, delta, int(positives.size), events)


def prepare(counts, totals, hidden=None):
    """A on visible target proportions; no closure of partial observed mass."""
    return _prepare(*_visible_input(counts, totals, hidden))


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
    """R1: raw observed locked; entire residual goes to missing targets only."""
    if np.shape(proposals) != prepared.raw.shape:
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
        # With one unknown the composition constraint determines it exactly;
        # no regression weight (possibly underflowed to zero) is needed.
        allocation = np.array([residual]) if missing[i].sum() == 1 else _allocate_budget(
            np.asarray(proposals)[i, missing[i]], residual)
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
        unassigned[i] = 0.
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
    def __init__(self, *, max_iter=120, tol=1e-8, reference=None):
        # Reuse core configuration validation; no fitted state held by this adapter.
        MICoDa(max_iter=max_iter, tol=tol, reference=reference)
        self.max_iter, self.tol, self.reference = max_iter, tol, reference

    def impute_counts(self, counts, totals, hidden=None):
        config = {'version': 'v2', 'semantics': 'paper', 'zero_policy': 'min2_closure',
                  'budget_policy': 'R1', 'all_zero_policy': 'exclude_initial_and_refit',
                  'max_iter': int(self.max_iter), 'tol': float(self.tol),
                  'reference_requested': None if self.reference is None else int(self.reference),
                  'covariates': 'intercept_only', 'W': 0., 'other_column': False,
                  'positive_state_required': True}
        raw = n = prepared = None
        try:
            raw, n = _visible_input(counts, totals, hidden)
            prepared = _prepare(raw, n)
            core_values = prepared.values[prepared.core_rows]
            initial_rows = int(np.isfinite(core_values).all(axis=1).sum())
            if initial_rows <= 1:
                raise InsufficientDataError('Complete-case initialization needs more than one row')
            core = MICoDa(max_iter=self.max_iter, tol=self.tol, reference=self.reference)
            core.fit(core_values, require_positive_state=True)
            proposals = np.full(raw.shape, np.nan)
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
                coefficient_target_indices=[j for j in range(raw.shape[1]) if j != core.reference_])
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
