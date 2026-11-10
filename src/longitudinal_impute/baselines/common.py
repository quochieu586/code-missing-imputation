"""Numerical contracts shared by paper references (numpy only)."""
from dataclasses import dataclass
from typing import Any
import numpy as np
from ..errors import (BaselineError, BaselineInputError, InsufficientDataError,
                      NumericalError, NotFittedError, PredictionUnavailableError,
                      RankDeficientError)


@dataclass(frozen=True)
class CoreResult:
    values: np.ndarray
    diagnostics: dict[str, Any]

    def __post_init__(self):
        # Every core exposes the same flat row-event channel. Method-specific
        # tuning data may remain alongside it, but consumers need not inspect
        # implementation-specific diagnostic trees.
        diagnostics = dict(self.diagnostics)
        diagnostics.setdefault('row_events', diagnostics.get('records', []))
        from .diagnostics import canonical_diagnostics
        diagnostics.update(canonical_diagnostics(diagnostics))
        object.__setattr__(self, 'diagnostics', diagnostics)


def matrix(values, *, positive=False, complete=False, width=None):
    x = np.array(values, dtype=np.float64, copy=True)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] < 2:
        raise BaselineInputError('Expected a nonempty N x D matrix with D >= 2')
    if width is not None and x.shape[1] != width:
        raise BaselineInputError('Feature dimension differs from fitted data')
    if np.isinf(x).any() or (complete and np.isnan(x).any()):
        raise BaselineInputError('Infinite values or missing values outside supported domain')
    finite = x[np.isfinite(x)]
    if (finite < 0).any() or (positive and (finite <= 0).any()):
        raise BaselineInputError('This reference requires positive parts' if positive
                                 else 'Negative compositions are not valid')
    return x


def simplex(values, *, partial=False, positive=False):
    x = matrix(values, positive=positive, complete=not partial)
    sums = np.nansum(x, axis=1)
    full = np.isfinite(x).all(axis=1)
    if (sums > 1 + 1e-10).any() or not np.allclose(sums[full], 1., atol=1e-10, rtol=0):
        raise BaselineInputError('Reference simplex requires total 1 for complete rows and observed mass <= 1')
    return x


def closure(values):
    x = np.asarray(values, dtype=float)
    mass = x.sum(axis=-1, keepdims=True)
    if not np.isfinite(x).all() or (x < 0).any() or (mass <= 0).any():
        raise BaselineError('Cannot close nonfinite, negative or zero-total vector')
    return x / mass


def stable_softmax(logits):
    z = np.asarray(logits, dtype=float)
    if not np.isfinite(z).all():
        raise NumericalError('Nonfinite log-ratio prediction')
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def clr(values):
    x = np.asarray(values, dtype=float)
    if not np.isfinite(x).all() or (x <= 0).any():
        raise BaselineInputError('CLR requires finite positive parts; use an explicit encoder zero policy')
    logs = np.log(x)
    return logs - logs.mean(axis=-1, keepdims=True)


def pivot_basis(d):
    """Eq. (8) Hron: first coordinate is log(gmean(rest)/first)."""
    basis = np.zeros((d, d - 1))
    for j in range(d - 1):
        remaining = d - j - 1
        basis[j, j] = -np.sqrt(remaining / (remaining + 1))
        basis[j + 1:, j] = 1 / np.sqrt(remaining * (remaining + 1))
    return basis


def cov_matrix(z):
    if len(z) < 2:
        raise InsufficientDataError('At least two rows required for covariance stopping')
    return np.atleast_2d(np.cov(z, rowvar=False, ddof=1))


def distance_order(distances, limit=None):
    """Stable donor-ID tie break including float-roundoff-equivalent distances.

    Sort first, then group only values within 1e-12 of the group's first value
    (relative tolerance 1e-12). This avoids chaining unrelated near ties.
    With limit=k only the first k entries have fully tie-resolved ordering;
    the entire cutoff tie group is resolved so its prefix is exactly unchanged.
    """
    values = np.asarray(distances, float)
    order = np.argsort(values, kind='stable')
    ranked = values[order]
    near = np.flatnonzero(np.isclose(ranked[1:], ranked[:-1], atol=1e-12, rtol=1e-12))
    cursor = 0
    while cursor < len(near):
        start = int(near[cursor])
        if limit is not None and start >= limit:
            break
        threshold = 1e-12 + 1e-12 * abs(ranked[start])
        stop = int(np.searchsorted(ranked, ranked[start] + threshold, side='right'))
        # Keep the original first-value tolerance at floating-point boundaries.
        while stop > start + 1 and abs(ranked[stop - 1] - ranked[start]) > threshold:
            stop -= 1
        stop = max(stop, start + 2)
        order[start:stop] = np.sort(order[start:stop])
        cursor = int(np.searchsorted(near, stop, side='left'))
    return order


def design_matrix(covariates, n):
    if covariates is None:
        return np.ones((n, 1))
    x = np.asarray(covariates, dtype=float)
    if x.ndim != 2 or len(x) != n or not np.isfinite(x).all():
        raise BaselineError('Covariates must be a complete finite N x P numeric matrix')
    return np.column_stack((np.ones(n), x))


def fill_within_residual(values, proposal):
    """Put per-cell proposals into the missing targets without exceeding the row residual.

    values: N x (D+1) model input (targets c/N then Other, NaN = missing).
    proposal: N x D proposed proportions; only the missing target cells are read.
    Per incomplete row, R = 1 - sum(observed targets). If the proposals sum to more than R
    they are scaled down to R; Other gets the rest, max(R - filled, 0). A row whose filled
    mass and Other are both 0 gets Other = 1 as a pure weight, so restore_counts sends any
    (rounding-level) residual to Other instead of failing. Returns (matrix, rows_scaled_down).
    """
    x = np.asarray(values, float)
    targets = x[:, :-1]
    missing = ~np.isfinite(targets)
    rows = missing.any(axis=1)
    proposal = np.asarray(proposal, float)
    if proposal.shape != targets.shape or not np.isfinite(proposal[missing]).all() or (proposal[missing] < 0).any():
        raise BaselineInputError('Proposals must be finite, nonnegative and N x D')
    residual = np.maximum(1. - np.nansum(targets, axis=1), 0.)
    fill = np.where(missing, proposal, 0.)
    mass = fill.sum(axis=1)
    over = rows & (mass > residual)
    fill[over] *= (residual[over] / mass[over])[:, None]
    out = x.copy()
    out[:, :-1][missing] = fill[missing]
    other = np.maximum(residual - fill.sum(axis=1), 0.)
    other[rows & (other == 0) & (fill.sum(axis=1) == 0)] = 1.
    out[rows, -1] = other[rows]
    return out, int(over.sum())


def observed_columns(values, model):
    """Model input check shared by the floor/time baselines: N x (D+1), finite cells in [0, 1]."""
    x = np.array(values, dtype=float)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] < 3:
        raise BaselineInputError(f'{model} expects a nonempty N x (D+1) matrix (targets plus Other)')
    finite = x[np.isfinite(x)]
    if np.isinf(x).any() or (finite < 0).any() or (finite > 1 + 1e-10).any():
        raise BaselineInputError(f'{model} expects proportions in [0, 1] or NaN')
    return x
