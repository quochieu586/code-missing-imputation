"""Explicit OLS and raw least-trimmed-squares solvers.

LTS enumerates h-subsets when tractable, otherwise seeded multi-start C-steps.
It is NOT robustbase's reweighted ltsReg; approximation status is reported.
"""
from itertools import combinations
from math import comb
import numpy as np
from .common import BaselineError, BaselineInputError, InsufficientDataError, NumericalError, RankDeficientError


def _regression_arrays(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise BaselineInputError('Regression design must be a nonempty N x P matrix')
    if y.ndim not in (1, 2) or y.shape[0] != x.shape[0]:
        raise BaselineInputError('Regression response must have the same row count as its design')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise BaselineInputError('Nonfinite regression data')
    if x.shape[0] < x.shape[1]:
        raise InsufficientDataError('Regression design has fewer observations than parameters')
    return x, y


def _solve_full_rank(x, y):
    x, y = _regression_arrays(x, y)
    try:
        coefficients, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    except np.linalg.LinAlgError as exc:
        raise NumericalError('Least-squares decomposition failed') from exc
    if rank < x.shape[1]:
        raise RankDeficientError('Regression design is rank deficient')
    if not np.isfinite(coefficients).all():
        raise NumericalError('Least-squares solution is nonfinite')
    return coefficients


def ols(x, y):
    return _solve_full_rank(x, y)


def lts(x, y, *, h=None, seed=42, starts=64, max_csteps=50, exact_limit=5000):
    x, y = _regression_arrays(x, y)
    n, p = x.shape
    h = (n + p + 1) // 2 if h is None else h
    if not isinstance(h, (int, np.integer)) or not p <= h <= n:
        raise BaselineError('LTS h must be an integer between parameter count and n')
    if starts < 1 or max_csteps < 1 or exact_limit < 1:
        raise BaselineError('LTS solver limits must be positive')
    exact = comb(n, h) <= exact_limit
    rng = np.random.default_rng(seed)
    subsets = combinations(range(n), h) if exact else (
        [np.arange(n)] + [rng.choice(n, p, replace=False) for _ in range(starts)])
    best = None
    valid = 0
    rank_deficient_subsets = 0
    for subset in subsets:
        subset = np.asarray(subset, dtype=int)
        try:
            beta = _solve_full_rank(x[subset], y[subset])
        except RankDeficientError:
            rank_deficient_subsets += 1
            continue
        # Exhaustive h-subsets solve the global trimmed objective. C-steps
        # only accelerate a start; they cannot worsen its retained objective.
        for _ in range(max_csteps):
            resid = (y - x @ beta) ** 2
            selected = np.argsort(resid, kind='stable')[:h]
            try:
                updated = _solve_full_rank(x[selected], y[selected])
            except RankDeficientError:
                break
            if np.allclose(updated, beta, atol=1e-12, rtol=1e-12):
                beta = updated
                break
            beta = updated
        residual = (y - x @ beta) ** 2
        score = float(np.sort(residual)[:h].sum())
        valid += 1
        if best is None or score < best[0]:
            best = (score, beta.copy())
    if best is None:
        raise RankDeficientError('No full-rank LTS subset')
    return best[1], {'solver': 'exact_h_subsets' if exact else 'multistart_csteps',
                     'global_optimum_certified': exact and rank_deficient_subsets == 0, 'h': int(h),
                     'objective': best[0], 'valid_starts': valid, 'seed': seed,
                     'rank_deficient_subsets_skipped': rank_deficient_subsets,
                     'reweighted': False}
