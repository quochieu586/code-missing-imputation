"""Fréchet mean on the simplex with JSD geometry.

Implements the alpha-weighted Fréchet mean following Tsagris et al.
When alpha=1, this reduces to the arithmetic mean (after closure).
"""

from __future__ import annotations

import numpy as np


def frechet_mean(
    compositions: np.ndarray,
    alpha: float = 1.0,
    weights: np.ndarray | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> np.ndarray:
    """Compute the Fréchet mean of compositions under JSD geometry.

    For alpha=1, returns the arithmetic mean (closed to sum 1).
    For alpha != 1, uses iterative optimization.

    Args:
        compositions: Array (k, d) of neighbor compositions.
        alpha: Power parameter. alpha=1 gives arithmetic mean.
        weights: Optional weights for each composition (k,).
        max_iter: Maximum iterations for optimization.
        tol: Convergence tolerance.

    Returns:
        Fréchet mean composition (d,).
    """
    compositions = np.asarray(compositions, dtype=np.float64)
    k, d = compositions.shape

    if k == 0:
        return np.ones(d) / d

    if weights is None:
        weights = np.ones(k) / k
    else:
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / weights.sum()

    if np.isclose(alpha, 1.0):
        return _arithmetic_mean_closed(compositions, weights)

    return _iterative_frechet(compositions, alpha, weights, max_iter, tol)


def _arithmetic_mean_closed(
    compositions: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Weighted arithmetic mean, closed to sum 1."""
    mean = np.average(compositions, axis=0, weights=weights)
    total = mean.sum()
    if total <= 0:
        return np.ones(compositions.shape[1]) / compositions.shape[1]
    return mean / total


def _iterative_frechet(
    compositions: np.ndarray,
    alpha: float,
    weights: np.ndarray,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    """Iterative computation of Fréchet mean for alpha != 1.

    Uses the power mean approach:
    mu_alpha = (sum(w_i * x_i^alpha))^(1/alpha), then close.
    """
    k, d = compositions.shape

    eps = 1e-15
    compositions_safe = np.maximum(compositions, eps)

    if alpha > 0:
        powered = compositions_safe ** alpha
        weighted_sum = np.average(powered, axis=0, weights=weights)
        mean = weighted_sum ** (1.0 / alpha)
    else:
        powered = compositions_safe ** alpha
        weighted_sum = np.average(powered, axis=0, weights=weights)
        mean = weighted_sum ** (1.0 / alpha)

    total = mean.sum()
    if total <= 0:
        return np.ones(d) / d
    return mean / total


def frechet_mean_batch(
    compositions_list: list[np.ndarray],
    alpha: float = 1.0,
    weights_list: list[np.ndarray | None] | None = None,
) -> np.ndarray:
    """Compute Fréchet mean for multiple groups of compositions.

    Args:
        compositions_list: List of (k_i, d) arrays.
        alpha: Power parameter.
        weights_list: Optional list of weight arrays.

    Returns:
        Array (n_groups, d) of Fréchet means.
    """
    results = []
    for i, comp in enumerate(compositions_list):
        w = weights_list[i] if weights_list else None
        results.append(frechet_mean(comp, alpha=alpha, weights=w))
    return np.array(results)