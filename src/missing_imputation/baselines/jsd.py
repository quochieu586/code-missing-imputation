"""Jensen-Shannon Divergence for compositional data.

Implements JSD that handles zeros directly (0 * log(0) = 0 convention),
following Tsagris et al.
"""

from __future__ import annotations

import numpy as np


def jsd_pairwise(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute pairwise JSD between rows of x and rows of y.

    JSD(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m) where m = (p+q)/2.
    Handles zeros: 0 * log(0) = 0.

    Args:
        x: Array (n, d) of compositions.
        y: Array (m, d) of compositions.

    Returns:
        JSD distance matrix (n, m). Values in [0, log(2)].
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    n = x.shape[0]
    m = y.shape[0]
    result = np.zeros((n, m), dtype=np.float64)

    for i in range(n):
        for j in range(m):
            result[i, j] = _jsd_single(x[i], y[j])

    return result


def _jsd_single(p: np.ndarray, q: np.ndarray) -> float:
    """Compute JSD between two composition vectors."""
    m = 0.5 * (p + q)
    return 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute KL(p || q) with 0 * log(0) = 0 convention."""
    mask = (p > 0) & (q > 0)
    if not mask.any():
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def jsd_distance_matrix(compositions: np.ndarray) -> np.ndarray:
    """Compute full pairwise JSD distance matrix.

    Args:
        compositions: Array (n, d) of compositions.

    Returns:
        Symmetric distance matrix (n, n).
    """
    n = compositions.shape[0]
    result = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            d = _jsd_single(compositions[i], compositions[j])
            result[i, j] = d
            result[j, i] = d

    return result


def jsd_distance_to_reference(
    compositions: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """Compute JSD from each row to a single reference composition.

    Args:
        compositions: Array (n, d).
        reference: Single composition vector (d,).

    Returns:
        Distance vector (n,).
    """
    n = compositions.shape[0]
    result = np.zeros(n, dtype=np.float64)
    for i in range(n):
        result[i] = _jsd_single(compositions[i], reference)
    return result


def jsd_one_vs_many(p: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Vectorized JSD between one composition and many rows.

    Args:
        p: Composition vector (d,).
        Q: Composition matrix (m, d).

    Returns:
        JSD vector (m,).
    """
    p = np.asarray(p, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    M = 0.5 * (p[np.newaxis, :] + Q)

    mask_p = p > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_p = np.sum(
            np.where(mask_p[np.newaxis, :], p[np.newaxis, :] * np.log(p[np.newaxis, :] / M), 0.0),
            axis=1,
        )

    mask_q = Q > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_q = np.sum(np.where(mask_q, Q * np.log(Q / M), 0.0), axis=1)

    return 0.5 * (kl_p + kl_q)


def partial_jsd(x_sub: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Vectorized JSD on observed sub-compositions, renormalized per row.

    Computes JSD between a single query sub-composition and multiple pool rows,
    using only the observed components. Both query and pool rows are renormalized
    to sum to 1 over the observed components before computing JSD.

    Args:
        x_sub: Observed components of query row (k,), unnormalized.
        Y: Observed components of pool rows (m, k), unnormalized.

    Returns:
        JSD vector (m,). Rows with zero sub-sum get log(2).
    """
    x_sum = x_sub.sum()
    m = Y.shape[0]
    if x_sum <= 0:
        return np.full(m, np.log(2), dtype=np.float64)

    xn = x_sub / x_sum
    s = Y.sum(axis=1)
    valid = s > 0
    out = np.full(m, np.log(2), dtype=np.float64)

    if not valid.any():
        return out

    Yn = Y[valid] / s[valid, np.newaxis]
    M = 0.5 * (xn[np.newaxis, :] + Yn)

    mask_x = xn > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_x = np.sum(
            np.where(mask_x[np.newaxis, :], xn[np.newaxis, :] * np.log(xn[np.newaxis, :] / M), 0.0),
            axis=1,
        )
    mask_y = Yn > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_y = np.sum(np.where(mask_y, Yn * np.log(Yn / M), 0.0), axis=1)

    out[valid] = 0.5 * (kl_x + kl_y)
    return out


# Alias for backward compatibility
partial_jsd_one_vs_many = partial_jsd
