"""Core compositional data transforms: clr, ilr, Aitchison distance.

All functions operate on numpy arrays. Compositions must be strictly positive.
References:
  [H] Hron, Templ, Filzmoser (2009)
  [S] Seki, Zhang, Imoto (2025)
"""
import numpy as np
from numpy.typing import NDArray


def clr(X: NDArray, axis: int = -1) -> NDArray:
    """Centered log-ratio transform.

    clr(x)_k = ln(x_k / g(x)) where g(x) = geometric mean of x.

    Properties (must hold):
      - clr(c*x) == clr(x)  for any c > 0  (scale invariance)
      - sum_k clr(x)_k == 0                (zero-sum)

    Parameters
    ----------
    X : array, shape (..., K) or as specified by axis
        Strictly positive compositions.
    axis : int
        Axis along which to compute clr. Default -1.

    Returns
    -------
    Z : array, same shape as X
        clr-transformed values. Sum along `axis` equals 0.

    Raises
    ------
    ValueError
        If any value <= 0.
    """
    X = np.asarray(X, dtype=np.float64)
    if not np.all(np.isfinite(X)) or np.any(X <= 0):
        raise ValueError("clr requires strictly positive values. Got values <= 0.")
    log_X = np.log(X)
    geom_mean_log = np.mean(log_X, axis=axis, keepdims=True)
    return log_X - geom_mean_log


def clr_inverse(Z: NDArray, axis: int = -1) -> NDArray:
    """Inverse clr = closure(exp(z)) = softmax(z).

    Properties:
      - softmax(z + c*1) == softmax(z)  (shift invariance)
      - Always returns compositions summing to 1 along axis.

    Parameters
    ----------
    Z : array
        Values in clr space.
    axis : int
        Axis along which to compute inverse.

    Returns
    -------
    X : array, same shape as Z
        Compositions (proportions summing to 1 along axis).
    """
    Z = np.asarray(Z, dtype=np.float64)
    exp_Z = np.exp(Z - np.max(Z, axis=axis, keepdims=True))  # numerically stable
    return exp_Z / np.sum(exp_Z, axis=axis, keepdims=True)


def clr_subcomposition(x: NDArray, idx: NDArray) -> NDArray:
    """clr on a subcomposition defined by index set.

    Used in Stage A when rows have NaN: geometric mean computed ONLY
    over the observed indices `idx`, NOT over all K.

    Parameters
    ----------
    x : array, shape (K,) or (n, K)
        Row(s) with possible NaN outside idx.
    idx : array of int
        Indices of observed (non-NaN) components.

    Returns
    -------
    z : array, shape (len(idx),) or (n, len(idx))
        clr values of the subcomposition.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        sub = x[idx]
    else:
        sub = x[:, idx]
    return clr(sub)


def aitchison_distance(x: NDArray, y: NDArray, idx: NDArray = None) -> float:
    """Aitchison distance = ||clr(x) - clr(y)||_2.

    Equivalent to [H] formula (2) but O(K) instead of O(K^2).

    Properties:
      - d_A(c*x, y) == d_A(x, y)  for any c > 0  (scale invariance)
      - d_A(x, y) == d_E(ilr(x), ilr(y))          (isometry)

    Parameters
    ----------
    x, y : array, shape (K,)
        Strictly positive compositions.
    idx : array of int, optional
        If provided, compute distance on subcomposition x[idx], y[idx].

    Returns
    -------
    d : float
        Aitchison distance.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if idx is not None:
        z_x = clr_subcomposition(x, idx)
        z_y = clr_subcomposition(y, idx)
    else:
        z_x = clr(x)
        z_y = clr(y)
    return float(np.linalg.norm(z_x - z_y))


def aitchison_distance_formula(x: NDArray, y: NDArray) -> float:
    """Aitchison distance via original formula (2) from [H].

    d_A(x,y) = sqrt( (1/D) * sum_{i<j} (ln(x_i/x_j) - ln(y_i/y_j))^2 )

    O(K^2). Used only for testing equivalence with the clr-based version.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    D = len(x)
    total = 0.0
    for i in range(D - 1):
        for j in range(i + 1, D):
            diff = np.log(x[i] / x[j]) - np.log(y[i] / y[j])
            total += diff ** 2
    return np.sqrt(total / D)


def ilr(X: NDArray) -> NDArray:
    """Isometric log-ratio transform using sequential binary partition.

    [H] formula (8):
    z_j = sqrt((D-j)/(D-j+1)) * ln( (prod_{l=j+1}^D x_l)^{1/(D-j)} / x_j )

    Used ONLY for baseline #5 (kNN + iterative LTS). Main pipeline uses clr.

    Parameters
    ----------
    X : array, shape (..., D), strictly positive

    Returns
    -------
    Z : array, shape (..., D-1)
    """
    X = np.asarray(X, dtype=np.float64)
    if np.any(X <= 0):
        raise ValueError("ilr requires strictly positive values.")
    D = X.shape[-1]
    log_X = np.log(X)
    Z = np.empty(X.shape[:-1] + (D - 1,), dtype=np.float64)
    for j in range(D - 1):
        geo_mean_log = np.mean(log_X[..., j + 1:], axis=-1)
        Z[..., j] = np.sqrt((D - j - 1) / (D - j)) * (geo_mean_log - log_X[..., j])
    return Z


def ilr_inverse(Z: NDArray) -> NDArray:
    """Inverse ilr using [H] formulas (9)-(11).

    Returns ONE REPRESENTATIVE of the equivalence class (not the original vector).
    Does NOT normalize to constant sum.

    Parameters
    ----------
    Z : array, shape (..., D-1)

    Returns
    -------
    X : array, shape (..., D)
    """
    Z = np.asarray(Z, dtype=np.float64)
    D = Z.shape[-1] + 1
    X = np.empty(Z.shape[:-1] + (D,), dtype=np.float64)

    # Formula (9): x_1 = exp(-sqrt((D-1)/D) * z_1)
    X[..., 0] = np.exp(-np.sqrt((D - 1) / D) * Z[..., 0])

    # Formula (10): x_j for j=2..D-1 (0-indexed: j=1..D-2)
    for j in range(1, D - 1):
        cumsum = np.zeros_like(Z[..., 0])
        for l in range(j):
            cumsum = cumsum + Z[..., l] / np.sqrt((D - l) * (D - l - 1))
        X[..., j] = np.exp(cumsum - np.sqrt((D - j - 1) / (D - j)) * Z[..., j])

    # Formula (11): x_D (0-indexed: D-1)
    cumsum = np.zeros_like(Z[..., 0])
    for l in range(D - 1):
        cumsum = cumsum + Z[..., l] / np.sqrt((D - l) * (D - l - 1))
    X[..., D - 1] = np.exp(cumsum)

    return X
