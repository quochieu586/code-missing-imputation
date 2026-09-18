"""Stage A: kNN imputation using Aitchison distance.

Implements [H] Direction 1, strategy 2a:
  - Fill each missing cell sequentially
  - Neighbors must have observed value at cell j AND at all O_i
  - Distance on subcomposition of observed indices (clr-based, O(K))
  - Scaling factor f* (median-based, formula 7) adjusts for scale differences
  - Final value = median of scaled neighbor values (formula 6)

References:
  [H] Hron, Templ, Filzmoser (2009), §3.1
  pipeline_knn_diffusion_handover.md, §3
"""
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class KNNImputeResult:
    """Output of kNN-Aitchison imputation."""
    X_imputed: NDArray       # (n, K) fully imputed, strictly positive
    n_cells_filled: int      # number of cells that were imputed
    n_fallback: int          # cells where insufficient neighbors forced fallback


def _find_neighbors_2a(
    x_i: NDArray,
    O_i: NDArray,
    j: int,
    X_pool: NDArray,
    M_pool: NDArray,
    k: int,
) -> NDArray:
    """Find k nearest neighbors using Aitchison distance on subcomposition O_i.

    Strategy 2a [H]: candidates must have observed values at position j
    AND at all positions in O_i.  The neighbor set changes per cell j.

    Returns indices into X_pool of the k nearest neighbors.
    """
    # Filter: must have observed value at j AND all O_i
    required_idx = np.append(O_i, j)
    valid_mask = M_pool[:, required_idx].all(axis=1)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return np.array([], dtype=int)

    # clr on subcomposition O_i for target row
    x_sub = x_i[O_i]
    if np.any(x_sub <= 0) or np.any(np.isnan(x_sub)):
        return np.array([], dtype=int)

    log_x = np.log(x_sub)
    geo_mean_x = np.mean(log_x)
    clr_x = log_x - geo_mean_x  # (|O_i|,)

    # Vectorized clr + distances for all valid candidates
    candidates = X_pool[valid_indices][:, O_i]  # (m, |O_i|)
    log_c = np.log(candidates)
    geo_mean_c = np.mean(log_c, axis=1, keepdims=True)
    clr_c = log_c - geo_mean_c  # (m, |O_i|)

    distances = np.linalg.norm(clr_c - clr_x, axis=1)  # (m,)

    # Principled tie-break. Rows that are pure pseudo-counts sit at exactly the
    # same Aitchison distance from a target, and np.argpartition runs
    # introselect, which returns an arbitrary subset of tied keys: for a pool of
    # 400 equidistant candidates it picks indices 392..399 rather than 0..7, and
    # which subset it lands on depends on array layout, not on the data. Making
    # distance primary and the pool index the tie-break fixes that choice.
    #
    # This is NOT what T7 tests: rescaling the target row leaves the candidate
    # distance vector unchanged, so argpartition saw identical input and T7 held
    # both before and after this change. The defect it fixes is that the
    # neighbour set was chosen by an implementation detail.
    k_actual = min(k, len(valid_indices))
    order = np.lexsort((valid_indices, distances))
    return valid_indices[order[:k_actual]]


def _scaling_factor_median(x_i: NDArray, x_nb: NDArray, O_i: NDArray) -> float:
    """Robust scaling factor f*_{i,i_l}, [H] formula (7).

    f* = median(x_i[O_i]) / median(x_nb[O_i])

    Adjusts for overall scale difference: neighbors may be 'similar in
    proportions' but at a very different total scale.
    """
    med_t = np.median(x_i[O_i])
    med_n = np.median(x_nb[O_i])
    if med_n == 0:
        return 1.0
    return med_t / med_n


def _scaling_factor_sum(x_i: NDArray, x_nb: NDArray, O_i: NDArray) -> float:
    """Original scaling factor, [H] formula (5).

    f = sum(x_i[O_i]) / sum(x_nb[O_i])
    """
    s_t = np.sum(x_i[O_i])
    s_n = np.sum(x_nb[O_i])
    if s_n == 0:
        return 1.0
    return s_t / s_n


def knn_aitchison_impute(
    X: NDArray,
    M0: NDArray,
    k: int = 8,
    adjust: Literal["median", "sum"] = "median",
    train_idx: Optional[NDArray] = None,
    verbose: bool = False,
) -> KNNImputeResult:
    """kNN imputation using Aitchison distance, strategy 2a.

    For each missing cell x_{ij}:
      A1. Filter candidates: observed at j AND all O_i (train set only)
      A2. Compute Aitchison distance on subcomposition O_i via clr
      A3. Select k nearest, compute scaling factor f* (formula 7 or 5)
      A4. Fill with median of scaled values (formula 6)

    Parameters
    ----------
    X : (n, K) array
        Observed cells > 0, missing cells = NaN.
    M0 : (n, K) bool array
        True = observed, False = missing. FROZEN from preprocessing.
    k : int
        Number of nearest neighbors. Default 8 ([H] simulation value).
    adjust : 'median' or 'sum'
        Scaling method. 'median' = formula (7) [default, robust].
    train_idx : array of int, optional
        Neighbor pool indices (R8: CV leakage prevention). None = all rows.
    verbose : bool
        Print progress every 500 rows.

    Returns
    -------
    KNNImputeResult
    """
    X = X.copy().astype(np.float64)
    M0 = np.asarray(M0, dtype=bool)
    n, K = X.shape

    if train_idx is None:
        train_idx = np.arange(n)

    scale_fn = _scaling_factor_median if adjust == "median" else _scaling_factor_sum

    # Pre-compute column medians for fallback
    col_medians = np.empty(K)
    for j in range(K):
        obs_j = X[train_idx, j][M0[train_idx, j]]
        obs_j_pos = obs_j[obs_j > 0]
        col_medians[j] = np.median(obs_j_pos) if len(obs_j_pos) > 0 else 1.0

    n_filled = 0
    n_fallback = 0

    for i in range(n):
        M_i = np.where(~M0[i])[0]   # missing indices
        O_i = np.where(M0[i])[0]    # observed indices

        if len(M_i) == 0:
            continue

        if verbose and i % 500 == 0:
            print(f"  kNN row {i}/{n}, missing={len(M_i)}")

        # Entirely missing row → column median fallback
        if len(O_i) == 0:
            for j in M_i:
                X[i, j] = col_medians[j]
                n_fallback += 1
            n_filled += len(M_i)
            continue

        # Pool = train rows excluding self
        pool_idx = train_idx[train_idx != i]
        X_pool = X[pool_idx]
        M_pool = M0[pool_idx]

        # Strategy 2a: fill each missing cell independently
        for j in M_i:
            neighbors = _find_neighbors_2a(X[i], O_i, j, X_pool, M_pool, k)

            if len(neighbors) == 0:
                X[i, j] = col_medians[j]
                n_fallback += 1
                n_filled += 1
                continue

            # Scaled values from each neighbor (formula 6)
            scaled = np.empty(len(neighbors))
            for ni, nb in enumerate(neighbors):
                f = scale_fn(X[i], X_pool[nb], O_i)
                scaled[ni] = f * X_pool[nb, j]

            X[i, j] = np.median(scaled)

            # Ensure strictly positive
            if X[i, j] <= 0:
                X[i, j] = col_medians[j]
                n_fallback += 1

            n_filled += 1

    return KNNImputeResult(
        X_imputed=X,
        n_cells_filled=n_filled,
        n_fallback=n_fallback,
    )
