"""JSD-kNN imputation (Variant 1).

Uses JSD distance to find k nearest neighbors and imputes missing
components using the arithmetic mean of neighbors.
Following Tsagris et al. 6-step algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .jsd import _jsd_single, partial_jsd


@dataclass
class JSDKNNResult:
    """Result of JSD-kNN imputation."""

    imputed_proportions: np.ndarray
    n_neighbors_used: np.ndarray
    k: int
    method: str = "jsd_knn"


class JSDKNN:
    """JSD-kNN imputation for compositional data.

    Algorithm (6 steps from Tsagris et al.):
    1. Identify observed components for each incomplete row.
    2. Compute JSD between incomplete row and all complete rows
       using only observed components.
    3. Select k nearest neighbors.
    4. Compute arithmetic mean of neighbors'"'"' full compositions.
    5. Fill missing components from the mean.
    6. Close to ensure sum = 1.
    """

    def __init__(self, k: int = 5):
        self.k = k

    def fit(self, complete_proportions: np.ndarray) -> "JSDKNN":
        """Store the neighbor pool (complete rows).

        Args:
            complete_proportions: (n_complete, d) proportions of complete rows.
        """
        self.pool = np.asarray(complete_proportions, dtype=np.float64)
        self.n_pool = self.pool.shape[0]
        self.d = self.pool.shape[1]
        return self

    def impute(
        self,
        incomplete_proportions: np.ndarray,
        observed_mask: np.ndarray,
    ) -> JSDKNNResult:
        """Impute missing components for incomplete rows.

        Args:
            incomplete_proportions: (n_incomplete, d) with observed values filled.
            observed_mask: (n_incomplete, d) boolean, True = observed.

        Returns:
            JSDKNNResult with imputed proportions.
        """
        n_incomplete = incomplete_proportions.shape[0]
        result = np.zeros((n_incomplete, self.d), dtype=np.float64)
        n_neighbors_used = np.zeros(n_incomplete, dtype=np.int64)

        for i in range(n_incomplete):
            obs_idx = np.where(observed_mask[i])[0]

            if len(obs_idx) == 0:
                result[i] = np.ones(self.d) / self.d
                n_neighbors_used[i] = 0
                continue

            # Use vectorized partial JSD
            distances = partial_jsd(
                incomplete_proportions[i, obs_idx], self.pool[:, obs_idx]
            )

            k_actual = min(self.k, self.n_pool)
            neighbor_idx = np.argpartition(distances, k_actual - 1)[:k_actual]
            n_neighbors_used[i] = k_actual

            neighbor_compositions = self.pool[neighbor_idx]
            mean_comp = neighbor_compositions.mean(axis=0)

            result[i, observed_mask[i]] = incomplete_proportions[i, observed_mask[i]]
            missing_idx = np.where(~observed_mask[i])[0]
            result[i, missing_idx] = mean_comp[missing_idx]

            total = result[i].sum()
            if total > 0:
                result[i] /= total
            else:
                result[i] = np.ones(self.d) / self.d

        return JSDKNNResult(
            imputed_proportions=result,
            n_neighbors_used=n_neighbors_used,
            k=self.k,
        )
