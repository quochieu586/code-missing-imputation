"""JSD-alpha-kNN imputation (Variant 2).

Uses JSD distance to find k nearest neighbors and imputes missing
components using the Fréchet mean with tunable alpha.
When alpha=1, equivalent to JSD-kNN.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frechet import frechet_mean
from .jsd import _jsd_single, partial_jsd


@dataclass
class JSDAlphaKNNResult:
    """Result of JSD-alpha-kNN imputation."""

    imputed_proportions: np.ndarray
    n_neighbors_used: np.ndarray
    k: int
    alpha: float
    method: str = "jsd_alpha_knn"


class JSDAlphaKNN:
    """JSD-alpha-kNN imputation for compositional data.

    Same as JSD-kNN but uses Fréchet mean with parameter alpha
    instead of arithmetic mean.
    """

    def __init__(self, k: int = 5, alpha: float = 1.0):
        self.k = k
        self.alpha = alpha

    def fit(self, complete_proportions: np.ndarray) -> "JSDAlphaKNN":
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
    ) -> JSDAlphaKNNResult:
        """Impute missing components using Fréchet mean of neighbors.

        Args:
            incomplete_proportions: (n_incomplete, d) with observed values.
            observed_mask: (n_incomplete, d) boolean, True = observed.

        Returns:
            JSDAlphaKNNResult with imputed proportions.
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
            neighbor_idx = np.argsort(distances)[:k_actual]
            n_neighbors_used[i] = k_actual

            neighbor_compositions = self.pool[neighbor_idx]
            mean_comp = frechet_mean(neighbor_compositions, alpha=self.alpha)

            result[i, observed_mask[i]] = incomplete_proportions[i, observed_mask[i]]
            missing_idx = np.where(~observed_mask[i])[0]
            result[i, missing_idx] = mean_comp[missing_idx]

            total = result[i].sum()
            if total > 0:
                result[i] /= total
            else:
                result[i] = np.ones(self.d) / self.d

        return JSDAlphaKNNResult(
            imputed_proportions=result,
            n_neighbors_used=n_neighbors_used,
            k=self.k,
            alpha=self.alpha,
        )
