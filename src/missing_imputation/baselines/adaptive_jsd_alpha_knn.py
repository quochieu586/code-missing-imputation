"""Adaptive JSD-alpha-kNN imputation (Variant 3).

Tunes (alpha, k) per missingness pattern when sufficient data exists.
Falls back to global best (alpha, k) for sparse patterns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frechet import frechet_mean
from .jsd import _jsd_single, partial_jsd
from ..data.masks import MissingnessPattern


@dataclass
class AdaptiveJSDAlphaKNNResult:
    """Result of Adaptive JSD-alpha-kNN imputation."""

    imputed_proportions: np.ndarray
    n_neighbors_used: np.ndarray
    params_per_pattern: dict[tuple[bool, ...], tuple[float, int]]
    fallback_patterns: list[tuple[bool, ...]]
    method: str = "adaptive_jsd_alpha_knn"


class AdaptiveJSDAlphaKNN:
    """Adaptive JSD-alpha-kNN: per-pattern tuned (alpha, k).

    For each missingness pattern with sufficient support (>= min_pattern_support
    complete rows), tunes (alpha, k) independently via CV.
    For sparse patterns, falls back to global best (alpha, k).
    """

    def __init__(
        self,
        k_grid: list[int] | None = None,
        alpha_grid: list[float] | None = None,
        min_pattern_support: int = 30,
        global_k: int = 5,
        global_alpha: float = 1.0,
    ):
        self.k_grid = k_grid or [3, 5, 7, 10, 15]
        self.alpha_grid = alpha_grid or [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        self.min_pattern_support = min_pattern_support
        self.global_k = global_k
        self.global_alpha = global_alpha
        self.pattern_params: dict[tuple[bool, ...], tuple[float, int]] = {}

    def fit(self, complete_proportions: np.ndarray) -> "AdaptiveJSDAlphaKNN":
        """Store the neighbor pool.

        Args:
            complete_proportions: (n_complete, d) proportions of complete rows.
        """
        self.pool = np.asarray(complete_proportions, dtype=np.float64)
        self.n_pool = self.pool.shape[0]
        self.d = self.pool.shape[1]
        return self

    def tune_patterns(
        self,
        complete_proportions: np.ndarray,
        patterns: list[MissingnessPattern],
        n_folds: int = 5,
        seed: int = 42,
    ) -> dict[tuple[bool, ...], tuple[float, int]]:
        """Tune (alpha, k) per pattern using masked CV on complete rows.

        Args:
            complete_proportions: Complete rows to use for CV.
            patterns: Missingness patterns to tune for.
            n_folds: Number of CV folds.
            seed: Random seed.

        Returns:
            Dict mapping pattern -> (best_alpha, best_k).
        """
        rng = np.random.default_rng(seed)
        self.pattern_params = {}

        for pattern in patterns:
            if pattern.n_rows >= self.min_pattern_support:
                best_alpha, best_k = self._tune_single_pattern(
                    complete_proportions, pattern.pattern, n_folds, rng
                )
                self.pattern_params[pattern.pattern] = (best_alpha, best_k)
            else:
                self.pattern_params[pattern.pattern] = (self.global_alpha, self.global_k)

        return self.pattern_params

    def _tune_single_pattern(
        self,
        complete_proportions: np.ndarray,
        pattern: tuple[bool, ...],
        n_folds: int,
        rng: np.random.Generator,
    ) -> tuple[float, int]:
        """Tune (alpha, k) for a single pattern via masked CV."""
        n = complete_proportions.shape[0]
        obs_mask = np.array([not m for m in pattern])
        obs_idx = np.where(obs_mask)[0]

        if len(obs_idx) == 0:
            return self.global_alpha, self.global_k

        best_score = np.inf
        best_alpha = self.global_alpha
        best_k = self.global_k

        indices = rng.permutation(n)
        fold_size = n // n_folds

        for alpha in self.alpha_grid:
            for k in self.k_grid:
                scores = []
                for fold in range(n_folds):
                    val_start = fold * fold_size
                    val_end = val_start + fold_size if fold < n_folds - 1 else n
                    val_idx = indices[val_start:val_end]
                    train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

                    if len(train_idx) < k:
                        continue

                    train_pool = complete_proportions[train_idx]
                    val_data = complete_proportions[val_idx]

                    jsd_errors = []
                    for vi in range(len(val_idx)):
                        masked_obs = val_data[vi, obs_idx]
                        masked_sum = masked_obs.sum()
                        if masked_sum <= 0:
                            continue

                        masked_comp = np.zeros(self.d)
                        masked_comp[obs_idx] = masked_obs / masked_sum

                        # Use vectorized partial JSD
                        distances = partial_jsd(masked_obs, train_pool[:, obs_idx])

                        k_actual = min(k, len(train_idx))
                        neighbor_idx = np.argsort(distances)[:k_actual]
                        neighbors = train_pool[neighbor_idx]
                        imputed = frechet_mean(neighbors, alpha=alpha)

                        jsd_errors.append(_jsd_single(val_data[vi], imputed))

                    if jsd_errors:
                        scores.append(np.mean(jsd_errors))

                if scores:
                    mean_score = np.mean(scores)
                    if mean_score < best_score:
                        best_score = mean_score
                        best_alpha = alpha
                        best_k = k

        return best_alpha, best_k

    def impute(
        self,
        incomplete_proportions: np.ndarray,
        observed_mask: np.ndarray,
    ) -> AdaptiveJSDAlphaKNNResult:
        """Impute missing components with per-pattern tuned parameters.

        Args:
            incomplete_proportions: (n_incomplete, d).
            observed_mask: (n_incomplete, d) boolean.

        Returns:
            AdaptiveJSDAlphaKNNResult.
        """
        n_incomplete = incomplete_proportions.shape[0]
        result = np.zeros((n_incomplete, self.d), dtype=np.float64)
        n_neighbors_used = np.zeros(n_incomplete, dtype=np.int64)
        fallback_patterns = []

        for i in range(n_incomplete):
            pattern = tuple(~observed_mask[i])
            alpha, k = self.pattern_params.get(
                pattern, (self.global_alpha, self.global_k)
            )

            if pattern not in self.pattern_params:
                fallback_patterns.append(pattern)

            obs_idx = np.where(observed_mask[i])[0]

            if len(obs_idx) == 0:
                result[i] = np.ones(self.d) / self.d
                n_neighbors_used[i] = 0
                continue

            # Use vectorized partial JSD
            distances = partial_jsd(
                incomplete_proportions[i, obs_idx], self.pool[:, obs_idx]
            )

            k_actual = min(k, self.n_pool)
            neighbor_idx = np.argsort(distances)[:k_actual]
            n_neighbors_used[i] = k_actual

            neighbor_compositions = self.pool[neighbor_idx]
            mean_comp = frechet_mean(neighbor_compositions, alpha=alpha)

            result[i, observed_mask[i]] = incomplete_proportions[i, observed_mask[i]]
            missing_idx = np.where(~observed_mask[i])[0]
            result[i, missing_idx] = mean_comp[missing_idx]

            total = result[i].sum()
            if total > 0:
                result[i] /= total
            else:
                result[i] = np.ones(self.d) / self.d

        return AdaptiveJSDAlphaKNNResult(
            imputed_proportions=result,
            n_neighbors_used=n_neighbors_used,
            params_per_pattern=self.pattern_params,
            fallback_patterns=fallback_patterns,
        )