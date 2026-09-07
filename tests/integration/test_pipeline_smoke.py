"""Integration smoke test: small fixture through full baseline pipeline."""

import numpy as np
import pytest

from missing_imputation.baselines.jsd_knn import JSDKNN
from missing_imputation.baselines.jsd_alpha_knn import JSDAlphaKNN
from missing_imputation.data.closure import (
    counts_to_proportions,
    proportions_to_counts_largest_remainder,
    validate_closure,
)


class TestSmallFixturePipeline:
    def setup_method(self):
        rng = np.random.default_rng(42)
        n_complete = 20
        n_features = 5
        self.complete_props = rng.dirichlet(np.ones(n_features) * 2, size=n_complete)
        self.total_seq = rng.integers(10, 100, size=n_complete).astype(np.float64)

    def test_full_pipeline_jsd_knn(self):
        model = JSDKNN(k=5)
        model.fit(self.complete_props)

        incomplete = self.complete_props[:3].copy()
        incomplete[:, 2:] = 0
        obs_mask = np.zeros((3, 5), dtype=bool)
        obs_mask[:, :2] = True

        result = model.impute(incomplete, obs_mask)
        assert result.imputed_proportions.shape == (3, 5)
        assert not np.any(np.isnan(result.imputed_proportions))
        np.testing.assert_allclose(
            result.imputed_proportions.sum(axis=1), np.ones(3), atol=1e-6
        )

    def test_full_pipeline_with_counts(self):
        counts = (self.complete_props * self.total_seq[:, None]).astype(np.int64)
        total_seq = self.total_seq

        complete_mask = np.ones(counts.shape[0], dtype=bool)
        complete_props = counts_to_proportions(counts[complete_mask], total_seq[complete_mask])

        incomplete_idx = [0, 1, 2]
        incomplete_counts = counts[incomplete_idx].copy()
        incomplete_counts[:, 2:] = 0
        obs_mask = np.zeros((3, 5), dtype=bool)
        obs_mask[:, :2] = True

        props = counts_to_proportions(incomplete_counts, total_seq[incomplete_idx])

        model = JSDKNN(k=5)
        model.fit(complete_props)
        result = model.impute(props, obs_mask)

        new_counts = proportions_to_counts_largest_remainder(
            result.imputed_proportions,
            total_seq[incomplete_idx],
            obs_mask,
            incomplete_counts,
        )

        assert np.all(new_counts >= 0)
        np.testing.assert_array_equal(
            new_counts.sum(axis=1), total_seq[incomplete_idx].astype(np.int64)
        )

        is_valid, n_viol = validate_closure(new_counts, total_seq[incomplete_idx])
        assert is_valid