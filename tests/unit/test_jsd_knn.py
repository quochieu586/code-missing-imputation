"""Unit tests for JSD-kNN variants."""

import numpy as np
import pytest

from missing_imputation.baselines.jsd_knn import JSDKNN
from missing_imputation.baselines.jsd_alpha_knn import JSDAlphaKNN
from missing_imputation.baselines.adaptive_jsd_alpha_knn import AdaptiveJSDAlphaKNN
from missing_imputation.data.masks import MissingnessPattern


class TestJSDKNN:
    def setup_method(self):
        self.complete = np.array([
            [0.2, 0.3, 0.5],
            [0.1, 0.4, 0.5],
            [0.3, 0.2, 0.5],
            [0.15, 0.35, 0.5],
            [0.25, 0.25, 0.5],
        ])

    def test_impute_fills_missing(self):
        model = JSDKNN(k=3)
        model.fit(self.complete)
        incomplete = np.array([[0.2, 0.0, 0.0]])
        obs_mask = np.array([[True, False, False]])
        result = model.impute(incomplete, obs_mask)
        assert not np.any(np.isnan(result.imputed_proportions))
        assert result.imputed_proportions.shape == (1, 3)

    def test_observed_values_preserved(self):
        model = JSDKNN(k=3)
        model.fit(self.complete)
        incomplete = np.array([[0.2, 0.3, 0.0]])
        obs_mask = np.array([[True, True, False]])
        result = model.impute(incomplete, obs_mask)
        assert result.imputed_proportions[0, 0] == pytest.approx(0.2, abs=1e-6)
        assert result.imputed_proportions[0, 1] == pytest.approx(0.3, abs=1e-6)

    def test_output_sums_to_one(self):
        model = JSDKNN(k=3)
        model.fit(self.complete)
        incomplete = np.array([[0.2, 0.0, 0.0]])
        obs_mask = np.array([[True, False, False]])
        result = model.impute(incomplete, obs_mask)
        assert result.imputed_proportions.sum() == pytest.approx(1.0, abs=1e-6)

    def test_no_nan_output(self):
        model = JSDKNN(k=2)
        model.fit(self.complete)
        incomplete = np.array([[0.0, 0.0, 0.5]])
        obs_mask = np.array([[False, False, True]])
        result = model.impute(incomplete, obs_mask)
        assert not np.any(np.isnan(result.imputed_proportions))


class TestJSDAlphaKNN:
    def setup_method(self):
        self.complete = np.array([
            [0.2, 0.3, 0.5],
            [0.1, 0.4, 0.5],
            [0.3, 0.2, 0.5],
            [0.15, 0.35, 0.5],
            [0.25, 0.25, 0.5],
        ])

    def test_alpha_1_matches_jsd_knn(self):
        knn = JSDKNN(k=3)
        knn.fit(self.complete)
        alpha_knn = JSDAlphaKNN(k=3, alpha=1.0)
        alpha_knn.fit(self.complete)

        incomplete = np.array([[0.2, 0.0, 0.0]])
        obs_mask = np.array([[True, False, False]])

        r1 = knn.impute(incomplete, obs_mask)
        r2 = alpha_knn.impute(incomplete, obs_mask)
        np.testing.assert_allclose(
            r1.imputed_proportions, r2.imputed_proportions, atol=1e-6
        )

    def test_different_alpha_gives_different_result(self):
        knn_a1 = JSDAlphaKNN(k=3, alpha=1.0)
        knn_a1.fit(self.complete)
        knn_a5 = JSDAlphaKNN(k=3, alpha=5.0)
        knn_a5.fit(self.complete)

        incomplete = np.array([[0.2, 0.0, 0.0]])
        obs_mask = np.array([[True, False, False]])

        r1 = knn_a1.impute(incomplete, obs_mask)
        r5 = knn_a5.impute(incomplete, obs_mask)
        assert not np.allclose(r1.imputed_proportions, r5.imputed_proportions)


class TestAdaptiveJSDAlphaKNN:
    def setup_method(self):
        self.complete = np.array([
            [0.2, 0.3, 0.5],
            [0.1, 0.4, 0.5],
            [0.3, 0.2, 0.5],
            [0.15, 0.35, 0.5],
            [0.25, 0.25, 0.5],
        ])

    def test_fallback_for_sparse_patterns(self):
        model = AdaptiveJSDAlphaKNN(
            k_grid=[3, 5],
            alpha_grid=[1.0, 2.0],
            min_pattern_support=100,
            global_k=3,
            global_alpha=1.0,
        )
        model.fit(self.complete)

        incomplete = np.array([[0.2, 0.0, 0.0]])
        obs_mask = np.array([[True, False, False]])
        result = model.impute(incomplete, obs_mask)
        assert not np.any(np.isnan(result.imputed_proportions))

    def test_pattern_tuning_with_sufficient_support(self):
        model = AdaptiveJSDAlphaKNN(
            k_grid=[3],
            alpha_grid=[1.0],
            min_pattern_support=1,
            global_k=3,
            global_alpha=1.0,
        )
        model.fit(self.complete)

        pattern = MissingnessPattern(
            pattern=(False, True, True),
            n_rows=5,
            row_indices=[0, 1, 2, 3, 4],
        )
        model.tune_patterns(self.complete, [pattern], n_folds=2, seed=42)
        assert (False, True, True) in model.pattern_params