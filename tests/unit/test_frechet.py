"""Unit tests for Fréchet mean."""

import numpy as np
import pytest

from missing_imputation.baselines.frechet import frechet_mean


class TestFrechetMean:
    def test_alpha_1_equals_arithmetic_mean(self):
        compositions = np.array([
            [0.2, 0.3, 0.5],
            [0.4, 0.1, 0.5],
        ])
        result = frechet_mean(compositions, alpha=1.0)
        expected = compositions.mean(axis=0)
        expected = expected / expected.sum()
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_output_sums_to_one(self):
        compositions = np.array([
            [0.2, 0.3, 0.5],
            [0.4, 0.1, 0.5],
            [0.1, 0.6, 0.3],
        ])
        for alpha in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            result = frechet_mean(compositions, alpha=alpha)
            assert result.sum() == pytest.approx(1.0, abs=1e-10)

    def test_non_negative_output(self):
        compositions = np.array([
            [0.0, 0.5, 0.5],
            [0.3, 0.0, 0.7],
        ])
        result = frechet_mean(compositions, alpha=2.0)
        assert np.all(result >= 0)

    def test_single_composition_returns_itself(self):
        comp = np.array([[0.2, 0.3, 0.5]])
        result = frechet_mean(comp, alpha=1.0)
        np.testing.assert_allclose(result, comp[0], atol=1e-10)

    def test_empty_returns_uniform(self):
        compositions = np.zeros((0, 3))
        result = frechet_mean(compositions, alpha=1.0)
        np.testing.assert_allclose(result, [1 / 3, 1 / 3, 1 / 3], atol=1e-10)

    def test_weighted_mean(self):
        compositions = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        weights = np.array([0.75, 0.25])
        result = frechet_mean(compositions, alpha=1.0, weights=weights)
        assert result[0] > result[1]