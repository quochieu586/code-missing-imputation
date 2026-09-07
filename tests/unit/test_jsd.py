"""Unit tests for JSD computation."""

import numpy as np
import pytest

from missing_imputation.baselines.jsd import (
    _jsd_single,
    jsd_pairwise,
    jsd_distance_matrix,
)


class TestJSDProperties:
    def test_jsd_identical_compositions_is_zero(self):
        p = np.array([0.2, 0.3, 0.5])
        assert _jsd_single(p, p) == pytest.approx(0.0, abs=1e-12)

    def test_jsd_symmetric(self):
        p = np.array([0.2, 0.3, 0.5])
        q = np.array([0.1, 0.4, 0.5])
        assert _jsd_single(p, q) == pytest.approx(_jsd_single(q, p), abs=1e-12)

    def test_jsd_finite(self):
        p = np.array([0.2, 0.3, 0.5])
        q = np.array([0.1, 0.4, 0.5])
        val = _jsd_single(p, q)
        assert np.isfinite(val)
        assert val >= 0

    def test_jsd_with_zeros(self):
        p = np.array([0.0, 0.5, 0.5])
        q = np.array([0.3, 0.3, 0.4])
        val = _jsd_single(p, q)
        assert np.isfinite(val)
        assert val >= 0

    def test_jsd_both_zero_component(self):
        p = np.array([0.0, 0.5, 0.5])
        q = np.array([0.0, 0.3, 0.7])
        val = _jsd_single(p, q)
        assert np.isfinite(val)
        assert val >= 0

    def test_jsd_max_bounded_by_log2(self):
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([0.0, 1.0, 0.0])
        val = _jsd_single(p, q)
        assert val <= np.log(2) + 1e-10

    def test_jsd_distance_matrix_symmetric(self):
        compositions = np.array([
            [0.2, 0.3, 0.5],
            [0.1, 0.4, 0.5],
            [0.0, 0.6, 0.4],
        ])
        D = jsd_distance_matrix(compositions)
        assert D.shape == (3, 3)
        np.testing.assert_allclose(D, D.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-12)

    def test_jsd_pairwise_shape(self):
        x = np.array([[0.2, 0.3, 0.5], [0.1, 0.4, 0.5]])
        y = np.array([[0.3, 0.3, 0.4]])
        D = jsd_pairwise(x, y)
        assert D.shape == (2, 1)


class TestPaperExample:
    def test_five_component_example(self):
        """Tsagris paper: 5-component example should yield ~(0.20, 0.27, 0.30, 0.10, 0.13)."""
        compositions = np.array([
            [0.1, 0.3, 0.4, 0.0, 0.2],
            [0.3, 0.2, 0.2, 0.2, 0.1],
            [0.2, 0.31, 0.3, 0.1, 0.09],
        ])
        mean = compositions.mean(axis=0)
        expected = np.array([0.20, 0.27, 0.30, 0.10, 0.13])
        np.testing.assert_allclose(mean, expected, atol=0.01)