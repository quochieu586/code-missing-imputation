"""Unit tests for closure and count/proportion conversion."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from missing_imputation.data.closure import (
    counts_to_proportions,
    proportions_to_counts_largest_remainder,
    enforce_closure,
    compute_other,
    validate_closure,
)


class TestCountsToProportions:
    def test_basic_conversion(self):
        counts = np.array([[2, 3, 5], [1, 1, 8]])
        total = np.array([10, 10])
        props = counts_to_proportions(counts, total)
        np.testing.assert_allclose(props.sum(axis=1), [1.0, 1.0], atol=1e-10)

    def test_zero_total_handled(self):
        counts = np.array([[0, 0, 0]])
        total = np.array([0])
        props = counts_to_proportions(counts, total)
        assert np.all(np.isfinite(props))


class TestLargestRemainder:
    def test_sums_to_total(self):
        props = np.array([[0.33, 0.33, 0.34], [0.5, 0.3, 0.2]])
        total = np.array([100, 50])
        counts = proportions_to_counts_largest_remainder(props, total)
        np.testing.assert_array_equal(counts.sum(axis=1), total)

    def test_observed_cells_locked(self):
        props = np.array([[0.33, 0.33, 0.34]])
        total = np.array([100])
        obs_mask = np.array([[True, False, False]])
        obs_counts = np.array([[20, 0, 0]])
        counts = proportions_to_counts_largest_remainder(
            props, total, obs_mask, obs_counts
        )
        assert counts[0, 0] == 20
        assert counts.sum() == 100

    def test_non_negative_output(self):
        props = np.array([[0.1, 0.2, 0.7]])
        total = np.array([3])
        counts = proportions_to_counts_largest_remainder(props, total)
        assert np.all(counts >= 0)

    @given(
        total=st.integers(min_value=1, max_value=10000),
        n_features=st.integers(min_value=2, max_value=20),
    )
    @settings(max_examples=50)
    def test_property_closure_holds(self, total, n_features):
        rng = np.random.default_rng(42)
        props = rng.dirichlet(np.ones(n_features), size=1)
        total_arr = np.array([total])
        counts = proportions_to_counts_largest_remainder(props, total_arr)
        assert counts.sum() == total
        assert np.all(counts >= 0)


class TestEnforceClosure:
    def test_adjusts_missing_only(self):
        counts = np.array([[10, 20, 30]])
        total = np.array([100])
        obs_mask = np.array([[True, True, False]])
        obs_counts = np.array([[10, 20, 0]])
        result = enforce_closure(counts, total, obs_mask, obs_counts)
        assert result[0, 0] == 10
        assert result[0, 1] == 20
        assert result.sum() == 100


class TestComputeOther:
    def test_other_computation(self):
        counts = np.array([[2, 3, 5], [1, 1, 8]])
        total = np.array([15, 12])
        other = compute_other(counts, total)
        np.testing.assert_array_equal(other, [5, 2])

    def test_validate_closure_passes(self):
        counts = np.array([[2, 3, 5]])
        total = np.array([15])
        is_valid, n_viol = validate_closure(counts, total)
        assert is_valid
        assert n_viol == 0