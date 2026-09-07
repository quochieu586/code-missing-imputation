"""Unit tests for mask handling."""

import numpy as np
import pytest

from missing_imputation.data.masks import (
    MissingnessPattern,
    extract_patterns,
    apply_mask_blending,
)


class TestMissingnessPattern:
    def test_n_missing(self):
        p = MissingnessPattern(pattern=(True, False, True))
        assert p.n_missing == 2
        assert p.n_observed == 1

    def test_equality(self):
        p1 = MissingnessPattern(pattern=(True, False))
        p2 = MissingnessPattern(pattern=(True, False))
        assert p1 == p2

    def test_hashable(self):
        p = MissingnessPattern(pattern=(True, False, True))
        d = {p: "test"}
        assert d[p] == "test"


class TestExtractPatterns:
    def test_basic_extraction(self):
        mask = np.array([
            [True, False, True],
            [True, False, True],
            [False, True, False],
        ])
        patterns = extract_patterns(mask)
        assert len(patterns) == 2

    def test_pattern_counts(self):
        mask = np.array([
            [True, False, True],
            [True, False, True],
            [True, False, True],
        ])
        patterns = extract_patterns(mask)
        assert len(patterns) == 1
        assert patterns[0].n_rows == 3


class TestMaskBlending:
    def test_observed_cells_never_modified(self):
        imputed = np.array([[0.5, 0.3, 0.2]])
        obs_mask = np.array([[True, True, False]])
        original = np.array([[0.4, 0.6, 0.0]])
        result = apply_mask_blending(imputed, obs_mask, original)
        assert result[0, 0] == 0.4
        assert result[0, 1] == 0.6

    def test_missing_cells_take_imputed(self):
        imputed = np.array([[0.5, 0.3, 0.2]])
        obs_mask = np.array([[True, True, False]])
        original = np.array([[0.4, 0.6, 0.0]])
        result = apply_mask_blending(imputed, obs_mask, original)
        assert result[0, 2] == 0.2