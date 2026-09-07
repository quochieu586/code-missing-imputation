"""Unit tests for baseline selection logic."""

import numpy as np
import pytest

from missing_imputation.pipeline.select_baseline import select_baseline


class TestSelectBaseline:
    def test_selects_lowest_mse(self):
        results = {
            "jsd_knn": {"mse_mean": 0.05, "mse_std": 0.01, "invariants_ok": True},
            "jsd_alpha_knn": {"mse_mean": 0.03, "mse_std": 0.01, "invariants_ok": True},
            "adaptive_jsd_alpha_knn": {"mse_mean": 0.04, "mse_std": 0.01, "invariants_ok": True},
        }
        champion, details = select_baseline(results)
        assert champion == "jsd_alpha_knn"

    def test_discards_invariant_violators(self):
        results = {
            "jsd_knn": {"mse_mean": 0.05, "mse_std": 0.01, "invariants_ok": True},
            "jsd_alpha_knn": {"mse_mean": 0.01, "mse_std": 0.01, "invariants_ok": False},
        }
        champion, details = select_baseline(results)
        assert champion == "jsd_knn"

    def test_simplicity_tiebreak(self):
        results = {
            "jsd_knn": {"mse_mean": 0.05, "mse_std": 0.02, "invariants_ok": True},
            "jsd_alpha_knn": {"mse_mean": 0.051, "mse_std": 0.02, "invariants_ok": True},
        }
        champion, details = select_baseline(results, tolerance_std_multiplier=2.0)
        assert champion == "jsd_knn"

    def test_all_violate_raises(self):
        results = {
            "jsd_knn": {"mse_mean": 0.05, "invariants_ok": False},
        }
        with pytest.raises(ValueError):
            select_baseline(results)