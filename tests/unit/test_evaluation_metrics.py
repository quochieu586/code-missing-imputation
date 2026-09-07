import numpy as np
import pytest

from missing_imputation.evaluation.metrics import (
    jsd_metric,
    mae_proportions,
    rmse_proportions,
    mse_proportions,
    wasserstein_per_variant,
    wasserstein_mean,
    distribution_fidelity,
    temporal_fidelity,
    count_metrics_by_bin,
    seed_stability,
    compute_all_metrics,
)


def test_mse_proportions():
    true = np.array([[0.2, 0.3, 0.5], [0.1, 0.4, 0.5]])
    imputed = np.array([[0.25, 0.25, 0.5], [0.15, 0.35, 0.5]])
    mse = mse_proportions(true, imputed)
    assert mse > 0
    assert mse < 0.01


def test_mse_zero_when_identical():
    true = np.array([[0.2, 0.3, 0.5]])
    mse = mse_proportions(true, true)
    assert mse == 0.0


def test_wasserstein_per_variant():
    true = np.array([[0.2, 0.3, 0.5], [0.1, 0.4, 0.5]])
    imputed = np.array([[0.25, 0.25, 0.5], [0.15, 0.35, 0.5]])
    result = wasserstein_per_variant(true, imputed, ["A", "B", "C"])
    assert len(result) == 3
    assert all(v >= 0 for v in result.values())


def test_wasserstein_mean():
    true = np.array([[0.2, 0.3, 0.5], [0.1, 0.4, 0.5]])
    imputed = np.array([[0.25, 0.25, 0.5], [0.15, 0.35, 0.5]])
    result = wasserstein_mean(true, imputed)
    assert result >= 0


def test_distribution_fidelity():
    true = np.array([[0.2, 0.3, 0.5], [0.1, 0.4, 0.5], [0.3, 0.3, 0.4]])
    imputed = np.array([[0.25, 0.25, 0.5], [0.15, 0.35, 0.5], [0.35, 0.25, 0.4]])
    result = distribution_fidelity(true, imputed)
    assert "mean_diff" in result
    assert "var_diff" in result
    assert "zero_prevalence_diff" in result
    assert "corr_matrix_diff_mean" in result
    assert np.isfinite(result["mean_diff"])
    assert np.isfinite(result["var_diff"])
    assert np.isfinite(result["zero_prevalence_diff"])


def test_temporal_fidelity():
    true_panel = np.random.dirichlet([1, 1, 1], size=(2, 5)).reshape(2, 5, 3)
    imputed_panel = true_panel + np.random.normal(0, 0.01, true_panel.shape)
    imputed_panel = np.abs(imputed_panel)
    imputed_panel = imputed_panel / imputed_panel.sum(axis=2, keepdims=True)
    result = temporal_fidelity(true_panel, imputed_panel)
    assert "roughness_true_mean" in result
    assert "roughness_imputed_mean" in result
    assert "lag1_change_true_mean" in result
    assert "lag1_change_imputed_mean" in result
    assert "time_window_jsd" in result
    assert result["roughness_true_mean"] >= 0


def test_count_metrics_by_bin():
    true_counts = np.array([[10, 20], [5, 15], [50, 100], [3, 7], [80, 120]])
    imputed_counts = np.array([[12, 18], [6, 14], [48, 102], [4, 6], [78, 122]])
    total_seq = np.array([30, 20, 150, 10, 200])
    result = count_metrics_by_bin(true_counts, imputed_counts, total_seq, n_bins=3)
    assert len(result) > 0
    for bin_key, bin_data in result.items():
        assert "mse" in bin_data
        assert "mae" in bin_data
        assert bin_data["mse"] >= 0


def test_seed_stability():
    metrics_per_seed = [
        {"mse_prop": 0.001, "jsd": 0.05},
        {"mse_prop": 0.0012, "jsd": 0.06},
        {"mse_prop": 0.0009, "jsd": 0.04},
    ]
    stability = seed_stability(metrics_per_seed)
    assert "mse_prop" in stability
    assert "jsd" in stability
    assert stability["mse_prop"]["mean"] > 0
    assert stability["mse_prop"]["std"] >= 0
    assert stability["mse_prop"]["n_seeds"] == 3


def test_compute_all_metrics():
    true = np.array([[0.2, 0.3, 0.5], [0.1, 0.4, 0.5]])
    imputed = np.array([[0.25, 0.25, 0.5], [0.15, 0.35, 0.5]])
    result = compute_all_metrics(true, imputed, variant_names=["A", "B", "C"])
    assert "jsd" in result
    assert "mse_prop" in result
    assert "rmse_prop" in result
    assert "mae_prop" in result
    assert "wasserstein_A" in result
    assert "wasserstein_B" in result
    assert "wasserstein_C" in result


def test_compute_all_metrics_with_counts():
    true_props = np.array([[0.5, 0.5], [0.3, 0.7]])
    imputed_props = np.array([[0.55, 0.45], [0.35, 0.65]])
    true_counts = np.array([[50, 50], [30, 70]])
    imputed_counts = np.array([[55, 45], [35, 65]])
    result = compute_all_metrics(true_props, imputed_props, true_counts, imputed_counts)
    assert "mae_count" in result