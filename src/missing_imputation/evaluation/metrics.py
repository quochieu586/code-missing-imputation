"""Evaluation metrics for compositional imputation."""

from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance

from ..baselines.jsd import _jsd_single


def jsd_metric(
    true_proportions: np.ndarray,
    imputed_proportions: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Mean JSD between true and imputed proportions.

    Args:
        true_proportions: (n, d) true compositions.
        imputed_proportions: (n, d) imputed compositions.
        mask: Optional (n, d) boolean mask for which cells to evaluate.

    Returns:
        Mean JSD value.
    """
    n = true_proportions.shape[0]
    jsd_values = []

    for i in range(n):
        jsd_values.append(_jsd_single(true_proportions[i], imputed_proportions[i]))

    return float(np.mean(jsd_values)) if jsd_values else 0.0


def jsd_per_variant(
    true_proportions: np.ndarray,
    imputed_proportions: np.ndarray,
    variant_names: list[str] | None = None,
) -> dict[str, float]:
    """JSD per variant (computed column-wise).

    Args:
        true_proportions: (n, d) true compositions.
        imputed_proportions: (n, d) imputed compositions.
        variant_names: Optional names for variants. Defaults to V0, V1, ...

    Returns:
        Dictionary mapping variant name -> JSD.
    """
    n, d = true_proportions.shape
    if variant_names is None:
        variant_names = [f"V{i}" for i in range(d)]

    result_dict = {}
    for j in range(d):
        true_col = true_proportions[:, j]
        imputed_col = imputed_proportions[:, j]
        # JSD between 1D marginal distributions
        # Use histogram-based approach for continuous marginals
        eps = 1e-12
        true_vals = true_col[true_col > eps]
        imputed_vals = imputed_col[imputed_col > eps]
        if len(true_vals) == 0 or len(imputed_vals) == 0:
            jsd_val = 0.0
        else:
            bins = np.linspace(0, 1, 51)
            true_hist, _ = np.histogram(true_vals, bins=bins, density=True)
            imputed_hist, _ = np.histogram(imputed_vals, bins=bins, density=True)
            true_hist = true_hist + 1e-12
            imputed_hist = imputed_hist + 1e-12
            true_hist = true_hist / true_hist.sum()
            imputed_hist = imputed_hist / imputed_hist.sum()
            m = 0.5 * (true_hist + imputed_hist)
            jsd_val = 0.5 * (np.sum(true_hist * np.log(true_hist / m)) + np.sum(imputed_hist * np.log(imputed_hist / m)))
        result_dict[variant_names[j]] = float(jsd_val)
    return result_dict


def mae_proportions(
    true_proportions: np.ndarray,
    imputed_proportions: np.ndarray,
) -> float:
    """Mean Absolute Error on proportions."""
    return float(np.mean(np.abs(true_proportions - imputed_proportions)))


def rmse_proportions(
    true_proportions: np.ndarray,
    imputed_proportions: np.ndarray,
) -> float:
    """Root Mean Squared Error on proportions."""
    return float(np.sqrt(np.mean((true_proportions - imputed_proportions) ** 2)))


def mse_proportions(
    true_proportions: np.ndarray,
    imputed_proportions: np.ndarray,
) -> float:
    """Mean Squared Error on proportions."""
    return float(np.mean((true_proportions - imputed_proportions) ** 2))


def mae_counts(
    true_counts: np.ndarray,
    imputed_counts: np.ndarray,
) -> float:
    """Mean Absolute Error on counts."""
    return float(np.mean(np.abs(true_counts - imputed_counts)))


def zero_precision(
    true_values: np.ndarray,
    imputed_values: np.ndarray,
    threshold: float = 1e-10,
) -> float:
    """Precision for zero prediction."""
    predicted_zero = imputed_values <= threshold
    actual_zero = true_values <= threshold

    if predicted_zero.sum() == 0:
        return 0.0
    return float((predicted_zero & actual_zero).sum() / predicted_zero.sum())


def zero_recall(
    true_values: np.ndarray,
    imputed_values: np.ndarray,
    threshold: float = 1e-10,
) -> float:
    """Recall for zero prediction."""
    predicted_zero = imputed_values <= threshold
    actual_zero = true_values <= threshold

    if actual_zero.sum() == 0:
        return 0.0
    return float((predicted_zero & actual_zero).sum() / actual_zero.sum())


def zero_f1(
    true_values: np.ndarray,
    imputed_values: np.ndarray,
    threshold: float = 1e-10,
) -> float:
    """F1 score for zero prediction."""
    p = zero_precision(true_values, imputed_values, threshold)
    r = zero_recall(true_values, imputed_values, threshold)
    if p + r == 0:
        return 0.0
    return float(2 * p * r / (p + r))


def wasserstein_per_variant(
    true_proportions: np.ndarray,
    imputed_proportions: np.ndarray,
    variant_names: list[str] | None = None,
) -> dict[str, float]:
    """Wasserstein distance per variant (1D).

    Args:
        true_proportions: (n, d) true compositions.
        imputed_proportions: (n, d) imputed compositions.
        variant_names: Optional names for variants. Defaults to V0, V1, ...

    Returns:
        Dictionary mapping variant name -> Wasserstein distance.
    """
    n, d = true_proportions.shape
    if variant_names is None:
        variant_names = [f"V{i}" for i in range(d)]

    result = {}
    for j in range(d):
        true_vals = true_proportions[:, j]
        imputed_vals = imputed_proportions[:, j]
        result[variant_names[j]] = float(wasserstein_distance(true_vals, imputed_vals))
    return result


def wasserstein_mean(
    true_proportions: np.ndarray,
    imputed_proportions: np.ndarray,
) -> float:
    """Mean Wasserstein distance across all variants."""
    dists = wasserstein_per_variant(true_proportions, imputed_proportions)
    return float(np.mean(list(dists.values())))


def distribution_fidelity(
    true_proportions: np.ndarray,
    imputed_proportions: np.ndarray,
    variant_names: list[str] | None = None,
) -> dict:
    """Distribution fidelity metrics: mean, variance, quantiles, zero prevalence, correlation.

    Args:
        true_proportions: (n, d) true compositions.
        imputed_proportions: (n, d) imputed compositions.
        variant_names: Optional variant names.

    Returns:
        Dictionary with distribution comparison metrics.
    """
    n, d = true_proportions.shape
    if variant_names is None:
        variant_names = [f"V{i}" for i in range(d)]

    # Per-variant statistics
    true_mean = true_proportions.mean(axis=0)
    imputed_mean = imputed_proportions.mean(axis=0)
    true_var = true_proportions.var(axis=0)
    imputed_var = imputed_proportions.var(axis=0)

    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    true_quantiles = np.quantile(true_proportions, quantiles, axis=0)
    imputed_quantiles = np.quantile(imputed_proportions, quantiles, axis=0)

    true_zero_prev = (true_proportions <= 1e-10).mean(axis=0)
    imputed_zero_prev = (imputed_proportions <= 1e-10).mean(axis=0)

    # Correlation matrix difference
    true_corr = np.corrcoef(true_proportions, rowvar=False)
    imputed_corr = np.corrcoef(imputed_proportions, rowvar=False)
    corr_diff = np.abs(true_corr - imputed_corr)
    corr_diff_mean = float(np.mean(corr_diff))
    corr_diff_max = float(np.max(corr_diff))

    # Aggregate metrics
    mean_diff = float(np.mean(np.abs(true_mean - imputed_mean)))
    var_diff = float(np.mean(np.abs(true_var - imputed_var)))
    quantile_diffs = {}
    for i, q in enumerate(quantiles):
        quantile_diffs[f"quantile_{q}"] = float(np.mean(np.abs(true_quantiles[i] - imputed_quantiles[i])))
    zero_prev_diff = float(np.mean(np.abs(true_zero_prev - imputed_zero_prev)))

    return {
        "mean_diff": mean_diff,
        "var_diff": var_diff,
        "quantile_diffs": quantile_diffs,
        "zero_prevalence_diff": zero_prev_diff,
        "corr_matrix_diff_mean": corr_diff_mean,
        "corr_matrix_diff_max": corr_diff_max,
        "per_variant": {
            variant_names[j]: {
                "true_mean": float(true_mean[j]),
                "imputed_mean": float(imputed_mean[j]),
                "true_var": float(true_var[j]),
                "imputed_var": float(imputed_var[j]),
                "true_zero_prev": float(true_zero_prev[j]),
                "imputed_zero_prev": float(imputed_zero_prev[j]),
            }
            for j in range(d)
        },
    }


def temporal_roughness(
    values: np.ndarray,
    time_indices: np.ndarray | None = None,
) -> float:
    """Temporal roughness: mean absolute change between consecutive timepoints.

    Args:
        values: (n_times, d) values over time for one location.
        time_indices: Optional time indices for irregular spacing.

    Returns:
        Mean absolute first difference.
    """
    if values.shape[0] < 2:
        return 0.0
    diffs = np.abs(np.diff(values, axis=0))
    return float(np.mean(diffs))


def lag1_change_distribution(
    values: np.ndarray,
) -> dict:
    """Lag-1 change distribution statistics.

    Args:
        values: (n_times, d) values over time for one location.

    Returns:
        Dictionary with mean, std, quantiles of absolute lag-1 changes.
    """
    if values.shape[0] < 2:
        return {"mean": 0.0, "std": 0.0, "quantiles": {}}
    diffs = np.abs(np.diff(values, axis=0))
    flat_diffs = diffs.flatten()
    return {
        "mean": float(flat_diffs.mean()),
        "std": float(flat_diffs.std()),
        "quantiles": {
            f"q{q}": float(np.quantile(flat_diffs, q))
            for q in [0.1, 0.25, 0.5, 0.75, 0.9]
        },
    }


def time_window_jsd(
    true_panel: np.ndarray,
    imputed_panel: np.ndarray,
    window_size: int = 4,
) -> dict:
    """Time-window JSD: JSD computed over rolling windows.

    Args:
        true_panel: (n_locations, n_times, d) true compositions.
        imputed_panel: (n_locations, n_times, d) imputed compositions.
        window_size: Number of timepoints per window.

    Returns:
        Dictionary with mean/std JSD per window across locations.
    """
    n_loc, n_time, d = true_panel.shape
    results = {}
    for start in range(0, n_time - window_size + 1, window_size):
        end = min(start + window_size, n_time)
        window_true = true_panel[:, start:end].reshape(-1, d)
        window_imputed = imputed_panel[:, start:end].reshape(-1, d)
        if len(window_true) > 0:
            jsd_val = jsd_metric(window_true, window_imputed)
            results[f"window_{start}_{end}"] = jsd_val
    return {
        "per_window": results,
        "mean": float(np.mean(list(results.values()))) if results else 0.0,
        "std": float(np.std(list(results.values()))) if results else 0.0,
    }


def temporal_fidelity(
    true_panel: np.ndarray,
    imputed_panel: np.ndarray,
    time_indices: np.ndarray | None = None,
    window_size: int = 4,
) -> dict:
    """Temporal fidelity metrics: roughness, lag-1 change, time-window JSD.

    Args:
        true_panel: (n_locations, n_times, d) true compositions.
        imputed_panel: (n_locations, n_times, d) imputed compositions.
        time_indices: Optional time indices.
        window_size: Window size for time-window JSD.

    Returns:
        Dictionary with temporal fidelity metrics.
    """
    n_loc = true_panel.shape[0]

    roughness_vals = []
    lag1_vals = []
    for loc in range(n_loc):
        roughness_vals.append(temporal_roughness(true_panel[loc], time_indices))
        roughness_vals.append(temporal_roughness(imputed_panel[loc], time_indices))
        lag1_true = lag1_change_distribution(true_panel[loc])
        lag1_imputed = lag1_change_distribution(imputed_panel[loc])
        lag1_vals.append(lag1_true["mean"])
        lag1_vals.append(lag1_imputed["mean"])

    tw_jsd = time_window_jsd(true_panel, imputed_panel, window_size)

    return {
        "roughness_true_mean": float(np.mean([temporal_roughness(true_panel[loc], time_indices) for loc in range(n_loc)])),
        "roughness_imputed_mean": float(np.mean([temporal_roughness(imputed_panel[loc], time_indices) for loc in range(n_loc)])),
        "lag1_change_true_mean": float(np.mean([lag1_change_distribution(true_panel[loc])["mean"] for loc in range(n_loc)])),
        "lag1_change_imputed_mean": float(np.mean([lag1_change_distribution(imputed_panel[loc])["mean"] for loc in range(n_loc)])),
        "time_window_jsd": tw_jsd,
    }


def count_metrics_by_bin(
    true_counts: np.ndarray,
    imputed_counts: np.ndarray,
    total_seq: np.ndarray,
    n_bins: int = 5,
) -> dict:
    """Count metrics binned by total_sequence quantiles.

    Args:
        true_counts: (n, d) true counts.
        imputed_counts: (n, d) imputed counts.
        total_seq: (n,) total sequence per row.
        n_bins: Number of quantile bins.

    Returns:
        Dictionary with MSE/MAE per bin.
    """
    bins = np.quantile(total_seq, np.linspace(0, 1, n_bins + 1))
    bins[0] = -1  # Ensure first bin includes minimum
    bins[-1] = np.inf  # Ensure last bin includes maximum

    results = {}
    for i in range(n_bins):
        mask = (total_seq > bins[i]) & (total_seq <= bins[i + 1])
        if mask.sum() == 0:
            continue
        tc = true_counts[mask]
        ic = imputed_counts[mask]
        results[f"bin_{i}"] = {
            "n_samples": int(mask.sum()),
            "total_seq_range": [float(bins[i] + 1), float(bins[i + 1])],
            "mse": float(np.mean((tc - ic) ** 2)),
            "mae": float(np.mean(np.abs(tc - ic))),
            "rmse": float(np.sqrt(np.mean((tc - ic) ** 2))),
        }
    return results


def seed_stability(metrics_per_seed: list[dict]) -> dict:
    """Seed stability: mean ± std across seeds, coefficient of variation.

    Args:
        metrics_per_seed: List of metric dictionaries, one per seed.

    Returns:
        Dictionary with mean, std, cv for each metric.
    """
    if not metrics_per_seed:
        return {}

    all_keys = set()
    for m in metrics_per_seed:
        all_keys.update(m.keys())

    result = {}
    for key in all_keys:
        vals = [m.get(key, np.nan) for m in metrics_per_seed]
        vals = np.array([v for v in vals if not np.isnan(v)])
        if len(vals) == 0:
            continue
        mean_val = float(vals.mean())
        std_val = float(vals.std())
        cv = std_val / mean_val if mean_val != 0 else np.inf
        result[key] = {
            "mean": mean_val,
            "std": std_val,
            "cv": float(cv),
            "n_seeds": len(vals),
        }
    return result


def compute_all_metrics(
    true_proportions: np.ndarray,
    imputed_proportions: np.ndarray,
    true_counts: np.ndarray | None = None,
    imputed_counts: np.ndarray | None = None,
    variant_names: list[str] | None = None,
) -> dict[str, float]:
    """Compute all evaluation metrics.

    Returns:
        Dictionary of metric name -> value.
    """
    metrics = {
        "jsd": jsd_metric(true_proportions, imputed_proportions),
        "mse_prop": mse_proportions(true_proportions, imputed_proportions),
        "mae_prop": mae_proportions(true_proportions, imputed_proportions),
        "rmse_prop": rmse_proportions(true_proportions, imputed_proportions),
        "wasserstein_mean": wasserstein_mean(true_proportions, imputed_proportions),
        "zero_precision": zero_precision(true_proportions, imputed_proportions),
        "zero_recall": zero_recall(true_proportions, imputed_proportions),
        "zero_f1": zero_f1(true_proportions, imputed_proportions),
    }

    if variant_names is not None:
        ws_per_var = wasserstein_per_variant(true_proportions, imputed_proportions, variant_names)
        for k, v in ws_per_var.items():
            metrics[f"wasserstein_{k}"] = v
        jsd_per_var = jsd_per_variant(true_proportions, imputed_proportions, variant_names)
        for k, v in jsd_per_var.items():
            metrics[f"jsd_{k}"] = v

    if true_counts is not None and imputed_counts is not None:
        metrics["mae_count"] = mae_counts(true_counts, imputed_counts)

    return metrics
