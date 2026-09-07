"""Probability calibration and occurrence metrics."""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


def fit_calibration_map(y_val: np.ndarray, p_val: np.ndarray) -> IsotonicRegression | None:
    if len(np.unique(y_val)) < 2:
        return None
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    ir.fit(p_val, y_val)
    return ir


def apply_calibration_map(ir: IsotonicRegression | None, p: np.ndarray) -> np.ndarray:
    if ir is None:
        return p
    return ir.predict(p)


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = y[mask].mean()
        bin_conf = p[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def calibration_curve(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    observed = np.zeros(n_bins)
    predicted = np.zeros(n_bins)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if mask.sum() > 0:
            observed[i] = y[mask].mean()
            predicted[i] = p[mask].mean()
    return predicted, observed


def compute_occurrence_metrics(
    y: np.ndarray, p: np.ndarray, threshold: float = 0.5, n_bins: int = 10
) -> dict:
    metrics: dict = {
        "n": int(len(y)),
        "log_loss": log_loss(y, p),
        "brier": brier_score(y, p),
        "ece": expected_calibration_error(y, p, n_bins),
        "prevalence": float(y.mean()) if len(y) > 0 else 0.0,
        "p_std": float(p.std()) if len(p) > 0 else 0.0,
    }

    if len(y) > 0 and len(np.unique(y)) == 2:
        from sklearn.metrics import average_precision_score, roc_auc_score

        metrics["roc_auc"] = float(roc_auc_score(y, p))
        metrics["pr_auc"] = float(average_precision_score(y, p))

    pred_pos = p >= threshold
    pred_zero = ~pred_pos
    metrics["precision_at_threshold"] = float(
        (pred_pos & (y == 1)).sum() / max(pred_pos.sum(), 1)
    )
    metrics["recall_at_threshold"] = float(
        (pred_pos & (y == 1)).sum() / max((y == 1).sum(), 1)
    )
    tp = (pred_pos & (y == 1)).sum()
    fp = (pred_pos & (y == 0)).sum()
    fn = (pred_zero & (y == 1)).sum()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    metrics["f1_at_threshold"] = float(
        2 * precision * recall / max(precision + recall, 1e-12)
    )

    return metrics


def prevalence_baseline_metrics(y: np.ndarray) -> dict:
    if len(y) == 0:
        return {"log_loss": float("nan"), "brier": float("nan"), "ece": float("nan")}
    prevalence = float(y.mean())
    p_const = np.full(len(y), prevalence)
    return {
        "log_loss": log_loss(y, p_const),
        "brier": brier_score(y, p_const),
        "ece": expected_calibration_error(y, p_const),
        "prevalence": prevalence,
    }
