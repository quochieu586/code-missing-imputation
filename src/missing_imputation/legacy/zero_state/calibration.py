from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .zinb import ReducedZINBModel, PooledSparseZINBModel, ZINBConfig, ModelType
from .routing import ModelRouting
from ..evaluation.splits import (
    random_cell_split,
    empirical_pattern_split,
    time_block_split,
    SplitResult,
)


@dataclass
class CalibrationMetrics:
    nll: float
    brier_score: float
    calibration_curve: tuple[np.ndarray, np.ndarray]
    precision: float
    recall: float
    f1: float
    threshold: float
    n_samples: int
    ece: float = 0.0

    def to_dict(self) -> dict:
        return {
            "nll": self.nll,
            "brier_score": self.brier_score,
            "calibration_curve_bins": self.calibration_curve[0].tolist(),
            "calibration_curve_observed": self.calibration_curve[1].tolist(),
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "threshold": self.threshold,
            "n_samples": self.n_samples,
            "ece": self.ece,
        }


def generate_artificial_masks(
    observed_mask: np.ndarray,
    recipe: str,
    df: pd.DataFrame | None = None,
    config=None,
    n_folds: int = 5,
    seed: int = 42,
) -> list[SplitResult]:
    splits = []
    for fold in range(n_folds):
        fold_seed = seed + fold * 1000
        if recipe == "random-cell":
            split = random_cell_split(observed_mask, test_fraction=0.2, seed=fold_seed)
        elif recipe == "empirical-pattern":
            split = empirical_pattern_split(observed_mask, n_patterns=20, seed=fold_seed)
        elif recipe == "time-block":
            if df is None or config is None:
                raise ValueError("time-block recipe requires df and config")
            split = time_block_split(
                observed_mask, block_size_days=28, blocks_per_location=2, seed=fold_seed,
            )
        else:
            raise ValueError(f"Unknown recipe: {recipe}")
        splits.append(split)
    return splits


def compute_calibration_metrics(
    y_true: np.ndarray,
    p_nonzero: np.ndarray,
    p_zero: np.ndarray,
    threshold: float = 0.5,
) -> CalibrationMetrics:
    y_binary = (np.asarray(y_true) > 0).astype(np.float64)
    p_nonzero = np.asarray(p_nonzero, dtype=np.float64)
    y_pred_binary = (p_nonzero >= threshold).astype(np.float64)

    eps = 1e-12
    p_nz = np.clip(p_nonzero, eps, 1 - eps)
    p_z = np.clip(p_zero, eps, 1 - eps)
    nll = -np.mean(y_binary * np.log(p_nz) + (1 - y_binary) * np.log(p_z))

    brier = np.mean((y_binary - p_nonzero) ** 2)

    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    observed_freq = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (p_nonzero >= bin_edges[i]) & (p_nonzero < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (p_nonzero >= bin_edges[i]) & (p_nonzero <= bin_edges[i + 1])
        if np.any(mask):
            observed_freq[i] = np.mean(y_binary[mask])
        else:
            observed_freq[i] = np.nan

    tp = np.sum((y_binary == 1) & (y_pred_binary == 1))
    fp = np.sum((y_binary == 0) & (y_pred_binary == 1))
    fn = np.sum((y_binary == 1) & (y_pred_binary == 0))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    ece = _compute_ece(y_binary, p_nonzero, n_bins, bin_edges)

    return CalibrationMetrics(
        nll=float(nll),
        brier_score=float(brier),
        calibration_curve=(bin_centers, observed_freq),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        threshold=threshold,
        n_samples=int(len(y_true)),
        ece=float(ece),
    )


def _compute_ece(
    y_binary: np.ndarray,
    p_nonzero: np.ndarray,
    n_bins: int,
    bin_edges: np.ndarray,
) -> float:
    n = len(y_binary)
    if n == 0:
        return 0.0
    ece = 0.0
    for i in range(n_bins):
        mask = (p_nonzero >= bin_edges[i]) & (p_nonzero < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (p_nonzero >= bin_edges[i]) & (p_nonzero <= bin_edges[i + 1])
        if np.any(mask):
            bin_acc = np.mean(y_binary[mask])
            bin_conf = np.mean(p_nonzero[mask])
            ece += np.abs(bin_acc - bin_conf) * mask.sum() / n
    return float(ece)


def select_recall_constrained_threshold(
    y_true: np.ndarray,
    p_nonzero: np.ndarray,
    threshold_grid: list[float],
    min_recall: float = 0.95,
    objective: str = "precision",
) -> tuple[float, CalibrationMetrics]:
    best_threshold = threshold_grid[0] if threshold_grid else 0.5
    best_score = -np.inf
    best_metrics = None

    for threshold in threshold_grid:
        metrics = compute_calibration_metrics(y_true, p_nonzero, 1 - p_nonzero, threshold)
        if metrics.recall < min_recall:
            continue

        if objective == "precision":
            score = metrics.precision
        elif objective == "ece":
            score = -metrics.ece
        elif objective == "f1":
            score = metrics.f1
        else:
            score = metrics.precision

        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = metrics

    if best_metrics is None:
        best_metrics = compute_calibration_metrics(
            y_true, p_nonzero, 1 - p_nonzero, threshold_grid[0] if threshold_grid else 0.5
        )
        best_threshold = threshold_grid[0] if threshold_grid else 0.5

    return best_threshold, best_metrics


def compute_lag_lead_features(
    counts_grid: np.ndarray,
    M_observed_variant: np.ndarray,
    loc_indices: np.ndarray,
    time_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(loc_indices)
    lag_counts = np.zeros(n, dtype=np.float64)
    lag_avail = np.zeros(n, dtype=np.float64)
    lead_counts = np.zeros(n, dtype=np.float64)
    lead_avail = np.zeros(n, dtype=np.float64)

    n_times = counts_grid.shape[1]

    for i in range(n):
        loc, t = int(loc_indices[i]), int(time_indices[i])
        if t - 1 >= 0 and M_observed_variant[loc, t - 1] == 1:
            lag_counts[i] = counts_grid[loc, t - 1]
            lag_avail[i] = 1.0
        if t + 1 < n_times and M_observed_variant[loc, t + 1] == 1:
            lead_counts[i] = counts_grid[loc, t + 1]
            lead_avail[i] = 1.0

    return lag_counts, lag_avail, lead_counts, lead_avail


def build_pooled_stack(
    panel,
    M_observed: np.ndarray,
    variant_names: list[str],
    include_masks: dict[str, np.ndarray] | None = None,
    avail_masks: dict[str, np.ndarray] | None = None,
    validation_fraction: float = 0.2,
) -> dict:
    stacked_counts = []
    stacked_total_seq = []
    stacked_day_index = []
    stacked_lag_counts = []
    stacked_lag_avail = []
    stacked_lead_counts = []
    stacked_lead_avail = []
    variant_codes = []

    for code, variant_name in enumerate(variant_names):
        variant_idx = panel.feature_names.index(variant_name)
        include = (M_observed[:, :, variant_idx] == 1)
        if include_masks is not None and variant_name in include_masks:
            include = include & include_masks[variant_name].astype(bool)

        avail = (M_observed[:, :, variant_idx] == 1)
        if avail_masks is not None and variant_name in avail_masks:
            avail = avail & avail_masks[variant_name].astype(bool)

        obs_locs, obs_times = np.where(include)
        if len(obs_locs) == 0:
            continue

        counts_grid = panel.counts[:, :, variant_idx].astype(np.float64)
        lag_counts, lag_avail, lead_counts, lead_avail = compute_lag_lead_features(
            counts_grid, avail.astype(np.uint8), obs_locs, obs_times
        )

        stacked_counts.append(counts_grid[obs_locs, obs_times])
        stacked_total_seq.append(panel.total_sequence[obs_locs, obs_times].astype(np.float64))
        stacked_day_index.append(_compute_global_day_indices(panel, obs_locs, obs_times))
        stacked_lag_counts.append(lag_counts)
        stacked_lag_avail.append(lag_avail)
        stacked_lead_counts.append(lead_counts)
        stacked_lead_avail.append(lead_avail)
        variant_codes.append(np.full(len(obs_locs), code, dtype=np.int64))

    if len(stacked_counts) == 0:
        return {
            "stacked_counts": np.array([]),
            "stacked_total_seq": np.array([]),
            "stacked_day_index": np.array([]),
            "stacked_lag_counts": np.array([]),
            "stacked_lag_avail": np.array([]),
            "stacked_lead_counts": np.array([]),
            "stacked_lead_avail": np.array([]),
            "variant_codes": np.array([]),
        }

    stacked = {
        "stacked_counts": np.concatenate(stacked_counts),
        "stacked_total_seq": np.concatenate(stacked_total_seq),
        "stacked_day_index": np.concatenate(stacked_day_index),
        "stacked_lag_counts": np.concatenate(stacked_lag_counts),
        "stacked_lag_avail": np.concatenate(stacked_lag_avail),
        "stacked_lead_counts": np.concatenate(stacked_lead_counts),
        "stacked_lead_avail": np.concatenate(stacked_lead_avail),
        "variant_codes": np.concatenate(variant_codes),
    }
    if validation_fraction > 0:
        n_total_obs = len(stacked["stacked_counts"])
        n_val_obs = max(1, int(n_total_obs * validation_fraction))
        vm = np.zeros(n_total_obs, dtype=bool)
        vm[-n_val_obs:] = True
        stacked["validation_mask"] = vm
    return stacked


def run_oof_calibration(
    variant_name: str,
    panel,
    variant_idx: int,
    M_observed: np.ndarray,
    df: pd.DataFrame,
    data_config,
    zinb_config: ZINBConfig,
    routing: ModelRouting,
    pooled_variant_names: list[str] | None = None,
    min_fold_coverage: float = 0.8,
) -> tuple[CalibrationMetrics, float, dict]:
    M_obs_variant = M_observed[:, :, variant_idx]
    counts_grid = panel.counts[:, :, variant_idx].astype(np.float64)

    obs_locs, obs_times = np.where(M_obs_variant == 1)
    if len(obs_locs) == 0:
        empty = compute_calibration_metrics(np.array([0]), np.array([0.5]), np.array([0.5]), 0.5)
        return empty, 0.5, {"status": "no_observed_cells", "fold_details": [], "fold_coverage_ok": False}

    all_y_true: list[np.ndarray] = []
    all_p_nonzero: list[np.ndarray] = []
    all_p_baseline: list[np.ndarray] = []
    oof_records: list[dict] = []
    fold_details: list[dict] = []

    n_folds_total = 0
    n_folds_ok = 0
    recipe_ok: dict[str, int] = {}
    recipe_total: dict[str, int] = {}

    for recipe in zinb_config.artificial_mask_recipes:
        recipe_ok[recipe] = 0
        recipe_total[recipe] = 0
        splits = generate_artificial_masks(
            M_obs_variant,
            recipe=recipe,
            df=df,
            config=data_config,
            n_folds=zinb_config.n_artificial_folds,
            seed=zinb_config.fit_seed,
        )

        for fold_idx, split in enumerate(splits):
            n_folds_total += 1
            recipe_total[recipe] += 1
            train_mask = split.train_mask
            test_mask = split.test_mask

            train_locs, train_times = np.where(train_mask)
            test_locs, test_times = np.where(test_mask)
            if len(train_locs) == 0 or len(test_locs) == 0:
                fold_details.append({
                    "recipe": recipe, "fold": fold_idx,
                    "status": "failed", "reason": "empty train or test split",
                })
                continue

            train_counts = counts_grid[train_locs, train_times]
            test_counts = counts_grid[test_locs, test_times]
            prevalence = float((train_counts > 0).mean())

            train_total_seq = panel.total_sequence[train_locs, train_times].astype(np.float64)
            train_day_index = _compute_global_day_indices(panel, train_locs, train_times)
            test_total_seq = panel.total_sequence[test_locs, test_times].astype(np.float64)
            test_day_index = _compute_global_day_indices(panel, test_locs, test_times)

            try:
                if routing.model_tier == "reduced_per_variant":
                    model = ReducedZINBModel(variant_name, zinb_config)
                    train_lag_counts, train_lag_avail, train_lead_counts, train_lead_avail = compute_lag_lead_features(
                        counts_grid, train_mask.astype(np.uint8), train_locs, train_times
                    )
                    test_lag_counts, test_lag_avail, test_lead_counts, test_lead_avail = compute_lag_lead_features(
                        counts_grid, train_mask.astype(np.uint8), test_locs, test_times
                    )

                    n_train_obs = len(train_counts)
                    n_val_obs = max(1, n_train_obs // 5)
                    validation_mask = np.zeros(n_train_obs, dtype=bool)
                    validation_mask[-n_val_obs:] = True

                    fit_result = model.fit(
                        counts=train_counts,
                        total_seq=train_total_seq,
                        day_index=train_day_index,
                        lag_counts=train_lag_counts,
                        lag_avail=train_lag_avail,
                        lead_counts=train_lead_counts,
                        lead_avail=train_lead_avail,
                        validation_mask=validation_mask,
                    )

                    if fit_result.model_type != ModelType.ZINB:
                        fold_details.append({
                            "recipe": recipe, "fold": fold_idx,
                            "status": "abstain",
                            "reason": fit_result.fallback_reason,
                        })
                        continue

                    X_zi_train, X_nb_train, _ = model.encoder.transform(
                        train_total_seq, train_day_index, train_lag_counts,
                        train_lag_avail, train_lead_counts, train_lead_avail
                    )
                    rank_ok = (np.linalg.matrix_rank(X_zi_train) == X_zi_train.shape[1] and
                               np.linalg.matrix_rank(X_nb_train) == X_nb_train.shape[1])

                    post = model.predict_proba(
                        total_seq=test_total_seq,
                        day_index=test_day_index,
                        lag_counts=test_lag_counts,
                        lag_avail=test_lag_avail,
                        lead_counts=test_lead_counts,
                        lead_avail=test_lead_avail,
                    )
                else:
                    if pooled_variant_names is None:
                        fold_details.append({
                            "recipe": recipe, "fold": fold_idx,
                            "status": "abstain",
                            "reason": "pooled_variant_names not provided",
                        })
                        continue

                    test_lag_counts, test_lag_avail, test_lead_counts, test_lead_avail = compute_lag_lead_features(
                        counts_grid, train_mask.astype(np.uint8), test_locs, test_times
                    )

                    stacked = build_pooled_stack(
                        panel, M_observed, pooled_variant_names,
                        include_masks={variant_name: train_mask},
                        avail_masks={variant_name: train_mask},
                    )
                    if len(stacked["stacked_counts"]) == 0:
                        fold_details.append({
                            "recipe": recipe, "fold": fold_idx,
                            "status": "failed", "reason": "empty pooled stack",
                        })
                        continue

                    fold_pooled = PooledSparseZINBModel(pooled_variant_names, zinb_config)
                    pooled_fit = fold_pooled.fit(**stacked)

                    if pooled_fit.model_type != ModelType.ZINB:
                        fold_details.append({
                            "recipe": recipe, "fold": fold_idx,
                            "status": "abstain",
                            "reason": pooled_fit.fallback_reason or "pooled fold fit failed",
                        })
                        continue

                    X_zi_train, X_nb_train, _ = fold_pooled.encoder.transform(
                        stacked["stacked_total_seq"], stacked["stacked_day_index"],
                        stacked["stacked_lag_counts"], stacked["stacked_lag_avail"],
                        stacked["stacked_lead_counts"], stacked["stacked_lead_avail"],
                    )
                    n_stack = len(stacked["stacked_counts"])
                    variant_dummies = fold_pooled._build_variant_dummies(stacked["variant_codes"], n_stack)
                    X_zi_full = np.hstack([X_zi_train, variant_dummies])
                    X_nb_full = np.hstack([X_nb_train, variant_dummies])
                    rank_ok = (np.linalg.matrix_rank(X_zi_full) == X_zi_full.shape[1] and
                               np.linalg.matrix_rank(X_nb_full) == X_nb_full.shape[1])

                    variant_code = pooled_variant_names.index(variant_name)
                    post = fold_pooled.predict_proba_variant(
                        variant_code=variant_code,
                        total_seq=test_total_seq,
                        day_index=test_day_index,
                        lag_counts=test_lag_counts,
                        lag_avail=test_lag_avail,
                        lead_counts=test_lead_counts,
                        lead_avail=test_lead_avail,
                    )
            except Exception as e:
                fold_details.append({
                    "recipe": recipe, "fold": fold_idx,
                    "status": "failed", "reason": f"exception: {e}",
                })
                continue

            p_nz = np.asarray(post.p_nonzero, dtype=np.float64)
            all_y_true.append(test_counts)
            all_p_nonzero.append(p_nz)
            all_p_baseline.append(np.full(len(test_counts), prevalence, dtype=np.float64))

            n_folds_ok += 1
            recipe_ok[recipe] += 1

            for i in range(len(test_counts)):
                oof_records.append({
                    "variant_name": variant_name,
                    "recipe": recipe,
                    "fold": fold_idx,
                    "loc_index": int(test_locs[i]),
                    "time_index": int(test_times[i]),
                    "y_true": float(test_counts[i]),
                    "p_nonzero": float(p_nz[i]),
                    "p_baseline": float(prevalence),
                })

            fold_details.append({
                "recipe": recipe, "fold": fold_idx,
                "status": "ok",
                "n_train": int(len(train_locs)),
                "n_test": int(len(test_locs)),
                "rank_ok": bool(rank_ok),
                "schema_match": True,
                "train_prevalence": prevalence,
            })

    if len(all_y_true) == 0:
        empty = compute_calibration_metrics(np.array([0]), np.array([0.5]), np.array([0.5]), 0.5)
        return empty, 0.5, {
            "fold_details": fold_details,
            "status": "no_oof_predictions",
            "fold_coverage_ok": False,
            "oof_records": [],
        }

    y_oof = np.concatenate(all_y_true)
    p_oof = np.concatenate(all_p_nonzero)
    p_base_oof = np.concatenate(all_p_baseline)

    best_threshold, best_metrics = select_recall_constrained_threshold(
        y_oof,
        p_oof,
        zinb_config.threshold_grid,
        min_recall=zinb_config.min_nonzero_recall,
        objective=zinb_config.threshold_objective,
    )

    y_binary = (y_oof > 0).astype(np.float64)
    eps = 1e-12
    p_b = np.clip(p_base_oof, eps, 1 - eps)
    baseline_nll = float(-np.mean(y_binary * np.log(p_b) + (1 - y_binary) * np.log(1 - p_b)))
    baseline_brier = float(np.mean((y_binary - p_base_oof) ** 2))

    rank_ok_all = all(fd.get("rank_ok", True) for fd in fold_details if fd.get("status") == "ok")

    coverage = (n_folds_ok / n_folds_total) if n_folds_total > 0 else 0.0
    recipe_coverage_ok = all(
        recipe_ok[r] >= max(1, int(np.ceil(min_fold_coverage * recipe_total[r])))
        for r in recipe_total if recipe_total[r] > 0
    )
    fold_coverage_ok = bool(coverage >= min_fold_coverage and recipe_coverage_ok)

    details = {
        "fold_details": fold_details,
        "n_oof_samples": int(len(y_oof)),
        "status": "ok",
        "rank_ok": bool(rank_ok_all),
        "schema_match": True,
        "best_threshold": float(best_threshold),
        "frac_nonzero_at_selected_threshold": float(np.mean(p_oof >= best_threshold)),
        "frac_nonzero_at_0_5": float(np.mean(p_oof >= 0.5)),
        "fold_coverage": float(coverage),
        "fold_coverage_ok": fold_coverage_ok,
        "baseline_nll": baseline_nll,
        "baseline_brier": baseline_brier,
        "oof_records": oof_records,
    }
    return best_metrics, best_threshold, details


def compute_baseline_metrics(y_true: np.ndarray) -> dict:
    y_binary = (np.asarray(y_true) > 0).astype(np.float64)
    prevalence = y_binary.mean()
    eps = 1e-12
    baseline_nll = -np.mean(
        y_binary * np.log(np.clip(prevalence, eps, 1 - eps))
        + (1 - y_binary) * np.log(np.clip(1 - prevalence, eps, 1 - eps))
    )
    baseline_brier = np.mean((y_binary - prevalence) ** 2)
    return {"baseline_nll": float(baseline_nll), "baseline_brier": float(baseline_brier)}


def _compute_global_day_indices(
    panel, loc_indices: np.ndarray, time_indices: np.ndarray
) -> np.ndarray:
    if len(panel.time_index) == 0:
        return np.zeros(len(loc_indices), dtype=np.float64)
    t0 = panel.time_index[0]
    day_indices = np.array([
        (panel.time_index[int(t)] - t0).days for t in time_indices
    ], dtype=np.float64)
    return day_indices


def create_m_artificial_zero(
    observed_mask: np.ndarray,
    artificial_mask: np.ndarray,
) -> np.ndarray:
    return observed_mask & artificial_mask


def create_m_artificial_mag(
    observed_mask: np.ndarray,
    artificial_mask: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    positive_mask = (counts > 0) & observed_mask
    return positive_mask & artificial_mask