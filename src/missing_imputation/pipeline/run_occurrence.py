"""Occurrence gate pipeline: fit pooled logistic, OOF, calibration, gate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..data.lineage import compute_sha256
from ..data.loader import load_config, load_covariants
from ..occurrence.artifacts import save_occurrence_artifacts
from ..occurrence.calibration import (
    apply_calibration_map,
    compute_occurrence_metrics,
    fit_calibration_map,
    prevalence_baseline_metrics,
)
from ..occurrence.features import (
    build_occurrence_features,
    build_target_prediction_features,
)
from ..occurrence.gating import (
    OccurrenceGateResult,
    apply_gate_to_targets,
    selective_false_zero_gate,
    fit_temperature_and_crc,
)
from ..occurrence.model import (
    OccurrenceConfig,
    PooledLogisticGate,
    run_oof,
    tune_C,
)
from ..data.lineage import compute_sha256
from ..data.loader import load_config, load_covariants


def load_split_ids(splits_dir: str | Path) -> dict:
    splits_dir = Path(splits_dir)
    splits = {}
    for name in ["random_cell", "empirical_pattern", "time_block", "country_holdout"]:
        path = splits_dir / f"{name}.json"
        with path.open(encoding="utf-8") as f:
            splits[name] = json.load(f)
    return splits


def _build_fold_splits(
    split_ids: dict,
    observed_mask: np.ndarray,
    n_cells: int,
    row_idx: np.ndarray,
    variant_idx: np.ndarray,
    n_folds: int,
    seed: int,
) -> tuple[list[list[tuple[np.ndarray, np.ndarray]]], list[str]]:
    recipe_names = ["random-cell", "empirical-pattern", "time-block", "country-holdout"]
    all_folds: list[list[tuple[np.ndarray, np.ndarray]]] = []

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_cells)
    fold_size = n_cells // n_folds
    random_folds = []
    for f in range(n_folds):
        s = f * fold_size
        e = s + fold_size if f < n_folds - 1 else n_cells
        test = perm[s:e]
        train = np.concatenate([perm[:s], perm[e:]])
        random_folds.append((train, test))
    all_folds.append(random_folds)

    ep_rows = set(split_ids["empirical_pattern"].get("test_row_indices", []))
    ep_mask = np.isin(row_idx, list(ep_rows))
    ep_test = np.where(ep_mask)[0]
    ep_train = np.where(~ep_mask)[0]
    all_folds.append([(ep_train, ep_test)] if len(ep_test) > 0 else [])

    tb_rows = set(split_ids["time_block"].get("test_row_indices", []))
    tb_mask = np.isin(row_idx, list(tb_rows))
    tb_test = np.where(tb_mask)[0]
    tb_train = np.where(~tb_mask)[0]
    all_folds.append([(tb_train, tb_test)] if len(tb_test) > 0 else [])

    held_out_locs = set(split_ids["country_holdout"].get("held_out_locations", []))
    ch_mask = np.zeros(n_cells, dtype=bool)
    ch_test = np.where(ch_mask)[0]
    ch_train = np.where(~ch_mask)[0]
    all_folds.append([(ch_train, ch_test)] if len(ch_test) > 0 else [])

    return all_folds, recipe_names


def run_occurrence_gate(
    data_path: str | Path,
    config_path: str | Path,
    splits_dir: str | Path,
    output_dir: str | Path,
    tsagris_path: str | Path | None = None,
) -> OccurrenceGateResult:
    project_root = Path(__file__).resolve().parents[3]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_config = load_config(config_path)
    df = load_covariants(data_path, data_config)
    variant_cols = data_config.variant_components
    observed_mask = df[variant_cols].notna().to_numpy()
    M_target = df[variant_cols].isna().to_numpy().astype(np.uint8)

    raw_checksum = compute_sha256(data_path)

    with Path(config_path).open(encoding="utf-8") as f:
        occ_cfg = yaml.safe_load(f).get("occurrence", {})

    occ_config = OccurrenceConfig(
        C_grid=occ_cfg.get("C_grid", [0.01, 0.1, 1.0, 10.0]),
        max_iter=occ_cfg.get("max_iter", 2000),
        n_folds=occ_cfg.get("n_folds", 5),
        min_fold_coverage=occ_cfg.get("min_fold_coverage", 0.8),
        min_valid_folds=occ_cfg.get("min_valid_folds", 3),
        seed=occ_cfg.get("seed", 42),
        max_false_zero_rate=occ_cfg.get("max_false_zero_rate", 0.05),
        min_zero_gate_support=occ_cfg.get("min_zero_gate_support", 50),
        threshold_grid=occ_cfg.get(
            "threshold_grid", [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        ),
    )

    feats = build_occurrence_features(
        df, variant_cols, observed_mask,
        location_col=data_config.location_col,
        date_col=data_config.date_col,
        total_seq_col=data_config.total_sequence_col,
    )
    X, y = feats.X, feats.y

    split_ids = load_split_ids(splits_dir)
    fold_splits, recipe_names = _build_fold_splits(
        split_ids, observed_mask, len(y), feats.row_idx, feats.variant_idx,
        occ_config.n_folds, occ_config.seed,
    )

    oof_results = run_oof(X, y, fold_splits, recipe_names, occ_config)

    metrics_rows = []
    for recipe, oof in oof_results.items():
        if oof.n_predictions == 0:
            metrics_rows.append({"recipe": recipe, "status": "no_predictions"})
            continue
        m = compute_occurrence_metrics(oof.y_true, oof.p_calibrated)
        baseline = prevalence_baseline_metrics(oof.y_true)
        metrics_rows.append({
            "recipe": recipe,
            **m,
            "baseline_log_loss": baseline["log_loss"],
            "baseline_brier": baseline["brier"],
            "baseline_ece": baseline["ece"],
            "coverage": oof.coverage,
            "n_folds_valid": oof.n_folds_valid,
        })

    release_ok = True
    release_reason = ""
    for recipe in ["time-block", "country-holdout"]:
        oof = oof_results.get(recipe)
        if oof is None or oof.n_predictions == 0:
            continue
        model_ll = compute_occurrence_metrics(oof.y_true, oof.p_calibrated)["log_loss"]
        base_ll = prevalence_baseline_metrics(oof.y_true)["log_loss"]
        if model_ll > base_ll + 0.1:
            release_ok = False
            release_reason = f"{recipe}: log-loss {model_ll:.4f} worse than baseline {base_ll:.4f}"
        if compute_occurrence_metrics(oof.y_true, oof.p_calibrated)["p_std"] < occ_config.collapse_std_min:
            release_ok = False
            release_reason = f"{recipe}: probability collapse (std < {occ_config.collapse_std_min})"

    rng = np.random.default_rng(occ_config.seed)
    n_val = max(1, len(y) // 5)
    val_pick = rng.choice(len(y), size=n_val, replace=False)
    val_mask = np.zeros(len(y), dtype=bool)
    val_mask[val_pick] = True
    train_mask = ~val_mask

    best_C = tune_C(X[train_mask], y[train_mask], occ_config.C_grid, val_mask, occ_config.max_iter, occ_config.seed)
    final_model = PooledLogisticGate(C=best_C, max_iter=occ_config.max_iter, seed=occ_config.seed)
    final_model.fit(X[train_mask], y[train_mask])
    p_val = final_model.predict_proba(X[val_mask])
    cal_map = fit_calibration_map(y[val_mask], p_val)

    from ..occurrence.gating import (
        fit_temperature_and_crc,
        temperature_scale,
        conformal_zero_bias,
        expit,
        logit,
    )

    X_targets, target_rows, target_vars = build_target_prediction_features(
        df, variant_cols, M_target, feats.min_date,
        location_col=data_config.location_col,
        date_col=data_config.date_col,
        total_seq_col=data_config.total_sequence_col,
    )
    p_targets_raw = final_model.predict_proba(X_targets) if len(X_targets) > 0 else np.array([])
    p_targets_cal = apply_calibration_map(cal_map, p_targets_raw)

    # Fit temperature scaling and CRC on OOF predictions
    # Build OOF DataFrame from all recipes
    oof_y_true = []
    oof_p_cal = []
    oof_variant = []
    for oof in oof_results.values():
        if oof.n_predictions == 0:
            continue
        oof_y_true.append(oof.y_true)
        oof_p_cal.append(oof.p_calibrated)
        # Map cell indices to variant names
        cell_var_idx = feats.variant_idx[oof.cell_indices]
        oof_variant.extend([feats.variant_names[v] for v in cell_var_idx])

    crc_result = fit_temperature_and_crc(
        oof_df=pd.DataFrame({
            "y_true": np.concatenate(oof_y_true),
            "p_calibrated": np.concatenate(oof_p_cal),
            "variant": np.array(oof_variant),
        }),
        y_true_col="y_true",
        p_calibrated_col="p_calibrated",
        variant_col="variant",
        variant_names=variant_cols,
        alpha=0.05,
        b_max=5.0,
        min_positive=10,
    )

    T = crc_result["temperature"]
    bias_per_variant = crc_result["bias_per_variant"]

    # Recalibrate target probabilities with temperature
    p_targets_raw = final_model.predict_proba(X_targets) if len(X_targets) > 0 else np.array([])
    p_targets_cal = apply_calibration_map(cal_map, p_targets_raw)

    # Apply temperature scaling to target probabilities
    if len(p_targets_cal) > 0:
        p_cal = p_targets_cal.clip(1e-12, 1 - 1e-12)
        logits = logit(p_cal)
        logits_cal = logits / T
        p_targets_recal = expit(logits_cal)
    else:
        p_targets_cal = np.array([])

    tau_per_variant: dict[str, float | None] = {}
    bias_per_variant: dict[str, float] = {}
    w_gan_per_variant: dict[str, float] = {}
    tau_hard_per_variant: dict[str, float] = {}
    threshold_rows = []

    for j, variant in enumerate(variant_cols):
        var_sel = target_vars == j
        if var_sel.sum() < occ_config.min_zero_gate_support:
            tau_per_variant[variant] = None
            bias_per_variant[variant] = 0.0
            w_gan_per_variant[variant] = 1.0
            tau_hard_per_variant[variant] = 0.0
            threshold_rows.append({"variant": variant, "status": "ABSTAIN_TO_GAN", "reason": "insufficient_support"})
            continue

        oof_key = "time-block" if oof_results.get("time-block") and oof_results["time-block"].n_predictions > 0 else "random-cell"
        oof = oof_results.get(oof_key)
        if oof is None or oof.n_predictions == 0:
            tau_per_variant[variant] = None
            bias_per_variant[variant] = 0.0
            w_gan_per_variant[variant] = 1.0
            tau_hard_per_variant[variant] = 0.0
            threshold_rows.append({"variant": variant, "status": "ABSTAIN_TO_GAN", "reason": "no_oof"})
            continue

        variant_oof_sel = np.isin(oof.cell_indices, np.where(feats.variant_idx == j)[0])
        if variant_oof_sel.sum() < occ_config.min_zero_gate_support:
            tau_per_variant[variant] = None
            bias_per_variant[variant] = 0.0
            w_gan_per_variant[variant] = 1.0
            tau_hard_per_variant[variant] = 0.0
            threshold_rows.append({"variant": variant, "status": "ABSTAIN_TO_GAN", "reason": "oof_support_low"})
            continue

        tau, diag = selective_false_zero_gate(
            oof.y_true[variant_oof_sel],
            oof.p_calibrated[variant_oof_sel],
            occ_config.threshold_grid,
            occ_config.max_false_zero_rate,
            occ_config.min_zero_gate_support,
        )
        tau_per_variant[variant] = tau

        # Get CRC bias
        b = crc_result["bias_per_variant"].get(variant, 0.0)
        bias_per_variant[variant] = b

        # Compute w_gan = sigma(logit(p_calibrated) - b) on calibration data
        variant_oof_sel_mask = variant_oof_sel
        if variant_oof_sel_mask.sum() > 0:
            p_cal = oof.p_calibrated[variant_oof_sel]
            logits = logit(np.clip(p_cal, 1e-12, 1 - 1e-12))
            w_gan = np.mean(expit(logits - b))
        else:
            w_gan = 1.0
        w_gan_per_variant[variant] = float(w_gan)

        # Compute hard-lock threshold tau_hard from variant zero-lock cap
        # For now, use the original tau as tau_hard
        tau_hard_per_variant[variant] = tau if tau is not None else 0.0

        if tau is not None:
            threshold_rows.append({
                "variant": variant, "status": "GATED", "tau_zero": tau,
                **diag.get("selected", {}),
            })
        else:
            threshold_rows.append({"variant": variant, "status": "ABSTAIN_TO_GAN", "reason": "no_threshold_meets_gate"})

    # Apply gate with CRC bias and soft fusion weights
    M_target_zero, M_gan, w_gan = apply_gate_to_targets(
        p_targets_cal, M_target, variant_cols, tau_per_variant,
        w_gan_per_variant=w_gan_per_variant,
        tau_hard_per_variant={v: 0.0 for v in variant_cols},  # no hard-lock at occurrence stage
    )

    gate_result = OccurrenceGateResult(
        M_target_zero=M_target_zero,
        M_gan=M_gan,
        M_hard_lock=np.zeros_like(M_target_zero),
        M_soft_fusion=np.zeros_like(M_target_zero),
        released=release_ok,
        release_reason=release_reason,
    )
    gate_result.validate(M_target)

    oof_df_rows = []
    for recipe, oof in oof_results.items():
        for i in range(oof.n_predictions):
            cell_i = oof.cell_indices[i]
            oof_df_rows.append({
                "recipe": recipe,
                "fold": oof.fold_ids[i],
                "row_idx": int(feats.row_idx[cell_i]),
                "variant_idx": int(feats.variant_idx[cell_i]),
                "variant": variant_cols[int(feats.variant_idx[cell_i])],
                "y_true": int(oof.y_true[i]),
                "p_raw": float(oof.p_raw[i]),
                "p_calibrated": float(oof.p_calibrated[i]),
            })
    oof_df = pd.DataFrame(oof_df_rows)

    feature_schema = {
        "feature_names": feats.feature_names,
        "variant_names": variant_cols,
        "min_date": str(feats.min_date),
        "n_features": len(feats.feature_names),
        "reference_variant": variant_cols[0],
    }

    manifest = {
        "model_type": "pooled_regularized_logistic",
        "C": best_C,
        "seed": occ_config.seed,
        "raw_checksum": raw_checksum,
        "config_checksum": compute_sha256(config_path),
        "split_ids_dir": str(splits_dir),
        "max_false_zero_rate": occ_config.max_false_zero_rate,
        "min_zero_gate_support": occ_config.min_zero_gate_support,
        "released": release_ok,
        "release_reason": release_reason,
        "n_target_zero": int(M_target_zero.sum()),
        "n_gan": int(M_gan.sum()),
        "temperature": float(T),
        "crc_bias": bias_per_variant,
        "code_version": "v2.1",
    }

    target_coordinates = np.column_stack([target_rows, target_vars]) if len(target_rows) > 0 else np.zeros((0, 2), dtype=np.int64)

    save_occurrence_artifacts(
        output_dir,
        oof_df,
        p_targets_cal,
        target_coordinates,
        M_target_zero,
        M_gan,
        metrics_rows,
        threshold_rows,
        feature_schema,
        manifest,
    )

    return OccurrenceGateResult(
        M_target_zero=M_target_zero,
        M_gan=M_gan,
        M_hard_lock=np.zeros_like(M_target_zero),
        M_soft_fusion=np.zeros_like(M_target_zero),
        released=release_ok,
        release_reason=release_reason,
    )