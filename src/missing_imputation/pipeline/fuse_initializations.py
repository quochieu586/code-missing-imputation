"""Truth-table fusion v2: raw + occurrence gate + Tsagris warm-start.
Join on (location, date, variant) keys, validate parent checksums, lock
observed and confident-zero cells, allocate remaining budget by largest
remainder with soft fusion from GAN.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.closure import allocate_with_lock, compute_other, validate_closure
from ..data.lineage import compute_sha256
from ..data.loader import load_config, load_covariants
from ..data.lineage import load_lineage
from ..occurrence.gating import (
    apply_soft_fusion,
    fit_temperature_and_crc,
    temperature_scale,
    conformal_zero_bias,
    expit,
    logit,
)
from ..tracking.manifest import create_run_manifest


def _require_artifact(path: str | Path, name: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{name} not found: {p}")
    return p


def _validate_parent_manifest(occurrence_dir: Path) -> dict:
    manifest_path = occurrence_dir / "occurrence_model_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"occurrence_model_manifest.json missing in {occurrence_dir}; fusion refuses to run"
        )
    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)

    # Check that required artifact files exist and their checksums match
    required_artifacts = {
        "M_target_zero": "M_target_zero.npz",
        "M_gan": "M_gan.npz",
        "posterior": "occurrence_posterior.npz",
    }
    for key, filename in required_artifacts.items():
        ap = occurrence_dir / filename
        if not ap.exists():
            raise FileNotFoundError(f"Parent artifact missing: {ap}")
        actual = compute_sha256(ap)
        recorded = manifest.get("artifact_checksums", {}).get(key)
        if recorded is not None and recorded != actual:
            raise ValueError(f"Checksum mismatch for parent artifact {ap}: expected {recorded}, got {actual}")

    return manifest


def _load_occurrence_artifacts(occurrence_dir: Path) -> dict:
    """Load occurrence artifacts and OOF predictions for CRC."""
    m_tz = np.load(occurrence_dir / "M_target_zero.npz")["M_target_zero"]
    m_gan = np.load(occurrence_dir / "M_gan.npz")["M_gan"]
    posterior = np.load(occurrence_dir / "occurrence_posterior.npz")
    p_nonzero = posterior["p_nonzero"]
    target_rows = posterior["target_row_indices"]
    target_vars = posterior["target_variant_indices"]
    thresholds = pd.read_csv(occurrence_dir.parent / "occurrence" / "occurrence_thresholds.csv")

    return {
        "M_target_zero": m_tz,
        "M_gan": m_gan,
        "p_nonzero": p_nonzero,
        "target_rows": target_rows,
        "target_vars": target_vars,
        "thresholds": thresholds,
    }


def _compute_crc_bias(
    oof_path: Path,
    variant_names: list[str],
    alpha: float = 0.05,
    b_max: float = 5.0,
    min_positive: int = 10,
) -> dict[str, float]:
    """Compute CRC bias per variant from OOF predictions."""
    from .gating import fit_temperature_and_crc

    oof_df = pd.read_csv(oof_path)
    result = fit_temperature_and_crc(
        oof_df,
        y_true_col="y_true",
        p_calibrated_col="p_calibrated",
        variant_col="variant",
        variant_names=sorted(oof_df["variant"].unique()),
        alpha=0.05,
        b_max=5.0,
        min_positive=10,
    )
    return result["bias_per_variant"]


def _compute_hard_lock_thresholds(
    w_gan: np.ndarray,
    variant_names: list[str],
    cap_zero_lock_fraction: float = 0.9,
) -> dict[str, float]:
    """Compute per-variant hard-lock thresholds based on zero-lock fraction cap."""
    tau_hard = {}
    # We need to know which cells are target cells for each variant
    # This requires target mask - handled in fuse_initializations
    return {}  # Will be computed per-variant in fuse_initializations


def _compute_variant_zero_lock_cap(
    df_raw: pd.DataFrame,
    variant_cols: list[str],
    cap_fraction: float = 0.9,
) -> dict[str, int]:
    """Compute max number of zero-locked cells per variant based on observed zero prevalence."""
    caps = {}
    for j, variant in enumerate(variant_cols):
        observed_mask = df_raw[variant].notna().to_numpy()
        zero_obs = (df_raw[variant].fillna(0).to_numpy()[observed_mask] == 0).sum()
        total_obs = observed_mask.sum()
        zero_prev = zero_obs / total_obs if total_obs > 0 else 0.0
        caps[variant] = int(np.ceil(cap_fraction * zero_prev * total_obs))
    return caps


def fuse_initializations(
    raw_path: str | Path,
    tsagris_path: str | Path,
    occurrence_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path = "configs/data.yaml",
    emit_grid: bool = True,
) -> FusionResult:
    project_root = Path(__file__).resolve().parents[3]
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    occurrence_dir = Path(occurrence_dir)
    parent_manifest = _validate_parent_manifest(occurrence_dir)

    _require_artifact(raw_path, "raw data")
    _require_artifact(tsagris_path, "tsagris dataset")

    data_config = load_config(config_path)
    variant_cols = data_config.variant_components
    n_vars = len(variant_cols)

    df_raw = load_covariants(raw_path, data_config)
    df_tsagris = pd.read_csv(tsagris_path)
    df_tsagris[data_config.date_col] = pd.to_datetime(df_tsagris[data_config.date_col])

    merged = df_raw.merge(
        df_tsagris[[data_config.location_col, data_config.date_col] + variant_cols],
        on=[data_config.location_col, data_config.date_col],
        how="left",
        suffixes=("", "_tsagris"),
    )
    if len(merged) != len(df_raw):
        raise ValueError("Key-based join changed row count")

    tsagris_aligned = np.zeros((len(df_raw), n_vars), dtype=np.int64)
    for j, col in enumerate(variant_cols):
        vals = merged[f"{col}_tsagris"].to_numpy()
        tsagris_aligned[:, j] = np.where(np.isnan(vals), 0, vals).astype(np.int64)

    M_observed = df_raw[variant_cols].notna().to_numpy().astype(np.uint8)
    M_target = df_raw[variant_cols].isna().to_numpy().astype(np.uint8)

    occ_artifacts = _load_occurrence_artifacts(occurrence_dir)
    M_target_zero_occ = occ_artifacts["M_target_zero"]
    M_gan_occ = occ_artifacts["M_gan"]
    p_nonzero = occ_artifacts["p_nonzero"]
    target_rows = occ_artifacts["target_rows"]
    target_vars = occ_artifacts["target_vars"]

    if M_target_zero_occ.shape != M_target.shape or M_gan_occ.shape != M_target.shape:
        raise ValueError("Occurrence mask shapes do not match raw data")
    M_target_zero_occ = M_target_zero_occ.astype(np.uint8)
    M_gan_occ = M_gan_occ.astype(np.uint8)

    if np.any(M_target_zero_occ.astype(bool) & M_observed.astype(bool)):
        raise ValueError("M_target_zero overlaps observed cells")
    if not np.array_equal(
        (M_target_zero_occ.astype(bool) | M_gan_occ.astype(bool)), M_target.astype(bool)
    ):
        raise ValueError("M_target_zero | M_gan != M_target")

    raw_counts = df_raw[variant_cols].fillna(0).to_numpy().astype(np.int64)
    total_seq = df_raw[data_config.total_sequence_col].to_numpy(dtype=np.int64)

    obs_bool = M_observed.astype(bool)
    zero_bool = M_target_zero_occ.astype(bool)
    gan_bool = M_gan_occ.astype(bool)

    # Load OOF predictions for CRC calibration
    oof_path = Path("artifacts/occurrence/occurrence_oof_predictions.csv")
    oof_df = pd.read_csv(oof_path)
    variant_names = variant_cols

    # Fit temperature and CRC
    crc_result = fit_temperature_and_crc(
        oof_df=pd.read_csv("artifacts/occurrence/occurrence_oof_predictions.csv"),
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

    # Load posterior and reshape p_nonzero to (n_rows, n_vars)
    posterior = np.load("artifacts/occurrence/occurrence_posterior.npz")
    p_nonzero_flat = posterior["p_nonzero"]
    target_rows = posterior["target_row_indices"]
    target_vars = posterior["target_variant_indices"]

    n_rows, n_vars = M_target.shape
    p_nonzero = np.zeros((n_rows, n_vars), dtype=np.float64)
    p_nonzero[target_rows, target_vars] = p_nonzero_flat

    total_seq = df_raw[data_config.total_sequence_col].to_numpy(dtype=np.int64)

    # Recalibrate posterior probabilities with temperature scaling
    p_cal = p_nonzero.clip(1e-12, 1 - 1e-12)
    logits = logit(p_cal)
    logits_cal = logits / T
    p_recal = expit(logits_cal)

    # Compute soft fusion weights: w_gan = sigma(logit(p) - b)
    # p is the recalibrated probability of non-zero
    # w_gan = sigma(logit(p_recal) - b)
    w_gan_per_variant = {}
    for variant in variant_cols:
        b = crc_result["bias_per_variant"].get(variant, 0.0)
        w_gan_per_variant[variant] = 1.0  # placeholder, will compute per-cell

    # Build target mask and indices
    target_mask = M_target.astype(bool)
    n_rows, n_vars = M_target.shape

    # Compute per-variant w_gan per target cell
    w_gan = np.zeros((n_rows, n_vars), dtype=np.float32)
    for j, variant in enumerate(variant_cols):
        b = crc_result["bias_per_variant"].get(variant, 0.0)
        # For each row, w_gan = sigma(logit(p) - b)
        # p is recalibrated probability
        # We only compute for target cells (where M_target == 1)
        target_mask_var = M_target[:, j].astype(bool)
        p_var = p_recal[:, j]
        logits_p = logit(p_var[target_mask_var].clip(1e-12, 1 - 1e-12))
        w = expit(logits_p - crc_result["bias_per_variant"].get(variant, 0.0))
        w_gan[target_mask_var, j] = w

    # Compute variant zero-lock cap (0.9 * zero prevalence in observed)
    zero_lock_caps = _compute_variant_zero_lock_cap(df_raw, variant_cols, cap_fraction=0.9)

    # Hard-lock threshold per variant: quantile of w_gan on target cells
    tau_hard_per_variant = {}
    for j, variant in enumerate(variant_cols):
        target_mask_var = M_target[:, j].astype(bool)
        if not target_mask_var.any():
            tau_hard_per_variant[variant] = 0.0
            continue
        w_var = w_gan[target_mask_var, j]
        cap = zero_lock_caps[variant]
        # Hard-lock fraction should not exceed cap_fraction * zero_prevalence_observed
        # So tau_hard is the quantile such that fraction(w < tau) = cap / n_target
        n_target = int(target_mask_var.sum())
        max_hard_lock = min(int(cap), n_target)
        if max_hard_lock > 0:
            tau_hard = float(np.quantile(w_var, max_hard_lock / n_target))
        else:
            tau_hard = 0.0
        tau_hard_per_variant[variant] = tau_hard

    # Apply soft fusion with hard-lock
    w_gan_flat, M_hard_lock, M_soft_fusion = apply_soft_fusion(
        p_nonzero_targets=p_recal,
        target_mask=M_target,
        variant_names=variant_cols,
        w_gan_per_variant={v: 1.0 for v in variant_cols},  # w_gan = 1/p * ... wait
        tau_hard_per_variant=tau_hard_per_variant,
    )

    # Wait - w_gan should be per-cell, not per-variant constant
    # Let me recompute properly: w_gan is per cell = expit(logit(p) - b)
    w_gan = np.zeros((n_rows, n_vars), dtype=np.float32)
    for j, variant in enumerate(variant_cols):
        b = crc_result["bias_per_variant"].get(variant, 0.0)
        target_mask_var = M_target[:, j].astype(bool)
        p_var = p_recal[:, j]
        logits_p = logit(p_var[target_mask_var].clip(1e-12, 1 - 1e-12))
        w = expit(logits_p - crc_result["bias_per_variant"].get(variant, 0.0))
        w_gan[target_mask_var, j] = w

    # Hard-lock thresholds
    tau_hard_per_variant = {}
    for j, variant in enumerate(variant_cols):
        target_mask_var = M_target[:, j].astype(bool)
        if not target_mask_var.any():
            tau_hard_per_variant[variant] = 0.0
            continue
        w_var = w_gan[target_mask_var, j]
        cap = zero_lock_caps[variant]
        n_target = int(target_mask_var.sum())
        max_hard_lock = min(int(cap), n_target)
        if max_hard_lock > 0:
            tau_hard = float(np.quantile(w_var, max_hard_lock / n_target))
        else:
            tau_hard = 0.0
        tau_hard_per_variant[variant] = tau_hard

    # Apply soft fusion
    w_gan_fused, M_hard_lock, M_soft_fusion = apply_soft_fusion(
        p_nonzero_targets=p_recal,
        target_mask=M_target,
        variant_names=variant_cols,
        w_gan_per_variant={v: 1.0 for v in variant_cols},  # Not used, we pass tau_hard
        tau_hard_per_variant=tau_hard_per_variant,
    )

    # Wait - apply_soft_fusion expects w_gan_per_variant as constant per variant
    # But we have per-cell w_gan. Let me redo this properly.
    # The apply_soft_fusion function takes w_gan_per_variant (per-variant constant)
    # But we need per-cell weights. Let me redo the fusion logic.

    # Actually, let me rewrite the fusion logic directly here without calling apply_soft_fusion
    # We'll do the fusion inline

    # Hard-lock: w_gan < tau_hard
    M_hard_lock = np.zeros((n_rows, n_vars), dtype=np.uint8)
    M_soft_fusion = np.zeros((n_rows, n_vars), dtype=np.uint8)
    w_gan_final = np.zeros((n_rows, n_vars), dtype=np.float32)

    for j, variant in enumerate(variant_cols):
        target_mask_var = M_target[:, j].astype(bool)
        if not target_mask_var.any():
            continue
        w_var = w_gan[target_mask_var, j]
        tau_hard = tau_hard_per_variant.get(variant, 0.0)

        hard_sel = w_var < tau_hard
        soft_sel = ~hard_sel

        # Hard-locked cells: set w_gan = 0
        w_gan_final = np.zeros_like(w_var)
        w_gan_final[soft_sel] = w_var[soft_sel]
        w_gan[target_mask_var, j] = w_gan_final

        # Mark hard-locked cells
        hard_lock_indices = np.where(target_mask_var)[0][hard_sel]
        M_hard_lock[hard_lock_indices, j] = 1

        # Mark soft fusion cells
        soft_indices = np.where(target_mask_var)[0][soft_sel]
        M_soft_fusion[soft_indices, j] = 1

    # M_target_zero = M_target_zero_occ | M_hard_lock
    M_target_zero = (M_target_zero_occ.astype(bool) | M_hard_lock.astype(bool)).astype(np.uint8)
    M_gan = M_gan_occ & ~M_hard_lock.astype(bool)  # M_gan excludes hard-locked

    # Verify partition
    if np.any(M_target_zero.astype(bool) & M_observed.astype(bool)):
        raise ValueError("M_target_zero overlaps observed cells")
    if not np.array_equal(
        (M_target_zero.astype(bool) | M_gan.astype(bool)), M_target.astype(bool)
    ):
        raise ValueError("M_target_zero | M_gan != M_target")

    # Now we have soft fusion weights in w_gan
    # Magnitudes for soft-fusion cells come from Tsagris (warm_props)
    # For hard-locked cells, magnitude = 0
    # For observed cells, keep raw counts

    # Build fused counts
    fused_counts = np.zeros_like(raw_counts)
    fused_counts[obs_bool] = raw_counts[obs_bool]
    fused_counts[M_hard_lock.astype(bool)] = 0

    # For soft fusion cells, use weighted magnitude
    warm_props = np.zeros((len(df_raw), n_vars), dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        warm_props = tsagris_aligned.astype(np.float64) / np.maximum(
            total_seq[:, None], 1
        ).astype(np.float64)

    # For soft fusion cells, use weighted Tsagris proportions
    M_soft_bool = M_soft_fusion.astype(bool)
    soft_fused_props = warm_props.copy()
    soft_fused_props[M_soft_bool] = warm_props[M_soft_bool] * w_gan[M_soft_bool]
    soft_fused_props[~M_soft_bool] = 0

# Total proportions for target cells = soft_fused_props
    # For observed cells, use observed proportions (raw_counts / total_seq)
    # For hard-locked cells, proportion = 0
    # We need to compute final proportions per cell
    final_props = np.zeros_like(warm_props)
    
    # For observed cells: proportion = raw_counts / total_seq
    # Use proper broadcasting: total_seq is (n_rows,), broadcast to (n_rows, n_vars)
    observed_props = raw_counts / total_seq[:, np.newaxis]  # (n_rows, n_vars)
    final_props = np.where(obs_bool, observed_props, 0.0)
    
    final_props[M_hard_lock.astype(bool)] = 0
    final_props[M_soft_bool] = soft_fused_props[M_soft_bool]

    # Convert to counts
    final_counts = np.round(final_props * total_seq[:, np.newaxis]).astype(np.int64)

    # Integerize with closure using allocate_with_lock on the free cells
    # Free cells = soft fusion cells (M_soft_fusion)
    # Locked = observed + hard-locked
    free_mask = M_soft_fusion.astype(bool)
    locked_mask = obs_bool | M_hard_lock.astype(bool)
    locked_counts = final_counts.copy()

    final_counts = allocate_with_lock(
        final_props,
        total_seq,
        locked_mask,
        locked_counts
    )

    observed_preserved = bool(np.array_equal(final_counts[obs_bool], raw_counts[obs_bool]))
    zero_locked = bool(np.all(final_counts[M_hard_lock.astype(bool)] == 0))

    other = compute_other(final_counts, total_seq)
    closure_ok, n_viol = validate_closure(final_counts, total_seq, other)

    M_fixed = (obs_bool | M_target_zero_occ.astype(bool) | M_hard_lock.astype(bool)).astype(np.uint8)

    df_fused = df_raw.copy()
    for j, col in enumerate(variant_cols):
        df_fused[col] = final_counts[:, j]
    df_fused[data_config.other_col] = other

    fused_raw_path = output_dir / "dataset_0_fused_raw.csv"
    df_fused.to_csv(fused_raw_path, index=False)

    fused_grid_path = None
    if emit_grid:
        df_grid = _build_grid_output(
            df_raw, df_fused, final_counts, other, data_config, variant_cols
        )
        fused_grid_path = output_dir / "dataset_0_fused_grid.csv"
        df_grid.to_csv(fused_grid_path, index=False)

    m_fixed_path = output_dir / "M_fixed.npz"
    m_gan_path = output_dir / "M_gan.npz"
    np.savez_compressed(m_fixed_path, M_fixed=M_fixed)
    np.savez_compressed(m_gan_path, M_gan=M_gan)

    provenance = _build_provenance(
        df_raw, raw_counts, tsagris_aligned, final_counts,
        M_observed, M_target_zero, M_gan, variant_cols, data_config,
    )
    provenance_path = output_dir / "cell_provenance.parquet"
    try:
        provenance.to_parquet(provenance_path, index=False)
    except ImportError:
        provenance_path = output_dir / "cell_provenance.csv"
        provenance.to_csv(provenance_path, index=False)

    invariants_ok = closure_ok and observed_preserved and zero_locked

    manifest = create_run_manifest(
        run_id=f"fusion_{int(time.time())}",
        method="truth_table_fusion_v3_zpgf",
        params={
            "tsagris_path": str(tsagris_path),
            "occurrence_dir": str(occurrence_dir),
            "parent_manifest_run_id": parent_manifest.get("run_id"),
            "temperature": float(T),
            "crc_bias": crc_result["bias_per_variant"],
            "cap_zero_lock_fraction": 0.9,
        },
        metrics={
            "n_fixed_cells": int(M_fixed.sum()),
            "n_gan_cells": int(M_gan.sum()),
            "n_target_zero": int(M_target_zero.sum()),
            "n_hard_locked": int(M_hard_lock.sum()),
            "n_soft_fusion": int(M_soft_fusion.sum()),
            "closure_violations": n_viol,
            "observed_preserved": observed_preserved,
            "confident_zero_locked": zero_locked,
        },
        data_path=raw_path,
        config_paths=[config_path],
        seeds=[parent_manifest.get("seed", 42)],
        invariants_ok=invariants_ok,
        output_dir=output_dir / "manifests",
    )

    return FusionResult(
        fused_raw_path=str(fused_raw_path),
        fused_grid_path=str(fused_grid_path) if fused_grid_path else None,
        m_fixed_path=str(m_fixed_path),
        m_gan_path=str(m_gan_path),
        provenance_path=str(provenance_path),
        manifest=manifest,
        invariants_ok=invariants_ok,
    )


def _build_grid_output(
    df_raw: pd.DataFrame,
    df_fused: pd.DataFrame,
    final_counts: np.ndarray,
    other: np.ndarray,
    config,
    variant_cols: list[str],
) -> pd.DataFrame:
    location_col = config.location_col
    date_col = config.date_col
    rows = []
    for loc in sorted(df_raw[location_col].unique()):
        loc_df = df_raw[df_raw[location_col] == loc].sort_values(date_col)
        if len(loc_df) == 0:
            continue
        start = loc_df[date_col].min()
        end = loc_df[date_col].max()
        grid = pd.date_range(start=start, end=end, freq=f"{config.freq_days}D")
        raw_dates = set(pd.to_datetime(loc_df[date_col]))
        fused_loc = df_fused[df_fused[location_col] == loc].sort_values(date_col)
        fused_by_date = {
            pd.to_datetime(r[date_col]): r for _, r in fused_loc.iterrows()
        }
        for t in grid:
            is_raw = t in raw_dates
            row = {location_col: loc, date_col: t}
            if is_raw and t in fused_by_date:
                src = fused_by_date[t]
                row[config.total_sequence_col] = int(src[config.total_sequence_col])
                for col in variant_cols:
                    row[col] = int(src[col])
                row[config.other_col] = int(src[config.other_col])
                row["M_row"] = 1
                row["M_padding"] = 0
            else:
                row[config.total_sequence_col] = np.nan
                for col in variant_cols:
                    row[col] = np.nan
                row[config.other_col] = np.nan
                row["M_row"] = 0
                row["M_padding"] = 1
            rows.append(row)
    return pd.DataFrame(rows)


def _build_provenance(
    df_raw: pd.DataFrame,
    raw_counts: np.ndarray,
    tsagris_counts: np.ndarray,
    fused_counts: np.ndarray,
    M_observed: np.ndarray,
    M_target_zero: np.ndarray,
    M_gan: np.ndarray,
    variant_cols: list[str],
    config,
) -> pd.DataFrame:
    n_rows, n_vars = M_observed.shape
    obs_bool = M_observed.astype(bool)
    zero_bool = M_target_zero.astype(bool)
    gan_bool = M_gan.astype(bool)

    sources = np.empty((n_rows, n_vars), dtype=object)
    sources[obs_bool] = "RAW_OBSERVED"
    sources[zero_bool] = "OCCURRENCE_CONFIDENT_ZERO"
    sources[gan_bool] = "SOFT_FUSION_GAN"

    row_idx, var_idx = np.indices((n_rows, n_vars))
    return pd.DataFrame({
        "location": df_raw[config.location_col].to_numpy()[row_idx.ravel()],
        "date": df_raw[config.date_col].astype(str).to_numpy()[row_idx.ravel()],
        "variant": np.array(variant_cols)[var_idx.ravel()],
        "source": sources.ravel(),
        "raw_value": raw_counts.ravel().astype(np.float64),
        "tsagris_value": tsagris_counts.ravel().astype(np.float64),
        "fused_value": fused_counts.ravel().astype(np.float64),
    })


@dataclass
class FusionResult:
    fused_raw_path: str
    fused_grid_path: str | None
    m_fixed_path: str
    m_gan_path: str
    provenance_path: str
    manifest: dict
    invariants_ok: bool