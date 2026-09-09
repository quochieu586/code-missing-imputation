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
import yaml

from ..data.closure import allocate_with_lock, validate_closure
from ..data.lineage import compute_sha256
from ..data.loader import load_config, load_covariants
from ..occurrence.gating import compute_soft_fusion, expit, fit_temperature_and_crc, logit
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
    """Load the occurrence stage outputs consumed by fusion."""
    posterior = np.load(occurrence_dir / "occurrence_posterior.npz")
    return {
        "M_target_zero": np.load(occurrence_dir / "M_target_zero.npz")["M_target_zero"],
        "M_gan": np.load(occurrence_dir / "M_gan.npz")["M_gan"],
        "p_nonzero": posterior["p_nonzero"],
        "target_rows": posterior["target_row_indices"],
        "target_vars": posterior["target_variant_indices"],
        "oof_path": occurrence_dir / "occurrence_oof_predictions.csv",
    }


def _load_zpgf_config(config_path: str | Path) -> dict:
    """Load ZPGF thresholds. Falls back to the documented defaults when absent."""
    defaults = {
        "crc_alpha": 0.05,
        "crc_b_max": 5.0,
        "crc_min_positive": 10,
        "tau_hard_abs": 0.05,
        "cap_fraction": 0.9,
    }
    path = Path(config_path)
    if not path.exists():
        return defaults
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    zpgf = raw.get("zpgf", {})
    calib = zpgf.get("calibration", {})
    hard = zpgf.get("hard_lock", {})
    return {
        "crc_alpha": float(calib.get("crc_alpha", defaults["crc_alpha"])),
        "crc_b_max": float(calib.get("crc_b_max", defaults["crc_b_max"])),
        "crc_min_positive": int(calib.get("crc_min_positive", defaults["crc_min_positive"])),
        "tau_hard_abs": float(hard.get("tau_hard_abs", defaults["tau_hard_abs"])),
        "cap_fraction": float(hard.get("cap_fraction", defaults["cap_fraction"])),
    }


def _compute_variant_zero_lock_cap(
    df_raw: pd.DataFrame,
    variant_cols: list[str],
    cap_fraction: float = 0.9,
) -> dict[str, float]:
    """Max FRACTION of a variant's target cells that may be hard-locked to zero.

    Plan S18.3 caps the locked fraction at cap_fraction x observed zero prevalence.
    Returning a fraction (not a cell count derived from the observed cells) is what
    keeps the cap meaningful: the earlier count-based form regularly exceeded the
    number of target cells, which set the lock quantile to 1.0 and locked every
    cell of the variant.
    """
    caps: dict[str, float] = {}
    for variant in variant_cols:
        observed_mask = df_raw[variant].notna().to_numpy()
        total_obs = int(observed_mask.sum())
        if total_obs == 0:
            caps[variant] = 0.0
            continue
        zero_obs = int((df_raw[variant].to_numpy()[observed_mask] == 0).sum())
        zero_prev = zero_obs / total_obs
        caps[variant] = float(np.clip(cap_fraction * zero_prev, 0.0, 1.0))
    return caps


def fuse_initializations(
    raw_path: str | Path,
    tsagris_path: str | Path,
    occurrence_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path = "configs/data.yaml",
    emit_grid: bool = True,
    zpgf_config_path: str | Path = "configs/fusion/zpgf.yaml",
) -> FusionResult:
    project_root = Path(__file__).resolve().parents[3]
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    zpgf_config_path = Path(zpgf_config_path)
    if not zpgf_config_path.is_absolute():
        zpgf_config_path = project_root / zpgf_config_path
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
    M_target_zero_occ = occ_artifacts["M_target_zero"].astype(np.uint8)
    M_gan_occ = occ_artifacts["M_gan"].astype(np.uint8)

    if M_target_zero_occ.shape != M_target.shape or M_gan_occ.shape != M_target.shape:
        raise ValueError("Occurrence mask shapes do not match raw data")
    if np.any(M_target_zero_occ.astype(bool) & M_observed.astype(bool)):
        raise ValueError("Parent M_target_zero overlaps observed cells")
    if not np.array_equal(
        (M_target_zero_occ.astype(bool) | M_gan_occ.astype(bool)), M_target.astype(bool)
    ):
        raise ValueError("Parent M_target_zero | M_gan != M_target")

    raw_counts = df_raw[variant_cols].fillna(0).to_numpy().astype(np.int64)
    total_seq = df_raw[data_config.total_sequence_col].to_numpy(dtype=np.int64)
    obs_bool = M_observed.astype(bool)
    n_rows = len(df_raw)

    zpgf_cfg = _load_zpgf_config(zpgf_config_path)

    # --- Post-hoc recalibration (plan S19.2 steps 1-2) -----------------------
    # Temperature scaling on the OOF logits, then the CRC bias per variant. Both
    # are fitted on held-out occurrence predictions, never on the target cells.
    oof_path = occ_artifacts["oof_path"]
    if not oof_path.exists():
        raise FileNotFoundError(f"OOF predictions required for CRC calibration: {oof_path}")
    crc_result = fit_temperature_and_crc(
        oof_df=pd.read_csv(oof_path),
        y_true_col="y_true",
        p_calibrated_col="p_calibrated",
        variant_col="variant",
        variant_names=variant_cols,
        alpha=zpgf_cfg["crc_alpha"],
        b_max=zpgf_cfg["crc_b_max"],
        min_positive=zpgf_cfg["crc_min_positive"],
    )
    T = float(crc_result["temperature"])

    p_nonzero = np.zeros((n_rows, n_vars), dtype=np.float64)
    p_nonzero[occ_artifacts["target_rows"], occ_artifacts["target_vars"]] = occ_artifacts[
        "p_nonzero"
    ]
    p_recal = expit(logit(np.clip(p_nonzero, 1e-12, 1 - 1e-12)) / T)

    # --- ZPGF gate (plan S18.3 / S19.2 steps 3-4) ---------------------------
    # The hard-lock decision lives here and here only. The occurrence stage's own
    # binary gate (M_gan_occ) is a diagnostic of the superseded scheme from S5.4;
    # intersecting the two gates would require a cell to clear both and starves
    # the GAN, so it is deliberately not used to restrict M_gan.
    zero_lock_caps = _compute_variant_zero_lock_cap(
        df_raw, variant_cols, cap_fraction=zpgf_cfg["cap_fraction"]
    )

    # Plan S5.4: an occurrence model that failed its release gate must not lock
    # anything to zero. Its probabilities can still shape the soft weight, which
    # is advisory and applied after the GAN, but a hard lock destroys the cell.
    parent_released = bool(parent_manifest.get("released", True))
    tau_hard_abs = zpgf_cfg["tau_hard_abs"] if parent_released else 0.0
    if not parent_released:
        print(
            "WARNING: parent occurrence gate was not released "
            f"({parent_manifest.get('release_reason', '')!r}); "
            "ZPGF hard-lock disabled, every target cell goes to the GAN"
        )

    w_gan, M_hard_lock, M_soft_fusion, gate_diagnostics = compute_soft_fusion(
        p_nonzero=p_recal,
        target_mask=M_target,
        variant_names=variant_cols,
        bias_per_variant=crc_result["bias_per_variant"],
        zero_lock_cap_fraction=zero_lock_caps,
        tau_hard_abs=tau_hard_abs,
    )

    M_target_zero = M_hard_lock
    M_gan = M_soft_fusion
    M_fixed = (obs_bool | M_target_zero.astype(bool)).astype(np.uint8)

    if np.any(M_target_zero.astype(bool) & obs_bool):
        raise ValueError("M_target_zero overlaps observed cells")
    if np.any(M_target_zero.astype(bool) & M_gan.astype(bool)):
        raise ValueError("M_target_zero and M_gan overlap")
    if not np.array_equal(
        (M_target_zero.astype(bool) | M_gan.astype(bool)), M_target.astype(bool)
    ):
        raise ValueError("M_target_zero | M_gan != M_target")
    if np.any(M_fixed.astype(bool) & M_gan.astype(bool)):
        raise ValueError("M_fixed and M_gan overlap")

    # --- Truth table (plan S6) ----------------------------------------------
    #   observed        -> raw count, locked
    #   hard-locked     -> 0, locked
    #   soft fusion     -> Tsagris warm-start, editable by the GAN
    # w_gan is NOT applied to the warm-start here: per S18.6 it multiplies the
    # GAN's own magnitude in gan/postprocess.py, after the GAN has produced it.
    # The residual "other" rides along as a free column so the leftover budget
    # lands there instead of being forced onto the variant cells.
    soft_bool = M_soft_fusion.astype(bool)
    lock_bool = obs_bool | M_hard_lock.astype(bool)

    tsagris_other = np.maximum(total_seq - tsagris_aligned.sum(axis=1), 0)
    denom = np.maximum(total_seq, 1)[:, None].astype(np.float64)

    props = np.zeros((n_rows, n_vars + 1), dtype=np.float64)
    props[:, :n_vars] = np.where(obs_bool, raw_counts / denom, 0.0)
    props[:, :n_vars][soft_bool] = (tsagris_aligned / denom)[soft_bool]
    props[:, n_vars] = tsagris_other / denom[:, 0]

    lock_mask = np.zeros((n_rows, n_vars + 1), dtype=bool)
    lock_mask[:, :n_vars] = lock_bool  # "other" is never locked: it absorbs the residual
    locked_counts = np.zeros((n_rows, n_vars + 1), dtype=np.int64)
    locked_counts[:, :n_vars] = np.where(obs_bool, raw_counts, 0)

    allocated = allocate_with_lock(props, total_seq, lock_mask, locked_counts)
    final_counts = allocated[:, :n_vars]
    other = allocated[:, n_vars]

    observed_preserved = bool(np.array_equal(final_counts[obs_bool], raw_counts[obs_bool]))
    # Every cell the gate locked to zero must actually be zero, not just the ones
    # from the current hard-lock pass.
    zero_locked = bool(np.all(final_counts[M_target_zero.astype(bool)] == 0))
    closure_ok, n_viol = validate_closure(final_counts, total_seq, other)

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
    w_gan_path = output_dir / "w_gan.npz"
    np.savez_compressed(m_fixed_path, M_fixed=M_fixed)
    np.savez_compressed(m_gan_path, M_gan=M_gan)
    # Consumed by gan/postprocess.py, which applies y_hat = w_gan * m.
    np.savez_compressed(w_gan_path, w_gan=w_gan)

    provenance = _build_provenance(
        df_raw, raw_counts, tsagris_aligned, final_counts,
        M_observed, M_target_zero, M_gan, variant_cols, data_config,
    )
    provenance_path = output_dir / "cell_provenance.parquet"
    provenance.to_parquet(provenance_path, index=False)
    # A CSV from an earlier fallback run would otherwise sit next to the fresh
    # parquet and be read as if it were current.
    stale_csv = output_dir / "cell_provenance.csv"
    if stale_csv.exists():
        stale_csv.unlink()

    gate_summary_path = output_dir / "zpgf_gate_summary.csv"
    pd.DataFrame(
        [{"variant": v, **d} for v, d in gate_diagnostics.items()]
    ).to_csv(gate_summary_path, index=False)

    invariants_ok = closure_ok and observed_preserved and zero_locked

    manifest = create_run_manifest(
        run_id=f"fusion_{int(time.time())}",
        method="truth_table_fusion_v3_zpgf",
        params={
            "tsagris_path": str(tsagris_path),
            "occurrence_dir": str(occurrence_dir),
            "parent_manifest_run_id": parent_manifest.get("run_id"),
            "temperature": T,
            "crc_bias": crc_result["bias_per_variant"],
            "zpgf_config": str(zpgf_config_path),
            "parent_occurrence_released": parent_released,
            "tau_hard_abs_effective": tau_hard_abs,
            **zpgf_cfg,
        },
        metrics={
            "n_fixed_cells": int(M_fixed.sum()),
            "n_gan_cells": int(M_gan.sum()),
            "n_target_zero": int(M_target_zero.sum()),
            "n_hard_locked": int(M_hard_lock.sum()),
            "n_soft_fusion": int(M_soft_fusion.sum()),
            "n_gan_cells_parent_occurrence": int(M_gan_occ.sum()),
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
        w_gan_path=str(w_gan_path),
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

    # Names follow the plan S6 truth table. The third row is the Tsagris
    # warm-start that the GAN is allowed to refine; ZPGF only decides which
    # cells land there, it does not change where the value comes from.
    sources = np.empty((n_rows, n_vars), dtype=object)
    sources[obs_bool] = "RAW_OBSERVED"
    sources[zero_bool] = "OCCURRENCE_CONFIDENT_ZERO"
    sources[gan_bool] = "TSAGRIS_WARM_START"

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
    w_gan_path: str
    provenance_path: str
    manifest: dict
    invariants_ok: bool