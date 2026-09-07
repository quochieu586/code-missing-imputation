"""Occurrence gate artifact persistence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.lineage import compute_sha256


def save_occurrence_artifacts(
    output_dir: str | Path,
    oof_predictions: pd.DataFrame,
    p_nonzero_targets: np.ndarray,
    target_coordinates: np.ndarray,
    M_target_zero: np.ndarray,
    M_gan: np.ndarray,
    metrics_rows: list[dict],
    threshold_rows: list[dict],
    feature_schema: dict,
    manifest: dict,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    oof_path = output_dir / "occurrence_oof_predictions.parquet"
    try:
        oof_predictions.to_parquet(oof_path, index=False)
    except ImportError:
        oof_path = output_dir / "occurrence_oof_predictions.csv"
        oof_predictions.to_csv(oof_path, index=False)
    paths["oof_predictions"] = str(oof_path)

    posterior_path = output_dir / "occurrence_posterior.npz"
    np.savez_compressed(
        posterior_path,
        p_nonzero=p_nonzero_targets,
        target_row_indices=target_coordinates[:, 0],
        target_variant_indices=target_coordinates[:, 1],
    )
    paths["posterior"] = str(posterior_path)

    m_target_zero_path = output_dir / "M_target_zero.npz"
    np.savez_compressed(m_target_zero_path, M_target_zero=M_target_zero)
    paths["M_target_zero"] = str(m_target_zero_path)

    m_gan_path = output_dir / "M_gan.npz"
    np.savez_compressed(m_gan_path, M_gan=M_gan)
    paths["M_gan"] = str(m_gan_path)

    metrics_path = output_dir / "occurrence_metrics.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    paths["metrics"] = str(metrics_path)

    thresholds_path = output_dir / "occurrence_thresholds.csv"
    pd.DataFrame(threshold_rows).to_csv(thresholds_path, index=False)
    paths["thresholds"] = str(thresholds_path)

    schema_path = output_dir / "occurrence_feature_schema.json"
    with schema_path.open("w", encoding="utf-8") as f:
        json.dump(feature_schema, f, indent=2, default=str)
    paths["feature_schema"] = str(schema_path)

    manifest_path = output_dir / "occurrence_model_manifest.json"
    artifact_checksums = {}
    for key, p in paths.items():
        if Path(p).exists():
            artifact_checksums[key] = compute_sha256(p)
    manifest["artifact_checksums"] = artifact_checksums
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    paths["manifest"] = str(manifest_path)

    return paths
