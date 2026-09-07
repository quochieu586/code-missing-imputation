from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _file_checksum(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def save_variant_support_report(
    supports: list,
    path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [s.to_dict() for s in supports]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def save_model_routing_report(
    routings: list,
    path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in routings:
        d = r.to_dict()
        gm = d.pop("gate_metrics", {})
        for k, v in gm.items():
            d[f"gate_{k}"] = v
        rows.append(d)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def save_zero_state_metrics(
    metrics_per_variant: dict[str, dict],
    path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant, m in metrics_per_variant.items():
        row = {"variant_name": variant}
        for k, v in m.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    row[f"{k}_{k2}"] = v2
            else:
                row[k] = v
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def save_design_schema(schema: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(schema, f, indent=2)
    return path


def save_model_manifest(
    run_id: str,
    variant_routings: list,
    fit_results: dict,
    quality_flags: dict[str, bool],
    artifact_paths: list[str | Path],
    config_params: dict,
    seeds: list[int],
    output_path: str | Path,
    git_commit: str | None = None,
) -> dict:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checksums = {}
    for p in artifact_paths:
        p = Path(p)
        if p.exists():
            checksums[p.name] = _file_checksum(p)

    manifest = {
        "run_id": run_id,
        "method": "zinb_zero_state_v2",
        "git_commit": git_commit or "unknown",
        "config_params": config_params,
        "seeds": seeds,
        "quality_flags": quality_flags,
        "routing": [r.to_dict() for r in variant_routings],
        "fit_results": {
            variant: {
                "model_type": fr.model_type.value if fr is not None else None,
                "converged": bool(fr.converged) if fr is not None else False,
                "n_iterations": int(fr.n_iterations) if fr is not None and fr.n_iterations is not None else None,
                "log_likelihood": float(fr.log_likelihood) if fr is not None and fr.log_likelihood is not None else None,
                "l1_penalty_used": float(fr.l1_penalty_used) if fr is not None else None,
                "fallback_reason": fr.fallback_reason if fr is not None else None,
                "n_observations": int(fr.n_observations) if fr is not None else 0,
            }
            for variant, fr in fit_results.items()
        },
        "checksums": checksums,
    }

    with output_path.open("w") as f:
        json.dump(manifest, f, indent=2, default=str)

    return manifest


def save_oof_predictions(
    oof_records: list[dict],
    path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(oof_records) > 0:
        df = pd.DataFrame(oof_records)
    else:
        df = pd.DataFrame(
            columns=["variant_name", "recipe", "fold", "cell_index", "y_true", "p_nonzero"]
        )
    if path.suffix == ".parquet":
        try:
            df.to_parquet(path, index=False)
        except ImportError:
            csv_path = path.with_suffix(".csv")
            df.to_csv(csv_path, index=False)
            print(f"  WARNING: pyarrow not available; saved OOF predictions to {csv_path.name} instead of .parquet")
            return csv_path
    else:
        df.to_csv(path, index=False)
    return path
