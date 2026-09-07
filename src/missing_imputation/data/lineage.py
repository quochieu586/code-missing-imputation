"""Data lineage: raw checksum freeze and verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


def compute_sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_lineage(config_path: str | Path = "configs/data.yaml") -> dict:
    with Path(config_path).open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    lineage = cfg.get("lineage", {})
    if not lineage:
        raise ValueError(f"No 'lineage' section in {config_path}")
    return lineage


def verify_raw_data(
    data_path: str | Path,
    config_path: str | Path = "configs/data.yaml",
) -> dict:
    lineage = load_lineage(config_path)
    actual_checksum = compute_sha256(data_path)
    expected = lineage.get("raw_checksum")
    result = {
        "data_path": str(data_path),
        "expected_checksum": expected,
        "actual_checksum": actual_checksum,
        "checksum_ok": expected is None or actual_checksum == expected,
    }
    if not result["checksum_ok"]:
        raise ValueError(
            f"Raw data checksum mismatch: expected {expected}, got {actual_checksum}"
        )
    return result


def freeze_lineage(
    data_path: str | Path,
    config_path: str | Path = "configs/data.yaml",
) -> dict:
    lineage = load_lineage(config_path)
    return {
        "raw_data_path": str(data_path),
        "raw_checksum": compute_sha256(data_path),
        "expected_m_target_count": lineage.get("expected_m_target_count"),
        "variant_order": lineage.get("variant_order", []),
    }


def save_lineage_record(record: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)


def load_lineage_record(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)
