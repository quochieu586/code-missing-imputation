"""Manifest and provenance tracking."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def compute_file_checksum(file_path: str | Path) -> str:
    """Compute SHA-256 checksum of a file."""
    sha256 = hashlib.sha256()
    with Path(file_path).open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_git_commit() -> str:
    """Get current Git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return "unknown"


def create_run_manifest(
    run_id: str,
    method: str,
    params: dict,
    metrics: dict,
    data_path: str | Path,
    config_paths: list[str | Path],
    seeds: list[int],
    invariants_ok: bool,
    output_dir: str | Path,
) -> dict:
    """Create a run manifest with full provenance."""
    manifest = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "params": params,
        "metrics": metrics,
        "data_checksum": compute_file_checksum(data_path),
        "config_checksums": {
            str(p): compute_file_checksum(p) for p in config_paths
        },
        "git_commit": get_git_commit(),
        "seeds": seeds,
        "invariants_ok": invariants_ok,
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
    }

    output_path = Path(output_dir) / f"run_{run_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(manifest, f, indent=2, default=str)

    return manifest


def load_manifest(manifest_path: str | Path) -> dict:
    """Load a manifest from JSON file."""
    with Path(manifest_path).open() as f:
        return json.load(f)


def verify_manifest_chain(
    manifest_paths: list[str | Path],
) -> tuple[bool, list[str]]:
    """Verify that a chain of manifests is consistent."""
    issues = []
    for i, path in enumerate(manifest_paths):
        if not Path(path).exists():
            issues.append(f"Manifest {i} not found: {path}")
            continue

        manifest = load_manifest(path)
        required_fields = ["run_id", "timestamp", "method", "metrics", "seeds"]
        for field in required_fields:
            if field not in manifest:
                issues.append(f"Manifest {i} missing field: {field}")

    return len(issues) == 0, issues
