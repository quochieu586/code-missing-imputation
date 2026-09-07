"""Generate versioned split ID files for the four OOF recipes.

Run from project root:
    python scripts/generate_split_ids.py

Writes configs/splits/{random_cell,empirical_pattern,time_block,country_holdout}.json
Each file is deterministic given its seed and the frozen raw data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from missing_imputation.data.loader import load_config, load_covariants

SEED = 20260901
OUTPUT_DIR = PROJECT_ROOT / "configs" / "splits"


def random_cell_split(observed_mask: np.ndarray, seed: int, test_fraction: float = 0.2) -> dict:
    rng = np.random.default_rng(seed)
    obs_indices = np.argwhere(observed_mask)
    n_obs = len(obs_indices)
    n_test = int(n_obs * test_fraction)
    perm = rng.permutation(n_obs)
    test_cells = obs_indices[perm[:n_test]].tolist()
    return {
        "recipe": "random-cell",
        "seed": seed,
        "test_fraction": test_fraction,
        "n_observed": int(n_obs),
        "n_test": int(n_test),
        "test_cell_row_indices": [int(c[0]) for c in test_cells],
        "test_cell_variant_indices": [int(c[1]) for c in test_cells],
    }


def empirical_pattern_split(observed_mask: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n_rows = observed_mask.shape[0]
    patterns: dict[tuple, list[int]] = {}
    for i in range(n_rows):
        p = tuple((~observed_mask[i]).tolist())
        patterns.setdefault(p, []).append(i)
    pattern_keys = sorted(patterns.keys(), key=lambda p: (sum(p), p))
    n_select = min(20, len(pattern_keys))
    selected = rng.choice(len(pattern_keys), size=n_select, replace=False)
    test_rows: list[int] = []
    for pi in sorted(int(s) for s in selected):
        test_rows.extend(patterns[pattern_keys[pi]])
    return {
        "recipe": "empirical-pattern",
        "seed": seed,
        "n_patterns_available": len(pattern_keys),
        "n_patterns_selected": int(n_select),
        "test_row_indices": sorted(test_rows),
    }


def time_block_split(
    df: pd.DataFrame, observed_mask: np.ndarray, seed: int,
    location_col: str, date_col: str, blocks_per_location: int = 2,
) -> dict:
    rng = np.random.default_rng(seed)
    locations = df[location_col].to_numpy()
    dates = df[date_col].to_numpy()
    test_rows: list[int] = []
    for loc in pd.unique(locations):
        loc_rows = np.where(locations == loc)[0]
        order = np.argsort(dates[loc_rows])
        loc_rows = loc_rows[order]
        n = len(loc_rows)
        if n < 4:
            continue
        n_blocks = min(blocks_per_location, n // 3)
        starts = rng.choice(max(n - 2, 1), size=n_blocks, replace=False)
        for s in starts:
            end = min(int(s) + 2, n)
            test_rows.extend(int(r) for r in loc_rows[int(s):end])
    return {
        "recipe": "time-block",
        "seed": seed,
        "blocks_per_location": blocks_per_location,
        "test_row_indices": sorted(set(test_rows)),
    }


def country_holdout_split(
    df: pd.DataFrame, seed: int, location_col: str, holdout_fraction: float = 0.1,
) -> dict:
    rng = np.random.default_rng(seed)
    locations = sorted(df[location_col].unique())
    n_holdout = max(1, int(len(locations) * holdout_fraction))
    held_out = sorted(rng.choice(locations, size=n_holdout, replace=False).tolist())
    return {
        "recipe": "country-holdout",
        "seed": seed,
        "holdout_fraction": holdout_fraction,
        "held_out_locations": held_out,
    }


def main() -> int:
    config = load_config(PROJECT_ROOT / "configs" / "data.yaml")
    df = load_covariants(config.path, config)
    variant_cols = config.variant_components
    observed_mask = df[variant_cols].notna().to_numpy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    splits = {
        "random_cell.json": random_cell_split(observed_mask, SEED),
        "empirical_pattern.json": empirical_pattern_split(observed_mask, SEED),
        "time_block.json": time_block_split(
            df, observed_mask, SEED, config.location_col, config.date_col
        ),
        "country_holdout.json": country_holdout_split(df, SEED, config.location_col),
    }
    for name, payload in splits.items():
        path = OUTPUT_DIR / name
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1, default=str)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())