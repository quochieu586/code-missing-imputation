"""Evaluation split generation following Section 6.1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SplitResult:
    """Result of a split operation."""

    train_mask: np.ndarray
    test_mask: np.ndarray
    split_type: str
    seed: int
    metadata: dict


def random_cell_split(
    observed_mask: np.ndarray,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> SplitResult:
    """Random cell-level split for sanity check.

    Masks a random fraction of observed cells for evaluation.

    Args:
        observed_mask: (n_rows, n_features) boolean, True = observed.
        test_fraction: Fraction of observed cells to mask.
        seed: Random seed.

    Returns:
        SplitResult with train/test masks.
    """
    rng = np.random.default_rng(seed)
    obs_indices = np.argwhere(observed_mask)
    n_obs = len(obs_indices)
    n_test = int(n_obs * test_fraction)

    perm = rng.permutation(n_obs)
    test_indices = obs_indices[perm[:n_test]]

    test_mask = np.zeros_like(observed_mask)
    for idx in test_indices:
        test_mask[idx[0], idx[1]] = True

    train_mask = observed_mask & ~test_mask

    return SplitResult(
        train_mask=train_mask,
        test_mask=test_mask,
        split_type="random-cell",
        seed=seed,
        metadata={"n_test_cells": n_test, "test_fraction": test_fraction},
    )


def empirical_pattern_split(
    observed_mask: np.ndarray,
    n_patterns: int = 20,
    seed: int = 42,
) -> SplitResult:
    """Empirical pattern split: mask entire rows matching real patterns.

    Args:
        observed_mask: (n_rows, n_features) boolean.
        n_patterns: Number of patterns to sample.
        seed: Random seed.

    Returns:
        SplitResult.
    """
    rng = np.random.default_rng(seed)
    n_rows = observed_mask.shape[0]

    patterns = {}
    for i in range(n_rows):
        p = tuple(~observed_mask[i])
        if p not in patterns:
            patterns[p] = []
        patterns[p].append(i)

    pattern_keys = list(patterns.keys())
    n_select = min(n_patterns, len(pattern_keys))
    selected_patterns = rng.choice(
        len(pattern_keys), size=n_select, replace=False
    )

    test_mask = np.zeros_like(observed_mask)
    for pi in selected_patterns:
        pattern = pattern_keys[pi]
        rows = patterns[pattern]
        for r in rows:
            test_mask[r] = observed_mask[r]

    train_mask = observed_mask & ~test_mask

    return SplitResult(
        train_mask=train_mask,
        test_mask=test_mask,
        split_type="empirical-pattern",
        seed=seed,
        metadata={"n_patterns_selected": n_select},
    )


def time_block_split(
    observed_mask: np.ndarray,
    block_size_days: int = 28,
    blocks_per_location: int = 2,
    seed: int = 42,
    df: pd.DataFrame | None = None,
    location_col: str = "location",
    date_col: str = "date",
) -> SplitResult:
    """Time-block split: hold out contiguous time blocks per location.

    Works with a 2D panel-format mask of shape (n_locations, n_times).
    Optional df/location_col/date_col are kept for API compatibility but
    are not needed when the mask is already in panel format.

    Args:
        observed_mask: (n_locations, n_times) boolean panel, True = observed.
        block_size_days: Size of each block in days.
        blocks_per_location: Number of blocks to hold out per location.
        seed: Random seed.
        df: (optional) DataFrame, currently unused for panel-format masks.
        location_col: (optional) location column name, currently unused.
        date_col: (optional) date column name, currently unused.

    Returns:
        SplitResult.
    """
    rng = np.random.default_rng(seed)
    test_mask = np.zeros_like(observed_mask)

    n_locations, n_times = observed_mask.shape
    block_size_steps = max(1, block_size_days // 14)

    for loc in range(n_locations):
        obs_times = np.where(observed_mask[loc])[0]
        if len(obs_times) < 3:
            continue

        n_blocks = min(blocks_per_location, len(obs_times) // 2)
        if n_blocks == 0:
            continue

        block_starts = rng.choice(
            len(obs_times) - 1, size=n_blocks, replace=False
        )

        for start in block_starts:
            end = min(start + block_size_steps, len(obs_times))
            block_times = obs_times[start:end]
            test_mask[loc, block_times] = True

    train_mask = observed_mask & ~test_mask

    return SplitResult(
        train_mask=train_mask,
        test_mask=test_mask,
        split_type="time-block",
        seed=seed,
        metadata={
            "block_size_days": block_size_days,
            "blocks_per_location": blocks_per_location,
        },
    )


def country_holdout_split(
    df: pd.DataFrame,
    observed_mask: np.ndarray,
    holdout_fraction: float = 0.1,
    location_col: str = "location",
    seed: int = 42,
) -> SplitResult:
    """Country/location holdout split.

    Args:
        df: DataFrame with location column.
        observed_mask: (n_rows, n_features) boolean.
        holdout_fraction: Fraction of locations to hold out.
        location_col: Location column name.
        seed: Random seed.

    Returns:
        SplitResult.
    """
    rng = np.random.default_rng(seed)
    locations = df[location_col].unique()
    n_holdout = max(1, int(len(locations) * holdout_fraction))

    held_out = rng.choice(locations, size=n_holdout, replace=False)
    held_out_set = set(held_out)

    test_mask = np.zeros_like(observed_mask)
    for i, loc in enumerate(df[location_col]):
        if loc in held_out_set:
            test_mask[i] = observed_mask[i]

    train_mask = observed_mask & ~test_mask

    return SplitResult(
        train_mask=train_mask,
        test_mask=test_mask,
        split_type="country-holdout",
        seed=seed,
        metadata={"n_locations_held_out": n_holdout, "locations": list(held_out)},
    )
