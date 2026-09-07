"""Artificial masking for evaluation using same recipes as occurrence gate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import json
import numpy as np
import pandas as pd

from .splits import (
    SplitResult,
    country_holdout_split,
    empirical_pattern_split,
    random_cell_split,
    time_block_split,
)


@dataclass
class EvaluationMask:
    """Mask for evaluation with ground truth."""
    train_mask: np.ndarray  # (n_rows, n_features) boolean, True = used for training
    test_mask: np.ndarray   # (n_rows, n_features) boolean, True = held out for evaluation
    test_values: np.ndarray # Ground truth values for test cells
    recipe: str
    split_metadata: dict


def load_split_ids(splits_dir: str | Path) -> dict:
    """Load pre-computed split IDs from configs/splits/."""
    splits_dir = Path(splits_dir)
    splits = {}
    for name in ["random_cell", "empirical_pattern", "time_block", "country_holdout"]:
        path = splits_dir / f"{name}.json"
        with path.open(encoding="utf-8") as f:
            splits[name] = json.load(f)
    return splits


def create_evaluation_masks_from_split_ids(
    observed_mask: np.ndarray,
    values: np.ndarray,
    total_seq: np.ndarray,
    split_ids: dict,
    recipe: Literal["random-cell", "empirical-pattern", "time-block", "country-holdout"],
    df: pd.DataFrame | None = None,
    location_col: str = "location",
    date_col: str = "date",
) -> "EvaluationMask":
    """Create evaluation masks from pre-computed split IDs.
    
    Args:
        observed_mask: (n_rows, n_features) boolean, True = observed in raw data.
        values: (n_rows, n_features) true values (counts).
        total_seq: (n_rows,) total sequence for proportion conversion.
        split_ids: Pre-computed split IDs from load_split_ids().
        recipe: Masking recipe to use.
        df: DataFrame with location/date columns (required for time-block, country-holdout).
        location_col: Location column name.
        date_col: Date column name.
    
    Returns:
        EvaluationMask with train/test masks and ground truth values (as proportions).
    """
    split_key = {
        "random-cell": "random_cell",
        "empirical-pattern": "empirical_pattern",
        "time-block": "time_block",
        "country-holdout": "country_holdout",
    }[recipe]
    
    split_data = split_ids.get(split_key, {})
    
    # Convert values to proportions for test_values
    props = values / total_seq[:, np.newaxis]
    
    if recipe == "random-cell":
        test_mask = np.zeros_like(observed_mask, dtype=bool)
        row_indices = split_data.get("test_cell_row_indices", [])
        var_indices = split_data.get("test_cell_variant_indices", [])
        for r, v in zip(row_indices, var_indices):
            test_mask[r, v] = True
    elif recipe == "empirical-pattern":
        test_mask = np.zeros_like(observed_mask, dtype=bool)
        test_row_indices = split_data.get("test_row_indices", [])
        for r in test_row_indices:
            test_mask[r] = observed_mask[r]
    elif recipe == "time-block":
        test_mask = np.zeros_like(observed_mask, dtype=bool)
        test_row_indices = split_data.get("test_row_indices", [])
        for r in test_row_indices:
            test_mask[r] = observed_mask[r]
    elif recipe == "country-holdout":
        test_mask = np.zeros_like(observed_mask, dtype=bool)
        held_out_locs = set(split_data.get("held_out_locations", []))
        if df is not None:
            for i, loc in enumerate(df[location_col]):
                if loc in held_out_locs:
                    test_mask[i] = observed_mask[i]
    else:
        raise ValueError(f"Unknown recipe: {recipe}")
    
    train_mask = observed_mask & ~test_mask
    
    # Extract test values (ground truth) as PROPORTIONS
    test_values = np.full_like(props, np.nan)
    test_values[test_mask] = props[test_mask]
    
    return EvaluationMask(
        train_mask=train_mask,
        test_mask=test_mask,
        test_values=test_values,
        recipe=recipe,
        split_metadata=split_data,
    )


def create_evaluation_masks(
    observed_mask: np.ndarray,
    values: np.ndarray,
    recipe: Literal["random-cell", "empirical-pattern", "time-block", "country-holdout"],
    df: pd.DataFrame | None = None,
    location_col: str = "location",
    date_col: str = "date",
    seed: int = 42,
    test_fraction: float = 0.2,
    n_patterns: int = 20,
    blocks_per_location: int = 2,
    holdout_fraction: float = 0.1,
) -> EvaluationMask:
    """Create artificial masks for evaluation using the same recipes as occurrence gate.
    
    This is the fallback function that generates splits on-the-fly (not using pre-computed IDs).
    """
    from .splits import (
        country_holdout_split,
        empirical_pattern_split,
        random_cell_split,
        time_block_split,
    )
    
    if recipe == "random-cell":
        split = random_cell_split(observed_mask, test_fraction=test_fraction, seed=seed)
    elif recipe == "empirical-pattern":
        split = empirical_pattern_split(observed_mask, n_patterns=n_patterns, seed=seed)
    elif recipe == "time-block":
        if df is None:
            raise ValueError("df required for time-block recipe")
        split = time_block_split(
            observed_mask, blocks_per_location=blocks_per_location, seed=seed,
            df=df, location_col=location_col, date_col=date_col,
        )
    elif recipe == "country-holdout":
        if df is None:
            raise ValueError("df required for country-holdout recipe")
        split = country_holdout_split(
            df, observed_mask, holdout_fraction=holdout_fraction, seed=seed,
        )
    else:
        raise ValueError(f"Unknown recipe: {recipe}")

    # Extract test values (ground truth)
    test_values = values.copy()
    test_values[~split.test_mask] = np.nan

    return EvaluationMask(
        train_mask=split.train_mask,
        test_mask=split.test_mask,
        test_values=test_values,
        recipe=split.split_type,
        split_metadata=split.metadata,
    )


def create_all_evaluation_masks(
    observed_mask: np.ndarray,
    values: np.ndarray,
    total_seq: np.ndarray,
    df: pd.DataFrame,
    location_col: str = "location",
    date_col: str = "date",
    seed: int = 42,
    test_fraction: float = 0.2,
    splits_dir: str | Path | None = None,
) -> dict[str, EvaluationMask]:
    """Create evaluation masks for all 4 recipes using pre-computed split IDs if available.
    
    Args:
        observed_mask: (n_rows, n_features) boolean, True = observed in raw data.
        values: (n_rows, n_features) true values (counts).
        total_seq: (n_rows,) total sequence per row for proportion conversion.
        df: DataFrame with location/date columns.
        location_col: Location column name.
        date_col: Date column name.
        seed: Random seed (used only for fallback).
        test_fraction: Fraction for random-cell recipe (fallback).
        splits_dir: Directory containing pre-computed split IDs. If None, generates on-the-fly.
    
    Returns:
        Dict mapping recipe name to EvaluationMask.
    """
    recipes = ["random-cell", "empirical-pattern", "time-block", "country-holdout"]
    masks = {}
    
    split_ids = None
    if splits_dir is not None:
        try:
            split_ids = load_split_ids(splits_dir)
        except Exception:
            split_ids = None
    
    for recipe in ["random-cell", "empirical-pattern", "time-block", "country-holdout"]:
        if split_ids is not None:
            masks[recipe] = create_evaluation_masks_from_split_ids(
                observed_mask, values, total_seq, split_ids, recipe, df, location_col, date_col
            )
        else:
            masks[recipe] = create_evaluation_masks(
                observed_mask, values, recipe, df, location_col, date_col, seed, test_fraction
            )
    
    return masks


def evaluate_with_masks(
    true_values: np.ndarray,
    imputed_values: np.ndarray,
    evaluation_mask: EvaluationMask,
    metric_fn,
) -> float:
    """Evaluate a metric only on test cells.

    Args:
        true_values: Full true values array (not used, kept for API compatibility).
        imputed_values: Full imputed values array.
        evaluation_mask: EvaluationMask with test_mask and test_values (ground truth).
        metric_fn: Function(true, imputed) -> float.

    Returns:
        Metric value computed only on test cells.
    """
    test_mask = evaluation_mask.test_mask
    if test_mask.sum() == 0:
        return np.nan
    # Use test_values from mask (ground truth for artificially masked cells)
    # instead of true_values which may have 0 for missing cells
    return metric_fn(evaluation_mask.test_values[test_mask], imputed_values[test_mask])
