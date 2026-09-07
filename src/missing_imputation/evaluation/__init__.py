"""Evaluation: splits, metrics, and reporting."""

from .metrics import (
    compute_all_metrics,
    jsd_metric,
    mae_counts,
    mae_proportions,
    rmse_proportions,
    temporal_roughness,
    zero_f1,
    zero_precision,
    zero_recall,
)
from .splits import (
    SplitResult,
    country_holdout_split,
    empirical_pattern_split,
    random_cell_split,
    time_block_split,
)

__all__ = [
    "SplitResult",
    "random_cell_split",
    "empirical_pattern_split",
    "time_block_split",
    "country_holdout_split",
    "compute_all_metrics",
    "jsd_metric",
    "mae_proportions",
    "rmse_proportions",
    "mae_counts",
    "zero_precision",
    "zero_recall",
    "zero_f1",
    "temporal_roughness",
]
