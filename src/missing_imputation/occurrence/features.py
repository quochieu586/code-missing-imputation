"""Feature construction for the pooled logistic Occurrence Gate.

Only raw observed context is used. No Tsagris, occurrence or GAN output is
ever read here (leakage rule, IMPLEMENTATION_PLAN.md Section 5.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

BASE_FEATURE_NAMES = [
    "scaled_global_day",
    "log1p_total_seq",
    "lag_positive",
    "lead_positive",
    "lag_avail",
    "lead_avail",
    "scaled_backward_gap",
]


@dataclass
class OccurrenceFeatures:
    X: np.ndarray
    y: np.ndarray
    row_idx: np.ndarray
    variant_idx: np.ndarray
    feature_names: list[str] = field(default_factory=list)
    min_date: pd.Timestamp | None = None
    variant_names: list[str] = field(default_factory=list)


def _lag_lead_arrays(
    df: pd.DataFrame,
    variant_cols: list[str],
    observed_mask: np.ndarray,
    location_col: str,
    date_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_rows = len(df)
    n_var = len(variant_cols)
    lag_avail = np.zeros((n_rows, n_var), dtype=np.float64)
    lag_positive = np.zeros((n_rows, n_var), dtype=np.float64)
    lead_avail = np.zeros((n_rows, n_var), dtype=np.float64)
    lead_positive = np.zeros((n_rows, n_var), dtype=np.float64)
    backward_gap = np.zeros(n_rows, dtype=np.float64)

    raw_counts = df[variant_cols].to_numpy(dtype=np.float64)
    locations = df[location_col].to_numpy()
    dates = df[date_col].to_numpy()

    for loc in pd.unique(locations):
        loc_rows = np.where(locations == loc)[0]
        order = np.argsort(dates[loc_rows], kind="stable")
        loc_rows = loc_rows[order]
        if len(loc_rows) < 2:
            continue
        for pos in range(1, len(loc_rows)):
            cur = loc_rows[pos]
            prev = loc_rows[pos - 1]
            gap_days = (pd.Timestamp(dates[cur]) - pd.Timestamp(dates[prev])).days
            backward_gap[cur] = float(gap_days)
            obs_prev = observed_mask[prev].astype(bool)
            lag_avail[cur, obs_prev] = 1.0
            lag_positive[cur] = np.where(
                obs_prev, (raw_counts[prev] > 0).astype(np.float64), 0.0
            )
            obs_cur = observed_mask[cur].astype(bool)
            lead_avail[prev, obs_cur] = 1.0
            lead_positive[prev] = np.where(
                obs_cur, (raw_counts[cur] > 0).astype(np.float64), 0.0
            )

    return lag_avail, lag_positive, lead_avail, lead_positive, backward_gap


def build_occurrence_features(
    df: pd.DataFrame,
    variant_cols: list[str],
    observed_mask: np.ndarray | None = None,
    location_col: str = "location",
    date_col: str = "date",
    total_seq_col: str = "total_sequence",
    include_spline: bool = False,
    spline_n_knots: int = 5,
    spline_degree: int = 3,
) -> OccurrenceFeatures:
    if observed_mask is None:
        observed_mask = df[variant_cols].notna().to_numpy()
    observed_mask = observed_mask.astype(bool)

    lag_avail, lag_positive, lead_avail, lead_positive, backward_gap = _lag_lead_arrays(
        df, variant_cols, observed_mask, location_col, date_col
    )

    min_date = pd.Timestamp(df[date_col].min())
    global_day = (pd.to_datetime(df[date_col]) - min_date).dt.days.to_numpy(
        dtype=np.float64
    )
    scaled_global_day = global_day / 365.25
    log1p_total_seq = np.log1p(df[total_seq_col].to_numpy(dtype=np.float64))
    raw_counts = df[variant_cols].to_numpy(dtype=np.float64)

    cell_rows, cell_vars = np.where(observed_mask)
    n_cells = len(cell_rows)

    y = (raw_counts[cell_rows, cell_vars] > 0).astype(np.int64)

    variant_names = list(variant_cols)
    n_variants = len(variant_names)
    variant_dummies = np.zeros((n_cells, max(n_variants - 1, 0)), dtype=np.float64)
    for k in range(1, n_variants):
        variant_dummies[cell_vars == k, k - 1] = 1.0
    variant_feature_names = [f"variant_effect_{variant_names[k]}" for k in range(1, n_variants)]

    base = np.column_stack(
        [
            scaled_global_day[cell_rows],
            log1p_total_seq[cell_rows],
            lag_positive[cell_rows, cell_vars],
            lead_positive[cell_rows, cell_vars],
            lag_avail[cell_rows, cell_vars],
            lead_avail[cell_rows, cell_vars],
            backward_gap[cell_rows] / 365.25,
        ]
    )

    feature_names = BASE_FEATURE_NAMES + variant_feature_names
    X = np.hstack([base, variant_dummies]) if n_cells > 0 else base

    if include_spline and n_cells > 0:
        from sklearn.preprocessing import SplineTransformer

        spline = SplineTransformer(
            n_knots=spline_n_knots, degree=spline_degree, include_bias=False
        )
        spline_cols = spline.fit_transform(scaled_global_day[cell_rows].reshape(-1, 1))
        spline_names = [f"time_spline_{i}" for i in range(spline_cols.shape[1])]
        X = np.hstack([X, spline_cols])
        feature_names = feature_names + spline_names

    return OccurrenceFeatures(
        X=X,
        y=y,
        row_idx=cell_rows.astype(np.int64),
        variant_idx=cell_vars.astype(np.int64),
        feature_names=feature_names,
        min_date=min_date,
        variant_names=variant_names,
    )


def build_target_prediction_features(
    df: pd.DataFrame,
    variant_cols: list[str],
    target_mask: np.ndarray,
    min_date: pd.Timestamp,
    location_col: str = "location",
    date_col: str = "date",
    total_seq_col: str = "total_sequence",
    include_spline: bool = False,
    spline_transformer=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build feature matrix for true target cells using raw observed context."""
    observed_mask = df[variant_cols].notna().to_numpy().astype(bool)
    lag_avail, lag_positive, lead_avail, lead_positive, backward_gap = _lag_lead_arrays(
        df, variant_cols, observed_mask, location_col, date_col
    )

    global_day = (pd.to_datetime(df[date_col]) - min_date).dt.days.to_numpy(
        dtype=np.float64
    )
    scaled_global_day = global_day / 365.25
    log1p_total_seq = np.log1p(df[total_seq_col].to_numpy(dtype=np.float64))

    target_mask = target_mask.astype(bool)
    cell_rows, cell_vars = np.where(target_mask)
    n_cells = len(cell_rows)
    n_variants = len(variant_cols)

    variant_dummies = np.zeros((n_cells, max(n_variants - 1, 0)), dtype=np.float64)
    for k in range(1, n_variants):
        variant_dummies[cell_vars == k, k - 1] = 1.0

    base = np.column_stack(
        [
            scaled_global_day[cell_rows],
            log1p_total_seq[cell_rows],
            lag_positive[cell_rows, cell_vars],
            lead_positive[cell_rows, cell_vars],
            lag_avail[cell_rows, cell_vars],
            lead_avail[cell_rows, cell_vars],
            backward_gap[cell_rows] / 365.25,
        ]
    )
    X = np.hstack([base, variant_dummies]) if n_cells > 0 else base

    if include_spline and n_cells > 0 and spline_transformer is not None:
        spline_cols = spline_transformer.transform(
            scaled_global_day[cell_rows].reshape(-1, 1)
        )
        X = np.hstack([X, spline_cols])

    return X, cell_rows.astype(np.int64), cell_vars.astype(np.int64)
