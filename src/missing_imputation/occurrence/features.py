"""Feature construction for the pooled logistic Occurrence Gate.

Only raw observed context is used. No Tsagris, occurrence or GAN output is
ever read here (leakage rule, IMPLEMENTATION_PLAN.md Section 5.2).

Training cells and target cells go through the SAME builder. They used to have
separate code paths, which is how a train/inference feature mismatch survived
unnoticed: on this dataset missingness is all-or-nothing per (location, variant),
so lag/lead context exists for 98.9% of observed cells and for 0 of the 86,050
target cells. A model fitted with lag/lead was therefore scored in a regime that
never occurs at prediction time. See COMPOSITIONAL_FEATURE_NAMES for the
replacements, all of which are computable on target cells.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TEMPORAL_FEATURE_NAMES = [
    "lag_positive",
    "lead_positive",
    "lag_avail",
    "lead_avail",
]

COMPOSITIONAL_FEATURE_NAMES = [
    # Share of the row's sequencing budget not yet accounted for by the other
    # observed variants. If the observed variants already sum to total_sequence
    # there is nothing left for this cell, so it is almost certainly zero.
    "residual_share",
    # How many other variants must share that residual.
    "n_missing_other",
    # Is this variant circulating at this time at all? Measured on observed cells
    # from OTHER locations, so it is available for a location that never reports
    # the variant.
    "circulation",
]

BASE_FEATURE_NAMES = ["scaled_global_day", "log1p_total_seq", "scaled_backward_gap"]


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
    availability_mask: np.ndarray,
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
            obs_prev = availability_mask[prev].astype(bool)
            lag_avail[cur, obs_prev] = 1.0
            lag_positive[cur] = np.where(
                obs_prev, (raw_counts[prev] > 0).astype(np.float64), 0.0
            )
            obs_cur = availability_mask[cur].astype(bool)
            lead_avail[prev, obs_cur] = 1.0
            lead_positive[prev] = np.where(
                obs_cur, (raw_counts[cur] > 0).astype(np.float64), 0.0
            )

    return lag_avail, lag_positive, lead_avail, lead_positive, backward_gap


def _compositional_arrays(
    df: pd.DataFrame,
    variant_cols: list[str],
    availability_mask: np.ndarray,
    location_col: str,
    date_col: str,
    total_seq_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row-level compositional context, defined identically for any cell.

    Every quantity is leave-one-out with respect to the cell itself and is built
    only from cells visible in `availability_mask`, so a training cell and a
    target cell in the same row see the same kind of context.
    """
    counts = df[variant_cols].fillna(0).to_numpy(dtype=np.float64)
    total = df[total_seq_col].to_numpy(dtype=np.float64)
    avail = availability_mask.astype(bool)

    # residual_share: budget left for this cell and the other non-visible ones
    visible_counts = np.where(avail, counts, 0.0)
    sum_excl_self = visible_counts.sum(axis=1, keepdims=True) - visible_counts
    residual_share = (total[:, None] - sum_excl_self) / np.maximum(total, 1.0)[:, None]

    # n_missing_other: how many OTHER variants are not visible in this row
    n_hidden = (~avail).sum(axis=1, keepdims=True)
    n_missing_other = (n_hidden - (~avail).astype(np.int64)).astype(np.float64)

    # circulation: P(count > 0) for this variant in this month, measured on
    # visible cells of OTHER locations (leave-one-location-out).
    dates = pd.to_datetime(df[date_col])
    month = dates.dt.to_period("M").astype(str).to_numpy()
    locations = df[location_col].to_numpy()
    positive = (counts > 0) & avail

    circulation = np.zeros_like(counts, dtype=np.float64)
    frame = pd.DataFrame({"month": month, "location": locations})
    for j in range(len(variant_cols)):
        frame["obs"] = avail[:, j].astype(np.int64)
        frame["pos"] = positive[:, j].astype(np.int64)
        by_month = frame.groupby("month")[["obs", "pos"]].sum()
        by_month_loc = frame.groupby(["month", "location"])[["obs", "pos"]].sum()

        tot_obs = frame["month"].map(by_month["obs"]).to_numpy(dtype=np.float64)
        tot_pos = frame["month"].map(by_month["pos"]).to_numpy(dtype=np.float64)
        idx = pd.MultiIndex.from_arrays([frame["month"], frame["location"]])
        own_obs = by_month_loc["obs"].reindex(idx).to_numpy(dtype=np.float64)
        own_pos = by_month_loc["pos"].reindex(idx).to_numpy(dtype=np.float64)

        denom = tot_obs - np.nan_to_num(own_obs)
        numer = tot_pos - np.nan_to_num(own_pos)
        circulation[:, j] = np.divide(
            numer, denom, out=np.zeros_like(denom), where=denom > 0
        )

    return residual_share, n_missing_other, circulation


def build_occurrence_features(
    df: pd.DataFrame,
    variant_cols: list[str],
    cell_mask: np.ndarray | None = None,
    availability_mask: np.ndarray | None = None,
    location_col: str = "location",
    date_col: str = "date",
    total_seq_col: str = "total_sequence",
    include_spline: bool = False,
    spline_n_knots: int = 5,
    spline_degree: int = 3,
    spline_transformer=None,
    use_temporal: bool = False,
    use_compositional: bool = True,
) -> OccurrenceFeatures:
    """Build the pooled long-table design matrix over cells.

    Two masks with different jobs:
      cell_mask         - which cells become rows of X (the cell set and its order)
      availability_mask - which cells may be READ as context

    They are separate because of plan S5.3 step 2: inside an OOF fold the
    held-out cells must be invisible to the feature builder while still being
    scored. Default: availability = cell_mask.

    use_temporal enables the lag/lead block of plan S5.2. It is OFF by default
    because on this dataset it is identically zero on every target cell (see the
    module docstring), which makes any model that relies on it untransferable.

    y is only meaningful for observed cells; on target cells it is zero-filled
    and must not be used.
    """
    if cell_mask is None:
        cell_mask = df[variant_cols].notna().to_numpy()
    cell_mask = cell_mask.astype(bool)
    if availability_mask is None:
        availability_mask = cell_mask
    availability_mask = availability_mask.astype(bool)

    lag_avail, lag_positive, lead_avail, lead_positive, backward_gap = _lag_lead_arrays(
        df, variant_cols, availability_mask, location_col, date_col
    )

    min_date = pd.Timestamp(df[date_col].min())
    global_day = (pd.to_datetime(df[date_col]) - min_date).dt.days.to_numpy(
        dtype=np.float64
    )
    scaled_global_day = global_day / 365.25
    log1p_total_seq = np.log1p(df[total_seq_col].to_numpy(dtype=np.float64))
    raw_counts = df[variant_cols].fillna(0).to_numpy(dtype=np.float64)

    cell_rows, cell_vars = np.where(cell_mask)
    n_cells = len(cell_rows)
    y = (raw_counts[cell_rows, cell_vars] > 0).astype(np.int64)

    columns: list[np.ndarray] = [
        scaled_global_day[cell_rows],
        log1p_total_seq[cell_rows],
        backward_gap[cell_rows] / 365.25,
    ]
    feature_names = list(BASE_FEATURE_NAMES)

    if use_compositional:
        residual_share, n_missing_other, circulation = _compositional_arrays(
            df, variant_cols, availability_mask, location_col, date_col, total_seq_col
        )
        columns.extend(
            [
                residual_share[cell_rows, cell_vars],
                n_missing_other[cell_rows, cell_vars],
                circulation[cell_rows, cell_vars],
            ]
        )
        feature_names += COMPOSITIONAL_FEATURE_NAMES

    if use_temporal:
        columns.extend(
            [
                lag_positive[cell_rows, cell_vars],
                lead_positive[cell_rows, cell_vars],
                lag_avail[cell_rows, cell_vars],
                lead_avail[cell_rows, cell_vars],
            ]
        )
        feature_names += TEMPORAL_FEATURE_NAMES

    variant_names = list(variant_cols)
    n_variants = len(variant_names)
    variant_dummies = np.zeros((n_cells, max(n_variants - 1, 0)), dtype=np.float64)
    for k in range(1, n_variants):
        variant_dummies[cell_vars == k, k - 1] = 1.0
    feature_names += [f"variant_effect_{variant_names[k]}" for k in range(1, n_variants)]

    base = np.column_stack(columns) if n_cells > 0 else np.zeros((0, len(columns)))
    X = np.hstack([base, variant_dummies]) if n_cells > 0 else base

    if include_spline and n_cells > 0:
        from sklearn.preprocessing import SplineTransformer

        values = scaled_global_day[cell_rows].reshape(-1, 1)
        if spline_transformer is None:
            spline_transformer = SplineTransformer(
                n_knots=spline_n_knots, degree=spline_degree, include_bias=False
            )
            spline_cols = spline_transformer.fit_transform(values)
        else:
            spline_cols = spline_transformer.transform(values)
        X = np.hstack([X, spline_cols])
        feature_names = feature_names + [
            f"time_spline_{i}" for i in range(spline_cols.shape[1])
        ]

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
    use_temporal: bool = False,
    use_compositional: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Design matrix for the true target cells, via the same builder as training.

    Context comes from the raw observed cells; the target cells themselves are
    never read.
    """
    observed_mask = df[variant_cols].notna().to_numpy().astype(bool)
    feats = build_occurrence_features(
        df,
        variant_cols,
        cell_mask=target_mask,
        availability_mask=observed_mask,
        location_col=location_col,
        date_col=date_col,
        total_seq_col=total_seq_col,
        include_spline=include_spline,
        spline_transformer=spline_transformer,
        use_temporal=use_temporal,
        use_compositional=use_compositional,
    )
    return feats.X, feats.row_idx, feats.variant_idx
