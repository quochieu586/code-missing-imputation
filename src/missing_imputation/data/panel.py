"""Longitudinal panel construction and reindexing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class PanelTensor:
    """Longitudinal panel tensor following Section 4.2 data contract.

    Attributes:
        X: Proportion tensor [n_locations, n_times, n_features]
        counts: Count tensor [n_locations, n_times, n_features]
        total_sequence: Total sequence per location-time [n_locations, n_times]
        M_observed: Observed mask [n_locations, n_times, n_features]
        M_row: Row existence mask [n_locations, n_times]
        location_index: Location names
        time_index: Time points (regular grid)
        feature_names: Variant component names
    """

    X: np.ndarray  # proportions
    counts: np.ndarray  # integer counts
    total_sequence: np.ndarray  # total_sequence per location-time
    M_observed: np.ndarray  # 1 = observed, 0 = missing
    M_row: np.ndarray  # 1 = row exists in original data
    location_index: pd.Index
    time_index: pd.DatetimeIndex
    feature_names: list[str]

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.X.shape

    @property
    def n_locations(self) -> int:
        return self.X.shape[0]

    @property
    def n_times(self) -> int:
        return self.X.shape[1]

    @property
    def n_features(self) -> int:
        return self.X.shape[2]

    def get_location_slice(self, location: str) -> "PanelTensor":
        """Extract sub-panel for a single location."""
        loc_idx = self.location_index.get_loc(location)
        return PanelTensor(
            X=self.X[loc_idx : loc_idx + 1],
            counts=self.counts[loc_idx : loc_idx + 1],
            total_sequence=self.total_sequence[loc_idx : loc_idx + 1],
            M_observed=self.M_observed[loc_idx : loc_idx + 1],
            M_row=self.M_row[loc_idx : loc_idx + 1],
            location_index=pd.Index([location]),
            time_index=self.time_index,
            feature_names=self.feature_names,
        )

    def to_dataframe(self, config) -> pd.DataFrame:
        """Convert panel back to DataFrame (only original rows)."""
        rows = []
        for loc_idx, loc in enumerate(self.location_index):
            for time_idx, time in enumerate(self.time_index):
                if self.M_row[loc_idx, time_idx] == 0:
                    continue  # Skip padded grid points

                row = {
                    config.location_col: loc,
                    config.date_col: time,
                    config.total_sequence_col: int(self.total_sequence[loc_idx, time_idx]),
                }
                for f_idx, feat in enumerate(self.feature_names):
                    row[feat] = float(self.counts[loc_idx, time_idx, f_idx])

                # Add 'other' column
                variant_sum = self.counts[loc_idx, time_idx, :].sum()
                row[config.other_col] = int(self.total_sequence[loc_idx, time_idx] - variant_sum)

                rows.append(row)

        return pd.DataFrame(rows)


def build_panel(
    df: pd.DataFrame,
    config,
    variant_cols: list[str],
    freq_days: int = 14,
    emit_grid: bool = False,
) -> PanelTensor:
    """Build longitudinal panel tensor from DataFrame.

    Reindexes to regular freq_days grid per location.
    Padded grid points have M_row=0 and are excluded from output by default.

    Args:
        df: Input DataFrame with covariants data
        config: DataConfig object
        variant_cols: List of variant column names
        freq_days: Grid frequency in days
        emit_grid: If True, include padded grid points in output

    Returns:
        PanelTensor with all required tensors
    """
    location_col = config.location_col
    date_col = config.date_col
    total_seq_col = config.total_sequence_col

    locations = sorted(df[location_col].unique())

    # Build 14-day grid per location
    all_times = []
    for loc in locations:
        loc_df = df[df[location_col] == loc].sort_values(date_col)
        if len(loc_df) == 0:
            continue
        start = loc_df[date_col].min()
        end = loc_df[date_col].max()
        grid = pd.date_range(start=start, end=end, freq=f"{freq_days}D")
        all_times.extend(grid)

    time_index = pd.DatetimeIndex(sorted(set(all_times)))

    n_locations = len(locations)
    n_times = len(time_index)
    n_features = len(variant_cols)

    # Initialize tensors
    X = np.zeros((n_locations, n_times, n_features), dtype=np.float32)  # proportions
    counts = np.zeros((n_locations, n_times, n_features), dtype=np.int32)
    total_sequence = np.zeros((n_locations, n_times), dtype=np.int32)
    M_observed = np.zeros((n_locations, n_times, n_features), dtype=np.uint8)
    M_row = np.zeros((n_locations, n_times), dtype=np.uint8)

    # Maps
    loc_to_idx = {loc: i for i, loc in enumerate(locations)}
    time_to_idx = {t: i for i, t in enumerate(time_index)}

    # Fill from data
    for _, row in df.iterrows():
        loc_idx = loc_to_idx[row[location_col]]
        time_idx = time_to_idx.get(pd.Timestamp(row[date_col]))
        if time_idx is None:
            continue

        total_seq = int(row[total_seq_col])
        total_sequence[loc_idx, time_idx] = total_seq
        M_row[loc_idx, time_idx] = 1

        for f_idx, col in enumerate(variant_cols):
            val = row[col]
            if pd.notna(val):
                count = int(val)
                counts[loc_idx, time_idx, f_idx] = count
                if total_seq > 0:
                    X[loc_idx, time_idx, f_idx] = count / total_seq
                M_observed[loc_idx, time_idx, f_idx] = 1

    return PanelTensor(
        X=X,
        counts=counts,
        total_sequence=total_sequence,
        M_observed=M_observed,
        M_row=M_row,
        location_index=pd.Index(locations),
        time_index=time_index,
        feature_names=variant_cols,
    )


def panel_to_proportions(panel: PanelTensor) -> np.ndarray:
    """Extract proportion tensor from panel (alias for panel.X)."""
    return panel.X


def proportions_to_panel(
    proportions: np.ndarray,
    panel: PanelTensor,
    config,
) -> PanelTensor:
    """Create new panel with updated proportions (counts recomputed via closure).

    This uses largest-remainder allocation on missing cells only.
    """
    from .closure import proportions_to_counts_with_closure

    new_counts = np.zeros_like(panel.counts)

    for loc_idx in range(panel.n_locations):
        for time_idx in range(panel.n_times):
            if panel.M_row[loc_idx, time_idx] == 0:
                continue

            total_seq = panel.total_sequence[loc_idx, time_idx]
            if total_seq == 0:
                continue

            prop = proportions[loc_idx, time_idx]
            obs_mask = panel.M_observed[loc_idx, time_idx].astype(bool)
            orig_counts = panel.counts[loc_idx, time_idx]

            # Convert proportions to counts with closure
            new_counts[loc_idx, time_idx] = proportions_to_counts_with_closure(
                prop, total_seq, obs_mask, orig_counts
            )

    return PanelTensor(
        X=proportions,
        counts=new_counts,
        total_sequence=panel.total_sequence,
        M_observed=panel.M_observed,
        M_row=panel.M_row,
        location_index=panel.location_index,
        time_index=panel.time_index,
        feature_names=panel.feature_names,
    )