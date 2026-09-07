"""GAN data preparation: panel construction, artificial magnitude masking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class GANDataset:
    X_clr: np.ndarray
    counts_raw: np.ndarray
    total_sequence: np.ndarray
    M_observed: np.ndarray
    M_fixed: np.ndarray
    M_gan: np.ndarray
    M_padding: np.ndarray
    M_row: np.ndarray
    p_nonzero: np.ndarray
    time_decay_f: np.ndarray
    time_decay_b: np.ndarray
    location_index: list
    time_index: list
    feature_names: list[str]
    pseudo_count: float


def build_gan_panel(
    df_fused: pd.DataFrame,
    M_fixed: np.ndarray,
    M_gan: np.ndarray,
    p_nonzero_flat: np.ndarray | None = None,
    variant_cols: list[str] | None = None,
    location_col: str = "location",
    date_col: str = "date",
    total_seq_col: str = "total_sequence",
    other_col: str = "other",
    pseudo_count: float | None = None,
) -> GANDataset:
    from .clr import clr_transform

    if variant_cols is None:
        raise ValueError("variant_cols required")

    locations = sorted(df_fused[location_col].unique())
    n_variants = len(variant_cols)
    # Include "other" as the 18th feature for proper closure
    feature_names = variant_cols + [other_col]
    n_features = len(feature_names)

    counts = df_fused[feature_names].fillna(0).to_numpy(dtype=np.float64)
    total_seq = df_fused[total_seq_col].to_numpy(dtype=np.float64)
    dates = pd.to_datetime(df_fused[date_col]).to_numpy()

    loc_to_idx = {loc: i for i, loc in enumerate(locations)}
    
    # Compute actual observed timepoints per location (like reference implementation)
    # This handles irregular dates and missing timepoints per location
    loc_timepoints = {}
    for loc in locations:
        loc_dates = pd.to_datetime(df_fused.loc[df_fused[location_col] == loc, date_col])
        loc_timepoints[loc] = sorted(loc_dates.unique())

    all_times = sorted(set(pd.to_datetime(df_fused[date_col])))
    time_to_idx = {t: i for i, t in enumerate(all_times)}
    n_times = len(all_times)
    n_locs = len(locations)

    X = np.zeros((n_locs, n_times, n_features), dtype=np.float64)
    M_obs = np.zeros((n_locs, n_times, n_features), dtype=np.uint8)
    M_pad = np.zeros((n_locs, n_times), dtype=np.uint8)
    M_row = np.zeros((n_locs, n_times), dtype=np.uint8)
    total_seq_panel = np.zeros((n_locs, n_times), dtype=np.float64)

    for i in range(len(df_fused)):
        loc = df_fused.iloc[i][location_col]
        loc_idx = loc_to_idx[loc]
        t = pd.to_datetime(df_fused.iloc[i][date_col])
        if t not in time_to_idx:
            continue
        t_idx = time_to_idx[t]
        M_row[loc_idx, t_idx] = 1
        total_seq_panel[loc_idx, t_idx] = total_seq[i]
        for j in range(n_features):
            X[loc_idx, t_idx, j] = counts[i, j]
            if j < n_variants:
                if M_fixed[i, j] and not M_gan[i, j]:
                    M_obs[loc_idx, t_idx, j] = 1
            else:
                # "other" is always observed (it's the residual)
                M_obs[loc_idx, t_idx, j] = 1

    M_pad = 1 - M_row

    # Compute time decay based on actual observed timepoints per location (like reference)
    # time_decay_f: forward time decay (gap to previous observed timepoint)
    # time_decay_b: backward time decay (gap to next observed timepoint)
    time_decay_f = np.zeros((n_locs, n_times, 1), dtype=np.float64)
    time_decay_b = np.zeros((n_locs, n_times, 1), dtype=np.float64)
    
    for loc_idx, loc in enumerate(locations):
        loc_times = loc_timepoints[loc]
        if len(loc_times) == 0:
            continue
            
        # Map actual timepoints to grid indices
        loc_time_indices = [time_to_idx[t] for t in loc_times if t in time_to_idx]
        
        # Forward decay: for each grid timepoint, compute gap to previous OBSERVED timepoint
        last_obs_idx = None
        for t_idx in range(n_times):
            if t_idx in loc_time_indices:
                # This grid point has an observation
                time_decay_f[loc_idx, t_idx, 0] = 0.0
                last_obs_idx = t_idx
            else:
                # Grid point without observation - accumulate gap from last observed
                if last_obs_idx is not None:
                    gap = (all_times[t_idx] - all_times[last_obs_idx]).days
                    time_decay_f[loc_idx, t_idx, 0] = float(gap)
                else:
                    time_decay_f[loc_idx, t_idx, 0] = 0.0  # No previous observation
        
        # Backward decay: for each grid timepoint, compute gap to next OBSERVED timepoint
        next_obs_idx = None
        for t_idx in range(n_times - 1, -1, -1):
            if t_idx in loc_time_indices:
                # This grid point has an observation
                time_decay_b[loc_idx, t_idx, 0] = 0.0
                next_obs_idx = t_idx
            else:
                # Grid point without observation - accumulate gap to next observed
                if next_obs_idx is not None:
                    gap = (all_times[next_obs_idx] - all_times[t_idx]).days
                    time_decay_b[loc_idx, t_idx, 0] = float(gap)
                else:
                    time_decay_b[loc_idx, t_idx, 0] = 0.0  # No next observation

    X_flat = X.reshape(n_locs * n_times, n_features)
    X_clr, pc = clr_transform(X_flat, pseudo_count)

    M_fixed_panel = np.zeros((n_locs, n_times, n_features), dtype=np.uint8)
    M_gan_panel = np.zeros((n_locs, n_times, n_features), dtype=np.uint8)
    p_nz_panel = np.zeros((n_locs, n_times, n_features), dtype=np.float64)
    flat_idx = 0
    for loc_idx in range(n_locs):
        for t_idx in range(n_times):
            if M_row[loc_idx, t_idx] == 1:
                M_fixed_panel[loc_idx, t_idx, :n_variants] = M_fixed[flat_idx]
                M_gan_panel[loc_idx, t_idx, :n_variants] = M_gan[flat_idx]
                # "other" is never GAN-editable
                M_fixed_panel[loc_idx, t_idx, n_variants] = 1
                M_gan_panel[loc_idx, t_idx, n_variants] = 0
                if p_nonzero_flat is not None:
                    for j in range(n_variants):
                        if M_gan[flat_idx, j]:
                            pass
                flat_idx += 1

    return GANDataset(
        X_clr=X_clr.reshape(n_locs, n_times, n_features),
        counts_raw=X,
        total_sequence=total_seq_panel,
        M_observed=M_obs,
        M_fixed=M_fixed_panel,
        M_gan=M_gan_panel,
        M_padding=M_pad,
        M_row=M_row,
        p_nonzero=p_nz_panel,
        time_decay_f=time_decay_f,
        time_decay_b=time_decay_b,
        location_index=locations,
        time_index=all_times,
        feature_names=feature_names,
        pseudo_count=pc,
    )