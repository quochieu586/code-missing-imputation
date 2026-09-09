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
    row_index: np.ndarray  # grid cell -> df_fused row index, -1 for padding
    p_nonzero: np.ndarray
    w_gan: np.ndarray
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
    p_nonzero_rows: np.ndarray | None = None,
    w_gan_rows: np.ndarray | None = None,
    variant_cols: list[str] | None = None,
    location_col: str = "location",
    date_col: str = "date",
    total_seq_col: str = "total_sequence",
    other_col: str = "other",
    pseudo_count: float | None = None,
) -> GANDataset:
    """Build the [location, time, feature] panel consumed by the GAN.

    p_nonzero_rows carries the occurrence-gate posterior aligned with df_fused rows:
    shape (n_rows, n_variants), defined on target cells and zero elsewhere. It is
    broadcast into the panel as a conditioning channel; on cells locked by M_fixed
    (raw observed and occurrence confident-zero) the channel holds the known
    indicator I(count > 0) instead, since occurrence there is not in question.

    w_gan_rows carries the ZPGF soft weights, same shape and alignment. They are
    not model inputs: postprocess_gan_output multiplies them into the generated
    magnitude (y_hat = w_gan * m) before closure projection, per plan S18.6.
    Cells outside M_gan get weight 1.0 so locked values pass through untouched.
    """
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

    if p_nonzero_rows is not None:
        p_nonzero_rows = np.asarray(p_nonzero_rows, dtype=np.float64).reshape(
            len(df_fused), n_variants
        )
    if w_gan_rows is not None:
        w_gan_rows = np.asarray(w_gan_rows, dtype=np.float64).reshape(
            len(df_fused), n_variants
        )

    X = np.zeros((n_locs, n_times, n_features), dtype=np.float64)
    M_obs = np.zeros((n_locs, n_times, n_features), dtype=np.uint8)
    M_pad = np.zeros((n_locs, n_times), dtype=np.uint8)
    M_row = np.zeros((n_locs, n_times), dtype=np.uint8)
    # Grid cell -> df_fused row index, so masks and posteriors are joined by
    # (location, date) instead of relying on row order matching grid order.
    row_index = np.full((n_locs, n_times), -1, dtype=np.int64)
    total_seq_panel = np.zeros((n_locs, n_times), dtype=np.float64)

    for i in range(len(df_fused)):
        loc = df_fused.iloc[i][location_col]
        loc_idx = loc_to_idx[loc]
        t = pd.to_datetime(df_fused.iloc[i][date_col])
        if t not in time_to_idx:
            continue
        t_idx = time_to_idx[t]
        M_row[loc_idx, t_idx] = 1
        row_index[loc_idx, t_idx] = i
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
    # Weight 1.0 by default: anything the GAN is not allowed to edit must survive
    # the multiplication unchanged.
    w_gan_panel = np.ones((n_locs, n_times, n_features), dtype=np.float64)
    for loc_idx in range(n_locs):
        for t_idx in range(n_times):
            i = int(row_index[loc_idx, t_idx])
            if i < 0:
                continue
            M_fixed_panel[loc_idx, t_idx, :n_variants] = M_fixed[i]
            M_gan_panel[loc_idx, t_idx, :n_variants] = M_gan[i]
            # "other" is never GAN-editable
            M_fixed_panel[loc_idx, t_idx, n_variants] = 1
            M_gan_panel[loc_idx, t_idx, n_variants] = 0

            # Conditioning channel: occurrence posterior on GAN-editable cells,
            # known I(count > 0) on everything locked by M_fixed.
            p_row = np.zeros(n_features, dtype=np.float64)
            if p_nonzero_rows is not None:
                p_row[:n_variants] = p_nonzero_rows[i]
            locked = M_fixed_panel[loc_idx, t_idx].astype(bool) & ~M_gan_panel[
                loc_idx, t_idx
            ].astype(bool)
            p_nz_panel[loc_idx, t_idx] = np.where(
                locked, (X[loc_idx, t_idx] > 0).astype(np.float64), p_row
            )

            if w_gan_rows is not None:
                gan_row = M_gan_panel[loc_idx, t_idx].astype(bool)
                w_row = np.ones(n_features, dtype=np.float64)
                w_row[:n_variants] = np.where(
                    gan_row[:n_variants], w_gan_rows[i], 1.0
                )
                w_gan_panel[loc_idx, t_idx] = w_row

    return GANDataset(
        X_clr=X_clr.reshape(n_locs, n_times, n_features),
        counts_raw=X,
        total_sequence=total_seq_panel,
        M_observed=M_obs,
        M_fixed=M_fixed_panel,
        M_gan=M_gan_panel,
        M_padding=M_pad,
        M_row=M_row,
        row_index=row_index,
        p_nonzero=p_nz_panel,
        w_gan=w_gan_panel,
        time_decay_f=time_decay_f,
        time_decay_b=time_decay_b,
        location_index=locations,
        time_index=all_times,
        feature_names=feature_names,
        pseudo_count=pc,
    )