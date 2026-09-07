"""Mask handling for missing data imputation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class MissingnessPattern:
    """Represents a missingness pattern across variant components."""

    pattern: tuple[bool, ...]  # True = missing, False = observed
    n_rows: int = 0
    row_indices: list[int] = field(default_factory=list)

    def __hash__(self):
        return hash(self.pattern)

    def __eq__(self, other):
        if not isinstance(other, MissingnessPattern):
            return False
        return self.pattern == other.pattern

    @property
    def n_missing(self) -> int:
        return sum(self.pattern)

    @property
    def n_observed(self) -> int:
        return len(self.pattern) - self.n_missing

    def to_string(self, variant_names: list[str]) -> str:
        """Human-readable pattern string."""
        parts = []
        for i, (name, is_missing) in enumerate(zip(variant_names, self.pattern)):
            if is_missing:
                parts.append(f"{name}=?")
            else:
                parts.append(f"{name}=obs")
        return ", ".join(parts)


@dataclass
class MaskCollection:
    """Collection of masks for the dataset.

    Following Section 4.2 data contract:
    - M_observed: [location, time, feature], 1 = original observed, 0 = original missing
    - M_row: [location, time], 1 = row exists in original CSV
    - delta_f: [location, time, feature], forward time gap
    - delta_b: [location, time, feature], backward time gap
    """

    M_observed: np.ndarray  # (n_locations, n_times, n_features)
    M_row: np.ndarray  # (n_locations, n_times)
    delta_f: np.ndarray  # (n_locations, n_times, n_features)
    delta_b: np.ndarray  # (n_locations, n_times, n_features)
    location_index: pd.Index  # location names
    time_index: pd.DatetimeIndex  # time points (14-day grid)
    feature_names: list[str]  # variant component names

    def save(self, path: str | Path) -> None:
        """Save masks to NPZ file."""
        np.savez_compressed(
            path,
            M_observed=self.M_observed,
            M_row=self.M_row,
            delta_f=self.delta_f,
            delta_b=self.delta_b,
            location_index=self.location_index.to_numpy(),
            time_index=self.time_index.to_numpy(),
            feature_names=np.array(self.feature_names, dtype=object),
        )

    @classmethod
    def load(cls, path: str | Path) -> MaskCollection:
        """Load masks from NPZ file."""
        data = np.load(path, allow_pickle=True)
        return cls(
            M_observed=data["M_observed"],
            M_row=data["M_row"],
            delta_f=data["delta_f"],
            delta_b=data["delta_b"],
            location_index=pd.Index(data["location_index"]),
            time_index=pd.DatetimeIndex(data["time_index"]),
            feature_names=data["feature_names"].tolist(),
        )

    def get_original_missing_mask(self) -> np.ndarray:
        """Get mask of originally missing cells (for locking during imputation).

        Returns:
            Boolean array (n_locations, n_times, n_features) where True = originally missing.
        """
        return self.M_observed == 0


def compute_missingness_patterns(
    df: pd.DataFrame, variant_cols: list[str]
) -> dict[MissingnessPattern, MissingnessPattern]:
    """Group rows by their missingness pattern.

    Returns:
        Dict mapping pattern -> MissingnessPattern with row indices.
    """
    patterns: dict[MissingnessPattern, MissingnessPattern] = {}
    missing_mask = df[variant_cols].isna().to_numpy()

    for idx, row_mask in enumerate(missing_mask):
        pattern_tuple = tuple(row_mask.tolist())
        pattern = MissingnessPattern(pattern=pattern_tuple)
        if pattern not in patterns:
            patterns[pattern] = MissingnessPattern(pattern=pattern_tuple)
        patterns[pattern].n_rows += 1
        patterns[pattern].row_indices.append(idx)

    return patterns


def build_longitudinal_masks(
    df: pd.DataFrame,
    config,
    variant_cols: list[str],
    freq_days: int = 14,
) -> MaskCollection:
    """Build longitudinal mask tensors from DataFrame.

    Reindexes to regular 14-day grid per location.
    """
    location_col = config.location_col
    date_col = config.date_col

    locations = sorted(df[location_col].unique())

    # Build 14-day grid per location
    all_times = []
    for loc in locations:
        loc_df = df[df[location_col] == loc].sort_values(date_col)
        if len(loc_df) == 0:
            continue
        start = loc_df[date_col].min()
        end = loc_df[date_col].max()
        # Create regular grid
        grid = pd.date_range(start=start, end=end, freq=f"{freq_days}D")
        all_times.extend(grid)

    # Unique sorted time index
    time_index = pd.DatetimeIndex(sorted(set(all_times)))

    n_locations = len(locations)
    n_times = len(time_index)
    n_features = len(variant_cols)

    # Initialize tensors
    M_observed = np.zeros((n_locations, n_times, n_features), dtype=np.uint8)
    M_row = np.zeros((n_locations, n_times), dtype=np.uint8)
    delta_f = np.full((n_locations, n_times, n_features), np.nan, dtype=np.float32)
    delta_b = np.full((n_locations, n_times, n_features), np.nan, dtype=np.float32)

    # Map location and time to indices
    loc_to_idx = {loc: i for i, loc in enumerate(locations)}
    time_to_idx = {t: i for i, t in enumerate(time_index)}

    # Fill M_observed and M_row from actual data
    for _, row in df.iterrows():
        loc_idx = loc_to_idx[row[location_col]]
        # Find nearest time index (exact match expected for original rows)
        time_idx = time_to_idx.get(row[date_col])
        if time_idx is None:
            # Should not happen if grid includes all original dates
            continue

        M_row[loc_idx, time_idx] = 1
        for f_idx, col in enumerate(variant_cols):
            if pd.notna(row[col]):
                M_observed[loc_idx, time_idx, f_idx] = 1

    # Compute time gaps (delta_f, delta_b) for each location-feature
    for loc_idx, loc in enumerate(locations):
        loc_df = df[df[location_col] == loc].sort_values(date_col)
        if len(loc_df) == 0:
            continue

        # For each feature, compute gaps at observed timepoints
        for f_idx, col in enumerate(variant_cols):
            observed_times = loc_df.loc[loc_df[col].notna(), date_col]
            if len(observed_times) == 0:
                continue

            # Map to grid indices
            obs_grid_idx = [time_to_idx[t] for t in observed_times if t in time_to_idx]

            # Forward gap: days to next observed
            for i, t_idx in enumerate(obs_grid_idx):
                if i + 1 < len(obs_grid_idx):
                    next_t_idx = obs_grid_idx[i + 1]
                    gap = (time_index[next_t_idx] - time_index[t_idx]).days
                    delta_f[loc_idx, t_idx, f_idx] = gap

            # Backward gap: days from previous observed
            for i, t_idx in enumerate(obs_grid_idx):
                if i > 0:
                    prev_t_idx = obs_grid_idx[i - 1]
                    gap = (time_index[t_idx] - time_index[prev_t_idx]).days
                    delta_b[loc_idx, t_idx, f_idx] = gap

    return MaskCollection(
        M_observed=M_observed,
        M_row=M_row,
        delta_f=delta_f,
        delta_b=delta_b,
        location_index=pd.Index(locations),
        time_index=time_index,
        feature_names=variant_cols,
    )


def apply_mask_to_proportions(
    proportions: np.ndarray,
    observed_mask: np.ndarray,
    original_proportions: np.ndarray,
) -> np.ndarray:
    """Lock observed cells to their original proportions.

    Args:
        proportions: Current proportions (n_rows, n_features)
        observed_mask: Boolean mask (n_rows, n_features), True = observed
        original_proportions: Original proportions for observed cells

    Returns:
        Proportions with observed cells locked.
    """
    result = proportions.copy()
    result[observed_mask] = original_proportions[observed_mask]
    return result


def project_to_simplex(x: np.ndarray) -> np.ndarray:
    """Project vector onto probability simplex (sum=1, x>=0).

    Uses the algorithm from Wang & Carreira-Perpinan (2013).
    """
    # Handle 2D array (batch projection)
    if x.ndim == 1:
        x = x.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False

    n_features = x.shape[1]
    result = np.zeros_like(x)

    for i in range(x.shape[0]):
        v = x[i]
        # Sort descending
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - 1
        ind = np.arange(n_features) + 1
        cond = u - cssv / ind > 0
        if not cond.any():
            result[i] = np.maximum(v, 0)
            result[i] /= result[i].sum()
            continue
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / rho
        w = np.maximum(v - theta, 0)
        result[i] = w

    return result.squeeze() if squeeze else result


def restore_observed_counts(
    imputed_counts: np.ndarray,
    observed_mask: np.ndarray,
    original_counts: np.ndarray,
) -> np.ndarray:
    """Restore observed counts in imputed array.

    Args:
        imputed_counts: Imputed integer counts (n_rows, n_features)
        observed_mask: Boolean mask (n_rows, n_features), True = observed
        original_counts: Original integer counts for observed cells

    Returns:
        Counts with observed cells restored.
    """
    result = imputed_counts.copy()
    result[observed_mask] = original_counts[observed_mask]
    return result


def extract_patterns(observed_mask: np.ndarray) -> list[MissingnessPattern]:
    """Extract unique missingness patterns from an observed mask.

    Args:
        observed_mask: Boolean array (n_rows, n_features), True = observed.

    Returns:
        List of MissingnessPattern with pattern tuple where True = missing.
    """
    patterns: dict[tuple[bool, ...], MissingnessPattern] = {}
    for idx in range(observed_mask.shape[0]):
        missing_tuple = tuple((~observed_mask[idx]).tolist())
        if missing_tuple not in patterns:
            patterns[missing_tuple] = MissingnessPattern(pattern=missing_tuple)
        patterns[missing_tuple].n_rows += 1
        patterns[missing_tuple].row_indices.append(idx)
    return list(patterns.values())


def apply_mask_blending(
    imputed_values: np.ndarray,
    observed_mask: np.ndarray,
    original_values: np.ndarray,
) -> np.ndarray:
    """Blend imputed values with original observed values.

    Observed cells take original values; missing cells keep imputed values.

    Args:
        imputed_values: Imputed array (n_rows, n_features).
        observed_mask: Boolean mask (n_rows, n_features), True = observed.
        original_values: Original values for observed cells.

    Returns:
        Blended array with observed cells locked.
    """
    result = imputed_values.copy()
    result[observed_mask] = original_values[observed_mask]
    return result


def save_mask(path: str | Path, **arrays: np.ndarray) -> None:
    """Save mask arrays to an NPZ file."""
    np.savez_compressed(path, **arrays)


def load_mask(path: str | Path) -> np.lib.npyio.NpzFile:
    """Load mask arrays from an NPZ file."""
    return np.load(path, allow_pickle=True)


@dataclass
class RawMasks:
    """Raw masks derived directly from the CSV, per Section 3.1.

    Shapes are (n_raw_rows, n_variants) unless noted.
    - M_observed: 1 if raw value is not NaN (observed zero and positive both 1).
    - M_target: 1 if raw value is NaN on a row present in the CSV.
    - M_row: 1 if the (location, date) row exists in the CSV.
    - M_padding: 1 only for cells created by internal grid reindexing.
    """

    M_observed: np.ndarray
    M_target: np.ndarray
    M_row: np.ndarray
    M_padding: np.ndarray
    locations: np.ndarray
    dates: np.ndarray
    variant_order: list[str]
    raw_checksum: str = ""

    def validate(self, expected_m_target_count: int | None = None) -> None:
        if np.any(self.M_observed.astype(bool) & self.M_target.astype(bool)):
            raise ValueError("M_observed AND M_target must be empty")
        if np.any(self.M_target.astype(bool) & self.M_padding.astype(bool)):
            raise ValueError("M_target must exclude padding cells")
        if expected_m_target_count is not None:
            actual = int(self.M_target.sum())
            if actual != expected_m_target_count:
                raise ValueError(
                    f"M_target.sum() = {actual}, expected {expected_m_target_count}"
                )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            M_observed=self.M_observed,
            M_target=self.M_target,
            M_row=self.M_row,
            M_padding=self.M_padding,
            locations=self.locations,
            dates=self.dates,
            variant_order=np.array(self.variant_order, dtype=object),
            raw_checksum=self.raw_checksum,
        )

    @classmethod
    def load(cls, path: str | Path) -> RawMasks:
        data = np.load(path, allow_pickle=True)
        return cls(
            M_observed=data["M_observed"],
            M_target=data["M_target"],
            M_row=data["M_row"],
            M_padding=data["M_padding"],
            locations=data["locations"],
            dates=data["dates"],
            variant_order=data["variant_order"].tolist(),
            raw_checksum=str(data.get("raw_checksum", "")),
        )


def create_raw_masks(
    df: pd.DataFrame,
    variant_cols: list[str],
    raw_checksum: str = "",
    expected_m_target_count: int | None = None,
) -> RawMasks:
    """Build raw masks directly from the CSV DataFrame.

    M_target = isnan(X_raw) AND broadcast(M_row) AND NOT broadcast(M_padding).
    On raw CSV rows every row exists (M_row=1) and no padding is present
    (M_padding=0), so M_target equals raw NaN cells exactly.
    """
    raw_nan = df[variant_cols].isna().to_numpy()
    n_rows, n_variants = raw_nan.shape
    M_observed = (~raw_nan).astype(np.uint8)
    M_row = np.ones((n_rows, n_variants), dtype=np.uint8)
    M_padding = np.zeros((n_rows, n_variants), dtype=np.uint8)
    M_target = (raw_nan & (M_row == 1) & (M_padding == 0)).astype(np.uint8)

    masks = RawMasks(
        M_observed=M_observed,
        M_target=M_target,
        M_row=M_row,
        M_padding=M_padding,
        locations=df["location"].to_numpy() if "location" in df.columns else np.array([]),
        dates=df["date"].astype(str).to_numpy() if "date" in df.columns else np.array([]),
        variant_order=list(variant_cols),
        raw_checksum=raw_checksum,
    )
    masks.validate(expected_m_target_count)
    return masks
