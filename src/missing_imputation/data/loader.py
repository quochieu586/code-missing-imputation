"""Data loading and preprocessing for covariants dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .schema import (
    AuditReport,
    DataConfig,
    compute_checksum,
    validate_dataframe,
)


VARIANT_COMPONENTS = [
    "recombinant",
    "20A",
    "20B",
    "20C",
    "20E",
    "Beta",
    "Alpha",
    "Gamma",
    "Delta",
    "Kappa",
    "Epsilon",
    "Eta",
    "Iota",
    "Lambda",
    "Mu",
    "Omicron",
    "S:677",
]


def load_config(config_path: str | Path) -> DataConfig:
    """Load data configuration from YAML.

    variant_components, derived, validation and time_grid are top-level sections
    in configs/data.yaml, not children of `data:`. Reading them off `data:` made
    every one of them silently fall back to a hardcoded default, so edits to the
    YAML had no effect. Both placements are accepted, top level winning.
    """
    with Path(config_path).open() as f:
        cfg = yaml.safe_load(f) or {}
    data_cfg = cfg.get("data", {})

    def section(name: str) -> dict:
        value = cfg.get(name, data_cfg.get(name, {}))
        return value if isinstance(value, dict) else {}

    variant_components = cfg.get(
        "variant_components", data_cfg.get("variant_components", VARIANT_COMPONENTS)
    )
    return DataConfig(
        path=Path(data_cfg.get("path", "data/covariants.csv")),
        index_cols=data_cfg.get("index_cols", ["location", "date"]),
        total_sequence_col=data_cfg.get("total_sequence_col", "total_sequence"),
        date_col=data_cfg.get("date_col", "date"),
        location_col=data_cfg.get("location_col", "location"),
        variant_components=list(variant_components),
        other_col=section("derived").get("other_col", "other"),
        freq_days=section("time_grid").get("freq_days", 14),
        validation=section("validation"),
    )


def load_covariants(
    path_or_config: str | Path | DataConfig | None = None,
    config: DataConfig | None = None,
    config_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load and validate covariants.csv.

    Args:
        path_or_config: CSV path, DataConfig, or None.
        config: DataConfig object (optional).
        config_path: Path to YAML config file (optional).

    Returns:
        Validated DataFrame with proper dtypes and sorted index.
    """
    explicit_path: Path | None = None
    if isinstance(path_or_config, DataConfig):
        config = path_or_config
    elif path_or_config is not None:
        explicit_path = Path(path_or_config)

    if config is None:
        if config_path is None:
            config_path = "configs/data.yaml"
        config = load_config(config_path)

    path = explicit_path if explicit_path is not None else config.path
    if not path.is_absolute():
        # Resolve relative to project root
        project_root = Path(__file__).parents[3]
        path = project_root / path

    # Load CSV
    df = pd.read_csv(path)

    # Parse date column
    df[config.date_col] = pd.to_datetime(df[config.date_col])

    # Ensure variant columns are float (NaN for missing)
    for col in config.variant_components:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure total_sequence is int
    df[config.total_sequence_col] = df[config.total_sequence_col].astype("int64")

    # Sort by location, date
    df = df.sort_values([config.location_col, config.date_col]).reset_index(drop=True)

    # Validate
    is_valid, errors = validate_dataframe(df, config)
    if not is_valid:
        raise ValueError("Data validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    return df


def compute_other(df: pd.DataFrame, config: DataConfig) -> pd.Series:
    """Compute 'other' component: total_sequence - sum(17 variants).

    For rows with any missing variant, other is also NaN (unknown).
    """
    variant_sum = df[config.variant_components].sum(axis=1, skipna=True)
    any_missing = df[config.variant_components].isna().any(axis=1)
    other = df[config.total_sequence_col] - variant_sum
    other = other.where(~any_missing, np.nan)
    return other


def add_other_column(df: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    """Add 'other' column to DataFrame."""
    df = df.copy()
    df[config.other_col] = compute_other(df, config)
    return df


def get_missing_mask(df: pd.DataFrame, variant_cols: list[str]) -> np.ndarray:
    """Get boolean mask of missing values for variant components.

    Returns:
        2D array (n_rows, n_variants) where True = missing.
    """
    return df[variant_cols].isna().to_numpy()


def get_observed_mask(df: pd.DataFrame, variant_cols: list[str]) -> np.ndarray:
    """Get boolean mask of observed (non-missing) values for variant components.

    Returns:
        2D array (n_rows, n_variants) where True = observed.
    """
    return df[variant_cols].notna().to_numpy()


def split_complete_incomplete(df: pd.DataFrame, variant_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into complete rows (no missing variants) and incomplete rows."""
    mask_complete = df[variant_cols].notna().all(axis=1)
    complete = df[mask_complete].copy().reset_index(drop=True)
    incomplete = df[~mask_complete].copy().reset_index(drop=True)
    return complete, incomplete


def audit_data(df: pd.DataFrame, config: DataConfig, checksum: str | None = None) -> AuditReport:
    """Generate comprehensive data audit report.

    The schema checks are run here and recorded on the report. Leaving
    validation_errors empty made AuditReport.is_valid unconditionally True, so
    the audit always claimed "All validation checks passed" whatever the data.
    """
    variant_cols = config.variant_components
    _, validation_errors = validate_dataframe(df, config)

    # Basic stats
    n_rows, n_cols = df.shape
    n_locations = df[config.location_col].nunique()
    n_timepoints = df[config.date_col].nunique()
    date_range = (df[config.date_col].min(), df[config.date_col].max())

    # Missingness
    missing_mask = get_missing_mask(df, variant_cols)
    n_missing_cells = int(missing_mask.sum())
    missing_cell_rate = n_missing_cells / (n_rows * len(variant_cols))

    n_rows_with_missing = int(missing_mask.any(axis=1).sum())
    n_complete_rows = n_rows - n_rows_with_missing

    # Time gaps per location
    time_gaps = []
    for loc in df[config.location_col].unique():
        loc_dates = df[df[config.location_col] == loc][config.date_col].sort_values()
        if len(loc_dates) > 1:
            gaps = loc_dates.diff().dt.days.dropna()
            time_gaps.extend(gaps.tolist())
    median_time_gap = float(np.median(time_gaps)) if time_gaps else 0.0
    max_time_gap = float(np.max(time_gaps)) if time_gaps else 0.0

    # Zero prevalence per variant
    zero_prevalence = {}
    for col in variant_cols:
        observed = df[col].dropna()
        if len(observed) > 0:
            zero_prevalence[col] = float((observed == 0).sum() / len(observed))
        else:
            zero_prevalence[col] = 0.0

    return AuditReport(
        n_rows=n_rows,
        n_cols=n_cols,
        n_locations=n_locations,
        n_timepoints=n_timepoints,
        date_range=date_range,
        n_variant_components=len(variant_cols),
        n_missing_cells=n_missing_cells,
        missing_cell_rate=missing_cell_rate,
        n_rows_with_missing=n_rows_with_missing,
        n_complete_rows=n_complete_rows,
        median_time_gap_days=median_time_gap,
        max_time_gap_days=max_time_gap,
        zero_prevalence_per_variant=zero_prevalence,
        checksum=checksum or "",
        validation_errors=validation_errors,
    )


def save_audit_report(report: AuditReport, output_path: str | Path) -> None:
    """Save audit report as markdown."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.to_markdown(), encoding="utf-8")


def run_audit(
    df: pd.DataFrame,
    config: DataConfig,
    data_path: str | Path | None = None,
) -> AuditReport:
    """Run full data audit on a DataFrame or CSV path."""
    if data_path is not None:
        path = Path(data_path)
        checksum = compute_checksum(path)
    else:
        checksum = None
    return audit_data(df, config, checksum=checksum)