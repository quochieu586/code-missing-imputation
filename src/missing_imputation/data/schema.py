"""Data schemas and validation for covariants dataset."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator


class CovariantsRow(BaseModel):
    """Schema for a single row of covariants.csv."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    location: str = Field(..., min_length=1)
    date: pd.Timestamp
    total_sequence: int = Field(..., gt=0)
    recombinant: float | None = None
    var_20A: float | None = Field(None, alias="20A")
    var_20B: float | None = Field(None, alias="20B")
    var_20C: float | None = Field(None, alias="20C")
    var_20E: float | None = Field(None, alias="20E")
    Beta: float | None = None
    Alpha: float | None = None
    Gamma: float | None = None
    Delta: float | None = None
    Kappa: float | None = None
    Epsilon: float | None = None
    Eta: float | None = None
    Iota: float | None = None
    Lambda: float | None = None
    Mu: float | None = None
    Omicron: float | None = None
    S_677: float | None = Field(None, alias="S:677")

    @field_validator("date", mode="before")
    @classmethod
    def parse_date(cls, v: str | pd.Timestamp) -> pd.Timestamp:
        if isinstance(v, str):
            return pd.Timestamp(v)
        return v

    @field_validator(
        "recombinant",
        "var_20A",
        "var_20B",
        "var_20C",
        "var_20E",
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
        "S_677",
        mode="before",
    )
    @classmethod
    def validate_count(cls, v: float | None) -> float | None:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        if v < 0:
            raise ValueError("Variant counts must be non-negative")
        return float(v)

    def get_variant_counts(self) -> dict[str, float | None]:
        """Return variant counts as dict with original column names."""
        return {
            "recombinant": self.recombinant,
            "20A": self.var_20A,
            "20B": self.var_20B,
            "20C": self.var_20C,
            "20E": self.var_20E,
            "Beta": self.Beta,
            "Alpha": self.Alpha,
            "Gamma": self.Gamma,
            "Delta": self.Delta,
            "Kappa": self.Kappa,
            "Epsilon": self.Epsilon,
            "Eta": self.Eta,
            "Iota": self.Iota,
            "Lambda": self.Lambda,
            "Mu": self.Mu,
            "Omicron": self.Omicron,
            "S:677": self.S_677,
        }


class DataConfig(BaseModel):
    """Configuration for data loading and validation."""

    path: Path
    index_cols: list[str] = ["location", "date"]
    total_sequence_col: str = "total_sequence"
    date_col: str = "date"
    location_col: str = "location"
    variant_components: list[str] = Field(
        default_factory=lambda: [
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
    )
    other_col: str = "other"
    freq_days: int = 14

    validation: dict = Field(
        default_factory=lambda: {
            "total_sequence_min": 1,
            "allow_negative_counts": False,
            "allow_observed_sum_exceed_total": False,
            "require_unique_index": True,
        }
    )


class AuditReport(BaseModel):
    """Data audit report."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    n_rows: int
    n_cols: int
    n_locations: int
    n_timepoints: int
    date_range: tuple[pd.Timestamp, pd.Timestamp]
    n_variant_components: int
    n_missing_cells: int
    missing_cell_rate: float
    n_rows_with_missing: int
    n_complete_rows: int
    median_time_gap_days: float
    max_time_gap_days: float
    zero_prevalence_per_variant: dict[str, float]
    checksum: str
    validation_errors: list[str] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.validation_errors) == 0

    @property
    def errors(self) -> list[str]:
        return self.validation_errors

    def to_markdown(self) -> str:
        """Convert to markdown report."""
        lines = [
            "# Data Audit Report",
            "",
            "## Summary",
            f"- **Rows**: {self.n_rows:,}",
            f"- **Columns**: {self.n_cols}",
            f"- **Locations**: {self.n_locations}",
            f"- **Timepoints**: {self.n_timepoints}",
            f"- **Date range**: {self.date_range[0].date()} to {self.date_range[1].date()}",
            f"- **Variant components**: {self.n_variant_components}",
            f"- **Missing cells**: {self.n_missing_cells:,} ({self.missing_cell_rate:.2%})",
            f"- **Rows with ≥1 missing**: {self.n_rows_with_missing:,} ({self.n_rows_with_missing/self.n_rows:.2%})",
            f"- **Complete rows**: {self.n_complete_rows:,}",
            f"- **Median time gap**: {self.median_time_gap_days:.1f} days",
            f"- **Max time gap**: {self.max_time_gap_days:.1f} days",
            f"- **SHA256 checksum**: `{self.checksum}`",
            "",
            "## Zero Prevalence by Variant",
            "",
            "| Variant | Zero Rate |",
            "|---------|-----------|",
        ]
        for var, rate in self.zero_prevalence_per_variant.items():
            lines.append(f"| {var} | {rate:.2%} |")
        lines.append("")

        if self.validation_errors:
            lines.append("## Validation Errors")
            lines.append("")
            for err in self.validation_errors:
                lines.append(f"- {err}")
        else:
            lines.append("## Validation")
            lines.append("")
            lines.append("All validation checks passed.")

        return "\n".join(lines)


def compute_checksum(path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_dataframe(df: pd.DataFrame, config: DataConfig) -> tuple[bool, list[str]]:
    """Validate DataFrame against schema. Returns (is_valid, errors)."""
    errors = []

    # Check required columns
    required = config.index_cols + [config.total_sequence_col] + config.variant_components
    missing_cols = set(required) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    # Check unique index
    if config.validation.get("require_unique_index", True):
        dup = df.duplicated(subset=config.index_cols, keep=False)
        if dup.any():
            n_dup = dup.sum()
            errors.append(f"Duplicate (location, date) keys: {n_dup} rows")

    # Check total_sequence > 0
    if (df[config.total_sequence_col] <= 0).any():
        errors.append("total_sequence must be > 0")

    # Check variant counts non-negative
    if not config.validation.get("allow_negative_counts", False):
        for col in config.variant_components:
            if col in df.columns:
                neg = df[col].dropna()
                if (neg < 0).any():
                    errors.append(f"Negative values in {col}")

    # Check observed sum <= total_sequence
    if not config.validation.get("allow_observed_sum_exceed_total", False):
        variant_sum = df[config.variant_components].sum(axis=1, skipna=True)
        exceed = variant_sum > df[config.total_sequence_col]
        if exceed.any():
            errors.append(f"Observed variant sum exceeds total_sequence in {exceed.sum()} rows")

    return len(errors) == 0, errors