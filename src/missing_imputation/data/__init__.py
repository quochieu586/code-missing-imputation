"""Data handling: schema, loading, masks, panel, closure."""

from .closure import (
    compute_other,
    counts_to_proportions,
    enforce_closure,
    proportions_to_counts_largest_remainder,
    validate_closure,
)
from .loader import load_config, load_covariants, run_audit, save_audit_report
from .masks import (
    MissingnessPattern,
    apply_mask_blending,
    extract_patterns,
    load_mask,
    save_mask,
)
from .panel import (
    PanelTensor,
    build_panel,
    panel_to_proportions,
    proportions_to_panel,
)
from .schema import AuditReport, CovariantsRow, DataConfig, validate_dataframe

__all__ = [
    "AuditReport",
    "CovariantsRow",
    "DataConfig",
    "MissingnessPattern",
    "PanelTensor",
    "apply_mask_blending",
    "build_panel",
    "compute_other",
    "counts_to_proportions",
    "enforce_closure",
    "extract_patterns",
    "load_config",
    "load_covariants",
    "load_mask",
    "panel_to_proportions",
    "proportions_to_counts_largest_remainder",
    "proportions_to_panel",
    "run_audit",
    "save_audit_report",
    "save_mask",
    "validate_closure",
    "validate_dataframe",
]