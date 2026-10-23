"""Interface contracts only; array validation/ownership is a future implementation.

Raw tables are N x D; model tensors are B x T x D. Array fields use Any until
the numerical backend is selected. Frozen dataclasses do NOT freeze arrays.
Do not expose EvaluationLabels or the original raw table to prediction code.
"""
from dataclasses import dataclass
from typing import Any, Mapping

Array = Any


@dataclass(frozen=True)
class DatasetSchema:
    entity_column: str
    time_column: str
    total_column: str
    target_columns: tuple[str, ...]
    metadata_columns: tuple[str, ...]


@dataclass(frozen=True)
class RawPanel:
    """Trusted boundary only. Preserve raw observed counts including zeros."""
    row_ids: tuple[str, ...]
    counts: Array
    original_observed: Array
    totals: Array
    entity_ids: Array
    timestamps: Array
    metadata: Mapping[str, Array]
    schema: DatasetSchema


@dataclass(frozen=True)
class ConditioningPanel:
    """Values hidden BEFORE any fit/transform; NaN outside conditioning_mask.

    Model code receives this object, not RawPanel. Metadata must have passed
    provenance review and masking of dependent derived variables.
    """
    row_ids: tuple[str, ...]
    counts: Array
    conditioning_mask: Array
    totals: Array
    entity_ids: Array
    timestamps: Array
    metadata: Mapping[str, Array]
    metadata_observed: Mapping[str, Array]
    mask_id: str


@dataclass(frozen=True)
class EvaluationLabels:
    """Only trainer/scorer may access; targets are originally observed cells."""
    row_ids: tuple[str, ...]
    values: Array
    target_mask: Array


@dataclass(frozen=True)
class FitScope:
    """Record the permitted fitting population, including transductive cases."""
    protocol: str
    train_row_ids: tuple[str, ...]
    split_id: str
    mask_id: str


@dataclass(frozen=True)
class InitializedPanel:
    counts: Array
    fallback_mask: Array
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class ModelBatch:
    """No true held-out counts. padding_mask=True means a padded time step.

    Conditioning mask is in raw feature space, never copied to ILR coordinates.
    Metadata tensors include missing indicators and train-fitted encodings.
    """
    features: Array
    conditioning_mask: Array
    timestamps: Array
    padding_mask: Array
    metadata: Array
    row_ids: tuple[str, ...]


@dataclass(frozen=True)
class ImputationResult:
    """Expected counts; integer rounding is a separate optional export policy."""
    row_ids: tuple[str, ...]
    counts: Array
    diagnostics: Mapping[str, Any]
