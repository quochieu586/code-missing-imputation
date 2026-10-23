"""Trusted input boundaries for the baseline application layer."""
import numpy as np
from ..contracts import ConditioningPanel, EvaluationLabels, FitScope, RawPanel
from .common import BaselineError


def safe_counts(panel):
    x = np.asarray(panel.counts, dtype=float)
    mask = np.asarray(panel.conditioning_mask)
    if x.ndim != 2 or mask.shape != x.shape or mask.dtype != bool:
        raise BaselineError('Counts and boolean conditioning mask must have matching N x D shapes')
    if len(panel.row_ids) != len(x) or len(set(panel.row_ids)) != len(x):
        raise BaselineError('row_ids must be unique and aligned with counts')
    # Index/replace before validation or arithmetic: hidden values are never used.
    safe = np.where(mask, x, np.nan)
    visible = safe[mask]
    n = np.asarray(panel.totals, float)
    if n.shape != (len(x),) or not np.isfinite(n).all() or (n <= 0).any():
        raise BaselineError('Totals must be finite positive and row-aligned')
    if not np.isfinite(visible).all() or (visible < 0).any():
        raise BaselineError('Visible counts must be finite nonnegative')
    if (np.nansum(safe, axis=1) > n).any():
        raise BaselineError('Observed counts exceed total; never clamp or modify observations')
    return safe


def fit_subset(panel, scope: FitScope):
    safe_counts(panel)
    if scope.protocol not in ('inductive', 'inductive_group_by_location', 'transductive'):
        raise BaselineError('Unknown fit protocol')
    if scope.mask_id != panel.mask_id:
        raise BaselineError('Fit scope mask_id differs from conditioning panel')
    requested = scope.train_row_ids
    if not requested or len(set(requested)) != len(requested):
        raise BaselineError('FitScope needs nonempty unique permitted training row IDs')
    index = {key: i for i, key in enumerate(panel.row_ids)}
    if not set(requested) <= set(index):
        raise BaselineError('FitScope refers to rows absent from panel')
    ids = np.array([index[key] for key in requested], dtype=int)
    return subset(panel, ids)


def subset(panel, ids):
    ids = np.asarray(ids, int)
    return ConditioningPanel(
        row_ids=tuple(panel.row_ids[i] for i in ids), counts=safe_counts(panel)[ids].copy(),
        conditioning_mask=np.asarray(panel.conditioning_mask)[ids].copy(),
        totals=np.asarray(panel.totals)[ids].copy(), entity_ids=np.asarray(panel.entity_ids)[ids].copy(),
        timestamps=np.asarray(panel.timestamps)[ids].copy(),
        metadata={k: np.asarray(v)[ids].copy() for k, v in panel.metadata.items()},
        metadata_observed={k: np.asarray(v)[ids].copy() for k, v in panel.metadata_observed.items()},
        mask_id=panel.mask_id)


def hide_targets(raw: RawPanel, hidden, *, mask_id, approved_metadata=()):
    """Exclude all unapproved metadata, including target-derived scores/Other.

    Caller must establish provenance of approved columns. This cannot infer
    whether an arbitrary column is derived from target truth by its name.
    """
    x = np.asarray(raw.counts, float)
    observed, hidden = np.asarray(raw.original_observed), np.asarray(hidden)
    if observed.dtype != bool or hidden.dtype != bool or observed.shape != x.shape or hidden.shape != x.shape:
        raise BaselineError('O/H must be matching boolean masks')
    if (hidden & ~observed).any():
        raise BaselineError('Artificial targets must be a subset of originally observed cells')
    conditioning = observed & ~hidden
    meta = {k: np.asarray(raw.metadata[k]).copy() for k in approved_metadata}
    meta_mask = {}
    for key, value in meta.items():
        if len(value) != len(x):
            raise BaselineError('Metadata not aligned with raw rows')
        meta_mask[key] = np.array([v is not None and str(v).strip().lower() not in
                                  ('', 'nan', 'na', 'none') for v in value], dtype=bool)
    panel = ConditioningPanel(raw.row_ids, np.where(conditioning, x, np.nan), conditioning.copy(),
                              np.asarray(raw.totals).copy(), np.asarray(raw.entity_ids).copy(),
                              np.asarray(raw.timestamps).copy(), meta, meta_mask, mask_id)
    safe_counts(panel)
    labels = EvaluationLabels(raw.row_ids, np.where(hidden, x, np.nan), hidden.copy())
    return panel, labels


def extended_proportions(panel):
    """Derived Other is known ONLY when all raw target parts are visible."""
    raw = safe_counts(panel)
    p = raw / np.asarray(panel.totals)[:, None]
    complete = np.asarray(panel.conditioning_mask).all(axis=1)
    other = np.full(len(p), np.nan)
    # Compute in count space first. Subtracting rounded normalized proportions
    # from 1 can manufacture tiny positive Other values and poison a fitted
    # minimum-positive zero policy (e.g. delta~=1e-17 instead of a real count).
    totals = np.asarray(panel.totals, float)
    other[complete] = (totals[complete] - raw[complete].sum(axis=1)) / totals[complete]
    return np.column_stack((p, other))
