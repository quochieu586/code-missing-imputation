"""CLR transform utilities for internal GAN representation.

Uses pseudo-count on proportions (not counts) with proper renormalization
to ensure invertibility.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def clr_transform(
    counts: np.ndarray,
    pseudo_count: float | None = None,
    total_sequence: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Apply CLR transform to counts or proportions.

    If total_sequence is provided, converts counts to proportions first.
    Adds pseudo_count to proportions and renormalizes to ensure sum=1.
    Returns (clr_values, pseudo_count).
    """
    counts = np.asarray(counts, dtype=np.float64)

    if total_sequence is not None:
        total_sequence = np.asarray(total_sequence, dtype=np.float64)
        total_sequence = np.where(total_sequence == 0, 1.0, total_sequence)
        proportions = counts / total_sequence[..., np.newaxis]
    else:
        # Assume input is already proportions
        proportions = counts

    n_features = proportions.shape[-1]

    if pseudo_count is None:
        positive = proportions[proportions > 0]
        pseudo_count = float(positive.min() / 2.0) if len(positive) > 0 else 0.5 / n_features

    # Add pseudo-count and renormalize to sum=1
    shifted = proportions + pseudo_count
    row_sums = shifted.sum(axis=-1, keepdims=True)
    shifted = shifted / row_sums

    log_shifted = np.log(shifted)
    geo_mean = log_shifted.mean(axis=-1, keepdims=True)
    clr_values = log_shifted - geo_mean

    # Return pseudo_count used; store n_features for inverse if needed
    return clr_values, pseudo_count


def inverse_clr_transform(
    clr_values: np.ndarray,
    pseudo_count: float,
    n_features: int | None = None,
) -> np.ndarray:
    """Inverse CLR: softmax then remove pseudo-count and renormalize.

    Input: clr_values from clr_transform
    Output: proportions (summing to 1 per row)
    
    The forward transform does:
    1. shifted = (proportions + c) / (1 + k*c) where k=n_features, c=pseudo_count
    2. clr = log(shifted) - mean(log(shifted))
    
    The inverse:
    1. shifted_props = softmax(clr_values)  # recovers the shifted proportions
    2. original_props = shifted_props * (1 + k*c) - c
    3. Renormalize to handle numerical errors
    """
    clr_values = np.asarray(clr_values, dtype=np.float64)
    if n_features is None:
        n_features = clr_values.shape[-1]

    # The composition lives on the LAST axis. Callers pass either a 2-D
    # (row, feature) matrix or a 3-D (location, time, feature) panel; using a
    # fixed axis=1 silently normalised the panel across time instead of across
    # features, which is not a composition at all.
    log_exp_sum = logsumexp(clr_values, axis=-1, keepdims=True)
    shifted_props = np.exp(clr_values - log_exp_sum)

    # Remove pseudo-count and renormalize
    # shifted_props = (original_props + c) / (1 + k*c)
    # original_props = shifted_props * (1 + k*c) - c
    k_c = 1.0 + n_features * pseudo_count
    original_props = shifted_props * k_c - pseudo_count
    original_props = np.maximum(original_props, 0.0)

    # Renormalize to handle numerical errors
    row_sums = original_props.sum(axis=-1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return original_props / row_sums


def closure_project_counts(
    candidate_counts: np.ndarray,
    total_sequence: np.ndarray,
    lock_mask: np.ndarray,
    locked_counts: np.ndarray,
) -> np.ndarray:
    from ..data.closure import allocate_with_lock

    total_sequence = np.asarray(total_sequence, dtype=np.float64)
    n_rows = candidate_counts.shape[0]
    proportions = np.zeros_like(candidate_counts, dtype=np.float64)
    for i in range(n_rows):
        s = candidate_counts[i].sum()
        proportions[i] = candidate_counts[i] / s if s > 0 else np.full_like(
            candidate_counts[i], 1.0 / candidate_counts.shape[1]
        )
    return allocate_with_lock(proportions, total_sequence, lock_mask, locked_counts)