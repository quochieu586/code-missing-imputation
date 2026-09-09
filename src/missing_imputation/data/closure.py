"""Count/proportion conversion and largest-remainder closure."""

from __future__ import annotations

import numpy as np


def counts_to_proportions(
    counts: np.ndarray, total_sequence: np.ndarray
) -> np.ndarray:
    """Convert integer counts to proportions.

    Args:
        counts: Integer counts array (n_rows, n_features).
        total_sequence: Total per row (n_rows,).

    Returns:
        Proportions array (n_rows, n_features), summing to 1 per row.
    """
    total_sequence = np.asarray(total_sequence, dtype=np.float64)
    total_sequence = np.where(total_sequence == 0, 1.0, total_sequence)
    return counts.astype(np.float64) / total_sequence[:, np.newaxis]


def build_full_composition(
    df,
    variant_cols: list[str],
    total_sequence_col: str = "total_sequence",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the full (n_variants + 1)-part composition including the residual.

    The 17 variants do NOT close on their own: `other = total_sequence - sum(17)`
    carries real mass (44% of the complete rows have other > 0). Imputing only the
    17 variants and then renormalising them to sum 1 silently forces other to 0
    on every imputed row, which contradicts plan S3 ("other chua biet va la
    residual trong closure"). Treating `other` as the last component of the
    composition lets the neighbour pool close exactly and lets kNN impute the
    residual alongside the variants.

    `other` is observed only when all 17 variants are observed; on any row with a
    missing variant the residual is unknown and is imputed too.

    Returns:
        counts:      (n_rows, n_variants + 1) int64, NaN -> 0, last column = other
        observed:    (n_rows, n_variants + 1) bool, last column = row is complete
        proportions: (n_rows, n_variants + 1) float64, counts / total_sequence
        total_seq:   (n_rows,) float64
    """
    variant_counts = df[variant_cols].fillna(0).to_numpy().astype(np.int64)
    variant_observed = df[variant_cols].notna().to_numpy()
    total_seq = df[total_sequence_col].to_numpy(dtype=np.float64)

    row_complete = variant_observed.all(axis=1)
    other = np.where(
        row_complete,
        total_seq.astype(np.int64) - variant_counts.sum(axis=1),
        0,
    ).astype(np.int64)

    counts = np.column_stack([variant_counts, other])
    observed = np.column_stack([variant_observed, row_complete])
    proportions = counts_to_proportions(counts, total_seq)
    return counts, observed, proportions, total_seq


def proportions_to_counts_largest_remainder(
    proportions: np.ndarray,
    total_sequence: np.ndarray,
    observed_mask: np.ndarray | None = None,
    observed_counts: np.ndarray | None = None,
) -> np.ndarray:
    """Convert proportions to integer counts using largest-remainder method.

    Ensures sum(counts) == total_sequence for each row.
    Observed cells are locked to their original values.

    Args:
        proportions: Proportion array (n_rows, n_features).
        total_sequence: Total per row (n_rows,).
        observed_mask: Boolean mask (n_rows, n_features), True = observed.
        observed_counts: Original counts for observed cells.

    Returns:
        Integer counts array (n_rows, n_features).
    """
    n_rows, n_features = proportions.shape
    total_sequence = np.asarray(total_sequence, dtype=np.float64)
    result = np.zeros((n_rows, n_features), dtype=np.int64)

    if observed_mask is not None and observed_counts is not None:
        result[observed_mask] = observed_counts[observed_mask].astype(np.int64)

    for i in range(n_rows):
        if observed_mask is not None:
            obs_sum = result[i, observed_mask[i]].sum()
            remaining = int(round(total_sequence[i])) - obs_sum
            missing_idx = np.where(~observed_mask[i])[0]
        else:
            remaining = int(round(total_sequence[i]))
            missing_idx = np.arange(n_features)

        if len(missing_idx) == 0:
            continue

        if remaining <= 0:
            result[i, missing_idx] = 0
            continue

        props_missing = proportions[i, missing_idx]
        props_sum = props_missing.sum()
        if props_sum <= 0:
            alloc = np.zeros(len(missing_idx), dtype=np.float64)
            alloc[0] = remaining
        else:
            alloc = props_missing / props_sum * remaining

        floors = np.floor(alloc).astype(np.int64)
        remainders = alloc - floors
        deficit = remaining - floors.sum()

        if deficit > 0:
            top_indices = np.argsort(-remainders)[:deficit]
            floors[top_indices] += 1

        result[i, missing_idx] = floors

    return result


def allocate_with_lock(
    proportions: np.ndarray,
    total_sequence: np.ndarray,
    lock_mask: np.ndarray,
    locked_counts: np.ndarray,
) -> np.ndarray:
    """Generalized largest-remainder allocation with arbitrary locked cells.

    lock_mask may include observed cells, occurrence confident-zeros, or any
    combination. Locked cells keep locked_counts exactly; the remaining budget
    per row is distributed over unlocked cells by largest remainder.
    """
    n_rows, n_features = proportions.shape
    total_sequence = np.asarray(total_sequence, dtype=np.float64)
    result = np.zeros((n_rows, n_features), dtype=np.int64)

    lock_mask = lock_mask.astype(bool)
    result[lock_mask] = locked_counts[lock_mask].astype(np.int64)

    for i in range(n_rows):
        locked_sum = result[i, lock_mask[i]].sum()
        remaining = int(round(total_sequence[i])) - locked_sum
        free_idx = np.where(~lock_mask[i])[0]

        if len(free_idx) == 0:
            continue

        if remaining <= 0:
            result[i, free_idx] = 0
            continue

        props_free = proportions[i, free_idx]
        props_sum = props_free.sum()
        if props_sum <= 0:
            alloc = np.zeros(len(free_idx), dtype=np.float64)
            alloc[0] = remaining
        else:
            alloc = props_free / props_sum * remaining

        floors = np.floor(alloc).astype(np.int64)
        remainders = alloc - floors
        deficit = remaining - floors.sum()

        if deficit > 0:
            top_indices = np.argsort(-remainders)[:deficit]
            floors[top_indices] += 1

        result[i, free_idx] = floors

    return result


def enforce_closure(
    counts: np.ndarray,
    total_sequence: np.ndarray,
    observed_mask: np.ndarray,
    observed_counts: np.ndarray,
) -> np.ndarray:
    """Enforce that sum(counts) == total_sequence per row.

    Adjusts missing cells only; observed cells are never modified.

    Args:
        counts: Current counts (n_rows, n_features).
        total_sequence: Total per row (n_rows,).
        observed_mask: Boolean mask, True = observed.
        observed_counts: Original observed counts.

    Returns:
        Adjusted counts satisfying closure.
    """
    result = counts.copy()
    result[observed_mask] = observed_counts[observed_mask].astype(np.int64)

    n_rows = counts.shape[0]
    for i in range(n_rows):
        total = int(round(total_sequence[i]))
        obs_sum = result[i, observed_mask[i]].sum()
        remaining = total - obs_sum
        missing_idx = np.where(~observed_mask[i])[0]

        if len(missing_idx) == 0:
            continue

        current_sum = result[i, missing_idx].sum()
        if current_sum == remaining:
            continue

        if remaining <= 0:
            result[i, missing_idx] = 0
            continue

        diff = remaining - current_sum
        if diff > 0:
            result[i, missing_idx[0]] += diff
        else:
            for j in missing_idx:
                subtract = min(result[i, j], -diff)
                result[i, j] -= subtract
                diff += subtract
                if diff >= 0:
                    break

    return result


def compute_other(
    counts: np.ndarray, total_sequence: np.ndarray
) -> np.ndarray:
    """Compute the 'other' component: total_sequence - sum(variant counts).

    Args:
        counts: Variant counts (n_rows, n_features).
        total_sequence: Total per row (n_rows,).

    Returns:
        Other component (n_rows,).
    """
    return total_sequence.astype(np.int64) - counts.sum(axis=1).astype(np.int64)


def validate_closure(
    counts: np.ndarray,
    total_sequence: np.ndarray,
    other: np.ndarray | None = None,
) -> tuple[bool, int]:
    """Validate closure invariant: sum(variants) + other == total_sequence.

    Also validates that all counts (including other) are non-negative.

    Args:
        counts: Variant counts (n_rows, n_features).
        total_sequence: Total per row (n_rows,).
        other: Other component (n_rows,). If None, computed from counts.

    Returns:
        (is_valid, n_violations).
    """
    if other is None:
        other = compute_other(counts, total_sequence)

    variant_sum = counts.sum(axis=1).astype(np.int64)
    total = total_sequence.astype(np.int64)
    closure_violations = variant_sum + other != total
    other_nonneg = other >= 0
    counts_nonneg = np.all(counts >= 0, axis=1)

    all_violations = closure_violations | ~other_nonneg | ~counts_nonneg
    n_violations = int(all_violations.sum())
    return n_violations == 0, n_violations
