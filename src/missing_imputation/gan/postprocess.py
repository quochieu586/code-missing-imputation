"""Inverse CLR and constrained closure postprocessing for GAN output."""

from __future__ import annotations

import numpy as np

from .clr import inverse_clr_transform, closure_project_counts


def postprocess_gan_output(
    final_clr: np.ndarray,
    dataset,
) -> np.ndarray:
    n_feats = len(dataset.feature_names)
    counts_candidate = inverse_clr_transform(final_clr, dataset.pseudo_count, n_features=n_feats)

    lock_mask = dataset.M_fixed.astype(bool)
    locked_counts = dataset.counts_raw.copy()

    n_locs, n_times, n_feats = counts_candidate.shape
    final_counts = np.zeros_like(counts_candidate, dtype=np.int64)

    for loc in range(n_locs):
        for t in range(n_times):
            # Skip padded grid points (M_row == 0 means not in original data)
            if dataset.M_row[loc, t] == 0:
                continue
            row_total = int(dataset.total_sequence[loc, t])
            if row_total <= 0:
                continue
            row_candidate = counts_candidate[loc, t]
            row_lock = lock_mask[loc, t]
            row_locked = locked_counts[loc, t]

            projected = closure_project_counts(
                row_candidate[None, :],
                np.array([row_total]),
                row_lock[None, :],
                row_locked[None, :],
            )
            final_counts[loc, t] = projected[0]

    for loc in range(n_locs):
        for t in range(n_times):
            if dataset.M_row[loc, t] == 0:
                continue
            lm = lock_mask[loc, t]
            final_counts[loc, t][lm] = dataset.counts_raw[loc, t][lm].astype(np.int64)

    assert np.all(final_counts >= 0), "Negative counts after postprocessing"
    for loc in range(n_locs):
        for t in range(n_times):
            if dataset.M_row[loc, t] == 0:
                continue
            # Closure: sum of all features (including "other") should equal total_sequence
            assert final_counts[loc, t].sum() == int(dataset.total_sequence[loc, t]), (
                f"Closure violation at ({loc}, {t}): sum={final_counts[loc, t].sum()}, total={int(dataset.total_sequence[loc, t])}"
            )

    return final_counts