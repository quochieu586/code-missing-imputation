"""Inverse CLR and constrained closure postprocessing for GAN output."""

from __future__ import annotations

import numpy as np

from .clr import inverse_clr_transform, closure_project_counts


def postprocess_gan_output(
    final_clr: np.ndarray,
    dataset,
    apply_soft_weights: bool = True,
) -> np.ndarray:
    """Inverse-CLR the generator output, apply the ZPGF weight, then project closure.

    Plan S18.6 puts the spike-and-slab posterior mean here: the generator produces a
    magnitude m and the fusion stage supplies w_gan, so the value written back is
    y_hat = w_gan * m, applied BEFORE closure projection so the residual budget is
    redistributed around the down-weighted cells. Weights are 1.0 outside M_gan, so
    observed and hard-locked cells are unaffected.

    apply_soft_weights=False reproduces the ungated behaviour for the S12 ablation.
    """
    n_feats = len(dataset.feature_names)
    counts_candidate = inverse_clr_transform(final_clr, dataset.pseudo_count, n_features=n_feats)

    if apply_soft_weights and getattr(dataset, "w_gan", None) is not None:
        counts_candidate = counts_candidate * dataset.w_gan

    # The last feature is the residual "other". It must stay free during closure
    # projection: it is what absorbs the mass that down-weighting removes from a
    # cell. Locking it would force the remaining budget back onto the M_gan cells
    # and make the ZPGF weight inert.
    lock_mask = dataset.M_fixed.astype(bool).copy()
    lock_mask[..., -1] = False
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