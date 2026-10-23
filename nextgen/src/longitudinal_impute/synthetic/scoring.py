"""Canonical hidden-cell metrics against sampled counts, X/N and latent p."""
import numpy as np


def score(labels, latent, panel, prediction):
    prediction = np.asarray(prediction, dtype=float)
    if labels.row_ids != panel.row_ids or prediction.shape != panel.counts.shape:
        raise ValueError('Score row identity/shape mismatch')
    if np.asarray(latent['p']).shape != prediction.shape or np.asarray(latent['strata']).shape != prediction.shape:
        raise ValueError('Score latent shape mismatch')
    truth = np.nan_to_num(np.asarray(labels.values, dtype=float), nan=0.0)
    hidden = np.asarray(labels.target_mask, dtype=bool)
    totals = np.asarray(panel.totals, dtype=float)[:, None]
    error = prediction - truth
    abs_error = np.abs(error)
    squared = error ** 2
    ratio_error = error / totals
    latent_error = prediction / totals - np.asarray(latent["p"])
    strata = np.asarray(latent["strata"])

    def metrics(mask):
        values = abs_error[mask]
        ratio_values = np.abs(ratio_error[mask])
        return {
            "n": int(mask.sum()),
            "latent_p_mae": float(np.abs(latent_error[mask]).mean()) if len(values) else None,
            "latent_p_rmse": float(np.sqrt((latent_error[mask] ** 2).mean())) if len(values) else None,
            "count_mae": float(values.mean()) if len(values) else None,
            "count_rmse": float(np.sqrt(squared[mask].mean())) if len(values) else None,
            "sample_ratio_mae": float(ratio_values.mean()) if len(values) else None,
            "sample_ratio_rmse": float(np.sqrt((ratio_error[mask] ** 2).mean())) if len(values) else None,
        }

    by_stratum = {}
    for code, name in ((0, "true_zero"), (1, "sampling_zero"), (2, "positive_count")):
        by_stratum[name] = metrics(hidden & (strata == code))

    conditioning = np.asarray(panel.conditioning_mask, dtype=bool)
    budget_excess = prediction.sum(axis=1) - np.asarray(panel.totals, dtype=float)
    return {
        "overall": metrics(hidden),
        "by_stratum": by_stratum,
        "checks": {
            "finite": bool(np.isfinite(prediction).all()),
            "nonnegative": bool((prediction >= 0).all()),
            "observed_unchanged": bool(np.array_equal(prediction[conditioning], panel.counts[conditioning])),
            "budget_violations": int((budget_excess > 1e-8).sum()),
            "max_budget_excess": float(max(0.0, budget_excess.max(initial=0.0))),
        },
    }

