"""Per-feature distribution checks for native imputation without labels."""
import numpy as np


def wasserstein(a, b):
    """Exact one-dimensional Wasserstein-1 for equal-weight samples."""
    a, b = np.sort(a), np.sort(b)
    if not len(a) or not len(b):
        return None
    edges = np.sort(np.r_[a, b])
    cdf_a = np.searchsorted(a, edges[:-1], side='right') / len(a)
    cdf_b = np.searchsorted(b, edges[:-1], side='right') / len(b)
    return float(np.sum(np.abs(cdf_a - cdf_b) * np.diff(edges)))


def distribution_scores(raw, completed):
    scores = {}
    for j, name in enumerate(raw.targets):
        observed = np.isfinite(raw.counts[:, j])
        before, after = raw.counts[observed, j], completed[:, j]
        scores[name] = {
            'observed_cells': int(observed.sum()),
            'imputed_cells': int((~observed).sum()),
            'wasserstein_before_after': wasserstein(before, after),
            'wasserstein_observed_imputed': wasserstein(before, after[~observed]),
        }
    return {'distribution_counts': scores}
