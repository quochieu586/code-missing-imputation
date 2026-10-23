"""Sparse-label scores, supported relationship diagnostics and location stats."""
import numpy as np


def _mean(value):
    return float(np.mean(value)) if len(value) else None


def score(raw, row_ids, prediction, hidden):
    ids = np.asarray(row_ids, int)
    truth = np.asarray(raw.counts)[ids]
    target = np.asarray(hidden)[ids]
    p = np.asarray(prediction, float)
    if p.shape != truth.shape or not np.isfinite(p).all():
        raise ValueError('Prediction is nonfinite or misaligned')
    original = np.asarray(raw.original_observed)[ids]
    conditioning = original & ~target
    total = np.asarray(raw.totals)[ids]
    errors = np.zeros_like(p)
    errors[target] = p[target] - truth[target]
    ratio_errors = errors / total[:, None]
    absolute, squared = np.abs(errors), errors ** 2
    zero = target & (truth == 0)
    positive = target & (truth > 0)
    def masked(mask):
        return {'n': int(mask.sum()), 'count_mae': _mean(absolute[mask]),
                'count_rmse': float(np.sqrt(squared[mask].mean())) if mask.any() else None,
                'ratio_mae': _mean(np.abs(ratio_errors[mask]))}
    by_location = []
    entities = np.asarray(raw.entity_ids)[ids]
    for entity in sorted(set(entities)):
        rows = entities == entity
        mask = target[rows]
        by_location.append({'location': str(entity), 'n': int(mask.sum()),
                            'absolute_sum': float(absolute[rows][mask].sum()),
                            'squared_sum': float(squared[rows][mask].sum()),
                            'ratio_absolute_sum': float(np.abs(ratio_errors[rows][mask]).sum()),
                            'ratio_squared_sum': float((ratio_errors[rows][mask] ** 2).sum())})
    per_feature = [dict(feature=raw.schema.target_columns[j], **masked(target &
                   (np.arange(p.shape[1]) == j)[None, :])) for j in range(p.shape[1])]
    full = original.all(axis=1) & target.any(axis=1)
    coverage = {'supported_rows': int(full.sum()), 'mae': None, 'subcomposition_jsd': None}
    if full.any():
        a, b = truth[full] / total[full, None], p[full] / total[full, None]
        coverage['mae'] = float(np.abs(a.sum(axis=1) - b.sum(axis=1)).mean())
        supported = (a.sum(axis=1) > 0) & (b.sum(axis=1) > 0)
        if supported.any():
            from ..baselines.tsagris import jsd
            aa, bb = a[supported], b[supported]
            coverage['subcomposition_jsd'] = float(jsd(aa / aa.sum(axis=1, keepdims=True),
                                                       bb / bb.sum(axis=1, keepdims=True)).mean())
        coverage['subcomposition_supported_rows'] = int(supported.sum())
        if supported.sum() >= 2:
            from ..baselines.common import clr, pivot_basis
            aa, bb = a[supported], b[supported]
            aa, bb = aa / aa.sum(axis=1, keepdims=True), bb / bb.sum(axis=1, keepdims=True)
            # Fixed evaluation-only zero policy; never used to construct input.
            ca, cb = clr(aa + 1e-6), clr(bb + 1e-6)
            va, vb = np.cov(ca, rowvar=False), np.cov(cb, rowvar=False)
            ta = va.diagonal()[:, None] + va.diagonal()[None, :] - 2 * va
            tb = vb.diagonal()[:, None] + vb.diagonal()[None, :] - 2 * vb
            basis = pivot_basis(p.shape[1])
            coverage['variation_matrix_frobenius_error'] = float(np.linalg.norm(ta - tb))
            coverage['ilr_covariance_frobenius_error'] = float(np.linalg.norm(basis.T @ (va - vb) @ basis))
            coverage['zero_policy'] = 'conditional17_proportions_add_fixed_1e-6_for_scoring_only'
    temporal_errors, temporal_gaps = [], []
    lag_truth_a, lag_truth_b, lag_pred_a, lag_pred_b, lag_valid = [], [], [], [], []
    ratio_p = p / total[:, None]
    ratio_truth = np.divide(truth, total[:, None])
    for entity in sorted(set(entities)):
        local = np.flatnonzero(entities == entity)
        local = local[np.argsort(np.asarray(raw.timestamps)[ids[local]], kind='stable')]
        if len(local) < 2:
            continue
        a, b = local[:-1], local[1:]
        gap = np.asarray(raw.timestamps)[ids[b]] - np.asarray(raw.timestamps)[ids[a]]
        valid = original[a] & original[b] & (target[a] | target[b]) & (gap > 0)[:, None]
        lag_truth_a.append(ratio_truth[a]); lag_truth_b.append(ratio_truth[b])
        lag_pred_a.append(ratio_p[a]); lag_pred_b.append(ratio_p[b]); lag_valid.append(valid)
        delta = (ratio_p[b] - ratio_p[a]) - (ratio_truth[b] - ratio_truth[a])
        temporal_errors.extend(np.abs(delta[valid]).tolist())
        gaps = np.broadcast_to(gap[:, None], delta.shape)
        temporal_gaps.extend(gaps[valid].tolist())
    lag_errors = []
    if lag_valid:
        la, lb, pa, pb, lv = [np.concatenate(x) for x in
                             [lag_truth_a, lag_truth_b, lag_pred_a, lag_pred_b, lag_valid]]
        for j in range(p.shape[1]):
            keep = lv[:, j]
            if keep.sum() >= 10 and all(np.std(x[keep, j]) > 0 for x in [la, lb, pa, pb]):
                lag_errors.append(float(abs(np.corrcoef(la[keep, j], lb[keep, j])[0, 1] -
                                            np.corrcoef(pa[keep, j], pb[keep, j])[0, 1])))
    correlations, covariance_errors = [], []
    ratio_p = p / total[:, None]
    for j in range(p.shape[1]):
        for k in range(j + 1, p.shape[1]):
            valid = original[:, j] & original[:, k] & (target[:, j] | target[:, k])
            if valid.sum() < 10:
                continue
            a, b = ratio_truth[valid][:, [j, k]], ratio_p[valid][:, [j, k]]
            covariance_errors.append(float(abs(np.cov(a, rowvar=False)[0, 1] - np.cov(b, rowvar=False)[0, 1])))
            if (np.std(a, axis=0) > 0).all() and (np.std(b, axis=0) > 0).all():
                correlations.append(float(abs(np.corrcoef(a.T)[0, 1] - np.corrcoef(b.T)[0, 1])))
    positive_ratio = truth / total[:, None]
    return {'overall': masked(target), 'zero': masked(zero), 'positive': masked(positive),
            'rare_positive': masked(positive & (positive_ratio < .01)),
            'common_positive': masked(positive & (positive_ratio >= .01)),
            'location_macro_count_mae': _mean([x['absolute_sum'] / x['n'] for x in by_location if x['n']]),
            'feature_macro_count_mae': _mean([x['count_mae'] for x in per_feature if x['n']]),
            'by_location': by_location, 'by_feature': per_feature, 'coverage': coverage,
            'temporal': {'pairs': len(temporal_errors), 'ratio_delta_mae': _mean(temporal_errors),
                         'mean_gap_days': _mean(temporal_gaps),
                         'ratio_slope_mae_per_day': _mean(np.asarray(temporal_errors) / np.asarray(temporal_gaps))
                         if temporal_errors else None,
                         'lag_correlation_supported_features': len(lag_errors),
                         'lag_correlation_absolute_error_mean': _mean(lag_errors),
                         'by_gap_days': {str(gap): {'pairs': temporal_gaps.count(gap),
                           'ratio_delta_mae': _mean(np.asarray(temporal_errors)[np.asarray(temporal_gaps) == gap])}
                           for gap in sorted(set(temporal_gaps))}},
            'feature_relationship': {'supported_covariance_pairs': len(covariance_errors),
                                     'covariance_absolute_error_mean': _mean(covariance_errors),
                                     'supported_correlation_pairs': len(correlations),
                                     'correlation_absolute_error_mean': _mean(correlations),
                                     'scope': 'pooled_locations_descriptive_not_causal'},
            'validity': {'observed_changed': int(np.count_nonzero(p[conditioning] != truth[conditioning])),
                         'negative': int((p < 0).sum()),
                         'budget_violations': int((p.sum(axis=1) > total + 1e-10 * np.maximum(1., total)).sum())}}


def bootstrap_locations(stats, seed=20261015, draws=2000):
    values = [x for x in stats if x['n']]
    if not values:
        return None
    n = np.array([x['n'] for x in values], float)
    errors = np.array([x['ratio_absolute_sum'] for x in values], float)
    samples = np.random.default_rng(seed).integers(0, len(n), (draws, len(n)))
    pooled = errors[samples].sum(axis=1) / n[samples].sum(axis=1)
    macro = (errors / n)[samples].mean(axis=1)
    return {'locations': len(values), 'draws': draws,
            'ratio_mae_pooled_ci95': np.quantile(pooled, [.025, .975]).tolist(),
            'ratio_mae_location_macro_ci95': np.quantile(macro, [.025, .975]).tolist()}
