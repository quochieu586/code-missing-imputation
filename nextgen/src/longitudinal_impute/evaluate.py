"""Scores for completed tables. Plain functions: edit the metric definitions here.

A missing (or hidden) cell is "forced to 0" when the visible targets of its row
already sum to the row total N: restore_counts then gives it 0 whatever the
model proposes. Every other missing cell is "model-decided".
"""
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


def forced_zero(data, hidden):
    """Cells that are missing to the model and lie in a row whose visible sum equals N."""
    visible = np.isfinite(data.counts) & ~hidden
    residual = data.totals - np.where(visible, data.counts, 0.).sum(axis=1)
    return ~visible & (residual == 0)[:, None]


def wasserstein_scores(data, completed):
    """Native run (no labels): per target, compare observed cells with the imputed ones."""
    missing = ~np.isfinite(data.counts)
    forced = forced_zero(data, np.zeros_like(missing))
    scores = {}
    for scale, table, values in (('count', data.counts, completed),
                                 ('proportion', data.counts / data.totals[:, None], completed / data.totals[:, None])):
        for j, name in enumerate(data.targets):
            seen, imputed = table[~missing[:, j], j], values[:, j]
            target = scores.setdefault(name, {
                'observed_cells': int((~missing[:, j]).sum()), 'imputed_cells': int(missing[:, j].sum()),
                'forced_zero_cells': int(forced[:, j].sum()),
                'model_cells': int((missing[:, j] & ~forced[:, j]).sum())})
            target[scale] = {
                'observed_vs_completed': wasserstein(seen, imputed),
                'observed_vs_imputed': wasserstein(seen, imputed[missing[:, j]]),
                'observed_vs_model_imputed': wasserstein(seen, imputed[missing[:, j] & ~forced[:, j]])}
    return scores


def masked_errors(data, completed, hidden):
    """Errors on the hidden cells H only, overall and per group, on counts and on c / N."""
    if hidden.shape != data.counts.shape or (hidden & ~np.isfinite(data.counts)).any():
        raise ValueError('hidden must be a subset of the observed cells')
    rows = np.nonzero(hidden)[0]
    truth, pred, totals = data.counts[hidden], completed[hidden], data.totals[rows]
    forced = forced_zero(data, hidden)[hidden]
    groups = {'all': np.ones(len(truth), bool), 'truth_zero': truth == 0, 'truth_positive': truth > 0,
              'forced_zero': forced, 'model': ~forced, 'model_truth_zero': ~forced & (truth == 0)}
    errors = {}
    for group, cells in groups.items():
        count = pred[cells] - truth[cells]
        proportion = count / totals[cells]
        errors[group] = {'cells': int(cells.sum()), **({
            'mse_count': float(np.mean(count ** 2)), 'mae_count': float(np.mean(np.abs(count))),
            'mse_proportion': float(np.mean(proportion ** 2)),
            'mae_proportion': float(np.mean(np.abs(proportion)))} if cells.any() else dict.fromkeys(
            ('mse_count', 'mae_count', 'mse_proportion', 'mae_proportion')))}
    return errors


def summarize(rows):
    """Group run rows by (model, pass, mask) and give mean / sd over the successful seeds."""
    groups = {}
    for row in rows:
        groups.setdefault((row['model'], row['pass'], row['mask']), []).append(row)
    out = []
    for (model, run, mask), members in groups.items():
        ok = [r for r in members if r['status'] == 'ok']
        summary = {'model': model, 'pass': run, 'mask': mask, 'runs_ok': len(ok),
                   'runs_failed': len(members) - len(ok)}
        metrics = dict.fromkeys(k for r in ok for k, v in r.items()
                                if k != 'seed' and isinstance(v, (int, float)) and not isinstance(v, bool))
        for key in metrics:
            values = np.array([r[key] for r in ok if r.get(key) is not None], float)
            summary[f'{key}_mean'] = float(values.mean()) if len(values) else None
            summary[f'{key}_sd'] = float(values.std(ddof=1)) if len(values) > 1 else None
        out.append(summary)
    return out
