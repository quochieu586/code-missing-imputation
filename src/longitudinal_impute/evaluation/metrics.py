"""Tier 1 scores per cell and the long prediction file of masked runs.

Scores for completed tables. Plain functions: edit the metric definitions here.

A missing (or hidden) cell is "forced to 0" when the visible targets of its row
already sum to the row total N: restore_counts then gives it 0 whatever the
model proposes. Every other missing cell is "model-decided".

Long table of the hidden cells of masked runs: one row per (model, mask, seed, hidden cell).
run.py --save-predictions writes it after each completed masked run. The truth columns are
read from data.counts here, after the model returned, so they are an output for scoring
only and never part of a model input. score_predictions recomputes cell_errors
(the numbers in summary.json) and the series-macro MAE from the file alone.

Columns: row_id (0-based data row), entity, date (raw time text), group, target, N,
truth_count, pred_count, truth_prop, pred_prop, model, mask, seed, and the 0/1 group flags
forced_zero, model_cell, truth_zero, truth_positive. The flag of the `model` group is
named model_cell because the column `model` holds the model name.
"""
import csv
import gzip
import io
from pathlib import Path

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


GROUPS = ('all', 'truth_zero', 'truth_positive', 'forced_zero', 'model', 'model_truth_zero')
METRICS = ('mse_count', 'mae_count', 'mse_proportion', 'mae_proportion')


def cell_groups(truth, forced):
    """Boolean selector of every scored group, from the truth and forced-zero flag of each hidden cell."""
    return {'all': np.ones(len(truth), bool), 'truth_zero': truth == 0, 'truth_positive': truth > 0,
            'forced_zero': forced, 'model': ~forced, 'model_truth_zero': ~forced & (truth == 0)}


def cell_errors(truth, pred, totals, forced):
    """Cell-weighted errors per group on counts and on c / N; an empty group gives None (NA), not 0."""
    errors = {}
    for group, cells in cell_groups(truth, forced).items():
        count = pred[cells] - truth[cells]
        proportion = count / totals[cells]
        errors[group] = {'cells': int(cells.sum()), **({
            'mse_count': float(np.mean(count ** 2)), 'mae_count': float(np.mean(np.abs(count))),
            'mse_proportion': float(np.mean(proportion ** 2)),
            'mae_proportion': float(np.mean(np.abs(proportion)))} if cells.any() else dict.fromkeys(METRICS))}
    return errors


def masked_errors(data, completed, hidden):
    """Errors on the hidden cells H only, overall and per group, on counts and on c / N."""
    if hidden.shape != data.counts.shape or (hidden & ~np.isfinite(data.counts)).any():
        raise ValueError('hidden must be a subset of the observed cells')
    rows = np.nonzero(hidden)[0]
    return cell_errors(data.counts[hidden], completed[hidden], data.totals[rows], forced_zero(data, hidden)[hidden])


def series_macro_mae(truth, pred, totals, series, cells):
    """MAE of c / N within each series (e.g. location x target), then the plain mean over series.

    Only the selected cells count; a series with no selected cell is left out. None (NA)
    when no cell is selected. Cell-weighted MAE weights a series by its number of cells.
    """
    cells = np.asarray(cells, bool)
    if not cells.any():
        return None
    error = np.abs((pred[cells] - truth[cells]) / totals[cells])
    _, index = np.unique(np.asarray(series)[cells], return_inverse=True)
    return float(np.mean(np.bincount(index, weights=error) / np.bincount(index)))


def implied_other_errors(row_ids, truth, pred, totals, entities, complete_rows):
    """Error of the implied Other share on rows fully observed before masking (step 4).

    Inputs are per hidden cell (e.g. the long file); complete_rows holds the ids of rows whose
    targets were all observed before masking, so their true Other = (N - sum c) / N is known.
    The implied Other after filling is (N - sum of filled targets) / N; on such a row every
    missing cell is hidden, so implied - true = (sum truth - sum pred) / N over its hidden cells.
    Returns rows and entities used, the MAE and the mean signed error (implied - true) of the
    proportion; both are None (NA) when no hidden cell lies in such a row.
    """
    row_ids = np.asarray(row_ids)
    keep = np.isin(row_ids, np.asarray(sorted(complete_rows), dtype=int))
    if not keep.any():
        return {'rows': 0, 'entities': 0, 'mae_proportion': None, 'mean_signed_proportion': None}
    ids, index = np.unique(row_ids[keep], return_inverse=True)
    gap = np.bincount(index, weights=(np.asarray(truth) - np.asarray(pred))[keep])
    n = np.zeros(len(ids))
    n[index] = np.asarray(totals)[keep]
    error = gap / n
    return {'rows': len(ids), 'entities': len(set(np.asarray(entities)[keep])),
            'mae_proportion': float(np.mean(np.abs(error))), 'mean_signed_proportion': float(np.mean(error))}


def summarize(rows):
    """Group run rows by (model, pass, mask) and give mean / sd over the successful seeds."""
    groups = {}
    for row in rows:
        groups.setdefault((row['model'], row['pass'], row['mask']), []).append(row)
    out = []
    for (model, run_name, mask_name), members in groups.items():
        ok = [r for r in members if r['status'] == 'ok']
        summary = {'model': model, 'pass': run_name, 'mask': mask_name,
                   'space': members[0].get('space'), 'runs_ok': len(ok),
                   'runs_failed': len(members) - len(ok)}
        metrics = dict.fromkeys(k for r in ok for k, v in r.items()
                                if k != 'seed' and isinstance(v, (int, float)) and not isinstance(v, bool))
        for key in metrics:
            values = np.array([r[key] for r in ok if r.get(key) is not None], float)
            summary[f'{key}_mean'] = float(values.mean()) if len(values) else None
            summary[f'{key}_sd'] = float(values.std(ddof=1)) if len(values) > 1 else None
        out.append(summary)
    return out


def paired_summary(rows):
    """Masked comparison on shared successful seeds and model space only."""
    masked = [r for r in rows if r['pass'] == 'masked']
    cases, selected = {}, []
    for kind in sorted({r['mask'] for r in masked}):
        members = [r for r in masked if r['mask'] == kind]
        models = sorted({r['model'] for r in members})
        seeds = sorted({r['seed'] for r in members})
        same_space = len({r.get('space') for r in members}) == 1
        shared = [s for s in seeds if same_space and all(
            any(r['seed'] == s and r['model'] == m and r['status'] == 'ok'
                and r.get('hidden_cells', 0) > 0
                and r.get('resolved_cells') == r.get('hidden_cells') for r in members)
            for m in models)]
        cases[kind] = {'models': models, 'seeds_included': shared,
                       'seeds_excluded': [s for s in seeds if s not in shared],
                       'comparable': same_space and bool(shared),
                       'reason': ('different_model_spaces' if not same_space else
                                  'common_success_only' if shared else 'no_common_success')}
        selected.extend(r for r in members if r['seed'] in shared)
    return {'cases': cases, 'aggregate': summarize(selected)}



COLUMNS = ('row_id', 'entity', 'date', 'group', 'target', 'N', 'truth_count', 'pred_count', 'truth_prop',
           'pred_prop', 'model', 'mask', 'seed', 'forced_zero', 'model_cell', 'truth_zero', 'truth_positive')
FLOATS = ('N', 'truth_count', 'pred_count', 'truth_prop', 'pred_prop')
INTS = ('row_id', 'seed', 'forced_zero', 'model_cell', 'truth_zero', 'truth_positive')


def prediction_rows(data, completed, hidden, model, mask, seed, time_column=None):
    """Rows of the hidden cells in row-major order, the order masked_errors scores them in."""
    if hidden.shape != data.counts.shape or (hidden & ~np.isfinite(data.counts)).any():
        raise ValueError('hidden must be a subset of the observed cells')
    forced = forced_zero(data, hidden)
    out = []
    for i, j in zip(*np.nonzero(hidden)):
        truth, pred, n = float(data.counts[i, j]), float(completed[i, j]), float(data.totals[i])
        out.append({'row_id': int(i), 'entity': data.entities[i],
                    'date': data.rows[i][time_column] if time_column else '',
                    'group': data.groups[i] if data.groups is not None else '', 'target': data.targets[j],
                    'N': n, 'truth_count': truth, 'pred_count': pred, 'truth_prop': truth / n,
                    'pred_prop': pred / n, 'model': model, 'mask': mask, 'seed': int(seed),
                    'forced_zero': int(forced[i, j]), 'model_cell': int(not forced[i, j]),
                    'truth_zero': int(truth == 0), 'truth_positive': int(truth > 0)})
    return out


def write_predictions(path, rows):
    """csv.gz with repr floats (exact round trip) and a fixed gzip header, so equal rows hash equal."""
    with Path(path).open('xb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as packed, \
            io.TextIOWrapper(packed, encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: repr(v) if isinstance(v, float) else v for k, v in row.items()})


def read_predictions(path):
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError(f'{path}: columns {reader.fieldnames} are not {list(COLUMNS)}')
        return [{**row, **{k: float(row[k]) for k in FLOATS}, **{k: int(row[k]) for k in INTS}} for row in reader]


def score_predictions(rows):
    """{(model, mask, seed): {'errors': cell_errors(...), 'series_macro_mae_proportion': ..., 'series': ...}}.

    Cell-weighted errors use the same function as masked_errors, on counts read back from the
    file. The series-macro MAE is on the `model` group, one series per (entity, target).
    """
    runs = {}
    for row in rows:
        runs.setdefault((row['model'], row['mask'], row['seed']), []).append(row)
    scores = {}
    for key, cells in runs.items():
        truth, pred, totals = (np.array([c[k] for c in cells]) for k in ('truth_count', 'pred_count', 'N'))
        forced = np.array([c['forced_zero'] for c in cells], bool)
        if not np.array_equal(~forced, np.array([c['model_cell'] for c in cells], bool)):
            raise ValueError(f'{key}: model_cell is not the complement of forced_zero')
        series = np.array([f"{c['entity']}\x1f{c['target']}" for c in cells])
        scores[key] = {'errors': cell_errors(truth, pred, totals, forced),
                       'series_macro_mae_proportion': series_macro_mae(truth, pred, totals, series, ~forced),
                       'series': len(set(series[~forced]))}
    return scores
