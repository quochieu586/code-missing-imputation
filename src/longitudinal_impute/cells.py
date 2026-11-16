"""Long table of the hidden cells of masked runs: one row per (model, mask, seed, hidden cell).

run.py --save-predictions writes it after each completed masked run. The truth columns are
read from data.counts here, after the model returned, so they are an output for scoring
only and never part of a model input. score_predictions recomputes evaluate.cell_errors
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

from .evaluate import cell_errors, forced_zero, series_macro_mae

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
