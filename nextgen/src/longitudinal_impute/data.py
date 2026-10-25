"""COVID-19 CSV input, observed masks and count-space output."""
import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import tomllib

import numpy as np


@dataclass
class CovidData:
    counts: np.ndarray
    totals: np.ndarray
    targets: tuple[str, ...]
    fields: list[str]
    rows: list[dict[str, str]]


def load_csv(path, schema_config):
    with Path(schema_config).open('rb') as stream:
        spec = tomllib.load(stream)['dataset']
    targets = tuple(spec['targets'])
    if len(targets) < 2 or len(set(targets)) != len(targets):
        raise ValueError('Declare at least two unique target columns')
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    required = {spec['entity'], spec['time'], spec['total'], *targets}
    if not rows or not fields or not required <= set(fields):
        raise ValueError('CSV is empty or lacks declared columns')
    if len(set(fields)) != len(fields) or set(targets) & {spec['entity'], spec['time'], spec['total']}:
        raise ValueError('CSV/schema columns overlap or repeat')
    counts = np.empty((len(rows), len(targets)))
    totals = np.empty(len(rows))
    keys = set()
    for i, row in enumerate(rows):
        entity, timestamp = row[spec['entity']], row[spec['time']]
        date.fromisoformat(timestamp)
        if not entity or (entity, timestamp) in keys:
            raise ValueError('Empty entity or duplicate entity/date')
        keys.add((entity, timestamp))
        totals[i] = float(row[spec['total']])
        for j, name in enumerate(targets):
            value = row[name].strip()
            counts[i, j] = np.nan if value.lower() in ('', 'na', 'nan', 'none') else float(value)
    if np.isinf(counts).any() or (counts[np.isfinite(counts)] < 0).any():
        raise ValueError('Observed counts must be finite and nonnegative')
    if not np.isfinite(totals).all() or (totals <= 0).any() or (np.nansum(counts, axis=1) > totals).any():
        raise ValueError('Invalid total or observed sum exceeds total')
    return CovidData(counts, totals, targets, fields, rows)


def evaluation_mask(counts, rate, seed):
    if not np.isfinite(rate) or not 0 <= rate < 1:
        raise ValueError('mask_rate must be in [0, 1)')
    observed = np.isfinite(counts)
    rng = np.random.default_rng(seed)
    hidden = (rng.random(counts.shape) < rate) & observed
    # Leave an anchor when a row originally contains observations.
    for i in np.flatnonzero(observed.any(axis=1) & ~(observed & ~hidden).any(axis=1)):
        hidden[i, rng.choice(np.flatnonzero(observed[i]))] = False
    return hidden


def model_input(data, hidden, *, positive=False):
    observed = np.isfinite(data.counts)
    if hidden.shape != observed.shape or hidden.dtype != bool or (hidden & ~observed).any():
        raise ValueError('Evaluation mask must be a boolean subset of observed cells')
    counts = np.where(observed & ~hidden, data.counts, np.nan)
    complete = np.isfinite(counts).all(axis=1)
    other = np.full(len(counts), np.nan)
    # Compute residual before division to avoid manufacturing tiny positive Other.
    other[complete] = (data.totals[complete] - counts[complete].sum(axis=1)) / data.totals[complete]
    values = np.column_stack((counts / data.totals[:, None], other))
    if positive:
        positives = values[np.isfinite(values) & (values > 0)]
        if not len(positives):
            raise ValueError('Hron requires positive observed values')
        values[values == 0] = positives.min() / 2
    return values


def restore_counts(data, proposed, hidden):
    """Preserve visible counts; share residual mass between missing and Other."""
    observed = np.isfinite(data.counts) & ~hidden
    if proposed.shape != (len(data.counts), len(data.targets) + 1):
        raise ValueError('Expected target proposals plus one Other column')
    if not np.isfinite(proposed).all() or (proposed < 0).any():
        raise ValueError('Model proposals must be finite and nonnegative')
    result = np.where(observed, data.counts, 0.)
    for i in range(len(result)):
        missing = ~observed[i]
        residual = data.totals[i] - result[i].sum()
        if not missing.any() or residual == 0:
            continue
        weights = np.r_[proposed[i, :-1][missing], proposed[i, -1]]
        scale = weights.max()
        if scale == 0:
            raise ValueError(f'row {i}: no proposed mass for missing targets and Other')
        with np.errstate(under='ignore', over='ignore'):
            scaled = weights / scale
            factor = (residual / scaled.sum()) / scale
            allocation = weights * factor if np.isfinite(factor) and factor > 0 else scaled / scaled.sum() * residual
        if not allocation.any():
            raise ValueError('Residual allocation is not representable at float64 precision')
        result[i, missing] = allocation[:-1]
    if not np.isfinite(result).all() or (result < 0).any() or (
            result.sum(axis=1) > data.totals + 1e-10 * np.maximum(1., data.totals)).any():
        raise ValueError('Completed counts violate the row total')
    return result


def write_completed_csv(path, data, completed):
    observed = np.isfinite(data.counts)
    if completed.shape != data.counts.shape or not np.array_equal(completed[observed], data.counts[observed]):
        raise ValueError('Output shape or observed counts changed')
    if not np.isfinite(completed).all() or (completed < 0).any() or (
            completed.sum(axis=1) > data.totals + 1e-10 * np.maximum(1., data.totals)).any():
        raise ValueError('Output is incomplete or violates the row total')
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=data.fields)
        writer.writeheader()
        for i, source in enumerate(data.rows):
            row = dict(source)
            for j, name in enumerate(data.targets):
                if not observed[i, j]:
                    row[name] = format(completed[i, j], '.17g')
            writer.writerow(row)
