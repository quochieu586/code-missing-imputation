"""CSV input for any count table, evaluation masks and count-space output."""
import csv
from dataclasses import dataclass
from pathlib import Path
import os
import tomllib
from tempfile import NamedTemporaryFile

import numpy as np

MISSING = ('', 'na', 'nan', 'none')
SCHEMA_KEYS = {'entity', 'time', 'total', 'targets'}


@dataclass
class Dataset:
    counts: np.ndarray            # N x D target counts, NaN = missing
    totals: np.ndarray            # N row totals (declared column, or the row sum of the targets)
    targets: tuple[str, ...]
    entities: tuple[str, ...]     # entity of every row, e.g. location or subject
    fields: list[str]
    rows: list[dict[str, str]]
    lines: list[int]              # CSV line of every row, for error messages


def read_schema(path):
    with Path(path).open('rb') as stream:
        spec = tomllib.load(stream).get('dataset')
    if not isinstance(spec, dict) or 'entity' not in spec or 'targets' not in spec:
        raise ValueError(f'{path}: [dataset] needs entity and targets')
    if set(spec) - SCHEMA_KEYS:
        raise ValueError(f'{path}: unknown [dataset] keys {sorted(set(spec) - SCHEMA_KEYS)}')
    targets = spec['targets']
    if len(targets) < 2 or len(set(targets)) != len(targets):
        raise ValueError(f'{path}: declare at least two unique target columns')
    keys = [spec[k] for k in ('entity', 'time', 'total') if k in spec]
    if len(set(keys)) != len(keys) or set(keys) & set(targets):
        raise ValueError(f'{path}: entity/time/total/target columns overlap')
    return spec


def load_csv(path, schema_config):
    spec = read_schema(schema_config)
    targets, entity, time, total = tuple(spec['targets']), spec['entity'], spec.get('time'), spec.get('total')
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields, rows, lines = reader.fieldnames, [], []
        for row in reader:
            rows.append(row)
            lines.append(reader.line_num)
    if not rows or not fields:
        raise ValueError(f'{path}: CSV is empty')
    absent = [c for c in (entity, time, total, *targets) if c is not None and c not in fields]
    if absent or len(set(fields)) != len(fields):
        raise ValueError(f'{path}: CSV lacks declared columns {absent} or repeats a column')
    counts = np.empty((len(rows), len(targets)))
    totals = np.empty(len(rows))
    keys = set()
    for i, (row, line) in enumerate(zip(rows, lines)):
        if None in row or any(value is None for value in row.values()):
            # Reject a row whose field count does not match the header (extra or
            # missing fields) instead of letting DictReader fabricate None keys
            # that later break parsing or add a bogus column to the output row.
            raise ValueError(f'{path}: CSV line {line} (entity {row.get(entity)!r}): '
                             f'expected {len(fields)} fields, found extra or missing fields')
        name = row[entity]
        where = f'CSV line {line} (entity {name!r})'
        if not name:
            raise ValueError(f'CSV line {line}: empty entity in column {entity!r}')
        if time is not None:
            if not row[time] or (name, row[time]) in keys:
                raise ValueError(f'{where}: empty or duplicate {time!r} = {row[time]!r}')
            keys.add((name, row[time]))
        for j, column in enumerate(targets):
            counts[i, j] = parse(row[column], where, column, missing_ok=True)
        if total is not None:
            totals[i] = parse(row[total], where, total, missing_ok=False)
            if totals[i] <= 0 or np.nansum(counts[i]) > totals[i]:
                raise ValueError(f'{where}: column {total!r} = {totals[i]:g} is not positive '
                                 f'or below the observed target sum {np.nansum(counts[i]):g}')
        else:
            if np.isnan(counts[i]).any():
                absent = [c for c, v in zip(targets, counts[i]) if np.isnan(v)]
                raise ValueError(f'{where}: targets {absent} are missing but the schema declares no total '
                                 'column, so the row total is unknown')
            totals[i] = counts[i].sum()
            if totals[i] <= 0:
                raise ValueError(f'{where}: all targets are 0, the row total must be positive')
    return Dataset(counts, totals, targets, tuple(row[entity] for row in rows), fields, rows, lines)


def parse(text, where, column, *, missing_ok):
    text = text.strip()
    if text.lower() in MISSING:
        if missing_ok:
            return np.nan
        raise ValueError(f'{where}: column {column!r} is empty')
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f'{where}: column {column!r} = {text!r} is not a number') from None
    if not np.isfinite(value) or value < 0:
        raise ValueError(f'{where}: column {column!r} = {text!r} must be finite and nonnegative')
    return value


def validate_mask(kind, rate):
    """Pre-flight check shared by make_mask and the runner, before any side effect."""
    if kind not in ('random_cell', 'whole_feature'):
        raise ValueError(f"mask kind must be 'random_cell' or 'whole_feature', got {kind!r}")
    if not np.isfinite(rate) or not 0 <= rate < 1:
        raise ValueError('mask rate must be in [0, 1)')


def make_mask(data, kind, rate, seed):
    """Hide a reproducible subset H of the observed cells. Returns (H, statistics)."""
    validate_mask(kind, rate)
    observed = np.isfinite(data.counts)
    rng = np.random.default_rng(seed)
    stats = {'kind': kind, 'rate': rate, 'seed': seed}
    if kind == 'random_cell':
        # Every observed cell is hidden with probability rate; keep one anchor in rows with data.
        hidden = (rng.random(observed.shape) < rate) & observed
        for i in np.flatnonzero(observed.any(axis=1) & ~(observed & ~hidden).any(axis=1)):
            hidden[i, rng.choice(np.flatnonzero(observed[i]))] = False
    elif kind == 'whole_feature':
        # Hide the whole observed series of random (entity, target) pairs until about
        # rate x observed cells are hidden. A pair that would leave a row with no visible
        # target is skipped.
        hidden = np.zeros_like(observed)
        entities = np.asarray(data.entities)
        r, c = np.nonzero(observed)
        pairs = sorted(set(zip(entities[r], c)))
        visible = observed.sum(axis=1)
        goal, count, used, skipped = rate * observed.sum(), 0, 0, 0
        for p in rng.permutation(len(pairs)):
            if count >= goal:
                break
            e, j = pairs[p]
            series = np.flatnonzero((entities == e) & observed[:, j])
            if (visible[series] == 1).any():
                skipped += 1
                continue
            hidden[series, j] = True
            visible[series] -= 1
            count += len(series)
            used += 1
        stats.update(pairs_with_observed_cells=len(pairs), pairs_hidden=used, pairs_skipped=skipped)
    complete = observed.all(axis=1)
    stats.update(observed_cells=int(observed.sum()), hidden_cells=int(hidden.sum()),
                 actual_rate=float(hidden.sum() / max(observed.sum(), 1)),
                 rows_affected=int(hidden.any(axis=1).sum()),
                 complete_rows_before=int(complete.sum()),
                 complete_rows_after=int((complete & ~hidden.any(axis=1)).sum()))
    return hidden, stats


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


def check_completed(data, completed, hidden):
    """Visible counts unchanged; every cell finite, nonnegative and within the row total."""
    visible = np.isfinite(data.counts) & ~hidden
    if completed.shape != data.counts.shape or not np.array_equal(completed[visible], data.counts[visible]):
        raise ValueError('Output shape or observed counts changed')
    if not np.isfinite(completed).all() or (completed < 0).any() or (
            completed.sum(axis=1) > data.totals + 1e-10 * np.maximum(1., data.totals)).any():
        raise ValueError('Output is incomplete or violates the row total')


def write_completed_csv(path, data, completed, *, allow_partial=False):
    """Publish only after a complete write: temp file, then an atomic hard link.

    allow_partial publishes a `*_partial.csv` whose missing targets stay empty;
    invariants are still enforced on the filled cells and on the row budget.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    temporary = None
    try:
        with NamedTemporaryFile(mode='w', encoding='utf-8', newline='', dir=path.parent,
                                prefix=f'.{path.name}.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            _write_completed_rows(stream, data, completed, allow_partial=allow_partial)
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _validate_completed(data, completed, allow_partial=False):
    observed = np.isfinite(data.counts)
    if completed.shape != data.counts.shape:
        raise ValueError('Output shape differs from the dataset')
    if not np.array_equal(completed[observed], data.counts[observed]):
        raise ValueError('Observed counts changed')
    if np.isinf(completed).any() or (completed < 0).any():
        raise ValueError('Output is incomplete or violates the row total')
    if not allow_partial and np.isnan(completed).any():
        raise ValueError('Output is incomplete or violates the row total')
    if (np.nansum(completed, axis=1) > data.totals + 1e-10 * np.maximum(1., data.totals)).any():
        raise ValueError('Output is incomplete or violates the row total')


def _write_completed_rows(stream, data, completed, *, allow_partial=False):
    observed = np.isfinite(data.counts)
    _validate_completed(data, completed, allow_partial=allow_partial)
    writer = csv.DictWriter(stream, fieldnames=data.fields)
    writer.writeheader()
    for i, source in enumerate(data.rows):
        row = dict(source)
        for j, name in enumerate(data.targets):
            if not observed[i, j]:
                row[name] = '' if np.isnan(completed[i, j]) else format(completed[i, j], '.17g')
        writer.writerow(row)
