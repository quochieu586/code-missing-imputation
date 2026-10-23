"""Explicit CSV boundary and export for general baselines; no pandas required."""
import csv
import json
from datetime import date
from pathlib import Path
import numpy as np
from ..contracts import RawPanel, DatasetSchema
from .common import BaselineError


def schema_from_config(config):
    spec = config['dataset']
    targets = tuple(spec['targets'])
    metadata = tuple(spec.get('metadata_candidates', ()))
    if len(targets) < 2 or len(set(targets)) != len(targets) or set(targets) & set(metadata):
        raise BaselineError('Target/metadata schema is invalid')
    return DatasetSchema(spec['entity'], spec['time'], spec['total'], targets, metadata)


def load_csv(path, schema):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames, rows = reader.fieldnames, list(reader)
    needed = [schema.entity_column, schema.time_column, schema.total_column,
              *schema.target_columns, *schema.metadata_columns]
    if not rows or not fieldnames or not set(needed) <= set(fieldnames):
        raise BaselineError('CSV is empty or lacks explicitly declared columns')
    counts = np.empty((len(rows), len(schema.target_columns)))
    totals, entities, times, keys = [], [], [], []
    try:
        for i, row in enumerate(rows):
            entity, timestamp = row[schema.entity_column], row[schema.time_column]
            if not entity:
                raise BaselineError('Empty location/entity ID')
            times.append(date.fromisoformat(timestamp).toordinal())
            entities.append(entity)
            keys.append(json.dumps([entity, timestamp], ensure_ascii=False, separators=(',', ':')))
            totals.append(float(row[schema.total_column]))
            for j, name in enumerate(schema.target_columns):
                value = row[name].strip()
                counts[i, j] = np.nan if value.lower() in ('', 'na', 'nan', 'none') else float(value)
    except (ValueError, TypeError, KeyError) as exc:
        raise BaselineError('Invalid CSV date/count/total') from exc
    if len(set(keys)) != len(keys) or np.isinf(counts).any() or (counts[np.isfinite(counts)] < 0).any():
        raise BaselineError('Duplicate entity/date keys or invalid count values')
    totals = np.asarray(totals)
    if not np.isfinite(totals).all() or (totals <= 0).any() or (np.nansum(counts, axis=1) > totals).any():
        raise BaselineError('Invalid total or observed sum exceeds total')
    raw = RawPanel(tuple(keys), counts, np.isfinite(counts), totals, np.asarray(entities),
                   np.asarray(times), {k: np.array([r[k] for r in rows], object)
                                      for k in schema.metadata_columns}, schema)
    return raw, fieldnames, rows


def write_completed_csv(path, raw, result, fieldnames, rows):
    if result.row_ids != raw.row_ids:
        raise BaselineError('Output row identity/order does not match raw CSV')
    if not np.array_equal(result.counts[raw.original_observed], raw.counts[raw.original_observed]):
        raise BaselineError('Final output changed raw observed counts')
    if not np.isfinite(result.counts).all() or (result.counts < 0).any() or (
            result.counts.sum(axis=1) > raw.totals + 1e-10 * np.maximum(1., raw.totals)).any():
        raise BaselineError('Final output is incomplete or infeasible')
    with Path(path).open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, source in enumerate(rows):
            row = dict(source)
            for j, name in enumerate(raw.schema.target_columns):
                if not raw.original_observed[i, j]:
                    row[name] = format(result.counts[i, j], '.17g')
            writer.writerow(row)
