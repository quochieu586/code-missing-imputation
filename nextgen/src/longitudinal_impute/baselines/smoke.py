"""Small, reproducible real-data checks; NOT a full longitudinal benchmark."""
import argparse
import json
from pathlib import Path
import platform
import time
import tomllib
import numpy as np
from ..contracts import FitScope
from ..metadata import MetadataEncoder
from .adapters import CountBaselineAdapter
from .csv_io import schema_from_config, load_csv
from .panels import hide_targets, subset
from .registry import make_core
from .runner import file_hash


def run_smoke(data, config_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Refusing to overwrite smoke evidence')
    with Path(config_path).open('rb') as f:
        schema = schema_from_config(tomllib.load(f))
    raw, _, _ = load_csv(data, schema)
    complete = raw.original_observed.all(axis=1)
    locations = sorted(set(raw.entity_ids[complete]))
    if len(locations) < 2:
        raise ValueError('Smoke needs complete data in at least two locations')
    train_locations = set(locations[:max(1, len(locations) // 2)])
    train_ids = np.flatnonzero(complete & np.isin(raw.entity_ids, list(train_locations)))[:200]
    test_complete = np.flatnonzero(complete & ~np.isin(raw.entity_ids, list(train_locations)))[:24]
    test_native = np.flatnonzero(~complete & ~np.isin(raw.entity_ids, list(train_locations)))[:24]
    hidden = np.zeros_like(raw.original_observed)
    # Same feature set across all sampled labelled query rows. Full-dataset
    # benchmark must mask full location histories, beyond this smoke subset.
    selected_features = np.argsort(raw.counts[test_complete].sum(axis=0))[-2:]
    hidden[np.ix_(test_complete, selected_features)] = True
    approved = ('median_age',) if 'median_age' in raw.metadata else ()
    panel, labels = hide_targets(raw, hidden, mask_id='smoke_global_hidden_v1', approved_metadata=approved)
    train, query = subset(panel, train_ids), subset(panel, np.r_[test_complete, test_native])
    assert not set(train.entity_ids) & set(query.entity_ids)
    scope = FitScope('inductive_group_by_location', train.row_ids, 'smoke_location_split_v1', train.mask_id)
    parameters = {'hron_knn': {'k': 3}, 'micoda_v2': {'W': 1e-6, 'max_iter': 10},
                  'tsagris_basic': {'k': 3}}
    output.mkdir(parents=True, exist_ok=False)
    summary = {'kind': 'real_data_smoke_not_full_benchmark', 'data_sha256': file_hash(data),
               'config_sha256': file_hash(config_path), 'python': platform.python_version(),
               'numpy': np.__version__, 'train_rows': len(train.row_ids), 'query_rows': len(query.row_ids),
               'train_locations': sorted(train_locations), 'target_features': selected_features.tolist(),
               'mask_sha256': __import__('hashlib').sha256(hidden.tobytes()).hexdigest(), 'methods': []}
    ids = np.r_[test_complete, test_native]
    target_mask = labels.target_mask[ids]
    truth = labels.values[ids]
    for method, params in parameters.items():
        start = time.perf_counter()
        adapter = CountBaselineAdapter(make_core(method, degenerate_policies=True, **params),
                                       metadata_encoder=MetadataEncoder(numeric=approved),
                                       fallback='train_mean')
        adapter.fit(train, scope)
        result = adapter.predict_result(query)
        error = np.abs(result.counts[target_mask] - truth[target_mask])
        mask = query.conditioning_mask
        checks = {'finite': bool(np.isfinite(result.counts).all()),
                  'nonnegative': bool((result.counts >= 0).all()),
                  'observed_unchanged': bool(np.array_equal(result.counts[mask], query.counts[mask])),
                  'budget': bool((result.counts.sum(axis=1) <= query.totals + 1e-9).all())}
        if not all(checks.values()):
            raise AssertionError(checks)
        summary['methods'].append({'method': method, 'parameters': params, 'checks': checks,
                                   'wall_seconds': time.perf_counter() - start,
                                   'hidden_count_mae': float(error.mean()), 'hidden_targets': int(target_mask.sum()),
                                   'fallback_rows': len(result.diagnostics['fallback_records']),
                                   'repair_l1': result.diagnostics['repair_l1'],
                                   'native_missing_accuracy': None})
        np.savetxt(output / (method + '_counts.csv'), result.counts, delimiter=',',
                   header=','.join(schema.target_columns), comments='')
        with (output / (method + '_diagnostics.json')).open('x', encoding='utf-8') as f:
            json.dump(result.diagnostics, f, ensure_ascii=False, indent=2, allow_nan=False)
    with (output / 'summary.json').open('x', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=False)
    with (output / 'rows.json').open('x', encoding='utf-8') as f:
        json.dump({'train_row_ids': train.row_ids, 'query_row_ids': query.row_ids}, f, ensure_ascii=False, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--schema-config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.data, args.schema_config, args.output), indent=2))


if __name__ == '__main__':
    main()
