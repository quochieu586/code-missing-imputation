"""Thin CLI: approved schema -> explicit fit IDs -> count adapter -> CSV.

Fit IDs JSON is a list of JSON-encoded [entity,date] strings, as emitted by
load_csv. The caller owns splitting. No automatic selection using test scores.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import tomllib
import numpy as np
from ..contracts import FitScope
from ..metadata import MetadataEncoder
from .adapters import CountBaselineAdapter
from .common import BaselineError
from .csv_io import schema_from_config, load_csv, write_completed_csv
from .panels import hide_targets
from .registry import METHODS, make_core, core_configuration


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute(*, data, schema_config, fit_ids, method, output, parameters=None,
            numeric=(), categorical=(), fallback='error', degenerate_policies=False,
            protocol='inductive'):
    output = Path(output)
    if output.exists():
        raise BaselineError('Refusing to overwrite output directory')
    with Path(schema_config).open('rb') as f:
        config = tomllib.load(f)
    schema = schema_from_config(config)
    approved = tuple(numeric) + tuple(categorical)
    if not set(approved) <= set(schema.metadata_columns):
        raise BaselineError('Metadata must be explicitly approved in the dataset schema')
    raw, fields, rows = load_csv(data, schema)
    panel, _ = hide_targets(raw, np.zeros_like(raw.original_observed), mask_id='native_observed',
                            approved_metadata=approved)
    ids = tuple(json.loads(Path(fit_ids).read_text(encoding='utf-8')))
    parameters = {} if parameters is None else dict(parameters)
    core = make_core(method, degenerate_policies=degenerate_policies, **parameters)
    adapter = CountBaselineAdapter(core, metadata_encoder=MetadataEncoder(numeric=numeric, categorical=categorical),
                                   fallback=fallback)
    start = time.perf_counter()
    adapter.fit(panel, FitScope(protocol, ids, file_hash(fit_ids), panel.mask_id))
    result = adapter.predict_result(panel)
    elapsed = time.perf_counter() - start
    output.mkdir(parents=True, exist_ok=False)
    write_completed_csv(output / 'completed.csv', raw, result, fields, rows)
    source = Path(__file__).parents[1]
    manifest = {'run_id': output.name, 'method': method, 'parameters': parameters,
                'resolved_core_config': core_configuration(adapter.core_), 'run_kind': 'dataset_adaptation',
                'protocol': protocol, 'fit_rows': len(ids), 'dataset_sha256': file_hash(data),
                'schema_sha256': file_hash(schema_config), 'fit_ids_sha256': file_hash(fit_ids),
                'source_sha256': {str(p.relative_to(source)): file_hash(p) for p in sorted(source.rglob('*.py'))},
                'python': platform.python_version(), 'numpy': np.__version__,
                'core_seed': getattr(core, 'seed', None),
                'conditioning_mask_sha256': hashlib.sha256(panel.conditioning_mask.tobytes()).hexdigest(),
                'metadata_numeric': list(numeric), 'metadata_categorical': list(categorical),
                'fallback': fallback, 'degenerate_policies': degenerate_policies,
                'wall_seconds': elapsed, 'accuracy_on_native_missing': None}
    for name, value in [('manifest.json', manifest), ('diagnostics.json', result.diagnostics)]:
        with (output / name).open('x', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--schema-config', required=True, type=Path)
    parser.add_argument('--fit-ids', required=True, type=Path)
    parser.add_argument('--method', required=True, choices=METHODS)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--parameters', default='{}', help='JSON object of core constructor arguments')
    parser.add_argument('--numeric-metadata', default='')
    parser.add_argument('--categorical-metadata', default='')
    parser.add_argument('--fallback', choices=['error', 'train_mean'], default='error')
    parser.add_argument('--degenerate-policies', action='store_true')
    parser.add_argument('--protocol', choices=['inductive', 'transductive'], default='inductive')
    args = parser.parse_args()
    result = execute(data=args.data, schema_config=args.schema_config, fit_ids=args.fit_ids,
                     method=args.method, output=args.output, parameters=json.loads(args.parameters),
                     numeric=tuple(filter(None, args.numeric_metadata.split(','))),
                     categorical=tuple(filter(None, args.categorical_metadata.split(','))),
                     fallback=args.fallback, degenerate_policies=args.degenerate_policies, protocol=args.protocol)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
