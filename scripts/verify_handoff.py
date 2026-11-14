"""Check the installed package, paths and CSV outputs from an unrelated cwd.

Run with --output NEW_DIRECTORY; add --data data/covariants.csv for full COVID.
This checks operational invariants, not the scientific comparability of models.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODELS = ('hron', 'tsagris', 'micoda')
SCHEMAS = {
    'covid': '[dataset]\nentity="location"\ntime="date"\ntotal="total_sequence"\n'
             'targets=["v1","v2","v3","v4"]\n',
    'microbiome': '[dataset]\nentity="subject"\ntime="visit"\n'
                  'targets=["Bacteroides","Prevotella","Faecalibacterium",'
                  '"Akkermansia","Bifidobacterium"]\n',
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def invoke(command, cwd, env, log, *, succeeds=True):
    result = subprocess.run([str(arg) for arg in command], cwd=cwd, env=env,
                            capture_output=True, text=True, encoding='utf-8')
    log.write_text(json.dumps([str(arg) for arg in command]) + '\n' + result.stdout + result.stderr,
                   encoding='utf-8')
    require((result.returncode == 0) == succeeds, f'Unexpected exit {result.returncode}: {log}')
    return result


def validate_run(source, schema, output, kind, rate, seed):
    from longitudinal_impute.data import load_csv, make_mask

    data = load_csv(source, schema)
    observed = np.isfinite(data.counts)
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    config = json.loads((output / 'config.json').read_text(encoding='utf-8'))
    require(Path(config['data']) == source, 'Config data path differs from requested input')
    require(Path(config['schema_path']) == schema, 'Config schema path differs from requested schema')
    require(config['data_sha256'] == sha(source), 'Config data fingerprint differs')
    require(set(summary['native']) == set(MODELS), 'Native model set differs')
    require(set(summary['masked']) == {str(seed)}, 'Masked seed set differs')
    hidden = np.load(output / f'mask_{kind}_seed{seed}.npy', allow_pickle=False)
    expected, _ = make_mask(data, kind, rate, seed)
    require(np.array_equal(hidden, expected), 'Saved mask differs from the shared seeded mask')
    require(hidden.any(), 'No cells were hidden; this run cannot verify masked scoring')
    masked = summary['masked'][str(seed)]
    require(masked['mask']['hidden_cells'] == int(hidden.sum()), 'Mask statistics differ')
    require(set(masked['models']) == set(MODELS), 'Masked model set differs')
    for name in MODELS:
        native = summary['native'][name]
        require(native['status'] == 'ok', f'{name}: native status is not ok')
        require(set(native['wasserstein']) == set(data.targets), f'{name}: target scores missing')
        completed = load_csv(output / f'{name}_completed.csv', schema)
        require(completed.fields == data.fields and len(completed.rows) == len(data.rows),
                f'{name}: CSV shape/header changed')
        require(np.isfinite(completed.counts).all() and (completed.counts >= 0).all(),
                f'{name}: invalid completed counts')
        require(np.array_equal(completed.counts[observed], data.counts[observed]),
                f'{name}: observed numeric values changed')
        require((completed.counts.sum(axis=1) <= data.totals + 1e-10 * np.maximum(1., data.totals)).all(),
                f'{name}: row total exceeded')
        for i, original in enumerate(data.rows):
            visible = {target for j, target in enumerate(data.targets) if observed[i, j]}
            for column in data.fields:
                if column not in data.targets or column in visible:
                    require(completed.rows[i][column] == original[column],
                            f'{name}: observed text or metadata changed at row {i}, column {column}')
        scores = masked['models'][name]
        require(scores['status'] == 'ok' and scores['coverage'] == 1., f'{name}: incomplete masked run')
        all_scores = scores['errors']['all']
        require(all_scores['cells'] == int(hidden.sum()), f'{name}: scored cells differ from shared mask')
        require(all(np.isfinite(all_scores[key]) and all_scores[key] >= 0 for key in
                    ('mse_count', 'mae_count', 'mse_proportion', 'mae_proportion')),
                f'{name}: invalid masked metrics')
    require(len(summary['rows']) == 6 and all(row['status'] == 'ok' for row in summary['rows']),
            'Summary does not contain six successful native/masked rows')
    with (output / 'summary.csv').open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        columns, csv_rows = reader.fieldnames, list(reader)
    require(columns and csv_rows == [
        {key: '' if row.get(key) is None else str(row.get(key, '')) for key in columns}
        for row in summary['rows']], 'CSV summary differs from JSON summary')
    require(len(summary['aggregate']) == 6 and all(row['runs_ok'] == 1 and row['runs_failed'] == 0
                                                  for row in summary['aggregate']),
            'Aggregate lost a run or hid a failure')
    require(not (output / 'errors.log').exists(), 'Unexpected errors.log')
    return {'models': list(MODELS), 'rows': len(data.rows), 'hidden_cells': int(hidden.sum()),
            'data_sha256': sha(source), 'observed_and_metadata_preserved': True,
            'counts_finite_nonnegative_within_total': True, 'masked_coverage': 1.,
            'csv_json_summaries_consistent': True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--schema', type=Path)
    args = parser.parse_args(argv)
    if args.schema and not args.data:
        parser.error('--schema requires --data')
    source = args.data.resolve(strict=True) if args.data else None
    schema = (args.schema or ROOT / 'configs/baselines_covariants.toml').resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cwd = output / 'unrelated cwd with spaces'
    cwd.mkdir()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1')
    env.pop('PYTHONPATH', None)
    report = {'python': sys.executable, 'cwd': str(cwd), 'runs': {}, 'status': 'running'}
    try:
        probe = invoke([sys.executable, '-c', 'import longitudinal_impute; print(longitudinal_impute.__file__)'],
                       cwd, env, output / 'import.log')
        imported = Path(probe.stdout.strip()).resolve()
        require(imported == ROOT / 'src/longitudinal_impute/__init__.py',
                f'Package imported from {imported}, expected this checkout')
        report['package'] = str(imported)
        for name in ('longitudinal-impute', 'general-impute'):
            console = Path(sys.executable).parent / (name + ('.exe' if os.name == 'nt' else ''))
            require(console.is_file(), f'Missing installed console entrypoint: {console}')
            invoke([console, '--help'], cwd, env, output / f'{name}-help.log')
        cases = []
        for name, text in SCHEMAS.items():
            fixture_schema = output / f'{name} schema.toml'
            fixture_schema.write_text(text, encoding='utf-8')
            cases.append((name, ROOT / f'tests/fixtures/{name}_small.csv', fixture_schema, .1))
        if source:
            cases.append(('real_covid', source, schema, .05))
        for name, data, spec, rate in cases:
            original = sha(data)
            for kind in ('random_cell', 'whole_feature'):
                label = name + '_' + kind
                destination = output / (label + ' output')
                command = [sys.executable, '-W', 'error::RuntimeWarning', '-m', 'longitudinal_impute.run',
                           '--data', data, '--schema', spec, '--models', ROOT / 'configs/models_default.toml',
                           '--output', destination, '--mask', kind, '--mask-rate', str(rate), '--seeds', '7']
                invoke(command, cwd, env, output / f'{label}.log')
                report['runs'][label] = validate_run(data, spec, destination, kind, rate, 7)
                before = {p.name: sha(p) for p in destination.iterdir() if p.is_file()}
                refusal = invoke(command, cwd, env, output / f'{label}-overwrite.log', succeeds=False)
                require('already exists; choose a new output directory' in refusal.stderr,
                        f'{label}: rerun failed for a reason other than output refusal')
                after = {p.name: sha(p) for p in destination.iterdir() if p.is_file()}
                require(before == after, f'{label}: rerun changed existing output')
                require(sha(data) == original, f'{label}: source data changed')
                report['runs'][label]['overwrite_refused_without_changes'] = True
                print(label + ': passed', flush=True)
        report['status'] = 'passed'
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        (output / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
