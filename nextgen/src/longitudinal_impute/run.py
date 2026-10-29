"""Impute one count table with every configured model, then score them on shared masks.

    python -m longitudinal_impute.run --data <csv> --schema <toml> --models <toml> --output <dir>
           [--mask whole_feature|random_cell|none] [--mask-rate 0.05] [--seeds 0 1 2 3 4]
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time
import tomllib

import numpy as np

from .data import load_csv, read_schema, make_mask, model_input, restore_counts, check_completed, write_completed_csv
from .evaluate import wasserstein_scores, masked_errors, summarize
from .baselines.hron import HronKNN
from .baselines.micoda import MICoDa
from .baselines.tsagris import TsagrisKNN

MODELS = {'hron': HronKNN, 'micoda': MICoDa, 'tsagris': TsagrisKNN}


def impute(data, model, hidden):
    """Fit on the visible cells, predict proportions and return checked counts."""
    values = model_input(data, hidden, positive=model.input_domain == 'positive')
    model.fit(values)
    completed = restore_counts(data, model.predict(values) * data.totals[:, None], hidden)
    check_completed(data, completed, hidden)
    return completed


def read_models(path):
    """[name] tables with constructor parameters; only the models listed are run."""
    with Path(path).open('rb') as stream:
        config = tomllib.load(stream)
    if not config or set(config) - set(MODELS):
        raise ValueError(f'{path}: declare one or more of {sorted(MODELS)}, got {sorted(config)}')
    for name, params in config.items():
        try:
            MODELS[name](**params)          # reject bad parameters before any run starts
        except TypeError as exc:
            raise ValueError(f'{path}: [{name}] {exc}') from None
    return config


def describe(exc, data):
    """Error message plus its context, with the row index translated to entity and CSV line."""
    context = dict(getattr(exc, 'context', {}))
    row = context.get('row')
    if isinstance(row, (int, np.integer)) and 0 <= row < len(data.entities):
        context.update(entity=data.entities[row], csv_line=data.lines[row])
    feature = context.get('feature')
    names = (*data.targets, 'Other')
    if isinstance(feature, (int, np.integer)) and 0 <= feature < len(names):
        context['feature'] = names[feature]
    extra = ', '.join(f'{k}={v}' for k, v in context.items())
    return f'{exc}' + (f' [{extra}]' if extra else '')


def git_commit():
    here = Path(__file__).parent
    try:
        head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=here, capture_output=True, text=True, check=True)
        dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=here,
                               capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return {'commit': head.stdout.strip(), 'uncommitted_changes': bool(dirty.stdout.strip())}


def plain(value):
    """json.dumps fallback for numpy scalars and arrays."""
    return value.tolist() if isinstance(value, (np.ndarray, np.generic)) else str(value)


def flat_errors(errors):
    """{'all': {...}, 'truth_zero': {...}} -> mse_count, ..., mse_count_truth_zero, cells_truth_zero, ..."""
    row = {}
    for group, scores in errors.items():
        for key, value in scores.items():
            row[key if group == 'all' else f'{key}_{group}'] = value
    return row


def print_table(aggregate):
    def number(value, digits):
        return '-' if value is None else f'{value:.{digits}g}'
    print(f"{'model':10} {'pass':7} {'mask':14} {'ok/fail':8} {'sec':>7} "
          f"{'MSE prop (mean +- sd)':>24} {'MSE count':>11} {'MAE prop':>9}")
    for r in aggregate:
        mse = number(r.get('mse_proportion_mean'), 4) + ' +- ' + number(r.get('mse_proportion_sd'), 2)
        print(f"{r['model']:10} {r['pass']:7} {r['mask']:14} {r['runs_ok']}/{r['runs_failed']:<6} "
              f"{number(r.get('seconds_mean'), 3):>7} {mse:>24} {number(r.get('mse_count_mean'), 6):>11} "
              f"{number(r.get('mae_proportion_mean'), 3):>9}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--schema', required=True, type=Path)
    parser.add_argument('--models', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--mask', choices=('none', 'random_cell', 'whole_feature'), default='none')
    parser.add_argument('--mask-rate', type=float, default=.05)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    args = parser.parse_args(argv)
    # Mask settings are checked here, before any output or model run (same rate rule as make_mask).
    if args.mask != 'none':
        if not np.isfinite(args.mask_rate) or not 0 <= args.mask_rate < 1:
            parser.error(f'--mask-rate must be in [0, 1), got {args.mask_rate!r}')
        if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
            parser.error(f'--seeds must be distinct nonnegative integers, got {args.seeds}')

    # 1. Load and validate everything before writing; never overwrite an earlier run.
    if args.output.exists():
        raise FileExistsError(f'{args.output} already exists; choose a new output directory')
    data = load_csv(args.data, args.schema)
    models = read_models(args.models)
    output = args.output
    output.mkdir(parents=True)
    config = {'data': str(args.data.resolve()), 'data_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
              'rows': len(data.counts), 'schema_path': str(args.schema.resolve()), 'schema': read_schema(args.schema),
              'models_path': str(args.models.resolve()), 'models': models,
              'mask': args.mask, 'mask_rate': args.mask_rate if args.mask != 'none' else None,
              'seeds': args.seeds if args.mask != 'none' else None, 'git': git_commit()}
    (output / 'config.json').write_text(json.dumps(config, indent=1, default=plain), encoding='utf-8')

    rows, detail, failures = [], {'native': {}, 'masked': {}}, []

    def failed(model, run, seed, exc):
        line = f'{model} | {run} | {"-" if seed is None else seed} | {type(exc).__name__} | {describe(exc, data)}'
        failures.append(line)
        with (output / 'errors.log').open('a', encoding='utf-8') as stream:
            stream.write(line + '\n')
        return {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}

    # 2. Native run: fill the real NaN cells from every observed cell.
    nothing = np.zeros(data.counts.shape, bool)
    for name, params in models.items():
        start = time.perf_counter()
        try:
            model = MODELS[name](**params)
            completed = impute(data, model, nothing)
            seconds = time.perf_counter() - start
            write_completed_csv(output / f'{name}_completed.csv', data, completed)
            result = {'status': 'ok', 'seconds': seconds, 'info': getattr(model, 'info_', None),
                      'wasserstein': wasserstein_scores(data, completed)}
        except ValueError as exc:
            result = {**failed(name, 'native', None, exc), 'seconds': time.perf_counter() - start}
        detail['native'][name] = result
        rows.append({'model': name, 'pass': 'native', 'mask': 'none', 'seed': None, 'status': result['status'],
                     'seconds': result['seconds'], 'imputed_cells': int((~np.isfinite(data.counts)).sum()),
                     'error': result.get('error', '')})

    # 3. Masked runs: one mask per seed, shared by every model; a fresh model per run.
    for seed in (args.seeds if args.mask != 'none' else []):
        hidden, stats = make_mask(data, args.mask, args.mask_rate, seed)
        np.save(output / f'mask_{args.mask}_seed{seed}.npy', hidden)
        detail['masked'][seed] = {'mask': stats, 'models': {}}
        for name, params in models.items():
            start = time.perf_counter()
            try:
                model = MODELS[name](**params)
                completed = impute(data, model, hidden)
                result = {'status': 'ok', 'seconds': time.perf_counter() - start,
                          'info': getattr(model, 'info_', None), 'errors': masked_errors(data, completed, hidden)}
            except ValueError as exc:
                result = {**failed(name, 'masked', seed, exc), 'seconds': time.perf_counter() - start}
            detail['masked'][seed]['models'][name] = result
            rows.append({'model': name, 'pass': 'masked', 'mask': args.mask, 'seed': seed,
                         'status': result['status'], 'seconds': result['seconds'],
                         'hidden_cells': stats['hidden_cells'], 'actual_rate': stats['actual_rate'],
                         'complete_rows_after_mask': stats['complete_rows_after'],
                         **flat_errors(result.get('errors', {})), 'error': result.get('error', '')})

    # 4. summary.csv: one row per model x pass x seed; summary.json: everything above.
    aggregate = summarize(rows)
    with (output / 'summary.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)), restval='')
        writer.writeheader()
        writer.writerows(rows)
    (output / 'summary.json').write_text(json.dumps({**detail, 'rows': rows, 'aggregate': aggregate},
                                                    indent=1, default=plain), encoding='utf-8')
    print_table(aggregate)
    if failures:
        print(f'\n{len(failures)} failed run(s), see {output / "errors.log"}:')
        print('\n'.join(failures))
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
