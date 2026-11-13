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

from .data import load_csv, read_schema, make_mask, validate_mask, model_input, restore_counts, check_completed, write_completed_csv
from .evaluate import wasserstein_scores, masked_errors, summarize, paired_summary
from .baselines.hron import HronKNN
from .baselines.tsagris import TsagrisKNN
from .baselines.micoda import MICoDaCovid
from .main_model import MainModel, initial_arrays, initial_from_arrays

MICODA_INFO_KEYS = ('status', 'delta', 'reference', 'iterations', 'converged',
                    'termination_reason', 'initial_complete_rows', 'excluded_all_zero_rows',
                    'initial_regression_support', 'warning_counts', 'coefficients',
                    'coefficient_target_indices', 'missing_cells', 'resolved_cells',
                    'unresolved_cells', 'error', 'row_events', 'config')
MODELS = {'hron': HronKNN, 'micoda': MICoDaCovid, 'tsagris': TsagrisKNN, 'main': MainModel}
SPACE = {'micoda': '18_with_other', 'hron': '18_with_other', 'tsagris': '18_with_other', 'main': '18_with_other'}


def run_model(data, model, hidden):
    """Fit on the visible cells and return (values, status, info) with counts checked."""
    if isinstance(model, MainModel):
        completed = model.impute(data, hidden)
        check_completed(data, completed, hidden)
        return completed, 'completed', model.info_
    if isinstance(model, MICoDaCovid):
        result = model.impute_counts(data.counts, data.totals, hidden)
        if result.values is None and result.status != 'failed':
            raise ValueError(f'counts adapter returned no values: {result.status}')
        info = {k: result.diagnostics[k] for k in MICODA_INFO_KEYS if k in result.diagnostics}
        info['unassigned'] = result.unassigned
        if result.status == 'completed':
            check_completed(data, result.values, hidden)
        return result.values, result.status, info
    values = model_input(data, hidden, positive=model.input_domain == 'positive')
    model.fit(values)
    completed = restore_counts(data, model.predict(values) * data.totals[:, None], hidden)
    check_completed(data, completed, hidden)
    return completed, 'completed', getattr(model, 'info_', None)


def impute(data, model, hidden):
    """Legacy helper: fit on the visible cells and return checked counts."""
    values, status, info = run_model(data, model, hidden)
    if status != 'completed':
        raise ValueError(adapter_error(status, info))
    return values


def adapter_error(status, info):
    error = (info or {}).get('error')
    if error:
        return f'{error["code"]}: {error["message"]}; context={error.get("context", {})}'
    return f'{status} completion; {(info or {}).get("unresolved_cells", "-")} cells unresolved'


def space(name, model=None):
    """Space label: the counts adapter is 17 targets (A17) or 18 with Other (A18)."""
    if isinstance(model, MICoDaCovid):
        return '18_with_other' if model.other else '17_targets'
    if name == 'micoda' and isinstance(model, dict):
        return '18_with_other' if model.get('other', True) else '17_targets'
    return SPACE.get(name, '18_with_other')


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
    print(f"{'model':10} {'space':14} {'pass':7} {'mask':14} {'ok/fail':8} {'sec':>7} "
          f"{'MSE prop (mean +- sd)':>24} {'MSE count':>11} {'MAE prop':>9}")
    for r in aggregate:
        mse = number(r.get('mse_proportion_mean'), 4) + ' +- ' + number(r.get('mse_proportion_sd'), 2)
        print(f"{r['model']:10} {r['space']:14} {r['pass']:7} {r['mask']:14} {r['runs_ok']}/{r['runs_failed']:<6} "
              f"{number(r.get('seconds_mean'), 3):>7} {mse:>24} {number(r.get('mse_count_mean'), 6):>11} "
              f"{number(r.get('mae_proportion_mean'), 3):>9}")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_prediction(output, case, name, values, artifacts):
    relative = f'predictions/{case}_{name}.npy'
    path = output / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        np.save(stream, values, allow_pickle=False)
    artifacts['files'][relative] = sha256(path)
    artifacts['predictions'][f'{case}/{name}'] = relative


def save_history(output, case, info, artifacts):
    relative = f'main/{case}/history.csv'
    path = output / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['iteration', 'loss', 'max_change_proportion', 'converged', 'seconds'])
        writer.writeheader()
        writer.writerows((info or {}).get('history', []))
    artifacts['files'][relative] = sha256(path)


def code_hashes():
    root = Path(__file__).parent
    return {p: sha256(root / p) for p in ('data.py', 'main_model.py', 'network.py', 'baselines/tsagris.py')}


def artifact_path(root, relative, files):
    """Bound artifact reads to this run, including symlinks; hashes are not signatures."""
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError('artifact path must be relative')
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or relative not in files or not path.is_file():
        raise ValueError(f'invalid/missing artifact: {relative}')
    if sha256(path) != files[relative]:
        raise ValueError(f'artifact checksum changed: {relative}')
    return path


def load_run(root):
    root = root.resolve()
    try:
        artifacts = json.loads((root / 'artifacts.json').read_text(encoding='utf-8'))
        if artifacts['version'] != 1 or not isinstance(artifacts['files'], dict):
            raise ValueError('unsupported artifact version')
        for relative in artifacts['files']:
            artifact_path(root, relative, artifacts['files'])
        config = json.loads(artifact_path(root, 'config.json', artifacts['files']).read_text(encoding='utf-8'))
        summary = json.loads(artifact_path(root, 'summary.json', artifacts['files']).read_text(encoding='utf-8'))
        validate_run_metadata(config, summary)
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f'invalid/incomplete source run artifacts: {exc}') from exc
    return root, config, summary, artifacts


def validate_run_metadata(config, summary):
    """Reject incomplete inherited recipes/results before defaults or output writes."""
    try:
        if not isinstance(config, dict) or not isinstance(summary, dict):
            raise ValueError('config and summary must be objects')
        if config['stage'] not in ('full', 'init', 'refine', 'evaluate'):
            raise ValueError('invalid stage')
        if config['pass'] not in ('native', 'masked', 'both'):
            raise ValueError('invalid pass')
        for key in ('data', 'schema_path', 'models_path', 'data_sha256', 'schema_sha256', 'models_sha256'):
            if not isinstance(config[key], str) or not config[key]:
                raise ValueError(f'invalid {key}')
        models = config['models']
        if not isinstance(models, dict) or not models or set(models) - set(MODELS):
            raise ValueError('invalid models')
        if config['pass'] == 'native':
            if config['mask'] != 'none' or config['seeds'] is not None or config['mask_rate'] is not None:
                raise ValueError('native recipe must not schedule masks')
            seeds = []
        else:
            rate, seeds = config['mask_rate'], config['seeds']
            if type(rate) not in (int, float):
                raise ValueError('invalid mask rate')
            validate_mask(config['mask'], rate)
            if (not isinstance(seeds, list) or not seeds
                    or any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds)):
                raise ValueError('invalid seeds')
        native, masked = summary['native'], summary['masked']
        if not isinstance(native, dict) or not isinstance(masked, dict) or set(masked) != {str(s) for s in seeds}:
            raise ValueError('missing or unexpected cases')
        cases = [native] if config['pass'] in ('native', 'both') else []
        if not cases and native:
            raise ValueError('unexpected native case')
        cases += [masked[str(seed)]['models'] for seed in seeds]
        for results in cases:
            if not isinstance(results, dict) or set(results) != set(models):
                raise ValueError('missing or unexpected case models')
            for result in results.values():
                if not isinstance(result, dict) or result['status'] not in ('ok', 'partial', 'failed', 'initialized'):
                    raise ValueError('invalid result status')
                seconds = result['seconds']
                if type(seconds) not in (int, float) or not np.isfinite(seconds) or seconds < 0:
                    raise ValueError('invalid compute seconds')
                info = result.get('info')
                if info is not None:
                    if not isinstance(info, dict):
                        raise ValueError('result info must be an object or null')
                    error = info.get('error')
                    if error is not None and (not isinstance(error, dict) or any(
                            not isinstance(error.get(key), str) for key in ('code', 'message'))):
                        raise ValueError('error diagnostic requires string code and message')
                    if 'history' in info:
                        history = info['history']
                        columns = {'iteration', 'loss', 'max_change_proportion', 'converged', 'seconds'}
                        if not isinstance(history, list) or any(
                                not isinstance(row, dict) or set(row) - columns for row in history):
                            raise ValueError('invalid history rows')
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'invalid source run metadata: {exc}') from exc


def scheduled_cases(config, data):
    if config['pass'] in ('native', 'both'):
        yield 'native', None, np.zeros(data.counts.shape, bool), None
    if config['pass'] in ('masked', 'both'):
        for seed in config['seeds']:
            hidden, stats = make_mask(data, config['mask'], config['mask_rate'], seed)
            yield f'{config["mask"]}_seed{seed}', seed, hidden, stats


def save_initial(output, case, initial, artifacts):
    relative = f'main/{case}/initial.npz'
    path = output / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        np.savez_compressed(stream, **initial_arrays(initial))
    artifacts['files'][relative] = sha256(path)
    artifacts['initials'][case] = relative


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--schema', type=Path)
    parser.add_argument('--models', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--mask', choices=('none', 'random_cell', 'whole_feature'))
    parser.add_argument('--mask-rate', type=float)
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--pass', dest='passes', choices=('native', 'masked', 'both'))
    parser.add_argument('--stage', choices=('full', 'init', 'refine', 'evaluate'), default='full')
    parser.add_argument('--from-run', type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f'{args.output} already exists; choose a new output directory')
    source = None
    if args.stage in ('refine', 'evaluate'):
        if args.from_run is None:
            raise ValueError(f'--stage {args.stage} requires --from-run')
        source = load_run(args.from_run)
        root, previous, previous_summary, previous_artifacts = source
        if args.output.resolve().is_relative_to(root):
            raise ValueError('output must not modify or be inside the source run')
        for name, path_key in (('data', 'data'), ('schema', 'schema_path'), ('models', 'models_path')):
            path = getattr(args, name) or Path(previous[path_key])
            if not path.is_file() or sha256(path) != previous[f'{name}_sha256']:
                raise ValueError(f'{name} input hash changed or file missing')
            setattr(args, name, path)
        for attr, key in (('passes', 'pass'), ('mask', 'mask'), ('mask_rate', 'mask_rate'), ('seeds', 'seeds')):
            requested = getattr(args, attr)
            if requested is not None and requested != previous[key]:
                raise ValueError(f'source artifact {key} is incompatible with the requested override')
            setattr(args, attr, previous[key])
        if args.stage == 'refine' and previous['stage'] != 'init':
            raise ValueError('refine requires a successful init-stage source run')
        if args.stage == 'evaluate' and previous['stage'] == 'init':
            raise ValueError('evaluate requires saved predictions, not initialized-only artifacts')
        if args.stage == 'refine' and previous.get('code_sha256') != code_hashes():
            raise ValueError('initial artifact preprocessing/network code changed')
    elif args.from_run is not None:
        raise ValueError('--from-run is only valid for refine/evaluate')
    if any(getattr(args, k) is None for k in ('data', 'schema', 'models')):
        raise ValueError('--data, --schema and --models are required for model computation')
    args.mask = args.mask or 'none'
    args.mask_rate = .05 if args.mask_rate is None else args.mask_rate
    args.seeds = [0, 1, 2, 3, 4] if args.seeds is None else args.seeds
    args.passes = args.passes or ('both' if args.mask != 'none' else 'native')
    if args.passes == 'native':
        args.mask = 'none'
    elif args.mask == 'none':
        raise ValueError('--pass masked/both requires a non-none --mask')
    # Mask settings are checked here, before any output or model run (same rate rule as make_mask).
    if args.mask != 'none':
        validate_mask(args.mask, args.mask_rate)
        if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
            raise ValueError(f'--seeds must be distinct nonnegative integers, got {args.seeds}')

    # 1. Load and validate everything before writing; never overwrite an earlier run.
    if args.output.exists():
        raise FileExistsError(f'{args.output} already exists; choose a new output directory')
    data = load_csv(args.data, args.schema)
    if args.stage == 'evaluate':
        with args.models.open('rb') as stream:
            models = tomllib.load(stream)  # replay must not construct a model
    else:
        models = read_models(args.models)
    if args.stage in ('init', 'refine') and set(models) != {'main'}:
        raise ValueError('init/refine stage requires main-only model configuration')
    initial_states, replay = {}, {}
    if source is not None:
        for case, seed, hidden, _ in scheduled_cases(previous, data):
            if seed is not None:
                relative = f'mask_{args.mask}_seed{seed}.npy'
                saved = np.load(artifact_path(root, relative, previous_artifacts['files']), allow_pickle=False)
                if saved.dtype != bool or not np.array_equal(saved, hidden):
                    raise ValueError('source artifact mask differs from the declared data/seed recipe')
            case_results = previous_summary['native'] if seed is None else previous_summary['masked'][str(seed)]['models']
            if set(case_results) != set(models):
                raise ValueError('source artifact model set changed')
            if args.stage == 'evaluate':
                for name in models:
                    result = case_results[name]
                    if result.get('status') not in ('ok', 'partial', 'failed'):
                        raise ValueError('source artifact has an unscorable status')
                    values = None
                    if result['status'] in ('ok', 'partial'):
                        try:
                            relative = previous_artifacts['predictions'][f'{case}/{name}']
                            values = np.load(artifact_path(root, relative, previous_artifacts['files']), allow_pickle=False)
                        except (OSError, KeyError, TypeError) as exc:
                            raise ValueError(f'invalid prediction artifact: {exc}') from exc
                        if result['status'] == 'ok':
                            check_completed(data, values, hidden)
                        else:
                            visible = np.isfinite(data.counts) & ~hidden
                            if (values.shape != data.counts.shape or np.isinf(values).any()
                                    or not np.array_equal(values[visible], data.counts[visible])):
                                raise ValueError('partial prediction artifact changed visible counts or shape')
                            check_completed(data, np.where(np.isnan(values), 0., values), hidden)
                    replay[f'{case}/{name}'] = (result, values)
                continue
            result = case_results['main']
            if result['status'] != 'initialized':
                raise ValueError('refine source case was not successfully initialized')
            try:
                relative = previous_artifacts['initials'][case]
                with np.load(artifact_path(root, relative, previous_artifacts['files']), allow_pickle=False) as saved:
                    if len(saved.files) != len(set(saved.files)):
                        raise ValueError('duplicate initial artifact arrays')
                    arrays = {key: saved[key] for key in saved.files}
                initial_states[case] = initial_from_arrays(data, hidden, arrays, MainModel(**models['main']))
            except (KeyError, TypeError, OSError) as exc:
                raise ValueError(f'invalid initial artifact: {exc}') from exc
    output = args.output
    output.mkdir(parents=True)
    config = {'data': str(args.data.resolve()), 'data_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
              'rows': len(data.counts), 'schema_path': str(args.schema.resolve()), 'schema': read_schema(args.schema),
              'models_path': str(args.models.resolve()), 'models': models,
              'mask': args.mask, 'mask_rate': args.mask_rate if args.mask != 'none' else None,
              'seeds': args.seeds if args.mask != 'none' else None, 'git': git_commit(),
              'stage': args.stage, 'pass': args.passes, 'schema_sha256': sha256(args.schema),
              'models_sha256': sha256(args.models),
              'code_sha256': previous.get('code_sha256') if args.stage == 'evaluate' else code_hashes(),
              'from_run': str(source[0]) if source is not None else None}
    if args.stage == 'evaluate':
        config['compute_git'] = previous.get('compute_git', previous.get('git'))
        config['compute_stage'] = previous.get('compute_stage', previous['stage'])
        config['evaluation_code_sha256'] = sha256(Path(__file__).with_name('evaluate.py'))
    (output / 'config.json').write_text(json.dumps(config, indent=1, default=plain), encoding='utf-8')

    rows, detail, failures = [], {'native': {}, 'masked': {}}, []
    artifacts = {'version': 1, 'files': {}, 'predictions': {}, 'initials': {}}

    def execute(name, model, hidden, case):
        if args.stage == 'evaluate':
            result, values = replay[f'{case}/{name}']
            status = 'completed' if result['status'] == 'ok' else result['status']
            info = result.get('info')
            if status == 'failed' and info is None:
                info = {'error': {'code': 'source_run_error', 'message': result.get('error', 'source run failed')}}
            return values, status, info
        if args.stage == 'init':
            initial = model.prepare(data, hidden)
            save_initial(output, case, initial, artifacts)
            return None, 'initialized', {'trained': False, 'iterations': 0, 'termination_reason': 'initialized'}
        if args.stage == 'refine':
            completed = model.impute(data, hidden, initial=initial_states[case])
            check_completed(data, completed, hidden)
            return completed, 'completed', model.info_
        return run_model(data, model, hidden)

    def failed(model, run, seed, exc):
        line = f'{model} | {run} | {"-" if seed is None else seed} | {type(exc).__name__} | {describe(exc, data)}'
        failures.append(line)
        with (output / 'errors.log').open('a', encoding='utf-8') as stream:
            stream.write(line + '\n')
        return {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}

    # 2. Native run: fill the real NaN cells from every observed cell.
    nothing = np.zeros(data.counts.shape, bool)
    for name, params in (models.items() if args.passes in ('native', 'both') else []):
        start = time.perf_counter()
        try:
            model = params if args.stage == 'evaluate' else MODELS[name](**params)
            completed, status, info = execute(name, model, nothing, 'native')
            seconds = replay[f'native/{name}'][0]['seconds'] if args.stage == 'evaluate' else time.perf_counter() - start
            suffix = '_completed' if status == 'completed' else (
                '_partial' if status == 'partial' else '')
            if status in ('completed', 'partial') and completed is not None:
                write_completed_csv(output / f'{name}{suffix}.csv', data, completed,
                                    allow_partial=status == 'partial')
                save_prediction(output, 'native', name, completed, artifacts)
                if name == 'main':
                    save_history(output, 'native', info, artifacts)
            if status == 'initialized':
                result = {'status': 'initialized', 'space': space(name, model), 'seconds': seconds, 'info': info}
            elif status == 'completed':
                result = {'status': 'ok', 'space': space(name, model), 'seconds': seconds,
                          'info': info, 'wasserstein': wasserstein_scores(data, completed)}
            elif status == 'partial':
                message = adapter_error(status, info)
                failed(name, 'native', None, ValueError(message))
                result = {'status': 'partial', 'space': space(name, model), 'seconds': seconds,
                          'info': info, 'error': message}
            else:
                message = adapter_error(status, info)
                failed(name, 'native', None, ValueError(message))
                result = {'status': 'failed', 'space': space(name, model), 'seconds': seconds,
                          'info': info, 'error': message}
        except ValueError as exc:
            result = {**failed(name, 'native', None, exc), 'seconds': time.perf_counter() - start,
                      'space': space(name)}
        detail['native'][name] = result
        rows.append({'model': name, 'pass': 'native', 'mask': 'none', 'seed': None,
                     'status': result['status'], 'space': result['space'],
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
                model = params if args.stage == 'evaluate' else MODELS[name](**params)
                completed, status, info = execute(name, model, hidden, f'{args.mask}_seed{seed}')
                seconds = (replay[f'{args.mask}_seed{seed}/{name}'][0]['seconds'] if args.stage == 'evaluate'
                           else time.perf_counter() - start)
                if status in ('completed', 'partial') and completed is not None:
                    case = f'{args.mask}_seed{seed}'
                    save_prediction(output, case, name, completed, artifacts)
                    if name == 'main':
                        save_history(output, case, info, artifacts)
                if status == 'initialized':
                    result = {'status': 'initialized', 'space': space(name, model), 'seconds': seconds, 'info': info}
                elif status != 'completed':
                    message = adapter_error(status, info)
                    failed(name, 'masked', seed, ValueError(message))
                    result = {'status': status, 'space': space(name, model), 'seconds': seconds,
                              'info': info, 'error': message,
                              'coverage': float((hidden & (np.isfinite(completed) if completed is not None
                                                           else np.zeros(hidden.shape, bool))).sum()
                                                / max(hidden.sum(), 1))}
                else:
                    resolved = hidden & np.isfinite(completed)
                    coverage = float(resolved.sum() / max(hidden.sum(), 1))
                    result = {'status': 'ok', 'space': space(name, model), 'seconds': seconds,
                              'info': info,
                              'errors': masked_errors(data, completed, hidden) if coverage == 1. else None,
                              'coverage': coverage}
            except ValueError as exc:
                result = {**failed(name, 'masked', seed, exc),
                          'seconds': time.perf_counter() - start, 'space': space(name),
                          'coverage': 0.}
            detail['masked'][seed]['models'][name] = result
            rows.append({'model': name, 'pass': 'masked', 'mask': args.mask, 'seed': seed,
                         'status': result['status'], 'space': result['space'],
                         'seconds': result['seconds'],
                         'hidden_cells': stats['hidden_cells'], 'actual_rate': stats['actual_rate'],
                         'complete_rows_after_mask': stats['complete_rows_after'],
                         'resolved_cells': result.get('coverage', 0.) * stats['hidden_cells'],
                          **flat_errors(result.get('errors') or {}), 'error': result.get('error', '')})

    # 4. summary.csv: one row per model x pass x seed; summary.json: everything above.
    aggregate = summarize(rows) if args.stage != 'init' else []
    with (output / 'summary.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)), restval='')
        writer.writeheader()
        writer.writerows(rows)
    paired = paired_summary(rows) if args.stage != 'init' else {'cases': {}, 'aggregate': []}
    (output / 'summary.json').write_text(json.dumps({**detail, 'rows': rows, 'aggregate': aggregate, 'paired': paired},
                                                    indent=1, default=plain), encoding='utf-8')
    for path in output.rglob('*'):
        if path.is_file():
            artifacts['files'][path.relative_to(output).as_posix()] = sha256(path)
    (output / 'artifacts.json').write_text(json.dumps(artifacts, indent=1), encoding='utf-8')
    print_table(aggregate)
    if paired['cases']:
        print('\nPaired masked comparison (shared successful seeds, same model space):')
        print_table(paired['aggregate'])
        for kind, case in paired['cases'].items():
            print(f'{kind}: included={case["seeds_included"]}, excluded={case["seeds_excluded"]}; {case["reason"]}')
    if failures:
        print(f'\n{len(failures)} failed run(s), see {output / "errors.log"}:')
        print('\n'.join(failures))
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
