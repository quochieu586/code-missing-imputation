"""Impute one count table with every configured model, then score them on shared masks.

    python -m longitudinal_impute.run --data <csv> --schema <toml> --models <toml> --output <dir>
           [--mask whole_feature|random_cell|whole_feature_matched|none] [--mask-target JSON]
           [--mask-rate 0.05] [--seeds 0 1 2 3 4]
           [--only NAME [NAME ...]] [--save-predictions]

Every [name] table of --models runs, except tables with `enabled = false`. A table named
after a registered model runs that model; any other name needs `model = "<registered name>"`
(e.g. [sdm_no_time] with model = "same_day_mean") and its outputs carry the table name.
--only runs exactly the named tables (enabled or not), in the given order.
--save-predictions also writes predictions_<mask>.csv.gz: one row per hidden cell of every
completed masked run (see cells.py), for scoring outside this runner.
--mask whole_feature_matched (step 7) needs --mask-target, the frozen strata target of
evaluation/masks.py; its path, sha256 and content are written to config.json.
"""
import argparse
import csv
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import time
import tomllib

import numpy as np

from .data import (load_csv, read_schema, make_mask, validate_mask, model_input, restore_counts, check_completed,
                   write_completed_csv, apply_residual_rule, RESIDUAL_RULES, MASK_KINDS)
from .evaluation.masks import read_target
from .evaluation.metrics import wasserstein_scores, masked_errors, summarize, paired_summary, forced_zero
from .evaluation.metrics import prediction_rows, write_predictions
from .baselines.hron import HronKNN
from .baselines.tsagris import TsagrisKNN
from .baselines.micoda import MICoDaCovid
from .main_model import MainModel, initial_arrays, initial_from_arrays
from .baselines.zero_fill import ZeroFill
from .baselines.mean_fill import MeanFill
from .baselines.marginal_sampler import MarginalSampler
from .baselines.same_day_mean import SameDayMean

MICODA_INFO_KEYS = ('status', 'delta', 'reference', 'iterations', 'converged',
                    'termination_reason', 'initial_complete_rows', 'excluded_all_zero_rows',
                    'initial_regression_support', 'warning_counts', 'coefficients',
                    'coefficient_target_indices', 'missing_cells', 'resolved_cells',
                    'unresolved_cells', 'error', 'row_events', 'config')
MODELS = {'hron': HronKNN, 'micoda': MICoDaCovid, 'tsagris': TsagrisKNN, 'main': MainModel,
          'zero_fill': ZeroFill, 'mean_fill': MeanFill, 'marginal_sampler': MarginalSampler,
          'same_day_mean': SameDayMean}
SPACE = {'micoda': '18_with_other', 'hron': '18_with_other', 'tsagris': '18_with_other', 'main': '18_with_other'}


def run_model(data, model, hidden, residual_rule=None):
    """Fit on the visible cells and return (values, status, info) with counts checked.

    residual_rule (step 4, Hron/Tsagris/MICoDa other = true only): restore the model's
    composition with data.apply_residual_rule instead of the model's own rule. None keeps
    every model's current path unchanged.
    """
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
        if residual_rule is not None and model.proposals_ is not None:
            # The adapter's own R2 result is discarded; its core composition is restored again.
            completed, info['residual_rule'] = apply_residual_rule(data, model.proposals_, hidden, residual_rule)
            check_completed(data, completed, hidden)
            info['adapter_status'] = result.status
            return completed, 'completed', info
        if result.status == 'completed':
            check_completed(data, result.values, hidden)
        return result.values, result.status, info
    values = model_input(data, hidden, positive=model.input_domain == 'positive')
    if getattr(model, 'needs_context', False):
        # Row facts known for every row whatever the mask: N, time, group, entity. No counts.
        model.fit(values, context={'totals': data.totals, 'days': data.days, 'time_kind': data.time_kind,
                                   'groups': data.groups, 'entities': data.entities})
    else:
        model.fit(values)
    if residual_rule is None:
        completed = restore_counts(data, model.predict(values) * data.totals[:, None], hidden)
        check_completed(data, completed, hidden)
        return completed, 'completed', getattr(model, 'info_', None)
    completed, rule_info = apply_residual_rule(data, model.predict(values), hidden, residual_rule)
    check_completed(data, completed, hidden)
    return completed, 'completed', {**(getattr(model, 'info_', None) or {}), 'residual_rule': rule_info}


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


def error_code(info=None, exc=None):
    """Stable code of a failed or partial run: the exception's code, else the adapter's error code."""
    if exc is not None:
        return getattr(exc, 'code', type(exc).__name__)
    return ((info or {}).get('error') or {}).get('code') or 'unresolved_cells'


def cell_counts(data, hidden):
    """Missing-to-the-model cells split into forced to 0 (row residual 0) and model-decided."""
    forced = forced_zero(data, hidden)
    missing = hidden | ~np.isfinite(data.counts)
    return {'forced_zero_cells': int((forced & hidden).sum()), 'model_cells': int((hidden & ~forced).sum()),
            'missing_forced_zero_cells': int((forced & missing).sum()),
            'missing_model_cells': int((missing & ~forced).sum())}


def space(name, model=None):
    """Space label: the counts adapter is 17 targets (A17) or 18 with Other (A18)."""
    if isinstance(model, MICoDaCovid):
        return '18_with_other' if model.other else '17_targets'
    if isinstance(model, dict):                  # a table's parameters (replay): label by its model
        name = model_kind(name, model)
        if name == 'micoda':
            return '18_with_other' if model.get('other', True) else '17_targets'
    return SPACE.get(name, '18_with_other')


def model_kind(name, params):
    """Registered model of a table: its `model` key, else the table name itself."""
    return params.get('model', name)


# Models whose `residual_rule` key is handled by the runner (step 4). For same_day_mean the key
# is the model's own parameter (cap / rescale, step 3); other models have no such key.
RUNNER_RULES = {'hron': RESIDUAL_RULES, 'tsagris': RESIDUAL_RULES, 'micoda': RESIDUAL_RULES}


def residual_rule(name, params):
    """The runner-level residual rule of table [name], or None (model's current rule)."""
    return params.get('residual_rule') if model_kind(name, params) in RUNNER_RULES else None


def constructor_params(name, params):
    drop = {'model'} | ({'residual_rule'} if model_kind(name, params) in RUNNER_RULES else set())
    return {k: v for k, v in params.items() if k not in drop}


def check_rule(name, params):
    """Refuse a residual rule a model cannot take, with the reason, before any side effect."""
    kind, rule = model_kind(name, params), params.get('residual_rule')
    if rule is None:
        return
    if kind == 'same_day_mean' and rule == 'prop_other':
        raise ValueError(f'[{name}] residual_rule = "prop_other" is not applicable to same_day_mean: '
                         'it predicts no Other share (use "cap" or "rescale")')
    if kind == 'same_day_mean':
        return                                   # its own parameter, checked by the constructor
    if kind not in RUNNER_RULES:
        raise ValueError(f'[{name}] residual_rule is not applicable to {kind}')
    if rule not in RUNNER_RULES[kind]:
        raise ValueError(f'[{name}] residual_rule must be one of {RESIDUAL_RULES}, got {rule!r}')
    # Without an explicit key, MICoDaCovid's own default applies (other = true since PR #5).
    if kind == 'micoda' and params.get('other', inspect.signature(MICoDaCovid).parameters['other'].default) is not True:
        raise ValueError(f'[{name}] residual_rule is not applicable to micoda with other = false (A17): it '
                         'predicts no Other share, and its R1 restore already is the "rescale" rule')


def build(name, params):
    """A fresh model of table [name]: the class of model_kind, with its constructor parameters."""
    return MODELS[model_kind(name, params)](**constructor_params(name, params))


def resolve_models(models):
    """{table: {'model', 'class', 'params' (as written), 'effective_params' (with constructor defaults)}}."""
    out = {}
    for name, params in models.items():
        cls = MODELS[model_kind(name, params)]
        given = constructor_params(name, params)
        defaults = {p.name: p.default for p in inspect.signature(cls).parameters.values()
                    if p.default is not inspect.Parameter.empty}
        out[name] = {'model': model_kind(name, params), 'class': cls.__name__, 'params': given,
                     'effective_params': {**defaults, **given}}
        if residual_rule(name, params) is not None:
            out[name]['runner_residual_rule'] = residual_rule(name, params)
    return out


def read_models(path, only=None, construct=True):
    """[name] tables with constructor parameters -> {name: params} of the models to run.

    A table with `enabled = false` is validated but only runs when --only names it. A table
    whose name is not a registered model must say which one with `model = "<name>"`; params
    keep that key (build() drops it).
    construct=False (stage evaluate, a replay) selects the same tables without building a model.
    """
    with Path(path).open('rb') as stream:
        config = tomllib.load(stream)
    unnamed = [name for name, params in config.items() if name not in MODELS and 'model' not in params]
    if not config or unnamed:
        raise ValueError(f'{path}: declare one or more of {sorted(MODELS)} (or a table with '
                         f'model = "<one of them>"), got {sorted(config)}')
    selected = {}
    for name, params in config.items():
        params = dict(params)
        enabled = params.pop('enabled', True)
        if not isinstance(enabled, bool):
            raise ValueError(f'{path}: [{name}] enabled must be true or false')
        if model_kind(name, params) not in MODELS:
            raise ValueError(f'{path}: [{name}] model = {params["model"]!r} is not one of {sorted(MODELS)}')
        try:
            check_rule(name, params)
            if construct:
                build(name, params)         # reject bad parameters before any run starts
        except TypeError as exc:
            raise ValueError(f'{path}: [{name}] {exc}') from None
        selected[name] = (params, enabled)
    if only:
        absent = [name for name in only if name not in selected]
        if absent or len(set(only)) != len(only):
            raise ValueError(f'--only {list(only)}: names must be distinct tables of {path}, '
                             f'missing {absent}')
        return {name: selected[name][0] for name in only}
    models = {name: params for name, (params, enabled) in selected.items() if enabled}
    if not models:
        raise ValueError(f'{path}: every model has enabled = false; pick some with --only')
    return models


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
    print(f"{'model':16} {'space':14} {'pass':7} {'mask':14} {'ok/fail':8} {'sec':>7} "
          f"{'MSE prop (mean +- sd)':>24} {'MSE count':>11} {'MAE prop':>9}")
    for r in aggregate:
        mse = number(r.get('mse_proportion_mean'), 4) + ' +- ' + number(r.get('mse_proportion_sd'), 2)
        print(f"{r['model']:16} {r['space']:14} {r['pass']:7} {r['mask']:14} {r['runs_ok']}/{r['runs_failed']:<6} "
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
        if not isinstance(models, dict) or not models or any(
                not isinstance(p, dict) or model_kind(n, p) not in MODELS for n, p in models.items()):
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
            hidden, stats = make_mask(data, config['mask'], config['mask_rate'], seed, config.get('mask_target'))
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
    parser.add_argument('--mask', choices=('none', *MASK_KINDS))
    parser.add_argument('--mask-target', type=Path, help='strata target JSON, for whole_feature_matched only')
    parser.add_argument('--mask-rate', type=float)
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--pass', dest='passes', choices=('native', 'masked', 'both'))
    parser.add_argument('--stage', choices=('full', 'init', 'refine', 'evaluate'), default='full')
    parser.add_argument('--from-run', type=Path)
    parser.add_argument('--only', nargs='+', metavar='NAME',
                        help='run only these [name] tables of --models, even those with enabled = false')
    parser.add_argument('--save-predictions', action='store_true',
                        help='write predictions_<mask>.csv.gz with every hidden cell of completed masked runs')
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
        if args.only is not None and args.only != previous.get('only'):
            raise ValueError('source artifact only is incompatible with the requested override')
        args.only = previous.get('only')
        recorded_target = previous.get('mask_target')
        if recorded_target:
            args.mask_target = args.mask_target or Path(recorded_target['path'])
            if not args.mask_target.is_file() or sha256(args.mask_target) != recorded_target['sha256']:
                raise ValueError('mask target hash changed or file missing')
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
    if (args.mask == 'whole_feature_matched') != (args.mask_target is not None):
        raise ValueError('--mask-target is needed for --mask whole_feature_matched and only for it')
    target = read_target(args.mask_target) if args.mask_target else None

    # 1. Load and validate everything before writing; never overwrite an earlier run.
    if args.output.exists():
        raise FileExistsError(f'{args.output} already exists; choose a new output directory')
    data = load_csv(args.data, args.schema)
    # replay (evaluate) must not construct a model; --only / enabled select the same tables
    models = read_models(args.models, args.only, construct=args.stage != 'evaluate')
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
              'resolved_models': resolve_models(models), 'only': args.only,
              'mask': args.mask, 'mask_rate': args.mask_rate if args.mask != 'none' else None,
              'seeds': args.seeds if args.mask != 'none' else None, 'git': git_commit(),
              'group_map': data.group_map, 'save_predictions': args.save_predictions,
              'stage': args.stage, 'pass': args.passes, 'schema_sha256': sha256(args.schema),
              'models_sha256': sha256(args.models),
              'code_sha256': previous.get('code_sha256') if args.stage == 'evaluate' else code_hashes(),
              'from_run': str(source[0]) if source is not None else None}
    if args.stage == 'evaluate':
        config['compute_git'] = previous.get('compute_git', previous.get('git'))
        config['compute_stage'] = previous.get('compute_stage', previous['stage'])
        config['evaluation_code_sha256'] = sha256(Path(__file__).parent / 'evaluation' / 'metrics.py')
    if target is not None:
        config['mask_target'] = {'path': str(args.mask_target.resolve()),
                                 'sha256': hashlib.sha256(args.mask_target.read_bytes()).hexdigest(), **target}
    (output / 'config.json').write_text(json.dumps(config, indent=1, default=plain), encoding='utf-8')

    nothing = np.zeros(data.counts.shape, bool)
    rows, detail, failures, predictions = [], {'native': {}, 'masked': {}}, [], []
    detail['native_cells'] = cell_counts(data, nothing)
    artifacts = {'version': 1, 'files': {}, 'predictions': {}, 'initials': {}}

    def execute(name, model, hidden, case, rule=None):
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
        return run_model(data, model, hidden, rule)

    def failed(model, run, seed, exc):
        line = f'{model} | {run} | {"-" if seed is None else seed} | {type(exc).__name__} | {describe(exc, data)}'
        failures.append(line)
        with (output / 'errors.log').open('a', encoding='utf-8') as stream:
            stream.write(line + '\n')
        return {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}', 'error_code': error_code(exc=exc)}

    # 2. Native run: fill the real NaN cells from every observed cell.
    for name, params in (models.items() if args.passes in ('native', 'both') else []):
        start = time.perf_counter()
        try:
            model = params if args.stage == 'evaluate' else build(name, params)
            completed, status, info = execute(name, model, nothing, 'native', residual_rule(name, params))
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
                          'info': info, 'error': message, 'error_code': error_code(info)}
            else:
                message = adapter_error(status, info)
                failed(name, 'native', None, ValueError(message))
                result = {'status': 'failed', 'space': space(name, model), 'seconds': seconds,
                          'info': info, 'error': message, 'error_code': error_code(info)}
        except ValueError as exc:
            result = {**failed(name, 'native', None, exc), 'seconds': time.perf_counter() - start,
                      'space': space(model_kind(name, params))}
        detail['native'][name] = result
        rows.append({'model': name, 'pass': 'native', 'mask': 'none', 'seed': None,
                     'status': result['status'], 'space': result['space'],
                     'seconds': result['seconds'], 'imputed_cells': int((~np.isfinite(data.counts)).sum()),
                     'error': result.get('error', '')})

    # 3. Masked runs: one mask per seed, shared by every model; a fresh model per run.
    for seed in (args.seeds if args.mask != 'none' else []):
        hidden, stats = make_mask(data, args.mask, args.mask_rate, seed, target)
        np.save(output / f'mask_{args.mask}_seed{seed}.npy', hidden)
        stats.update(cell_counts(data, hidden))
        detail['masked'][seed] = {'mask': stats, 'models': {}}
        for name, params in models.items():
            start = time.perf_counter()
            try:
                model = params if args.stage == 'evaluate' else build(name, params)
                completed, status, info = execute(name, model, hidden, f'{args.mask}_seed{seed}',
                                                  residual_rule(name, params))
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
                              'info': info, 'error': message, 'error_code': error_code(info),
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
                    if args.save_predictions:
                        predictions += prediction_rows(data, completed, hidden, name, args.mask, seed,
                                                       config['schema'].get('time'))
            except ValueError as exc:
                result = {**failed(name, 'masked', seed, exc),
                          'seconds': time.perf_counter() - start, 'space': space(model_kind(name, params)),
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
    if args.save_predictions and args.mask != 'none':
        write_predictions(output / f'predictions_{args.mask}.csv.gz', predictions)
    for path in output.rglob('*'):
        if path.is_file():
            artifacts['files'][path.relative_to(output).as_posix()] = sha256(path)
    (output / 'artifacts.json').write_text(json.dumps(artifacts, indent=1), encoding='utf-8')
    print_table(aggregate)
    print('(mean/sd over the seeds each model completed, so models may be averaged over different '
          'seeds: not a ranking. Rank with tv2_step2/tier01.py, which uses the seeds common to all models.)')
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
