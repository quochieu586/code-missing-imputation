"""Frozen, resumable offline benchmark jobs. Every file uses exclusive writes.

One process owns each fold directory. No worker imports another runner. Job
completion is a result.json written AFTER prediction arrays; partial failures
remain as evidence and are never treated as completed or silently overwritten.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import tomllib
import traceback
import numpy as np
from ..contracts import FitScope
from ..metadata import MetadataEncoder
from ..baselines.adapters import CountBaselineAdapter
from ..baselines.csv_io import load_csv, schema_from_config, write_completed_csv
from ..baselines.panels import hide_targets, subset
from ..baselines.registry import make_core, core_configuration
from ..baselines.common import BaselineError
from ..errors import NumericalError
from .protocol import make_folds, digest
from .masks import make_mask, mask_hash
from .metrics import score
from .artifacts import (write_json, publish_json, new_attempt, file_hash,
                        prediction_artifact_path, final_csv_artifact_path, get_mask,
                        evaluation_identity as _evaluation_identity,
                        read_completed_evaluation as _read_completed_evaluation)
from .completion import RunArtifacts
from .candidates import supplementary_candidate, final_candidate


def failure_status(error):
    if isinstance(error, NumericalError):
        return 'numerical_failure'
    if isinstance(error, BaselineError):
        return 'not_applicable'
    return 'failed_runtime'


def structured_error(error, *, stage):
    """Serializable error detail; traceback is captured at the catch site."""
    return {'stage': stage, 'error_code': getattr(error, 'code', 'unexpected_error'),
            'error_type': type(error).__name__, 'error': str(error),
            'context': dict(getattr(error, 'context', {})),
            'traceback': traceback.format_exc()}


def source_hash():
    source = Path(__file__).parents[1]
    return digest({p.relative_to(source).as_posix(): file_hash(p) for p in sorted(source.rglob('*.py'))})


def prepare(root, config_path, output):
    root, output = Path(root), Path(output)
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    with (root / config['schema']).open('rb') as stream:
        schema = schema_from_config(tomllib.load(stream))
    raw, _, _ = load_csv(root / config['data'], schema)
    folds = make_folds(raw, config['folds'], config['split_seed'])
    output.mkdir(parents=True, exist_ok=False)
    masks = output / 'masks'; masks.mkdir()
    hashes = {}
    for scenario in config['scenarios']:
        for seed in config['mask_seeds']:
            mask = make_mask(raw, scenario, seed, config['mask_rate'])
            with (masks / f'{scenario}_{seed}.npz').open('xb') as stream:
                np.savez_compressed(stream, hidden=mask)
            hashes[f'{scenario}_{seed}'] = {'hash': mask_hash(mask), 'targets': int(mask.sum())}
    manifest = {'config': config, 'config_hash': digest(config), 'dataset_hash': file_hash(root / config['data']),
                'schema_hash': file_hash(root / config['schema']), 'source_hash': source_hash(),
                'python': platform.python_version(), 'numpy': np.__version__, 'rows': len(raw.row_ids),
                'locations': len(set(raw.entity_ids)), 'targets': list(schema.target_columns),
                'original_observed': int(raw.original_observed.sum()), 'native_missing': int((~raw.original_observed).sum()),
                'split_hash': digest(folds), 'mask_registry': hashes,
                'temporal': 'offline_bidirectional', 'native_missing_accuracy': None}
    write_json(output / 'manifest.json', manifest)
    write_json(output / 'folds.json', folds)
    return manifest


def load_run(root, output):
    root, output = Path(root), Path(output)
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    config = manifest['config']
    if digest(config) != manifest['config_hash']:
        raise ValueError('Frozen configuration hash mismatch')
    if source_hash() != manifest['source_hash']:
        raise ValueError('Source differs from frozen manifest; use a NEW experiment directory')
    if file_hash(root / config['data']) != manifest['dataset_hash']:
        raise ValueError('Dataset differs from frozen manifest')
    if file_hash(root / config['schema']) != manifest['schema_hash']:
        raise ValueError('Dataset schema differs from frozen manifest')
    if digest(json.loads((output / 'folds.json').read_text(encoding='utf-8'))) != manifest['split_hash']:
        raise ValueError('Frozen location split hash mismatch')
    with (root / config['schema']).open('rb') as stream:
        raw, fields, rows = load_csv(root / config['data'], schema_from_config(tomllib.load(stream)))
    return manifest, raw, fields, rows


def compact_diagnostics(result):
    diag = result.diagnostics
    reasons, policies = Counter(), Counter()
    for record in diag.get('fallback_records', []):
        reasons[record.get('error_code', record.get('reason', 'unknown'))] += 1
    for event in diag.get('core_row_events', []):
        if 'policy' in event:
            policies[event['policy']] += 1
    return {'fallback_rows': sum(reasons.values()), 'fallback_reasons': dict(reasons),
            'core_policies': dict(policies), 'repair_l1': diag['repair_l1'],
            'zero_delta': diag['zero_delta'], 'metadata_features': diag['metadata_features'],
            'fit_summary': diag['fit_summary'], 'prediction_summary': diag['prediction_summary'],
            'tuning': diag['tuning']}


def make_adapter(config, candidate):
    core = make_core(candidate['method'], degenerate_policies=config['degenerate_policies'],
                     **candidate['parameters'])
    return CountBaselineAdapter(core,
                                metadata_encoder=MetadataEncoder(numeric=config['numeric_metadata']),
                                fallback=config['fallback'])


def train_panel(raw, config, ids, fold):
    native, _ = hide_targets(raw, np.zeros_like(raw.original_observed), mask_id=f'fold{fold}_train_native',
                             approved_metadata=config['numeric_metadata'])
    train = subset(native, ids)
    return train, FitScope('inductive_group_by_location', train.row_ids, f'fold{fold}', train.mask_id)


def query_panel(raw, config, output, ids, scenario, seed):
    hidden = get_mask(output, scenario, seed)
    # Only query-region masks become held-out targets for this fit. All donors
    # are already frozen from training locations, never from these queries.
    panel, _ = hide_targets(raw, hidden, mask_id=mask_hash(hidden), approved_metadata=config['numeric_metadata'])
    return subset(panel, ids), hidden


def evaluate(adapter, raw, config, output, ids, scenario, seed, job):
    job = Path(job)
    query, hidden = query_panel(raw, config, output, ids, scenario, seed)
    identity = _evaluation_identity(output, job, ids, scenario, seed, hidden)
    cached = _read_completed_evaluation(job, ids, scenario, seed, hidden, identity, raw.counts.shape[1])
    if cached is not None:
        return cached
    attempt = new_attempt(job)
    start = time.perf_counter()
    result = adapter.predict_result(query)
    metrics = score(raw, ids, result.counts, hidden)
    if any(metrics['validity'].values()):
        raise AssertionError(metrics['validity'])
    prediction_file = attempt / 'predictions.npz'
    with prediction_file.open('xb') as stream:
        np.savez_compressed(stream, row_indices=np.asarray(ids, dtype=int), counts=result.counts)
    record = {'status': 'ok', 'scenario': scenario, 'mask_seed': seed, 'mask_hash': mask_hash(hidden),
              'rows': len(ids), 'seconds': time.perf_counter() - start,
              'metrics': metrics, 'diagnostics': compact_diagnostics(result),
              'job_identity': identity,
              'predictions_artifact': prediction_file.relative_to(job).as_posix(),
              'predictions_hash': file_hash(prediction_file)}
    write_json(attempt / 'result.json', record)
    publish_json(job / 'result.json', record)
    return record


def run_fold(root, output, fold, stage='all'):
    output = Path(output)
    manifest, raw, _, _ = load_run(root, output)
    artifacts = RunArtifacts(output, manifest, raw)
    config = manifest['config']
    if (output / f'fold{fold}' / 'completion.json').exists():
        artifacts.fold(fold)
        return
    split = json.loads((output / 'folds.json').read_text(encoding='utf-8'))['folds'][fold]
    train, scope = train_panel(raw, config, split['train_rows'], fold)
    assert not set(train.entity_ids) & set(np.asarray(raw.entity_ids)[split['test_rows']])
    assert not set(train.entity_ids) & set(np.asarray(raw.entity_ids)[split['validation_rows']])
    base = output / f'fold{fold}'; base.mkdir(exist_ok=True)
    selection_path = base / 'selection.json'
    if stage in ('all', 'validation') and not selection_path.exists():
        selections, validation_table = {}, {}
        for family, candidates in config['candidates'].items():
            table = []
            for index, candidate in enumerate(candidates):
                directory = base / 'validation' / family / f'candidate{index}'
                summary = directory / 'candidate.json'
                if summary.exists():
                    record, _ = artifacts.candidate(fold, family, index)
                else:
                    start = time.perf_counter()
                    try:
                        adapter = make_adapter(config, candidate).fit(train, scope)
                        evaluations = [evaluate(adapter, raw, config, output, split['validation_rows'],
                                                'whole_feature', seed, directory / f'seed{seed}')
                                       for seed in config['mask_seeds']]
                        record = {'status': 'ok', 'candidate': candidate, 'index': index,
                                  'ratio_mae': float(np.mean([r['metrics']['overall']['ratio_mae'] for r in evaluations])),
                                  'seconds': time.perf_counter() - start}
                    except BaselineError as exc:
                        directory.mkdir(parents=True, exist_ok=True)
                        record = {'status': failure_status(exc), 'candidate': candidate, 'index': index,
                                  **structured_error(exc, stage='validation')}
                    publish_json(summary, record)
                table.append(record)
                print(json.dumps({'fold': fold, 'stage': 'validation', 'family': family,
                                  'index': index, 'status': record['status'], 'ratio_mae': record.get('ratio_mae')}), flush=True)
            valid = [x for x in table if x['status'] == 'ok']
            if not valid:
                raise BaselineError(f'No applicable validation candidate for {family}: {table}')
            selections[family] = min(valid, key=lambda x: (x['ratio_mae'], x['index']))
            validation_table[family] = table
        inventory = artifacts.validation_inventory(fold, validation_table)
        publish_json(selection_path, {'selected': selections, 'validation': validation_table,
                                      **artifacts.identity(fold), 'artifacts': inventory})
    selection = artifacts.selection(fold)['selected']
    if stage == 'validation':
        return
    if stage in ('all', 'test'):
        for family, selected in selection.items():
            directory = base / 'test' / family
            cached = True
            for scenario in config['scenarios']:
                for seed in config['mask_seeds']:
                    job = directory / f'{scenario}_{seed}'
                    hidden = get_mask(output, scenario, seed)
                    identity = _evaluation_identity(output, job, split['test_rows'], scenario, seed, hidden)
                    if _read_completed_evaluation(job, split['test_rows'], scenario, seed, hidden,
                                                  identity, raw.counts.shape[1]) is None:
                        cached = False
            if cached:
                continue
            adapter = make_adapter(config, selected['candidate']).fit(train, scope)
            for scenario in config['scenarios']:
                for seed in config['mask_seeds']:
                    record = evaluate(adapter, raw, config, output, split['test_rows'], scenario, seed,
                                      directory / f'{scenario}_{seed}')
                    print(json.dumps({'fold': fold, 'stage': 'test', 'family': family, 'scenario': scenario,
                                      'seed': seed, 'ratio_mae': record['metrics']['overall']['ratio_mae']}), flush=True)
    if stage in ('all', 'supplementary'):
        for method in config['supplementary_methods']:
            directory = base / 'supplementary' / method
            if artifacts.supplementary(fold, method, selection) is not None:
                continue
            attempt = new_attempt(directory)
            command = [sys.executable, '-B', '-m', __spec__.name, 'supplementary_one',
                       '--root', str(Path(root).resolve()), '--output', str(output.resolve()),
                       '--fold', str(fold), '--method', method]
            limit = config['supplementary_wall_seconds_per_fold_method']
            execution_log = attempt / 'execution.txt'
            failure = None
            with execution_log.open('x', encoding='utf-8') as log:
                try:
                    process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=limit)
                    if not (directory / 'finished.json').exists():
                        failure = {'status': 'failed_runtime',
                                   'error': f'Child exited {process.returncode} without a completion marker'}
                except subprocess.TimeoutExpired:
                    failure = {'status': 'resource_limit',
                               'error': f'Exceeded preregistered {limit}s wall budget per supplementary fold/method',
                               'partial_outputs_not_scored': True}
            if failure is not None and not (directory / 'finished.json').exists():
                failure.update({**artifacts.identity(fold, method),
                    'candidate': supplementary_candidate(config, selection, method),
                    'limit_seconds': limit, 'execution_log': execution_log.relative_to(output).as_posix(),
                    'execution_log_hash': file_hash(execution_log)})
                failure['artifacts'] = artifacts.supplementary_inventory(fold, method, failure)
                publish_json(directory / 'finished.json', failure)
            record = artifacts.supplementary(fold, method, selection)
            print(json.dumps({'fold': fold, 'stage': 'supplementary', 'method': method,
                              'status': record['status']}), flush=True)


def supplementary_one(root, output, fold, method):
    output = Path(output)
    manifest, raw, _, _ = load_run(root, output)
    artifacts = RunArtifacts(output, manifest, raw)
    config = manifest['config']
    if method not in config['supplementary_methods']:
        raise ValueError('Method is not preregistered')
    if (output / f'fold{fold}' / 'completion.json').exists():
        artifacts.fold(fold)
        return
    split = json.loads((output / 'folds.json').read_text(encoding='utf-8'))['folds'][fold]
    train, scope = train_panel(raw, config, split['train_rows'], fold)
    selection = artifacts.selection(fold)['selected']
    directory = output / f'fold{fold}' / 'supplementary' / method
    if artifacts.supplementary(fold, method, selection) is not None:
        return
    candidate = supplementary_candidate(config, selection, method)
    start = time.perf_counter()
    stage, scenario, seed = 'fit', None, None
    try:
        adapter = make_adapter(config, candidate).fit(train, scope)
        stage = 'prediction'
        for scenario in config['supplementary_scenarios']:
            for seed in config['mask_seeds']:
                evaluate(adapter, raw, config, output, split['test_rows'], scenario, seed,
                         directory / f'{scenario}_{seed}')
        record = {'status': 'ok', 'candidate': candidate, 'seconds': time.perf_counter() - start}
    except BaselineError as exc:
        exc.add_context(stage=stage, scenario=scenario, seed=seed, method=method, fold=fold)
        directory.mkdir(parents=True, exist_ok=True)
        record = {'status': failure_status(exc), 'candidate': candidate,
                  **structured_error(exc, stage=stage), 'seconds': time.perf_counter() - start}
    record.update(artifacts.identity(fold, method))
    record['artifacts'] = artifacts.supplementary_inventory(fold, method, record)
    publish_json(directory / 'finished.json', record)


def final_refit(root, output):
    output = Path(output)
    manifest, raw, fields, rows = load_run(root, output)
    artifacts = RunArtifacts(output, manifest, raw)
    config = manifest['config']
    selections = [artifacts.selection(fold) for fold in range(config['folds'])]
    train, scope = train_panel(raw, config, list(range(len(raw.row_ids))), 'final')
    summaries = []
    for family in config['candidates']:
        directory = output / 'final' / family
        candidate = final_candidate(config, selections, family)
        record = artifacts.final(family, candidate)
        if record is not None:
            summaries.append(record)
            continue
        start = time.perf_counter()
        adapter = make_adapter(config, candidate).fit(train, scope)
        result = adapter.predict_result(train)
        attempt = new_attempt(directory)
        completed_csv = attempt / 'completed.csv'
        write_completed_csv(completed_csv, raw, result, fields, rows)
        compact = compact_diagnostics(result)
        record = {'family': family, 'candidate': candidate, 'selection': 'mean_validation_score_across_folds',
                  'rows': len(raw.row_ids), 'native_missing_filled': int((~raw.original_observed).sum()),
                  'seconds': time.perf_counter() - start, 'dataset_hash': manifest['dataset_hash'],
                  'csv_artifact': completed_csv.relative_to(directory).as_posix(),
                  'csv_hash': file_hash(completed_csv), 'native_missing_accuracy': None,
                  'diagnostics': compact, 'resolved_core_config': core_configuration(adapter.core_),
                  **artifacts.identity(), 'artifacts': {}}
        artifacts.add(record['artifacts'], completed_csv)
        write_json(attempt / 'manifest.json', record)
        publish_json(directory / 'manifest.json', record)
        summaries.append(record)
        print(json.dumps({'stage': 'final_refit', 'family': family, 'seconds': record['seconds']}), flush=True)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'validation', 'test', 'supplementary', 'supplementary_one', 'all', 'final'])
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--fold', type=int)
    parser.add_argument('--method')
    args = parser.parse_args()
    if args.stage == 'prepare':
        print(json.dumps(prepare(args.root, args.config, args.output), indent=2))
    elif args.stage == 'final':
        final_refit(args.root, args.output)
    elif args.stage == 'supplementary_one':
        supplementary_one(args.root, args.output, args.fold, args.method)
    else:
        if args.fold is None:
            parser.error('--fold required')
        run_fold(args.root, args.output, args.fold, args.stage)


if __name__ == '__main__':
    main()
