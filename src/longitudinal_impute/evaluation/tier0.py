"""Tier 0: run status, reliability, pairing of run directories, failure policy.

A run counts only when completed with coverage 1. Every scheduled run stays in the status table,
failed or missing ones included (F07); nothing is dropped to make a table look better.
validate_pair (F2) refuses to merge the main and micoda_other run directories of a mask unless
data, schema, mask kind, rate, seeds, group map and every saved mask agree.
"""
import json
from pathlib import Path

import numpy as np

from .report import digest, na
from .report import ROOT


STATUS = {'ok': 'completed', 'partial': 'partial', 'failed': 'failed'}


def status_rows(run_dir, config, summary, rename):
    """One row per scheduled run of one run directory: native per model, then model x seed."""
    rows = []
    cells = summary.get('native_cells', {})
    for name in config['models']:
        result = summary['native'].get(name)
        rows.append(status_row(run_dir, rename.get(name, name), 'native', 'none', None, result,
                               {'hidden_cells': 0, 'forced_zero_cells': cells.get('missing_forced_zero_cells'),
                                'model_cells': cells.get('missing_model_cells')}))
    for seed in (config['seeds'] or []):
        masked = summary['masked'].get(str(seed), {'mask': {}, 'models': {}})
        stats = masked['mask']
        for name in config['models']:
            rows.append(status_row(run_dir, rename.get(name, name), 'masked', config['mask'], str(seed),
                                   masked['models'].get(name), {k: stats.get(k) for k in (
                                       'hidden_cells', 'forced_zero_cells', 'model_cells', 'complete_rows_after')}))
    return rows


def status_row(run_dir, model, run_pass, mask, seed, result, cells):
    base = {'run_dir': run_dir, 'model': model, 'pass': run_pass, 'mask': mask, 'seed': seed}
    if result is None:
        return {**base, 'status': 'missing', 'coverage': None, 'error_code': 'missing_result',
                'error_message': 'scheduled run has no entry in summary.json', **dict.fromkeys(
                    ('converged', 'termination_reason', 'iterations')), **cells}
    info = result.get('info') or {}
    status = STATUS.get(result['status'], result['status'])
    coverage = result.get('coverage')
    if coverage is None and run_pass == 'native':
        if status == 'completed':
            coverage = 1.
        elif info.get('missing_cells'):
            coverage = info.get('resolved_cells', 0) / info['missing_cells']
    return {**base, 'status': status, 'coverage': coverage, 'error_code': result.get('error_code', ''),
            'error_message': result.get('error', ''), 'converged': info.get('converged'),
            'termination_reason': info.get('termination_reason'), 'iterations': info.get('iterations'), **cells}


def reliability(status):
    """Per model, pass and mask: every scheduled run counted, failed seeds listed with their code."""
    table = {}
    for r in status:
        entry = table.setdefault((r['model'], r['pass'], r['mask']), {
            'model': r['model'], 'pass': r['pass'], 'mask': r['mask'], 'scheduled': 0, 'completed': 0,
            'partial': 0, 'failed': 0, 'missing': 0, 'not_ok': []})
        entry['scheduled'] += 1
        entry[r['status']] += 1
        if r['status'] != 'completed' or r['coverage'] != 1.:
            entry['not_ok'].append(f"{r['seed'] or r['run_dir']}: {r['status']} ({r['error_code']})")
    return list(table.values())


def status_markdown(status, rel):
    out = ['| Model | Lượt | Mask | Lên lịch | completed | partial | failed | missing | Run không ok |',
           '|---|---|---|---:|---:|---:|---:|---:|---|']
    for e in rel:
        out.append(f"| {e['model']} | {e['pass']} | {e['mask']} | {e['scheduled']} | {e['completed']} | "
                   f"{e['partial']} | {e['failed']} | {e['missing']} | {'; '.join(e['not_ok']) or '-'} |")
    return out


def seed_markdown(status, seed):
    rows = [r for r in status if r['seed'] == seed]
    if not rows:
        return []
    r0 = rows[0]
    out = [f"Seed {seed} ({r0['mask']}): {r0['hidden_cells']} ô che, {r0['forced_zero_cells']} ép về 0, "
           f"{r0['model_cells']} ô model; {r0['complete_rows_after']} dòng đầy đủ sau khi che.", '',
           '| Model | status | coverage | mã lỗi | thông điệp | hội tụ (MICoDa) |', '|---|---|---:|---|---|---|']
    for r in rows:
        message = (r['error_message'] or '-').replace('|', '/')[:110]
        out.append(f"| {r['model']} | {r['status']} | {na(r['coverage'], 3)} | {r['error_code'] or '-'} | "
                   f"{message} | {'-' if r['converged'] is None else r['converged']} |")
    return out



SEEDS = ('0', '1', '2', '3', '4')     # default seeds of validate_pair


def micoda_other(config, table='micoda'):
    """The `other` flag MICoDa ran with in a run directory's config.json.

    An explicit key wins. Without it, the default of the runner that wrote the config applies:
    runs written before PR #5 (no 'stage' key in config.json) used other = false (A17); the
    runner of PR #5 and later defaults to other = true (A18).
    """
    params = config['models'].get(table) or {}
    return params.get('other', 'stage' in config)


def run_record(folder, mask, seeds=SEEDS):
    """Manifest entry of one run directory, read from its config.json and saved masks."""
    config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    masks = {}
    for seed in seeds:
        path = folder / f'mask_{mask}_seed{seed}.npy'
        if not path.exists():
            raise ValueError(f'{folder.name}: missing saved mask {path.name}')
        masks[seed] = np.load(path)
    return {'dir': folder.name, 'git': config.get('git'), 'data': Path(config['data']).name,
            'data_sha256': config['data_sha256'], 'schema_sha256': digest(config['schema']),
            'mask': config['mask'], 'rate': config['mask_rate'],
            'seeds': [str(s) for s in config['seeds'] or []],
            'mask_sha256': {seed: digest(m) for seed, m in masks.items()},
            'models': config['models'], 'models_sha256': digest(config['models']),
            'group_map': config.get('group_map', 'absent')}, masks


def validate_pair(root, mask, seeds=SEEDS):
    """F2: the main and micoda_other runs of one mask may be paired only if these all agree."""
    seeds = tuple(str(s) for s in seeds)
    main, main_masks = run_record(root / f'main_{mask}', mask, seeds)
    other, other_masks = run_record(root / f'micoda_other_{mask}', mask, seeds)
    problems = []
    for key in ('data_sha256', 'schema_sha256', 'mask', 'rate', 'seeds', 'group_map'):
        if main[key] != other[key]:
            problems.append(f'{key}: main {main[key]!r} != micoda_other {other[key]!r}')
    if main['mask'] != mask or main['seeds'] != list(seeds):
        problems.append(f'expected mask {mask!r} with seeds {list(seeds)}, got {main["mask"]!r} {main["seeds"]}')
    for seed in seeds:
        if not np.array_equal(main_masks[seed], other_masks[seed]):
            problems.append(f'seed {seed}: saved masks differ')
    configs = {name: json.loads((root / f'{name}_{mask}' / 'config.json').read_text(encoding='utf-8'))
               for name in ('main', 'micoda_other')}
    if 'micoda' not in main['models'] or micoda_other(configs['main']) is not False:
        problems.append(f'main run must contain MICoDa with other = false, got {main["models"].get("micoda")}')
    if set(other['models']) != {'micoda'} or micoda_other(configs['micoda_other']) is not True:
        problems.append(f'micoda_other run must contain only MICoDa with other = true, got {other["models"]}')
    if problems:
        raise ValueError(f'refusing to pair main_{mask} with micoda_other_{mask}:\n  ' + '\n  '.join(problems))
    for record in (main, other):        # the committed manifest keeps repository-relative paths only
        if isinstance(record['group_map'], dict) and Path(record['group_map']['path']).is_relative_to(ROOT):
            record['group_map'] = {**record['group_map'],
                                   'path': Path(record['group_map']['path']).relative_to(ROOT).as_posix()}
    return [main, other]



def failed_seeds(runs, model, seeds):
    """Seeds where the model does not count (not completed with coverage 1, or no run)."""
    return [s for s in seeds if not runs.get(model, {}).get(s, {}).get('eligible')]
