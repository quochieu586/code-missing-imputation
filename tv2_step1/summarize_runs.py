"""Step 1 multi-seed tables from the run outputs in tv2_step1/runs/ (not tracked).

    PYTHONPATH=nextgen/src python tv2_step1/summarize_runs.py [--runs DIR] [--out DIR]

Reads summary.json of <runs>/main_<mask> (7 models) and <runs>/micoda_other_<mask> (MICoDa
other = true, reported as micoda_other). Only metric groups already computed by
evaluate.masked_errors are used. The same_day_mean fallback levels on the hidden cells
are recomputed from the saved masks (same model, same parameters as the run).

F2 (PR #4 review): before the two runs of a mask are merged, validate_pair() checks that
they share the data hash, schema, mask kind, rate, seeds and group map, that every saved
mask is identical, that the main run has MICoDa with other = false and the second run
MICoDa alone with other = true. Any mismatch raises ValueError and no output file is
written. Outputs (only after every check passed): results_multiseed.md,
results_multiseed.csv and results_manifest.json (hashes of data, schema, masks, models
and group map per run) in --out.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
MASKS = ('whole_feature', 'random_cell')
SEEDS = ('0', '1', '2', '3', '4')
PAIRS = (('same_day_mean', 'hron'), ('hron', 'tsagris'), ('micoda_other', 'mean_fill'), ('micoda', 'mean_fill'))


def digest(value):
    """sha256 of bytes, of a numpy array (dtype, shape, data) or of a JSON-able object."""
    if isinstance(value, np.ndarray):
        value = f'{value.dtype}{value.shape}'.encode() + np.ascontiguousarray(value).tobytes()
    elif not isinstance(value, bytes):
        value = json.dumps(value, sort_keys=True).encode()
    return hashlib.sha256(value).hexdigest()


def run_record(folder, mask):
    """Manifest entry of one run directory, read from its config.json and saved masks."""
    config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    masks = {}
    for seed in SEEDS:
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


def validate_pair(root, mask):
    """F2: the main and micoda_other runs of one mask may be paired only if these all agree."""
    main, main_masks = run_record(root / f'main_{mask}', mask)
    other, other_masks = run_record(root / f'micoda_other_{mask}', mask)
    problems = []
    for key in ('data_sha256', 'schema_sha256', 'mask', 'rate', 'seeds', 'group_map'):
        if main[key] != other[key]:
            problems.append(f'{key}: main {main[key]!r} != micoda_other {other[key]!r}')
    if main['mask'] != mask or main['seeds'] != list(SEEDS):
        problems.append(f'expected mask {mask!r} with seeds {list(SEEDS)}, got {main["mask"]!r} {main["seeds"]}')
    for seed in SEEDS:
        if not np.array_equal(main_masks[seed], other_masks[seed]):
            problems.append(f'seed {seed}: saved masks differ')
    if 'micoda' not in main['models'] or main['models']['micoda'].get('other', False) is not False:
        problems.append(f'main run must contain MICoDa with other = false, got {main["models"].get("micoda")}')
    if set(other['models']) != {'micoda'} or other['models']['micoda'].get('other') is not True:
        problems.append(f'micoda_other run must contain only MICoDa with other = true, got {other["models"]}')
    if problems:
        raise ValueError(f'refusing to pair main_{mask} with micoda_other_{mask}:\n  ' + '\n  '.join(problems))
    return [main, other]


def load(root, mask):
    """{model: {seed: result}} plus the native results, with micoda_other merged in."""
    runs, native = {}, {}
    for prefix, rename in (('main', {}), ('micoda_other', {'micoda': 'micoda_other'})):
        summary = json.loads((root / f'{prefix}_{mask}' / 'summary.json').read_text())
        for name, result in summary['native'].items():
            native[rename.get(name, name)] = result
        for seed in SEEDS:
            for name, result in summary['masked'][seed]['models'].items():
                runs.setdefault(rename.get(name, name), {})[seed] = result
    return runs, native


def metric(result, group, key):
    errors = result.get('errors')
    return None if result['status'] != 'ok' or errors is None else errors[group][key]


def fmt(values, digits=4):
    values = np.array([v for v in values if v is not None], float)
    if not len(values):
        return '-'
    sd = values.std(ddof=1) if len(values) > 1 else 0.
    return f'{values.mean():.{digits}f} ± {sd:.{digits}f}'


def order_key(runs, model, key='mae_proportion'):
    """F4: complete models by mean error (0.0 first), then models with a missing/failed seed."""
    values = [metric(runs[model][s], 'model', key) for s in SEEDS]
    present = [v for v in values if v is not None and np.isfinite(v)]
    incomplete = len(present) < len(values)
    return (incomplete, float(np.mean(present)) if present else np.inf, model)


def main_table(runs, out):
    order = sorted(runs, key=lambda m: order_key(runs, m))
    out.append('| Model | ok/5 | MAE tỷ lệ | MSE tỷ lệ | MAE count | Số ô model (seed 0–4) |')
    out.append('|---|---:|---:|---:|---:|---|')
    rows = []
    for m in order:
        r = [runs[m][s] for s in SEEDS]
        ok = sum(x['status'] == 'ok' for x in r)
        cells = '/'.join(str(metric(x, 'model', 'cells')) for x in r)
        out.append(f"| {m} | {ok} | {fmt([metric(x, 'model', 'mae_proportion') for x in r])} | "
                   f"{fmt([metric(x, 'model', 'mse_proportion') for x in r])} | "
                   f"{fmt([metric(x, 'model', 'mae_count') for x in r], 1)} | {cells} |")
        for s, x in zip(SEEDS, r):
            rows.append({'model': m, 'seed': s, 'status': x['status'],
                         **{f'{k}_{g}': metric(x, g, k) for g in ('model', 'truth_zero', 'truth_positive',
                                                                    'model_truth_zero')
                            for k in ('cells', 'mae_proportion', 'mse_proportion', 'mae_count')}})
    return rows


def pair_table(runs, out):
    out.append('| Cặp (A vs B) | MAE tỷ lệ: A thắng | hiệu TB (A − B) | MSE tỷ lệ: A thắng | hiệu TB (A − B) |')
    out.append('|---|---:|---:|---:|---:|')
    for a, b in PAIRS:
        cells = []
        for key in ('mae_proportion', 'mse_proportion'):
            diff = [metric(runs[a][s], 'model', key) - metric(runs[b][s], 'model', key)
                    for s in SEEDS if metric(runs[a][s], 'model', key) is not None
                    and metric(runs[b][s], 'model', key) is not None]
            cells += [f'{sum(d < 0 for d in diff)}/{len(diff)} seed', f'{np.mean(diff):+.4f}']
        out.append(f'| {a} vs {b} | ' + ' | '.join(cells) + ' |')


def truth_table(runs, out):
    out.append('| Model | MAE tỷ lệ, đáp án = 0 (truth_zero) | đáp án = 0, ô model (model_truth_zero) | '
               'đáp án > 0 (truth_positive) | Số ô 0 / ô model 0 / ô > 0 (seed 0) |')
    out.append('|---|---:|---:|---:|---|')
    for m in ('hron', 'tsagris', 'same_day_mean'):
        r = [runs[m][s] for s in SEEDS]
        out.append(f"| {m} | {fmt([metric(x, 'truth_zero', 'mae_proportion') for x in r])} | "
                   f"{fmt([metric(x, 'model_truth_zero', 'mae_proportion') for x in r])} | "
                   f"{fmt([metric(x, 'truth_positive', 'mae_proportion') for x in r])} | "
                   f"{metric(r[0], 'truth_zero', 'cells')} / {metric(r[0], 'model_truth_zero', 'cells')} / "
                   f"{metric(r[0], 'truth_positive', 'cells')} |")


def levels_table(root, mask, native, out):
    sys.path.insert(0, str(REPO / 'nextgen/src'))
    from longitudinal_impute import run
    from longitudinal_impute.data import load_csv
    from longitudinal_impute.evaluate import forced_zero
    config = json.loads((root / f'main_{mask}' / 'config.json').read_text())
    data = load_csv(config['data'], config['schema_path'])
    # F3: recompute only from the same data and the same group map the run used.
    if hashlib.sha256(Path(config['data']).read_bytes()).hexdigest() != config['data_sha256']:
        raise ValueError(f"main_{mask}: {config['data']} changed since the run (data_sha256 differs)")
    if 'group_map' not in config:
        raise ValueError(f'main_{mask}: config.json has no group_map provenance; rerun with the current run.py')
    recorded, current = config['group_map'], data.group_map
    if (recorded is None) != (current is None) or (recorded and (
            recorded['path'], recorded['sha256']) != (current['path'], current['sha256'])):
        raise ValueError(f'main_{mask}: group map differs from the one the run used: '
                         f'recorded {recorded}, current {current}')
    params = config['models']['same_day_mean']
    names = ['window', 'widened_1', 'widened_2', 'global']
    out.append('| Seed | Ô che (nhóm model): gốc / nới 1 / nới 2 / toàn cục | Mọi ô che: gốc / nới 1 / nới 2 / toàn cục |')
    out.append('|---|---|---|')
    for seed in SEEDS:
        hidden = np.load(root / f'main_{mask}' / f'mask_{mask}_seed{seed}.npy')
        model = run.MODELS['same_day_mean'](**params)
        run.run_model(data, model, hidden)
        cells = []
        for group in (hidden & ~forced_zero(data, hidden), hidden):
            levels = model.levels_[group]
            cells.append(' / '.join(str(int((levels == k).sum())) for k in range(len(names)))
                         + f' (toàn cục {np.mean(levels == 3):.1%})')
        out.append(f'| {seed} | ' + ' | '.join(cells) + ' |')
    levels = native['same_day_mean']['info']['cells_by_level']
    total = sum(levels.values())
    out.append(f"\nÔ thiếu thật (lượt native): " + ', '.join(f'{k} {v}' for k, v in levels.items())
               + f"; toàn cục {levels['global'] / total:.1%} trên {total} ô.")


def summarize(root, out_dir, levels=True):
    """Validate every pair first; write the three outputs only when all checks passed."""
    manifest = {mask: validate_pair(root, mask) for mask in MASKS}
    out, rows = ['# Step 1: kết quả nhiều seed (COVID đầy đủ, che 5%, seed 0–4)', '',
                 'Nhóm `model` = ô bị che mà phần dư của dòng > 0 (evaluate.forced_zero). '
                 'Sắp theo MAE tỷ lệ trung bình. mean ± sd (ddof=1) qua 5 seed: độ lệch chuẩn mẫu, '
                 'không phải khoảng tin cậy. Nguồn: results_manifest.json.', ''], []
    for mask in MASKS:
        runs, native = load(root, mask)
        out += [f'## Mask `{mask}`' + (' (chính)' if mask == 'whole_feature' else ' (tham khảo)'), '']
        statuses = {m: [runs[m][s]['status'] for s in SEEDS] for m in runs}
        bad = {m: s for m, s in statuses.items() if set(s) != {'ok'}}
        native_bad = {m: r['status'] for m, r in native.items() if r['status'] != 'ok'}
        out.append(f'Trạng thái: mọi lượt ok' if not bad and not native_bad
                   else f'Trạng thái KHÔNG ok: masked {bad}, native {native_bad}')
        out.append('')
        out += ['### Nhóm ô model', '']
        rows += [{'mask': mask, **r} for r in main_table(runs, out)]
        out += ['', '### Thắng/thua theo seed (ghép cặp, cùng mask; thắng = sai số nhỏ hơn)', '']
        pair_table(runs, out)
        out += ['', '### MAE tỷ lệ theo đáp án', '']
        truth_table(runs, out)
        if levels:
            out += ['', '### same_day_mean: mức cửa sổ trên ô bị che', '']
            levels_table(root, mask, native, out)
        out.append('')
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'results_multiseed.md').write_text('\n'.join(out), encoding='utf-8')
    with (out_dir / 'results_multiseed.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / 'results_manifest.json').write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n',
                                                   encoding='utf-8')
    return out, rows, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', type=Path, default=HERE / 'runs')
    parser.add_argument('--out', type=Path, default=HERE)
    args = parser.parse_args(argv)
    summarize(args.runs, args.out)


if __name__ == '__main__':
    main()
