"""Step 1 multi-seed tables from the run outputs in tv2_step1/runs/ (not tracked).

    PYTHONPATH=nextgen/src python tv2_step1/summarize_runs.py > tv2_step1/results_multiseed.md

Reads summary.json of runs/main_<mask> (7 models) and runs/micoda_other_<mask> (MICoDa
other = true, reported as micoda_other). Only metric groups already computed by
evaluate.masked_errors are used. The same_day_mean fallback levels on the hidden cells
are recomputed from the saved masks (same model, same parameters as the run).
"""
import csv
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
MASKS = ('whole_feature', 'random_cell')
SEEDS = ('0', '1', '2', '3', '4')
PAIRS = (('same_day_mean', 'hron'), ('hron', 'tsagris'), ('micoda_other', 'mean_fill'), ('micoda', 'mean_fill'))


def load(mask):
    """{model: {seed: result}} plus the native results, with micoda_other merged in."""
    runs, native = {}, {}
    for prefix, rename in (('main', {}), ('micoda_other', {'micoda': 'micoda_other'})):
        summary = json.loads((HERE / 'runs' / f'{prefix}_{mask}' / 'summary.json').read_text())
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


def main_table(runs, out):
    order = sorted(runs, key=lambda m: np.mean([metric(runs[m][s], 'model', 'mae_proportion') or np.inf
                                                for s in SEEDS]))
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


def levels_table(mask, native, out):
    sys.path.insert(0, str(REPO / 'nextgen/src'))
    from longitudinal_impute import run
    from longitudinal_impute.data import load_csv
    from longitudinal_impute.evaluate import forced_zero
    config = json.loads((HERE / 'runs' / f'main_{mask}' / 'config.json').read_text())
    data = load_csv(config['data'], config['schema_path'])
    params = config['models']['same_day_mean']
    names = ['window', 'widened_1', 'widened_2', 'global']
    out.append('| Seed | Ô che (nhóm model): gốc / nới 1 / nới 2 / toàn cục | Mọi ô che: gốc / nới 1 / nới 2 / toàn cục |')
    out.append('|---|---|---|')
    for seed in SEEDS:
        hidden = np.load(HERE / 'runs' / f'main_{mask}' / f'mask_{mask}_seed{seed}.npy')
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


def main():
    out, rows = ['# Step 1: kết quả nhiều seed (COVID đầy đủ, che 5%, seed 0–4)', '',
                 'Nhóm `model` = ô bị che mà phần dư của dòng > 0 (evaluate.forced_zero). '
                 'Sắp theo MAE tỷ lệ trung bình. mean ± sd (ddof=1) qua 5 seed.', ''], []
    for mask in MASKS:
        runs, native = load(mask)
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
        out += ['', '### same_day_mean: mức cửa sổ trên ô bị che', '']
        levels_table(mask, native, out)
        out.append('')
    print('\n'.join(out))
    with (HERE / 'results_multiseed.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
