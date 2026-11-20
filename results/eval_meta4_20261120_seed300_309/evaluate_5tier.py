"""Score the eval_meta4 runs with the evaluation module (tier 0, 1, 2). No metric is reimplemented.

Run from the repository root:  PYTHONPATH=src .venv/Scripts/python.exe <this file>
Tier 3 (U1-U4) validates the diagnostics D1-D3 themselves on seeds 0-4 and is model-independent; it is
not rerun here (see the README of the evaluation module for its verdicts).
"""
import csv
import json
from pathlib import Path

import numpy as np

from longitudinal_impute.data import load_csv
from longitudinal_impute.evaluation import tier0, tier1, tier2
from longitudinal_impute.evaluation.masks import step5_cuts
from longitudinal_impute.evaluation.metrics import read_predictions
from longitudinal_impute.evaluation.report import ROOT

OUT = Path(__file__).resolve().parent
MASKS = ('whole_feature', 'random_cell', 'whole_feature_matched')
MODELS = ('hron', 'tsagris', 'micoda', 'same_day_mean', 'main', 'main_nometa')
PAIRS = [('main', b) for b in MODELS if b != 'main'] + [('main_nometa', 'same_day_mean')]
METRICS = ('mae_prop', 'mse_prop', 'mae_count', 'series_macro_mae_prop', 'mae_prop_truth_zero',
           'mae_prop_truth_positive')
NATIVE_DIR = OUT / 'whole_feature_matched'          # the run with --pass both holds the native fills


def write(name, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / 'report' / name).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def f(x, d=4):
    return 'NA' if x is None else f'{x:.{d}f}'


def main():
    (OUT / 'report').mkdir(exist_ok=True)
    md = ['# Đánh giá 5 tầng: MainModel (metadata 4 trường) vs baseline — exploratory', '',
          'Seed 300–309, tỷ lệ che 5%, 6 model: ' + ', '.join(f'`{m}`' for m in MODELS) + '. '
          'Điểm tính bằng `longitudinal_impute.evaluation` (tier0, tier1, tier2), không viết lại metric. '
          'Seed đã được nhìn trong lúc phát triển, nên đây là exploratory, không phải confirmatory.', '']
    cuts = step5_cuts(ROOT / 'configs/evaluation/step5_tier2/tier2_results.md')
    status_all, summary_rows, pair_out, strata_out = [], [], [], []
    tier1_means = {}
    for mask in MASKS:
        folder = OUT / mask
        status, scores, _, _ = tier1.load_run(folder, {}, mask)
        status = [s for s in status if s['pass'] == 'masked' and s['model'] in MODELS]
        status_all += status
        runs = {m: v for m, v in tier1.table_runs(status, scores).items() if m in MODELS}
        seeds = sorted({s for v in runs.values() for s in v})
        common = [s for s in seeds if all(runs.get(m, {}).get(s, {}).get('eligible') for m in MODELS)]
        tier1_means[mask] = {}
        for m in MODELS:
            row = {'mask': mask, 'model': m, 'seeds_ok': sum(1 for s in seeds if runs.get(m, {}).get(s, {}).get('eligible')),
                   'common_seeds': len(common)}
            for k in METRICS:
                vals = [runs[m][s][k] for s in common if runs[m][s].get(k) is not None]
                row[f'{k}_mean'] = float(np.mean(vals)) if vals else None
                row[f'{k}_sd'] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
            tier1_means[mask][m] = row
            summary_rows.append(row)
        for a, b in PAIRS:
            for metric in ('mae_prop', 'mse_prop'):
                p = tier1.pair(runs, a, b, common, metric)
                p['sign_p'] = tier1.sign_test(p['wins'], p['n'] - p['wins'] - p['ties'])
                pair_out.append({'mask': mask, 'a': a, 'b': b, 'metric': metric, 'n': p['n'], 'wins': p['wins'],
                                 'ties': p['ties'], 'mean_diff': p['mean_diff'], 'sd_diff': p['sd_diff'],
                                 'sign_p': p['sign_p']})
        rows = [r for r in read_predictions(folder / f'predictions_{mask}.csv.gz') if r['model'] in MODELS]
        for (m, s, kind, label), v in tier1.stratum_scores(rows, cuts).items():
            if str(s) in common:
                strata_out.append({'mask': mask, 'model': m, 'seed': s, 'kind': kind, 'stratum': label, 'mae_prop': v})
        del rows

    # ---- tier 0 ----
    rel = tier0.reliability(status_all)
    md += ['## Tầng 0 — hợp lệ', '', '| Model | Mask | Lên lịch | Completed | Partial | Failed | Missing |',
           '|---|---|---:|---:|---:|---:|---:|']
    md += [f"| {r['model']} | {r['mask']} | {r['scheduled']} | {r['completed']} | {r['partial']} | {r['failed']} | "
           f"{r['missing']} |" for r in rel]
    # ---- tier 1 ----
    md += ['', '## Tầng 1 — che giả có đáp án', '',
           'MAE / MSE của tỷ lệ c/N trên ô model quyết định (bỏ ô bị ép 0), mean ± sd qua seed chung. Thấp hơn là tốt hơn.', '',
           '| Model | ' + ' | '.join(f'MAE {m}' for m in MASKS) + ' | ' + ' | '.join(f'MSE {m}' for m in MASKS) + ' |',
           '|---|' + '---:|' * 6]
    for m in MODELS:
        cells = [f"{f(tier1_means[k][m]['mae_prop_mean'])} ± {f(tier1_means[k][m]['mae_prop_sd'])}" for k in MASKS]
        cells += [f"{f(tier1_means[k][m]['mse_prop_mean'])} ± {f(tier1_means[k][m]['mse_prop_sd'])}" for k in MASKS]
        md.append(f'| `{m}` | ' + ' | '.join(cells) + ' |')
    md += ['', '**So cặp** (Δ = A − B trung bình qua seed; thắng = seed A thấp hơn; sign test hai phía, mô tả biến thiên mask):', '',
           '| A | B | Metric | ' + ' | '.join(MASKS) + ' |', '|---|---|---|' + '---|' * 3]
    for a, b in PAIRS:
        for metric in ('mae_prop', 'mse_prop'):
            cols = []
            for mask in MASKS:
                p = next(x for x in pair_out if (x['mask'], x['a'], x['b'], x['metric']) == (mask, a, b, metric))
                cols.append(f"{p['mean_diff']:+.4f}, {p['wins']}/{p['n']}, p={p['sign_p']:.3g}" if p['n'] else 'NA')
            md.append(f'| `{a}` | `{b}` | {metric} | ' + ' | '.join(cols) + ' |')
    md += ['', '**MAE theo châu lục (whole_feature, mean qua seed)**:', '']
    groups = sorted({r['stratum'] for r in strata_out if r['kind'] == 'continent'})
    md += ['| Model | ' + ' | '.join(groups) + ' |', '|---|' + '---:|' * len(groups)]
    for m in MODELS:
        vals = []
        for g in groups:
            v = [r['mae_prop'] for r in strata_out if (r['mask'], r['model'], r['kind'], r['stratum']) ==
                 ('whole_feature', m, 'continent', g) and r['mae_prop'] is not None]
            vals.append(f(float(np.mean(v))) if v else 'NA')
        md.append(f'| `{m}` | ' + ' | '.join(vals) + ' |')

    # ---- tier 2 ----
    data = load_csv(tier2.DATA, tier2.SCHEMA)
    facts, cells = tier2.row_facts(data), tier2.native_cells(data)
    support = tier2.local_support(facts, cells['eval'], cells['ref'])
    t2 = {}
    for m in MODELS:
        path = NATIVE_DIR / f'{m}_completed.csv'
        if path.exists():
            values = tier2.read_completed(path, data)
            t2[m] = tier2.diagnose(values, facts, cells['eval'], cells['ref'], support, list(data.targets))
    md += ['', '## Tầng 2 — ô thiếu thật (mô tả, không phải độ chính xác)', '',
           'D1 = chênh tỷ lệ ô bằng 0 so với reference; D2 = Wasserstein W1 trên tỷ lệ c/N phần dương; '
           'D3 = ô điền > 0 ngoài khoảng thời gian có dữ liệu của biến thể. R-local là reference chính.', '',
           *tier2.overview_lines(t2)]
    write('tier2_summary.csv', [{'model': m, **{f'{k}_{w}': r['summary'][k][w] for k in r['summary']
                                                for w in ('equal', 'weighted')}, 'd3_cells': r['d3_cells'],
                                 'd3_mass': r['d3_mass']} for m, r in t2.items()])
    md += ['', '## Tầng 3 / Tầng 4', '',
           '- Tầng 3 (U1–U4) kiểm định chính các chẩn đoán D1–D3 trên seed 0–4, không phụ thuộc model; không chạy lại.',
           '- Tầng 4 (downstream) module chưa hiện thực.', '']
    write('tier0_status.csv', status_all)
    write('tier1_summary.csv', summary_rows)
    write('tier1_pairs.csv', pair_out)
    write('tier1_strata.csv', strata_out)
    (OUT / 'report' / 'REPORT_5TIER_VI.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
