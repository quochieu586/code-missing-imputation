"""Step 3: same_day_mean ablation tables (exploratory), built on tv2_step2/tier01.py.

    PYTHONPATH=nextgen/src python tv2_step3/ablation.py [--runs DIR] [--step2-runs DIR] [--out DIR]

Reads <runs>/ablation_<mask> (the six variants of configs/models_sameday_ablation.toml) and the
step 2 baselines from <step2-runs>/main_<mask>. No second scorer: status rows, scores from the
long files, the summary.json regression check and the pairing all come from tier01. Before any
table, and before anything is written:
1. check_same_masks (F2-style): data, schema, mask kind, rate, group map and every saved mask of
   seeds 0-4 must agree between the ablation run and the step 2 run;
2. check_full: sdm_full must equal step 2 same_day_mean byte for byte (native completed CSV,
   every hidden-cell row of the long file);
3. window_levels re-runs the windowed variants on the saved masks and checks their hidden-cell
   predictions against the long file before counting levels on the model cells.
Outputs in --out: ablation_status.csv, ablation_runs.csv, ablation_pairs.csv,
ablation_results.md, ablation_manifest.json; the auto:* blocks of REPORT_vi.md.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
_spec = importlib.util.spec_from_file_location('tier01', REPO / 'tv2_step2/tier01.py')
tier01 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tier01)            # also puts nextgen/src on sys.path
from longitudinal_impute import run  # noqa: E402
from longitudinal_impute.cells import read_predictions  # noqa: E402
from longitudinal_impute.data import load_csv  # noqa: E402
from longitudinal_impute.evaluate import forced_zero  # noqa: E402

MASKS = ('whole_feature', 'random_cell')
SEEDS = ('0', '1', '2', '3', '4')
VARIANTS = ('sdm_full', 'sdm_no_time', 'sdm_no_group', 'sdm_unweighted', 'sdm_rescale', 'sdm_none')
FACTORS = (('sdm_no_time', 'T thời gian'), ('sdm_no_group', 'G châu lục'), ('sdm_unweighted', 'W trọng số N'),
           ('sdm_rescale', 'R phần dư'))
BASELINES = ('hron', 'mean_fill')
EXTRA_PAIRS = (('sdm_none', 'mean_fill'), ('sdm_full', 'hron'))
WINDOWED = ('sdm_full', 'sdm_no_group', 'sdm_unweighted', 'sdm_rescale')     # use_time = true
LEVELS = ('window', 'widened_1', 'widened_2', 'global')


def relative(record):
    gm = record['group_map']
    if isinstance(gm, dict) and Path(gm['path']).is_relative_to(REPO):
        record['group_map'] = {**gm, 'path': Path(gm['path']).relative_to(REPO).as_posix()}
    return record


def check_same_masks(ablation, step2, mask):
    """F2-style: the ablation run may be paired with the step 2 run only if all of these agree."""
    a, a_masks = tier01.step1.run_record(ablation, mask, SEEDS)
    b, b_masks = tier01.step1.run_record(step2, mask, SEEDS)
    problems = [f'{key}: ablation {a[key]!r} != step 2 {b[key]!r}'
                for key in ('data_sha256', 'schema_sha256', 'mask', 'rate', 'group_map') if a[key] != b[key]]
    if a['mask'] != mask or a['seeds'] != list(SEEDS) or not set(SEEDS) <= set(b['seeds']):
        problems.append(f'expected mask {mask!r}, seeds {list(SEEDS)}; got {a["mask"]!r} {a["seeds"]} / {b["seeds"]}')
    problems += [f'seed {s}: saved masks differ' for s in SEEDS if not np.array_equal(a_masks[s], b_masks[s])]
    if set(a['models']) != set(VARIANTS):
        problems.append(f'ablation run must hold exactly {VARIANTS}, got {sorted(a["models"])}')
    if problems:
        raise ValueError(f'refusing to pair {ablation.name} with {step2.name}:\n  ' + '\n  '.join(problems))
    return relative(a), relative(b)


def hidden_rows(path, model, seeds=SEEDS):
    return [{k: v for k, v in r.items() if k != 'model'} for r in read_predictions(path)
            if r['model'] == model and str(r['seed']) in seeds]


def check_full(ablation, step2, mask):
    """sdm_full must reproduce step 2 same_day_mean byte for byte, or nothing is written."""
    native = (ablation / 'sdm_full_completed.csv').read_bytes() == (step2 / 'same_day_mean_completed.csv').read_bytes()
    new = hidden_rows(ablation / f'predictions_{mask}.csv.gz', 'sdm_full')
    old = hidden_rows(step2 / f'predictions_{mask}.csv.gz', 'same_day_mean')
    if not native or new != old or not new:
        raise ValueError(f'{mask}: sdm_full differs from step 2 same_day_mean '
                         f'(native CSV identical: {native}; hidden rows {len(new)} vs {len(old)})')
    return {'native_completed_csv_identical': True, 'hidden_cell_rows_identical': len(new)}


def none_vs_mean_fill(ablation, step2, mask):
    """Largest |pred_count| difference between sdm_none and step 2 mean_fill on the same hidden cells."""
    a = hidden_rows(ablation / f'predictions_{mask}.csv.gz', 'sdm_none')
    b = hidden_rows(step2 / f'predictions_{mask}.csv.gz', 'mean_fill')
    same = [(x['row_id'], x['target'], x['seed']) for x in a] == [(y['row_id'], y['target'], y['seed']) for y in b]
    if not same:
        raise ValueError(f'{mask}: sdm_none and mean_fill do not cover the same hidden cells')
    diff = np.array([abs(x['pred_count'] - y['pred_count']) for x, y in zip(a, b)])
    prop = np.array([abs(x['pred_prop'] - y['pred_prop']) for x, y in zip(a, b)])
    return {'cells': len(a), 'identical_cells': int((diff == 0).sum()), 'max_abs_count': float(diff.max()),
            'max_abs_prop': float(prop.max())}


def window_levels(ablation, mask):
    """Hidden model-cell counts per window level, from a re-run checked against the long file."""
    config = json.loads((ablation / 'config.json').read_text(encoding='utf-8'))
    if hashlib.sha256(Path(config['data']).read_bytes()).hexdigest() != config['data_sha256']:
        raise ValueError(f"{ablation.name}: {config['data']} changed since the run")
    data = load_csv(config['data'], config['schema_path'])
    if (config['group_map'] or {}).get('sha256') != (data.group_map or {}).get('sha256'):
        raise ValueError(f'{ablation.name}: group map differs from the one the run used')
    rows = read_predictions(ablation / f'predictions_{mask}.csv.gz')
    table = {}
    for name in WINDOWED:
        for seed in SEEDS:
            hidden = np.load(ablation / f'mask_{mask}_seed{seed}.npy')
            model = run.build(name, config['models'][name])
            completed = run.run_model(data, model, hidden)[0]
            stored = np.array([r['pred_count'] for r in rows if r['model'] == name and str(r['seed']) == seed])
            if not np.array_equal(completed[hidden], stored):
                raise ValueError(f'{name} seed {seed}: re-run differs from the long file')
            levels = model.levels_[hidden & ~forced_zero(data, hidden)]
            table[name, seed] = [int((levels == k).sum()) for k in range(len(LEVELS))]
    summary = json.loads((ablation / 'summary.json').read_text(encoding='utf-8'))
    native = {name: summary['native'][name]['info']['cells_by_level'] for name in ('sdm_full', 'sdm_no_group')}
    return table, native


# ---------- markdown ----------

def cell(p, digits=4):
    if p['mean_diff'] is None:
        return 'NA'
    tie = f", hòa {p['ties']}" if p['ties'] else ''
    sd = 'NA' if p['sd_diff'] is None else f"{p['sd_diff']:.2g}"
    return f"{p['mean_diff']:+.{digits}g} ± {sd} ({p['wins']}/{p['n']}{tie})"


def wide_pairs(runs, pairs, seeds):
    """A vs B on every metric: mean ± sd of A − B over the common seeds, (A wins / n)."""
    out = ['| A vs B | ' + ' | '.join(label for *_, label in tier01.METRIC_COLUMNS) + ' |',
           '|---|' + '---|' * len(tier01.METRIC_COLUMNS)]
    rows = []
    for a, b in pairs:
        cells = []
        for column, *_ in tier01.METRIC_COLUMNS:
            p = tier01.pair(runs, a, b, seeds, column)
            rows.append(p)
            cells.append(cell(p))
        out.append(f'| {a} vs {b} | ' + ' | '.join(cells) + ' |')
    return out, rows


def contributions(runs, seeds):
    """(variant − full) / (none − full) on the means over the common seeds; descriptive only."""
    _, used, _, _ = tier01.ranking(runs, VARIANTS, seeds)
    mean = {(m, c): tier01.mean_sd([runs[m][s][c] for s in used])[0] for m in VARIANTS
            for c, *_ in tier01.METRIC_COLUMNS}
    out = ['| Yếu tố tắt | ' + ' | '.join(label for *_, label in tier01.METRIC_COLUMNS) + ' |',
           '|---|' + '---:|' * len(tier01.METRIC_COLUMNS)]
    span = []
    for c, *_ in tier01.METRIC_COLUMNS:
        full, none = mean['sdm_full', c], mean['sdm_none', c]
        span.append('NA' if None in (full, none) else f'{none - full:+.4g}')
    out.append('| sdm_none − sdm_full (mẫu số) | ' + ' | '.join(span) + ' |')
    for variant, label in FACTORS:
        cells = []
        for c, *_ in tier01.METRIC_COLUMNS:
            full, none, value = mean['sdm_full', c], mean['sdm_none', c], mean[variant, c]
            if None in (full, none, value) or none == full:
                cells.append('NA')
            else:
                cells.append(f'{(value - full) / (none - full):+.0%} (Δ {value - full:+.3g})')
        out.append(f'| {label} ({variant}) | ' + ' | '.join(cells) + ' |')
    return out


def levels_markdown(table, native):
    out = ['| Biến thể | ' + ' | '.join(f'seed {s}' for s in SEEDS) + ' | Tổng: toàn cục / ô model |',
           '|---|' + '---|' * len(SEEDS) + '---:|']
    for name in WINDOWED:
        counts = [table[name, s] for s in SEEDS]
        total = np.sum(counts, axis=0)
        out.append(f'| {name} | ' + ' | '.join(' / '.join(map(str, c)) for c in counts)
                   + f' | {total[-1]} / {total.sum()} ({total[-1] / total.sum():.1%}) |')
    out += ['', 'Ô thiếu thật (lượt native), mức toàn cục:']
    for name, levels in native.items():
        total = sum(levels.values())
        out.append(f"- {name}: {levels['global']} / {total} ({levels['global'] / total:.1%}); "
                   + ', '.join(f'{k} {v}' for k, v in levels.items()))
    return out


# ---------- driver ----------

def summarize(root, step2_root, out_dir):
    manifest, status_all, run_rows, pair_rows, blocks = {}, [], [], [], {}
    md = ['# Step 3: ablation same_day_mean (COVID đầy đủ, che 5%, seed 0–4) — **exploratory**', '',
          'Sinh bởi `ablation.py` (chấm điểm bằng `tier01`); không sửa tay. Metric chính: MAE tỷ lệ, '
          'cell-weighted, nhóm ô model. mean ± sd (ddof=1) qua seed, không phải khoảng tin cậy.', '']
    for mask in MASKS:
        ablation, step2 = root / f'ablation_{mask}', step2_root / f'main_{mask}'
        records = check_same_masks(ablation, step2, mask)
        full = check_full(ablation, step2, mask)
        status, scores, record, (checked, worst) = tier01.load_run(ablation, {}, mask)
        base_status, base_scores, base_record, base_check = tier01.load_run(step2, {}, mask)
        base_status = [r for r in base_status if r['model'] in BASELINES]
        levels, native = window_levels(ablation, mask)
        none_mean = none_vs_mean_fill(ablation, step2, mask)
        runs = tier01.table_runs(status + base_status, {**scores, **base_scores})
        models = [*VARIANTS, *BASELINES]
        status_all += status
        run_rows += [{'mask': mask, 'model': m, 'seed': s, **runs[m][s]} for m in models for s in SEEDS]
        main, _ = tier01.main_markdown(runs, models, SEEDS)
        versus, rows = wide_pairs(runs, [(v, 'sdm_full') for v in VARIANTS[1:]] + list(EXTRA_PAIRS), SEEDS)
        pair_rows += [{'mask': mask, **{k: v for k, v in p.items() if k not in ('seeds_used', 'dropped')},
                       'seeds_used': ' '.join(p['seeds_used'])} for p in rows]
        primary = [tier01.pair(runs, v, 'sdm_full', SEEDS) for v in VARIANTS[1:]]
        primary += [tier01.pair(runs, a, b, SEEDS) for a, b in EXTRA_PAIRS]
        blocks[f'status_{mask}'] = tier01.status_markdown(status, tier01.reliability(status))
        blocks[f'versus_{mask}'] = tier01.pairs_markdown(primary, f'**{mask}**, biến thể vs sdm_full')
        blocks[f'wide_{mask}'] = ['Ô: Δ = A − B (mean ± sd), (A thắng / n seed). Thắng = sai số nhỏ hơn.', '',
                                  *versus]
        blocks[f'share_{mask}'] = contributions(runs, SEEDS)
        blocks[f'levels_{mask}'] = levels_markdown(levels, native)
        blocks[f'macro_{mask}'] = tier01.macro_markdown(runs, models, SEEDS)
        blocks[f'main_{mask}'] = main
        blocks[f'none_{mask}'] = [
            f"sdm_none vs mean_fill, {mask}: {none_mean['identical_cells']} / {none_mean['cells']} ô che dự đoán "
            f"giống hệt; chênh lớn nhất {none_mean['max_abs_count']:.3g} (count), {none_mean['max_abs_prop']:.3g} (tỷ lệ)."]
        manifest[mask] = {'ablation_run': {**records[0], **record}, 'step2_run': {**records[1], **base_record},
                          'masks_equal_step2': True, 'sdm_full_vs_step2_same_day_mean': full,
                          'regression_check': {'values_checked': checked + base_check[0],
                                               'max_abs_diff': max(worst, base_check[1])},
                          'levels_rerun_matches_long_file': True, 'sdm_none_vs_mean_fill': none_mean}
        label = 'chính' if mask == 'whole_feature' else 'sensitivity'
        md += [f'## {mask} ({label})', '', '### Trạng thái', '', *blocks[f'status_{mask}'], '',
               '### Bảng metric', '', *main, '', '### Biến thể vs sdm_full (mọi metric)', '',
               *blocks[f'wide_{mask}'], '', '### Đóng góp (mô tả, không cộng tính)', '', *blocks[f'share_{mask}'],
               '', '### Mức cửa sổ trên ô che thuộc nhóm model (gốc / nới 1 / nới 2 / toàn cục)', '',
               *blocks[f'levels_{mask}'], '', '### Series-macro so với cell-weighted', '', *blocks[f'macro_{mask}'],
               '', *blocks[f'none_{mask}'], '',
               f"Regression check: {checked + base_check[0]} giá trị khớp summary.json, sai khác lớn nhất "
               f"{max(worst, base_check[1]):.3g}. sdm_full = same_day_mean step 2: CSV native giống từng byte, "
               f"{full['hidden_cell_rows_identical']} dòng ô che giống hệt.", '']
    out_dir.mkdir(parents=True, exist_ok=True)
    tier01.write_csv(out_dir / 'ablation_status.csv', status_all)
    tier01.write_csv(out_dir / 'ablation_runs.csv', run_rows)
    tier01.write_csv(out_dir / 'ablation_pairs.csv', pair_rows)
    (out_dir / 'ablation_results.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    (out_dir / 'ablation_manifest.json').write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n',
                                                    encoding='utf-8')
    tier01.update_report(out_dir / 'REPORT_vi.md', blocks)
    return manifest, run_rows, pair_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', type=Path, default=HERE / 'runs')
    parser.add_argument('--step2-runs', type=Path, default=REPO / 'tv2_step2/runs')
    parser.add_argument('--out', type=Path, default=HERE)
    args = parser.parse_args(argv)
    summarize(args.runs, args.step2_runs, args.out)


if __name__ == '__main__':
    main()
