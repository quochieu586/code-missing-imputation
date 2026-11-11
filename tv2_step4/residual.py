"""Step 4: compare algorithms under one residual rule (exploratory), built on tier01 and ablation.

    PYTHONPATH=nextgen/src python tv2_step4/residual.py [--runs DIR] [--step2-runs DIR]
                                                       [--step3-runs DIR] [--out DIR]

Cells of the matrix (variant = model + rule) come from four run directories per mask:
  step 2 main_<mask>          hron_prop, tsagris_prop, micoda_rescale (A17 R1), mean_fill_cap
  step 2 micoda_other_<mask>  micoda_other_prop (adapter R2 branch)
  step 3 ablation_<mask>      sdm_cap (= sdm_full), sdm_rescale
  step 4 residual_<mask>      hron/tsagris/micoda_other x cap/rescale (data.apply_residual_rule)
plus step 4 check_<mask> (prop_other through apply_residual_rule, consistency only). Before any
table and before anything is written: every directory must share data, schema, mask kind,
rate, group map and the saved masks of seeds 0-4 with step 2 main_<mask> (F2-style), and the
long-file scores of every directory must match its summary.json (tier01.load_run).
Outputs in --out: residual_runs.csv, residual_pairs.csv, residual_results.md,
residual_manifest.json, and the auto:* blocks of REPORT_vi.md.
"""
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
_spec = importlib.util.spec_from_file_location('ablation', REPO / 'tv2_step3/ablation.py')
ablation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ablation)           # loads tier01 and puts nextgen/src on sys.path
tier01 = ablation.tier01
from longitudinal_impute.cells import read_predictions  # noqa: E402
from longitudinal_impute.data import load_csv  # noqa: E402
from longitudinal_impute.evaluate import implied_other_errors  # noqa: E402

MASKS = ('whole_feature', 'random_cell')
SEEDS = ('0', '1', '2', '3', '4')
VARIANTS = {   # label: (model, rule)
    'hron_prop': ('hron', 'prop_other'), 'hron_cap': ('hron', 'cap'), 'hron_rescale': ('hron', 'rescale'),
    'tsagris_prop': ('tsagris', 'prop_other'), 'tsagris_cap': ('tsagris', 'cap'),
    'tsagris_rescale': ('tsagris', 'rescale'), 'micoda_other_prop': ('micoda_other', 'prop_other'),
    'micoda_other_cap': ('micoda_other', 'cap'), 'micoda_other_rescale': ('micoda_other', 'rescale'),
    'micoda_rescale': ('micoda (A17)', 'rescale'), 'sdm_cap': ('same_day_mean', 'cap'),
    'sdm_rescale': ('same_day_mean', 'rescale'), 'mean_fill_cap': ('mean_fill', 'cap')}
RULES = ('prop_other', 'cap', 'rescale')
NEW = ('hron_cap', 'hron_rescale', 'tsagris_cap', 'tsagris_rescale', 'micoda_other_cap', 'micoda_other_rescale')
WITHIN_RULE = (('sdm', 'hron'), ('sdm', 'tsagris'), ('hron', 'tsagris'), ('micoda_other', 'hron'))
WITHIN_MODEL = ('hron', 'tsagris', 'micoda_other')
CHECKS = (('hron_prop_check', 'main', 'hron'), ('tsagris_prop_check', 'main', 'tsagris'),
          ('micoda_other_prop_check', 'micoda_other', 'micoda'))


def sources(step2, step3, step4, mask):
    """(folder, rename) of every directory that holds cells of the matrix."""
    return [(step2 / f'main_{mask}', {'hron': 'hron_prop', 'tsagris': 'tsagris_prop', 'micoda': 'micoda_rescale',
                                      'mean_fill': 'mean_fill_cap'}),
            (step2 / f'micoda_other_{mask}', {'micoda': 'micoda_other_prop'}),
            (step3 / f'ablation_{mask}', {'sdm_full': 'sdm_cap', 'sdm_rescale': 'sdm_rescale'}),
            (step4 / f'residual_{mask}', {v: v for v in NEW})]


def check_same(folder, reference, mask):
    """F2-style: the run may join the tables only if data, schema, mask and group map agree."""
    a, a_masks = tier01.step1.run_record(folder, mask, SEEDS)
    b, b_masks = tier01.step1.run_record(reference, mask, SEEDS)
    problems = [f'{key}: {a[key]!r} != {b[key]!r}' for key in ('data_sha256', 'schema_sha256', 'mask', 'rate',
                                                              'group_map') if a[key] != b[key]]
    problems += [f'seed {s}: saved masks differ' for s in SEEDS if not np.array_equal(a_masks[s], b_masks[s])]
    if not set(SEEDS) <= set(a['seeds']):
        problems.append(f'seeds {a["seeds"]} lack some of {list(SEEDS)}')
    if problems:
        raise ValueError(f'refusing to join {folder} with {reference.name}:\n  ' + '\n  '.join(problems))
    return ablation.relative(a)


def compare(path_a, model_a, path_b, model_b):
    """Hidden-cell predictions of two runs on the same cells: identical count and largest difference."""
    a, b = ablation.hidden_rows(path_a, model_a), ablation.hidden_rows(path_b, model_b)
    if [(r['row_id'], r['target'], r['seed']) for r in a] != [(r['row_id'], r['target'], r['seed']) for r in b]:
        raise ValueError(f'{model_a} and {model_b} do not cover the same hidden cells')
    diff = np.array([abs(x['pred_count'] - y['pred_count']) for x, y in zip(a, b)])
    return {'cells': len(a), 'identical': int((diff == 0).sum()), 'max_abs_count': float(diff.max())}


def compare_native(csv_a, csv_b, schema):
    if csv_a.read_bytes() == csv_b.read_bytes():
        return {'bytes_identical': True, 'max_abs_count': 0.}
    a, b = load_csv(csv_a, schema).counts, load_csv(csv_b, schema).counts
    return {'bytes_identical': False, 'max_abs_count': float(np.nanmax(np.abs(a - b)))}


def consistency(step2, step4, mask, schema):
    out = {}
    for check, prefix, model in CHECKS:
        folder = step2 / f'{prefix}_{mask}'
        out[check] = {**compare(step4 / f'check_{mask}' / f'predictions_{mask}.csv.gz', check,
                                folder / f'predictions_{mask}.csv.gz', model),
                      'native': compare_native(step4 / f'check_{mask}' / f'{check}_completed.csv',
                                               folder / f'{model}_completed.csv', schema)}
    path = step4 / f'residual_{mask}' / f'predictions_{mask}.csv.gz'
    out['tsagris_cap_vs_prop'] = compare(path, 'tsagris_cap', step2 / f'main_{mask}' / f'predictions_{mask}.csv.gz',
                                         'tsagris')
    return out


def p_rows(step4, mask):
    """From summary.json: the composition each new variant restored (row sums before closure)."""
    rows = []
    for prefix in ('residual', 'check'):
        summary = json.loads((step4 / f'{prefix}_{mask}' / 'summary.json').read_text(encoding='utf-8'))
        for name in summary['native']:
            infos = [summary['masked'][s]['models'][name]['info']['residual_rule'] for s in SEEDS]
            sums = [i['composition_row_sum'] for i in infos]
            rows.append(f"| {name} | {infos[0]['rule']} | {min(x['min'] for x in sums):.4g} / "
                        f"{np.median([x['median'] for x in sums]):.4g} / {max(x['max'] for x in sums):.4g} | "
                        + ('-' if 'rows_scaled' not in infos[0] else
                           f"{sum(i['rows_scaled'] for i in infos)} / {sum(i['rows'] for i in infos)}") + ' |')
    return ['| Biến thể | Quy tắc | Tổng dòng của composition trước khi đóng (min / median / max) | '
            'Dòng bị co/chuẩn hóa / dòng có ô thiếu |', '|---|---|---|---|', *rows]


def decomposition(runs, other, column=tier01.PRIMARY):
    """gap(sdm_cap - B_prop) split along the cap path and the rescale path; per seed, then mean."""
    def series(a, b):
        return [runs[a][s][column] - runs[b][s][column] for s in SEEDS]
    gap = series('sdm_cap', f'{other}_prop')
    paths = {'cap': [('thuật toán @cap', series('sdm_cap', f'{other}_cap')),
                     (f'quy tắc {other} (cap − prop_other)', series(f'{other}_cap', f'{other}_prop'))],
             'rescale': [('quy tắc same_day_mean (cap − rescale)', series('sdm_cap', 'sdm_rescale')),
                         ('thuật toán @rescale', series('sdm_rescale', f'{other}_rescale')),
                         (f'quy tắc {other} (rescale − prop_other)', series(f'{other}_rescale', f'{other}_prop'))]}
    mean_gap = float(np.mean(gap))
    out = [f'| Đường tách | Thành phần | Δ mean ± sd | âm / 5 seed | % khoảng cách |', '|---|---|---|---:|---:|',
           f"| — | khoảng cách sdm_cap − {other}_prop | {mean_gap:+.4f} ± {np.std(gap, ddof=1):.4f} | "
           f"{sum(g < 0 for g in gap)} | 100% |"]
    for path, terms in paths.items():
        for label, values in terms:
            out.append(f'| {path} | {label} | {np.mean(values):+.4f} ± {np.std(values, ddof=1):.4f} | '
                       f'{sum(v < 0 for v in values)} | {np.mean(values) / mean_gap:+.0%} |')
    return out


def truth_markdown(runs):
    columns = [c for c in tier01.METRIC_COLUMNS if c[1] in ('truth_zero', 'truth_positive', 'model_truth_zero')]
    out = ['| Biến thể | Model | Quy tắc | ' + ' | '.join(label for *_, label in columns) + ' |',
           '|---|---|---|' + '---:|' * len(columns)]
    for label, (model, rule) in VARIANTS.items():
        cells = []
        for column, *_ in columns:
            mean, sd = tier01.mean_sd([runs[label][s][column] for s in SEEDS])
            cells.append(f'{tier01.na(mean)} ± {tier01.na(sd)}')
        out.append(f'| {label} | {model} | {rule} | ' + ' | '.join(cells) + ' |')
    return out


def implied_other(paths, data, mask):
    """Implied Other MAE per variant and seed, on rows complete before masking with a hidden cell."""
    complete = set(np.flatnonzero(np.isfinite(data.counts).all(axis=1)).tolist())
    table, support = {}, {}
    for path, rename in paths:
        for model, rows in itertools.groupby(sorted(read_predictions(path), key=lambda r: (r['model'], r['seed'])),
                                             key=lambda r: (r['model'], r['seed'])):
            label = rename.get(model[0])
            if label is None or str(model[1]) not in SEEDS:
                continue
            rows = list(rows)
            score = implied_other_errors([r['row_id'] for r in rows], [r['truth_count'] for r in rows],
                                         [r['pred_count'] for r in rows], [r['N'] for r in rows],
                                         [r['entity'] for r in rows], complete)
            table[label, str(model[1])] = score
            support[str(model[1])] = (score['rows'], score['entities'])
    out = [f"Dòng đầy đủ trước khi che: {len(complete)}, thuộc {len({data.entities[i] for i in complete})} nước. "
           'Dòng dùng / số nước theo seed: ' + ', '.join(f'seed {s}: {r} / {e}' for s, (r, e) in sorted(support.items())),
           '', '| Biến thể | Quy tắc | MAE tỷ lệ Other ngầm | Sai số có dấu (ngầm − thật) |', '|---|---|---:|---:|']
    for label, (_, rule) in VARIANTS.items():
        mae = tier01.mean_sd([table[label, s]['mae_proportion'] for s in SEEDS])
        signed = tier01.mean_sd([table[label, s]['mean_signed_proportion'] for s in SEEDS])
        out.append(f'| {label} | {rule} | {tier01.na(mae[0])} ± {tier01.na(mae[1])} | '
                   f'{tier01.na(signed[0], 4)} ± {tier01.na(signed[1])} |')
    return out, {f'{k[0]}|{k[1]}': v for k, v in table.items()}


def load_data(folder):
    config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    if hashlib.sha256(Path(config['data']).read_bytes()).hexdigest() != config['data_sha256']:
        raise ValueError(f"{folder.name}: {config['data']} changed since the run")
    return load_csv(config['data'], config['schema_path']), config['schema_path']


def by_rule(rule):
    return [v for v, (_, r) in VARIANTS.items() if r == rule]


def label(model, rule):
    return {'sdm': f'sdm_{rule}', 'micoda_other': f'micoda_other_{rule}'}.get(model, f'{model}_{rule}')


def summarize(step2, step3, step4, out_dir):
    manifest, run_rows, pair_rows, blocks = {}, [], [], {}
    md = ['# Step 4: so sánh theo quy tắc phần dư (COVID đầy đủ, che 5%, seed 0–4) — **exploratory**', '',
          'Sinh bởi `residual.py` (chấm điểm bằng `tier01`); không sửa tay. Metric chính: MAE tỷ lệ, '
          'cell-weighted, nhóm ô model. mean ± sd (ddof=1) qua seed, không phải khoảng tin cậy.', '']
    for mask in MASKS:
        reference = step2 / f'main_{mask}'
        folders = sources(step2, step3, step4, mask)
        records = {f.name: check_same(f, reference, mask) for f, _ in folders}
        records[f'check_{mask}'] = check_same(step4 / f'check_{mask}', reference, mask)
        status, scores, checks = [], {}, {}
        for folder, rename in folders:
            own_status, own_scores, record, (checked, worst) = tier01.load_run(folder, rename, mask)
            status += [r for r in own_status if r['model'] in rename.values()]
            scores.update({k: v for k, v in own_scores.items() if k[0] in rename.values()})
            records[folder.name].update(record)
            checks[folder.name] = {'values_checked': checked, 'max_abs_diff': worst}
        _, _, record, (checked, worst) = tier01.load_run(step4 / f'check_{mask}', {}, mask)
        records[f'check_{mask}'].update(record)
        checks[f'check_{mask}'] = {'values_checked': checked, 'max_abs_diff': worst}
        data, schema = load_data(reference)
        runs = tier01.table_runs(status, scores)
        missing = [v for v in VARIANTS if v not in runs]
        if missing:
            raise ValueError(f'{mask}: no runs for {missing}')
        run_rows += [{'mask': mask, 'variant': v, 'model': VARIANTS[v][0], 'rule': VARIANTS[v][1], 'seed': s,
                      **runs[v][s]} for v in VARIANTS for s in SEEDS]
        same = consistency(step2, step4, mask, schema)
        blocks[f'p_{mask}'] = p_rows(step4, mask)
        blocks[f'consistency_{mask}'] = [
            f"- {name}: {c['identical']} / {c['cells']} ô che giống hệt, chênh lớn nhất {c['max_abs_count']:.3g} count"
            + (f"; CSV native {'giống từng byte' if c['native']['bytes_identical'] else 'khác'}"
               f" (chênh lớn nhất {c['native']['max_abs_count']:.3g})" if 'native' in c else '')
            for name, c in same.items()]
        for rule in RULES:
            blocks[f'rule_{rule}_{mask}'] = tier01.main_markdown(runs, by_rule(rule), SEEDS)[0]
        pairs = [(label(a, r), label(b, r)) for r in ('cap', 'rescale') for a, b in WITHIN_RULE]
        blocks[f'within_rule_{mask}'], rows = ablation.wide_pairs(runs, pairs, SEEDS)
        pair_rows += [{'mask': mask, 'kind': 'within_rule', **p} for p in rows]
        pairs = [(f'{m}_{r}', f'{m}_prop') for m in WITHIN_MODEL for r in ('cap', 'rescale')]
        blocks[f'within_model_{mask}'], rows = ablation.wide_pairs(runs, pairs, SEEDS)
        pair_rows += [{'mask': mask, 'kind': 'within_model', **p} for p in rows]
        blocks[f'gap_{mask}'] = [*decomposition(runs, 'hron'), '', *decomposition(runs, 'tsagris')]
        blocks[f'truth_{mask}'] = truth_markdown(runs)
        blocks[f'other_{mask}'], other = implied_other([(f / f'predictions_{mask}.csv.gz', r) for f, r in folders],
                                                       data, mask)
        blocks[f'macro_{mask}'] = [line for rule in RULES for line in
                                   (f'**{rule}**', *tier01.macro_markdown(runs, by_rule(rule), SEEDS), '')]
        blocks[f'macro_{mask}'] += ['**mọi biến thể**', *tier01.macro_markdown(runs, list(VARIANTS), SEEDS)]
        manifest[mask] = {'runs': records, 'regression_check': checks, 'masks_equal_step2_main': True,
                          'consistency': same, 'implied_other': other}
        md += [f"## {mask} ({'chính' if mask == 'whole_feature' else 'sensitivity'})", '',
               '### Composition được chia (xác minh p)', '', *blocks[f'p_{mask}'], '',
               '### Consistency check', '', *blocks[f'consistency_{mask}'], '']
        for rule in RULES:
            md += [f'### Bảng 1: quy tắc `{rule}`', '', *blocks[f'rule_{rule}_{mask}'], '']
        md += ['### Bảng 2: ghép cặp trong cùng quy tắc', '', *blocks[f'within_rule_{mask}'], '',
               '### Bảng 3: ghép cặp trong cùng model', '', *blocks[f'within_model_{mask}'], '',
               '### Bảng 4: tách khoảng cách (MAE tỷ lệ; mô tả, không phải phân rã chặt)', '', *blocks[f'gap_{mask}'],
               '', '### Bảng 5: theo nhóm đáp án', '', *blocks[f'truth_{mask}'], '',
               '### Bảng 6: Other ngầm', '', *blocks[f'other_{mask}'], '',
               '### Bảng 7: series-macro so với cell-weighted', '', *blocks[f'macro_{mask}'], '']
    out_dir.mkdir(parents=True, exist_ok=True)
    tier01.write_csv(out_dir / 'residual_runs.csv', run_rows)
    tier01.write_csv(out_dir / 'residual_pairs.csv', [{k: v for k, v in p.items() if k != 'dropped'}
                                                      | {'seeds_used': ' '.join(p['seeds_used'])} for p in pair_rows])
    (out_dir / 'residual_results.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    (out_dir / 'residual_manifest.json').write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n',
                                                    encoding='utf-8')
    tier01.update_report(out_dir / 'REPORT_vi.md', blocks)
    return manifest, run_rows, pair_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', type=Path, default=HERE / 'runs')
    parser.add_argument('--step2-runs', type=Path, default=REPO / 'tv2_step2/runs')
    parser.add_argument('--step3-runs', type=Path, default=REPO / 'tv2_step3/runs')
    parser.add_argument('--out', type=Path, default=HERE)
    args = parser.parse_args(argv)
    summarize(args.step2_runs, args.step3_runs, args.runs, args.out)


if __name__ == '__main__':
    main()
