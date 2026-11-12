"""Step 5: tier 2 diagnostics on the real missing cells (native), baselines only. Descriptive, NOT accuracy.

    PYTHONPATH=nextgen/src python tv2_step5/tier2.py [--out DIR] [--rerun DIR]

Definitions: nextgen/src/longitudinal_impute/tier2_diagnostics.py, frozen (committed) before this
script read any output. Order of work:
1. Reuse the native completed CSV of each model from the step 2 / step 4 run directories, after
   checking data sha256, schema, group map sha256, the model's table against the current models
   file, a clean git tree, native status ok, the same CSV bytes in the random_cell directory, and
   the observed counts unchanged. Any mismatch stops the script. --rerun DIR (optional): native
   CSVs written again by the current code (run.py --mask none) must be byte-identical.
2. E = M_m (real missing, not forced to 0), V = O (observed). The R-local support is computed once.
3. D1, D2 (R-local, R-pooled), D3 per model x target; strata; the two references; decoys.
Outputs in --out: tier2_targets.csv, tier2_strata.csv, tier2_support.csv, tier2_results.md,
tier2_manifest.json; the auto:* blocks of REPORT_vi.md are regenerated. No ranking anywhere.
"""
import argparse
import csv
from datetime import date, timedelta
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / 'nextgen/src'))
from longitudinal_impute.data import load_csv, read_schema, check_completed  # noqa: E402
from longitudinal_impute.run import read_models  # noqa: E402
from longitudinal_impute import tier2_diagnostics as t2  # noqa: E402

_spec = importlib.util.spec_from_file_location('tier01', REPO / 'tv2_step2/tier01.py')
tier01 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tier01)

DATA, SCHEMA = REPO / 'data/covariants.csv', REPO / 'nextgen/configs/baselines_covariants.toml'
GROUP_MAP, CONFIGS = REPO / 'nextgen/configs/covid_location_continent.csv', REPO / 'nextgen/configs'
MAIN, OTHER, RESIDUAL = 'tv2_step2/runs/main', 'tv2_step2/runs/micoda_other', 'tv2_step4/runs/residual'
# reported name: (run directory prefix, table in that run's config, models file, --rerun subfolder)
SOURCES = {
    'hron': (MAIN, 'hron', 'models_default.toml', 'main'),
    'tsagris': (MAIN, 'tsagris', 'models_default.toml', 'main'),
    'micoda': (MAIN, 'micoda', 'models_default.toml', 'main'),
    'micoda_other': (OTHER, 'micoda', 'models_micoda_other.toml', 'other'),
    'same_day_mean': (MAIN, 'same_day_mean', 'models_default.toml', 'main'),
    'mean_fill': (MAIN, 'mean_fill', 'models_default.toml', 'main'),
    'zero_fill': (MAIN, 'zero_fill', 'models_default.toml', 'main'),
    'marginal_sampler': (MAIN, 'marginal_sampler', 'models_default.toml', 'main'),
    'hron_cap': (RESIDUAL, 'hron_cap', 'models_residual_rule.toml', 'residual'),
    'tsagris_cap': (RESIDUAL, 'tsagris_cap', 'models_residual_rule.toml', 'residual'),
    'micoda_other_cap': (RESIDUAL, 'micoda_other_cap', 'models_residual_rule.toml', 'residual'),
}
MASKS = ('whole_feature', 'random_cell')     # every run directory holds the native run; read the first
TIER1 = {'tv2_step2/tier01_runs.csv': ('model', ('hron', 'tsagris', 'micoda', 'micoda_other', 'same_day_mean',
                                                 'mean_fill', 'zero_fill', 'marginal_sampler')),
         'tv2_step4/residual_runs.csv': ('variant', ('hron_cap', 'tsagris_cap', 'micoda_other_cap'))}
TIER1_SEEDS = ('0', '1', '2', '3', '4')
sha256, na, update_report = tier01.sha256, tier01.na, tier01.update_report


def signed(value, digits=3):
    return 'NA' if value is None else f'{value:+.{digits}f}'


# ---------- step 1: reuse checks ----------

def load_source(name, data, rerun=None):
    """Checked native completed counts of one model, and its provenance record."""
    prefix, table, models_file, rerun_dir = SOURCES[name]
    folder = REPO / f'{prefix}_{MASKS[0]}'
    config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    path, twin = folder / f'{table}_completed.csv', REPO / f'{prefix}_{MASKS[1]}' / f'{table}_completed.csv'
    group_map = config.get('group_map') or {}
    problems = [text for bad, text in (
        (config['data_sha256'] != sha256(DATA), 'data sha256 differs from data/covariants.csv'),
        (config['schema'] != read_schema(SCHEMA), 'schema differs from the current schema'),
        (group_map.get('sha256') != sha256(GROUP_MAP) or group_map.get('rows_resolved') != len(data.counts),
         'group map differs'),
        (config['models'].get(table) != read_models(CONFIGS / models_file, [table])[table],
         f'[{table}] differs from {models_file}'),
        ((config.get('git') or {}).get('uncommitted_changes') is not False, 'run made from an unclean tree'),
        ((summary['native'].get(table) or {}).get('status') != 'ok', 'native status is not ok'),
        (not path.exists() or not twin.exists() or sha256(path) != sha256(twin),
         f'native CSV missing or not identical in the {MASKS[1]} run'),
    ) if bad]
    if problems:
        raise ValueError(f'{name} ({folder.relative_to(REPO)}): ' + '; '.join(problems))
    counts = read_completed(path, data)
    check_completed(data, counts, np.zeros(data.counts.shape, bool))
    record = {'dir': folder.relative_to(REPO).as_posix(), 'table': table, 'models_file': models_file,
              'config_sha256': sha256(folder / 'config.json'), 'csv': path.name, 'csv_sha256': sha256(path),
              'git_commit': (config.get('git') or {}).get('commit'), f'same_csv_in_{MASKS[1]}': True}
    if rerun is not None:
        again = rerun / rerun_dir / path.name
        record['rerun_identical'] = again.exists() and sha256(again) == sha256(path)
    return counts, record


def read_completed(path, data):
    """Target columns of a completed CSV, row by row equal to the data apart from the filled targets.

    Not load_csv: it refuses a row whose filled sum exceeds N by a rounding error, which
    check_completed accepts (1e-10 relative), so the checks are done by check_completed."""
    with path.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(data.rows):
        raise ValueError(f'{path.name}: {len(rows)} rows, the data has {len(data.rows)}')
    fixed = [c for c in data.fields if c not in data.targets]
    if any(row[c] != source[c] for row, source in zip(rows, data.rows) for c in fixed):
        raise ValueError(f'{path.name}: non-target columns differ from the data')
    return np.array([[float(row[t]) for t in data.targets] for row in rows])


# ---------- tables ----------

def day_text(first, day):
    return 'NA' if day is None else (first + timedelta(days=int(day))).isoformat()


def support_lines(data, cells, results, first):
    one = next(iter(results.values()))
    m, f, mm = (int(cells[k].sum()) for k in ('missing', 'forced', 'eval'))
    q1, q2 = one['n_cuts']
    out = [f"Ô thiếu thật M = {m}; ép về 0 M_f = {f} ({f / m:.2%}); model quyết định M_m = {mm} ({mm / m:.2%}); "
           f"ô quan sát O = {int(cells['ref'].sum())}. R-local có support: {one['supported']} / {mm} ô M_m "
           f"(coverage {one['supported'] / mm:.2%}). Ngưỡng nhóm N (tam phân vị trên dòng có ô M_m): "
           f"N1 ≤ {q1:g} < N2 ≤ {q2:g} < N3.", '',
           '| Target | ô O | ô M_m | có support | coverage | TB ô O trong reference | TB location | '
           'T_j (ngày đầu – cuối có count dương) | ô M_m ngoài T_j |', '|---|' + '---:|' * 6 + '---|---:|']
    rows = []
    for name, v in one['targets'].items():
        span = v['span'] or (None, None)
        out.append(f"| {name} | {v['ref_cells']} | {v['eval_cells']} | {v['supported']} | {na(v['coverage'], 3)} | "
                   f"{na(v['mean_ref_cells'], 1)} | {na(v['mean_ref_locations'], 1)} | "
                   f"{day_text(first, span[0])} – {day_text(first, span[1])} | {v['outside_cells']} |")
        rows.append({'kind': 'target', 'stratum': name, **{k: v[k] for k in (
            'ref_cells', 'eval_cells', 'supported', 'coverage', 'mean_ref_cells', 'mean_ref_locations',
            'outside_cells')}, 'span_first': day_text(first, span[0]), 'span_last': day_text(first, span[1])})
    out += ['', '| Tầng | ô M_m | có support | coverage | số target có ≥ 30 ô có support |', '|---|---:|---:|---:|---:|']
    for s in one['strata']:
        enough = s['d1']['targets']
        out.append(f"| {s['stratum']} | {s['eval_cells']} | {s['supported']} | "
                   f"{na(s['supported'] / s['eval_cells'] if s['eval_cells'] else None, 3)} | {enough} |")
        rows.append({'kind': s['kind'], 'stratum': s['stratum'], 'eval_cells': s['eval_cells'],
                     'supported': s['supported'], 'targets_scored': enough})
    return out, rows


def target_table(results, targets, ref, key, digits, fmt):
    """Model x target table of results[m]['targets'][t][ref][key], with the two aggregates."""
    head = ['| Model | ' + ' | '.join(targets) + ' | TB đều | TB trọng số (số ô) |',
            '|---|' + '---:|' * (len(targets) + 2)]
    for m, r in results.items():
        agg = r['summary'][f'{ref}_{key}']
        head.append(f'| {m} | ' + ' | '.join(fmt(r['targets'][t][ref][key], digits) for t in targets)
                    + f" | {fmt(agg['equal'], digits)} | {fmt(agg['weighted'], digits)} ({agg['targets']} target) |")
    return head


def d1_lines(results, targets):
    out = []
    for ref, label in (('local', 'R-local (chính)'), ('pooled', 'R-pooled')):
        one = next(iter(results.values()))['targets']
        out += [f'**D1 {label}** = Pr̂(zero, ô M_m) − Pr_ref(zero). Dòng đầu: Pr_ref(zero) của reference.', '',
                *target_table(results, targets, ref, 'd1', 3, signed)[:2],
                '| *Pr_ref(zero)* | ' + ' | '.join(na(one[t][ref]['pr_zero_ref'], 3) for t in targets) + ' | | |',
                *target_table(results, targets, ref, 'd1', 3, signed)[2:], '']
    return out


def d2_lines(results, targets):
    out = []
    for ref, label in (('local', 'R-local (chính)'), ('pooled', 'R-pooled')):
        out += [f'**D2 {label}** = W1 trên tỷ lệ c/N, phần dương (count ≥ 0.5). NA = một phía không có ô dương.',
                '', *target_table(results, targets, ref, 'd2', 4, na), '']
    return out


def d3_lines(results, targets):
    one = next(iter(results.values()))['targets']
    out = ['Số ô M_m được điền dương (≥ 0.5 count) ở ngày ngoài T_j; dòng đầu: số ô M_m ngoài T_j (mọi model).', '',
           '| Model | ' + ' | '.join(targets) + ' | tổng ô | tổng khối lượng tỷ lệ |',
           '|---|' + '---:|' * (len(targets) + 2),
           '| *ô M_m ngoài T_j* | ' + ' | '.join(str(one[t]['outside_cells']) for t in targets)
           + f" | {sum(one[t]['outside_cells'] for t in targets)} | |"]
    for m, r in results.items():
        out.append(f'| {m} | ' + ' | '.join(str(r['targets'][t]['d3_cells']) for t in targets)
                   + f" | {r['d3_cells']} | {r['d3_mass']:.3f} |")
    return out


def strata_lines(results):
    out = []
    one = next(iter(results.values()))['strata']
    for kind, title in (('group', 'châu lục'), ('n', 'nhóm N')):
        labels = [s['stratum'] for s in one if s['kind'] == kind]
        for key, label in (('d1', 'D1 R-local'), ('d2', 'D2 R-local')):
            out += [f'**{label} theo {title}**: TB đều / TB trọng số qua các target có ≥ 30 ô có support '
                    '(số target trong ngoặc).', '', '| Model | ' + ' | '.join(labels) + ' |',
                    '|---|' + '---:|' * len(labels)]
            for m, r in results.items():
                cells = {s['stratum']: s[key] for s in r['strata'] if s['kind'] == kind}
                fmt = signed if key == 'd1' else na
                out.append(f'| {m} | ' + ' | '.join(
                    f"{fmt(cells[x]['equal'], 3)} / {fmt(cells[x]['weighted'], 3)} ({cells[x]['targets']})"
                    for x in labels) + ' |')
            out.append('')
        out += [f'**D3 theo {title}**: số ô cờ / khối lượng tỷ lệ (NA khi tầng có < 30 ô M_m).', '',
                '| Model | ' + ' | '.join(labels) + ' |', '|---|' + '---:|' * len(labels)]
        for m, r in results.items():
            cells = {s['stratum']: s for s in r['strata'] if s['kind'] == kind}
            out.append(f'| {m} | ' + ' | '.join('NA' if cells[x]['d3_cells'] is None else
                                                 f"{cells[x]['d3_cells']} / {cells[x]['d3_mass']:.3f}"
                                                 for x in labels) + ' |')
        out.append('')
    return out


def shift_lines(results, targets):
    """Table 6: per target and model, does D1 change sign or by >= MATERIAL_CHANGE between references?"""
    one = next(iter(results.values()))['targets']
    out = [f'Ký hiệu: **±** đổi dấu, **L** |D1 local − D1 pooled| ≥ {t2.MATERIAL_CHANGE:g}, · không có cả hai. '
           'Cột Δref = Pr_ref(zero) local − pooled (chung mọi model).', '',
           '| Target | Δref | ' + ' | '.join(results) + ' |', '|---|---:|' + ':---:|' * len(results)]
    counts = {m: [0, 0, 0] for m in results}
    for t in targets:
        ref = one[t]
        delta = None if ref['local']['pr_zero_ref'] is None else ref['local']['pr_zero_ref'] - ref['pooled'][
            'pr_zero_ref']
        marks = []
        for m, r in results.items():
            s = t2.reference_shift(r['targets'][t]['local']['d1'], r['targets'][t]['pooled']['d1'])
            if s['change'] is None:
                marks.append('NA')
                continue
            counts[m][0] += s['sign_differs']
            counts[m][1] += s['large']
            counts[m][2] += 1
            marks.append(('±' if s['sign_differs'] else '') + ('L' if s['large'] else '') or '·')
        out.append(f'| {t} | {signed(delta)} | ' + ' | '.join(marks) + ' |')
    out += ['', 'Số target đổi dấu / thay đổi lớn (trên số target có cả hai giá trị): ' + '; '.join(
        f'{m} {c[0]} / {c[1]} (/{c[2]})' for m, c in counts.items()) + '.']
    return out


def decoy_lines(results, targets):
    """Table 7: behaviour of the diagnostics on the two decoys. Not a gate."""
    out = []
    if 'zero_fill' in results:
        z = results['zero_fill']['targets']
        has = {ref: [t for t in targets if z[t][ref]['d1'] is not None] for ref in ('local', 'pooled')}
        exact = sum(z[t]['pooled']['d1'] == 1 - z[t]['pooled']['pr_zero_ref'] for t in has['pooled'])
        top = {ref: sum(all(r['targets'][t][ref]['d1'] is None or r['targets'][t][ref]['d1'] <= z[t][ref]['d1']
                            for r in results.values()) for t in has[ref]) for ref in has}
        d2na = {ref: sum(z[t][ref]['d2'] is None for t in has[ref]) for ref in has}
        s = results['zero_fill']['summary']
        n = {ref: len(v) for ref, v in has.items()}
        out += [f"- zero_fill: D1 pooled = 1 − Pr_ref(zero) ở {exact}/{n['pooled']} target có D1 (mọi ô M_m "
                f"điền 0). D1 TB đều R-local {signed(s['local_d1']['equal'])}, R-pooled "
                f"{signed(s['pooled_d1']['equal'])}. D1 lớn nhất (hoặc bằng) trong các model ở "
                f"{top['local']}/{n['local']} target (R-local), {top['pooled']}/{n['pooled']} (R-pooled). D2 = NA ở "
                f"{d2na['local']}/{n['local']} (R-local), {d2na['pooled']}/{n['pooled']} (R-pooled) target có D1. "
                f"Target không có D1 (không có ô M_m, hoặc không ô nào có support): "
                f"{', '.join(t for t in targets if t not in has['local']) or 'không'} (R-local); "
                f"{', '.join(t for t in targets if t not in has['pooled']) or 'không'} (R-pooled)."]
    if 'marginal_sampler' in results:
        r = results['marginal_sampler']
        s = r['summary']
        has = [t for t in targets if r['targets'][t]['pooled']['d1'] is not None]
        small = sum(abs(r['targets'][t]['pooled']['d1']) <= .05 for t in has)
        out += [f"- marginal_sampler: D1 R-pooled TB đều {signed(s['pooled_d1']['equal'])}, trọng số "
                f"{signed(s['pooled_d1']['weighted'])}; |D1 pooled| ≤ 0.05 ở {small}/{len(has)} target có D1. "
                f"D2 R-pooled TB đều {na(s['pooled_d2']['equal'])}, trọng số {na(s['pooled_d2']['weighted'])}. "
                f"R-local: D1 {signed(s['local_d1']['equal'])} / {signed(s['local_d1']['weighted'])}, "
                f"D2 {na(s['local_d2']['equal'])} / {na(s['local_d2']['weighted'])} (đều / trọng số)."]
    return out


def overview_lines(results):
    out = ['| Model | D1 local (đều / trọng số) | D2 local (đều / trọng số) | D1 pooled (đều / trọng số) | '
           'D2 pooled (đều / trọng số) | D3 ô | D3 khối lượng |', '|---|---|---|---|---|---:|---:|']
    for m, r in results.items():
        s = r['summary']
        out.append(f"| {m} | {signed(s['local_d1']['equal'])} / {signed(s['local_d1']['weighted'])} | "
                   f"{na(s['local_d2']['equal'])} / {na(s['local_d2']['weighted'])} | "
                   f"{signed(s['pooled_d1']['equal'])} / {signed(s['pooled_d1']['weighted'])} | "
                   f"{na(s['pooled_d2']['equal'])} / {na(s['pooled_d2']['weighted'])} | {r['d3_cells']} | "
                   f"{r['d3_mass']:.3f} |")
    return out


def tier1_rows():
    """Tier 1 means (committed step 2 / step 4 tables) over the seeds where all 11 models count."""
    runs = {}
    for path, (column, models) in TIER1.items():
        with (REPO / path).open(encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                if row[column] in models and row['seed'] in TIER1_SEEDS:
                    runs.setdefault(row['mask'], {}).setdefault(row[column], {})[row['seed']] = row
    out = {}
    for mask, models in runs.items():
        seeds = [s for s in TIER1_SEEDS if all(models.get(m, {}).get(s, {}).get('eligible') == 'True'
                                               for m in SOURCES)]
        out[mask] = {'seeds': seeds, 'means': {m: {k: float(np.mean([float(models[m][s][k]) for s in seeds]))
                                                   for k in ('mae_prop', 'mae_prop_model_truth_zero',
                                                             'mae_prop_truth_positive')} for m in SOURCES}}
    return out


def tier1_lines(results, tier1):
    seeds = {m: ', '.join(v['seeds']) for m, v in tier1.items()}
    out = [f"Tầng 1: trung bình trên seed chung của 11 model (whole_feature: {seeds.get('whole_feature')}; "
           f"random_cell: {seeds.get('random_cell')}), lấy từ tier01_runs.csv và residual_runs.csv đã commit. "
           'Nhóm ô model; z = đáp án 0, + = đáp án dương. Tầng 2: native, R-local, TB trọng số.', '',
           '| Model | T1 wf MAE | T1 wf MAE z | T1 wf MAE + | T1 rc MAE | T1 rc MAE z | T1 rc MAE + | '
           'T2 D1 | T2 D2 | T2 D3 ô |', '|---|' + '---:|' * 9]
    for m, r in results.items():
        cols = [na(tier1[mask]['means'][m][k]) for mask in ('whole_feature', 'random_cell')
                for k in ('mae_prop', 'mae_prop_model_truth_zero', 'mae_prop_truth_positive')]
        s = r['summary']
        out.append(f"| {m} | " + ' | '.join(cols) + f" | {signed(s['local_d1']['weighted'])} | "
                   f"{na(s['local_d2']['weighted'])} | {r['d3_cells']} |")
    return out


# ---------- driver ----------

def write_csv(path, rows):
    tier01.write_csv(path, rows)


def summarize(out_dir, rerun=None):
    data = load_csv(DATA, SCHEMA)
    first = date.fromisoformat(data.rows[int(np.argmin(data.days))]['date'])
    loaded = {name: load_source(name, data, rerun) for name in SOURCES}
    facts, cells = t2.row_facts(data), t2.native_cells(data)
    support = t2.local_support(facts, cells['eval'], cells['ref'])
    targets = list(data.targets)
    results = {name: t2.diagnose(values, facts, cells['eval'], cells['ref'], support, targets)
               for name, (values, _) in loaded.items()}
    blocks = {}
    blocks['support'], support_rows = support_lines(data, cells, results, first)
    blocks['d1'], blocks['d2'] = d1_lines(results, targets), d2_lines(results, targets)
    blocks['d3'], blocks['strata'] = d3_lines(results, targets), strata_lines(results)
    blocks['shift'], blocks['decoys'] = shift_lines(results, targets), decoy_lines(results, targets)
    blocks['overview'], blocks['tier1'] = overview_lines(results), tier1_lines(results, tier1_rows())
    target_rows = [{'model': m, 'ref': ref, 'target': t, 'eval_cells': v['eval_cells'] if ref == 'pooled' else
                    v['supported'], **v[ref]} for m, r in results.items() for t, v in r['targets'].items()
                   for ref in ('local', 'pooled')]
    target_rows += [{'model': m, 'ref': 'd3', 'target': t, 'eval_cells': v['eval_cells'], 'outside_cells':
                     v['outside_cells'], 'd3_cells': v['d3_cells'], 'd3_mass': v['d3_mass']}
                    for m, r in results.items() for t, v in r['targets'].items()]
    strata_rows = [{'model': m, 'kind': s['kind'], 'stratum': s['stratum'], 'eval_cells': s['eval_cells'],
                    'supported': s['supported'], **{f'{d}_{k}': s[d][k] for d in ('d1', 'd2')
                                                    for k in ('equal', 'weighted', 'targets', 'cells')},
                    'd3_cells': s['d3_cells'], 'd3_mass': s['d3_mass']} for m, r in results.items() for s in r['strata']]
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / 'tier2_targets.csv', target_rows)
    write_csv(out_dir / 'tier2_strata.csv', strata_rows)
    write_csv(out_dir / 'tier2_support.csv', support_rows)
    md = ['# Step 5: Tầng 2 native (mô tả, không xếp hạng) — **exploratory**', '',
          'Sinh bởi `tier2.py`; không sửa tay. Không có đáp án trên ô thiếu thật: mọi số là mô tả so với '
          'reference từ ô quan sát, không phải độ chính xác.', '']
    for name, title in (('support', '1. Support'), ('d1', '2. D1'), ('d2', '3. D2'), ('d3', '4. D3'),
                        ('strata', '5. Phân tầng'), ('shift', '6. R-local vs R-pooled'),
                        ('decoys', '7. Đối thủ giả'), ('overview', 'Tổng hợp'), ('tier1', 'So với Tầng 1')):
        md += [f'## {title}', '', *blocks[name], '']
    (out_dir / 'tier2_results.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    manifest = {'data_sha256': sha256(DATA), 'schema_sha256': sha256(SCHEMA), 'group_map_sha256': sha256(GROUP_MAP),
                'diagnostics_sha256': sha256(REPO / 'nextgen/src/longitudinal_impute/tier2_diagnostics.py'),
                'constants': {k: getattr(t2, k) for k in ('ZERO_BELOW', 'WINDOW_DAYS', 'MIN_CELLS', 'MIN_LOCATIONS',
                                                          'MIN_STRATUM_CELLS', 'MATERIAL_CHANGE')},
                'cells': {k: int(v.sum()) for k, v in cells.items()}, 'models': {m: rec for m, (_, rec) in loaded.items()},
                'results_sha256': {m: __import__('hashlib').sha256(json.dumps(r, sort_keys=True).encode()).hexdigest()
                                   for m, r in results.items()}}
    (out_dir / 'tier2_manifest.json').write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n',
                                                 encoding='utf-8')
    blocks['check'] = check_lines(manifest, rerun is not None)
    update_report(out_dir / 'REPORT_vi.md', blocks)
    return results, manifest


def check_lines(manifest, rerun):
    models = manifest['models']
    out = [f"data sha256 `{manifest['data_sha256'][:12]}`, schema `{manifest['schema_sha256'][:12]}`, group map "
           f"`{manifest['group_map_sha256'][:12]}`, tier2_diagnostics.py `{manifest['diagnostics_sha256'][:12]}`.", '',
           '| Model | Thư mục run | commit của run | CSV native sha256 | cùng byte ở random_cell | chạy lại = cùng byte |',
           '|---|---|---|---|:---:|:---:|']
    for m, rec in models.items():
        again = rec.get('rerun_identical', 'không chạy') if rerun else 'không chạy'
        out.append(f"| {m} | `{rec['dir']}` | `{(rec['git_commit'] or '')[:7]}` | `{rec['csv_sha256'][:12]}` | "
                   f"{rec[f'same_csv_in_{MASKS[1]}']} | {again} |")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=HERE)
    parser.add_argument('--rerun', type=Path, help='folder with main/, other/, residual/ native runs of the current code')
    args = parser.parse_args(argv)
    summarize(args.out, args.rerun)


if __name__ == '__main__':
    main()
