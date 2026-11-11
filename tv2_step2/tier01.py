"""Step 2: tier 0 (run status) and tier 1 (errors on masked cells) tables, all labelled exploratory.

    PYTHONPATH=nextgen/src python tv2_step2/tier01.py [--runs DIR] [--out DIR] [--step1-runs DIR]

Reads <runs>/main_<mask> (7 models) and <runs>/micoda_other_<mask> (MICoDa other = true, reported
as micoda_other), written by run_all.sh with --save-predictions. Order of work:
1. F2 (step 1): validate_pair refuses to merge the two runs of a mask unless data, schema,
   mask, rate, seeds, group map and every saved mask agree.
2. Tier 0: one status row per scheduled run (config models x seeds, plus native), read from
   summary.json; a scheduled run with no result is reported as 'missing', never dropped.
3. Tier 1: every metric is recomputed from predictions_<mask>.csv.gz (cells.score_predictions)
   and must equal the errors in summary.json to 1e-12, or nothing is written.
4. F07: a run counts only if completed with coverage 1. Model means are taken over the seeds
   where every model counts; each pair is compared over the seeds where both count. The
   reliability table keeps every run. No per-model success-only mean is ranked.
Outputs in --out: tier01_status.csv, tier01_runs.csv, tier01_pairs.csv, tier01_results.md,
tier01_manifest.json; the auto:* blocks of REPORT_vi.md are regenerated.
"""
import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / 'nextgen/src'))
from longitudinal_impute.cells import read_predictions, score_predictions  # noqa: E402
from longitudinal_impute.evaluate import GROUPS, METRICS  # noqa: E402

_spec = importlib.util.spec_from_file_location('summarize_runs', REPO / 'tv2_step1/summarize_runs.py')
step1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(step1)

SEEDS = {'whole_feature': ('0', '1', '2', '3', '4', '109'), 'random_cell': ('0', '1', '2', '3', '4')}
PASSES = (('main', {}), ('micoda_other', {'micoda': 'micoda_other'}))
# (column, group, key, label); the first one is the primary metric.
METRIC_COLUMNS = (('mae_prop', 'model', 'mae_proportion', 'MAE tỷ lệ'),
                  ('mse_prop', 'model', 'mse_proportion', 'MSE tỷ lệ'),
                  ('mae_count', 'model', 'mae_count', 'MAE count'),
                  ('series_macro_mae_prop', None, 'series_macro_mae_proportion', 'MAE tỷ lệ series-macro'),
                  ('mae_prop_truth_zero', 'truth_zero', 'mae_proportion', 'MAE tỷ lệ, đáp án = 0'),
                  ('mae_prop_truth_positive', 'truth_positive', 'mae_proportion', 'MAE tỷ lệ, đáp án > 0'),
                  ('mae_prop_model_truth_zero', 'model_truth_zero', 'mae_proportion', 'MAE tỷ lệ, ô model đáp án = 0'))
PRIMARY = 'mae_prop'
KEY_PAIRS = (('same_day_mean', 'hron'), ('tsagris', 'hron'), ('micoda_other', 'mean_fill'),
             ('micoda', 'mean_fill'), ('micoda_other', 'micoda'), ('hron', 'mean_fill'))
TOLERANCE = 1e-12
STATUS = {'ok': 'completed', 'partial': 'partial', 'failed': 'failed'}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def na(value, digits=4):
    return 'NA' if value is None else f'{value:.{digits}f}'


# ---------- tier 0 ----------

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
                                       'hidden_cells', 'forced_zero_cells', 'model_cells')}))
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


# ---------- tier 1 ----------

def check_scores(scores, summary, rename, mask):
    """Regression check: long-file metrics equal summary.json errors; returns (values checked, max diff)."""
    back = {v: k for k, v in rename.items()}
    expected = {(rename.get(m, m), mask, int(seed)): r['errors'] for seed, s in summary['masked'].items()
                for m, r in s['models'].items() if r['status'] == 'ok' and r.get('errors')}
    if set(scores) != set(expected):
        raise ValueError(f'{mask}: runs in the long file {sorted(set(scores) ^ set(expected))} do not match '
                         'the completed runs of summary.json')
    checked, worst = 0, 0.
    for key, stored in expected.items():
        for group in GROUPS:
            for metric in ('cells', *METRICS):
                a, b = scores[key]['errors'][group][metric], stored[group][metric]
                if (a is None) != (b is None) or (a is not None and not abs(a - b) <= TOLERANCE):
                    raise ValueError(f'{back.get(key[0], key[0])} {key[1:]} {group} {metric}: '
                                     f'long file {a} != summary.json {b}')
                if a is not None:
                    checked, worst = checked + 1, max(worst, abs(a - b))
    return checked, worst


def run_metrics(score):
    if score is None:
        return dict.fromkeys(c for c, *_ in METRIC_COLUMNS)
    return {column: (score[key] if group is None else score['errors'][group][key])
            for column, group, key, _ in METRIC_COLUMNS}


def table_runs(status, scores):
    """{model: {seed: {'eligible', 'status', 'code', **metrics}}} for the masked runs of one mask."""
    runs = {}
    for r in status:
        if r['pass'] != 'masked':
            continue
        score = scores.get((r['model'], r['mask'], int(r['seed'])))
        eligible = r['status'] == 'completed' and r['coverage'] == 1. and score is not None
        runs.setdefault(r['model'], {})[r['seed']] = {
            'eligible': eligible, 'status': r['status'], 'code': r['error_code'],
            **run_metrics(score if eligible else None)}
    return runs


def why(run):
    return 'không có run' if run is None else f"{run['status']} ({run['code'] or 'coverage < 1'})"


def common_seeds(runs, models, seeds):
    """Seeds where every listed model counts, and {seed: reasons} for the others."""
    used, dropped = [], {}
    for seed in seeds:
        bad = [f'{m} {why(runs[m].get(seed))}' for m in models if not runs[m].get(seed, {}).get('eligible')]
        if bad:
            dropped[seed] = bad
        else:
            used.append(seed)
    return used, dropped


def mean_sd(values):
    values = np.array([v for v in values if v is not None], float)
    if not len(values):
        return None, None
    return float(values.mean()), (float(values.std(ddof=1)) if len(values) > 1 else None)


def ranking(runs, models, seeds, column=PRIMARY):
    """F07: order models by the mean over the seeds common to all of them (explicit None last)."""
    used, dropped = common_seeds(runs, models, seeds)
    means = {m: mean_sd([runs[m][s][column] for s in used]) for m in models}
    order = sorted(models, key=lambda m: (means[m][0] is None, means[m][0] if means[m][0] is not None else 0., m))
    return order, used, dropped, means


def pair(runs, a, b, seeds, column=PRIMARY):
    """A vs B over the seeds where both count: mean and sample sd of A - B, A wins (smaller error), ties."""
    used, dropped = common_seeds(runs, (a, b), seeds)
    diffs = [runs[a][s][column] - runs[b][s][column] for s in used
             if runs[a][s][column] is not None and runs[b][s][column] is not None]
    mean, sd = mean_sd(diffs)
    return {'a': a, 'b': b, 'metric': column, 'seeds_used': used, 'n': len(diffs), 'dropped': dropped,
            'mean_diff': mean, 'sd_diff': sd, 'wins': sum(d < 0 for d in diffs), 'ties': sum(d == 0 for d in diffs)}


# ---------- markdown ----------

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
           f"{r0['model_cells']} ô model.", '',
           '| Model | status | coverage | mã lỗi | thông điệp | hội tụ (MICoDa) |', '|---|---|---:|---|---|---|']
    for r in rows:
        message = (r['error_message'] or '-').replace('|', '/')[:110]
        out.append(f"| {r['model']} | {r['status']} | {na(r['coverage'], 3)} | {r['error_code'] or '-'} | "
                   f"{message} | {'-' if r['converged'] is None else r['converged']} |")
    return out


def main_markdown(runs, models, seeds):
    order, used, dropped, _ = ranking(runs, models, seeds)
    out = [f"Tập seed chung (mọi model completed, coverage = 1): {', '.join(used) or 'rỗng'}."]
    out += [f"Seed {s} bị loại: {'; '.join(r)}." for s, r in dropped.items()]
    out += ['', 'Sắp theo MAE tỷ lệ (nhóm ô model) trung bình trên tập seed chung. mean ± sd (ddof=1) qua seed; '
            'sd không phải khoảng tin cậy. **exploratory**.', '',
            '| Model | ' + ' | '.join(label for *_, label in METRIC_COLUMNS) + ' |',
            '|---|' + '---:|' * len(METRIC_COLUMNS)]
    for m in order:
        cells = []
        for column, _, key, _ in METRIC_COLUMNS:
            mean, sd = mean_sd([runs[m][s][column] for s in used])
            digits = 1 if key == 'mae_count' else 4
            cells.append('NA' if mean is None else f'{na(mean, digits)} ± {na(sd, digits)}')
        out.append(f'| {m} | ' + ' | '.join(cells) + ' |')
    return out, order


def pairs_markdown(pairs, title):
    out = [f'{title} (có điều kiện trên tập seed chung của từng cặp; thắng = MAE tỷ lệ nhóm model nhỏ hơn; '
           'Δ = A − B, mean ± sd ddof=1)', '',
           '| A vs B | n seed | A thắng | Δ MAE tỷ lệ | Seed loại (lý do) |', '|---|---:|---:|---|---|']
    for p in pairs:
        tie = f", hòa {p['ties']}" if p['ties'] else ''
        dropped = '; '.join(f"{s}: {', '.join(r)}" for s, r in p['dropped'].items()) or '-'
        delta = 'NA' if p['mean_diff'] is None else f"{p['mean_diff']:+.5f} ± {na(p['sd_diff'], 5)}"
        out.append(f"| {p['a']} vs {p['b']} | {p['n']} | {p['wins']}/{p['n']}{tie} | {delta} | {dropped} |")
    return out


def macro_markdown(runs, models, seeds):
    """Does the order change between cell-weighted and series-macro MAE (same common seeds)?"""
    cell, used, _, _ = ranking(runs, models, seeds, 'mae_prop')
    macro, _, _, _ = ranking(runs, models, seeds, 'series_macro_mae_prop')
    out = [f"Seed chung: {', '.join(used)}. Cell-weighted: {' < '.join(cell)}.",
           f"Series-macro: {' < '.join(macro)}."]
    flips = [(a, b) for a, b in itertools.combinations(cell, 2) if macro.index(a) > macro.index(b)]
    if not flips:
        out.append('Thứ hạng **không đổi**.')
    for a, b in flips:
        pc, pm = pair(runs, a, b, seeds, 'mae_prop'), pair(runs, a, b, seeds, 'series_macro_mae_prop')
        out.append(f"- Đổi chỗ {a} / {b}: cell-weighted {a} thắng {pc['wins']}/{pc['n']} "
                   f"(Δ {pc['mean_diff']:+.5f}); series-macro {a} thắng {pm['wins']}/{pm['n']} "
                   f"(Δ {pm['mean_diff']:+.5f}).")
    return out


# ---------- driver ----------

def load_mask(root, mask):
    seeds = SEEDS[mask]
    records = step1.validate_pair(root, mask, seeds)              # F2, raises on any mismatch
    status, scores, check, files = [], {}, [0, 0.], []
    for prefix, rename in PASSES:
        folder = root / f'{prefix}_{mask}'
        config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
        summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
        status += status_rows(folder.name, config, summary, rename)
        path = folder / f'predictions_{mask}.csv.gz'
        rows = read_predictions(path)
        own = {(rename.get(m, m), k, s): v for (m, k, s), v in score_predictions(rows).items()}
        checked, worst = check_scores(own, summary, rename, mask)
        check = [check[0] + checked, max(check[1], worst)]
        scores.update(own)
        files.append({'dir': folder.name, 'config_sha256': sha256(folder / 'config.json'),
                      'summary_sha256': sha256(folder / 'summary.json'), 'predictions': path.name,
                      'predictions_sha256': sha256(path), 'prediction_rows': len(rows),
                      'git': config.get('git')})
    for record, extra in zip(records, files):
        record.update(extra)
    return status, scores, records, {'values_checked': check[0], 'max_abs_diff': check[1]}


def compare_step1(root, records, mask, step1_runs):
    """Reuse check: same masks as the committed step 1 manifest; same errors as the step 1 runs if present."""
    manifest = json.loads((REPO / 'tv2_step1/results_manifest.json').read_text(encoding='utf-8'))[mask]
    out = {'masks_equal_step1_manifest': all(
        new['mask_sha256'][s] == old['mask_sha256'][s] for new, old in zip(records, manifest) for s in step1.SEEDS)}
    if step1_runs is None or not (step1_runs / f'main_{mask}').exists():
        out['errors_equal_step1_runs'] = 'step 1 runs not available'
        return out
    same, compared = True, 0
    for prefix, _ in PASSES:
        new, old = (json.loads((r / f'{prefix}_{mask}' / 'summary.json').read_text()) for r in (root, step1_runs))
        for seed in step1.SEEDS:
            for m, result in old['masked'][seed]['models'].items():
                compared += 1
                other = new['masked'][seed]['models'][m]
                same &= (result['status'], result.get('errors')) == (other['status'], other.get('errors'))
    out['errors_equal_step1_runs'] = same
    out['runs_compared'] = compared
    return out


def update_report(path, blocks):
    if not path.exists():
        return
    text = path.read_text(encoding='utf-8')
    for name, lines in blocks.items():
        start, end = f'<!-- auto:{name}:start -->', f'<!-- auto:{name}:end -->'
        if text.count(start) != 1 or text.count(end) != 1:
            continue
        head, rest = text.split(start)
        _, tail = rest.split(end)
        text = head + start + '\n' + '\n'.join(lines).rstrip() + '\n' + end + tail
    path.write_text(text, encoding='utf-8')


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)), restval='')
        writer.writeheader()
        writer.writerows([{k: 'NA' if v is None else v for k, v in r.items()} for r in rows])


def summarize(root, out_dir, step1_runs=None):
    all_status, run_rows, pair_rows, manifest = [], [], [], {}
    md = ['# Step 2: Tầng 0–1 (COVID đầy đủ, che 5%) — **exploratory**', '',
          'Sinh bởi `tier01.py`; không sửa tay. Metric chính: MAE tỷ lệ trên nhóm ô `model`, cell-weighted '
          'trong từng seed. Seed 0–4 đã được nhìn và chỉnh trước đó.', '']
    blocks = {}
    for mask in SEEDS:
        status, scores, records, check = load_mask(root, mask)
        manifest[mask] = {'runs': records, 'regression_check': check,
                          'step1_reuse': compare_step1(root, records, mask, step1_runs)}
        all_status += status
        runs = table_runs(status, scores)
        models = list(runs)
        for m in models:
            for s in SEEDS[mask]:
                run_rows.append({'mask': mask, 'model': m, 'seed': s, **runs[m][s]})
        main, order = main_markdown(runs, models, SEEDS[mask])
        pairs = [pair(runs, a, b, SEEDS[mask], column) for a, b in itertools.combinations(order, 2)
                 for column, *_ in METRIC_COLUMNS]
        pair_rows += [{'mask': mask, **{k: v for k, v in p.items() if k not in ('seeds_used', 'dropped')},
                       'seeds_used': ' '.join(p['seeds_used']),
                       'dropped': '; '.join(f"{s}: {', '.join(r)}" for s, r in p['dropped'].items())}
                      for p in pairs]
        primary = [p for p in pairs if p['metric'] == PRIMARY]
        key = [pair(runs, a, b, SEEDS[mask]) for a, b in KEY_PAIRS if a in runs and b in runs]
        label = 'chính' if mask == 'whole_feature' else 'sensitivity'
        rel = reliability(status)
        blocks[f'status_{mask}'] = status_markdown(status, rel)
        blocks[f'main_{mask}'] = main
        blocks[f'pairs_{mask}'] = pairs_markdown(key, f'**{mask}**, cặp chính')
        blocks[f'macro_{mask}'] = macro_markdown(runs, models, SEEDS[mask])
        md += [f'## {mask} ({label})', '', '### Tầng 0: reliability (mọi run đã lên lịch)', '',
               *blocks[f'status_{mask}'], '', '### Tầng 1: bảng metric', '', *main, '',
               '### Ghép cặp, mọi cặp (MAE tỷ lệ)', '', *pairs_markdown(primary, f'**{mask}**'), '',
               '### Series-macro so với cell-weighted', '', *blocks[f'macro_{mask}'], '',
               f"Regression check: {check['values_checked']} giá trị khớp summary.json, "
               f"sai khác lớn nhất {check['max_abs_diff']:.3g} (ngưỡng {TOLERANCE:g}).", '']
    blocks['seed109'] = seed_markdown(all_status, '109')
    md += ['## Seed 109 (whole_feature, ca thử chính sách failure)', '', *blocks['seed109'], '']
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / 'tier01_status.csv', all_status)
    write_csv(out_dir / 'tier01_runs.csv', run_rows)
    write_csv(out_dir / 'tier01_pairs.csv', pair_rows)
    (out_dir / 'tier01_results.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    (out_dir / 'tier01_manifest.json').write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n',
                                                  encoding='utf-8')
    update_report(out_dir / 'REPORT_vi.md', blocks)
    return manifest, all_status, run_rows, pair_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', type=Path, default=HERE / 'runs')
    parser.add_argument('--out', type=Path, default=HERE)
    parser.add_argument('--step1-runs', type=Path, default=REPO / 'tv2_step1/runs')
    args = parser.parse_args(argv)
    summarize(args.runs, args.out, args.step1_runs)


if __name__ == '__main__':
    main()
