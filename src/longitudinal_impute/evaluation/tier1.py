"""Tier 1: errors on masked cells with known truth, common-seed pairing (F07), sign test.

Every metric is recomputed from predictions_<mask>.csv.gz (score_predictions) and must equal the
errors in summary.json to 1e-12 (check_scores). A model mean is taken over the seeds where every model
counts; each pair is compared over the seeds where both count. The sign test is a description of
mask variation on a fixed table, not a confidence interval over countries.
summarize() writes the tier 0 / 1 tables of a set of exploratory runs (main_<mask>, micoda_other_<mask>).
"""
import itertools
import json

from math import comb

import numpy as np

from .report import mean_sd, na, sha256, update_report, write_csv
from .metrics import GROUPS, METRICS, cell_errors, read_predictions, score_predictions
from .masks import strata
from .tier0 import reliability, seed_markdown, status_markdown, status_rows, validate_pair


SEEDS = {'whole_feature': ('0', '1', '2', '3', '4', '109'), 'random_cell': ('0', '1', '2', '3', '4')}
UNDECLARED = ('109',)
PASSES = (('main', {}), ('micoda_other', {'micoda': 'micoda_other'}))
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


def main_markdown(runs, models, seeds, undeclared=()):
    order, used, dropped, _ = ranking(runs, models, seeds)
    out = [f"Tập seed chung (mọi model completed, coverage = 1): {', '.join(used) or 'rỗng'}."]
    if undeclared:
        out.append(f"Seed không khai báo trước ({', '.join(undeclared)}): chỉ có trong bảng trạng thái và "
                   'reliability, không vào bảng metric hay hiệu ghép cặp.')
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


def load_run(folder, rename, mask):
    """One run directory: status rows, scores from its long file (checked against summary.json), record."""
    config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    status = status_rows(folder.name, config, summary, rename)
    path = folder / f'predictions_{mask}.csv.gz'
    rows = read_predictions(path)
    scores = {(rename.get(m, m), k, s): v for (m, k, s), v in score_predictions(rows).items()}
    check = check_scores(scores, summary, rename, mask)
    record = {'dir': folder.name, 'config_sha256': sha256(folder / 'config.json'),
              'summary_sha256': sha256(folder / 'summary.json'), 'predictions': path.name,
              'predictions_sha256': sha256(path), 'prediction_rows': len(rows), 'git': config.get('git'),
              'resolved_models': config.get('resolved_models')}
    return status, scores, record, check


def load_mask(root, mask):
    records = validate_pair(root, mask, SEEDS[mask])        # F2, raises on any mismatch
    status, scores, check = [], {}, [0, 0.]
    for record, (prefix, rename) in zip(records, PASSES):
        own_status, own, extra, (checked, worst) = load_run(root / f'{prefix}_{mask}', rename, mask)
        status += own_status
        scores.update(own)
        record.update(extra)
        check = [check[0] + checked, max(check[1], worst)]
    return status, scores, records, {'values_checked': check[0], 'max_abs_diff': check[1]}


def summarize(root, out_dir):
    all_status, run_rows, pair_rows, manifest = [], [], [], {}
    md = ['# Step 2: Tầng 0–1 (COVID đầy đủ, che 5%) — **exploratory**', '',
          'Sinh bởi `evaluation.tier1.summarize`; không sửa tay. Metric chính: MAE tỷ lệ trên nhóm ô `model`, cell-weighted '
          'trong từng seed. Seed 0–4 đã được nhìn và chỉnh trước đó.', '']
    blocks = {}
    for mask in SEEDS:
        status, scores, records, check = load_mask(root, mask)
        manifest[mask] = {'runs': records, 'regression_check': check}
        all_status += status
        runs = table_runs(status, scores)
        seeds = tuple(s for s in SEEDS[mask] if s not in UNDECLARED)
        models = list(runs)
        for m in models:
            for s in seeds:
                run_rows.append({'mask': mask, 'model': m, 'seed': s, **runs[m][s]})
        main, order = main_markdown(runs, models, seeds, [s for s in SEEDS[mask] if s in UNDECLARED])
        pairs = [pair(runs, a, b, seeds, column) for a, b in itertools.combinations(order, 2)
                 for column, *_ in METRIC_COLUMNS]
        pair_rows += [{'mask': mask, **{k: v for k, v in p.items() if k not in ('seeds_used', 'dropped')},
                       'seeds_used': ' '.join(p['seeds_used']),
                       'dropped': '; '.join(f"{s}: {', '.join(r)}" for s, r in p['dropped'].items())}
                      for p in pairs]
        primary = [p for p in pairs if p['metric'] == PRIMARY]
        key = [pair(runs, a, b, seeds) for a, b in KEY_PAIRS if a in runs and b in runs]
        label = 'chính' if mask == 'whole_feature' else 'sensitivity'
        rel = reliability(status)
        if mask == 'whole_feature':
            blocks['seed109'] = seed_markdown(status, '109')
        blocks[f'status_{mask}'] = status_markdown(status, rel)
        blocks[f'main_{mask}'] = main
        blocks[f'pairs_{mask}'] = pairs_markdown(key, f'**{mask}**, cặp chính')
        blocks[f'macro_{mask}'] = macro_markdown(runs, models, seeds)
        md += [f'## {mask} ({label})', '', '### Tầng 0: reliability (mọi run đã lên lịch)', '',
               *blocks[f'status_{mask}'], '', '### Tầng 1: bảng metric', '', *main, '',
               '### Ghép cặp, mọi cặp (MAE tỷ lệ)', '', *pairs_markdown(primary, f'**{mask}**'), '',
               '### Series-macro so với cell-weighted', '', *blocks[f'macro_{mask}'], '',
               f"Regression check: {check['values_checked']} giá trị khớp summary.json, "
               f"sai khác lớn nhất {check['max_abs_diff']:.3g} (ngưỡng {TOLERANCE:g}).", '']
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



def sign_test(wins, losses):
    """Exact two-sided binomial sign test, p = 0.5, ties already dropped; None when n = 0."""
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    return min(1., 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)



STRATUM_MIN_CELLS = 30


def stratum_scores(rows, cuts):
    """{(model, seed, kind, stratum): MAE of c / N on the model cells, None below STRATUM_MIN_CELLS cells}."""
    groups = {}
    for r in rows:
        if r['model_cell']:
            groups.setdefault((r['model'], str(r['seed'])), []).append(r)
    out = {}
    for (model, seed), cells in groups.items():
        truth, pred, n = (np.array([c[k] for c in cells]) for k in ('truth_count', 'pred_count', 'N'))
        labels = {'continent': np.array([c['group'] for c in cells]),
                  'n': np.array([s.split('|')[1] for s in strata([''] * len(n), n, cuts)])}
        for kind, label in labels.items():
            for value in sorted(set(label)):
                sel = label == value
                out[model, seed, kind, value] = (cell_errors(truth[sel], pred[sel], n[sel], np.zeros(sel.sum(), bool))
                                                 ['all']['mae_proportion'] if sel.sum() >= STRATUM_MIN_CELLS else None)
    return out
