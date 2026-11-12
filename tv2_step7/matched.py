"""Step 7: a whole_feature mask matched to the support of the native missing cells; tiers 0, 1, 3 again.

    PYTHONPATH=nextgen/src python tv2_step7/matched.py target    # pi -> nextgen/configs/mask_target_covid.json
    PYTHONPATH=nextgen/src python tv2_step7/matched.py [summary] [--out DIR]

Mask: nextgen/src/longitudinal_impute/matched_mask.py (frozen before any mask was drawn). The N cut
points are read from the step 5 output (tv2_step5/tier2_results.md), not recomputed.
Declared before the runs (no definition of tiers 1-3 changes):
- Tier 0 / 1: tier01 on the three passes (7 models, micoda_other, the cap variants) of tv2_step7/runs,
  11 models, mask seeds 0-4; ranking on the seeds common to all; compared with whole_feature (step 2
  / 4 runs, same 11 models, same code).
- Tier 1 by stratum: MAE of c / N on the hidden model cells of a continent, or of an N group (step 5
  cuts), with evaluate.cell_errors, per model and seed; a stratum with < 30 such cells in a seed is
  NA in that seed; mean ± sd over seeds with a value; same_day_mean vs hron over seeds where both have one.
- Tier 3: the step 6 specification unchanged. Its 10 cases become whole_feature_matched (5) +
  random_cell (5, the step 6 cases, recomputed and checked equal to tv2_step6/tier3_cases.json), so
  the 'in each mask' and '>= 9 / 10 cases' rules keep their meaning. Side by side with step 6's
  verdicts (tv2_step6/tier3_manifest.json).
- Match (reported, not a gate): total variation between the strata shares of the hidden model cells
  and pi, per seed, for whole_feature (step 2 masks) and whole_feature_matched.
"""
import argparse
import csv
import importlib.util
import itertools
import json
from pathlib import Path
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / 'nextgen/src'))
from longitudinal_impute import matched_mask as mm, tier2_diagnostics as t2, tier3_calibration as t3  # noqa: E402
from longitudinal_impute.cells import read_predictions  # noqa: E402
from longitudinal_impute.evaluate import cell_errors, forced_zero  # noqa: E402

_spec = importlib.util.spec_from_file_location('tier3', REPO / 'tv2_step6/tier3.py')
tier3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tier3)
tier01, tier2 = tier3.tier01, tier3.tier2
na, signed, ms = tier01.na, tier2.signed, tier3.ms

MASK, WF, RC = 'whole_feature_matched', 'whole_feature', 'random_cell'
SEEDS = tier3.SEEDS
TARGET_FILE = REPO / 'nextgen/configs/mask_target_covid.json'
STEP5 = REPO / 'tv2_step5/tier2_results.md'
RUNS7 = {'tv2_step7/runs/main': tier3.RUNS[tier2.MAIN], 'tv2_step7/runs/micoda_other': tier3.RUNS[tier2.OTHER],
         'tv2_step7/runs/residual': tier3.RUNS[tier2.RESIDUAL]}
MODELS = ('hron', 'tsagris', 'micoda', 'micoda_other', 'same_day_mean', 'mean_fill', 'zero_fill', 'marginal_sampler',
          'hron_cap', 'tsagris_cap', 'micoda_other_cap')
DIRS = {WF: (('tv2_step2/runs/main_whole_feature', {}), ('tv2_step2/runs/micoda_other_whole_feature',
                                                          {'micoda': 'micoda_other'}),
             ('tv2_step4/runs/residual_whole_feature', {})),
        MASK: (('tv2_step7/runs/main_whole_feature_matched', {}),
               ('tv2_step7/runs/micoda_other_whole_feature_matched', {'micoda': 'micoda_other'}),
               ('tv2_step7/runs/residual_whole_feature_matched', {}))}
MIN_CELLS = 30
LABEL = tier3.LABEL


def step5_cuts():
    """N cut points as printed by step 5 (table 1): 'N1 ≤ q1 < N2 ≤ q2 < N3'."""
    found = re.findall(r'N1 ≤ ([0-9.e+-]+) < N2 ≤ ([0-9.e+-]+) < N3', STEP5.read_text(encoding='utf-8'))
    if len(found) != 1:
        raise ValueError(f'{STEP5}: expected one line with the N cut points, found {len(found)}')
    return [float(found[0][0]), float(found[0][1])]


def write_target():
    data = tier2.load_csv(tier2.DATA, tier2.SCHEMA)
    target = mm.native_target(data, step5_cuts())
    TARGET_FILE.write_text(json.dumps(target, indent=1, sort_keys=True) + '\n', encoding='utf-8')
    return target


# ---------- tier 0 / 1 ----------

def tier1(mask):
    """Status rows, {model: {seed: run}} and long rows of the 11 models on one mask."""
    status, scores, rows = [], {}, []
    for folder, rename in DIRS[mask]:
        own, own_scores, _, _ = tier01.load_run(REPO / folder, rename, mask)
        status += [r for r in own if r['model'] in MODELS and (r['seed'] in SEEDS or r['pass'] == 'native')]
        scores.update(own_scores)
        rows += [r for r in read_predictions(REPO / folder / f'predictions_{mask}.csv.gz')
                 if rename.get(r['model'], r['model']) in MODELS and str(r['seed']) in SEEDS
                 and not r.update(model=rename.get(r['model'], r['model']))]
    runs = {m: {s: v for s, v in seeds.items() if s in SEEDS} for m, seeds in tier01.table_runs(status, scores).items()}
    return status, runs, rows


def ranking_lines(runs):
    lines, orders = [], {}
    for mask in (WF, MASK):
        main, orders[mask] = tier01.main_markdown(runs[mask], list(MODELS), SEEDS)
        lines += [f'**{mask}**', '', *main, '', *tier01.pairs_markdown(
            [tier01.pair(runs[mask], a, b, SEEDS) for a, b in (*tier01.KEY_PAIRS, ('same_day_mean', 'hron_cap'))],
            f'**{mask}**, cặp chính'), '']
    a, b = orders[WF], orders[MASK]
    flips = [(x, y) for x, y in itertools.combinations(a, 2) if b.index(x) > b.index(y)]
    lines += [f'Thứ tự whole_feature: {" < ".join(a)}.', f'Thứ tự whole_feature_matched: {" < ".join(b)}.',
              'Cặp đổi chỗ: ' + ('; '.join(f'{x} / {y}' for x, y in flips) or 'không') + '.']
    return lines, orders


def stratum_scores(rows, cuts):
    """{(model, seed, kind, stratum): MAE of c / N on the model cells, None below MIN_CELLS cells}."""
    groups = {}
    for r in rows:
        if r['model_cell']:
            groups.setdefault((r['model'], str(r['seed'])), []).append(r)
    out = {}
    for (model, seed), cells in groups.items():
        truth, pred, n = (np.array([c[k] for c in cells]) for k in ('truth_count', 'pred_count', 'N'))
        labels = {'continent': np.array([c['group'] for c in cells]),
                  'n': np.array([s.split('|')[1] for s in mm.strata([''] * len(n), n, cuts)])}
        for kind, label in labels.items():
            for value in sorted(set(label)):
                sel = label == value
                out[model, seed, kind, value] = (cell_errors(truth[sel], pred[sel], n[sel], np.zeros(sel.sum(), bool))
                                                 ['all']['mae_proportion'] if sel.sum() >= MIN_CELLS else None)
    return out


def strata_lines(scores, strata):
    out = []
    for kind, labels in strata.items():
        out += [f'**MAE tỷ lệ (ô model) theo {"châu lục" if kind == "continent" else "nhóm N"}** — mean ± sd qua '
                f'seed có ≥ {MIN_CELLS} ô; n = số seed khi thiếu.', '',
                '| Model | Mask | ' + ' | '.join(labels) + ' |', '|---|---|' + '---:|' * len(labels)]
        for m in MODELS:
            for mask in (WF, MASK):
                out.append(f'| {m} | {tier3.SHORT[mask]} | ' + ' | '.join(
                    ms([scores[mask].get((m, s, kind, x)) for s in SEEDS]) for x in labels) + ' |')
        out += ['', f'same_day_mean − hron theo {"châu lục" if kind == "continent" else "nhóm N"} '
                '(Δ MAE, mean ± sd; số seed same_day_mean thắng / số seed có cả hai).', '',
                '| Tầng | wf | wfm |', '|---|---|---|']
        for x in labels:
            cols = []
            for mask in (WF, MASK):
                d = [scores[mask][('same_day_mean', s, kind, x)] - scores[mask][('hron', s, kind, x)] for s in SEEDS
                     if scores[mask].get(('same_day_mean', s, kind, x)) is not None
                     and scores[mask].get(('hron', s, kind, x)) is not None]
                cols.append('NA' if not d else f'{ms(d, 4, signed)} ({sum(v < 0 for v in d)}/{len(d)})')
            out.append(f'| {x} | ' + ' | '.join(cols) + ' |')
        out.append('')
    return out


# ---------- match (table 1) ----------

def hidden_masks(mask):
    folder = REPO / DIRS[mask][0][0]
    return {s: np.load(folder / f'mask_{mask}_seed{s}.npy') for s in SEEDS}


def match_lines(data, facts, target, masks, stats):
    labels = mm.strata(data.groups, data.totals, target['n_cuts'])
    keys = sorted(set(target['pi']) | set(labels))
    shares, tv, cover, by_target = {}, {}, {}, {}
    for mask, hidden in masks.items():
        e = {s: h & ~forced_zero(data, h) for s, h in hidden.items()}
        got = [mm.shares(labels, e[s]) for s in SEEDS]
        shares[mask] = {k: float(np.mean([g.get(k, 0.) for g in got])) for k in keys}
        tv[mask] = [mm.total_variation(g, target['pi']) for g in got]
        sup = [t2.local_support(facts, e[s], np.isfinite(data.counts) & ~hidden[s]) for s in SEEDS]
        cover[mask] = [sum(x['supported'].sum() for x in sp.values()) / e[s].sum() for s, sp in zip(SEEDS, sup)]
        by_target[mask] = np.mean([e[s].sum(axis=0) / e[s].sum() for s in SEEDS], axis=0)
    native = t2.native_cells(data)['eval']
    info = stats[MASK]
    out = [f"π: ô M_m native theo 18 tầng (châu lục × nhóm N, cắt {target['n_cuts']}; từ `mask_target_covid.json`). "
           'Phân bố đạt được: trên ô che thuộc nhóm model, TB 5 seed.', '',
           '| Tầng | π | wf | wfm | p_k (ô ứng viên) |', '|---|---:|---:|---:|---:|']
    for k in keys:
        out.append(f"| {k.replace('|', ' · ')} | {target['pi'].get(k, 0.):.3f} | {shares[WF][k]:.3f} | {shares[MASK][k]:.3f} | "
                   f"{info[0]['p'].get(k, float('nan')) if k in target['pi'] else float('nan'):.3f} |")
    out += ['', '| Mask | TV tới π (mean ± sd) | coverage R-local trên E | ứng viên bị guard bỏ / seed | '
            'cặp được che / seed | đạt rate |', '|---|---|---|---|---|---|']
    for mask in (WF, MASK):
        out.append(f"| {mask} | {ms(tv[mask], 3)} | {ms(cover[mask], 3)} | "
                   f"{', '.join(str(s['pairs_skipped']) for s in stats[mask])} | "
                   f"{', '.join(str(s['pairs_hidden']) for s in stats[mask])} | "
                   f"{', '.join(str(s.get('reached_goal', s['hidden_cells'] >= .05 * s['observed_cells'])) for s in stats[mask])} |")
    out += ['', f"Khối lượng π không với tới (tầng của π không có ô ứng viên nào): {info[0]['unreachable_mass']:.4f} "
            f"({', '.join(info[0]['unreachable_strata']) or 'không'}). Cặp ứng viên có w = 0: "
            f"{info[0]['zero_weight_pairs']} / {info[0]['candidate_pairs']}.", '',
            'Phân bố theo target (chỉ báo, không khớp): tỷ lệ ô của từng target.', '',
            '| Target | native M_m | wf E | wfm E |', '|---|---:|---:|---:|']
    nat = native.sum(axis=0) / native.sum()
    for j, t in enumerate(data.targets):
        out.append(f'| {t} | {nat[j]:.3f} | {by_target[WF][j]:.3f} | {by_target[MASK][j]:.3f} |')
    return out, {m: {'tv': tv[m], 'coverage': cover[m]} for m in tv}


def mask_stats(mask):
    summary = json.loads((REPO / DIRS[mask][0][0] / 'summary.json').read_text(encoding='utf-8'))
    return [summary['masked'][s]['mask'] for s in SEEDS]


# ---------- tier 3 ----------

def tier3_cases(data, facts, target):
    cases, cells, records = {}, {}, {}
    for mask, kwargs in ((MASK, {'runs_of': RUNS7, 'pair_root': REPO / 'tv2_step7/runs', 'seeds': SEEDS,
                                 'target': target}), (RC, {})):
        hidden, preds, runs, records[mask] = tier3.load_mask(data, mask, **kwargs)
        cases[mask], cells[mask] = [], []
        for s in SEEDS:
            case, c, support = tier3.one_case(data, facts, hidden[s], preds, runs, s)
            cases[mask].append(case)
            cells[mask].append((c['eval'], support))
    old = json.loads((REPO / 'tv2_step6/tier3_cases.json').read_text(encoding='utf-8'))[RC]
    if json.loads(json.dumps(cases[RC], sort_keys=True, default=lambda x: x.tolist() if hasattr(x, 'tolist')
                             else str(x))) != old:
        raise ValueError('random_cell cases differ from tv2_step6/tier3_cases.json')
    return cases, cells, records[MASK]


def tier3_lines(cases):
    blocks, v = {}, {}
    blocks['u1'], v['u1'] = tier3.u1_table(cases)
    blocks['u2'], v['u2'] = tier3.u2_table(cases)
    blocks['u3'], v['u3'] = tier3.u3_table(cases)
    blocks['sens'], extra = tier3.sensitivity_lines(cases)
    v.update(extra)
    blocks['u4'] = tier3.u4_table({MASK: cases[MASK]})
    old = json.loads((REPO / 'tv2_step6/tier3_manifest.json').read_text(encoding='utf-8'))['verdicts']
    rows = [('D1', 'U1 thiếu', 'u1', 'u1_under'), ('D1', 'U1 thừa', 'u1', 'u1_over'), ('D2', 'U2', 'u2', None),
            ('D3', 'U3', 'u3', None), ('D1 c/N < 0.001', 'U1 thiếu', 'prop', 'u1_under'),
            ('D1 c/N < 0.001', 'U1 thừa', 'prop', 'u1_over'), ('D1 R-pooled', 'U1 thiếu', 'pooled_u1', 'u1_under'),
            ('D1 R-pooled', 'U1 thừa', 'pooled_u1', 'u1_over'), ('D2 R-pooled', 'U2', 'pooled_u2', None)]
    pick = lambda d, k, sub: d[k][sub] if sub else d[k]
    blocks['summary'] = ['| Diagnostic | Use case | step 6: wf + rc | step 7: wfm + rc |', '|---|---|---|---|'] + [
        f'| {d} | {u} | {LABEL[pick(old, k, sub)]} | **{LABEL[pick(v, k, sub)]}** |' for d, u, k, sub in rows]
    return blocks, v


# ---------- driver ----------

def summarize(out_dir):
    data = tier2.load_csv(tier2.DATA, tier2.SCHEMA)
    facts = t2.row_facts(data)
    target = mm.read_target(TARGET_FILE)
    if target['n_cuts'] != step5_cuts():
        raise ValueError('the target file cut points differ from step 5')
    for folder, _ in DIRS[MASK]:
        saved = json.loads((REPO / folder / 'config.json').read_text(encoding='utf-8')).get('mask_target') or {}
        if saved.get('sha256') != tier2.sha256(TARGET_FILE):
            raise ValueError(f'{folder}: run made with another mask target')
    blocks, manifest = {}, {'mask_target_sha256': tier2.sha256(TARGET_FILE), 'target': target}
    stats = {m: mask_stats(m) for m in (WF, MASK)}
    blocks['match'], manifest['match'] = match_lines(data, facts, target, {m: hidden_masks(m) for m in (WF, MASK)},
                                                     stats)
    status, runs, scores = {}, {}, {}
    for mask in (WF, MASK):
        status[mask], runs[mask], rows = tier1(mask)
        scores[mask] = stratum_scores(rows, target['n_cuts'])
    blocks['status'] = tier01.status_markdown(status[MASK], tier01.reliability(status[MASK])) + [''] + [
        tier01.seed_markdown(status[MASK], s)[0] for s in SEEDS]
    blocks['status_seeds'] = [line for s in SEEDS for line in [''] + tier01.seed_markdown(status[MASK], s)]
    blocks['tier1'], orders = ranking_lines(runs)
    strata = {'continent': sorted(set(data.groups)), 'n': list(mm.N_LABELS)}
    blocks['strata'] = strata_lines(scores, strata)
    cases, cells, manifest['runs'] = tier3_cases(data, facts, target)
    t3_blocks, verdicts = tier3_lines(cases)
    blocks['tier3'] = t3_blocks['summary']
    blocks['tier3_detail'] = t3_blocks['u1'] + [''] + t3_blocks['u2'] + [''] + t3_blocks['u3'] + [''] + t3_blocks['u4']
    blocks['tier3_sens'] = t3_blocks['sens']
    wf_hidden = hidden_masks(WF)
    wf_cells = []
    for s in SEEDS:
        c = t2.calibration_cells(data, wf_hidden[s])
        wf_cells.append((c['eval'], t2.local_support(facts, c['eval'], c['ref'])))
    blocks['transfer'] = tier3.transfer_table(data, facts, None, {WF: wf_cells, RC: cells[RC], MASK: cells[MASK]})
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / 'matched_tier1_runs.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['mask', 'model', 'seed', 'eligible', 'status', *(c for c, *_ in tier01.METRIC_COLUMNS)])
        for mask in (WF, MASK):
            for m in MODELS:
                for s in SEEDS:
                    r = runs[mask][m][s]
                    writer.writerow([mask, m, s, r['eligible'], r['status'], *(
                        'NA' if r[c] is None else repr(r[c]) for c, *_ in tier01.METRIC_COLUMNS)])
    with (out_dir / 'matched_strata.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['mask', 'model', 'seed', 'kind', 'stratum', 'mae_prop_model'])
        for mask in (WF, MASK):
            for key in sorted(scores[mask]):
                v = scores[mask][key]
                writer.writerow([mask, *key, 'NA' if v is None else repr(v)])
    payload = json.dumps(cases[MASK], sort_keys=True, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    (out_dir / 'matched_cases.json').write_text(payload + '\n', encoding='utf-8')
    manifest.update(orders=orders, verdicts=verdicts, mask_stats=stats[MASK], cases_sha256=tier2.sha256(
        out_dir / 'matched_cases.json'))
    (out_dir / 'matched_manifest.json').write_text(json.dumps(manifest, indent=1, sort_keys=True, default=str) + '\n',
                                                   encoding='utf-8')
    md = ['# Step 7: mask khớp support của ô thiếu thật — **exploratory**', '', 'Sinh bởi `matched.py`; không sửa tay.', '']
    for name, title in (('match', '1. Khớp phân bố'), ('status', '2. Tầng 0'), ('status_seeds', '2b. Tầng 0 theo seed'), ('tier1', '3. Tầng 1'),
                        ('strata', '4. Tầng 1 theo tầng'), ('tier3', '5. Tầng 3'), ('tier3_detail', '5b. Tầng 3 chi tiết'),
                        ('tier3_sens', '5c. Tầng 3 sensitivity'), ('transfer', '6. Chuyển giao')):
        md += [f'## {title}', '', *blocks[name], '']
    (out_dir / 'matched_results.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    tier01.update_report(out_dir / 'REPORT_vi.md', blocks)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', nargs='?', choices=('summary', 'target'), default='summary')
    parser.add_argument('--out', type=Path, default=HERE)
    args = parser.parse_args(argv)
    if args.command == 'target':
        print(json.dumps(write_target(), indent=1))
    else:
        summarize(args.out)


if __name__ == '__main__':
    main()
