"""Step 6: tier 3, test the tier 2 diagnostics on masked cells where the truth is known (baselines only).

    PYTHONPATH=nextgen/src python tv2_step6/tier3.py [--out DIR]

Use cases, thresholds and sensitivities: nextgen/src/longitudinal_impute/tier3_calibration.py, frozen
(committed) before this script read any output. Diagnostics: tier2_diagnostics.py (step 5), unchanged.
1. Reuse the masked runs of step 2 (main_*, micoda_other_*) and step 4 (residual_*): tier01.load_run
   (long file = summary.json to 1e-12), F2 (tier01 / step 1 validate_pair), and here: data sha256,
   schema, group map, clean tree and each model's table against its models file, the saved masks
   equal across the three run directories and equal to data.make_mask. Any mismatch stops.
2. Ten cases (2 masks x seeds 0-4; seed 109 is not used). E = hidden model cells, V = O minus H.
3. Tables 1-10; the auto:* blocks of REPORT_vi.md are regenerated. No ranking, no composite score.
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
sys.path.insert(0, str(REPO / 'nextgen/src'))
from longitudinal_impute.cells import read_predictions  # noqa: E402
from longitudinal_impute.data import load_csv, make_mask, read_schema  # noqa: E402
from longitudinal_impute.run import read_models  # noqa: E402
from longitudinal_impute import tier2_diagnostics as t2, tier3_calibration as t3  # noqa: E402


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, REPO / path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


tier01 = module('tier01', 'tv2_step2/tier01.py')
tier2 = module('tier2', 'tv2_step5/tier2.py')
MASKS, SEEDS, RATE = ('whole_feature', 'random_cell'), ('0', '1', '2', '3', '4'), .05
MODELS = t3.REAL + t3.DECOYS
# run directory prefix: (rename of tables, models file, tables used)
RUNS = {tier2.MAIN: ({}, 'models_default.toml', ('hron', 'tsagris', 'micoda', 'same_day_mean', *t3.DECOYS)),
        tier2.OTHER: ({'micoda': 'micoda_other'}, 'models_micoda_other.toml', ('micoda',)),
        tier2.RESIDUAL: ({}, 'models_residual_rule.toml', ('hron_cap',))}
DROPPED = {'tsagris_cap': 'trùng tsagris (`cap` ≡ `prop_other`, step 4–5)',
           'micoda_other_cap': 'gần trùng micoda_other (step 4–5)'}
LABEL = {'pass': 'đạt', 'fail': 'không đạt', 'insufficient': 'không đủ support'}
na, signed = tier01.na, tier2.signed


# ---------- 1. reuse checks ----------

def load_mask(data, mask):
    """Hidden masks, prediction tables {(model, seed): N x D}, tier 1 runs and provenance of one mask."""
    hidden, preds, runs, records, checked = {}, {}, {}, [], 0
    tier01.step1.validate_pair(REPO / 'tv2_step2/runs', mask, tier01.SEEDS[mask])      # F2
    col = {t: j for j, t in enumerate(data.targets)}
    for prefix, (rename, models_file, tables) in RUNS.items():
        folder = REPO / f'{prefix}_{mask}'
        config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
        expected = read_models(tier2.CONFIGS / models_file, list(tables))
        problems = [text for bad, text in (
            (config['data_sha256'] != tier2.sha256(tier2.DATA), 'data sha256'),
            (config['schema'] != read_schema(tier2.SCHEMA), 'schema'),
            ((config.get('group_map') or {}).get('sha256') != tier2.sha256(tier2.GROUP_MAP), 'group map'),
            ((config.get('git') or {}).get('uncommitted_changes') is not False, 'unclean tree'),
            (any(config['models'].get(t) != expected[t] for t in tables), f'model tables vs {models_file}'),
            (config['mask'] != mask or config['mask_rate'] != RATE, 'mask kind or rate')) if bad]
        for seed in SEEDS:
            saved = np.load(folder / f'mask_{mask}_seed{seed}.npy')
            hidden.setdefault(seed, saved)
            if not np.array_equal(saved, hidden[seed]) or not np.array_equal(saved, make_mask(data, mask, RATE,
                                                                                             int(seed))[0]):
                problems.append(f'mask seed {seed}')
        if problems:
            raise ValueError(f'{folder.relative_to(REPO)}: ' + ', '.join(problems) + ' differ')
        status, scores, record, (values, _) = tier01.load_run(folder, rename, mask)
        checked += values
        for model, seeds in tier01.table_runs(status, scores).items():
            if model in MODELS:
                runs[model] = {s: seeds[s] for s in SEEDS}
        for row in read_predictions(folder / f'predictions_{mask}.csv.gz'):
            name, seed = rename.get(row['model'], row['model']), str(row['seed'])
            if name in MODELS and seed in SEEDS:
                table = preds.setdefault((name, seed), np.full(data.counts.shape, np.nan))
                table[row['row_id'], col[row['target']]] = row['pred_count']
        records.append({'dir': folder.relative_to(REPO).as_posix(), 'config_sha256': record['config_sha256'],
                        'predictions_sha256': record['predictions_sha256'], 'git': (config.get('git') or {}).get(
                            'commit')})
    for (name, seed), table in preds.items():
        if not np.array_equal(np.isfinite(table), hidden[seed]):
            raise ValueError(f'{mask} {name} seed {seed}: long file does not cover exactly the hidden cells')
    masks = {s: hashlib.sha256(np.ascontiguousarray(h).tobytes()).hexdigest() for s, h in hidden.items()}
    return hidden, preds, runs, {'runs': records, 'mask_sha256': masks, 'regression_values_checked': checked}


# ---------- 2. one case ----------

def scalars(diags, ref, key, models, data, support, cells, values, truth_key='truth'):
    """Common targets, scored cells, per-model distance / raw values and tier 1 errors on those cells."""
    common = t3.common_targets(diags, ref, key, [truth_key, *t3.REAL])
    scored = t3.scored_cells(support, cells['eval'], ref, [data.targets.index(t) for t in common])
    out = {'targets': len(common), 'cells': int(scored.sum()), 'dist': {}, 'raw': {}, 'abs_raw': {}, 'err': {}}
    for m in models:
        ok = all(diags[m]['targets'][t][ref][key] is not None for t in common) and common
        out['dist'][m] = t3.scalar(diags[m], ref, key, common, diags[truth_key]) if ok else None
        out['raw'][m] = t3.scalar(diags[m], ref, key, common) if ok else None
        out['abs_raw'][m] = t3.scalar(diags[m], ref, key, common, absolute=True) if ok else None
        out['err'][m] = t3.errors_on(data.counts, values[m], data.totals, scored) if common else None
    return out


def mae(err, group='all'):
    return {m: (None if e is None else e[group]['mae_proportion']) for m, e in err.items()}


def u_pair(d, direction, group, use='dist', absolute=False):
    diag = {m: (None if v is None else abs(v) if absolute else v) for m, v in d[use].items()}
    return t3.pairwise(diag, mae(d['err'], group), list(t3.REAL), direction)


def one_case(data, facts, hidden, preds, runs, seed):
    cells = t2.calibration_cells(data, hidden)
    support = t2.local_support(facts, cells['eval'], cells['ref'])
    targets = list(data.targets)
    ok = [m for m in MODELS if runs.get(m, {}).get(seed, {}).get('eligible') and (m, seed) in preds]
    values = {m: t3.case_table(data.counts, hidden, preds[m, seed]) for m in ok}
    values['truth'] = t3.truth_table(data.counts, cells['eval'], cells['ref'])
    diags = {m: t2.diagnose(v, facts, cells['eval'], cells['ref'], support, targets) for m, v in values.items()}
    out = {'seed': seed, 'models_ok': ok, 'E': int(cells['eval'].sum()), 'H': int(hidden.sum()),
           'supported': int(sum(s['supported'].sum() for s in support.values())), 'diag': {}, 'u': {}}
    for m in ok + ['truth']:
        s = diags[m]['summary']
        out['diag'][m] = {f'{r}_{d}': s[f'{r}_{d}']['weighted'] for r in ('local', 'pooled') for d in ('d1', 'd2')}
        out['diag'][m]['d3'] = diags[m]['d3_cells']
    for ref in ('local', 'pooled'):
        d1 = scalars(diags, ref, 'd1', ok, data, support, cells, values)
        d2 = scalars(diags, ref, 'd2', ok, data, support, cells, values)
        full = {m: runs[m][seed]['mae_prop'] for m in ok}
        u = {'d1': d1, 'd2': d2, 'u1_under': u_pair(d1, 1, 'truth_positive'),
             'u1_over': u_pair(d1, -1, 'truth_zero'), 'u2': u_pair(d2, 1, 'all', absolute=True),
             'u2_raw': u_pair(d2, 1, 'all', 'raw'), 'u2_full': t3.pairwise(
                 {m: abs(v) if v is not None else None for m, v in d2['dist'].items()}, full, list(t3.REAL))}
        u['regret'] = t3.regret({m: abs(v) for m, v in d2['dist'].items() if v is not None}, mae(d2['err']),
                                list(t3.REAL))
        u['regret_raw'] = t3.regret({m: v for m, v in d2['raw'].items() if v is not None}, mae(d2['err']),
                                    list(t3.REAL))
        u['regret_full'] = t3.regret({m: abs(v) for m, v in d2['dist'].items() if v is not None}, full,
                                     list(t3.REAL))
        out['u'][ref] = u
    prop = {m: t3.d1_proportion(values[m], facts, cells['eval'], cells['ref'], support, targets)
            for m in [*t3.REAL, 'truth'] if m in values}
    if all(m in prop for m in t3.REAL):
        out['u']['prop'] = {'d1': (p := scalars(prop, 'local', 'd1', t3.REAL, data, support, cells, values))}
        out['u']['prop'].update(u1_under=u_pair(p, 1, 'truth_positive'), u1_over=u_pair(p, -1, 'truth_zero'))
    flagged = {m: t3.flags(values[m], facts, cells['eval'], cells['ref']) for m in ok}
    real = [m for m in t3.REAL if m in flagged]
    zero = data.counts == 0
    out['u3'] = t3.flag_precision(np.concatenate([zero[flagged[m][0]] for m in real]),
                                  np.concatenate([flagged[m][1][flagged[m][0]] for m in real]))
    out['u3_models'] = {m: t3.flag_precision(zero[p], f[p]) for m, (p, f) in flagged.items()}
    local = out['u']['local']
    scores = {'d1_dist': {m: None if v is None else abs(v) for m, v in local['d1']['dist'].items()},
              'd1_abs_raw': local['d1']['abs_raw'], 'd2_dist': {
                  m: None if v is None else abs(v) for m, v in local['d2']['dist'].items()},
              'd2_raw': local['d2']['raw'], 'd3_rate': {m: f[1].sum() / cells['eval'].sum()
                                                        for m, f in flagged.items()}}
    out['u4'] = {name: {d: t3.decoy_worse(s, d) for d in t3.DECOYS} for name, s in scores.items()}
    out['u4_scores'] = {name: {m: None if v is None else float(v) for m, v in s.items()} for name, s in scores.items()}
    return out, cells, support


# ---------- 3. tables ----------

U4_LABELS = {'d1_dist': '\\|D1 − D1_truth\\|', 'd1_abs_raw': '\\|D1\\| thô', 'd2_dist': '\\|D2 − D2_truth\\|',
             'd2_raw': 'D2 thô', 'd3_rate': 'D3: ô cờ / ô E'}


def ms(values, digits=4, fmt=None):
    """mean ± sd over the cases with a value; (n = k) when some case has none."""
    mean, sd = tier01.mean_sd(values)
    fmt = fmt or (lambda v, d: na(v, d))
    k = sum(v is not None for v in values)
    return 'NA' if mean is None else f'{fmt(mean, digits)} ± {na(sd, digits)}' + (
        f' (n = {k})' if k < len(values) else '')


def rate_cell(p):
    return 'NA' if p['rate'] is None else f"{p['rate']:.2f} ({p['match']}/{p['match'] + p['mismatch']}" + (
        f", hòa {p['tie']}" if p['tie'] else '') + (f", NA {p['na']}" if p['na'] else '') + ')'


def verdict_lines(cases, key, ref='local'):
    rates = {mask: [c['u'][ref][key]['rate'] if key in c['u'].get(ref, {}) else None for c in cases[mask]]
             for mask in MASKS}
    per, verdict = t3.rate_verdict(rates)
    text = '; '.join(f"{mask}: TB {na(v['mean'], 2)}, min {na(v['min'], 2)} ({v['cases']} ca)" for mask, v in per.items())
    return text, verdict


def support_table(cases):
    out = ['| Mask | Seed | ô H | ô E (model) | ô E có support R-local | coverage | target chung D1 / D2 (R-local) | '
           'ô chấm D1 / D2 |', '|---|---:|---:|---:|---:|---:|---|---|']
    for mask in MASKS:
        for c in cases[mask]:
            u = c['u']['local']
            out.append(f"| {mask} | {c['seed']} | {c['H']} | {c['E']} | {c['supported']} | {c['supported'] / c['E']:.3f} | "
                       f"{u['d1']['targets']} / {u['d2']['targets']} | {u['d1']['cells']} / {u['d2']['cells']} |")
    return out


def diag_table(cases):
    out = []
    for mask in MASKS:
        out += [f'**{mask}** (mean ± sd qua 5 seed; D1, D2 = TB trọng số qua target; khoảng cách = trên target chung)',
                '', '| Model | D1 local | D2 local | D1 pooled | D2 pooled | D3 ô | D1 − D1_truth (local) | '
                '\\|D2 − D2_truth\\| (local) |', '|---|' + '---:|' * 7]
        for m in ['truth', *MODELS]:
            got = [c for c in cases[mask] if m in c['diag']]
            col = [ms([c['diag'][m][k] for c in got], 3 if 'd1' in k else 4, signed if 'd1' in k else None)
                   for k in ('local_d1', 'local_d2', 'pooled_d1', 'pooled_d2')]
            dist1 = ms([c['u']['local']['d1']['dist'].get(m) for c in got], 3, signed) if m != 'truth' else '0'
            dist2 = ms([abs(v) if (v := c['u']['local']['d2']['dist'].get(m)) is not None else None for c in got]) \
                if m != 'truth' else '0'
            out.append(f"| {'**truth**' if m == 'truth' else m} | " + ' | '.join(col)
                       + f" | {ms([c['diag'][m]['d3'] for c in got], 0)} | {dist1} | {dist2} |")
        out.append('')
    return out


def truth_table(truth_rows, targets):
    out = ['Giá trị của chính đáp án trên E so với reference (mean qua 5 seed). D1_truth ≠ 0 và D2_truth > 0 '
           'là độ lệch sẵn có của reference khỏi phân phối thật của ô che.', '',
           '| Target | ' + ' | '.join(f'{m[:2]} {k}' for m in ('wf', 'rc') for k in (
               'D1t local', 'D2t local', 'D1t pooled', 'D2t pooled')) + ' |', '|---|' + '---:|' * 8]
    for t in targets:
        cols = []
        for mask in MASKS:
            for ref, key in (('local', 'd1'), ('local', 'd2'), ('pooled', 'd1'), ('pooled', 'd2')):
                v = [r[t][ref][key] for r in truth_rows[mask] if r[t][ref][key] is not None]
                cols.append(f'NA' if not v else (signed(np.mean(v)) if key == 'd1' else na(np.mean(v))))
        out.append(f'| {t} | ' + ' | '.join(cols) + ' |')
    return out


def u1_table(cases, ref='local', block='u1'):
    keys = ('u1_under', 'u1_over')
    out = ['| Mask | Seed | thiếu: khớp (khớp/quyết định) | thừa: khớp |', '|---|---:|---|---|']
    for mask in MASKS:
        for c in cases[mask]:
            u = c['u'].get(ref, {})
            out.append(f"| {mask} | {c['seed']} | " + ' | '.join(rate_cell(u[k]) if k in u else 'NA' for k in keys) + ' |')
    verdicts = {}
    for key, label in zip(keys, ('thiếu (D1 cao ↔ MAE ô đáp án > 0)', 'thừa (D1 thấp ↔ MAE ô đáp án = 0)')):
        text, verdicts[key] = verdict_lines(cases, key, ref)
        out.append(f'- U1 {label}: {text} → **{LABEL[verdicts[key]]}**.')
    return out, verdicts


def u2_table(cases, ref='local'):
    out = ['| Mask | Seed | khớp \\|D2 − D2t\\| | chọn → regret | khớp D2 thô | regret thô | khớp, MAE toàn ô model | '
           'regret, MAE toàn ô model |', '|---|---:|---|---|---|---:|---|---:|']
    regrets = []
    for mask in MASKS:
        for c in cases[mask]:
            u = c['u'][ref]
            regrets.append(u['regret'][1])
            out.append(f"| {mask} | {c['seed']} | {rate_cell(u['u2'])} | {u['regret'][0]} → {na(u['regret'][1])} | "
                       f"{rate_cell(u['u2_raw'])} | {na(u['regret_raw'][1])} | {rate_cell(u['u2_full'])} | "
                       f"{na(u['regret_full'][1])} |")
    text, verdict = verdict_lines(cases, 'u2', ref)
    good, enough = t3.regret_ok(regrets)
    final = verdict if verdict != 'pass' else ('pass' if enough else 'fail')
    out.append(f'- U2: {text}; regret ≤ {t3.REGRET_MAX} ở {good}/10 ca → **{LABEL[final]}**.')
    return out, final


def u3_table(cases):
    out = ['| Mask | Seed | ô dương bị cờ | tỷ lệ đáp án 0 khi bị cờ | ô dương không cờ | khi không cờ | chênh |',
           '|---|---:|---:|---:|---:|---:|---:|']
    gaps = []
    for mask in MASKS:
        for c in cases[mask]:
            u = c['u3']
            gaps.append(u['gap'])
            out.append(f"| {mask} | {c['seed']} | {u['flagged']} | {na(u['p_flagged'], 3)} | {u['unflagged']} | "
                       f"{na(u['p_unflagged'], 3)} | {signed(u['gap'])} |")
    good, verdict = t3.u3_verdict(gaps)
    out.append(f'- U3: chênh ≥ {t3.U3_GAP} ở {good}/10 ca → **{LABEL[verdict]}**.')
    return out, verdict


def u4_table(cases):
    names = list(cases[MASKS[0]][0]['u4'])
    out = ['Số ca diagnostic xếp đối thủ giả kém hơn **mọi** model thật (trên số ca có giá trị), R-local.', '',
           '| Đối thủ giả | Mask | ' + ' | '.join(U4_LABELS[n] for n in names) + ' |', '|---|---|' + ':---:|' * len(names)]
    for d in t3.DECOYS:
        for mask in MASKS:
            cells = []
            for n in names:
                got = [c['u4'][n][d] for c in cases[mask] if c['u4'][n][d] is not None]
                cells.append(f'{sum(got)}/{len(got)}' if got else 'NA')
            out.append(f'| {d} | {mask} | ' + ' | '.join(cells) + ' |')
    out += ['', 'Giá trị trung bình qua 5 seed (R-local, trên target chung của model thật):', '',
            '| Model | Mask | ' + ' | '.join(U4_LABELS[n] for n in names) + ' |', '|---|---|' + '---:|' * len(names)]
    for m in MODELS:
        for mask in MASKS:
            out.append(f'| {m} | {mask} | ' + ' | '.join(
                ms([c['u4_scores'][n].get(m) for c in cases[mask]], 4) for n in names) + ' |')
    return out


def transfer_table(data, facts, cases, masked_cells):
    native = t2.native_cells(data)
    native_support = t2.local_support(facts, native['eval'], native['ref'])
    n_label, _ = t2.n_strata(facts, native['eval'])
    groups = sorted(set(facts.groups))

    def profile(eval_cells, support):
        rows = np.nonzero(eval_cells)[0]
        cov = sum(s['supported'].sum() for s in support.values()) / max(eval_cells.sum(), 1)
        weights = eval_cells.sum(axis=1)
        share = lambda labels, v: float(weights[labels == v].sum() / weights.sum())
        return [cov] + [share(facts.groups, g) for g in groups] + [share(n_label, n) for n in t2.N_LABELS], rows

    head = ['coverage R-local', *groups, *(f'{n} (N cắt native)' for n in t2.N_LABELS)]
    out = ['| Tập ô | ' + ' | '.join(head) + ' |', '|---|' + '---:|' * len(head)]
    nat, _ = profile(native['eval'], native_support)
    out.append('| native M_m | ' + ' | '.join(f'{v:.3f}' for v in nat) + ' |')
    for mask in MASKS:
        prof = [profile(e, s)[0] for e, s in masked_cells[mask]]
        out.append(f'| masked E, {mask} (TB 5 seed) | ' + ' | '.join(f'{v:.3f}' for v in np.mean(prof, axis=0)) + ' |')
    out += ['', 'Tỷ lệ 0 theo target: native = tỷ lệ zero quan sát (ô O); masked = tỷ lệ đáp án 0 trên E (TB 5 seed).',
            '', '| Target | native O | wf E | rc E |', '|---|---:|---:|---:|']
    for j, t in enumerate(data.targets):
        obs = data.counts[native['ref'][:, j], j]
        cols = [f'{np.mean(obs == 0):.3f}']
        for mask in MASKS:
            v = [np.mean(data.counts[e[:, j], j] == 0) for e, _ in masked_cells[mask] if e[:, j].any()]
            cols.append(f'{np.mean(v):.3f}' if v else 'NA')
        out.append(f'| {t} | ' + ' | '.join(cols) + ' |')
    return out


def summary_table(v):
    rows = [('D1', 'U1 thiếu (cảnh báo điền thiếu)', v['u1']['u1_under']), ('D1', 'U1 thừa (cảnh báo điền thừa)',
                                                                         v['u1']['u1_over']),
            ('D2', 'U2 proxy xếp hạng MAE', v['u2']), ('D3', 'U3 cờ ngoài support', v['u3']),
            ('D1, ngưỡng c/N < 0.001 (sensitivity)', 'U1 thiếu', v['prop']['u1_under']),
            ('D1, ngưỡng c/N < 0.001 (sensitivity)', 'U1 thừa', v['prop']['u1_over']),
            ('D1 R-pooled (sensitivity)', 'U1 thiếu', v['pooled_u1']['u1_under']),
            ('D1 R-pooled (sensitivity)', 'U1 thừa', v['pooled_u1']['u1_over']),
            ('D2 R-pooled (sensitivity)', 'U2', v['pooled_u2'])]
    return ['| Diagnostic | Use case | Kết quả (10 ca, 2 mask) |', '|---|---|---|'] + [
        f'| {d} | {u} | **{LABEL[r]}** |' for d, u, r in rows] + [
        '| D1, D2, D3 | U4 đối thủ giả | báo kết quả, không có ngưỡng (bảng 7) |']


def sensitivity_lines(cases):
    out, v = ['**D1 với ngưỡng zero c/N < 0.001 (R-local)**', ''], {}
    lines, v['prop'] = u1_table(cases, 'prop')
    out += lines + ['', '**D1 với R-pooled**', '']
    lines, v['pooled_u1'] = u1_table(cases, 'pooled')
    out += lines + ['', '**D2 với R-pooled**', '']
    lines, v['pooled_u2'] = u2_table(cases, 'pooled')
    return out + lines, v


def summarize(out_dir):
    data = tier2.load_csv(tier2.DATA, tier2.SCHEMA)
    facts, targets = t2.row_facts(data), list(data.targets)
    cases, manifest, masked_cells, truth_rows = {}, {}, {}, {}
    for mask in MASKS:
        hidden, preds, runs, manifest[mask] = load_mask(data, mask)
        cases[mask], masked_cells[mask], truth_rows[mask] = [], [], []
        for seed in SEEDS:
            case, cells, support = one_case(data, facts, hidden[seed], preds, runs, seed)
            cases[mask].append(case)
            masked_cells[mask].append((cells['eval'], support))
            truth = t2.diagnose(t3.truth_table(data.counts, cells['eval'], cells['ref']), facts, cells['eval'],
                                cells['ref'], support, targets)
            truth_rows[mask].append(truth['targets'])
    blocks, verdicts = {}, {}
    blocks['support'] = support_table(cases)
    blocks['diag'] = diag_table(cases)
    blocks['truth'] = truth_table(truth_rows, targets)
    blocks['u1'], verdicts['u1'] = u1_table(cases)
    blocks['u2'], verdicts['u2'] = u2_table(cases)
    blocks['u3'], verdicts['u3'] = u3_table(cases)
    blocks['u4'] = u4_table(cases)
    blocks['sensitivity'], extra = sensitivity_lines(cases)
    verdicts.update(extra)
    blocks['transfer'] = transfer_table(data, facts, cases, masked_cells)
    blocks['summary'] = summary_table(verdicts)
    blocks['check'] = [f"{mask}: {manifest[mask]['regression_values_checked']} giá trị long file khớp summary.json "
                       f"(tier01.load_run); mask seed 0–4 giống nhau ở 3 thư mục run và bằng `make_mask`; "
                       f"model loại bỏ: {', '.join(f'{k} ({v})' for k, v in DROPPED.items())}."
                       for mask in MASKS]
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(cases, sort_keys=True, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    (out_dir / 'tier3_cases.json').write_text(payload + '\n', encoding='utf-8')
    md = ['# Step 6: Tầng 3, kiểm định diagnostic Tầng 2 trên ô che giả — **exploratory**', '',
          'Sinh bởi `tier3.py`; không sửa tay.', '']
    for name, title in (('summary', '10. Tóm tắt'), ('support', '1. Support của E'), ('diag', '2. D1, D2, D3'),
                        ('truth', '3. Độ lệch của reference'), ('u1', '4. U1'), ('u2', '5. U2'), ('u3', '6. U3'),
                        ('u4', '7. U4'), ('sensitivity', '8. Sensitivity'), ('transfer', '9. Chuyển giao'),
                        ('check', 'Kiểm tra')):
        md += [f'## {title}', '', *blocks[name], '']
    (out_dir / 'tier3_results.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    manifest.update(verdicts={k: v for k, v in verdicts.items()}, data_sha256=tier2.sha256(tier2.DATA),
                    calibration_sha256=tier2.sha256(REPO / 'nextgen/src/longitudinal_impute/tier3_calibration.py'),
                    diagnostics_sha256=tier2.sha256(REPO / 'nextgen/src/longitudinal_impute/tier2_diagnostics.py'),
                    cases_sha256=hashlib.sha256(payload.encode()).hexdigest())
    (out_dir / 'tier3_manifest.json').write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n',
                                                 encoding='utf-8')
    tier01.update_report(out_dir / 'REPORT_vi.md', blocks)
    return cases, verdicts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=HERE)
    summarize(parser.parse_args(argv).out)


if __name__ == '__main__':
    main()
