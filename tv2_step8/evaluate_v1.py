"""Evaluation v1 (step 8): one confirmation pass of the baselines on unseen mask seeds.

    PYTHONPATH=nextgen/src python tv2_step8/evaluate_v1.py check       # config + unused seeds, no side effect
    PYTHONPATH=nextgen/src python tv2_step8/evaluate_v1.py masks       # draw the masks, write masks_manifest.json
    PYTHONPATH=nextgen/src python tv2_step8/evaluate_v1.py run         # every pass x mask with run.py, once
    PYTHONPATH=nextgen/src python tv2_step8/evaluate_v1.py [evaluate] [--out DIR]

Everything is read from eval_v1.toml (v1.read_config refuses unknown or missing keys) and every
command first checks that no confirmation seed appears in steps 1-7. Scores come from the existing
code: tier01 (tier 0 / 1), tier2 (native diagnostics, must equal step 5 byte for byte), tier3 (step 6
use cases), matched (strata scores). Pairs add the exact sign test as a description of mask variation.
"""
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / 'nextgen/src'))
sys.path.insert(0, str(HERE))
from longitudinal_impute import matched_mask as mm, tier2_diagnostics as t2  # noqa: E402
from longitudinal_impute.cells import read_predictions  # noqa: E402
from longitudinal_impute.data import make_mask  # noqa: E402
import v1  # noqa: E402

_spec = importlib.util.spec_from_file_location('matched', REPO / 'tv2_step7/matched.py')
matched = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(matched)
tier3, tier2, tier01 = matched.tier3, matched.tier2, matched.tier01
na, signed, ms = tier01.na, tier2.signed, tier3.ms
CONFIG = HERE / 'eval_v1.toml'
RUNS_DIR = HERE / 'runs'
MASKS_MANIFEST = HERE / 'masks_manifest.json'
STEP5_FILES = ('tier2_targets.csv', 'tier2_strata.csv', 'tier2_support.csv', 'tier2_results.md', 'tier2_manifest.json')
SHORT = {'whole_feature': 'wf', 'whole_feature_matched': 'wfm', 'random_cell': 'rc'}
VERDICT_ROWS = (('D1', 'U1 thiếu', 'u1', 'u1_under'), ('D1', 'U1 thừa', 'u1', 'u1_over'),
                ('D2 (chỉ mô tả)', 'U2', 'u2', None), ('D3', 'U3', 'u3', None),
                ('D1 c/N < 0.001', 'U1 thiếu', 'prop', 'u1_under'), ('D1 c/N < 0.001', 'U1 thừa', 'prop', 'u1_over'),
                ('D1 R-pooled', 'U1 thiếu', 'pooled_u1', 'u1_under'), ('D1 R-pooled', 'U1 thừa', 'pooled_u1', 'u1_over'),
                ('D2 R-pooled (chỉ mô tả)', 'U2', 'pooled_u2', None))
H_USE = {'u1_under': ('u1', 'u1_under'), 'u1_over': ('u1', 'u1_over'), 'u2': ('u2', None), 'u3': ('u3', None)}


def setup(path=None):
    """Config, then the unused-seed check: both before any side effect."""
    config = v1.read_config(path or CONFIG)
    files = v1.check_unused(config['masks']['seeds'], REPO, config['masks']['previous'])
    return config, files


def masks_of(config):
    return [config['masks']['primary'], *config['masks']['sensitivity']]


def seeds_of(config):
    return [str(s) for s in config['masks']['seeds']]


def target_for(config, mask):
    return mm.read_target(REPO / config['masks']['target']) if mask == 'whole_feature_matched' else None


# ---------- masks and runs ----------

def draw_masks(config):
    if MASKS_MANIFEST.exists():
        raise FileExistsError(f'{MASKS_MANIFEST} exists: the confirmation masks are drawn once')
    data = tier2.load_csv(REPO / config['data']['data'], REPO / config['data']['schema'])
    out = {'config_sha256': tier2.sha256(CONFIG), 'masks': {}}
    for mask in masks_of(config):
        out['masks'][mask] = {}
        for seed in config['masks']['seeds']:
            hidden, stats = make_mask(data, mask, config['data']['rate'], seed, target_for(config, mask))
            out['masks'][mask][str(seed)] = {'sha256': tier01.step1.digest(hidden), 'hidden_cells': stats['hidden_cells']}
    MASKS_MANIFEST.write_text(json.dumps(out, indent=1, sort_keys=True) + '\n', encoding='utf-8')
    return out


def run_commands(config):
    """run.py argument lists, one per pass x mask (output tv2_step8/runs/<pass>_<mask>)."""
    commands = []
    for mask in masks_of(config):
        for name, run in config['runs'].items():
            args = ['--data', config['data']['data'], '--schema', config['data']['schema'], '--models',
                    run['models_file'], '--output', f'tv2_step8/runs/{name}_{mask}', '--mask', mask,
                    '--mask-rate', str(config['data']['rate']), '--seeds', *map(str, config['masks']['seeds']),
                    '--save-predictions', '--only', *run['tables']]
            if mask == 'whole_feature_matched':
                args += ['--mask-target', config['masks']['target']]
            commands.append(args)
    return commands


def run_all(config):
    manifest = json.loads(MASKS_MANIFEST.read_text(encoding='utf-8'))
    if manifest['config_sha256'] != tier2.sha256(CONFIG):
        raise ValueError('eval_v1.toml changed after the masks were drawn')
    RUNS_DIR.mkdir(exist_ok=True)
    env = {**os.environ, 'PYTHONPATH': str(REPO / 'nextgen/src')}
    for args in run_commands(config):
        log = REPO / (args[args.index('--output') + 1] + '.log')
        with log.open('w', encoding='utf-8') as stream:
            subprocess.run([sys.executable, '-m', 'longitudinal_impute.run', *args], cwd=REPO, env=env,
                           stdout=stream, stderr=subprocess.STDOUT, check=False)


# ---------- tier 0 / 1 ----------

def runs_of(config):
    return {f'tv2_step8/runs/{name}': (run['rename'], Path(run['models_file']).name, tuple(run['tables']))
            for name, run in config['runs'].items()}


def tier1(config, mask, data, cuts, masks_manifest):
    """Status rows, {model: {seed: run}}, strata scores and provenance of one mask (11 models)."""
    seeds = seeds_of(config)
    tier01.step1.validate_pair(RUNS_DIR, mask, seeds)                     # F2
    status, scores, strata, records = [], {}, {}, []
    for name, run in config['runs'].items():
        folder = RUNS_DIR / f'{name}_{mask}'
        for s in seeds:
            if tier01.step1.digest(np.load(folder / f'mask_{mask}_seed{s}.npy')) != masks_manifest[mask][s]['sha256']:
                raise ValueError(f'{folder.name}: mask seed {s} differs from masks_manifest.json')
        own, own_scores, record, _ = tier01.load_run(folder, run['rename'], mask)
        status += own
        scores.update(own_scores)
        rows = read_predictions(folder / f'predictions_{mask}.csv.gz')
        for r in rows:
            r['model'] = run['rename'].get(r['model'], r['model'])
        strata.update(matched.stratum_scores(rows, cuts))
        del rows
        records.append({k: record[k] for k in ('dir', 'config_sha256', 'predictions_sha256', 'git')})
    runs = {m: {s: by_seed[s] for s in seeds} for m, by_seed in tier01.table_runs(status, scores).items()}
    return status, runs, strata, records


def pair_rows(runs, pairs, seeds, metric='mae_prop'):
    out = []
    for a, b in pairs:
        p = tier01.pair(runs, a, b, seeds, metric)
        p['sign_p'] = v1.sign_test(p['wins'], p['n'] - p['wins'] - p['ties'])
        out.append(p)
    return out


def pair_lines(rows, seeds, title):
    out = [f'{title}. Δ = A − B theo seed; thắng = A nhỏ hơn; p = sign test chính xác hai phía (bỏ hòa), '
           'mô tả biến thiên mask trên bảng cố định, không phải CI theo nước.', '',
           '| A vs B | metric | n seed | A thắng | hòa | Δ mean ± sd | sign p | ghi chú |', '|---|---|---:|---:|---:|---|---:|---|']
    for p in rows:
        note = 'có điều kiện: ' + '; '.join(f"{s}: {', '.join(r)}" for s, r in p['dropped'].items()) if p['dropped'] else '-'
        delta = 'NA' if p['mean_diff'] is None else f"{p['mean_diff']:+.5f} ± {na(p['sd_diff'], 5)}"
        out.append(f"| {p['a']} vs {p['b']} | {p['metric']} | {p['n']} | {p['wins']} | {p['ties']} | {delta} | "
                   f"{na(p['sign_p'], 4)} | {note} |")
    return out


def strata_lines(config, strata, seeds, labels):
    models = [*config['models']['real'], *config['models']['decoys'], *config['models']['tier1_only']]
    masks = [m for m in masks_of(config) if m != 'random_cell']
    out = []
    for kind, values in labels.items():
        title = 'châu lục' if kind == 'continent' else 'nhóm N'
        out += [f'**MAE tỷ lệ (ô model) theo {title}**: mean ± sd qua seed có ≥ '
                f"{config['tier1']['min_stratum_cells']} ô (n khi thiếu).", '',
                '| Model | Mask | ' + ' | '.join(values) + ' |', '|---|---|' + '---:|' * len(values)]
        for m in models:
            for mask in masks:
                out.append(f'| {m} | {SHORT[mask]} | ' + ' | '.join(
                    ms([strata[mask].get((m, s, kind, x)) for s in seeds]) for x in values) + ' |')
        out += ['', f'same_day_mean − hron theo {title}: Δ mean ± sd (thắng / số seed, sign p).', '',
                '| Tầng | ' + ' | '.join(SHORT[m] for m in masks) + ' |', '|---|' + '---|' * len(masks)]
        for x in values:
            cols = []
            for mask in masks:
                d = [strata[mask][('same_day_mean', s, kind, x)] - strata[mask][('hron', s, kind, x)] for s in seeds
                     if strata[mask].get(('same_day_mean', s, kind, x)) is not None
                     and strata[mask].get(('hron', s, kind, x)) is not None]
                wins, losses = sum(v < 0 for v in d), sum(v > 0 for v in d)
                cols.append('NA' if not d else f'{ms(d, 4, signed)} ({wins}/{len(d)}, p {na(v1.sign_test(wins, losses), 3)})')
            out.append(f'| {x} | ' + ' | '.join(cols) + ' |')
        out.append('')
    return out


# ---------- tier 2 / 3 ----------

def tier2_check(out_dir):
    """Native diagnostics again through tier2, with the step 8 native CSVs as the rerun; compare with step 5."""
    view = RUNS_DIR / 'native_view'
    view.mkdir(exist_ok=True)
    for link, run in (('main', 'main_whole_feature'), ('other', 'micoda_other_whole_feature'),
                      ('residual', 'residual_whole_feature')):
        if not (view / link).exists():
            (view / link).symlink_to(RUNS_DIR / run)
    tier2.summarize(out_dir, view)
    return {name: (out_dir / name).read_bytes() == (REPO / 'tv2_step5' / name).read_bytes() for name in STEP5_FILES}


def tier3_mask(config, mask, data, facts):
    hidden, preds, runs, record = tier3.load_mask(data, mask, runs_of(config), RUNS_DIR, seeds_of(config),
                                                  target_for(config, mask))
    cases = [tier3.one_case(data, facts, hidden[s], preds, runs, s)[0] for s in seeds_of(config)]
    one = {mask: cases}
    blocks, verdicts = {}, {}
    blocks['u1'], verdicts['u1'] = tier3.u1_table(one)
    blocks['u2'], verdicts['u2'] = tier3.u2_table(one)
    blocks['u3'], verdicts['u3'] = tier3.u3_table(one)
    blocks['sens'], extra = tier3.sensitivity_lines(one)
    verdicts.update(extra)
    blocks['u4'] = tier3.u4_table(one)
    return cases, blocks, verdicts, record


def pick(verdicts, key, sub):
    return verdicts[key][sub] if sub else verdicts[key]


def verdict_lines(verdicts, masks):
    out = ['| Diagnostic | Use case | ' + ' | '.join(SHORT[m] for m in masks) + ' | cả ba |',
           '|---|---|' + '---|' * (len(masks) + 1)]
    for d, u, key, sub in VERDICT_ROWS:
        per = {m: pick(verdicts[m], key, sub) for m in masks}
        out.append(f'| {d} | {u} | ' + ' | '.join(tier3.LABEL[per[m]] for m in masks)
                   + f' | **{tier3.LABEL[v1.combine_masks(per)]}** |')
    return out


# ---------- hypotheses and the exploratory comparison ----------

def hypotheses(config, runs, verdicts, seeds):
    max_failed, real = config['tier0']['max_failed_seeds'], config['models']['real']
    out = {}
    for h in config['hypotheses']:
        out[h['id']] = {}
        for mask in masks_of(config):
            if h['kind'] == 'tier3':
                key, sub = H_USE[h['use_case']]
                failures = {m: len(v1.failed_seeds(runs[mask], m, seeds)) for m in real}
                out[h['id']][mask] = (v1.judge_tier3(h, pick(verdicts[mask], key, sub), failures, max_failed), {})
            else:
                out[h['id']][mask] = v1.judge_pairs(h, runs[mask], seeds, max_failed, tier01.pair)
    return out


def hypothesis_lines(config, judged):
    masks = masks_of(config)
    out = ['| ID | Giả thuyết | ' + ' | '.join(f'{SHORT[m]}' + (' (chính)' if m == masks[0] else '') for m in masks)
           + ' | Chi tiết ở mask chính |', '|---|---|' + '---|' * (len(masks) + 1)]
    for h in config['hypotheses']:
        cells = []
        for mask in masks:
            outcome = v1.OUTCOMES[judged[h['id']][mask][0]]
            cells.append(f'**{outcome}**' if mask == masks[0] else outcome)
        parts = judged[h['id']][masks[0]][1]
        detail = '; '.join(f"vs {b}: Δ {signed(p['mean_diff'], 4)}, thắng {p['wins']}/{p['n']}, p {na(p['sign_p'], 3)}"
                           if p else f'vs {b}: không đủ support' for b, (_, p) in parts.items()) or '-'
        out.append(f"| {h['id']} | {h['text']} | " + ' | '.join(cells) + f' | {detail} |')
    return out


def exploratory_runs():
    """Seeds 0-4 tier 1 runs of the committed step 2 / 4 / 7 tables, as {mask: {model: {seed: run}}}."""
    out = {}

    def add(mask, model, row):
        out.setdefault(mask, {}).setdefault(model, {})[row['seed']] = {
            'eligible': row['eligible'] == 'True', 'status': row['status'], 'code': row.get('code', ''), **{c: None if row.get(c) in (None, '', 'NA') else float(row[c])
                                                      for c, *_ in tier01.METRIC_COLUMNS}}
    for path, column in (('tv2_step7/matched_tier1_runs.csv', 'model'), ('tv2_step2/tier01_runs.csv', 'model'),
                         ('tv2_step4/residual_runs.csv', 'variant')):
        with (REPO / path).open(encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                name = row[column]
                if path.endswith('residual_runs.csv') and name not in ('hron_cap', 'tsagris_cap', 'micoda_other_cap'):
                    continue
                if row['mask'] == 'random_cell' or path.startswith('tv2_step7'):
                    add(row['mask'], name, row)
    return out


def exploratory_lines(config, runs, seeds, verdicts):
    old = exploratory_runs()
    pairs = list(dict.fromkeys((h['a'], b, h['metric']) for h in config['hypotheses'] if h['kind'] != 'tier3'
                               for b in h['b']))
    out = ['Exploratory: seed 0–4 (đã nhìn ở step 2–7). Xác nhận: seed 200–209. Δ = A − B, thắng = A nhỏ hơn.', '',
           '| A vs B | metric | ' + ' | '.join(f'{SHORT[m]} 0–4 | {SHORT[m]} 200–209' for m in masks_of(config)) + ' |',
           '|---|---|' + '---|---|' * len(masks_of(config))]
    for a, b, metric in pairs:
        cols = []
        for mask in masks_of(config):
            for source, use in ((old[mask], ('0', '1', '2', '3', '4')), (runs[mask], seeds)):
                p = tier01.pair(source, a, b, use, metric)
                cols.append('NA' if p['mean_diff'] is None else f"{p['mean_diff']:+.4f} ({p['wins']}/{p['n']})")
        out.append(f'| {a} vs {b} | {metric} | ' + ' | '.join(cols) + ' |')
    out += ['', 'MAE tỷ lệ trên seed chung của mọi model trong từng mask và dải seed (F07).', '',
            '| Model | ' + ' | '.join(f'{SHORT[m]} MAE 0–4 | {SHORT[m]} MAE 200–209' for m in masks_of(config)) + ' |',
            '|---|' + '---:|---:|' * len(masks_of(config))]
    models = [*config['models']['real'], *config['models']['decoys'], *config['models']['tier1_only']]
    for m in models:
        cols = []
        for mask in masks_of(config):
            for source, use in ((old[mask], ('0', '1', '2', '3', '4')), (runs[mask], seeds)):
                common = tier01.common_seeds(source, models, use)[0]
                cols.append(ms([source[m][s]['mae_prop'] for s in common]) + f' ({len(common)})')
        out.append(f'| {m} | ' + ' | '.join(cols) + ' |')
    step6 = json.loads((REPO / 'tv2_step6/tier3_manifest.json').read_text(encoding='utf-8'))['verdicts']
    step7 = json.loads((REPO / 'tv2_step7/matched_manifest.json').read_text(encoding='utf-8'))['verdicts']
    out += ['', 'Tầng 3: exploratory gộp 10 ca (step 6: wf + rc; step 7: wfm + rc), v1 theo từng mask 10 ca.', '',
            '| Diagnostic | Use case | step 6 wf+rc | step 7 wfm+rc | ' + ' | '.join(f'v1 {SHORT[m]}' for m in masks_of(config))
            + ' |', '|---|---|---|---|' + '---|' * len(masks_of(config))]
    for d, u, key, sub in VERDICT_ROWS:
        out.append(f'| {d} | {u} | {tier3.LABEL[pick(step6, key, sub)]} | {tier3.LABEL[pick(step7, key, sub)]} | '
                   + ' | '.join(tier3.LABEL[pick(verdicts[m], key, sub)] for m in masks_of(config)) + ' |')
    return out


# ---------- driver ----------

def evaluate(config, seed_files, out_dir):
    seeds, masks = seeds_of(config), masks_of(config)
    masks_manifest = json.loads(MASKS_MANIFEST.read_text(encoding='utf-8'))
    if masks_manifest['config_sha256'] != tier2.sha256(CONFIG):
        raise ValueError('eval_v1.toml changed after the masks were drawn')
    data = tier2.load_csv(REPO / config['data']['data'], REPO / config['data']['schema'])
    facts, cuts = t2.row_facts(data), matched.step5_cuts()
    blocks, runs, strata, status, verdicts, records, cases = {}, {}, {}, {}, {}, {}, {}
    models = [*config['models']['real'], *config['models']['decoys'], *config['models']['tier1_only']]
    for mask in masks:
        status[mask], runs[mask], strata[mask], records[mask] = tier1(config, mask, data, cuts, masks_manifest['masks'])
        cases[mask], t3_blocks, verdicts[mask], _ = tier3_mask(config, mask, data, facts)
        blocks[f'tier3_{SHORT[mask]}'] = (t3_blocks['u1'] + [''] + t3_blocks['u2'] + [''] + t3_blocks['u3'] + ['']
                                          + t3_blocks['u4'] + [''] + t3_blocks['sens'])
    all_status = [r for mask in masks for r in status[mask]]
    blocks['status'] = tier01.status_markdown(all_status, tier01.reliability(all_status))
    blocks['tier1'], pairs_csv = [], []
    for mask in masks:
        main, _ = tier01.main_markdown(runs[mask], models, seeds)
        rows = pair_rows(runs[mask], config['tier1']['pairs'], seeds) + pair_rows(
            runs[mask], [('tsagris', 'hron')], seeds, 'mse_prop')
        blocks['tier1'] += [f'**{mask}**', '', *main, '']
        blocks.setdefault('pairs', []).extend(pair_lines(rows, seeds, f'**{mask}**') + [''])
        pairs_csv += [{'mask': mask, **{k: v for k, v in p.items() if k not in ('seeds_used', 'dropped')}} for p in rows]
    labels = {'continent': sorted(set(data.groups)), 'n': list(mm.N_LABELS)}
    blocks['strata'] = strata_lines(config, strata, seeds, labels)
    same = tier2_check(out_dir / 'tier2')
    blocks['tier2'] = ['Lượt native chạy lại qua `evaluate_v1.py` (`tier2.summarize`, CSV native của step 8 làm lượt '
                       'chạy lại); so từng byte với output step 5:', '',
                       '| File | giống step 5 |', '|---|:---:|'] + [f'| {k} | {v} |' for k, v in same.items()]
    blocks['tier3'] = verdict_lines(verdicts, masks)
    judged = hypotheses(config, runs, verdicts, seeds)
    blocks['hypotheses'] = hypothesis_lines(config, judged)
    blocks['exploratory'] = exploratory_lines(config, runs, seeds, verdicts)
    blocks['checks'] = [f'Seed xác nhận {seeds[0]}–{seeds[-1]}: không xuất hiện trong {len(seed_files)} file '
                        'config / manifest của step 1–7 (`v1.check_unused`).',
                        f"Mask: {len(masks) * len(seeds)} mask đã lưu khớp `masks_manifest.json`; F2 đạt ở cả ba mask; "
                        'long file khớp `summary.json` (tier01.load_run).']
    out_dir.mkdir(parents=True, exist_ok=True)
    tier01.write_csv(out_dir / 'v1_pairs.csv', pairs_csv)
    payload = json.dumps(cases, sort_keys=True, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    (out_dir / 'v1_cases.json').write_text(payload + '\n', encoding='utf-8')
    manifest = {'config_sha256': tier2.sha256(CONFIG), 'masks_manifest_sha256': tier2.sha256(MASKS_MANIFEST),
                'seed_check_files': seed_files, 'runs': records, 'tier2_equal_step5': same, 'verdicts': verdicts,
                'hypotheses': {k: {m: o for m, (o, _) in v.items()} for k, v in judged.items()},
                'cases_sha256': tier2.sha256(out_dir / 'v1_cases.json')}
    (out_dir / 'v1_manifest.json').write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n', encoding='utf-8')
    md = ['# Step 8: evaluation v1, lượt xác nhận (seed 200–209)', '', 'Sinh bởi `evaluate_v1.py`; không sửa tay.', '']
    for name, title in (('hypotheses', '7. Giả thuyết'), ('checks', 'Kiểm tra'), ('status', '1. Tầng 0'),
                        ('tier1', '2. Tầng 1'), ('pairs', '3. Ghép cặp'), ('strata', '4. Tầng 1 theo tầng'),
                        ('tier2', '5. Tầng 2'), ('tier3', '6. Tầng 3'), *((f'tier3_{SHORT[m]}', f'6. Tầng 3 chi tiết {m}')
                                                                       for m in masks),
                        ('exploratory', '8. So với exploratory')):
        md += [f'## {title}', '', *blocks[name], '']
    (out_dir / 'v1_results.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    tier01.update_report(out_dir / 'REPORT_vi.md', blocks)
    return judged


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', nargs='?', choices=('check', 'masks', 'run', 'evaluate'), default='evaluate')
    parser.add_argument('--out', type=Path, default=HERE)
    args = parser.parse_args(argv)
    config, files = setup()
    if args.command == 'check':
        print(f'ok: {len(files)} files of steps 1-7 read, no confirmation seed among them')
    elif args.command == 'masks':
        draw_masks(config)
    elif args.command == 'run':
        run_all(config)
    else:
        evaluate(config, files, args.out)


if __name__ == '__main__':
    main()
