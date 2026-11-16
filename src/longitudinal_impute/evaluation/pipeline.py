"""Evaluation v1 pipeline: the commands check, masks, run and evaluate (CLI: __main__.py).

Everything is read from the frozen configuration (protocol.read_config refuses unknown or missing keys)
and every command first checks that no confirmation seed appears in the used-seed snapshot. Scores come
from tier1 (tier 0 / 1), tier2 (native diagnostics, must reproduce the step 5 reference tables), tier3
(use cases) and tier1.stratum_scores. Pairs add the exact sign test as a description of mask variation.

runs_dir holds the confirmation runs <pass>_<mask> (pass: main, micoda_other, residual). The native
tier 2 check also reads the exploratory runs (seeds 0-4, same pass names), from one or more directories.
The tier 2 check compares the four step 5 tables byte for byte and, for tier2_manifest.json, the result
part: the run directory of each model and the hash of the code file are provenance (they change when the
runs or the code move) and are left out on both sides (MANIFEST_PROVENANCE).
"""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from .report import digest, ms, na, sha256, signed, update_report, write_csv
from .metrics import read_predictions
from .masks import N_LABELS, read_target, step5_cuts
from .tier0 import failed_seeds, reliability, status_markdown, validate_pair
from .tier1 import METRIC_COLUMNS, common_seeds, load_run, main_markdown, pair, sign_test, stratum_scores, table_runs
from .tier3 import LABEL, load_mask, one_case, sensitivity_lines, u1_table, u2_table, u3_table, u4_table
from .protocol import OUTCOMES, check_unused, combine_masks, frozen_sha256, judge_pairs, judge_tier3, read_config
from ..data import load_csv, make_mask
from . import tier2
from .tier2 import EXPLORATORY, row_facts
from .report import ROOT
MANIFEST_PROVENANCE = ('diagnostics_sha256', 'dir')



STEP5_FILES = ('tier2_targets.csv', 'tier2_strata.csv', 'tier2_support.csv', 'tier2_results.md', 'tier2_manifest.json')
SHORT = {'whole_feature': 'wf', 'whole_feature_matched': 'wfm', 'random_cell': 'rc'}
VERDICT_ROWS = (('D1', 'U1 thiếu', 'u1', 'u1_under'), ('D1', 'U1 thừa', 'u1', 'u1_over'),
                ('D2 (chỉ mô tả)', 'U2', 'u2', None), ('D3', 'U3', 'u3', None),
                ('D1 c/N < 0.001', 'U1 thiếu', 'prop', 'u1_under'), ('D1 c/N < 0.001', 'U1 thừa', 'prop', 'u1_over'),
                ('D1 R-pooled', 'U1 thiếu', 'pooled_u1', 'u1_under'), ('D1 R-pooled', 'U1 thừa', 'pooled_u1', 'u1_over'),
                ('D2 R-pooled (chỉ mô tả)', 'U2', 'pooled_u2', None))
H_USE = {'u1_under': ('u1', 'u1_under'), 'u1_over': ('u1', 'u1_over'), 'u2': ('u2', None), 'u3': ('u3', None)}


def setup(config_path):
    """Config, then the unused-seed check: both before any side effect."""
    config = read_config(config_path)
    files = check_unused(config['masks']['seeds'], [ROOT / p for p in config['masks']['previous']])
    return config, files


def masks_of(config):
    return [config['masks']['primary'], *config['masks']['sensitivity']]


def seeds_of(config):
    return [str(s) for s in config['masks']['seeds']]


def target_for(config, mask):
    return read_target(ROOT / config['masks']['target']) if mask == 'whole_feature_matched' else None


def masks_manifest_path(config_path):
    return Path(config_path).with_name('masks_manifest.json')


def draw_masks(config, config_path):
    manifest = masks_manifest_path(config_path)
    if manifest.exists():
        raise FileExistsError(f'{manifest} exists: the confirmation masks are drawn once')
    data = load_csv(ROOT / config['data']['data'], ROOT / config['data']['schema'])
    out = {'config_sha256': frozen_sha256(config_path), 'masks': {}}
    for mask in masks_of(config):
        out['masks'][mask] = {}
        for seed in config['masks']['seeds']:
            hidden, stats = make_mask(data, mask, config['data']['rate'], seed, target_for(config, mask))
            out['masks'][mask][str(seed)] = {'sha256': digest(hidden), 'hidden_cells': stats['hidden_cells']}
    manifest.write_text(json.dumps(out, indent=1, sort_keys=True) + '\n', encoding='utf-8')
    return out


def run_commands(config, runs_dir):
    """run.py argument lists, one per pass x mask (output <runs_dir>/<pass>_<mask>)."""
    commands = []
    for mask in masks_of(config):
        for name, run in config['runs'].items():
            args = ['--data', config['data']['data'], '--schema', config['data']['schema'], '--models',
                    run['models_file'], '--output', str(Path(runs_dir) / f'{name}_{mask}'), '--mask', mask,
                    '--mask-rate', str(config['data']['rate']), '--seeds', *map(str, config['masks']['seeds']),
                    '--save-predictions', '--only', *run['tables']]
            if mask == 'whole_feature_matched':
                args += ['--mask-target', config['masks']['target']]
            commands.append(args)
    return commands


def run_all(config, config_path, runs_dir):
    manifest = json.loads(masks_manifest_path(config_path).read_text(encoding='utf-8'))
    if manifest['config_sha256'] != frozen_sha256(config_path):
        raise ValueError('eval_v1.toml changed after the masks were drawn')
    Path(runs_dir).mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src')}
    for args in run_commands(config, runs_dir):
        log = Path(args[args.index('--output') + 1] + '.log')
        with log.open('w', encoding='utf-8') as stream:
            subprocess.run([sys.executable, '-m', 'longitudinal_impute.run', *args], cwd=ROOT, env=env,
                           stdout=stream, stderr=subprocess.STDOUT, check=False)


def runs_of(config):
    return {name: (run['rename'], Path(run['models_file']).name, tuple(run['tables']))
            for name, run in config['runs'].items()}


def tier1_mask(config, mask, data, cuts, masks_manifest, runs_dir):
    """Status rows, {model: {seed: run}}, strata scores and provenance of one mask (11 models)."""
    seeds = seeds_of(config)
    validate_pair(runs_dir, mask, seeds)                     # F2
    status, scores, strata, records = [], {}, {}, []
    for name, run in config['runs'].items():
        folder = Path(runs_dir) / f'{name}_{mask}'
        for s in seeds:
            if digest(np.load(folder / f'mask_{mask}_seed{s}.npy')) != masks_manifest[mask][s]['sha256']:
                raise ValueError(f'{folder.name}: mask seed {s} differs from masks_manifest.json')
        own, own_scores, record, _ = load_run(folder, run['rename'], mask)
        status += own
        scores.update(own_scores)
        rows = read_predictions(folder / f'predictions_{mask}.csv.gz')
        for r in rows:
            r['model'] = run['rename'].get(r['model'], r['model'])
        strata.update(stratum_scores(rows, cuts))
        del rows
        records.append({k: record[k] for k in ('dir', 'config_sha256', 'predictions_sha256', 'git')})
    runs = {m: {s: by_seed[s] for s in seeds} for m, by_seed in table_runs(status, scores).items()}
    return status, runs, strata, records


def pair_rows(runs, pairs, seeds, metric='mae_prop'):
    out = []
    for a, b in pairs:
        p = pair(runs, a, b, seeds, metric)
        p['sign_p'] = sign_test(p['wins'], p['n'] - p['wins'] - p['ties'])
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
                cols.append('NA' if not d else f'{ms(d, 4, signed)} ({wins}/{len(d)}, p {na(sign_test(wins, losses), 3)})')
            out.append(f'| {x} | ' + ' | '.join(cols) + ' |')
        out.append('')
    return out


def without_provenance(value):
    if isinstance(value, dict):
        return {k: without_provenance(v) for k, v in value.items() if k not in MANIFEST_PROVENANCE}
    return value


def same_file(name, new, reference):
    if name != 'tier2_manifest.json':
        return new.read_bytes() == reference.read_bytes()
    return (without_provenance(json.loads(new.read_text(encoding='utf-8')))
            == without_provenance(json.loads(reference.read_text(encoding='utf-8'))))


def tier2_check(config, out_dir, runs_dir, exploratory):
    """Native diagnostics again through tier2 from the exploratory runs, with the confirmation native CSVs
    as the rerun; compare with the step 5 reference tables."""
    runs_dir = Path(runs_dir)
    rerun = {'main': runs_dir / 'main_whole_feature', 'other': runs_dir / 'micoda_other_whole_feature',
             'residual': runs_dir / 'residual_whole_feature'}
    tier2.summarize(out_dir, rerun, exploratory)
    reference = ROOT / config['tier2']['compare_with']
    return {name: same_file(name, out_dir / name, reference / name) for name in STEP5_FILES}


def tier3_mask(config, mask, data, facts, runs_dir):
    hidden, preds, runs, record = load_mask(data, mask, runs_of(config), Path(runs_dir), seeds_of(config),
                                                  target_for(config, mask), roots=[runs_dir])
    cases = [one_case(data, facts, hidden[s], preds, runs, s)[0] for s in seeds_of(config)]
    one = {mask: cases}
    blocks, verdicts = {}, {}
    blocks['u1'], verdicts['u1'] = u1_table(one)
    blocks['u2'], verdicts['u2'] = u2_table(one)
    blocks['u3'], verdicts['u3'] = u3_table(one)
    blocks['sens'], extra = sensitivity_lines(one)
    verdicts.update(extra)
    blocks['u4'] = u4_table(one)
    return cases, blocks, verdicts, record


def pick(verdicts, key, sub):
    return verdicts[key][sub] if sub else verdicts[key]


def verdict_lines(verdicts, masks):
    out = ['| Diagnostic | Use case | ' + ' | '.join(SHORT[m] for m in masks) + ' | cả ba |',
           '|---|---|' + '---|' * (len(masks) + 1)]
    for d, u, key, sub in VERDICT_ROWS:
        per = {m: pick(verdicts[m], key, sub) for m in masks}
        out.append(f'| {d} | {u} | ' + ' | '.join(LABEL[per[m]] for m in masks)
                   + f' | **{LABEL[combine_masks(per)]}** |')
    return out


def hypotheses(config, runs, verdicts, seeds):
    max_failed, real = config['tier0']['max_failed_seeds'], config['models']['real']
    out = {}
    for h in config['hypotheses']:
        out[h['id']] = {}
        for mask in masks_of(config):
            if h['kind'] == 'tier3':
                key, sub = H_USE[h['use_case']]
                failures = {m: len(failed_seeds(runs[mask], m, seeds)) for m in real}
                out[h['id']][mask] = (judge_tier3(h, pick(verdicts[mask], key, sub), failures, max_failed), {})
            else:
                out[h['id']][mask] = judge_pairs(h, runs[mask], seeds, max_failed, pair)
    return out


def hypothesis_lines(config, judged):
    masks = masks_of(config)
    out = ['| ID | Giả thuyết | ' + ' | '.join(f'{SHORT[m]}' + (' (chính)' if m == masks[0] else '') for m in masks)
           + ' | Chi tiết ở mask chính |', '|---|---|' + '---|' * (len(masks) + 1)]
    for h in config['hypotheses']:
        cells = []
        for mask in masks:
            outcome = OUTCOMES[judged[h['id']][mask][0]]
            cells.append(f'**{outcome}**' if mask == masks[0] else outcome)
        parts = judged[h['id']][masks[0]][1]
        detail = '; '.join(f"vs {b}: Δ {signed(p['mean_diff'], 4)}, thắng {p['wins']}/{p['n']}, p {na(p['sign_p'], 3)}"
                           if p else f'vs {b}: không đủ support' for b, (_, p) in parts.items()) or '-'
        out.append(f"| {h['id']} | {h['text']} | " + ' | '.join(cells) + f' | {detail} |')
    return out


def exploratory_runs():
    """Seeds 0-4 tier 1 runs of the step 2 / 4 / 7 tables (configs/evaluation/exploratory), as
    {mask: {model: {seed: run}}}."""
    out = {}

    def add(mask, model, row):
        out.setdefault(mask, {}).setdefault(model, {})[row['seed']] = {
            'eligible': row['eligible'] == 'True', 'status': row['status'], 'code': row.get('code', ''), **{c: None if row.get(c) in (None, '', 'NA') else float(row[c])
                                                      for c, *_ in METRIC_COLUMNS}}
    for path, column in (('matched_tier1_runs.csv', 'model'), ('tier01_runs.csv', 'model'),
                         ('residual_runs.csv', 'variant')):
        with (EXPLORATORY / path).open(encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                name = row[column]
                if path.endswith('residual_runs.csv') and name not in ('hron_cap', 'tsagris_cap', 'micoda_other_cap'):
                    continue
                if row['mask'] == 'random_cell' or path.startswith('matched'):
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
                p = pair(source, a, b, use, metric)
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
                common = common_seeds(source, models, use)[0]
                cols.append(ms([source[m][s]['mae_prop'] for s in common]) + f' ({len(common)})')
        out.append(f'| {m} | ' + ' | '.join(cols) + ' |')
    earlier = json.loads((EXPLORATORY / 'verdicts.json').read_text(encoding='utf-8'))
    step6, step7 = earlier['step6'], earlier['step7']
    out += ['', 'Tầng 3: exploratory gộp 10 ca (step 6: wf + rc; step 7: wfm + rc), v1 theo từng mask 10 ca.', '',
            '| Diagnostic | Use case | step 6 wf+rc | step 7 wfm+rc | ' + ' | '.join(f'v1 {SHORT[m]}' for m in masks_of(config))
            + ' |', '|---|---|---|---|' + '---|' * len(masks_of(config))]
    for d, u, key, sub in VERDICT_ROWS:
        out.append(f'| {d} | {u} | {LABEL[pick(step6, key, sub)]} | {LABEL[pick(step7, key, sub)]} | '
                   + ' | '.join(LABEL[pick(verdicts[m], key, sub)] for m in masks_of(config)) + ' |')
    return out


def evaluate(config, seed_files, out_dir, config_path, runs_dir, exploratory):
    seeds, masks = seeds_of(config), masks_of(config)
    masks_manifest = json.loads(masks_manifest_path(config_path).read_text(encoding='utf-8'))
    if masks_manifest['config_sha256'] != frozen_sha256(config_path):
        raise ValueError('eval_v1.toml changed after the masks were drawn')
    data = load_csv(ROOT / config['data']['data'], ROOT / config['data']['schema'])
    facts, cuts = row_facts(data), step5_cuts(ROOT / config['tier1']['n_cuts_from'])
    blocks, runs, strata, status, verdicts, records, cases = {}, {}, {}, {}, {}, {}, {}
    models = [*config['models']['real'], *config['models']['decoys'], *config['models']['tier1_only']]
    for mask in masks:
        status[mask], runs[mask], strata[mask], records[mask] = tier1_mask(config, mask, data, cuts, masks_manifest['masks'],
                                                                             runs_dir)
        cases[mask], t3_blocks, verdicts[mask], _ = tier3_mask(config, mask, data, facts, runs_dir)
        blocks[f'tier3_{SHORT[mask]}'] = (t3_blocks['u1'] + [''] + t3_blocks['u2'] + [''] + t3_blocks['u3'] + ['']
                                          + t3_blocks['u4'] + [''] + t3_blocks['sens'])
    all_status = [r for mask in masks for r in status[mask]]
    blocks['status'] = status_markdown(all_status, reliability(all_status))
    blocks['tier1'], pairs_csv = [], []
    for mask in masks:
        main, _ = main_markdown(runs[mask], models, seeds)
        rows = pair_rows(runs[mask], config['tier1']['pairs'], seeds) + pair_rows(
            runs[mask], [('tsagris', 'hron')], seeds, 'mse_prop')
        blocks['tier1'] += [f'**{mask}**', '', *main, '']
        blocks.setdefault('pairs', []).extend(pair_lines(rows, seeds, f'**{mask}**') + [''])
        pairs_csv += [{'mask': mask, **{k: v for k, v in p.items() if k not in ('seeds_used', 'dropped')}} for p in rows]
    labels = {'continent': sorted(set(data.groups)), 'n': list(N_LABELS)}
    blocks['strata'] = strata_lines(config, strata, seeds, labels)
    same = tier2_check(config, out_dir / 'tier2', runs_dir, exploratory)
    blocks['tier2'] = ['Lượt native chạy lại qua `python -m longitudinal_impute.evaluation` (`tier2.summarize`, CSV native '
                       'của lượt xác nhận làm lượt chạy lại); so với output step 5 (configs/evaluation/step5_tier2): '
                       'bốn bảng so từng byte, manifest so phần kết quả (bỏ thư mục run và hash file code):', '',
                       '| File | giống step 5 |', '|---|:---:|'] + [f'| {k} | {v} |' for k, v in same.items()]
    blocks['tier3'] = verdict_lines(verdicts, masks)
    judged = hypotheses(config, runs, verdicts, seeds)
    blocks['hypotheses'] = hypothesis_lines(config, judged)
    blocks['exploratory'] = exploratory_lines(config, runs, seeds, verdicts)
    blocks['checks'] = [f'Seed xác nhận {seeds[0]}–{seeds[-1]}: không xuất hiện trong {len(seed_files)} file '
                        'config / manifest của step 1–7 (`protocol.check_unused`, ảnh chụp used_seeds.json).',
                        f"Mask: {len(masks) * len(seeds)} mask đã lưu khớp `masks_manifest.json`; F2 đạt ở cả ba mask; "
                        'long file khớp `summary.json` (tier1.load_run).']
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / 'v1_pairs.csv', pairs_csv)
    payload = json.dumps(cases, sort_keys=True, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    (out_dir / 'v1_cases.json').write_text(payload + '\n', encoding='utf-8')
    manifest = {'config_sha256': frozen_sha256(config_path), 'masks_manifest_sha256': sha256(masks_manifest_path(config_path)),
                'seed_check_files': seed_files, 'runs': records, 'tier2_equal_step5': same, 'verdicts': verdicts,
                'hypotheses': {k: {m: o for m, (o, _) in v.items()} for k, v in judged.items()},
                'cases_sha256': sha256(out_dir / 'v1_cases.json')}
    (out_dir / 'v1_manifest.json').write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n', encoding='utf-8')
    md = ['# Step 8: evaluation v1, lượt xác nhận (seed 200–209)', '', 'Sinh bởi `python -m longitudinal_impute.evaluation evaluate`; không sửa tay.', '']
    for name, title in (('hypotheses', '7. Giả thuyết'), ('checks', 'Kiểm tra'), ('status', '1. Tầng 0'),
                        ('tier1', '2. Tầng 1'), ('pairs', '3. Ghép cặp'), ('strata', '4. Tầng 1 theo tầng'),
                        ('tier2', '5. Tầng 2'), ('tier3', '6. Tầng 3'), *((f'tier3_{SHORT[m]}', f'6. Tầng 3 chi tiết {m}')
                                                                       for m in masks),
                        ('exploratory', '8. So với exploratory')):
        md += [f'## {title}', '', *blocks[name], '']
    (out_dir / 'v1_results.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    update_report(out_dir / 'REPORT_vi.md', blocks)
    return judged
