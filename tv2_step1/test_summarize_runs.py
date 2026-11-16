"""Tests of summarize_runs.py on small fake run directories (no model is run).

    python -W error::RuntimeWarning -m pytest tv2_step1/test_summarize_runs.py
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('summarize_runs', Path(__file__).with_name('summarize_runs.py'))
summarize_runs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summarize_runs)

MAIN = ('hron', 'tsagris', 'micoda', 'same_day_mean', 'mean_fill', 'marginal_sampler', 'zero_fill')
GROUPS = ('model', 'truth_zero', 'truth_positive', 'model_truth_zero')


def errors(value):
    return {g: {'cells': 10, 'mae_proportion': value, 'mse_proportion': value ** 2,
                'mae_count': 100 * value, 'mse_count': 1e4 * value ** 2} for g in GROUPS}


def write_run(folder, mask, models, scores, *, data_sha='d' * 64, mask_offset=0, rate=.05):
    """scores: {model: [mae per seed] or None for a failed run}."""
    folder.mkdir(parents=True)
    for seed in summarize_runs.SEEDS:
        hidden = np.zeros((6, 3), bool)
        hidden[(int(seed) + mask_offset) % 6, 1] = True
        np.save(folder / f'mask_{mask}_seed{seed}.npy', hidden)
    config = {'data': '/nowhere/data.csv', 'data_sha256': data_sha, 'schema': {'entity': 'e', 'targets': ['a', 'b']},
              'mask': mask, 'mask_rate': rate, 'seeds': [0, 1, 2, 3, 4], 'models': models,
              'git': None, 'group_map': None}
    (folder / 'config.json').write_text(json.dumps(config))
    masked = {seed: {'models': {m: ({'status': 'failed', 'errors': None} if values is None else
                                    {'status': 'ok', 'errors': errors(values[int(seed)])})
                                for m, values in scores.items()}} for seed in summarize_runs.SEEDS}
    native = {m: {'status': 'ok', 'info': {}} for m in scores}
    (folder / 'summary.json').write_text(json.dumps({'native': native, 'masked': masked}))


def fake_runs(root, **other):
    """Valid pair of runs for both masks; keyword arguments perturb the micoda_other runs."""
    other_models = {'micoda': {'max_iter': 120, 'tol': 1e-8, 'other': other.pop('other_flag', True)}}
    for mask in summarize_runs.MASKS:
        scores = {m: [.1 + .01 * k + .001 * s for s in range(5)] for k, m in enumerate(MAIN)}
        write_run(root / f'main_{mask}', mask, {m: {} for m in MAIN}, scores)
        write_run(root / f'micoda_other_{mask}', mask, other_models, {'micoda': [.05] * 5}, **other)


def test_valid_pair_writes_tables_and_manifest(tmp_path):
    fake_runs(tmp_path / 'runs')
    summarize_runs.summarize(tmp_path / 'runs', tmp_path / 'out', levels=False)
    assert {p.name for p in (tmp_path / 'out').iterdir()} == {
        'results_multiseed.md', 'results_multiseed.csv', 'results_manifest.json'}
    manifest = json.loads((tmp_path / 'out' / 'results_manifest.json').read_text())
    main, other = manifest['whole_feature']
    assert main['mask_sha256'] == other['mask_sha256'] and len(main['mask_sha256']) == 5
    assert other['models']['micoda']['other'] is True and main['data_sha256'] == other['data_sha256']


@pytest.mark.parametrize('perturbation,message', [
    ({'mask_offset': 1}, 'saved masks differ'),
    ({'data_sha': 'e' * 64}, 'data_sha256'),
    ({'other_flag': False}, 'other = true'),
    ({'rate': .1}, 'rate'),
])
def test_mismatched_runs_are_refused_and_nothing_is_written(tmp_path, perturbation, message):
    fake_runs(tmp_path / 'runs', **perturbation)
    with pytest.raises(ValueError, match=message):
        summarize_runs.summarize(tmp_path / 'runs', tmp_path / 'out', levels=False)
    assert not (tmp_path / 'out').exists()


def test_main_run_with_other_true_is_refused(tmp_path):
    fake_runs(tmp_path / 'runs')
    config_path = tmp_path / 'runs' / 'main_whole_feature' / 'config.json'
    config = json.loads(config_path.read_text())
    config['models']['micoda'] = {'other': True}
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='other = false'):
        summarize_runs.summarize(tmp_path / 'runs', tmp_path / 'out', levels=False)


def test_zero_error_sorts_first_and_incomplete_models_last(tmp_path):
    """F4 (PR #4 review): `metric or inf` used to send an exact 0.0 to the bottom."""
    runs = {'worse': {s: {'status': 'ok', 'errors': errors(.1)} for s in summarize_runs.SEEDS},
            'missing': {s: {'status': 'failed', 'errors': None} for s in summarize_runs.SEEDS},
            'partly': {s: ({'status': 'ok', 'errors': errors(.01)} if s != '3' else
                           {'status': 'failed', 'errors': None}) for s in summarize_runs.SEEDS},
            'perfect': {s: {'status': 'ok', 'errors': errors(0.)} for s in summarize_runs.SEEDS}}
    out = []
    summarize_runs.main_table(runs, out)
    assert [line.split('|')[1].strip() for line in out[2:]] == ['perfect', 'worse', 'partly', 'missing']


def test_win_counts_come_from_the_csv_per_metric(tmp_path):
    """F5 (PR #4 review): a low mean must not be reported as a per-seed win; each metric apart."""
    import csv
    keys = [f'{k}_model' for k, _ in summarize_runs.PAIR_METRICS]
    rows = []
    for mask in summarize_runs.MASKS:
        for s in summarize_runs.SEEDS:
            k = int(s)
            hron = {'mae_proportion': .09, 'mse_proportion': .066, 'mae_count': 20., 'mse_count': 5e4}
            # tsagris: worse MAE prop every seed, better MSE prop every seed, MAE count better on
            # seeds 0 and 4 only but with a much lower mean (one huge Hron seed is not a win).
            tsagris = {'mae_proportion': .1, 'mse_proportion': .05,
                       'mae_count': 10. if k in (0, 4) else 25., 'mse_count': 5e4}
            for name, values in (('hron', hron), ('tsagris', tsagris)):
                rows.append({'mask': mask, 'model': name, 'seed': s, **{f'{m}_model': values[m] for m in values}})
    path = tmp_path / 'r.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['mask', 'model', 'seed', *keys])
        writer.writeheader()
        writer.writerows(rows)
    pairs = {(p['mask'], p['metric']): p for p in summarize_runs.pairs_from_csv(path)
             if (p['a'], p['b']) == ('tsagris', 'hron')}
    got = {m: (pairs['whole_feature', m]['wins'], pairs['whole_feature', m]['ties']) for m in
           ('mae_proportion', 'mse_proportion', 'mae_count', 'mse_count')}
    assert got == {'mae_proportion': (0, 0), 'mse_proportion': (5, 0), 'mae_count': (2, 0), 'mse_count': (0, 5)}
    block = summarize_runs.pair_markdown(summarize_runs.pairs_from_csv(path))
    assert '- whole_feature: tsagris thắng hron ở MAE tỷ lệ 0/5, MSE tỷ lệ 5/5, MAE count 2/5, MSE count 0/5 seed.' in block
    # The docs get exactly the generated block between the markers, the rest untouched.
    doc = tmp_path / 'REPORT_vi.md'
    doc.write_text(f'before\n{summarize_runs.START}\nhand typed 5/5\n{summarize_runs.END}\nafter\n')
    summarize_runs.update_docs(tmp_path, block)
    text = doc.read_text()
    assert 'hand typed' not in text and text.startswith('before\n') and text.endswith('\nafter\n')
    assert '\n'.join(block).strip() in text


def test_recorded_paths_from_before_the_restructure_map_to_the_repository_root():
    old = Path('/somewhere/repo/nextgen/configs/covid_location_continent.csv')
    assert summarize_runs.current_path(old) == summarize_runs.REPO / 'configs/covid_location_continent.csv'
    assert summarize_runs.current_path(str(old)) == summarize_runs.REPO / 'configs/covid_location_continent.csv'
    other = Path('/somewhere/repo/data/covariants.csv')
    assert summarize_runs.current_path(other) == other


def test_micoda_other_default_follows_the_runner_that_wrote_the_config():
    assert summarize_runs.micoda_other({'models': {'micoda': {}}}) is False                  # before PR #5: A17
    assert summarize_runs.micoda_other({'models': {'micoda': {}}, 'stage': 'full'}) is True  # PR #5 runner: A18
    assert summarize_runs.micoda_other({'models': {'micoda': {'other': False}}, 'stage': 'full'}) is False
    assert summarize_runs.micoda_other({'models': {'micoda': {'other': True}}}) is True
