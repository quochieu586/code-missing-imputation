"""Tier 0: pairing of the main and micoda_other run directories (F2) on fake runs (no model is run).

    PYTHONPATH=src python -W error::RuntimeWarning -m pytest tests/evaluation/test_tier0.py
"""
import json

import numpy as np
import pytest

from longitudinal_impute.evaluation import tier0

MASKS = ('whole_feature', 'random_cell')

MAIN = ('hron', 'tsagris', 'micoda', 'same_day_mean', 'mean_fill', 'marginal_sampler', 'zero_fill')
GROUPS = ('model', 'truth_zero', 'truth_positive', 'model_truth_zero')


def errors(value):
    return {g: {'cells': 10, 'mae_proportion': value, 'mse_proportion': value ** 2,
                'mae_count': 100 * value, 'mse_count': 1e4 * value ** 2} for g in GROUPS}


def write_run(folder, mask, models, scores, *, data_sha='d' * 64, mask_offset=0, rate=.05):
    """scores: {model: [mae per seed] or None for a failed run}."""
    folder.mkdir(parents=True)
    for seed in tier0.SEEDS:
        hidden = np.zeros((6, 3), bool)
        hidden[(int(seed) + mask_offset) % 6, 1] = True
        np.save(folder / f'mask_{mask}_seed{seed}.npy', hidden)
    config = {'data': '/nowhere/data.csv', 'data_sha256': data_sha, 'schema': {'entity': 'e', 'targets': ['a', 'b']},
              'mask': mask, 'mask_rate': rate, 'seeds': [0, 1, 2, 3, 4], 'models': models,
              'git': None, 'group_map': None}
    (folder / 'config.json').write_text(json.dumps(config))
    masked = {seed: {'models': {m: ({'status': 'failed', 'errors': None} if values is None else
                                    {'status': 'ok', 'errors': errors(values[int(seed)])})
                                for m, values in scores.items()}} for seed in tier0.SEEDS}
    native = {m: {'status': 'ok', 'info': {}} for m in scores}
    (folder / 'summary.json').write_text(json.dumps({'native': native, 'masked': masked}))


def fake_runs(root, **other):
    """Valid pair of runs for both masks; keyword arguments perturb the micoda_other runs."""
    other_models = {'micoda': {'max_iter': 120, 'tol': 1e-8, 'other': other.pop('other_flag', True)}}
    for mask in MASKS:
        scores = {m: [.1 + .01 * k + .001 * s for s in range(5)] for k, m in enumerate(MAIN)}
        write_run(root / f'main_{mask}', mask, {m: {} for m in MAIN}, scores)
        write_run(root / f'micoda_other_{mask}', mask, other_models, {'micoda': [.05] * 5}, **other)


def test_valid_pair_is_accepted_with_the_saved_masks(tmp_path):
    fake_runs(tmp_path / 'runs')
    main, other = tier0.validate_pair(tmp_path / 'runs', 'whole_feature')
    assert main['mask_sha256'] == other['mask_sha256'] and len(main['mask_sha256']) == 5
    assert other['models']['micoda']['other'] is True and main['data_sha256'] == other['data_sha256']


@pytest.mark.parametrize('perturbation,message', [
    ({'mask_offset': 1}, 'saved masks differ'),
    ({'data_sha': 'e' * 64}, 'data_sha256'),
    ({'other_flag': False}, 'other = true'),
    ({'rate': .1}, 'rate'),
])
def test_mismatched_runs_are_refused(tmp_path, perturbation, message):
    fake_runs(tmp_path / 'runs', **perturbation)
    with pytest.raises(ValueError, match=message):
        tier0.validate_pair(tmp_path / 'runs', 'whole_feature')


def test_main_run_with_other_true_is_refused(tmp_path):
    fake_runs(tmp_path / 'runs')
    config_path = tmp_path / 'runs' / 'main_whole_feature' / 'config.json'
    config = json.loads(config_path.read_text())
    config['models']['micoda'] = {'other': True}
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='other = false'):
        tier0.validate_pair(tmp_path / 'runs', 'whole_feature')


def test_micoda_other_default_follows_the_runner_that_wrote_the_config():
    assert tier0.micoda_other({'models': {'micoda': {}}}) is False                  # before PR #5: A17
    assert tier0.micoda_other({'models': {'micoda': {}}, 'stage': 'full'}) is True  # PR #5 runner: A18
    assert tier0.micoda_other({'models': {'micoda': {'other': False}}, 'stage': 'full'}) is False
    assert tier0.micoda_other({'models': {'micoda': {'other': True}}}) is True
