"""Tests of ablation.py on the small COVID fixture (random_cell only, seeds 0-4).

    PYTHONPATH=src python -W error::RuntimeWarning -m pytest tv2_step3/test_ablation.py
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('ablation', HERE / 'ablation.py')
ablation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ablation)

from longitudinal_impute import run  # noqa: E402

REPO = HERE.parent
FIXTURE = REPO / 'tests/fixtures/covid_small.csv'
SCHEMA = '[dataset]\nentity="location"\ntime="date"\ntotal="total_sequence"\ntargets=["v1","v2","v3","v4"]\n'
ARGS = ('--mask', 'random_cell', '--mask-rate', '.2', '--seeds', '0', '1', '2', '3', '4', '--save-predictions')


def runs(tmp_path, monkeypatch, step2_models='[hron]\n[same_day_mean]\n[mean_fill]\n'):
    monkeypatch.setattr(ablation, 'MASKS', ('random_cell',))
    (tmp_path / 'schema.toml').write_text(SCHEMA)
    (tmp_path / 'step2.toml').write_text(step2_models)
    for root, models in ((tmp_path / 'step2', tmp_path / 'step2.toml'),
                         (tmp_path / 'step3', REPO / 'configs/models_sameday_ablation.toml')):
        name = 'main_random_cell' if root.name == 'step2' else 'ablation_random_cell'
        run.main(['--data', str(FIXTURE), '--schema', str(tmp_path / 'schema.toml'), '--models', str(models),
                  '--output', str(root / name), *ARGS])
    return tmp_path / 'step3', tmp_path / 'step2'


def test_end_to_end_checks_and_tables(tmp_path, monkeypatch):
    step3, step2 = runs(tmp_path, monkeypatch)
    manifest, run_rows, pair_rows = ablation.summarize(step3, step2, tmp_path / 'out')
    m = manifest['random_cell']
    assert m['sdm_full_vs_step2_same_day_mean']['native_completed_csv_identical']
    assert m['sdm_full_vs_step2_same_day_mean']['hidden_cell_rows_identical'] > 0
    assert m['regression_check']['max_abs_diff'] == 0 and m['regression_check']['values_checked'] > 0
    assert m['ablation_run']['resolved_models']['sdm_none']['effective_params']['weighting'] == 'none'
    assert len(m['ablation_run']['predictions_sha256']) == 64
    assert {r['model'] for r in run_rows} == {*ablation.VARIANTS, *ablation.BASELINES}
    versus = {(r['a'], r['b']) for r in pair_rows}
    assert versus == {(v, 'sdm_full') for v in ablation.VARIANTS[1:]} | set(ablation.EXTRA_PAIRS)
    assert all(r['n'] == 5 for r in pair_rows if r['metric'] == 'mae_prop')
    text = (tmp_path / 'out' / 'ablation_results.md').read_text()
    assert 'sdm_none vs mean_fill, random_cell' in text and 'Đóng góp' in text


def test_refuses_different_masks_before_writing(tmp_path, monkeypatch):
    step3, step2 = runs(tmp_path, monkeypatch)
    path = step3 / 'ablation_random_cell' / 'mask_random_cell_seed3.npy'
    hidden = np.load(path)
    hidden[np.argwhere(hidden)[0][0]] = False
    np.save(path, hidden)
    with pytest.raises(ValueError, match='seed 3: saved masks differ'):
        ablation.summarize(step3, step2, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_refuses_when_sdm_full_differs_from_step2(tmp_path, monkeypatch):
    step3, step2 = runs(tmp_path, monkeypatch, '[hron]\n[same_day_mean]\nmin_obs = 2\n[mean_fill]\n')
    with pytest.raises(ValueError, match='sdm_full differs from step 2 same_day_mean'):
        ablation.summarize(step3, step2, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_contribution_share_by_hand():
    def ok(value):
        return {'eligible': True, 'status': 'completed', 'code': '',
                **{c: value for c, *_ in ablation.tier01.METRIC_COLUMNS}}
    values = {'sdm_full': .1, 'sdm_none': .3, 'sdm_no_time': .25, 'sdm_no_group': .1,
              'sdm_unweighted': .05, 'sdm_rescale': .2}
    table = {m: {'0': ok(v), '1': ok(v)} for m, v in values.items()}
    rows = ablation.contributions(table, ('0', '1'))
    assert rows[2].startswith('| sdm_none − sdm_full (mẫu số) | +0.2 |')
    assert '| +75% (Δ +0.15) |' in rows[3] and '| +0% (Δ +0) |' in rows[4]
    assert '| -25% (Δ -0.05) |' in rows[5] and '| +50% (Δ +0.1) |' in rows[6]
