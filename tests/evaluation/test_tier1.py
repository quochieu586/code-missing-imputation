"""Tier 0 / 1 tables (tier1): F07 common-seed pairing, failed runs in the status table, end to end on a fixture.

    PYTHONPATH=src python -W error::RuntimeWarning -m pytest tests/evaluation/test_tier1.py
"""
import csv
import json
from pathlib import Path

import pytest

from longitudinal_impute import run
from longitudinal_impute.baselines.tsagris import TsagrisKNN
from longitudinal_impute.errors import InsufficientDataError
from longitudinal_impute.evaluation import tier0, tier1 as tier01

FIXTURE = Path(__file__).resolve().parents[1] / 'fixtures/covid_small.csv'
SCHEMA = '[dataset]\nentity="location"\ntime="date"\ntotal="total_sequence"\ntargets=["v1","v2","v3","v4"]\n'


def ok(value):
    return {'eligible': True, 'status': 'completed', 'code': '', **{c: value for c, *_ in tier01.METRIC_COLUMNS}}


FAILED = {'eligible': False, 'status': 'failed', 'code': 'insufficient_data',
          **{c: None for c, *_ in tier01.METRIC_COLUMNS}}


def test_f07_toy_pairs_on_common_seeds_and_a_wins():
    runs = {'A': {'0': ok(.10), '1': ok(.80)}, 'B': {'0': ok(.20), '1': FAILED}}
    p = tier01.pair(runs, 'A', 'B', ('0', '1'))
    assert p['seeds_used'] == ['0'] and p['n'] == 1 and p['wins'] == 1
    assert p['mean_diff'] == pytest.approx(-.10) and p['sd_diff'] is None
    assert p['dropped'] == {'1': ['B failed (insufficient_data)']}
    # A success-only mean would rank B (0.20) before A (0.45); the common-seed ranking does not.
    order, used, dropped, means = tier01.ranking(runs, ['A', 'B'], ('0', '1'))
    assert order == ['A', 'B'] and used == ['0'] and list(dropped) == ['1']
    assert means['A'][0] == pytest.approx(.10) and means['B'][0] == pytest.approx(.20)


def test_empty_common_seed_set_gives_na_not_zero():
    runs = {'A': {'0': FAILED}, 'B': {'0': ok(.2)}}
    p = tier01.pair(runs, 'A', 'B', ('0',))
    assert p['n'] == 0 and p['mean_diff'] is None and p['wins'] == 0
    _, used, _, means = tier01.ranking(runs, ['A', 'B'], ('0',))
    assert used == [] and means == {'A': (None, None), 'B': (None, None)}


def test_failed_and_missing_runs_stay_in_the_status_table():
    config = {'models': {'hron': {}, 'micoda': {}}, 'seeds': [0, 1], 'mask': 'whole_feature'}
    stats = {'hidden_cells': 10, 'forced_zero_cells': 4, 'model_cells': 6}
    summary = {'native': {'hron': {'status': 'ok', 'info': {}},
                          'micoda': {'status': 'ok', 'info': {'converged': True, 'iterations': 7}}},
               'native_cells': {'missing_forced_zero_cells': 1, 'missing_model_cells': 2},
               'masked': {'0': {'mask': stats, 'models': {
                   'hron': {'status': 'ok', 'coverage': 1., 'errors': {}},
                   'micoda': {'status': 'failed', 'coverage': 0., 'error_code': 'insufficient_fit',
                              'error': 'Complete-case initialization needs more than one row',
                              'info': {'converged': None}}}},
                          '1': {'mask': stats, 'models': {'hron': {'status': 'ok', 'coverage': 1.}}}}}
    rows = tier0.status_rows('main_whole_feature', config, summary, {'micoda': 'micoda_x'})
    assert len(rows) == 2 + 2 * 2
    by = {(r['model'], r['seed']): r for r in rows}
    assert by['micoda_x', '0']['status'] == 'failed' and by['micoda_x', '0']['error_code'] == 'insufficient_fit'
    assert by['micoda_x', '1']['status'] == 'missing'
    assert by['micoda_x', None]['converged'] is True and by['hron', None]['coverage'] == 1.
    assert (by['hron', '0']['forced_zero_cells'], by['hron', '0']['model_cells']) == (4, 6)
    rel = {(e['model'], e['pass']): e for e in tier0.reliability(rows)}
    assert (rel['micoda_x', 'masked']['scheduled'], rel['micoda_x', 'masked']['completed']) == (2, 0)
    assert rel['micoda_x', 'masked']['not_ok'] == ['0: failed (insufficient_fit)', '1: missing (missing_result)']


def fixture_runs(tmp_path, monkeypatch):
    """Main + micoda_other runs on the fixture, seeds 1 and 2; tsagris fails on seed 2."""
    calls = []

    class FailsOnSecondSeed(TsagrisKNN):
        def fit(self, values):
            calls.append(1)
            if len(calls) == 3:                    # native, seed 1, then seed 2 fails
                raise InsufficientDataError('no complete row')
            return super().fit(values)

    monkeypatch.setitem(run.MODELS, 'tsagris', FailsOnSecondSeed)
    monkeypatch.setattr(tier01, 'SEEDS', {'whole_feature': ('1', '2')})
    (tmp_path / 'schema.toml').write_text(SCHEMA)
    for prefix, models in (('main', '[hron]\n[tsagris]\n[micoda]\nother = false\n[mean_fill]\n'),
                           ('micoda_other', '[micoda]\nother = true\n')):
        (tmp_path / f'{prefix}.toml').write_text(models)
        run.main(['--data', str(FIXTURE), '--schema', str(tmp_path / 'schema.toml'), '--models',
                  str(tmp_path / f'{prefix}.toml'), '--output', str(tmp_path / 'runs' / f'{prefix}_whole_feature'),
                  '--mask', 'whole_feature', '--mask-rate', '.2', '--seeds', '1', '2', '--save-predictions'])


def test_end_to_end_on_fixture_with_one_failing_run(tmp_path, monkeypatch):
    fixture_runs(tmp_path, monkeypatch)
    manifest, status, run_rows, pair_rows = tier01.summarize(tmp_path / 'runs', tmp_path / 'out')
    failed = [r for r in status if r['status'] != 'completed']
    assert [(r['model'], r['seed'], r['error_code']) for r in failed] == [('tsagris', '2', 'insufficient_data')]
    assert manifest['whole_feature']['regression_check']['values_checked'] > 0
    assert manifest['whole_feature']['regression_check']['max_abs_diff'] <= 1e-12
    assert {r['dir'] for r in manifest['whole_feature']['runs']} == {'main_whole_feature',
                                                                     'micoda_other_whole_feature'}
    assert all(len(r['predictions_sha256']) == 64 for r in manifest['whole_feature']['runs'])
    pairs = {(r['a'], r['b'], r['metric']): r for r in pair_rows}
    with_tsagris = [r for (a, b, m), r in pairs.items() if 'tsagris' in (a, b) and m == 'mae_prop']
    assert with_tsagris and all(r['n'] == 1 and r['seeds_used'] == '1' for r in with_tsagris)
    assert all(r['n'] == 2 for (a, b, m), r in pairs.items() if 'tsagris' not in (a, b) and m == 'mae_prop')
    with (tmp_path / 'out' / 'tier01_status.csv').open(newline='') as stream:
        assert sum(1 for r in csv.DictReader(stream)) == 5 * (1 + 2)
    text = (tmp_path / 'out' / 'tier01_results.md').read_text()
    assert 'Seed 2 bị loại: tsagris failed (insufficient_data).' in text


def test_regression_check_refuses_a_mismatch():
    errors = {g: {'cells': 1, **{m: .5 for m in tier01.METRICS}} for g in tier01.GROUPS}
    summary = {'masked': {'0': {'models': {'hron': {'status': 'ok', 'errors': errors}}}}}
    scores = {('hron', 'random_cell', 0): {'errors': json.loads(json.dumps(errors))}}
    assert tier01.check_scores(scores, summary, {}, 'random_cell')[0] == len(tier01.GROUPS) * 5
    scores['hron', 'random_cell', 0]['errors']['model']['mae_proportion'] = .5 + 1e-9
    with pytest.raises(ValueError, match='long file'):
        tier01.check_scores(scores, summary, {}, 'random_cell')


def test_undeclared_seed_is_in_status_only(tmp_path, monkeypatch):
    fixture_runs(tmp_path, monkeypatch)
    monkeypatch.setattr(tier01, 'UNDECLARED', ('2',))     # the seed where tsagris fails
    _, status, run_rows, pair_rows = tier01.summarize(tmp_path / 'runs', tmp_path / 'out')
    assert [(r['model'], r['seed'], r['status']) for r in status if r['status'] != 'completed'] == [
        ('tsagris', '2', 'failed')]
    assert {r['seed'] for r in run_rows} == {'1'}
    assert all(r['seeds_used'] == '1' and r['dropped'] == '' for r in pair_rows)
    text = (tmp_path / 'out' / 'tier01_results.md').read_text()
    assert 'Seed không khai báo trước (2)' in text and 'Seed 2 bị loại' not in text
