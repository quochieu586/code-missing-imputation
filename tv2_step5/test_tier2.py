"""Tests of the step 5 table logic (reuse checks, table 6, table 7) on toy inputs.

    PYTHONPATH=nextgen/src python -W error::RuntimeWarning -m pytest tv2_step5/test_tier2.py
"""
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('tier2', HERE / 'tier2.py')
tier2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tier2)


def result(d1_local, d1_pooled, pr_ref=.5, d2=None):
    side = {'d1': None, 'pr_zero_ref': pr_ref, 'd2': d2}
    agg = {'equal': None, 'weighted': None, 'targets': 0, 'cells': 0}
    return {'targets': {'t': {'local': {**side, 'd1': d1_local}, 'pooled': {**side, 'd1': d1_pooled}}},
            'summary': {f'{r}_{d}': agg for r in ('local', 'pooled') for d in ('d1', 'd2')}}


def test_shift_table_marks_sign_flips_and_large_changes():
    lines = tier2.shift_lines({'a': result(.05, -.02), 'b': result(.4, .1), 'c': result(None, .1)}, ['t'])
    row = next(line for line in lines if line.startswith('| t |'))
    assert row.endswith('| ± | L | NA |')
    assert 'a 1 / 0 (/1); b 0 / 1 (/1); c 0 / 0 (/0)' in lines[-1]


def test_zero_fill_check_uses_one_minus_the_reference_zero_share():
    results = {'zero_fill': result(.5, .4, pr_ref=.6), 'other': result(.1, .2)}
    line = tier2.decoy_lines(results, ['t'])[0]
    assert 'D1 pooled = 1 − Pr_ref(zero) ở 1/1 target' in line
    assert 'ở 1/1 target (R-local), 1/1 (R-pooled)' in line and 'D2 = NA ở 1/1 (R-local)' in line
    results['other'] = result(.6, .2)
    assert 'ở 0/1 target (R-local)' in tier2.decoy_lines(results, ['t'])[0]


def test_reuse_stops_on_any_mismatch(tmp_path, monkeypatch):
    """A changed models table or a different native CSV in the random_cell run stops the script."""
    data = tier2.load_csv(tier2.DATA, tier2.SCHEMA)
    for mask in tier2.MASKS:
        source = tier2.REPO / f'{tier2.MAIN}_{mask}'
        if not (source / 'zero_fill_completed.csv').exists():
            pytest.skip('step 2 run directories are not available')
        target = tmp_path / f'run_{mask}'
        target.mkdir()
        for name in ('config.json', 'summary.json', 'zero_fill_completed.csv'):
            shutil.copy(source / name, target / name)
    monkeypatch.setattr(tier2, 'REPO', tmp_path)
    monkeypatch.setitem(tier2.SOURCES, 'zero_fill', ('run', 'zero_fill', 'models_default.toml', 'main'))
    values, record = tier2.load_source('zero_fill', data)
    assert values.shape == data.counts.shape and record['same_csv_in_random_cell']
    config = json.loads((tmp_path / 'run_whole_feature/config.json').read_text())
    config['models']['zero_fill'] = {'unexpected': 1}
    (tmp_path / 'run_whole_feature/config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError, match='differs from models_default.toml'):
        tier2.load_source('zero_fill', data)
    config['models']['zero_fill'] = {}
    (tmp_path / 'run_whole_feature/config.json').write_text(json.dumps(config))
    with (tmp_path / 'run_random_cell/zero_fill_completed.csv').open('a') as stream:
        stream.write('\n')
    with pytest.raises(ValueError, match='not identical'):
        tier2.load_source('zero_fill', data)
