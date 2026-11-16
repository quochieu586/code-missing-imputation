"""Tests of residual.py on the small COVID fixture (random_cell only, seeds 0-4).

    PYTHONPATH=src python -W error::RuntimeWarning -m pytest tv2_step4/test_residual.py
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('residual', HERE / 'residual.py')
residual = importlib.util.module_from_spec(spec)
spec.loader.exec_module(residual)

from longitudinal_impute import run  # noqa: E402

REPO = HERE.parent
FIXTURE = REPO / 'tests/fixtures/covid_small.csv'
SCHEMA = '[dataset]\nentity="location"\ntime="date"\ntotal="total_sequence"\ntargets=["v1","v2","v3","v4"]\n'
ARGS = ('--mask', 'random_cell', '--mask-rate', '.2', '--seeds', '0', '1', '2', '3', '4', '--save-predictions')
SDM = 'window_days = 31\nmin_obs = 2\nmax_widen = 1\n'
DIRS = {
    'step2/main': '[hron]\n[tsagris]\n[micoda]\nother = false\nmax_iter = 40\n[mean_fill]\n',
    'step2/micoda_other': '[micoda]\nmax_iter = 40\nother = true\n',
    'step3/ablation': f'[sdm_full]\nmodel = "same_day_mean"\n{SDM}[sdm_rescale]\nmodel = "same_day_mean"\n{SDM}'
                      'residual_rule = "rescale"\n',
    'step4/residual': ''.join(f'[{m}_{r}]\nmodel = "{"micoda" if m == "micoda_other" else m}"\nresidual_rule = "{r}"\n'
                              + ('max_iter = 40\nother = true\n' if m == 'micoda_other' else '')
                              for m in ('hron', 'tsagris', 'micoda_other') for r in ('cap', 'rescale')),
    'step4/check': ('[hron_prop_check]\nmodel = "hron"\nresidual_rule = "prop_other"\n'
                    '[tsagris_prop_check]\nmodel = "tsagris"\nresidual_rule = "prop_other"\n'
                    '[micoda_other_prop_check]\nmodel = "micoda"\nmax_iter = 40\nother = true\n'
                    'residual_rule = "prop_other"\n')}


@pytest.fixture(scope='module')
def roots(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('runs')
    (tmp / 'schema.toml').write_text(SCHEMA)
    for key, models in DIRS.items():
        root, prefix = key.split('/')
        (tmp / f'{prefix}.toml').write_text(models)
        run.main(['--data', str(FIXTURE), '--schema', str(tmp / 'schema.toml'), '--models', str(tmp / f'{prefix}.toml'),
                  '--output', str(tmp / root / f'{prefix}_random_cell'), *ARGS])
    return tmp


def test_end_to_end_tables_and_checks(roots, tmp_path, monkeypatch):
    monkeypatch.setattr(residual, 'MASKS', ('random_cell',))
    manifest, run_rows, pair_rows = residual.summarize(roots / 'step2', roots / 'step3', roots / 'step4', tmp_path)
    m = manifest['random_cell']
    assert {r['variant'] for r in run_rows} == set(residual.VARIANTS)
    assert all(r['eligible'] for r in run_rows)
    for name in ('hron_prop_check', 'tsagris_prop_check'):
        c = m['consistency'][name]
        assert c['identical'] == c['cells'] > 0 and c['native']['bytes_identical']
    assert m['consistency']['micoda_other_prop_check']['max_abs_count'] < 1e-9
    assert m['consistency']['tsagris_cap_vs_prop']['max_abs_count'] < 1e-9
    assert all(c['max_abs_diff'] == 0 for c in m['regression_check'].values())
    assert {r['a'] for r in pair_rows if r['kind'] == 'within_model'} == {
        f'{x}_{r}' for x in residual.WITHIN_MODEL for r in ('cap', 'rescale')}
    text = (tmp_path / 'residual_results.md').read_text()
    assert 'Bảng 4' in text and 'Other ngầm' in text


def test_refuses_a_run_with_another_mask(roots, tmp_path, monkeypatch):
    monkeypatch.setattr(residual, 'MASKS', ('random_cell',))
    path = roots / 'step4' / 'residual_random_cell' / 'mask_random_cell_seed2.npy'
    original = np.load(path)
    changed = original.copy()
    changed[np.argwhere(changed)[0][0]] = False
    np.save(path, changed)
    try:
        with pytest.raises(ValueError, match='seed 2: saved masks differ'):
            residual.summarize(roots / 'step2', roots / 'step3', roots / 'step4', tmp_path / 'out')
        assert not (tmp_path / 'out').exists()
    finally:
        np.save(path, original)


def test_decomposition_terms_add_up_by_hand():
    def ok(value):
        return {'eligible': True, 'status': 'completed', 'code': '',
                **{c: value for c, *_ in residual.tier01.METRIC_COLUMNS}}
    values = {'sdm_cap': .03, 'sdm_rescale': .08, 'hron_prop': .09, 'hron_cap': .05, 'hron_rescale': .10}
    runs = {m: {s: ok(v + .001 * int(s)) for s in residual.SEEDS} for m, v in values.items()}
    rows = residual.decomposition(runs, 'hron')
    assert '| khoảng cách sdm_cap − hron_prop | -0.0600 ± 0.0000 | 5 | 100% |' in rows[2]
    assert '| thuật toán @cap | -0.0200 ± 0.0000 | 5 | +33% |' in rows[3]
    assert '| quy tắc hron (cap − prop_other) | -0.0400 ± 0.0000 | 5 | +67% |' in rows[4]
    assert '| thuật toán @rescale | -0.0200 ± 0.0000 | 5 | +33% |' in rows[6]
    assert '| quy tắc hron (rescale − prop_other) | +0.0100 ± 0.0000 | 0 | -17% |' in rows[7]
