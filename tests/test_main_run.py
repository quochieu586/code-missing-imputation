"""Main integration through the existing CLI boundary, sharing masks and failures."""
import builtins
import json
from pathlib import Path
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from longitudinal_impute import run, main_model
from longitudinal_impute.data import load_csv, check_completed
from test_run import call, write, FIXTURES, COVID_SCHEMA, MICROBIOME_SCHEMA

MAIN = ('[main]\nchannels=8\nheads=2\nouter_iters=2\nepochs_per_iter=1\n'
        'batch_size=2\nseed=0\ntol=0.0\n')
ALL = '[hron]\n[tsagris]\n[micoda]\nother=true\n' + MAIN


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_main_is_accepted_in_existing_model_configuration(tmp_path):
    params = run.read_models(write(tmp_path, 'main.toml', MAIN))
    assert set(params) == {'main'}


@pytest.mark.parametrize('case,schema', [('covid', COVID_SCHEMA), ('microbiome', MICROBIOME_SCHEMA)])
@pytest.mark.parametrize('kind', ['random_cell', 'whole_feature'])
def test_cli_runs_all_four_with_the_same_mask_and_space(tmp_path, case, schema, kind):
    source = FIXTURES / f'{case}_small.csv'
    code, output = call(tmp_path, source, schema, ALL, '--mask', kind, '--mask-rate', '.1', '--seeds', '7')
    assert code == 0
    detail = json.loads((output / 'summary.json').read_text())
    assert set(detail['native']) == {'hron', 'tsagris', 'micoda', 'main'}
    assert all(r['status'] == 'ok' and r['space'] == '18_with_other' for r in detail['rows'])
    hidden = np.load(output / f'mask_{kind}_seed7.npy')
    for name, result in detail['masked']['7']['models'].items():
        assert result['coverage'] == 1.
        assert result['errors']['all']['cells'] == int(hidden.sum())
    assert detail['paired']['cases'][kind]['seeds_included'] == [7]
    assert detail['paired']['cases'][kind]['models'] == ['hron', 'main', 'micoda', 'tsagris']
    info = detail['masked']['7']['models']['main']['info']
    assert info['trained'] is True and 1 <= info['iterations'] <= 2
    assert info['budget_policy'] == 'R2'
    data = load_csv(source, tmp_path / 'schema.toml')
    completed = load_csv(output / 'main_completed.csv', tmp_path / 'schema.toml')
    check_completed(data, completed.counts, np.zeros_like(hidden))
    observed = np.isfinite(data.counts)
    for i in range(len(data.rows)):
        for j, target in enumerate(data.targets):
            if observed[i, j]:
                assert data.rows[i][target] == completed.rows[i][target]
    if case == 'microbiome':
        assert detail['native']['main']['info']['termination_reason'] == 'no_missing'


def test_main_gets_fresh_instance_for_native_and_each_mask(tmp_path, monkeypatch):
    class RecordingMain(main_model.MainModel):
        fitted = []
        def impute(self, data, hidden=None):
            assert self.network_ is None
            self.fitted.append(self)
            return super().impute(data, hidden)
    monkeypatch.setitem(run.MODELS, 'main', RecordingMain)
    code, _ = call(tmp_path, FIXTURES / 'covid_small.csv', COVID_SCHEMA, MAIN,
                   '--mask', 'random_cell', '--mask-rate', '.1', '--seeds', '7', '8')
    assert code == 0
    assert len(RecordingMain.fitted) == 3
    assert len({id(m) for m in RecordingMain.fitted}) == 3
    assert all(m.network_ is not None for m in RecordingMain.fitted)


@pytest.mark.parametrize('cause', ['empty_q', 'no_torch', 'dll', 'runtime'])
def test_main_failure_does_not_stop_baselines(tmp_path, monkeypatch, cause):
    models = MAIN + '[hron]\n[tsagris]\n[micoda]\n'
    if cause == 'empty_q':
        models = models.replace('[main]\n', '[main]\ntrain_mask_rate=0.0\n')
    elif cause in ('no_torch', 'dll'):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name == 'torch':
                if cause == 'dll':
                    raise OSError(126, 'controlled Windows torch DLL load failure')
                raise ModuleNotFoundError('torch deliberately unavailable', name='torch')
            return original(name, *args, **kwargs)
        monkeypatch.setattr(builtins, '__import__', guarded)
    else:
        def fails(*args, **kwargs):
            raise RuntimeError('controlled neural runtime failure')
        monkeypatch.setattr(main_model, 'train_one_round', fails)
    code, output = call(tmp_path, FIXTURES / 'covid_small.csv', COVID_SCHEMA, models,
                        '--mask', 'random_cell', '--mask-rate', '.1', '--seeds', '7')
    assert code == 1
    detail = json.loads((output / 'summary.json').read_text())
    assert detail['native']['main']['status'] == 'failed'
    assert detail['masked']['7']['models']['main']['status'] == 'failed'
    assert not (output / 'main_completed.csv').exists()
    for name in ('hron', 'tsagris', 'micoda'):
        assert detail['native'][name]['status'] == 'ok'
        assert detail['masked']['7']['models'][name]['status'] == 'ok'
        assert (output / f'{name}_completed.csv').is_file()
    assert not detail['paired']['aggregate']
    text = (output / 'errors.log').read_text()
    assert ('nonempty' if cause == 'empty_q' else 'extra main' if cause in ('no_torch', 'dll')
            else 'controlled neural runtime failure') in text
    if cause == 'dll':
        assert 'controlled Windows torch DLL load failure' in text
    if cause == 'runtime':
        assert 'iteration=1' in text and 'stage=train' in text


def test_invalid_main_params_fail_before_output(tmp_path):
    with pytest.raises(ValueError, match='outer_iters'):
        call(tmp_path, FIXTURES / 'covid_small.csv', COVID_SCHEMA, '[main]\nouter_iters=0\n')
    assert not (tmp_path / 'out').exists()


def test_shipped_all_model_preset_preserves_baselines_and_main_recipe():
    root = Path(__file__).resolve().parents[1]
    baseline = run.read_models(root / 'configs/models_default.toml')
    main = run.read_models(root / 'configs/main.toml')
    combined = run.read_models(root / 'configs/models_all.toml')
    assert set(combined) == {'hron', 'tsagris', 'micoda', 'main'}
    assert {k: combined[k] for k in baseline} == baseline
    assert combined['main'] == main['main']
