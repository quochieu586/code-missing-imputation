"""Final CLI boundaries: persistence, no-fit evaluation and staged parity."""
import json
from pathlib import Path
import numpy as np
import pytest
from longitudinal_impute import run
from longitudinal_impute.data import load_csv
from test_run import FIXTURES, COVID_SCHEMA, MICROBIOME_SCHEMA, write

MAIN = '[main]\nchannels=8\nheads=2\nouter_iters=1\nepochs_per_iter=1\nbatch_size=2\nseed=0\n'


def command(tmp_path, output, models='[hron]\n[tsagris]\n[micoda]\n', case='covid', *extra):
    schema = tmp_path / f'{case}.toml'
    if not schema.exists():
        schema.write_text(COVID_SCHEMA if case == 'covid' else MICROBIOME_SCHEMA)
    spec = tmp_path / 'models.toml'
    if not spec.exists():
        spec.write_text(models)
    return run.main(['--data', str(FIXTURES / f'{case}_small.csv'), '--schema', str(schema),
                     '--models', str(spec), '--output', str(output), *extra])


def test_masked_only_saves_reusable_predictions_without_native_pass(tmp_path):
    output = tmp_path / 'masked'
    assert command(tmp_path, output, '[hron]\n', 'covid', '--pass', 'masked',
                   '--mask', 'random_cell', '--mask-rate', '.1', '--seeds', '7') == 0
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['native'] == {} and all(r['pass'] == 'masked' for r in summary['rows'])
    manifest = json.loads((output / 'artifacts.json').read_text())
    path = manifest['predictions']['random_cell_seed7/hron']
    values = np.load(output / path, allow_pickle=False)
    assert values.shape == (24, 4) and np.isfinite(values).all()
    assert not list(output.glob('*_completed.csv'))


def test_explicit_native_pass_does_not_schedule_masks(tmp_path):
    output = tmp_path / 'native'
    assert command(tmp_path, output, '[hron]\n', 'covid', '--pass', 'native', '--mask', 'whole_feature') == 0
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['masked'] == {} and all(r['pass'] == 'native' for r in summary['rows'])
    assert (output / 'predictions/native_hron.npy').is_file()
    assert not list(output.glob('mask_*.npy'))


def test_default_full_persists_main_history(tmp_path):
    torch = pytest.importorskip('torch')
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        output = tmp_path / 'full'
        assert command(tmp_path, output, MAIN) == 0
        history = output / 'main/native/history.csv'
        assert history.is_file() and 'max_change_proportion' in history.read_text()
        manifest = json.loads((output / 'artifacts.json').read_text())
        assert 'main/native/history.csv' in manifest['files']
    finally:
        torch.set_num_threads(old)


@pytest.mark.parametrize('case', ['covid', 'microbiome'])
@pytest.mark.parametrize('kind', ['random_cell', 'whole_feature'])
def test_init_refine_matches_full_without_running_initializer_again(tmp_path, monkeypatch, case, kind):
    torch = pytest.importorskip('torch')
    from longitudinal_impute import main_model
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        flags = ('--mask', kind, '--mask-rate', '.1', '--seeds', '7')
        full, initial, refined = (tmp_path / n for n in ('full', 'init', 'refine'))
        assert command(tmp_path, full, MAIN, case, *flags) == 0
        def no_torch(name, *args, **kwargs):
            if name == 'torch':
                raise AssertionError('init must not import torch')
            return original(name, *args, **kwargs)
        import builtins
        original = builtins.__import__
        with monkeypatch.context() as m:
            m.setattr(builtins, '__import__', no_torch)
            assert command(tmp_path, initial, MAIN, case, '--stage', 'init', *flags) == 0
        initialized = json.loads((initial / 'summary.json').read_text())
        assert all(r['status'] == 'initialized' for r in initialized['rows'])
        assert initialized['aggregate'] == [] and not list(initial.glob('*_completed.csv'))
        manifest = json.loads((initial / 'artifacts.json').read_text())
        assert set(manifest['initials']) == {'native', f'{kind}_seed7'}
        def no_init(*args, **kwargs):
            raise AssertionError('refine must not rerun initialization')
        monkeypatch.setattr(main_model, 'initialize', no_init)
        assert run.main(['--stage', 'refine', '--from-run', str(initial), '--output', str(refined)]) == 0
        actual = json.loads((refined / 'artifacts.json').read_text())
        expected = json.loads((full / 'artifacts.json').read_text())
        assert set(actual['predictions']) == set(expected['predictions'])
        for key, relative in actual['predictions'].items():
            np.testing.assert_allclose(np.load(refined / relative), np.load(full / expected['predictions'][key]),
                                       atol=1e-10, rtol=1e-10)
    finally:
        torch.set_num_threads(old)


@pytest.mark.parametrize('change', ['data', 'schema', 'models', 'mask', 'state'])
def test_refine_rejects_changed_or_corrupt_source_before_output(tmp_path, change):
    import shutil
    source = tmp_path / 'data.csv'
    shutil.copyfile(FIXTURES / 'covid_small.csv', source)
    schema = write(tmp_path, 'schema.toml', COVID_SCHEMA)
    models = write(tmp_path, 'models.toml', MAIN)
    initial, refined = tmp_path / 'init', tmp_path / 'refine'
    assert run.main(['--stage', 'init', '--data', str(source), '--schema', str(schema), '--models', str(models),
                     '--output', str(initial), '--mask', 'random_cell', '--mask-rate', '.1', '--seeds', '7']) == 0
    if change in ('data', 'schema', 'models'):
        path = {'data': source, 'schema': schema, 'models': models}[change]
        path.write_bytes(path.read_bytes() + b'\n')
    else:
        manifest = json.loads((initial / 'artifacts.json').read_text())
        path = (initial / 'mask_random_cell_seed7.npy' if change == 'mask'
                else initial / manifest['initials']['native'])
        path.write_bytes(path.read_bytes() + b'corruption')
    with pytest.raises(ValueError, match='(hash|checksum|changed|artifact)'):
        run.main(['--stage', 'refine', '--from-run', str(initial), '--output', str(refined)])
    assert not refined.exists()


def test_invalid_stage_combinations_are_rejected_before_output(tmp_path):
    output = tmp_path / 'invalid'
    with pytest.raises(ValueError):
        command(tmp_path, output, '[hron]\n', 'covid', '--stage', 'init')
    assert not output.exists()
    with pytest.raises(ValueError):
        run.main(['--stage', 'refine', '--output', str(output)])
    assert not output.exists()


def test_evaluate_uses_saved_predictions_without_fit_or_torch(tmp_path, monkeypatch):
    import builtins
    output, rescored = tmp_path / 'full', tmp_path / 'rescored'
    assert command(tmp_path, output, '[hron]\n[tsagris]\n[micoda]\n', 'covid',
                   '--mask', 'random_cell', '--mask-rate', '.1', '--seeds', '7') == 0
    before = {p.relative_to(output).as_posix(): p.read_bytes() for p in output.rglob('*') if p.is_file()}
    expected = json.loads((output / 'summary.json').read_text())
    def no_fit(*args, **kwargs):
        raise AssertionError('evaluate must not construct or fit models')
    for name in run.MODELS:
        monkeypatch.setitem(run.MODELS, name, no_fit)
    monkeypatch.setattr(run, 'run_model', no_fit)
    original = builtins.__import__
    def no_torch(name, *args, **kwargs):
        if name == 'torch':
            raise AssertionError('evaluate must not import torch')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', no_torch)
    assert run.main(['--stage', 'evaluate', '--from-run', str(output), '--output', str(rescored)]) == 0
    actual = json.loads((rescored / 'summary.json').read_text())
    assert actual['rows'] == expected['rows']
    for name in ('hron', 'tsagris', 'micoda'):
        assert actual['masked']['7']['models'][name]['errors'] == expected['masked']['7']['models'][name]['errors']
        assert (rescored / f'{name}_completed.csv').read_bytes() == (output / f'{name}_completed.csv').read_bytes()
    assert before == {p.relative_to(output).as_posix(): p.read_bytes() for p in output.rglob('*') if p.is_file()}


def test_evaluate_recomputes_metrics_not_just_copies_summary(tmp_path, monkeypatch):
    full, rescored = tmp_path / 'full', tmp_path / 'rescored'
    assert command(tmp_path, full, '[hron]\n', 'covid', '--mask', 'random_cell', '--seeds', '7') == 0
    original = run.masked_errors
    def changed_metric(*args):
        result = original(*args)
        result['all']['mse_count'] += 1.
        return result
    expected = json.loads((full / 'summary.json').read_text())['masked']['7']['models']['hron']['errors']['all']['mse_count']
    monkeypatch.setattr(run, 'masked_errors', changed_metric)
    assert run.main(['--stage', 'evaluate', '--from-run', str(full), '--output', str(rescored)]) == 0
    actual = json.loads((rescored / 'summary.json').read_text())['masked']['7']['models']['hron']['errors']['all']['mse_count']
    assert actual == expected + 1.


@pytest.mark.parametrize('status', ['failed', 'partial'])
def test_evaluate_preserves_failed_and_partial_without_cherry_picking(tmp_path, monkeypatch, status):
    from longitudinal_impute.baselines.micoda import CovidResult
    def result(self, counts, totals, hidden):
        values = np.where(hidden, np.nan, counts).copy()
        if status == 'partial':
            values[tuple(np.argwhere(np.isnan(values))[0])] = 0.
        return CovidResult(values, status, {'status': status, 'unresolved_cells': int(np.isnan(values).sum()),
                           'error': {'code': 'fixture_error', 'message': 'controlled failure'}},
                           totals - np.nansum(values, axis=1))
    monkeypatch.setattr(run.MICoDaCovid, 'impute_counts', result)
    full, rescored = tmp_path / 'full', tmp_path / 'rescored'
    assert command(tmp_path, full, '[micoda]\n', 'covid', '--mask', 'random_cell', '--seeds', '7') == 1
    def no_fit(*args, **kwargs):
        raise AssertionError('evaluation must not retry failed or partial models')
    monkeypatch.setattr(run, 'run_model', no_fit)
    assert run.main(['--stage', 'evaluate', '--from-run', str(full), '--output', str(rescored)]) == 1
    details = json.loads((rescored / 'summary.json').read_text())
    assert details['native']['micoda']['status'] == status
    assert details['masked']['7']['models']['micoda']['status'] == status
    assert 'errors' not in details['masked']['7']['models']['micoda']
    assert not details['paired']['aggregate']


def test_evaluate_rejects_artifact_traversal_before_output(tmp_path):
    full = tmp_path / 'full'
    assert command(tmp_path, full, '[hron]\n') == 0
    manifest = json.loads((full / 'artifacts.json').read_text())
    manifest['files']['../outside.npy'] = '0' * 64
    (full / 'artifacts.json').write_text(json.dumps(manifest))
    rescored = tmp_path / 'rescored'
    with pytest.raises(ValueError, match='artifact'):
        run.main(['--stage', 'evaluate', '--from-run', str(full), '--output', str(rescored)])
    assert not rescored.exists()


@pytest.mark.parametrize('change', ['q', 'labels', 'counts', 'delta', 'mask', 'object_array'])
def test_rehashed_but_incompatible_initial_payload_is_rejected(tmp_path, change):
    import hashlib
    initial, refined = tmp_path / 'init', tmp_path / 'refine'
    assert command(tmp_path, initial, MAIN, 'covid', '--stage', 'init', '--pass', 'masked',
                   '--mask', 'random_cell', '--mask-rate', '.1', '--seeds', '7') == 0
    manifest = json.loads((initial / 'artifacts.json').read_text())
    relative = ('mask_random_cell_seed7.npy' if change == 'mask'
                else manifest['initials']['random_cell_seed7'])
    path = initial / relative
    if change == 'mask':
        mask = np.load(path)
        mask[0, 0] = ~mask[0, 0]
        np.save(path, mask)
    else:
        with np.load(path, allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        if change == 'q':
            arrays['q'] = ~arrays['q']
        elif change == 'labels':
            arrays['labels'][tuple(np.argwhere(~arrays['q'])[0])] = .25
        elif change == 'counts':
            arrays['inference'][1, 0] += 1.
        elif change == 'delta':
            arrays['delta_training'] = np.array(float('nan'))
        else:
            arrays['no_missing'] = np.array(False, dtype=object)
        np.savez_compressed(path, **arrays)
    manifest['files'][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    (initial / 'artifacts.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        run.main(['--stage', 'refine', '--from-run', str(initial), '--output', str(refined)])
    assert not refined.exists()


def test_init_is_not_evaluable_and_source_run_cannot_be_modified(tmp_path):
    initial = tmp_path / 'init'
    assert command(tmp_path, initial, MAIN, 'covid', '--stage', 'init') == 0
    with pytest.raises(ValueError, match='predictions'):
        run.main(['--stage', 'evaluate', '--from-run', str(initial), '--output', str(tmp_path / 'evaluate')])
    with pytest.raises(ValueError, match='inside'):
        run.main(['--stage', 'refine', '--from-run', str(initial), '--output', str(initial / 'child')])
    assert not (initial / 'child').exists()
    with pytest.raises(FileExistsError):
        run.main(['--stage', 'refine', '--from-run', str(initial), '--output', str(initial)])


def test_main_predictions_can_be_rescored_without_neural_dependencies(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    import builtins
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        full, rescored = tmp_path / 'full', tmp_path / 'rescored'
        assert command(tmp_path, full, MAIN, 'covid', '--mask', 'whole_feature', '--mask-rate', '.1', '--seeds', '7') == 0
        expected = json.loads((full / 'summary.json').read_text())
        def no_fit(*args, **kwargs):
            raise AssertionError('no neural construction/fit during evaluate')
        monkeypatch.setitem(run.MODELS, 'main', no_fit)
        monkeypatch.setattr(run.MainModel, 'impute', no_fit)
        original = builtins.__import__
        def guard(name, *args, **kwargs):
            if name == 'torch':
                raise AssertionError('no torch import during evaluate')
            return original(name, *args, **kwargs)
        monkeypatch.setattr(builtins, '__import__', guard)
        assert run.main(['--stage', 'evaluate', '--from-run', str(full), '--output', str(rescored)]) == 0
        actual = json.loads((rescored / 'summary.json').read_text())
        assert actual['rows'] == expected['rows']
        assert actual['masked']['7']['models']['main']['errors'] == expected['masked']['7']['models']['main']['errors']
    finally:
        torch.set_num_threads(old)


@pytest.mark.parametrize('stage', ['evaluate', 'refine'])
@pytest.mark.parametrize('change', ['missing_seconds', 'null_pass', 'bad_pass', 'bad_stage',
                                   'bad_seeds', 'negative_seconds', 'string_seconds', 'absent_case',
                                   'missing_error_code', 'missing_error_message', 'string_error',
                                   'list_error_code', 'string_info', 'list_info',
                                   'null_history', 'invalid_history_row'])
def test_invalid_source_metadata_is_rejected_before_output(tmp_path, stage, change):
    import hashlib
    source, output = tmp_path / 'source', tmp_path / 'output'
    flags = ('--mask', 'random_cell', '--mask-rate', '.1', '--seeds', '7')
    if stage == 'refine':
        assert command(tmp_path, source, MAIN, 'covid', '--stage', 'init', *flags) == 0
        name = 'main'
    else:
        assert command(tmp_path, source, '[hron]\n', 'covid', *flags) == 0
        name = 'hron'
    relative = 'config.json' if change in ('null_pass', 'bad_pass', 'bad_stage', 'bad_seeds') else 'summary.json'
    path = source / relative
    payload = json.loads(path.read_text())
    if change == 'missing_seconds':
        del payload['native'][name]['seconds']
    elif change in ('negative_seconds', 'string_seconds'):
        payload['masked']['7']['models'][name]['seconds'] = -1 if change == 'negative_seconds' else '1.0'
    elif change == 'absent_case':
        del payload['masked']['7']
    elif change in ('missing_error_code', 'missing_error_message', 'string_error', 'list_error_code',
                    'string_info', 'list_info', 'null_history', 'invalid_history_row'):
        payload['native'][name]['info'] = {
            'missing_error_code': {'error': {'message': 'missing code'}},
            'missing_error_message': {'error': {'code': 'missing_message'}},
            'string_error': {'error': 'invalid'},
            'list_error_code': {'error': {'code': [], 'message': 'invalid code'}},
            'string_info': 'invalid', 'list_info': [],
            'null_history': {'history': None},
            'invalid_history_row': {'history': [{'unexpected_column': 0}]},
        }[change]
        if stage == 'evaluate':
            payload['native'][name]['status'] = 'failed'
    else:
        key, value = {'null_pass': ('pass', None), 'bad_pass': ('pass', 'unknown'),
                      'bad_stage': ('stage', 'unknown'), 'bad_seeds': ('seeds', [])}[change]
        payload[key] = value
    path.write_text(json.dumps(payload))
    manifest = json.loads((source / 'artifacts.json').read_text())
    manifest['files'][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    (source / 'artifacts.json').write_text(json.dumps(manifest))
    before = {p.relative_to(source).as_posix(): p.read_bytes() for p in source.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='source.*metadata'):
        run.main(['--stage', stage, '--from-run', str(source), '--output', str(output)])
    assert not output.exists()
    assert before == {p.relative_to(source).as_posix(): p.read_bytes() for p in source.rglob('*') if p.is_file()}
