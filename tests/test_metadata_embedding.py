import importlib
import importlib.util
import os
from pathlib import Path
import sys
from dataclasses import replace
import numpy as np
import pytest
import torch
from longitudinal_impute.data import load_csv
from longitudinal_impute.main_model import MainModel, initialize, prepare_views, train_one_round, predict_counts
from longitudinal_impute.network import CompositionNet


def fixture(tmp_path, meta=True):
    (tmp_path / 'data.csv').write_text('id,time,total,a,b\nz,2,100,20,30\na,1,100,10,20\nz,1,100,30,10\nb,1,100,20,30\n')
    (tmp_path / 'schema.toml').write_text('[dataset]\nentity="id"\ntime="time"\ntotal="total"\ntargets=["a","b"]\n' + ('metadata_path="meta.csv"\n' if meta else ''))
    (tmp_path / 'meta.csv').write_text('continent,location,median_age,classification,group,ignored\nX,z,40,LOWER-MIDDLE INCOME,G,abc\nY,a,20,HIGH,H,\nX,b,30,HIGH,G,\nX,z,40,LOWER-MIDDLE-INCOME,G,\n')
    return load_csv(tmp_path / 'data.csv', tmp_path / 'schema.toml')


def model(**kw):
    return MainModel(channels=4, heads=1, outer_iters=1, epochs_per_iter=1,
                     batch_size=2, train_mask_rate=.2, seed=7, **kw)


def test_off_parity_against_prechange(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'dont_write_bytecode', True)
    default = Path(os.environ['LOCALAPPDATA']) / 'Temp/opencode/pre_metadata_src'
    root = Path(os.environ.get('METADATA_PARITY_ORACLE', str(default))) / 'longitudinal_impute'
    if not root.is_dir():
        pytest.skip(f'Pre-change parity oracle directory absent: {root}; set METADATA_PARITY_ORACLE to its source root')
    spec = importlib.util.spec_from_file_location('pre_metadata_oracle', root / '__init__.py', submodule_search_locations=[str(root)])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pkg
    spec.loader.exec_module(pkg)
    old = importlib.import_module('pre_metadata_oracle.main_model')
    old_data = importlib.import_module('pre_metadata_oracle.data')
    for meta in (False, True):
        data = fixture(tmp_path, meta)
        reference = old_data.Dataset(**{k: getattr(data, k) for k in old_data.Dataset.__dataclass_fields__})
        hidden = np.zeros_like(data.counts, bool)
        hidden[0, 0] = True
        current, previous = model(metadata=False), old.MainModel(channels=4, heads=1, outer_iters=1, epochs_per_iter=1, batch_size=2, train_mask_rate=.2, seed=7)
        np.testing.assert_array_equal(current.impute(data, hidden), previous.impute(reference, hidden))
        assert current.network_.state_dict().keys() == previous.network_.state_dict().keys()
        for k, v in current.network_.state_dict().items():
            assert torch.equal(v, previous.network_.state_dict()[k])
        assert len(current.info_['history']) == len(previous.info_['history'])
        for new_step, old_step in zip(current.info_['history'], previous.info_['history']):
            assert {k: v for k, v in new_step.items() if k != 'seconds'} == {
                k: v for k, v in old_step.items() if k != 'seconds'}
        assert 'metadata' not in current.info_


def test_existing_parameter_initialization_unchanged(tmp_path):
    spec = model(metadata=True, metadata_location=True)._fit_metadata(fixture(tmp_path))
    torch.manual_seed(5)
    off = CompositionNet(3, 4, 1)
    torch.manual_seed(5)
    on = CompositionNet(3, 4, 1, metadata_spec=spec)
    for k, v in off.state_dict().items():
        assert torch.equal(v, on.state_dict()[k])
    assert set(on.metadata_embeddings) == {'continent', 'location', 'classification', 'group'}


def test_loader_validation_and_statistics(tmp_path):
    data = fixture(tmp_path)
    assert set(data.metadata['z']) == {'continent', 'location', 'median_age', 'classification', 'group'}
    assert data.metadata['z']['classification'] == 'LOWER-MIDDLE-INCOME'
    assert len(data.metadata_provenance['sha256']) == 64
    spec = model(metadata=True)._fit_metadata(data)
    assert spec['age_mean'] == 30
    assert spec['age_sd'] == pytest.approx(np.std([20, 30, 40]))
    path = tmp_path / 'meta.csv'
    original = path.read_text()
    for text in (original.replace('Y,a,20,HIGH,H,\n', ''), original.replace('a,20', 'a,nan'), original.replace('a,20', 'a,abc'), original.replace('X,z,40,LOWER-MIDDLE-INCOME', 'X,z,41,LOWER-MIDDLE-INCOME'), original.replace('a,20', 'a,40').replace('b,30', 'b,40'), original.replace('Y,a,20,HIGH', 'Y,a,20,')):
        path.write_text(text)
        with pytest.raises(ValueError, match='metadata'):
            load_csv(tmp_path / 'data.csv', tmp_path / 'schema.toml')
    assert fixture(tmp_path, False).metadata is None


def test_alignment_two_views_and_short_batch(tmp_path):
    data = fixture(tmp_path)
    net = CompositionNet(3, 4, 1, metadata_spec=model(metadata=True, metadata_location=True)._fit_metadata(data))
    hidden = np.zeros_like(data.counts, bool)
    hidden[0, 0] = True
    q = np.zeros_like(hidden)
    q[1, 0] = q[3, 0] = q[0, 1] = True
    inference, training, q, labels = prepare_views(data, hidden, train_mask=q, seed=7)
    seen = []
    handle = net.register_forward_pre_hook(lambda module, args: seen.append(args[4]['location'].tolist()))
    train_one_round(net, training, q, labels, epochs=1, batch_size=2)
    predict_counts(net, inference, batch_size=2)
    handle.remove()
    assert seen[-2:] == [[0, 1], [2]]
    assert seen[:2] == [[0, 1], [2]]
    assert all(batch in ([0, 1], [2]) for batch in seen)
    order = [3, 2, 0, 1]
    shuffled = replace(data, counts=data.counts[order], totals=data.totals[order], entities=tuple(data.entities[i] for i in order), times=tuple(data.times[i] for i in order), rows=[data.rows[i] for i in order])
    out = predict_counts(net, initialize(shuffled, hidden[order]), batch_size=2)
    np.testing.assert_array_equal(out[np.argsort(order)], predict_counts(net, inference, batch_size=2))


def test_forward_contract_and_gradients(tmp_path):
    from longitudinal_impute.main_model import _metadata_inputs
    data = fixture(tmp_path)
    net = CompositionNet(3, 4, 1, metadata_spec=model(metadata=True)._fit_metadata(data))
    meta = _metadata_inputs(net, data)
    x = torch.arange(18, dtype=torch.float32).reshape(3, 2, 3) / 10
    valid = torch.tensor([[True, True], [True, False], [True, False]])
    args = (x, torch.ones_like(x, dtype=torch.bool), torch.zeros(3, 2), valid)
    out = net(*args, meta)
    assert out.shape == x.shape and torch.isfinite(out).all()
    assert torch.equal(out[~valid], torch.zeros_like(out[~valid]))
    torch.testing.assert_close(out[valid].sum(-1), torch.ones(4))
    out[..., 0].sum().backward()
    assert net.metadata_age.weight.grad.abs().sum() > 0
    assert all(e.weight.grad.abs().sum() > 0 for e in net.metadata_embeddings.values())
    assert not torch.equal(out, net(*args, {k: v.flip(0) for k, v in meta.items()}))
    with pytest.raises(ValueError, match='metadata'):
        net(*args)
    with pytest.raises(ValueError, match='metadata'):
        net(*args, {k: v[:1] for k, v in meta.items()})


def test_runner_alias_and_noloc(tmp_path):
    import json
    from longitudinal_impute.run import read_models, build, run_model, main
    data = fixture(tmp_path)
    assert model().metadata is True and model().metadata_location is False
    configs = {k: v for k, v in read_models(Path('configs/models_all.toml')).items() if k.startswith('main')}
    assert set(configs) == {'main', 'main_nometa'}
    assert model().metadata is True and model().metadata_location is False
    for name in configs:
        m = build(name, {**configs[name], 'channels': 4, 'heads': 1, 'outer_iters': 1,
                        'epochs_per_iter': 1, 'batch_size': 2, 'train_mask_rate': .2, 'seed': 7})
        hidden = np.zeros_like(data.counts, bool)
        hidden[0, 0] = True
        _, status, _ = run_model(data, m, hidden)
        assert status == 'completed'
        if name != 'main_nometa':
            assert set(m.network_.metadata_embeddings) == {'continent', 'classification', 'group'}
            assert m.network_.metadata_age is not None
        else:
            assert m.network_.metadata_spec is None
    with pytest.raises(ValueError, match='metadata'):
        model().impute(replace(data, metadata=None))
    # Exercise the actual full-stage CLI route for both aliases with shipped defaults.
    recipe = tmp_path / 'models.toml'
    recipe.write_text('\n'.join(
        f'[{name}]\n' + ('model="main"\n' if name != 'main' else '') +
        ('metadata=false\n' if name == 'main_nometa' else '') +
        'channels=4\nheads=1\nouter_iters=1\nepochs_per_iter=1\ntrain_mask_rate=0.2\nseed=7\n'
        for name in configs))
    output = tmp_path / 'run'
    main(['--data', str(tmp_path / 'data.csv'), '--schema', str(tmp_path / 'schema.toml'),
          '--models', str(recipe), '--output', str(output), '--mask', 'random_cell',
          '--mask-rate', '.125', '--seeds', '300', '--pass', 'masked', '--stage', 'full'])
    assert json.loads((output / 'config.json').read_text())['stage'] == 'full'
    assert 'failed' not in (output / 'summary.csv').read_text()


def test_baseline_predictions_ignore_metadata_key(tmp_path):
    from longitudinal_impute.run import build, impute
    from test_run import FIXTURES, COVID_SCHEMA
    schema = tmp_path / 'schema.toml'
    schema.write_text(COVID_SCHEMA)
    plain = load_csv(FIXTURES / 'covid_small.csv', schema)
    entities = sorted(set(plain.entities))
    (tmp_path / 'meta.csv').write_text('continent,location,median_age,classification,group\n' +
        ''.join(f'X,{entity},{20+i},HIGH,G\n' for i, entity in enumerate(entities)))
    schema.write_text(COVID_SCHEMA + '\nmetadata_path="meta.csv"\n')
    enriched = load_csv(FIXTURES / 'covid_small.csv', schema)
    hidden = np.zeros_like(plain.counts, bool)
    hidden[0, 0] = True
    for name in ('hron', 'tsagris', 'micoda', 'same_day_mean'):
        params = {'min_obs': 1} if name == 'same_day_mean' else {}
        a = impute(plain, build(name, params), hidden)
        b = impute(enriched, build(name, params), hidden)
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
