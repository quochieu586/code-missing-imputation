"""Fault injection for scientific inputs and attempt-based legacy persistence."""
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import torch
from src.stage_b import experiment_artifacts as store
from scripts.frozen_sources import source_hashes, historical_snapshot


def checkpoints():
    return {f'budget{budget}': {'best_epoch': 1, 'best_loss': 1.,
                               'state_dict': {'w': torch.ones(2)}} for budget in (280, 1120)}


def save(root):
    store.train_cache_save(root, 'probe', checkpoints(), [], np.zeros(2), 1., {})


def test_interrupted_checkpoint_retained_retry_is_new_attempt(tmp_path):
    (tmp_path / 'config.json').write_text('{}')
    def interrupted(data, path):
        Path(path).write_bytes(b'partial')
        raise OSError('disk interruption')
    with patch.object(store.torch, 'save', side_effect=interrupted), pytest.raises(OSError):
        save(tmp_path)
    assert store.train_cache_load(tmp_path, 'probe') is None
    partial = next(tmp_path.rglob('checkpoint.pt'))
    save(tmp_path)
    assert partial.read_bytes() == b'partial'
    blob = store.train_cache_load(tmp_path, 'probe')
    assert blob is not None and Path(blob['artifact_path']) != partial
    assert len(list(tmp_path.rglob('checkpoint.pt'))) == 2


@pytest.mark.parametrize('kind', ['train', 'eval'])
def test_corrupt_completed_artifact_rejected_without_deletion(tmp_path, kind):
    (tmp_path / 'config.json').write_text('{}')
    if kind == 'train':
        save(tmp_path)
        load = store.train_cache_load
    else:
        store.eval_cache_save(tmp_path, 'probe', {'a': {'score': 1.}}, {'a': np.ones((2, 3))})
        load = store.eval_cache_load
    blob = load(tmp_path, 'probe')
    artifact = Path(blob['artifact_path'])
    artifact.write_bytes(b'corrupt')
    assert load(tmp_path, 'probe') is None
    assert artifact.read_bytes() == b'corrupt'


def test_legacy_eval_index_without_npz_is_not_completion(tmp_path):
    (tmp_path / 'config.json').write_text('{}')
    base = tmp_path / 'cache_eval'
    base.mkdir()
    (base / 'probe.json').write_text(json.dumps({'scores': {}, 'config_sha256': store._cache_sha(tmp_path)}))
    assert store.eval_cache_load(tmp_path, 'probe') is None


def test_usable_legacy_checkpoint_migrates_without_retraining_or_overwrite(tmp_path):
    (tmp_path / 'config.json').write_text('{}')
    base = tmp_path / 'cache_train'
    base.mkdir()
    original = base / 'probe.pt'
    torch.save({'state_dict_280': {'w': torch.ones(2)},
                'state_dict_1120': {'w': torch.ones(2)}}, original)
    before = original.read_bytes()
    (base / 'probe.json').write_text(json.dumps({'config_sha256': store._cache_sha(tmp_path),
        'mean': [0., 0.], 'scale': 1., 'trace': [], 'extra': {},
        'checkpoints': {f'budget{b}': {'best_epoch': 1, 'best_loss': 1.} for b in (280, 1120)}}))
    blob = store.train_cache_load(tmp_path, 'probe')
    assert blob['migration'] == 'validated_legacy_unsealed'
    assert Path(blob['artifact_path']).read_bytes() == original.read_bytes() == before
    assert Path(blob['artifact_path']) != original


def test_missing_completed_npz_is_rejected_and_attempt_retained(tmp_path):
    (tmp_path / 'config.json').write_text('{}')
    store.eval_cache_save(tmp_path, 'probe', {'a': {}}, {'a': np.ones((2, 3))})
    blob = store.eval_cache_load(tmp_path, 'probe')
    artifact = Path(blob['artifact_path'])
    artifact.rename(artifact.with_suffix('.retained'))
    assert store.eval_cache_load(tmp_path, 'probe') is None
    assert artifact.with_suffix('.retained').exists()


def test_nonfinite_checkpoint_never_publishes_completion(tmp_path):
    (tmp_path / 'config.json').write_text('{}')
    bad = checkpoints()
    bad['budget280']['state_dict']['w'][0] = float('nan')
    with pytest.raises(ValueError, match='tensor'):
        store.train_cache_save(tmp_path, 'probe', bad, [], np.zeros(2), 1., {})
    assert store.train_cache_load(tmp_path, 'probe') is None
    assert not list(tmp_path.rglob('complete.json'))


def test_sources_migrated_exactly_to_historical_snapshot():
    root = Path(__file__).resolve().parents[3]
    baseline = json.loads((root / 'artifacts/e3e4_rework_2026-10-05/frozen_hashes_before.json').read_text())
    from scripts.diagnose_e5_rework import frozen_hashes
    assert historical_snapshot(frozen_hashes(), baseline, root) == baseline
    assert any('_assets' in k for k in source_hashes(root))


@pytest.mark.parametrize('present', [False, True])
def test_missing_or_changed_required_source_fails_closed(tmp_path, present):
    (tmp_path / 'directive').mkdir()
    if present:
        (tmp_path / 'directive/source.pdf').write_bytes(b'changed')
    (tmp_path / 'directive/source_registry.json').write_text(json.dumps({'sources': [
        {'path': 'directive/source.pdf', 'sha256': '0'*64, 'role': 'required_original'}]}))
    with pytest.raises(RuntimeError, match='source'):
        source_hashes(tmp_path)
