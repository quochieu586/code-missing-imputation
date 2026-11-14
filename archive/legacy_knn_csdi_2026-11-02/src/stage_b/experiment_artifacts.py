"""Validated append-only attempts for legacy experiment checkpoint/prediction caches.

Artifacts are written and read back before publishing a completion manifest.
Incomplete/corrupt attempts remain on disk and never qualify for resume.
"""
from pathlib import Path
import hashlib
import json
import os
import pickle
import shutil
import uuid
import numpy as np
import torch


class CacheConfigurationMismatch(RuntimeError):
    """A completed attempt belongs to another experiment configuration."""


def _hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _cache_sha(out_dir):
    return _hash(Path(out_dir) / 'config.json')


def _base(out_dir, kind, key):
    if not key or Path(key).name != key or '/' in key or '\\' in key or key in ('.', '..'):
        raise ValueError('Cache key must be a single filename component')
    return Path(out_dir) / ('cache_' + kind) / key


def _validate(path, kind):
    if kind == 'train':
        data = torch.load(path, map_location='cpu', weights_only=True)
        if set(data) != {'state_dict_280', 'state_dict_1120'}:
            raise ValueError('Incomplete checkpoint budgets')
        for state in data.values():
            if not isinstance(state, dict) or not state:
                raise ValueError('Empty checkpoint state')
            for tensor in state.values():
                if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
                    raise ValueError('Invalid checkpoint tensor')
        return sorted(data)
    with np.load(path, allow_pickle=False) as data:
        if not data.files or any(not np.isfinite(data[k]).all() for k in data.files):
            raise ValueError('Invalid prediction arrays')
        return {k: list(data[k].shape) for k in data.files}


def _validate_metadata(blob, kind, schema):
    if kind == 'train':
        for budget in (280, 1120):
            cp = blob['checkpoints'][f'budget{budget}']
            if not 1 <= cp['best_epoch'] <= budget // 7 or not np.isfinite(cp['best_loss']):
                raise ValueError('Invalid checkpoint selection metadata')
        mean = np.asarray(blob['mean'], float)
        if mean.ndim != 1 or not len(mean) or not np.isfinite(mean).all():
            raise ValueError('Invalid training mean')
        if not np.isfinite(blob['scale']) or blob['scale'] <= 0:
            raise ValueError('Invalid training scale')
        if not isinstance(blob['extra'], dict) or not isinstance(blob['trace'], list):
            raise ValueError('Invalid training provenance')
    elif {f'pred__{k}' for k in blob['scores']} != set(schema):
        raise ValueError('Score and prediction scenarios differ')


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value))


def _load(out_dir, kind, key):
    base = _base(out_dir, kind, key)
    for manifest in sorted(base.glob('attempt-*/complete.json'), reverse=True):
        try:
            blob = json.loads(manifest.read_text(encoding='utf-8'))
            if blob['key'] != key or blob['kind'] != kind:
                continue
            if blob['config_sha256'] != _cache_sha(out_dir):
                raise CacheConfigurationMismatch(f'stale cache for {key}: config changed; use a new --out-dir')
            artifact = manifest.parent / ('checkpoint.pt' if kind == 'train' else 'predictions.npz')
            if _hash(artifact) != blob['artifact_sha256'] or _validate(artifact, kind) != blob['schema']:
                continue
            _validate_metadata(blob, kind, blob['schema'])
            return {**blob, 'artifact_path': str(artifact)}
        except CacheConfigurationMismatch:
            raise
        except (OSError, ValueError, KeyError, TypeError, EOFError, pickle.UnpicklingError, RuntimeError):
            continue
    # Validate and seal usable legacy artifacts instead of silently retraining.
    legacy_index = base.parent / (key + '.json')
    legacy_artifact = base.parent / (key + ('.pt' if kind == 'train' else '.npz'))
    if legacy_index.exists() and legacy_artifact.exists():
        try:
            payload = json.loads(legacy_index.read_text(encoding='utf-8'))
            if payload.get('config_sha256') != _cache_sha(out_dir):
                raise CacheConfigurationMismatch(f'stale cache for {key}: config changed')
            schema = _validate(legacy_artifact, kind)
            _validate_metadata(payload, kind, schema)
            _save(out_dir, kind, key, {**payload, 'migration': 'validated_legacy_unsealed',
                                     'legacy_sha256_at_migration': _hash(legacy_artifact)},
                  lambda p: shutil.copyfile(legacy_artifact, p))
            return _load(out_dir, kind, key)
        except CacheConfigurationMismatch:
            raise
        except (OSError, ValueError, KeyError, TypeError, EOFError, pickle.UnpicklingError, RuntimeError):
            pass
    return None


def _save(out_dir, kind, key, payload, writer):
    attempt = _base(out_dir, kind, key) / ('attempt-' + uuid.uuid4().hex)
    attempt.mkdir(parents=True, exist_ok=False)
    path = attempt / ('checkpoint.pt' if kind == 'train' else 'predictions.npz')
    writer(path)
    with path.open('rb+') as stream:
        os.fsync(stream.fileno())
    schema = _validate(path, kind)
    _validate_metadata(payload, kind, schema)
    blob = {**payload, 'key': key, 'kind': kind, 'config_sha256': _cache_sha(out_dir),
            'artifact_sha256': _hash(path), 'schema': schema, 'schema_version': 1}
    pending = attempt / 'manifest.pending'
    with pending.open('x', encoding='utf-8') as stream:
        json.dump(blob, stream, indent=2, allow_nan=False, default=_json_default)
        stream.flush()
        os.fsync(stream.fileno())
    pending.rename(attempt / 'complete.json')


def train_cache_load(out_dir, key):
    return _load(out_dir, 'train', key)


def train_cache_save(out_dir, key, checkpoints, trace, mean, scale, extra):
    payload = {'checkpoints': {k: {f: checkpoints[k][f] for f in ('best_epoch', 'best_loss')}
                               for k in ('budget280', 'budget1120')},
               'trace': trace, 'mean': mean.tolist(), 'scale': float(scale), 'extra': extra}
    states = {f'state_dict_{b}': checkpoints[f'budget{b}']['state_dict'] for b in (280, 1120)}
    _save(out_dir, 'train', key, payload, lambda p: torch.save(states, p))


def eval_cache_load(out_dir, key):
    return _load(out_dir, 'eval', key)


def eval_cache_save(out_dir, key, scores, preds):
    if set(scores) != set(preds):
        raise ValueError('Every scored scenario requires a prediction artifact')
    _save(out_dir, 'eval', key, {'scores': scores},
          lambda p: np.savez_compressed(p, **{f'pred__{k}': v for k, v in preds.items()}))


def _train_cache_path(out_dir, key):
    blob = train_cache_load(out_dir, key)
    if blob is None:
        raise FileNotFoundError('No valid checkpoint attempt: ' + key)
    return Path(blob['artifact_path'])
