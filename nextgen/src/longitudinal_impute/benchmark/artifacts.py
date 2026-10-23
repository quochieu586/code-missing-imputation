"""Canonical immutable artifacts, retained attempts and leaf validation."""
import hashlib
import json
import os
from pathlib import Path
import uuid
import numpy as np
from .masks import mask_hash
from .protocol import digest


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def publish_json(path, value):
    """Publish last, atomically; never replace an existing marker."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.pending')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.rename(path)


def new_attempt(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    attempts = directory / 'attempts'
    attempts.mkdir(exist_ok=True)
    attempt = attempts / f'attempt-{uuid.uuid4().hex}'
    attempt.mkdir()
    return attempt


def artifact_path(directory, record, field, legacy_name):
    relative = Path(record.get(field, legacy_name))
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'Invalid {field} path in completion record')
    root = Path(directory).resolve()
    artifact = (root / relative).resolve()
    try:
        artifact.relative_to(root)
    except ValueError as exc:
        raise ValueError(f'{field} escapes its job directory') from exc
    return artifact


def prediction_artifact_path(job, record):
    return artifact_path(job, record, 'predictions_artifact', 'predictions.npz')


def final_csv_artifact_path(directory, record):
    return artifact_path(directory, record, 'csv_artifact', 'completed.csv')


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def get_mask(output, scenario, seed):
    output = Path(output)
    with np.load(output / 'masks' / f'{scenario}_{seed}.npz', allow_pickle=False) as values:
        hidden = values['hidden'].copy()
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    if mask_hash(hidden) != manifest['mask_registry'][f'{scenario}_{seed}']['hash']:
        raise ValueError('Frozen mask hash mismatch')
    return hidden


def evaluation_identity(output, job, ids, scenario, seed, hidden):
    output, job = Path(output).resolve(), Path(job).resolve()
    try:
        job_key = job.relative_to(output).as_posix()
    except ValueError as exc:
        raise ValueError('Evaluation job must be inside its run directory') from exc
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    return digest({'job': job_key, 'config_hash': manifest['config_hash'],
                   'scenario': scenario, 'mask_seed': int(seed),
                   'mask_hash': mask_hash(hidden), 'row_indices': [int(i) for i in ids]})


def read_completed_evaluation(job, ids, scenario, seed, hidden, identity, feature_count):
    marker = Path(job) / 'result.json'
    if not marker.exists():
        return None
    record = json.loads(marker.read_text(encoding='utf-8'))
    expected = {'status': 'ok', 'scenario': scenario, 'mask_seed': int(seed),
                'mask_hash': mask_hash(hidden), 'rows': len(ids), 'job_identity': identity}
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError(f'Completed evaluation identity mismatch: {marker}')
    predictions = prediction_artifact_path(job, record)
    if not predictions.is_file() or file_hash(predictions) != record.get('predictions_hash'):
        raise ValueError(f'Completed evaluation prediction artifact is missing or corrupt: {marker}')
    with np.load(predictions, allow_pickle=False) as saved:
        if not np.array_equal(saved['row_indices'], np.asarray(ids, dtype=int)):
            raise ValueError(f'Completed evaluation row identities mismatch: {marker}')
        counts = saved['counts']
        if counts.shape != (len(ids), feature_count) or not np.isfinite(counts).all():
            raise ValueError(f'Completed evaluation prediction shape/values mismatch: {marker}')
    return record
