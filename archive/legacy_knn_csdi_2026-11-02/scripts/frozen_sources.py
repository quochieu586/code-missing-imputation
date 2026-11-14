"""Pinned scientific inputs, including canonical-to-historical path identities."""
import hashlib
import json
from pathlib import Path


def source_hashes(root, *, historical=False):
    root = Path(root).resolve()
    registry = json.loads((root / 'directive/source_registry.json').read_text(encoding='utf-8'))
    result = {}
    for entry in registry['sources']:
        path = (root / entry['path']).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise RuntimeError('Required scientific source absent: ' + entry['path'])
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != entry['sha256']:
            raise RuntimeError('Scientific source hash mismatch: ' + entry['path'])
        if historical and 'legacy_path' not in entry:
            continue
        identity = entry.get('legacy_path', entry['path']) if historical else entry['path']
        result[str(Path(identity))] = actual
    return result


def historical_snapshot(current, baseline, root):
    """Project verified migrated sources onto a historical snapshot's identities.

    New registered inputs are validated by source_hashes, but cannot be claimed
    as members of an older snapshot. Missing historical inputs fail explicitly.
    """
    projected = {k: v for k, v in current.items() if not k.startswith('directive')}
    migrated = source_hashes(root, historical=True)
    for key in baseline:
        if key.startswith('directive'):
            if key not in migrated:
                raise RuntimeError('Unmapped historical scientific source: ' + key)
            projected[key] = migrated[key]
    return projected
