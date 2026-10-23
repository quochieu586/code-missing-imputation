"""Global raw-cell IDs define masks BEFORE any donor, transform or window."""
import hashlib
import numpy as np

SCENARIOS = ('whole_feature', 'random_cell', 'time_block', 'whole_row')


def make_mask(raw, scenario, seed, rate=.30):
    if scenario not in SCENARIOS or not 0 < rate < 1:
        raise ValueError('Unknown scenario or invalid masking rate')
    observed = np.asarray(raw.original_observed)
    hidden = np.zeros_like(observed)
    # Per-location random stream prevents row order or subset construction
    # from changing a different location's mask.
    for entity in sorted(set(raw.entity_ids)):
        ids = np.flatnonzero(np.asarray(raw.entity_ids) == entity)
        ids = ids[np.argsort(np.asarray(raw.timestamps)[ids], kind='stable')]
        key = f'{scenario}|{seed}|{entity}'.encode()
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], 'little'))
        if scenario == 'whole_feature':
            features = np.flatnonzero(observed[ids].any(axis=0))
            count = min(max(1, int(np.ceil(rate * len(features)))), max(1, len(features) - 1))
            selected = rng.choice(features, count, replace=False)
            hidden[np.ix_(ids, selected)] = True
        elif scenario == 'random_cell':
            hidden[ids] = rng.random(observed[ids].shape) < rate
        else:
            count = max(1, int(np.ceil(rate * len(ids))))
            if scenario == 'time_block':
                start = int(rng.integers(0, len(ids) - count + 1))
                selected = ids[start:start + count]
            else:
                selected = rng.choice(ids, count, replace=False)
            hidden[selected] = True
    return hidden & observed


def mask_hash(mask):
    return hashlib.sha256(np.asarray(mask, dtype=bool).tobytes()).hexdigest()
