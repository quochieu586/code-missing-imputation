"""Location-grouped five-fold split; stratify only on reporting completeness."""
import hashlib
import json
import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def make_folds(raw, n_folds=5, seed=20261015):
    entities = np.asarray(raw.entity_ids)
    locations = sorted(set(entities))
    complete = np.asarray(raw.original_observed).all(axis=1)
    complete_locations = sorted(set(entities[complete]))
    if n_folds < 3 or len(complete_locations) < n_folds:
        raise ValueError('Need >=3 folds and at least one complete-data location in each')
    rng = np.random.default_rng(seed)
    special = list(rng.permutation(complete_locations))
    folds = [[] for _ in range(n_folds)]
    counts = [0] * n_folds
    for index, location in enumerate(special):
        f = index % n_folds
        folds[f].append(str(location)); counts[f] += int((entities == location).sum())
    rest = [x for x in locations if x not in complete_locations]
    # Random tie-break followed by decreasing panel length, greedily balance rows.
    rest = list(rng.permutation(rest))
    rest.sort(key=lambda key: -int((entities == key).sum()))
    for key in rest:
        f = min(range(n_folds), key=lambda j: (counts[j], len(folds[j]), j))
        folds[f].append(str(key)); counts[f] += int((entities == key).sum())
    records = []
    for f in range(n_folds):
        test, validation = set(folds[f]), set(folds[(f + 1) % n_folds])
        train = set(locations) - test - validation
        records.append({'fold': f, 'train_locations': sorted(train),
                        'validation_locations': sorted(validation), 'test_locations': sorted(test),
                        'train_rows': np.flatnonzero(np.isin(entities, list(train))).tolist(),
                        'validation_rows': np.flatnonzero(np.isin(entities, list(validation))).tolist(),
                        'test_rows': np.flatnonzero(np.isin(entities, list(test))).tolist()})
    return {'seed': seed, 'stratification': 'original_observation_completeness_only',
            'folds': records, 'test_row_counts': counts}
