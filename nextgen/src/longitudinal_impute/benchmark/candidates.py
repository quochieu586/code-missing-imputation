"""One derivation of preregistered supplementary candidates for run and audit."""
import numpy as np


def supplementary_candidate(config, selection, method):
    if method not in config['supplementary_methods']:
        raise ValueError('Method is not preregistered')
    if method.startswith('hron'):
        params = dict(selection['hron']['candidate']['parameters'], max_iter=50)
        if method == 'hron_lts':
            params.update(lts_starts=config['supplementary_lts_starts'], seed=42)
    elif method.startswith('micoda'):
        params = dict(selection['micoda']['candidate']['parameters'])
    elif method == 'tsagris_basic':
        params = {'k': selection['tsagris']['candidate']['parameters']['k']}
    else:
        params = dict(config['supplementary_cv'])
    return {'method': method, 'parameters': params}


def final_candidate(config, selections, family):
    rankings = []
    for index, candidate in enumerate(config['candidates'][family]):
        scores = [selection['validation'][family][index].get('ratio_mae') for selection in selections]
        if all(score is not None for score in scores):
            rankings.append((float(np.mean(scores)), index, candidate))
    return min(rankings)[2]
