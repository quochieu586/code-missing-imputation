"""Evaluation v1 logic (step 8): configuration, unused seeds, sign test, failure policy, hypotheses.

Frozen with eval_v1.toml before any confirmation mask was drawn. No score is computed here: the
numbers come from tier01 (tier 1 runs) and tier3 (tier 3 verdicts); this module only decides.
"""
from math import comb
import json
from pathlib import Path
import hashlib
import tomllib

TOP = {'version', 'data', 'masks', 'models', 'runs', 'tier0', 'tier1', 'tier2', 'tier3', 'diagnostics', 'hypotheses'}
SECTIONS = {'data': {'data', 'schema', 'group_map', 'rate'},
            'masks': {'primary', 'sensitivity', 'seeds', 'target', 'previous'},
            'models': {'real', 'decoys', 'tier1_only'},
            'tier0': {'max_failed_seeds'},
            'tier1': {'primary', 'secondary', 'strata', 'n_cuts_from', 'min_stratum_cells', 'pairs'},
            'tier2': {'references', 'compare_with'},
            'tier3': {'spec'},
            'diagnostics': {'D1', 'D2', 'D3'}}
RUN_KEYS = {'models_file', 'tables', 'rename'}
HYPOTHESIS_KEYS = {'pair_better': {'id', 'text', 'kind', 'metric', 'a', 'b', 'min_wins'},
                   'pair_not_better': {'id', 'text', 'kind', 'metric', 'a', 'b', 'max_wins'},
                   'tier3': {'id', 'text', 'kind', 'use_case', 'expect'}}
TIER3_USES = {'u1_under', 'u1_over', 'u2', 'u3'}
OUTCOMES = {'confirmed': 'xác nhận', 'not_confirmed': 'không xác nhận', 'insufficient': 'không đủ support'}


def exact_keys(where, got, expected):
    if set(got) != set(expected):
        raise ValueError(f'{where}: unknown keys {sorted(set(got) - set(expected))}, '
                         f'missing keys {sorted(set(expected) - set(got))}')


# Amendment of eval_v1.toml after the rebase on main with PR #5 (repository restructure, MICoDa default
# A18). eval_v1.toml was frozen before the confirmation masks were drawn; masks_manifest.json keeps the
# sha256 of that frozen file. The amendment only moves the paths (nextgen/ -> root, models_default ->
# models_default_a17 with MICoDa A17 explicit) and adds a comment block between AMENDMENT markers.
# frozen_bytes() removes the block and moves the paths back, so the freeze check still compares the
# frozen file byte for byte and fails on any other edit.
AMENDMENT = ('# --- amendment (rebase on PR #5) ---\n', '# --- end of amendment ---\n')
PATH_MOVES = (('"configs/models_default_a17.toml"', '"nextgen/configs/models_default.toml"'),
              ('"configs/', '"nextgen/configs/'), ('"src/', '"nextgen/src/'))


def frozen_bytes(path):
    """eval_v1.toml as frozen: without the amendment block, paths moved back. A file without the
    block is returned unchanged."""
    text = Path(path).read_bytes().decode('utf-8')
    start, end = AMENDMENT
    if start not in text:
        return text.encode('utf-8')
    if text.count(start) != 1 or text.count(end) != 1 or text.index(end) < text.index(start):
        raise ValueError('eval_v1.toml: malformed amendment block')
    text = text[:text.index(start)] + text[text.index(end) + len(end):]
    for new, old in PATH_MOVES:
        text = text.replace(new, old)
    return text.encode('utf-8')


def frozen_sha256(path):
    return hashlib.sha256(frozen_bytes(path)).hexdigest()


def read_config(path):
    """eval_v1.toml, checked: any unknown or missing key raises before anything else happens."""
    with Path(path).open('rb') as stream:
        config = tomllib.load(stream)
    exact_keys('top level', config, TOP)
    for name, keys in SECTIONS.items():
        exact_keys(f'[{name}]', config[name], keys)
    if set(config['runs']) != {'main', 'micoda_other', 'residual'}:
        raise ValueError(f"[runs] must be main, micoda_other, residual, got {sorted(config['runs'])}")
    for name, run in config['runs'].items():
        exact_keys(f'[runs.{name}]', run, RUN_KEYS)
    models = set(config['models']['real']) | set(config['models']['decoys']) | set(config['models']['tier1_only'])
    produced = {run['rename'].get(t, t) for run in config['runs'].values() for t in run['tables']}
    if models != produced:
        raise ValueError(f'[models] {sorted(models)} differ from the models the runs produce {sorted(produced)}')
    seeds = config['masks']['seeds']
    if len(set(seeds)) != len(seeds) or any(not isinstance(s, int) or s < 0 for s in seeds):
        raise ValueError('[masks] seeds must be distinct nonnegative integers')
    ids = [h.get('id') for h in config['hypotheses']]
    if len(set(ids)) != len(ids):
        raise ValueError(f'hypothesis ids repeat: {ids}')
    for h in config['hypotheses']:
        if h.get('kind') not in HYPOTHESIS_KEYS:
            raise ValueError(f"{h.get('id')}: kind must be one of {sorted(HYPOTHESIS_KEYS)}")
        exact_keys(f"hypothesis {h['id']}", h, HYPOTHESIS_KEYS[h['kind']])
        if h['kind'] == 'tier3' and (h['use_case'] not in TIER3_USES or h['expect'] not in ('pass', 'fail')):
            raise ValueError(f"{h['id']}: use_case in {sorted(TIER3_USES)} and expect pass / fail")
        if h['kind'] != 'tier3' and ({h['a'], *h['b']} - models):
            raise ValueError(f"{h['id']}: unknown models {sorted({h['a'], *h['b']} - models)}")
    return config


# ---------- seeds already used ----------

def json_seeds(obj, out=None):
    """Every value under a key 'seeds' (list), 'seed' (scalar) or the keys of 'mask_sha256', as ints."""
    out = set() if out is None else out
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == 'seeds' and isinstance(value, list):
                out.update(int(v) for v in value if str(v).isdigit())
            elif key == 'seed' and str(value).isdigit():
                out.add(int(value))
            elif key == 'mask_sha256' and isinstance(value, dict):
                out.update(int(k) for k in value if str(k).isdigit())
            json_seeds(value, out)
    elif isinstance(obj, list):
        for value in obj:
            json_seeds(value, out)
    return out


def used_seeds(root, patterns):
    """{seed: [files]} over the JSON files matching the patterns under root."""
    found = {}
    for pattern in patterns:
        for path in sorted(Path(root).glob(pattern)):
            for seed in json_seeds(json.loads(path.read_text(encoding='utf-8'))):
                found.setdefault(seed, []).append(str(path.relative_to(root)))
    return found


def check_unused(seeds, root, patterns):
    """Refuse confirmation seeds that a previous step already used; returns the files read."""
    found = used_seeds(root, patterns)
    clash = sorted(set(seeds) & set(found))
    if clash:
        raise ValueError(f'confirmation seeds already used: {clash} (e.g. {found[clash[0]][:3]})')
    return sorted({f for files in found.values() for f in files})


# ---------- statistics and decisions ----------

def sign_test(wins, losses):
    """Exact two-sided binomial sign test, p = 0.5, ties already dropped; None when n = 0."""
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    return min(1., 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def failed_seeds(runs, model, seeds):
    """Seeds where the model does not count (not completed with coverage 1, or no run)."""
    return [s for s in seeds if not runs.get(model, {}).get(s, {}).get('eligible')]


def pair_outcome(p, kind, threshold):
    """Outcome of one pair from a tier01.pair dict (wins = A strictly smaller)."""
    if kind == 'pair_better':
        ok = p['mean_diff'] is not None and p['mean_diff'] < 0 and p['wins'] >= threshold
    else:
        ok = p['n'] > 0 and p['wins'] <= threshold
    return 'confirmed' if ok else 'not_confirmed'


def judge_pairs(h, runs, seeds, max_failed, pair):
    """pair_better / pair_not_better: one outcome per B, then combined (any not confirmed wins,
    then any insufficient). pair is tier01.pair."""
    parts = {}
    for b in h['b']:
        if any(len(failed_seeds(runs, m, seeds)) > max_failed for m in (h['a'], b)):
            parts[b] = ('insufficient', None)
            continue
        p = pair(runs, h['a'], b, seeds, h['metric'])
        p['sign_p'] = sign_test(p['wins'], p['n'] - p['wins'] - p['ties'])
        parts[b] = (pair_outcome(p, h['kind'], h['min_wins'] if h['kind'] == 'pair_better' else h['max_wins']), p)
    outcomes = [o for o, _ in parts.values()]
    overall = ('not_confirmed' if 'not_confirmed' in outcomes else
               'insufficient' if 'insufficient' in outcomes else 'confirmed')
    return overall, parts


def judge_tier3(h, verdict, real_failures, max_failed):
    """tier3: verdict of the use case on one mask ('pass' / 'fail' / 'insufficient')."""
    if any(n > max_failed for n in real_failures.values()) or verdict == 'insufficient':
        return 'insufficient'
    return 'confirmed' if verdict == h['expect'] else 'not_confirmed'


def combine_masks(verdicts):
    """Over the three masks: pass if all pass, fail if any fails, else insufficient."""
    values = list(verdicts.values())
    return 'pass' if all(v == 'pass' for v in values) else 'fail' if 'fail' in values else 'insufficient'
