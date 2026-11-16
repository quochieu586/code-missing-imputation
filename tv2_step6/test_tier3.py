"""Tests of the step 6 table logic (verdicts per mask, U2 regret rule, U4 counts) on toy cases.

    PYTHONPATH=src python -W error::RuntimeWarning -m pytest tv2_step6/test_tier3.py
"""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('tier3', HERE / 'tier3.py')
tier3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tier3)


def pairs(rate, n=10):
    match = round(rate * n) if rate is not None else 0
    return {'rate': rate, 'match': match, 'mismatch': n - match if rate is not None else 0, 'tie': 0, 'na': 0}


def case(seed, rate, regret):
    u = {'u2': pairs(rate), 'u2_raw': pairs(rate), 'u2_full': pairs(rate), 'regret': ('a', regret),
         'regret_raw': ('a', regret), 'regret_full': ('a', regret),
         'u1_under': pairs(rate), 'u1_over': pairs(rate)}
    return {'seed': str(seed), 'u': {'local': u}}


def cases(rates, regrets):
    return {mask: [case(s, r, g) for s, (r, g) in enumerate(zip(rates[k * 5:k * 5 + 5], regrets[k * 5:k * 5 + 5]))]
            for k, mask in enumerate(tier3.MASKS)}


def test_u2_needs_both_the_rate_rule_and_the_regret_rule():
    good = cases([.9] * 10, [0.] * 9 + [.04])
    assert tier3.u2_table(good)[1] == 'pass'
    assert tier3.u2_table(cases([.9] * 10, [0.] * 8 + [.04, .04]))[1] == 'fail'
    assert tier3.u2_table(cases([.9] * 9 + [.4], [0.] * 10))[1] == 'fail'


def test_rate_verdict_is_per_mask_and_na_cases_mean_insufficient_support():
    lines, verdicts = tier3.u1_table(cases([1.] * 5 + [.7] * 5, [0.] * 10))
    assert verdicts == {'u1_under': 'fail', 'u1_over': 'fail'}          # random_cell mean 0.7 < 0.8
    lines, verdicts = tier3.u1_table(cases([1.] * 8 + [None, None], [0.] * 10))
    assert verdicts['u1_under'] == 'insufficient' and 'random_cell: TB 1.00, min 1.00 (3 ca)' in lines[-1]


def test_u4_counts_only_cases_with_a_value():
    def toy(seed, worse):
        return {'seed': str(seed), 'u4': {k: {d: worse if k != 'd2_dist' or d != 'zero_fill' else None
                                              for d in tier3.t3.DECOYS} for k in tier3.U4_LABELS},
                'u4_scores': {k: {} for k in tier3.U4_LABELS}}
    table = tier3.u4_table({m: [toy(s, s < 3) for s in range(5)] for m in tier3.MASKS})
    row = next(line for line in table if line.startswith('| zero_fill | whole_feature |'))
    assert row == '| zero_fill | whole_feature | 3/5 | 3/5 | NA | 3/5 | 3/5 | 3/5 |'
    assert tier3.ms([.1, None, .3]) == '0.2000 ± 0.1414 (n = 2)'


def test_one_case_with_a_failed_real_model_marks_its_pairs_na():
    """Step 8 bug: a real model that failed in a case is absent, its pairs are NA (not a KeyError)."""
    import numpy as np
    import sys
    sys.path.insert(0, str(tier3.REPO / 'tests'))
    from test_tier2_diagnostics import panel
    data = panel(seed=3)
    rng = np.random.default_rng(0)
    hidden = np.isfinite(data.counts) & (rng.random(data.counts.shape) < .15)
    facts = tier3.t2.row_facts(data)
    preds, runs = {}, {}
    for k, m in enumerate(tier3.MODELS):
        ok = m != 'hron'
        runs[m] = {'0': {'eligible': ok, 'mae_prop': .1 + .01 * k if ok else None}}
        if ok:
            preds[m, '0'] = np.where(hidden, rng.integers(0, 4, data.counts.shape), np.nan)
    case = tier3.one_case(data, facts, hidden, preds, runs, '0')[0]
    assert 'hron' not in case['models_ok']
    pairs = case['u']['local']['u2']['pairs']
    assert all(s == 'na' for a, b, s in pairs if 'hron' in (a, b))
    assert case['u']['local']['u2']['na'] == 5 and case['u']['local']['u2']['match'] + case['u']['local']['u2'][
        'mismatch'] + case['u']['local']['u2']['tie'] == 10
    assert all(v is None for v in (case['u4']['d1_dist'][d] for d in tier3.t3.DECOYS))   # a real model is missing
