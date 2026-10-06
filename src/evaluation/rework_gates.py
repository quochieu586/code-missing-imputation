"""Pure-numpy gate logic for the E5 rework experiment (preregistration section 11).

Score layout (as produced by ``scripts/run_e5_rework.py``):
  scores['<mode>__<transform>__seed<s>'][scenario] -> evaluate() dict
  scores['init_only__<init>__<scenario>']          -> evaluate() dict
Missing entries FAIL a gate; they are never skipped.
"""
from __future__ import annotations

PRIMARY = ('whole_rare_n2', 'whole_middle_n2', 'whole_common_n2')
EFFICACY_THRESHOLD = 0.10


def _m2(scores, key, scenario):
    value = scores.get(key, {}).get(scenario, {}).get('m2')
    return None if value is None else float(value)


def conditioning_gate(scores, transforms, seeds, scenarios=PRIMARY):
    """Mask-only M2 strictly > No-init M2, per transform x scenario x seed.

    PASS for a (transform, scenario) needs every seed to satisfy the inequality
    and a positive seed-mean difference. The gate needs every pair to pass.
    """
    pairs = {}
    for t in transforms:
        for scn in scenarios:
            rows = {}
            for s in seeds:
                no = _m2(scores, f'no_init__{t}__seed{s}', scn)
                mask = _m2(scores, f'mask_only__{t}__seed{s}', scn)
                ok = no is not None and mask is not None and mask > no
                rows[f'seed{s}'] = {'no_init_m2': no, 'mask_only_m2': mask, 'passed': bool(ok)}
            diffs = [r['mask_only_m2'] - r['no_init_m2'] for r in rows.values()
                     if r['mask_only_m2'] is not None and r['no_init_m2'] is not None]
            mean_diff = sum(diffs) / len(diffs) if diffs else None
            pairs[f'{t}__{scn}'] = {
                'seeds': rows, 'mean_mask_minus_no_init': mean_diff,
                'passed': bool(len(diffs) == len(seeds) and all(r['passed'] for r in rows.values())
                               and mean_diff is not None and mean_diff > 0)}
    return {'passed': bool(pairs and all(v['passed'] for v in pairs.values())), 'pairs': pairs}


def efficacy_gate(scores, transforms, inits, seeds, scenarios=PRIMARY,
                  threshold=EFFICACY_THRESHOLD):
    """Init+CSDI vs same-initializer Init-only: gain strictly > threshold and
    nonzero CLR MAE not worse, for every seed and scenario of a pair.
    Gate PASS = at least one (initializer, transform) pair qualifies."""
    pairs = {}
    for t in transforms:
        for init in inits:
            cells, ok_all = {}, True
            for scn in scenarios:
                base = scores.get(f'init_only__{init}__{scn}', {})
                base_m2, base_mae = base.get('m2'), base.get('clr_mae_nonzero_hidden')
                for s in seeds:
                    sc = scores.get(f'{init}__{t}__seed{s}', {}).get(scn, {})
                    m2, mae = sc.get('m2'), sc.get('clr_mae_nonzero_hidden')
                    gain = (1.0 - m2 / base_m2) if (m2 is not None and base_m2) else None
                    fidelity = bool(mae is not None and base_mae is not None and mae <= base_mae)
                    ok = bool(gain is not None and gain > threshold and fidelity)
                    cells[f'{scn}__seed{s}'] = {
                        'init_only_m2': base_m2, 'diffusion_m2': m2, 'm2_gain_fraction': gain,
                        'nonzero_clr_mae_not_worse': fidelity, 'passed': ok}
                    ok_all = ok_all and ok
            pairs[f'{init}__{t}'] = {'cells': cells, 'passed': bool(ok_all)}
    return {'passed': bool(any(v['passed'] for v in pairs.values())), 'pairs': pairs,
            'threshold_strictly_greater_than': threshold}


def numerical_gate(scores):
    """Finite, nonnegative, observed counts restored exactly for every scored
    method/scenario. Scale-guard flags are recorded elsewhere, never removed."""
    bad = []
    n = 0
    for key, value in scores.items():
        entries = value.items() if _is_scenario_map(value) else [(None, value)]
        for scn, sc in entries:
            n += 1
            if (sc.get('status') != 'OK' or sc.get('nonfinite_cells', 1) != 0
                    or sc.get('negative_cells', 1) != 0
                    or not sc.get('observed_restoration_exact', False)):
                bad.append(f'{key}/{scn}')
    return {'passed': bool(n > 0 and not bad), 'n_checked': n, 'violations': bad}


def _is_scenario_map(value):
    return isinstance(value, dict) and value and all(isinstance(v, dict) for v in value.values()) \
        and 'm2' not in value


def scale_guard_summary(scores):
    """Recorded, not gating (Protocol U engineering flag)."""
    out = {}
    for key, value in scores.items():
        entries = value.items() if _is_scenario_map(value) else [(None, value)]
        for scn, sc in entries:
            diag = sc.get('numerical_diagnostics', {})
            out[f'{key}/{scn}'] = {'n_scale_guard_rows': diag.get('n_scale_guard_rows'),
                                   'max_ratio': diag.get('count_sum_over_visible_sum_max')}
    return out


def validation_gates(scores, transforms, inits, seeds, scenarios=PRIMARY):
    cond = conditioning_gate(scores, transforms, seeds, scenarios)
    eff = efficacy_gate(scores, transforms, inits, seeds, scenarios)
    num = numerical_gate(scores)
    return {'passed': bool(cond['passed'] and eff['passed'] and num['passed']),
            'conditioning_gate': cond, 'efficacy_gate': eff, 'numerical_gate': num,
            'restoration_gate': {'passed': num['passed'],
                                 'note': 'observed exactness folded into numerical gate'},
            'scenarios_gated': list(scenarios)}


def rank_key(scores, init, transform, seeds, scenarios=PRIMARY):
    """Registered selection rule: arithmetic mean over the primary whole-variant
    scenarios of the seed-mean validation M2 (ranking only, not a reported metric)."""
    means = []
    for scn in scenarios:
        vals = [_m2(scores, f'{init}__{transform}__seed{s}', scn) for s in seeds]
        if any(v is None for v in vals):
            return None
        means.append(sum(vals) / len(vals))
    return sum(means) / len(means)


def _tiebreak(scores, init, transform, seeds, scenario='random_cell_bank0'):
    vals = [_m2(scores, f'{init}__{transform}__seed{s}', scenario) for s in seeds]
    return sum(vals) / len(vals) if vals and all(v is not None for v in vals) else float('inf')


def select_finalists(scores, gates, transforms, inits, seeds, n=3, scenarios=PRIMARY):
    """Only when every validation gate passed; candidates must individually
    satisfy the efficacy criterion. Otherwise finalists == [] (negative result)."""
    ranked = []
    for t in transforms:
        for init in inits:
            key = f'{init}__{t}'
            qualified = gates['efficacy_gate']['pairs'].get(key, {}).get('passed', False)
            r = rank_key(scores, init, t, seeds, scenarios)
            if r is not None:
                ranked.append({'config': key, 'mode': init, 'transform': t,
                               'rank_stat': r, 'qualified': bool(qualified),
                               'tiebreak_random_cell_m2': _tiebreak(scores, init, t, seeds)})
    ranked.sort(key=lambda x: (x['rank_stat'], x['tiebreak_random_cell_m2'], x['config']))
    diagnostic = ranked[:n]
    finalists = ([c for c in ranked if c['qualified']][:n] if gates['passed'] else [])
    return finalists, diagnostic
