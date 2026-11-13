"""Tests for E5 mask-conditioning experiment (§11.1–11.5).

Covers R4: mask constant per location, 9-stratum coverage, paired banks,
budget/checkpoint replay, weighted fixture, observed locking, gate computation
with policy prefix.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
import rework_runtime  # noqa: F401,E402

import run_e5_mask_conditioning as M  # noqa: E402
from src.evaluation import rework_gates as RG  # noqa: E402
from src.stage_b.research_csdi import schedule  # noqa: E402

SEEDS = M.ALL_SEEDS
SCN = RG.PRIMARY


def sc(m2, mae=1.0, **kw):
    d = {'m2': m2, 'clr_mae_nonzero_hidden': mae, 'status': 'OK', 'nonfinite_cells': 0,
         'negative_cells': 0, 'observed_restoration_exact': True}
    d.update(kw)
    return d


# ---------------------------------------------------------------------------
# Training mask generation
# ---------------------------------------------------------------------------

class FakePool:
    def __init__(self, n_rows=100, n_cols=17, seed=0):
        rng = np.random.default_rng(seed)
        self.data = rng.integers(1, 50, size=(n_rows, n_cols)).astype(float)
        self.columns = [f'v{i}' for i in range(n_cols)]
        import pandas as pd
        self.df = pd.DataFrame(self.data, columns=self.columns)
        self.df['location'] = ['NL'] * (n_rows // 2) + ['UK'] * (n_rows - n_rows // 2)


def test_control_generates_18_random_cell_banks():
    pool = FakePool(n_rows=50)
    banks = M.make_training_mask_banks('control', pool.df, pool.columns, 50)
    assert len(banks) == M.MASK_BANKS
    for b in banks:
        assert b.shape == (50, M.N_VARIANTS)
        assert b.dtype == bool
        n_hidden_per_row = (~b).sum(axis=1)
        assert np.all(n_hidden_per_row == M.MASKED_PARTS)


def test_intervention_generates_9_paired_plus_9_whole_variant():
    pool = FakePool(n_rows=50)
    banks = M.make_training_mask_banks('intervention', pool.df, pool.columns, 50)
    assert len(banks) == M.MASK_BANKS
    for i in range(9, 18):
        visible = banks[i]
        hidden_cols = np.where(~visible[0])[0]
        assert len(hidden_cols) in (1, 2, 3)
        assert np.all(np.array_equal(~visible[0], ~visible[j]) for j in range(1, 50))


def test_paired_banks_are_identical():
    pool = FakePool(n_rows=50)
    ctrl = M.make_training_mask_banks('control', pool.df, pool.columns, 50)
    intv = M.make_training_mask_banks('intervention', pool.df, pool.columns, 50)
    M.assert_paired_banks(ctrl, intv)
    with pytest.raises(AssertionError):
        intv[0] = ~intv[0]
        M.assert_paired_banks(ctrl, intv)


def test_whole_variant_banks_cover_all_9_strata():
    pool = FakePool(n_rows=50)
    intv = M.make_training_mask_banks('intervention', pool.df, pool.columns, 50)
    M.assert_n_strata_covered(intv)


def test_assert_mask_constant_per_location_passes_for_uniform():
    hidden = np.zeros((50, 17), dtype=bool)
    hidden[:, 3] = True
    hidden[:, 7] = True
    M.assert_mask_constant_per_location(hidden)


def test_assert_mask_constant_per_location_fails_for_varying():
    hidden = np.zeros((50, 17), dtype=bool)
    hidden[0, 3] = True
    hidden[1, 5] = True
    with pytest.raises(AssertionError, match='mask varies across rows'):
        M.assert_mask_constant_per_location(hidden)


def test_unknown_policy_raises():
    pool = FakePool(n_rows=10)
    with pytest.raises(ValueError, match='unknown policy'):
        M.make_training_mask_banks('bogus', pool.df, pool.columns, 10)


# ---------------------------------------------------------------------------
# Gate computation with policy prefix
# ---------------------------------------------------------------------------

def _build_policy_scores(no=10., mask=11., init_only=10., csdi=8.,
                         mae_csdi=1.0, mae_init=1.0, policies=M.POLICIES):
    all_scores = {}
    for policy in policies:
        for mode in M.ALL_MODES:
            for seed in M.ALL_SEEDS:
                key = M.policy_key(policy, mode, seed)
                all_scores[key] = {
                    'budget280': {c: sc(no if mode == 'no_init' else
                                        mask if mode == 'mask_only' else
                                        csdi, mae_csdi) for c in SCN},
                    'budget1120': {c: sc(no if mode == 'no_init' else
                                         mask if mode == 'mask_only' else
                                         csdi, mae_csdi) for c in SCN},
                }
    control_scores = {}
    for im in M.INIT_MODES:
        for c in SCN:
            control_scores[f'init_only__{im}__{c}'] = sc(init_only, mae_init)
    return all_scores, control_scores


def test_gate_computation_per_policy_conditioning_pass():
    all_scores, control_scores = _build_policy_scores(no=10., mask=11., csdi=5.)
    budgets = {'budget280': '280', 'budget1120': '1120'}
    gates = {}
    for budget_name in budgets:
        policy_gates = {}
        for policy in M.POLICIES:
            budget_scores = {}
            for mode in M.ALL_MODES:
                for seed in M.ALL_SEEDS:
                    key = M.policy_key(policy, mode, seed)
                    scn_scores = all_scores.get(key, {}).get(budget_name, {})
                    if scn_scores:
                        gate_key = f'{mode}__clr16__seed{seed}'
                        budget_scores[gate_key] = scn_scores
            budget_scores.update(control_scores)
            cond = RG.conditioning_gate(budget_scores, M.ALL_TRANSFORMS, M.ALL_SEEDS)
            eff = RG.efficacy_gate(budget_scores, M.ALL_TRANSFORMS, M.INIT_MODES, M.ALL_SEEDS)
            num = RG.numerical_gate(budget_scores)
            policy_gates[policy] = {
                'conditioning_gate': cond,
                'efficacy_gate': eff,
                'numerical_gate': num,
            }
        cond_pass = all(g['conditioning_gate']['passed'] for g in policy_gates.values())
        eff_pass = any(g['efficacy_gate']['passed'] for g in policy_gates.values())
        num_pass = all(g['numerical_gate']['passed'] for g in policy_gates.values())
        gates[budget_name] = {
            'passed': bool(cond_pass and eff_pass and num_pass),
            'per_policy': policy_gates,
        }
    assert gates['budget280']['passed']
    assert gates['budget1120']['passed']


def test_gate_conditioning_fails_if_one_policy_fails():
    all_scores, control_scores = _build_policy_scores(no=10., mask=11., csdi=5.)
    for seed in M.ALL_SEEDS:
        key = M.policy_key('control', 'mask_only', seed)
        for c in SCN:
            all_scores[key]['budget280'][c]['m2'] = 9.0
    budgets = {'budget280': '280'}
    for budget_name in budgets:
        policy_gates = {}
        for policy in M.POLICIES:
            budget_scores = {}
            for mode in M.ALL_MODES:
                for seed in M.ALL_SEEDS:
                    key = M.policy_key(policy, mode, seed)
                    scn_scores = all_scores.get(key, {}).get(budget_name, {})
                    if scn_scores:
                        gate_key = f'{mode}__clr16__seed{seed}'
                        budget_scores[gate_key] = scn_scores
            budget_scores.update(control_scores)
            cond = RG.conditioning_gate(budget_scores, M.ALL_TRANSFORMS, M.ALL_SEEDS)
            policy_gates[policy] = {'conditioning_gate': cond}
        cond_pass = all(g['conditioning_gate']['passed'] for g in policy_gates.values())
        assert not cond_pass
        assert not policy_gates['control']['conditioning_gate']['passed']
        assert policy_gates['intervention']['conditioning_gate']['passed']


# ---------------------------------------------------------------------------
# Cache roundtrip
# ---------------------------------------------------------------------------

def test_train_cache_roundtrip_and_stale_detection(tmp_path):
    out_dir = tmp_path / 'exp'
    out_dir.mkdir()
    (out_dir / 'config.json').write_text('{"x": 1}')
    key = 'control_no_init__clr16__seed42'
    assert M.train_cache_load(out_dir, key) is None
    checkpoints = {
        'budget280': {'best_epoch': 10, 'best_loss': 0.5, 'state_dict': {'w': torch.zeros(1)}},
        'budget1120': {'best_epoch': 50, 'best_loss': 0.3, 'state_dict': {'w': torch.ones(1)}},
    }
    M.train_cache_save(out_dir, key, checkpoints, [{'epoch': 1}], np.zeros(17), 1.0,
                       {'policy': 'control', 'mode': 'no_init', 'seed': 42})
    blob = M.train_cache_load(out_dir, key)
    assert blob['checkpoints']['budget280']['best_epoch'] == 10
    assert blob['checkpoints']['budget1120']['best_epoch'] == 50
    (out_dir / 'config.json').write_text('{"x": 2}')
    with pytest.raises(RuntimeError, match='stale cache'):
        M.train_cache_load(out_dir, key)


def test_eval_cache_roundtrip(tmp_path):
    out_dir = tmp_path / 'exp'
    out_dir.mkdir()
    (out_dir / 'config.json').write_text('{"x": 1}')
    key = 'control_no_init__clr16__seed42_budget280'
    assert M.eval_cache_load(out_dir, key) is None
    scores = {'whole_rare_n2': sc(5.0)}
    preds = {'whole_rare_n2': np.ones((2, 17))}
    M.eval_cache_save(out_dir, key, scores, preds)
    blob = M.eval_cache_load(out_dir, key)
    assert blob['scores']['whole_rare_n2']['m2'] == 5.0


# ---------------------------------------------------------------------------
# Dual-budget checkpoint selection
# ---------------------------------------------------------------------------

def test_dual_budget_checkpoint_boundaries():
    rng = np.random.default_rng(0)
    target = rng.normal(size=(24, 16))
    windows = [np.arange(i, min(i + 8, 24)) for i in range(0, 24, 8)]
    times = rng.normal(size=24)
    cond = rng.normal(size=(24, 85))
    cond_banks = {('no_init', b): cond for b in range(M.MASK_BANKS)}
    _, alpha_bar = schedule()
    losses = [100 - i * 0.5 for i in range(M.BUDGET_1120_EPOCHS)]
    calls = iter(losses)
    model, mean, scale, checkpoints, trace, _ = M.fit_with_dual_budget(
        target, cond_banks, 'no_init', windows, times, alpha_bar,
        16, 16, 42, lambda m: float(next(calls)), M.BUDGET_1120_EPOCHS)
    assert 1 <= checkpoints['budget280']['best_epoch'] <= M.BUDGET_280_EPOCHS
    assert 1 <= checkpoints['budget1120']['best_epoch'] <= M.BUDGET_1120_EPOCHS
    assert checkpoints['budget280']['best_epoch'] <= checkpoints['budget1120']['best_epoch']
    assert len(trace) == M.BUDGET_1120_EPOCHS


def test_dual_budget_earliest_argmin_at_ties():
    rng = np.random.default_rng(0)
    target = rng.normal(size=(24, 16))
    windows = [np.arange(i, min(i + 8, 24)) for i in range(0, 24, 8)]
    times = rng.normal(size=24)
    cond = rng.normal(size=(24, 85))
    cond_banks = {('no_init', b): cond for b in range(M.MASK_BANKS)}
    _, alpha_bar = schedule()
    losses = [3., 1., 1., 2.] + [5.] * (M.BUDGET_1120_EPOCHS - 4)
    calls = iter(losses)
    _, _, _, checkpoints, trace, _ = M.fit_with_dual_budget(
        target, cond_banks, 'no_init', windows, times, alpha_bar,
        16, 16, 42, lambda m: float(next(calls)), M.BUDGET_1120_EPOCHS)
    assert checkpoints['budget280']['best_epoch'] == 2
    assert checkpoints['budget1120']['best_epoch'] == 2


# ---------------------------------------------------------------------------
# Key format
# ---------------------------------------------------------------------------

def test_policy_key_format():
    assert M.policy_key('control', 'no_init', 42) == 'control_no_init__clr16__seed42'
    assert M.policy_key('intervention', 'hron_2a', 43) == 'intervention_hron_2a__clr16__seed43'


def test_budget_key_format():
    assert M.budget_key('control', 'no_init', 42, '280') == 'control_no_init__clr16__seed42_budget280'
    assert M.budget_key('intervention', 'mask_only', 44, '1120') == 'intervention_mask_only__clr16__seed44_budget1120'

