"""Tests for the E5 rework gate logic and the corrected runner (2026-10-06 audit).

Gate tests are pure-python boundary tests; runner tests use tiny synthetic tensors
and a temporary experiment folder so no real artifact is touched.
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
import run_e5_rework as R  # noqa: E402

from src.evaluation import rework_gates as RG  # noqa: E402
from src.stage_b.research_csdi import schedule  # noqa: E402

SEEDS = (42, 43, 44)
SCN = RG.PRIMARY


def sc(m2, mae=1.0, **kw):
    d = {'m2': m2, 'clr_mae_nonzero_hidden': mae, 'status': 'OK', 'nonfinite_cells': 0,
         'negative_cells': 0, 'observed_restoration_exact': True}
    d.update(kw)
    return d


def build(no=10., mask=11., init_only=10., csdi=8., mae_csdi=1.0, mae_init=1.0,
          transforms=('clr16',), inits=('hron_2a',)):
    scores = {}
    for t in transforms:
        for s in SEEDS:
            scores[f'no_init__{t}__seed{s}'] = {c: sc(no) for c in SCN}
            scores[f'mask_only__{t}__seed{s}'] = {c: sc(mask) for c in SCN}
            for i in inits:
                scores[f'{i}__{t}__seed{s}'] = {c: sc(csdi, mae_csdi) for c in SCN}
    for i in inits:
        for c in SCN:
            scores[f'init_only__{i}__{c}'] = sc(init_only, mae_init)
    return scores


# ---------------------------------------------------------------- conditioning

def test_conditioning_requires_strict_inequality():
    assert RG.conditioning_gate(build(no=10., mask=10.0001), ['clr16'], SEEDS)['passed']
    assert not RG.conditioning_gate(build(no=10., mask=10.), ['clr16'], SEEDS)['passed']
    assert not RG.conditioning_gate(build(no=10., mask=9.99), ['clr16'], SEEDS)['passed']


def test_conditioning_fails_if_one_seed_or_one_scenario_fails():
    s = build()
    s['mask_only__clr16__seed43']['whole_rare_n2']['m2'] = 9.0
    gate = RG.conditioning_gate(s, ['clr16'], SEEDS)
    assert not gate['passed']
    assert not gate['pairs']['clr16__whole_rare_n2']['passed']
    assert gate['pairs']['clr16__whole_middle_n2']['passed']


def test_conditioning_missing_entry_fails_not_skips():
    s = build()
    del s['no_init__clr16__seed44']
    assert not RG.conditioning_gate(s, ['clr16'], SEEDS)['passed']


def test_conditioning_needs_every_transform():
    s = build(transforms=('clr16', 'ilr16'))
    for seed in SEEDS:
        s[f'mask_only__ilr16__seed{seed}']['whole_common_n2']['m2'] = 1.0
    assert not RG.conditioning_gate(s, ['clr16', 'ilr16'], SEEDS)['passed']


# -------------------------------------------------------------------- efficacy

def test_efficacy_threshold_is_strictly_greater_than_ten_percent():
    assert not RG.efficacy_gate(build(init_only=10., csdi=9.), ['clr16'], ['hron_2a'], SEEDS)['passed']
    assert RG.efficacy_gate(build(init_only=10., csdi=8.99), ['clr16'], ['hron_2a'], SEEDS)['passed']


def test_efficacy_requires_nonzero_clr_mae_not_worse():
    gate = RG.efficacy_gate(build(csdi=5., mae_csdi=1.5, mae_init=1.0), ['clr16'], ['hron_2a'], SEEDS)
    assert not gate['passed']
    assert RG.efficacy_gate(build(csdi=5., mae_csdi=1.0, mae_init=1.0), ['clr16'], ['hron_2a'], SEEDS)['passed']


def test_efficacy_one_good_seed_is_not_enough():
    s = build(csdi=5.)
    s['hron_2a__clr16__seed44']['whole_middle_n2']['m2'] = 50.
    assert not RG.efficacy_gate(s, ['clr16'], ['hron_2a'], SEEDS)['passed']


def test_efficacy_one_qualifying_pair_suffices():
    s = build(csdi=5., transforms=('clr16', 'ilr16'))
    for seed in SEEDS:
        s[f'hron_2a__ilr16__seed{seed}']['whole_middle_n2']['m2'] = 99.
    gate = RG.efficacy_gate(s, ['clr16', 'ilr16'], ['hron_2a'], SEEDS)
    assert gate['passed'] and gate['pairs']['hron_2a__clr16']['passed']
    assert not gate['pairs']['hron_2a__ilr16']['passed']


def test_efficacy_missing_init_only_baseline_fails():
    s = build(csdi=5.)
    del s['init_only__hron_2a__whole_rare_n2']
    assert not RG.efficacy_gate(s, ['clr16'], ['hron_2a'], SEEDS)['passed']


# ------------------------------------------------------------------- numerical

@pytest.mark.parametrize('field,value', [('observed_restoration_exact', False),
                                         ('nonfinite_cells', 1), ('negative_cells', 2),
                                         ('status', 'FAIL_OUTPUT_CONTRACT')])
def test_numerical_gate_catches_each_contract_violation(field, value):
    s = build()
    s['hron_2a__clr16__seed42']['whole_rare_n2'][field] = value
    gate = RG.numerical_gate(s)
    assert not gate['passed'] and any('hron_2a__clr16__seed42' in v for v in gate['violations'])


def test_numerical_gate_passes_clean_and_ignores_scale_flags():
    s = build()
    s['hron_2a__clr16__seed42']['whole_rare_n2']['numerical_diagnostics'] = {'n_scale_guard_rows': 9}
    assert RG.numerical_gate(s)['passed']  # flags are recorded, never gate-removed rows


# ------------------------------------------------------------------- selection

def test_no_finalists_when_any_gate_fails_even_if_a_pair_qualifies():
    s = build(no=10., mask=9., csdi=5.)  # conditioning fails
    g = RG.validation_gates(s, ['clr16'], ['hron_2a'], SEEDS)
    assert not g['conditioning_gate']['passed'] and g['efficacy_gate']['passed']
    finalists, diagnostic = RG.select_finalists(s, g, ['clr16'], ['hron_2a'], SEEDS)
    assert finalists == [] and len(diagnostic) == 1


def test_finalists_ranked_by_mean_m2_with_random_cell_then_name_tiebreak():
    tr, inits = ('clr16', 'ilr16', 'hkglr16_actual'), ('hron_2a', 'aitchison_complete')
    s = build(csdi=5., transforms=tr, inits=inits)
    for t in tr:
        for seed in SEEDS:
            for i in inits:
                s[f'{i}__{t}__seed{seed}']['random_cell_bank0'] = sc(7.)
    for seed in SEEDS:
        s[f'hron_2a__ilr16__seed{seed}'] = {c: sc(4.) for c in SCN} | {'random_cell_bank0': sc(7.)}
        s[f'aitchison_complete__clr16__seed{seed}']['random_cell_bank0']['m2'] = 1.
    g = RG.validation_gates(s, list(tr), list(inits), SEEDS)
    assert g['passed']
    finalists, _ = RG.select_finalists(s, g, list(tr), list(inits), SEEDS, n=3)
    assert [f['config'] for f in finalists][0] == 'hron_2a__ilr16'
    # equal rank statistic -> random-cell M2 breaks the tie, then name
    assert [f['config'] for f in finalists][1] == 'aitchison_complete__clr16'
    assert len(finalists) == 3


# ----------------------------------------------------------- runner: cache, fit

@pytest.fixture
def tmp_exp(tmp_path, monkeypatch):
    (tmp_path / 'config.json').write_text('{"x": 1}')
    monkeypatch.setattr(R, 'B', tmp_path)
    return tmp_path


def test_cache_roundtrip_and_stale_detection(tmp_exp, monkeypatch):
    d = tmp_exp / 'cache'
    assert R.cache_load(d, 'k') is None
    R.cache_save(d, 'k', {'whole_rare_n2': {'m2': 1.5}}, {'whole_rare_n2': np.ones((2, 3))},
                 {'whole_rare_n2': np.zeros((R.MC_SAMPLES, 2, 3))}, {'w': torch.zeros(1)},
                 7, 0.25, {})
    blob = R.cache_load(d, 'k')
    assert blob['best_epoch'] == 7 and blob['scores']['whole_rare_n2']['m2'] == 1.5
    assert np.load(d / 'k.npz')['pred__whole_rare_n2'].shape == (2, 3)
    monkeypatch.setattr(R, 'MC_SAMPLES', R.MC_SAMPLES + 1)
    with pytest.raises(RuntimeError, match='stale cache'):
        R.cache_load(d, 'k')


def test_incomplete_cache_entry_is_not_a_hit(tmp_exp):
    d = tmp_exp / 'cache'
    d.mkdir()
    (d / 'k.npz').write_bytes(b'partial')  # crash before the JSON marker was written
    assert R.cache_load(d, 'k') is None


def _toy_training_inputs(n=24, k=4):
    rng = np.random.default_rng(0)
    target = rng.normal(size=(n, k))
    windows = [np.arange(i, min(i + 8, n)) for i in range(0, n, 8)]
    times = rng.normal(size=n)
    cond = rng.normal(size=(n, 85))
    banks = {('no_init', b): cond for b in range(R.MASK_BANKS)}
    return target, banks, windows, times


def test_transferred_epochs_reproduce_checkpoint_run_exactly():
    """Canada trains for the validation-selected epoch count: the weights must equal
    the validation run's checkpoint at that epoch."""
    target, banks, windows, times = _toy_training_inputs()
    _, alpha_bar = schedule()
    # decreasing fixture loss => selected epoch is the last one
    calls = iter(range(100, 0, -1))
    model_a, *_ = R.fit_with_checkpoint(target, banks, 'no_init', windows, times, alpha_bar,
                                        8, target.shape[1], 3, lambda m: float(next(calls)), 4)
    model_b, *_ = R.fit_transferred(target, banks, 'no_init', windows, times, alpha_bar,
                                    8, target.shape[1], 3, 4)
    for (ka, va), (kb, vb) in zip(model_a.state_dict().items(), model_b.state_dict().items()):
        assert ka == kb and torch.equal(va, vb), ka


def test_checkpoint_selection_picks_argmin_with_earliest_tie():
    target, banks, windows, times = _toy_training_inputs()
    _, alpha_bar = schedule()
    losses = iter([3., 1., 1., 2.])
    *_, best_epoch, best_loss, trace, _ = R.fit_with_checkpoint(
        target, banks, 'no_init', windows, times, alpha_bar, 8, target.shape[1], 1,
        lambda m: next(losses), 4)
    assert (best_epoch, best_loss) == (2, 1.) and len(trace) == 4


def test_fixture_rejects_incomplete_or_zero_sum_rows_and_uses_full_label_target():
    q = np.abs(np.random.default_rng(1).normal(size=(4, 17))) + 1
    bad = q.copy()
    bad[1, 3] = np.nan
    visible = np.ones_like(q, dtype=bool)
    visible[:, :5] = False
    times = np.arange(4, dtype=float)

    class T:
        def forward(self, x):
            self.last = np.array(x)
            return np.log(x)

    with pytest.raises(AssertionError):
        R.make_fixture(bad, visible, times, np.arange(4), {}, {}, {}, 5.0, T(),
                       np.zeros(17), 1.0, 'mask_only')
    t = T()
    R.make_fixture(q, visible, times, np.arange(4), {}, {}, {}, 5.0, t, np.zeros(17), 1.0,
                   'mask_only')
    assert np.array_equal(t.last, q)  # target is the unmasked label, not masked-with-zeros


def test_restoration_sensitivity_only_touches_no_anchor_rows_and_keeps_observed(tmp_exp):
    d = tmp_exp / 'cache'
    d.mkdir()
    truth = np.array([[5., 3., 2.], [4., 4., 2.], [6., 2., 2.]] * 2)
    hidden = np.zeros_like(truth, dtype=bool)
    hidden[0] = True          # row 0 fully hidden -> no positive anchor
    hidden[1, 0] = True
    pred = truth / truth.sum(1, keepdims=True)
    q = np.where(~hidden, truth, np.nan)
    pred = np.where(np.isfinite(q), q, pred)
    np.savez_compressed(d / 'm.npz', **{'pred__s': pred})
    bins = {'all': np.arange(3)}
    out = R.restoration_sensitivity(d, ['m'], {'s': hidden}, truth, 0.5, bins,
                                    np.array([10., 20., 30.]))
    rec = out['models']['m/s']
    assert out['train_only_scale'] == 20. and rec['n_no_anchor_rows'] == 1
    assert rec['clipping_fraction'] == 0.0
    # control numbers are the untouched legacy predictions
    ctl = R.evaluate(truth, pred, hidden, q, 0.5, bins)
    assert rec['control_m2'] == ctl['m2']


def test_run_gate_helpers_never_open_canada_without_gate_pass_is_wired_in_main():
    """Static guard: the only call that evaluates the outer test fold is behind `if finalists`."""
    src = (ROOT / 'scripts/run_e5_rework.py').read_text(encoding='utf-8')
    assert src.count("'test', pool_df, query_all") == 1
    pre = src[:src.index("'test', pool_df, query_all")]
    assert 'if finalists and not args.dry_run:' in pre.splitlines()[-40:].__str__() or \
        'if finalists and not args.dry_run:' in pre[-3000:]
    assert 'canada_evaluated.lock' in pre[-3000:]


# ---------------------------------------------- control regressions (2026-10-06 audit)

def _baseline_inputs(seed=0, n=12):
    import pandas as pd
    rng = np.random.default_rng(seed)
    names = [f'v{i}' for i in range(17)]
    truth = rng.integers(1, 50, size=(n, 17)).astype(float)
    frame = pd.DataFrame(truth, columns=names)
    hidden = np.zeros((n, 17), dtype=bool)
    hidden[3:6, 2] = True
    hidden[:, 7] = True                      # whole variant hidden
    return frame, names, hidden, np.arange(n, dtype=float), frame.copy()


@pytest.mark.parametrize('bname', ['linear', 'locf_nocb'])
def test_baseline_never_reads_hidden_truth(bname):
    frame, names, hidden, times, pool = _baseline_inputs()
    bins = {'all': np.arange(17)}
    pred1, s1 = R.evaluate_baseline_on_scenario(frame, names, hidden, times, pool, 0.5, bins, bname)
    poisoned = frame.copy()
    poisoned.loc[:, names] = np.where(hidden, 1e9, frame[names].to_numpy())
    pred2, _ = R.evaluate_baseline_on_scenario(poisoned, names, hidden, times, pool, 0.5, bins, bname)
    assert np.array_equal(pred1, pred2)                     # hidden truth cannot influence output
    assert not np.array_equal(pred1[hidden], frame[names].to_numpy()[hidden])
    assert s1['m2'] > 0                                     # the old leak gave exactly 0


def test_init_only_scores_raw_initializer_output_without_epsilon_floor():
    src = (ROOT / 'scripts/run_e5_rework.py').read_text(encoding='utf-8')
    body = src[src.index('def evaluate_init_only_on_scenario'):src.index('def evaluate_baseline_on_scenario')]
    assert '1e-10' not in body and 'pred = init[init_mode]' in body
