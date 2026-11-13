"""Numerical parity, head/channel policy, scale-guard and observed-locking tests
for the reworked CSDI core (src/stage_b/research_csdi_rework.py).

These lock the experiment's architectural rule: every configured model uses
``heads=1`` so the channels 16 vs 17 vs 8 comparison varies only the latent
width, not the head count. The frozen 16-channel/4-head model is the bit-level
reference (archived control, not retrained).
"""
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from src.stage_b.research_csdi import (CSDICore, epsilon_loss, forward_noise,
    project_final, reverse_step, sample_latents, schedule, time_embedding)
from src.stage_b.research_csdi_rework import (CSDICoreRework, time_embedding_rework,
    ResidualBlockRework)
from src.evaluation.research_pilot import numerical_diagnostics, evaluate


ROOT = Path(__file__).resolve().parents[3]


def _pair(k, heads, channels=None):
    torch.manual_seed(42)
    frozen = CSDICore(k)
    torch.manual_seed(42)
    rework = CSDICoreRework(k, channels=channels or 16, heads=heads)
    return frozen, rework


def test_rework_matches_frozen_core_bit_for_bit():
    torch.set_num_threads(2)
    frozen, rework = _pair(16, heads=4)
    # same parameter tensors, same keys, same values
    fd, rd = frozen.state_dict(), rework.state_dict()
    assert set(fd) == set(rd)
    for key in fd:
        assert torch.equal(fd[key], rd[key]), key
    # identical forward on identical float64 inputs
    k = 16
    z = torch.randn(2, 3, k, dtype=torch.float64)
    c = torch.randn(2, 3, 85, dtype=torch.float64)
    t = torch.arange(3, dtype=torch.float64).expand(2, -1)
    steps = torch.tensor([1, 2])
    frozen.eval(); rework.eval()
    with torch.no_grad():
        assert torch.equal(frozen(z, c, t, steps), rework(z, c, t, steps))


def test_time_embedding_rework_equals_frozen():
    pos = torch.linspace(0, 1, 6, dtype=torch.float64).expand(3, -1)
    a = time_embedding(pos, 16)
    b = time_embedding_rework(pos, 16)
    assert torch.equal(a, b)


def test_single_head_policy_is_single_attention_head():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    model = CSDICoreRework(16, channels=16, heads=1)
    assert model.heads == 1
    assert model.layers[0].time_attention.self_attn.num_heads == 1
    assert model.layers[0].feature_attention.self_attn.num_heads == 1


def test_heads_policy_controls_attention_head_count():
    torch.set_num_threads(2)
    for heads, expected in ((1, 1), (2, 2), (4, 4)):
        torch.manual_seed(0)
        model = CSDICoreRework(16, channels=16, heads=heads)
        assert model.layers[0].time_attention.self_attn.num_heads == expected
        assert model.layers[0].feature_attention.self_attn.num_heads == expected


def test_heads_one_differs_from_heads_four_forward():
    """head count must be the only architectural variable along the channel axis."""
    torch.set_num_threads(2)
    k = 16
    z = torch.randn(2, 3, k, dtype=torch.float64)
    c = torch.randn(2, 3, 85, dtype=torch.float64)
    t = torch.arange(3, dtype=torch.float64).expand(2, -1)
    steps = torch.tensor([1, 2])
    with torch.no_grad():
        torch.manual_seed(7)
        a = CSDICoreRework(k, channels=16, heads=1)
        torch.nn.init.normal_(a.output[-1].weight)
        torch.nn.init.normal_(a.output[-1].bias)
        a.eval()
        out_a = a(z, c, t, steps)
        torch.manual_seed(7)
        b = CSDICoreRework(k, channels=16, heads=4)
        torch.nn.init.normal_(b.output[-1].weight)
        torch.nn.init.normal_(b.output[-1].bias)
        b.eval()
        out_b = b(z, c, t, steps)
    assert torch.isfinite(out_a).all()
    assert torch.isfinite(out_b).all()
    assert not torch.equal(out_a, out_b)


def test_clr16_channels16_and_clr17_channels17_parity():
    torch.set_num_threads(2)
    for k, channels in ((16, 16), (17, 17)):
        torch.manual_seed(3)
        model = CSDICoreRework(k, channels=channels, heads=1)
        assert model.channels == channels
        assert model.latent_dim == k
        z = torch.randn(2, 4, k, dtype=torch.float64)
        c = torch.randn(2, 4, 85, dtype=torch.float64)
        t = torch.arange(4, dtype=torch.float64).expand(2, -1)
        steps = torch.tensor([0, 1])
        model.eval()
        with torch.no_grad():
            out = model(z, c, t, steps)
        assert out.shape == z.shape
        assert out.dtype == torch.float64
        assert torch.isfinite(out).all()


def test_capacity_branch_channels8_is_narrower():
    torch.set_num_threads(2)
    torch.manual_seed(3)
    core = CSDICoreRework(16, channels=8, heads=1)
    assert core.channels == 8
    wide = CSDICoreRework(16, channels=16, heads=1)
    narrow_params = sum(p.numel() for p in core.parameters())
    wide_params = sum(p.numel() for p in wide.parameters())
    assert narrow_params < wide_params


def test_sample_latents_runs_with_rework_core_and_is_finite():
    torch.set_num_threads(2)
    torch.manual_seed(11)
    model = CSDICoreRework(16, channels=16, heads=1)
    model.eval()
    c = torch.randn(1, 5, 85, dtype=torch.float64)
    t = torch.arange(5, dtype=torch.float64).expand(1, -1)
    beta, ab = schedule()
    out, diag = sample_latents(model, c, t, beta, ab,
                               torch.zeros(1, 5, 16, dtype=torch.float64),
                               torch.tensor(1.0), 'ilr16', (), samples=4, seed=123)
    assert out.shape == (4, 1, 5, 16)
    assert torch.isfinite(out).all()
    assert diag['clipped_fraction'] == 0.0
    assert diag['projection'] == 'final_only'


def test_scale_guard_records_rows_without_dropping():
    query = np.array([[1., 2., 3.], [np.nan, np.nan, np.nan]])
    pred = np.array([[1., 2., 3.], [999999., 0., 0.]])
    diag = numerical_diagnostics(pred, query)
    assert diag['guard_threshold'] == 1000.0
    assert diag['guard_basis'] == 'visible_sum_or_one'
    assert 1 in diag['scale_guard_row_positions']
    assert 0 not in diag['scale_guard_row_positions']
    assert diag['n_scale_guard_rows'] == 1


def test_observed_locking_is_exact_through_restore():
    from scripts.run_e3_revised import load_transform_module
    tm = load_transform_module()
    comp = np.array([0.1, 0.2, 0.05, 0.3, 0.05, 0.05, 0.05, 0.05,
                     0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.07])
    comp = comp / comp.sum()
    query = comp * 100.0
    query[[0, 3, 6, 10]] = np.nan
    restored, info = tm.restore_observed_counts(comp, query)
    visible = np.isfinite(query)
    assert np.array_equal(restored[visible], query[visible])
    assert info['observed_exact'] is True
    assert np.all(restored >= 0)
    assert np.isfinite(restored).all()


def test_float64_dtype_enforced():
    torch.set_num_threads(2)
    model = CSDICoreRework(16, channels=16, heads=1)
    z = torch.randn(1, 2, 16, dtype=torch.float32)
    c = torch.randn(1, 2, 85, dtype=torch.float64)
    t = torch.zeros(2, dtype=torch.float64)
    steps = torch.tensor([0])
    with pytest.raises(TypeError):
        model(z, c, t, steps)
    assert all(p.dtype == torch.float64 for p in model.parameters())


def test_latent_dimension_mismatch_rejected():
    torch.set_num_threads(2)
    model = CSDICoreRework(16, channels=16, heads=1)
    z = torch.randn(1, 2, 12, dtype=torch.float64)
    c = torch.randn(1, 2, 85, dtype=torch.float64)
    t = torch.zeros(2, dtype=torch.float64)
    steps = torch.tensor([0])
    with pytest.raises(ValueError):
        model(z, c, t, steps)
