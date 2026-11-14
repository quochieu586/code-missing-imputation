"""T11 is defined before the Stage B implementation (handoff requirement)."""
import numpy as np
import pytest
import torch
from src.stage_b.single_pass import (masked_noise_loss, complete_counts, single_pass,
                                     training_loss, sample, center)
from src.bridge.clr_bridge import feature_order
from src.stage_b.csdi.model import Denoiser
from src.core.transforms import clr
from src.stage_a.knn_aitchison import knn_aitchison_impute


def test_T11_loss_ignores_non_targets():
    pred = torch.randn(2, 3, 5, dtype=torch.float64)
    truth = torch.randn_like(pred)
    mask = torch.rand_like(pred) > .5
    expected = masked_noise_loss(pred, truth, mask)
    for v in [float('nan'), 1e9, -1e9]:
        poisoned = truth.clone()
        poisoned[~mask] = v
        assert torch.equal(expected, masked_noise_loss(pred, poisoned, mask))
        poisoned_pred = pred.clone()
        poisoned_pred[~mask] = v
        assert torch.equal(expected, masked_noise_loss(poisoned_pred, poisoned, mask))


def test_T12_lock_and_clr():
    x = np.array([[[1., 2., 3.], [4., 5., 6.]]])
    m = np.array([[[True, False, True], [False, True, True]]])
    out = complete_counts(x, m, np.array([[[8., 3., -1.], [3., 5., 1.]]]))
    assert np.array_equal(out[m], x[m])
    assert np.all(out > 0)


def test_T15_order_train_only():
    x = np.random.default_rng(2).random((8, 4))
    a = feature_order(x, np.arange(4), ['a'] * 4)
    x[4:] *= 100
    assert np.array_equal(a, feature_order(x, np.arange(4), ['a'] * 4))


def test_T16_no_outer_loop():
    with pytest.raises(ValueError, match='max_iter=1'):
        single_pass(None, None, None, max_iter=2)


def test_train_sample_and_frozen_mask():
    torch.manual_seed(1)
    torch.set_num_threads(2)
    x = torch.exp(torch.randn(2,4,5))
    mask = torch.rand(2,4,5) > .3
    frozen = mask.clone()
    times = torch.arange(4).float().expand(2,-1)
    model = Denoiser(5,16,groups=['a','a','b','b','b'])
    beta = torch.linspace(.001,.5,4)
    loss = training_loss(model,x,mask,times,torch.ones(5),torch.cumprod(1-beta,0))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    z = sample(model,x,mask,times,beta,2).numpy().astype(float)
    result = single_pass(x.numpy(),mask.numpy(),z)
    assert torch.equal(mask,frozen)
    assert result.n_iter == 1 and not result.converged
    assert np.array_equal(result.X_hat[mask],x.numpy()[mask])
    assert np.max(np.abs(clr(result.X_hat).sum(-1))) < 1e-8


def test_knn_fallback_train_only_and_small_pool():
    x = np.array([[1.,2.],[2.,4.],[100.,900.],[np.nan,np.nan]])
    mask = np.isfinite(x)
    a = knn_aitchison_impute(x,mask,train_idx=np.array([0,1])).X_imputed
    x[2] *= 1e6
    b = knn_aitchison_impute(x,mask,train_idx=np.array([0,1])).X_imputed
    assert np.array_equal(a[3],b[3])
    x = np.array([[1.,2.],[2.,np.nan]])
    out = knn_aitchison_impute(x,np.isfinite(x),k=8,train_idx=np.array([0])).X_imputed
    assert out[1,1] == 4.


def test_metrics_use_final_compositions():
    from src.evaluation.metrics import evaluate
    x = np.array([[1.,2.,3.],[2.,3.,4.]])
    mask = np.array([[True,False,False],[False,True,False]])
    result = evaluate(x,x,mask,np.zeros_like(mask))
    assert result['mae_clr_all'] == result['m2'] == result['m3'] == 0.
    assert result['mae_clr_pseudocount'] is None


def test_perfect_epsilon_has_zero_training_loss():
    # Regression: projecting a partially masked noisy tensor into CLR before
    # epsilon loss incorrectly penalizes an exact denoiser.
    x = torch.tensor([[[1.,10.,100.],[2.,4.,90.]]])
    alpha = torch.tensor([.9999,.99,.5])
    truth = x.log()-x.log().mean(-1,keepdim=True)
    class Oracle:
        def __call__(self,noisy,context,observed,times,step):
            a = alpha[step,None,None]
            return (noisy-a.sqrt()*truth)/(1-a).sqrt()
    torch.manual_seed(4)
    loss = training_loss(Oracle(),x,torch.ones_like(x,dtype=torch.bool),
                         torch.tensor([[0.,1.]]),torch.ones(3),alpha)
    assert loss.item() < 1e-10


def test_ddpm_matches_reference_transition_before_final_projection():
    class ZeroNoise:
        def eval(self):
            return self
        def __call__(self,noisy,context,observed,times,step):
            return torch.zeros_like(noisy)
    x = torch.ones(1,2,3)
    beta = torch.tensor([.01,.1,.5])
    alpha = torch.cumprod(1-beta,0)
    torch.manual_seed(17)
    expected = torch.randn_like(x)
    for t in reversed(range(3)):
        expected = expected/(1-beta[t]).sqrt()
        if t:
            expected += ((1-alpha[t-1])/(1-alpha[t])*beta[t]).sqrt()*torch.randn_like(x)
    expected -= expected.mean(-1,keepdim=True)
    torch.manual_seed(17)
    actual = sample(ZeroNoise(),x,torch.zeros_like(x,dtype=torch.bool),torch.zeros(1,2),beta,1)
    assert torch.allclose(actual,expected,atol=1e-6)


def test_sampler_uses_observed_context_not_unconstrained_latents():
    torch.manual_seed(31)
    x = torch.tensor([[[1.,2.,3.],[3.,4.,9.]]])
    model = Denoiser(3,16)
    mask = torch.ones_like(x,dtype=torch.bool)
    z = sample(model,x,mask,torch.zeros(1,2),torch.tensor([.01,.1,.5]),3)
    expected = x.log()-x.log().mean(-1,keepdim=True)
    assert torch.allclose(z,expected,atol=1e-6)
    mask[...,1] = False
    z = sample(model,x,mask,torch.zeros(1,2),torch.tensor([.01,.1,.5]),3)
    # Observed log-ratios must be invariant even though CLR gauge shifts.
    assert torch.allclose(z[...,0]-z[...,2],expected[...,0]-expected[...,2],atol=1e-5)


def test_diverged_scale_is_rejected_before_it_is_written():
    # complete_counts clips the exponent, so a diverged CLR yields a finite
    # number that passes the mask and finiteness checks. Bug #4: nothing
    # compared it to anything, so 1e146 was written out as a count.
    from src.stage_b.single_pass import check_counts_are_physical
    x1 = np.full((3, 4), 2.)
    depth = np.array([8., 8., 8.])
    check_counts_are_physical(x1, x1, depth)
    blown = x1.copy()
    blown[1, 2] = 1e146
    with pytest.raises(ValueError, match='exceed 10x both'):
        check_counts_are_physical(blown, x1, depth)
    m = np.ones((3, 4), dtype=bool)
    m[1, 2] = False
    z = np.zeros((3, 4))
    z[1, 2] = 400.
    with pytest.raises(ValueError, match='exceed 10x both'):
        single_pass(x1, m, z, total_sequence=depth)
    # Without a depth the guard cannot run, and the caller keeps the old behaviour.
    assert single_pass(x1, m, z).X_hat[1, 2] > 1e10


def test_high_depth_row_of_pseudocounts_may_legitimately_grow():
    # A location reporting only pseudo-counts while 121k sequences were
    # sequenced SHOULD gain orders of magnitude once its unreported variants
    # are filled. Bounding against the initializer alone rejected these.
    from src.stage_b.single_pass import check_counts_are_physical
    x1 = np.array([[.5, .5, .5, .5]])
    filled = np.array([[.5, .5, 2.7e4, 1e3]])
    check_counts_are_physical(filled, x1, np.array([1.21e5]))
    with pytest.raises(ValueError, match='exceed 10x both'):
        check_counts_are_physical(filled, x1, np.array([4.]))


def test_cells_without_truth_are_denoised_like_the_cells_being_predicted():
    # Bug #3: missingness is all-or-nothing per (location, variant), so a
    # never-reported column can never be a self-supervised target. It must
    # still reach the denoiser in the same input state as a target cell,
    # otherwise the reverse chain meets it for the first time at sampling.
    seen = {}

    class Spy(torch.nn.Module):
        def forward(self, noisy, context, condition, times, step):
            seen.update(noisy=noisy, context=context, condition=condition)
            return torch.zeros_like(noisy)

    torch.manual_seed(0)
    x = torch.exp(torch.randn(1, 3, 4, dtype=torch.float64))
    eligible = torch.ones(1, 3, 4, dtype=torch.bool)
    eligible[..., 3] = False           # a variant this location never reports
    medians = torch.full((4,), 2., dtype=torch.float64)
    beta = torch.linspace(.001, .5, 4, dtype=torch.float64)
    loss = training_loss(Spy(), x, eligible, torch.arange(3).double().expand(1, -1),
                         medians, torch.cumprod(1 - beta, 0), ratio=1.)

    never = (slice(None), slice(None), 3)
    assert not seen['condition'][never].any(), 'must not be used as conditioning'
    assert (seen['noisy'][never] != 0).all(), 'must carry noise, like a target'
    # ratio=1. draws every eligible cell, so the whole context is the median
    # substitute: the never-reported column is indistinguishable from a target.
    assert torch.allclose(seen['context'], center(medians.expand_as(x).log()))
    # The loss is still restricted to cells that have a truth to compare against.
    assert torch.isfinite(loss)


def test_sampling_context_matches_the_training_context():
    # Training replaces every cell being predicted with the train medians
    # before the geometric mean; sampling must do the same, or the cells to
    # impute look like cells that were never targets during training.
    seen = {}

    class Spy(torch.nn.Module):
        def forward(self, noisy, context, condition, times, step):
            seen['context'] = context
            return torch.zeros_like(noisy)

    x = torch.exp(torch.randn(1, 2, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(3)))
    observed = torch.tensor([[[True, False, True], [True, True, False]]])
    medians = torch.full((3,), 5., dtype=torch.float64)
    beta = torch.linspace(.001, .5, 2, dtype=torch.float64)
    sample(Spy(), x, observed, torch.arange(2).double().expand(1, -1), beta, 1, medians)
    expected = center(torch.where(observed, x, medians).log())
    assert torch.allclose(seen['context'], expected)
    # And the X1 values at cells to impute never reach the denoiser.
    poisoned = x.clone()
    poisoned[~observed] = 1e9
    sample(Spy(), poisoned, observed, torch.arange(2).double().expand(1, -1), beta, 1, medians)
    assert torch.allclose(seen['context'], expected)


def test_unclipped_sampling_matches_the_epsilon_form():
    # The reverse step was rewritten as the q(x_{t-1}|x_t,x0_hat) posterior mean
    # so x0_hat can be bounded. Without a clip it must be the same chain.
    torch.manual_seed(5)
    beta = torch.linspace(1e-4 ** .5, .5 ** .5, 50).square().double()
    alpha = torch.cumprod(1 - beta, 0)
    for t in range(len(beta)):
        cur, eps = torch.randn(64, dtype=torch.float64) * 3, torch.randn(64, dtype=torch.float64)
        old = (cur - beta[t] / (1 - alpha[t]).sqrt() * eps) / (1 - beta[t]).sqrt()
        x0 = (cur - (1 - alpha[t]).sqrt() * eps) / alpha[t].sqrt()
        previous = alpha[t - 1] if t else torch.ones_like(alpha[0])
        new = (previous.sqrt() * beta[t] / (1 - alpha[t]) * x0
               + (1 - beta[t]).sqrt() * (1 - previous) / (1 - alpha[t]) * cur)
        assert torch.allclose(old, new, atol=1e-9)


def test_clipping_bounds_a_chain_that_would_otherwise_blow_up():
    # An epsilon predictor stuck at zero multiplies `current` by
    # prod 1/sqrt(1-beta_t) = 172x. The clip must keep the output in range.
    class Zero(torch.nn.Module):
        def forward(self, noisy, context, condition, times, step):
            return torch.zeros_like(noisy)

    torch.manual_seed(0)
    x = torch.exp(torch.randn(1, 4, 5, dtype=torch.float64))
    observed = torch.rand(1, 4, 5) > .5
    times = torch.arange(4).double().expand(1, -1)
    beta = torch.linspace(1e-4 ** .5, .5 ** .5, 50).square().double()
    medians = torch.ones(5, dtype=torch.float64)

    loose = sample(Zero(), x, observed, times, beta, 1, medians)
    tight = sample(Zero(), x, observed, times, beta, 1, medians, clip=(-4., 4.))
    assert loose[~observed].abs().max() > 50, 'expected the unclipped chain to diverge'
    assert tight[~observed].abs().max() < 12, 'clipped chain must stay bounded'
