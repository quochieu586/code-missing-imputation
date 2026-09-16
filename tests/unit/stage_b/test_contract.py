"""T11 is defined before the Stage B implementation (handoff requirement)."""
import numpy as np
import pytest
import torch
from src.stage_b.single_pass import masked_noise_loss, complete_counts, single_pass, training_loss, sample
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
