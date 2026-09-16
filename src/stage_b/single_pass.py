"""Exactly one outer pass; DDPM reverse timesteps are not outer iterations."""
from dataclasses import dataclass, field
import numpy as np
import torch
from src.core.transforms import clr, clr_inverse


def center(z):
    return z - z.mean(dim=-1, keepdim=True)


def masked_noise_loss(prediction, noise, target_mask):
    # Index before arithmetic: NaN * 0 would violate T11.
    # Epsilon is Gaussian noise, NOT a CLR vector: do not center epsilon.
    if not target_mask.any():
        return prediction.sum() * 0
    return (prediction[target_mask] - noise[target_mask]).square().mean()


def training_loss(model, x, eligible, times, medians, alpha, ratio=.5):
    target = (torch.rand_like(x) < ratio) & eligible
    condition = eligible & ~target
    log_x = x.log()
    truth = center(log_x)
    # Remove target values BEFORE computing the conditioning geometric mean.
    # Otherwise CLR(condition) leaks hidden values through its row mean.
    safe = torch.where(target, medians, x)
    context = center(safe.log())
    step = torch.randint(len(alpha), (len(x),), device=x.device)
    a = alpha[step, None, None]
    noise = torch.randn_like(x)
    noisy = torch.where(target, a.sqrt()*truth + (1-a).sqrt()*noise, torch.zeros_like(x))
    epsilon = model(noisy, context, condition, times, step)
    # Equation (5) in the handoff is epsilon MSE. Centering a partial noisy
    # tensor and mapping it back to epsilon changes this objective, introducing
    # a large spurious term at low noise. Only actual CLR outputs (sampling,
    # metrics) are centered; Gaussian epsilon labels are not CLR coordinates.
    return masked_noise_loss(epsilon, noise, target)


@torch.no_grad()
def sample(model, x, observed, times, beta, n_samples=10):
    model.eval()
    alpha = torch.cumprod(1-beta, 0)
    context = center(x.log())
    samples = []
    for _ in range(n_samples):
        current = torch.randn_like(x)
        for t in reversed(range(len(beta))):
            noisy = torch.where(~observed, current, torch.zeros_like(current))
            eps = model(noisy, context, observed, times, torch.full((len(x),), t, device=x.device))
            # Reverse the same Gaussian process used in training. Project only
            # the final CLR output; projecting intermediate states would change
            # the transition kernel and the distribution seen by the denoiser.
            current = (current-beta[t]/(1-alpha[t]).sqrt()*eps)/(1-beta[t]).sqrt()
            if t:
                sigma = ((1-alpha[t-1])/(1-alpha[t])*beta[t]).sqrt()
                current += sigma*torch.randn_like(current)
        # CSDI samples only the missing coordinates. Observed coordinates of
        # `current` are unconstrained latent noise, never observations. Restore
        # the conditioning CLR BEFORE projection/aggregation/inverse scaling.
        samples.append(center(torch.where(observed, context, current)))
    return center(torch.stack(samples).median(0).values)


def complete_counts(x1, m0, z):
    """Inverse CLR once, then restore original units and lock observed counts.

    Softmax alone cannot preserve arbitrary observed counts (handoff §4.3 vs
    R5/T12). Anchor the composition scale by the median observed log-ratio;
    overwrite observed cells exactly. Closure is provided separately for metrics.
    """
    x1 = np.asarray(x1, dtype=np.float64)
    m0 = np.asarray(m0, dtype=bool)
    z = np.asarray(z, dtype=np.float64)
    if x1.shape != m0.shape or z.shape != x1.shape:
        raise ValueError('X1, M0 and predicted CLR must have equal shapes')
    if not np.isfinite(x1).all() or np.any(x1 <= 0) or not np.isfinite(z).all():
        raise ValueError('X1 must be finite/positive and predicted CLR finite')
    p = clr_inverse(z)
    tiny = np.finfo(float).tiny
    p = np.maximum(p, tiny)
    log_scale = np.where(m0, np.log(x1)-np.log(p), np.nan)
    flat = log_scale.reshape(-1, x1.shape[-1])
    scales = np.array([np.median(row[np.isfinite(row)]) if np.isfinite(row).any()
                       else np.log(x1.reshape(flat.shape)[i].sum()) for i, row in enumerate(flat)])
    out = np.exp(np.clip(np.log(p) + scales.reshape(x1.shape[:-1])[..., None], -700, 700))
    out[m0] = x1[m0]
    return out


@dataclass
class LoopResult:
    X_hat: np.ndarray
    Z_history: list
    converged: bool = False
    n_iter: int = 1
    c1_history: list = field(default_factory=list)
    c2_history: list = field(default_factory=list)
    c3_history: list = field(default_factory=list)
    alpha_div_history: list = field(default_factory=list)
    zero_prop_history: list = field(default_factory=list)


def single_pass(x1, m0, predicted_clr, *, max_iter=1, validation_mae=None):
    if max_iter != 1:
        raise ValueError('This implementation requires max_iter=1')
    frozen = m0.copy()
    output = complete_counts(x1, m0, predicted_clr)
    before, after = clr(x1), clr(output)
    k = x1.shape[-1]
    c1 = np.linalg.norm(np.cov(after.reshape(-1,k).T)-np.cov(before.reshape(-1,k).T))/(k-1)
    c2 = np.abs(after-before)[~m0].mean() if (~m0).any() else 0.
    p = output / output.sum(-1, keepdims=True)
    assert np.array_equal(output[m0], x1[m0])
    assert np.array_equal(m0, frozen)
    return LoopResult(output, [before, after], c1_history=[float(c1)], c2_history=[float(c2)],
                      c3_history=[validation_mae], alpha_div_history=[float(-(p*np.log(p)).sum(-1).mean())],
                      zero_prop_history=[float((p < 1e-4).mean())])
