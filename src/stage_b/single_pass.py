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
    drawn = (torch.rand_like(x) < ratio) & eligible
    # Missingness in this data is all-or-nothing per (location, variant): a
    # location either reports a variant at every date or never. Self-supervised
    # targets can therefore only ever be drawn from the variants a location DOES
    # report, while at inference the model must fill the variants it NEVER
    # reports -- two disjoint column sets. With `target = drawn` alone those
    # never-reported columns were the only cells carrying neither noise nor a
    # conditioning flag, a third input state that does not exist at sampling
    # time, so the denoiser met them for the first time during the reverse chain
    # and its epsilon there was arbitrary.
    #
    # Put them in exactly the state the cells being predicted occupy: carrying
    # noise, excluded from conditioning, median-substituted in the context.
    # Supervise only `drawn`, because only there is the CLR truth identifiable.
    target = drawn | ~eligible
    condition = eligible & ~drawn
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
    return masked_noise_loss(epsilon, noise, drawn)


@torch.no_grad()
def sample(model, x, observed, times, beta, n_samples=10, medians=None, clip=None):
    model.eval()
    alpha = torch.cumprod(1-beta, 0)
    # Build the context exactly as training does. Training replaced every cell
    # being predicted with the train medians before the geometric mean; leaving
    # the X1 values in here instead made the cells to be imputed look, to the
    # denoiser, like cells that were NOT being predicted during training.
    # Dropping them also removes the last path by which the kNN initializer
    # could steer its own replacement.
    safe = x if medians is None else torch.where(observed, x, medians.to(x.dtype))
    context = center(safe.log())
    samples = []
    for _ in range(n_samples):
        current = torch.randn_like(x)
        for t in reversed(range(len(beta))):
            noisy = torch.where(~observed, current, torch.zeros_like(current))
            eps = model(noisy, context, observed, times, torch.full((len(x),), t, device=x.device))
            # Reverse the same Gaussian process used in training. Project only
            # the final CLR output; projecting intermediate states would change
            # the transition kernel and the distribution seen by the denoiser.
            #
            # Written as the q(x_{t-1} | x_t, x0_hat) posterior mean so that x0_hat
            # can be held inside the data range, as published DDPM ancestral
            # sampling does. Without that bound a cell where epsilon is inaccurate
            # is multiplied by prod 1/sqrt(1-beta_t) = 172x over these 50 steps,
            # which in CLR coordinates reaches exp(+-600) and saturates the clip
            # inside complete_counts. With clip=None the two forms are
            # algebraically identical, so this is a strict generalisation.
            x0 = (current-(1-alpha[t]).sqrt()*eps)/alpha[t].sqrt()
            if clip is not None:
                x0 = x0.clamp(clip[0], clip[1])
            previous = alpha[t-1] if t else torch.ones_like(alpha[0])
            current = (previous.sqrt()*beta[t]/(1-alpha[t])*x0
                       + (1-beta[t]).sqrt()*(1-previous)/(1-alpha[t])*current)
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


def check_counts_are_physical(x_hat, x1, total_sequence, max_ratio=10.):
    """Reject an output whose row totals are neither physical nor initializer-like.

    ``complete_counts`` clips its exponent at +-700, so a diverged CLR does not
    produce inf or NaN; it produces a finite number like 1e146 that satisfies
    every finiteness and mask check and is written out as an imputation.

    A row is allowed to exceed BOTH of two reference quantities by ``max_ratio``
    before it is rejected:

    * its sequencing depth -- a variant count cannot exceed the number of
      sequences the variants were called from;
    * the initializer's own total -- because kNN-Aitchison already returns rows
      summing to ~100x their depth when the depth is tiny, and rejecting those
      would reject the baseline rather than the regression.

    Only failing both matters. A high-depth row whose reported variants are all
    pseudo-counts (X1 total 48, depth 121134) legitimately gains orders of
    magnitude when its unreported variants are filled, and must not be flagged.
    """
    x_hat = np.asarray(x_hat, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)
    if x_hat.shape != x1.shape:
        raise ValueError('output and initializer must have equal shapes')
    if not np.isfinite(x_hat).all():
        raise ValueError('reconstructed counts contain non-finite values')
    after = x_hat.reshape(-1, x_hat.shape[-1]).sum(axis=-1)
    before = x1.reshape(-1, x1.shape[-1]).sum(axis=-1)
    depth = np.asarray(total_sequence, dtype=np.float64).reshape(-1)
    if depth.shape != after.shape:
        raise ValueError('total_sequence must have one entry per row')
    bound = max_ratio * np.maximum(np.maximum(depth, before), 1.)
    bad = after > bound
    if bad.any():
        worst = int(np.argmax(after / bound))
        raise ValueError(
            f'{int(bad.sum())} of {len(after)} rows exceed {max_ratio:g}x both their '
            f'sequencing depth and their initializer total. Worst row {worst}: '
            f'X1 sum {before[worst]:.3e}, depth {depth[worst]:.0f}, output '
            f'{after[worst]:.3e}, max cell '
            f'{x_hat.reshape(-1, x_hat.shape[-1])[worst].max():.3e}. The predicted '
            'CLR diverged; these values are not counts and must not be written '
            'out as imputations.')


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


def single_pass(x1, m0, predicted_clr, *, max_iter=1, validation_mae=None,
                total_sequence=None, max_ratio=10.):
    if max_iter != 1:
        raise ValueError('This implementation requires max_iter=1')
    frozen = m0.copy()
    output = complete_counts(x1, m0, predicted_clr)
    if total_sequence is not None:
        check_counts_are_physical(output, x1, total_sequence, max_ratio)
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
