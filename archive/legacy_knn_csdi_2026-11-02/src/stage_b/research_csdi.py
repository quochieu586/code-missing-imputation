"""CSDI adaptation: separate raw condition and full noisy LR target.

Temporal/feature attention and gated residual/skip blocks follow the frozen
ermongroup/CSDI snapshot in ref/csdi_official. This is a compositional adaptation,
not a replication of the original paper's experiments. Tensor convention B,L,K;
no raw visibility mask is copied to the latent branch. All numerics are float64.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn


def schedule(steps=20):
    beta = torch.tensor(np.linspace(1e-4 ** .5, .5 ** .5, steps) ** 2,
                        dtype=torch.float64)
    return beta, torch.cumprod(1 - beta, dim=0)


def forward_noise(z0, noise, alpha_bar):
    return alpha_bar.sqrt() * z0 + (1 - alpha_bar).sqrt() * noise


def epsilon_loss(pred, noise, eligible):
    # Index first: unknown labels, even NaNs, must never enter arithmetic.
    if not eligible.any():
        raise ValueError('no identifiable full-LR labels')
    return (pred[eligible] - noise[eligible]).square().mean()


def reverse_step(current, epsilon, t, beta, alpha_bar, noise=None):
    """Stock CSDI epsilon mean/posterior variance; final step adds no noise."""
    mean = (current - beta[t] / (1 - alpha_bar[t]).sqrt() * epsilon) / (1 - beta[t]).sqrt()
    if t == 0:
        return mean
    if noise is None:
        raise ValueError('reverse noise required for a non-final step')
    sigma = (beta[t] * (1 - alpha_bar[t-1]) / (1 - alpha_bar[t])).sqrt()
    return mean + sigma * noise


def project_final(z, name, references=()):
    if name.startswith('clr'):
        return z - z.mean(dim=-1, keepdim=True)
    if name.startswith('hkglr'):
        return z - z[..., list(references)].mean(dim=-1, keepdim=True)
    if name.startswith('ilr'):
        return z
    raise ValueError(name)


def condition_features(raw, visible, mode, raw_scale, initialized=None,
                       fallback=None, effective_k=None):
    """Raw 17-part condition; hidden target poisoning has no effect here.

    Slots are visible log1p, visibility, initialized log1p, fallback and k/4.
    Every mode/transform uses the same 5*D input size and width-16 encoder.
    """
    x = np.asarray(raw, dtype=np.float64)
    mask = np.asarray(visible, dtype=bool)
    safe = np.where(mask, x, 0.)
    if x.shape != mask.shape or not np.isfinite(safe).all() or (safe < 0).any():
        raise ValueError('invalid visible raw input')
    zeros = np.zeros_like(safe)
    if mode == 'mask_only':
        values, init, fb, k = zeros, zeros, zeros, zeros
    elif mode == 'no_init':
        values, init, fb, k = np.log1p(safe) / raw_scale, zeros, zeros, zeros
    elif mode == 'init':
        init_raw = np.asarray(initialized, dtype=np.float64)
        if init_raw.shape != x.shape or not np.isfinite(init_raw).all() or (init_raw < 0).any():
            raise ValueError('init mode requires finite raw initialization')
        values = np.log1p(safe) / raw_scale
        init = np.log1p(init_raw) / raw_scale
        fb = np.asarray(fallback, dtype=np.float64)
        k = np.asarray(effective_k, dtype=np.float64) / 4.
        if fb.shape != x.shape or k.shape != x.shape:
            raise ValueError('initializer provenance shape mismatch')
    else:
        raise ValueError(mode)
    return np.concatenate([values, mask.astype(np.float64), init, fb, k], axis=-1)


def time_embedding(pos, channels):
    frequencies = torch.exp(-math.log(10000.) * torch.arange(0, channels, 2,
                            dtype=pos.dtype, device=pos.device) / channels)
    phase = pos[..., None] * frequencies
    return torch.cat([phase.sin(), phase.cos()], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, channels=16):
        super().__init__()
        self.time_attention = nn.TransformerEncoderLayer(channels, 4, 64,
            dropout=0., activation='gelu', batch_first=True, dtype=torch.float64)
        self.feature_attention = nn.TransformerEncoderLayer(channels, 4, 64,
            dropout=0., activation='gelu', batch_first=True, dtype=torch.float64)
        self.step_projection = nn.Linear(channels, channels, dtype=torch.float64)
        self.mid_projection = nn.Linear(channels, 2*channels, dtype=torch.float64)
        self.condition_projection = nn.Linear(channels, 2*channels, dtype=torch.float64)
        self.output_projection = nn.Linear(channels, 2*channels, dtype=torch.float64)

    def forward(self, x, side, step):
        b, l, k, c = x.shape
        y = x + self.step_projection(step)[:, None, None, :]
        if l > 1:
            y = self.time_attention(y.permute(0,2,1,3).reshape(b*k,l,c))
            y = y.reshape(b,k,l,c).permute(0,2,1,3)
        y = self.feature_attention(y.reshape(b*l,k,c)).reshape(b,l,k,c)
        y = self.mid_projection(y) + self.condition_projection(side)
        gate, filt = y.chunk(2, dim=-1)
        residual, skip = self.output_projection(gate.sigmoid()*filt.tanh()).chunk(2,dim=-1)
        return (x+residual)/math.sqrt(2.), skip


class CSDICore(nn.Module):
    def __init__(self, latent_dim, raw_dim=17, channels=16, steps=20, layers=2):
        super().__init__()
        if channels != 16:
            raise ValueError('E3 capacity frozen at 16 channels')
        self.latent_dim = latent_dim
        self.condition_encoder = nn.Sequential(nn.Linear(5*raw_dim, channels,
            dtype=torch.float64), nn.SiLU(), nn.Linear(channels,channels,dtype=torch.float64))
        self.input_projection = nn.Linear(1,channels,dtype=torch.float64)
        self.feature_embedding = nn.Embedding(latent_dim,channels,dtype=torch.float64)
        self.step_embedding = nn.Sequential(nn.Linear(channels,channels,dtype=torch.float64),
            nn.SiLU(),nn.Linear(channels,channels,dtype=torch.float64),nn.SiLU())
        half = channels//2
        freq = 10.**(torch.arange(half,dtype=torch.float64)/(half-1)*4.)
        phase = torch.arange(steps,dtype=torch.float64)[:,None]*freq
        self.register_buffer('step_table',torch.cat([phase.sin(),phase.cos()],dim=-1))
        self.layers = nn.ModuleList([ResidualBlock(channels) for _ in range(layers)])
        self.output = nn.Sequential(nn.Linear(channels,channels,dtype=torch.float64),
                                   nn.ReLU(),nn.Linear(channels,1,dtype=torch.float64))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
        self.channels = channels

    def forward(self, noisy, condition, times, steps):
        if noisy.dtype != torch.float64 or condition.dtype != torch.float64:
            raise TypeError('CSDI requires float64')
        if noisy.shape[-1] != self.latent_dim:
            raise ValueError('latent dimension mismatch')
        side = self.condition_encoder(condition)[:,:,None,:]
        side = side + time_embedding(times,self.channels)[:,:,None,:]
        side = side + self.feature_embedding.weight[None,None,:,:]
        x = torch.relu(self.input_projection(noisy[...,None]))
        step = self.step_embedding(self.step_table[steps])
        skips = []
        for layer in self.layers:
            x, skip = layer(x,side,step)
            skips.append(skip)
        return self.output(torch.stack(skips).sum(0)/math.sqrt(len(skips))).squeeze(-1)


@torch.no_grad()
def sample_latents(model, condition, times, beta, alpha_bar, mean, scale,
                   name, references=(), samples=2, seed=42):
    """IID Gaussian on full LR; project once after reverse, before inverse.

    No per-step projection or clipping. Nonfinite chains raise, never silently
    become a baseline. Observed raw restoration belongs to the inverse adapter.
    """
    model.eval()
    b,l,_ = condition.shape
    rng = torch.Generator().manual_seed(seed)
    results = []
    max_norm, max_eps = 0.,0.
    for _ in range(samples):
        current = torch.randn((b,l,model.latent_dim),dtype=torch.float64,generator=rng)
        for t in reversed(range(len(beta))):
            steps = torch.full((b,),t,dtype=torch.int64)
            eps = model(current,condition,times,steps)
            noise = torch.randn(current.shape,dtype=torch.float64,generator=rng) if t else None
            current = reverse_step(current,eps,t,beta,alpha_bar,noise)
            if not torch.isfinite(current).all():
                raise FloatingPointError(f'nonfinite DDPM chain at step {t}')
            max_norm = max(max_norm,float(current.abs().max()))
            max_eps = max(max_eps,float(eps.abs().max()))
        latent = current*scale + mean
        results.append(project_final(latent,name,references))
    return torch.stack(results), {'max_abs_standardized_latent':max_norm,
        'max_abs_epsilon':max_eps,'clipped_fraction':0.,'projection':'final_only'}
