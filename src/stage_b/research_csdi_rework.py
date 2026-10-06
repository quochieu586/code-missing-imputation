"""Reworked CSDI core: a compositional adaptation of the frozen ermongroup/CSDI
snapshot, generalized to arbitrary channel widths and a configurable number of
attention heads. Numerically identical to the frozen ``CSDICore`` when called
with ``channels=16`` and ``heads=4`` (verified by a parity test); the rework
experiment itself uses a single-head policy so that a channels 16 vs 17 vs 8
comparison varies only the latent width, not the head count.

Tensor convention B,L,K; float64 throughout. The schedule, forward/noise/loss,
conditioning, projection and sampling helpers are imported verbatim from the
frozen :mod:`src.stage_b.research_csdi` so the numerical core cannot drift.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn

from src.stage_b.research_csdi import (condition_features, forward_noise,
    epsilon_loss, project_final, reverse_step, schedule, sample_latents)


class ResidualBlockRework(nn.Module):
    def __init__(self, channels=16, heads=4):
        super().__init__()
        self.time_attention = nn.TransformerEncoderLayer(channels, heads, 64,
            dropout=0., activation='gelu', batch_first=True, dtype=torch.float64)
        self.feature_attention = nn.TransformerEncoderLayer(channels, heads, 64,
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


class CSDICoreRework(nn.Module):
    def __init__(self, latent_dim, raw_dim=17, channels=16, steps=20, layers=2, heads=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.channels = channels
        self.heads = heads
        self.condition_encoder = nn.Sequential(nn.Linear(5*raw_dim, channels,
            dtype=torch.float64), nn.SiLU(), nn.Linear(channels,channels,dtype=torch.float64))
        self.input_projection = nn.Linear(1,channels,dtype=torch.float64)
        self.feature_embedding = nn.Embedding(latent_dim,channels,dtype=torch.float64)
        self.step_embedding = nn.Sequential(nn.Linear(channels,channels,dtype=torch.float64),
            nn.SiLU(),nn.Linear(channels,channels,dtype=torch.float64),nn.SiLU())
        half = (channels + 1) // 2
        freq = 10.**(torch.arange(half, dtype=torch.float64) / (half - 1) * 4.)
        phase = torch.arange(steps, dtype=torch.float64)[:, None] * freq
        step_table = torch.cat([phase.sin(), phase.cos()], dim=-1)
        if step_table.shape[-1] > channels:
            step_table = step_table[..., :channels]
        self.register_buffer('step_table', step_table)
        self.layers = nn.ModuleList([ResidualBlockRework(channels, heads) for _ in range(layers)])
        self.output = nn.Sequential(nn.Linear(channels,channels,dtype=torch.float64),
                                   nn.ReLU(),nn.Linear(channels,1,dtype=torch.float64))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, noisy, condition, times, steps):
        if noisy.dtype != torch.float64 or condition.dtype != torch.float64:
            raise TypeError('CSDICore requires float64')
        if noisy.shape[-1] != self.latent_dim:
            raise ValueError('latent dimension mismatch')
        side = self.condition_encoder(condition)[:,:,None,:]
        side = side + time_embedding_rework(times,self.channels)[:,:,None,:]
        side = side + self.feature_embedding.weight[None,None,:,:]
        x = torch.relu(self.input_projection(noisy[...,None]))
        step = self.step_embedding(self.step_table[steps])
        skips = []
        for layer in self.layers:
            x, skip = layer(x,side,step)
            skips.append(skip)
        return self.output(torch.stack(skips).sum(0)/math.sqrt(len(skips))).squeeze(-1)


def time_embedding_rework(pos, channels):
    frequencies = torch.exp(-math.log(10000.) * torch.arange(0, channels, 2,
                            dtype=pos.dtype, device=pos.device) / channels)
    phase = pos[..., None] * frequencies
    embedding = torch.cat([phase.sin(), phase.cos()], dim=-1)
    if embedding.shape[-1] > channels:
        embedding = embedding[..., :channels]
    return embedding
