"""Conditional epsilon diffusion with separate time/feature transformers.

Architecture and DDPM equations follow the local Seki CSDI_phylo source.
Reimplemented for (B,L,K), cell masks, initialization context and optional
group CNNs. No microbiome phylum labels are inferred for viral variants.
"""
import math
import torch
from torch import nn


class Denoiser(nn.Module):
    def __init__(self, k, channels=32, groups=None):
        super().__init__()
        self.k = k
        self.input = nn.Linear(4, channels)
        self.feature = nn.Embedding(k, channels)
        self.step = nn.Sequential(nn.Linear(channels, channels), nn.SiLU(), nn.Linear(channels, channels))
        self.channels = channels
        self.time_attention = nn.TransformerEncoderLayer(channels, 4, channels * 2, dropout=0., batch_first=True)
        self.feature_attention = nn.TransformerEncoderLayer(channels, 4, channels * 2, dropout=0., batch_first=True)
        self.output = nn.Sequential(nn.Linear(channels, channels), nn.ReLU(), nn.Linear(channels, 1))
        self.group_indices = [] if groups is None else [
            [i for i, g in enumerate(groups) if g == name] for name in sorted(set(groups))]
        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(channels, channels, (3, 1), padding=(1, 0)), nn.ReLU(),
            nn.Conv2d(channels, channels, (3, 1), padding=(1, 0)), nn.ReLU())
            for _ in self.group_indices])

    def embedding(self, pos):
        freq = torch.exp(-math.log(10000) * torch.arange(self.channels // 2, device=pos.device) / (self.channels // 2))
        phase = pos[..., None] * freq
        return torch.cat([phase.sin(), phase.cos()], -1)

    def forward(self, noisy, context, observed, times, step):
        b, l, k = noisy.shape
        x = self.input(torch.stack([noisy, context, observed.float(), (~observed).float()], -1))
        x = x + self.feature.weight[None, None] + self.embedding(times)[:, :, None]
        x = x + self.step(self.embedding(step.float()))[:, None, None]
        if self.group_indices:
            y = torch.zeros_like(x)
            for indices, conv in zip(self.group_indices, self.convs):
                y[:, :, indices] = conv(x[:, :, indices].permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            x = x + y
        x = self.time_attention(x.permute(0, 2, 1, 3).reshape(b*k, l, -1))
        x = x.reshape(b, k, l, -1).permute(0, 2, 1, 3)
        x = self.feature_attention(x.reshape(b*l, k, -1)).reshape(b, l, k, -1)
        return self.output(x).squeeze(-1)
