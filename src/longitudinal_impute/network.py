"""Small direct-composition network; no data loading or training loop."""
import math
import torch
from torch import nn


class CompositionNet(nn.Module):
    def __init__(self, components, channels=32, heads=4, *, metadata_spec=None):
        super().__init__()
        if (not isinstance(components, int) or isinstance(components, bool) or components < 2
                or not isinstance(channels, int) or isinstance(channels, bool) or channels < 2
                or not isinstance(heads, int) or isinstance(heads, bool) or heads < 1
                or channels % 2 or channels % heads):
            raise ValueError('components >= 2; channels must be positive, even and divisible by heads')
        self.components = components
        self.input = nn.Linear(2, channels)
        self.feature = nn.Embedding(components, channels)
        self.register_buffer('frequencies', torch.exp(
            -math.log(10000.) * torch.arange(channels // 2) / (channels // 2)))
        self.temporal = nn.MultiheadAttention(channels, heads, dropout=0., batch_first=True)
        self.features = nn.MultiheadAttention(channels, heads, dropout=0., batch_first=True)
        self.time_norm = nn.LayerNorm(channels)
        self.feature_norm = nn.LayerNorm(channels)
        self.gate_norm = nn.LayerNorm(channels)
        self.gate = nn.Linear(channels, 2 * channels)
        self.head = nn.Sequential(nn.Linear(channels, channels), nn.ReLU(), nn.Linear(channels, 1))
        self.metadata_spec = metadata_spec
        if metadata_spec is not None:
            self.metadata_embeddings = nn.ModuleDict({
                field: nn.Embedding(len(vocab), channels)
                for field, vocab in metadata_spec['vocabularies'].items()})
            self.metadata_age = nn.Linear(1, channels)

    def forward(self, features, observed_mask, times, valid_steps, metadata=None):
        if (features.ndim != 3 or features.shape[-1] != self.components
                or observed_mask.shape != features.shape or observed_mask.dtype != torch.bool
                or times.shape != features.shape[:2] or valid_steps.shape != times.shape
                or valid_steps.dtype != torch.bool or not features.is_floating_point()):
            raise ValueError('network: invalid shapes, feature dtype or masks')
        b, length, k = features.shape
        if not b or not length or not valid_steps.any(dim=1).all():
            raise ValueError('network: each entity needs at least one valid step')
        if not torch.isfinite(features[valid_steps]).all() or not torch.isfinite(times[valid_steps]).all():
            raise ValueError('network: real features and times must be finite')
        # Padding is removed BEFORE projections/attention, including NaN padding.
        z = features.masked_fill(~valid_steps[..., None], 0.)
        observed = observed_mask & valid_steps[..., None]
        t = times.masked_fill(~valid_steps, 0.)
        phase = t[..., None] * self.frequencies
        time_embedding = torch.cat((phase.sin(), phase.cos()), dim=-1)
        h = self.input(torch.stack((z, observed.to(z.dtype)), dim=-1))
        h = h + self.feature.weight[None, None] + time_embedding[:, :, None]
        if self.metadata_spec is not None:
            expected = set(self.metadata_embeddings) | {'median_age'}
            if not isinstance(metadata, dict) or set(metadata) != expected:
                raise ValueError('network: metadata tensors required')
            age = metadata['median_age']
            if age.shape != (b, 1) or not torch.isfinite(age).all():
                raise ValueError('network: invalid metadata age shape or values')
            m = self.metadata_age(age)
            for field, encoder in self.metadata_embeddings.items():
                codes = metadata[field]
                if (codes.shape != (b,) or codes.dtype != torch.long
                        or (codes < 0).any() or (codes >= encoder.num_embeddings).any()):
                    raise ValueError(f'network: invalid metadata {field}')
                m = m + encoder(codes)
            h = h + m[:, None, None, :]

        # Temporal attention: independent sequences for each entity x component.
        a = self.time_norm(h).permute(0, 2, 1, 3).reshape(b * k, length, -1)
        padding = (~valid_steps)[:, None, :].expand(b, k, length).reshape(b * k, length)
        a = self.temporal(a, a, a, key_padding_mask=padding, need_weights=False)[0]
        h = h + a.reshape(b, k, length, -1).permute(0, 2, 1, 3)
        # Feature attention never crosses a timepoint or an entity boundary.
        a = self.feature_norm(h).reshape(b * length, k, -1)
        a = self.features(a, a, a, need_weights=False)[0]
        h = h + a.reshape(b, length, k, -1)
        content, gate = self.gate(self.gate_norm(h)).chunk(2, dim=-1)
        h = h + content.tanh() * gate.sigmoid()
        proposals = self.head(h).squeeze(-1).softmax(dim=-1)
        return proposals.masked_fill(~valid_steps[..., None], 0.)
