"""PyTorch rewrite of the DeepMicroGen architecture.

CNN feature extractor + bidirectional RNN generator with time-decay
weighting + LSTM discriminator with time-classification auxiliary loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CNNFeatureExtractor(nn.Module):
    def __init__(self, n_features: int, hidden1: int = 16, hidden2: int = 8):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, hidden1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden1, hidden2, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dropout(self.act(self.conv1(x)))
        h = self.dropout(self.act(self.conv2(h)))
        return h


class DeepMicroGenGenerator(nn.Module):
    def __init__(
        self,
        n_features: int,
        rnn_hidden: int = 10,
        cnn_hidden1: int = 16,
        cnn_hidden2: int = 8,
    ):
        super().__init__()
        self.n_features = n_features
        # CNN input size includes n_features + p_nonzero (which is 1 per variant when concatenated)
        # p_nonzero is per-variant, so adds n_features dimensions
        cnn_input_size = n_features * 2  # features + p_nonzero per feature
        self.cnn = CNNFeatureExtractor(cnn_input_size, cnn_hidden1, cnn_hidden2)
        self.rnn_f = nn.RNN(cnn_hidden2, rnn_hidden, batch_first=True)
        self.rnn_b = nn.RNN(cnn_hidden2, rnn_hidden, batch_first=True)
        self.dense_f = nn.Linear(rnn_hidden, n_features)
        self.dense_b = nn.Linear(rnn_hidden, n_features)
        self.decay_f_fc = nn.Linear(1, 1)
        self.decay_b_fc = nn.Linear(1, 1)
        self.act = nn.LeakyReLU(0.2)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        time_decay_f: torch.Tensor,
        time_decay_b: torch.Tensor,
        p_nonzero: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            estimated: imputed values
            imputed: x_real + x_est (with mask applied)
            est_f: forward RNN output
            est_b: backward RNN output
        """
        # Apply p_nonzero conditioning BEFORE CNN (concatenate as additional feature per variant)
        if p_nonzero is not None:
            # p_nonzero shape: [batch, time, features] or [batch, time, 1]
            if p_nonzero.dim() == 2:
                p_nonzero = p_nonzero.unsqueeze(-1)
            # Expand p_nonzero to match feature dimension if needed
            if p_nonzero.size(-1) == 1:
                p_nonzero = p_nonzero.expand(-1, -1, self.n_features)
            x = torch.cat([x, p_nonzero], dim=-1)

        cnn_input = x.transpose(1, 2)
        feats = self.cnn(cnn_input).transpose(1, 2)

        out_f, _ = self.rnn_f(feats)
        feats_rev = torch.flip(feats, dims=[1])
        out_b, _ = self.rnn_b(feats_rev)
        out_b = torch.flip(out_b, dims=[1])

        est_f = self.dense_f(out_f)
        est_b = self.dense_b(out_b)

        lam_f = torch.exp(-torch.clamp(self.act(self.decay_f_fc(time_decay_f)), min=0.0))
        lam_b = torch.exp(-torch.clamp(self.act(self.decay_b_fc(time_decay_b)), min=0.0))

        estimated = lam_f * est_f + lam_b * est_b

        x_real = x[:, :, :self.n_features] * mask  # Only use original features for x_real
        x_est = estimated * (1.0 - mask)
        imputed = x_real + x_est
        return estimated, imputed, est_f, est_b


class DeepMicroGenDiscriminator(nn.Module):
    def __init__(self, n_features: int, n_timepoints: int, lstm_hidden: int = 10):
        super().__init__()
        self.lstm = nn.LSTM(n_features, lstm_hidden, batch_first=True)
        self.time_classifier = nn.Linear(lstm_hidden, n_timepoints)
        self.real_fake_head = nn.Linear(lstm_hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, _ = self.lstm(x)
        logits_time = self.time_classifier(h)
        logits_rt = self.real_fake_head(h)
        return torch.sigmoid(logits_rt), logits_time


class DeepMicroGenGAN(nn.Module):
    def __init__(self, n_features: int, n_timepoints: int):
        super().__init__()
        self.generator = DeepMicroGenGenerator(n_features)
        self.discriminator = DeepMicroGenDiscriminator(n_features, n_timepoints)