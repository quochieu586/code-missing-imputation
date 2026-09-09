"""GAN training loop on M_artificial_mag masked positives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .data import GANDataset
from .model import DeepMicroGenGAN


@dataclass
class TrainConfig:
    epochs: int = 300
    lr: float = 0.001
    d_steps: int = 5
    g_steps: int = 1
    patience: int = 50
    artificial_mask_fraction: float = 0.2
    seed: int = 42
    device: str = "cpu"
    consistency_weight: float = 1.0  # weight for consistency loss
    time_class_weight: float = 1.0  # weight for time classification loss
    grad_clip: float = 1.0  # gradient clipping threshold
    cnn_hidden1: int = 16
    cnn_hidden2: int = 8
    rnn_hidden: int = 10
    lstm_hidden: int = 10


def variant_nonzero_prior(dataset: GANDataset) -> np.ndarray:
    """Per-feature P(count > 0) over cells locked by M_fixed, used as a neutral prior."""
    observed = dataset.M_observed == 1
    n_obs = observed.sum(axis=(0, 1))
    n_pos = ((dataset.counts_raw > 0) & observed).sum(axis=(0, 1))
    return np.divide(
        n_pos.astype(np.float64),
        n_obs.astype(np.float64),
        out=np.full(n_obs.shape, 0.5, dtype=np.float64),
        where=n_obs > 0,
    )


def create_artificial_mag_mask(
    dataset: GANDataset, fraction: float, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    positive_obs = (dataset.counts_raw > 0) & (dataset.M_observed == 1)
    mask = np.zeros_like(dataset.M_observed, dtype=np.uint8)
    pos_locs, pos_times, pos_feats = np.where(positive_obs)
    n_sel = int(len(pos_locs) * fraction)
    if n_sel > 0:
        sel = rng.choice(len(pos_locs), size=n_sel, replace=False)
        mask[pos_locs[sel], pos_times[sel], pos_feats[sel]] = 1
    return mask


def train_gan(
    dataset: GANDataset,
    config: TrainConfig,
) -> tuple[np.ndarray, dict]:
    torch.manual_seed(config.seed)

    n_locs, n_times, n_feats = dataset.counts_raw.shape
    device = torch.device(config.device)

    model = DeepMicroGenGAN(
        n_feats,
        n_times,
        cnn_hidden1=config.cnn_hidden1,
        cnn_hidden2=config.cnn_hidden2,
        rnn_hidden=config.rnn_hidden,
        lstm_hidden=config.lstm_hidden,
    ).to(device)
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=config.lr)
    opt_g = torch.optim.Adam(model.generator.parameters(), lr=config.lr)
    bce = nn.BCELoss()
    ce = nn.CrossEntropyLoss()

    M_art = create_artificial_mag_mask(dataset, config.artificial_mask_fraction, config.seed)

    # Prepare p_nonzero conditioning for generator. On artificially masked cells the
    # panel channel holds the observed indicator (always 1 there, since only positive
    # observed cells are masked), which would leak the reconstruction target and would
    # not match the calibrated probabilities the generator sees on M_gan at inference.
    # Replace those with the per-feature observed prevalence.
    p_nonzero = None
    if dataset.p_nonzero is not None:
        p_nonzero = dataset.p_nonzero.astype(np.float32)
        prior = variant_nonzero_prior(dataset).astype(np.float32)
        p_nonzero = np.where(
            M_art.astype(bool), np.broadcast_to(prior, p_nonzero.shape), p_nonzero
        ).astype(np.float32)

    x_np = dataset.X_clr.astype(np.float32)
    decay_f_np = dataset.time_decay_f.astype(np.float32)
    decay_b_np = dataset.time_decay_b.astype(np.float32)

    best_recon = np.inf
    patience_counter = 0
    best_imputed: np.ndarray | None = None
    history: list[dict] = []

    for epoch in range(config.epochs):
        x_masked = np.where(M_art == 1, 0.0, x_np)

        x_t = torch.tensor(x_masked, device=device)
        mask_t = torch.tensor(1.0 - M_art.astype(np.float32), device=device)
        decay_f_t = torch.tensor(decay_f_np, device=device)
        decay_b_t = torch.tensor(decay_b_np, device=device)
        p_nonzero_t = torch.tensor(p_nonzero, device=device) if p_nonzero is not None else None

        # Time labels for discriminator time classification (one-hot encoded time indices)
        time_labels = torch.arange(n_locs * n_times, device=device) % n_times
        time_labels = time_labels.view(n_locs, n_times).long()

        for _ in range(config.d_steps):
            opt_d.zero_grad()
            estimated, imputed, est_f, est_b = model.generator(x_t, mask_t, decay_f_t, decay_b_t, p_nonzero_t)
            real_part = x_t * mask_t
            fake_part = estimated * (1.0 - mask_t)
            d_real_pred, d_real_time_logits = model.discriminator(real_part)
            d_fake_pred, d_fake_time_logits = model.discriminator(fake_part.detach())
            
            # Adversarial loss
            d_adv_loss = bce(d_real_pred, torch.ones_like(d_real_pred)) + bce(
                d_fake_pred, torch.zeros_like(d_fake_pred)
            )
            # Time classification loss (only on real data)
            d_time_loss = ce(d_real_time_logits.view(-1, d_real_time_logits.size(-1)), time_labels.view(-1))
            
            d_loss = d_adv_loss + config.time_class_weight * d_time_loss
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), config.grad_clip)
            opt_d.step()

        for _ in range(config.g_steps):
            opt_g.zero_grad()
            estimated, imputed, est_f, est_b = model.generator(x_t, mask_t, decay_f_t, decay_b_t, p_nonzero_t)
            fake_part = estimated * (1.0 - mask_t)
            d_fake_pred, _ = model.discriminator(fake_part)
            g_adv = bce(d_fake_pred, torch.ones_like(d_fake_pred))
            
            # Reconstruction loss on artificial masked positives
            art_mask_t = torch.tensor(M_art.astype(np.float32), device=device)
            recon = torch.mean(
                torch.abs((x_t - estimated) * art_mask_t)
            ) / max(art_mask_t.mean().item(), 1e-6)
            
            # Consistency loss: |est_f - est_b|
            consistency = torch.mean(torch.abs(est_f - est_b))
            
            g_loss = g_adv + recon + config.consistency_weight * consistency
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.generator.parameters(), config.grad_clip)
            opt_g.step()

        with torch.no_grad():
            _, imputed_eval, _, _ = model.generator(x_t, mask_t, decay_f_t, decay_b_t, p_nonzero_t)
            recon_val = float(
                torch.mean(torch.abs((x_t - imputed_eval) * torch.tensor(
                    M_art.astype(np.float32), device=device
                ))) / max(M_art.mean(), 1e-6)
            )

        history.append({"epoch": epoch, "recon": recon_val, "g_loss": float(g_loss.item())})

        if recon_val < best_recon:
            best_recon = recon_val
            patience_counter = 0
            with torch.no_grad():
                best_imputed = imputed_eval.cpu().numpy()
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

    if best_imputed is None:
        best_imputed = x_np

    final_clr = np.where(M_art == 0, x_np, best_imputed)
    final_clr = np.where(dataset.M_observed == 1, x_np, final_clr)

    return final_clr, {"history": history, "best_recon": best_recon, "epochs_run": len(history)}