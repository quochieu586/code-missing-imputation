"""Selective false-zero risk gate with Wilson upper confidence bound and Conformal Risk Control."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from scipy.stats import norm


def wilson_upper_bound(successes: int, n: int, confidence: float = 0.95) -> float:
    if n == 0:
        return 1.0
    z = norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / n
    denom = 1 + z * z / n
    center = p_hat + z * z / (2 * n)
    spread = z * np.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))
    return float((center + spread) / denom)


def temperature_scale(logits: np.ndarray, y: np.ndarray, t_bounds: tuple[float, float] = (0.1, 10.0)) -> float:
    """Fit temperature scaling parameter T on logits to minimize NLL."""
    def nll(T: float) -> float:
        q = expit(logits / T)
        # Avoid log(0)
        q = np.clip(q, 1e-15, 1 - 1e-15)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

    res = minimize_scalar(lambda T: nll(T), bounds=t_bounds, method="bounded")
    return float(res.x)


def conformal_zero_bias(
    y: np.ndarray,
    p_hat: np.ndarray,
    alpha: float = 0.05,
    b_max: float = 5.0,
    n_iter: int = 40,
    min_positive: int = 10,
) -> tuple[float, dict]:
    """
    Select bias b for soft zero-gating using Conformal Risk Control.

    Returns:
        b_opt: optimal bias (>= 0)
        info: diagnostics dict
    """
    y = np.asarray(y)
    p = np.asarray(p_hat)

    if y.sum() < min_positive:
        return 0.5, {"status": "fallback_low_support"}

    def false_zero_rate(b: float) -> float:
        # w = sigma(logit(p) - b); predicted zero when w < 0.5 -> logit(p) < b -> p < expit(b)
        pred_zero = p < expit(b)
        n_zero = int(pred_zero.sum())
        if n_zero == 0:
            return 0.0
        false_zero = int((pred_zero & (y == 1)).sum())
        return false_zero / n_zero

    # L(b) = false-zero rate is monotone non-decreasing in b
    # Find largest b such that L(b) <= alpha (strongest zero bias with guarantee)
    lo, hi = 0.0, b_max

    if false_zero_rate(lo) > alpha:
        return 0.0, {"status": "fzr_at_zero_exceeds_alpha", "fzr_at_zero": float(false_zero_rate(lo))}

    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if false_zero_rate(mid) <= alpha:
            lo = mid
        else:
            hi = mid

    b_opt = lo
    fzr_opt = false_zero_rate(b_opt)
    return b_opt, {"b_opt": b_opt, "fzr_opt": fzr_opt, "status": "converged"}


def fit_temperature_and_crc(
    oof_df,
    y_true_col: str = "y_true",
    p_calibrated_col: str = "p_calibrated",
    variant_col: str = "variant",
    variant_names: list[str] | None = None,
    alpha: float = 0.05,
    b_max: float = 5.0,
    min_positive: int = 10,
) -> dict:
    """
    Fit temperature scaling and CRC bias per variant on OOF calibration data.

    Returns:
        dict with:
            - 'temperature': fitted temperature T
            - 'bias_per_variant': {variant: b}
            - 'diagnostics': per-variant CRC diagnostics
    """
    # Recover logits from calibrated probabilities (clip to avoid inf)
    p_cal = oof_df[p_calibrated_col].clip(1e-12, 1 - 1e-12).to_numpy()
    logits = logit(oof_df[p_calibrated_col].clip(1e-12, 1 - 1e-12).to_numpy())
    y = oof_df[y_true_col].to_numpy().astype(int)

    # Fit temperature scaling
    T = temperature_scale(logits, y)

    # Recalibrate probabilities
    logits_cal = logits / T
    p_cal = expit(logits_cal)

    # Per-variant CRC bias
    variants = variant_names or sorted(oof_df[variant_col].unique())
    bias_per_variant = {}
    diagnostics = {}

    for variant in variants:
        mask = (oof_df[variant_col] == variant).to_numpy()
        if mask.sum() == 0:
            bias_per_variant[variant] = 0.0
            diagnostics[variant] = {"status": "no_calibration_data"}
            continue

        y_var = y[mask]

        if y_var.sum() < 10:  # fallback if too few positives
            bias_per_variant[variant] = 0.0
            diagnostics[variant] = {"status": "fallback_low_positive"}
            continue

        b, diag = conformal_zero_bias(
            y_var,
            p_cal[mask],  # use calibrated probs directly
            alpha=0.05,
            b_max=5.0,
            min_positive=10,
        )
        bias_per_variant[variant] = b
        diagnostics[variant] = diag

    return {
        "temperature": T,
        "bias_per_variant": bias_per_variant,
        "diagnostics": diagnostics,
    }


@dataclass
class VariantGateResult:
    variant: str
    status: str
    tau_zero: float = 0.0
    n_predicted_zero: int = 0
    false_zero_rate: float = 0.0
    false_zero_ucb: float = 0.0
    n_support: int = 0
    w_gan: float = 1.0  # soft weight for GAN magnitude
    b_crc: float = 0.0  # CRC bias


@dataclass
class OccurrenceGateResult:
    per_variant: list[VariantGateResult] = field(default_factory=list)
    M_target_zero: np.ndarray | None = None
    M_gan: np.ndarray | None = None
    M_hard_lock: np.ndarray | None = None  # hard-locked zeros (w < tau_hard)
    M_soft_fusion: np.ndarray | None = None  # cells with soft fusion
    released: bool = False
    release_reason: str = ""

    def validate(self, M_target: np.ndarray) -> None:
        if self.M_target_zero is None or self.M_gan is None:
            raise ValueError("Gate not applied")
        if np.any(self.M_target_zero.astype(bool) & self.M_gan.astype(bool)):
            raise ValueError("M_target_zero and M_gan overlap")
        if not np.array_equal(
            (self.M_target_zero.astype(bool) | self.M_gan.astype(bool)),
            M_target.astype(bool),
        ):
            raise ValueError("M_target_zero OR M_gan != M_target")


def selective_false_zero_gate(
    y_heldout: np.ndarray,
    p_heldout: np.ndarray,
    threshold_grid: list[float],
    max_false_zero_rate: float = 0.05,
    min_support: int = 50,
    confidence: float = 0.95,
) -> tuple[float | None, dict]:
    """Legacy: Select the most conservative tau such that predicted-zero cells have
    false-zero UCB <= max_false_zero_rate. Kept for backward compatibility.
    """
    y = np.asarray(y_heldout)
    p = np.asarray(p_heldout)
    best_tau = None
    diagnostics: dict = {"n_heldout": len(y), "candidates": []}

    for tau in sorted(threshold_grid):
        pred_zero = p < tau
        n_pred_zero = int(pred_zero.sum())
        if n_pred_zero < min_support:
            continue
        false_zero_count = int((pred_zero & (y == 1)).sum())
        fzr = false_zero_count / n_pred_zero
        ucb = wilson_upper_bound(false_zero_count, n_pred_zero, 0.95)
        diagnostics["candidates"].append(
            {"tau": tau, "n": n_pred_zero, "fzr": fzr, "ucb": ucb}
        )
        if fzr <= max_false_zero_rate:
            best_tau = tau
            diagnostics["selected"] = {"tau": tau, "n": n_pred_zero, "fzr": fzr, "ucb": ucb}
            break

    return best_tau, diagnostics


def apply_gate_to_targets(
    p_nonzero_targets: np.ndarray,
    target_mask: np.ndarray,
    variant_names: list[str],
    tau_per_variant: dict[str, float | None],
    w_gan_per_variant: dict[str, float] | None = None,
    tau_hard_per_variant: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Produce M_target_zero, M_gan, and soft fusion weights from p_nonzero and per-variant parameters.

    Returns:
        M_target_zero: hard-locked zeros
        M_gan: cells available for GAN magnitude (soft fusion)
        w_gan: soft fusion weights for target cells (n_targets,)
    """
    target_bool = target_mask.astype(bool)
    n_rows, n_vars = target_mask.shape

    M_target_zero = np.zeros(target_mask.shape, dtype=np.uint8)
    M_gan = np.zeros(target_mask.shape, dtype=np.uint8)
    w_gan = np.zeros(target_mask.shape, dtype=np.float32)

    target_rows, target_cols = np.where(target_bool)
    p_flat = p_nonzero_targets

    for j, variant in enumerate(variant_names):
        tau = tau_per_variant.get(variant)
        w_gan_var = w_gan_per_variant.get(variant, 1.0) if w_gan_per_variant else 1.0

        variant_sel = target_cols == j
        rows_j = target_rows[variant_sel]
        p_j = p_flat[variant_sel]

        if tau is None:
            # No gate threshold - all target cells go to GAN (soft fusion)
            free_sel = np.ones_like(p_j, dtype=bool)
        else:
            # Hard-lock: lock to zero if p < tau
            hard_zero_sel = p_j < tau
            free_sel = ~hard_zero_sel
            M_target_zero[rows_j[hard_zero_sel], j] = 1

        # Soft fusion weight for remaining target cells
        w_gan[rows_j[free_sel], j] = w_gan_var

        # M_gan = target cells not hard-locked
        M_gan[rows_j[free_sel], j] = 1

    return M_target_zero, M_gan, w_gan


def compute_soft_fusion(
    p_nonzero: np.ndarray,
    target_mask: np.ndarray,
    variant_names: list[str],
    bias_per_variant: dict[str, float],
    zero_lock_cap_fraction: dict[str, float],
    tau_hard_abs: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Zero-Preserving Gated Fusion weights and hard-lock masks (plan S18.3 / S19.2).

    Per target cell the soft weight is the spike-and-slab posterior gate

        w = sigmoid(logit(p) - b_variant)

    with b_variant the Conformal-Risk-Control bias. A cell is hard-locked to zero
    only when BOTH conditions of S18.3 hold: w is below the absolute floor
    tau_hard_abs, and the cell is among the lowest-w cells within the variant's
    zero-lock cap. The cap is a fraction of that variant's target cells, so a
    variant can never be locked in its entirety.

    Args:
        p_nonzero: (n_rows, n_vars) calibrated P(Y > 0); read only on target cells.
        target_mask: (n_rows, n_vars) raw-NaN target cells.
        variant_names: column order of p_nonzero / target_mask.
        bias_per_variant: CRC bias b per variant.
        zero_lock_cap_fraction: max fraction of a variant's target cells to lock.
        tau_hard_abs: absolute weight floor below which locking is permitted.

    Returns:
        w_gan: (n_rows, n_vars) float32, soft weight on target cells, 0 elsewhere
               and 0 on hard-locked cells.
        M_hard_lock: (n_rows, n_vars) uint8, target cells locked to zero.
        M_soft_fusion: (n_rows, n_vars) uint8, target cells handed to the GAN.
        diagnostics: per-variant tau/cap/counts.
    """
    target_bool = target_mask.astype(bool)

    w_gan = np.zeros(target_mask.shape, dtype=np.float32)
    M_hard_lock = np.zeros(target_mask.shape, dtype=np.uint8)
    M_soft_fusion = np.zeros(target_mask.shape, dtype=np.uint8)
    diagnostics: dict[str, dict] = {}

    for j, variant in enumerate(variant_names):
        target_var = target_bool[:, j]
        n_target = int(target_var.sum())
        if n_target == 0:
            diagnostics[variant] = {"n_target": 0, "status": "no_target_cells"}
            continue

        b = float(bias_per_variant.get(variant, 0.0))
        p_j = np.clip(p_nonzero[target_var, j], 1e-12, 1 - 1e-12)
        w_j = expit(logit(p_j) - b).astype(np.float32)

        cap = float(np.clip(zero_lock_cap_fraction.get(variant, 0.0), 0.0, 1.0))
        # Quantile of w below which locking stays inside the cap. cap == 0 leaves
        # tau at the minimum, so `w < tau` selects nothing.
        tau_cap = float(np.quantile(w_j, cap)) if cap > 0.0 else float(w_j.min())
        tau_eff = min(tau_hard_abs, tau_cap)

        hard_sel = w_j < tau_eff
        soft_sel = ~hard_sel

        rows_j = np.where(target_var)[0]
        w_col = np.zeros(len(rows_j), dtype=np.float32)
        w_col[soft_sel] = w_j[soft_sel]
        w_gan[rows_j, j] = w_col
        M_hard_lock[rows_j[hard_sel], j] = 1
        M_soft_fusion[rows_j[soft_sel], j] = 1

        diagnostics[variant] = {
            "n_target": n_target,
            "b_crc": b,
            "cap_fraction": cap,
            "tau_cap": tau_cap,
            "tau_hard_abs": tau_hard_abs,
            "tau_effective": tau_eff,
            "n_hard_locked": int(hard_sel.sum()),
            "n_soft_fusion": int(soft_sel.sum()),
            "hard_lock_fraction": float(hard_sel.sum() / n_target),
            "w_mean_soft": float(w_j[soft_sel].mean()) if soft_sel.any() else 0.0,
        }

    return w_gan, M_hard_lock, M_soft_fusion, diagnostics