"""Baseline selection following Section 6.3 rules — MSE primary, JSD tie-breaker."""

from __future__ import annotations

import numpy as np


SIMPLICITY_ORDER = ["jsd_knn", "jsd_alpha_knn", "adaptive_jsd_alpha_knn"]


def select_baseline(
    results: dict[str, dict],
    tolerance_std_multiplier: float = 2.0,
) -> tuple[str, dict]:
    """Select the best baseline following Section 6.3 rules.

    Rules:
    1. Discard methods violating invariants.
    2. Rank by mean MSE on proportions (primary metric).
    3. If MSE diff within uncertainty, prefer simpler method.
    4. Never select adaptive if worse on time-block MSE.
    5. Report mean, std, seed list, runtime.

    Args:
        results: Dict mapping method -> metrics.
        tolerance_std_multiplier: Multiplier for std-based tie-breaking.

    Returns:
        (selected_method, selection_details).
    """
    valid_methods = {
        name: metrics
        for name, metrics in results.items()
        if metrics.get("invariants_ok", True)
    }

    if not valid_methods:
        raise ValueError("All methods violated invariants")

    # Primary: rank by mean MSE (lower is better)
    ranked = sorted(
        valid_methods.items(),
        key=lambda x: x[1].get("mse_mean", np.inf),
    )

    if len(ranked) == 1:
        return ranked[0][0], {"reason": "only_valid_method"}

    best_name, best_metrics = ranked[0]
    best_mean = best_metrics["mse_mean"]
    best_std = best_metrics.get("mse_std", 0)

    for name, metrics in ranked[1:]:
        diff = metrics["mse_mean"] - best_mean
        threshold = tolerance_std_multiplier * max(best_std, metrics.get("mse_std", 0))

        if diff <= threshold:
            if SIMPLICITY_ORDER.index(name) < SIMPLICITY_ORDER.index(best_name):
                best_name = name
                best_metrics = metrics
                best_mean = metrics["mse_mean"]
                best_std = metrics.get("mse_std", 0)

    # Adaptive never selected if worse on time-block MSE
    if best_name == "adaptive_jsd_alpha_knn":
        time_block_mse = best_metrics.get("time_block_mse", best_mean)
        simple_methods = [
            (n, m) for n, m in ranked
            if n != "adaptive_jsd_alpha_knn"
        ]
        if simple_methods:
            simple_best_mse = min(m.get("time_block_mse", m["mse_mean"]) for _, m in simple_methods)
            if time_block_mse > simple_best_mse:
                best_name = simple_methods[0][0]
                best_metrics = simple_methods[0][1]

    return best_name, {
        "reason": "best_mse_with_simplicity_tiebreak",
        "selected_mean_mse": best_metrics.get("mse_mean"),
        "selected_std_mse": best_metrics.get("mse_std"),
        "all_results": {n: m.get("mse_mean") for n, m in results.items()},
    }