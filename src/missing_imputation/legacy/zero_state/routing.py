from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass
class RoutingConfig:
    high_support_n_positive: int = 500
    medium_support_n_positive: int = 200
    min_locations_with_positive: int = 10
    min_events_per_param: int = 10

    @classmethod
    def from_zinb_config(cls, config) -> "RoutingConfig":
        routing = getattr(config, "routing", None)
        if routing is not None:
            return cls(
                high_support_n_positive=routing.get("high_support_n_positive", 500),
                medium_support_n_positive=routing.get("medium_support_n_positive", 200),
                min_locations_with_positive=routing.get("min_locations_with_positive", 10),
                min_events_per_param=routing.get("min_events_per_param", 10),
            )
        return cls()


@dataclass
class VariantSupport:
    variant_name: str
    n_observed: int
    n_zero: int
    n_positive: int
    positive_rate: float
    n_locations_with_positive: int
    events_per_parameter: float
    missing_rate: float
    tier: Literal["high", "medium", "sparse", "abstain"] = "sparse"

    def to_dict(self) -> dict:
        return {
            "variant_name": self.variant_name,
            "n_observed": self.n_observed,
            "n_zero": self.n_zero,
            "n_positive": self.n_positive,
            "positive_rate": self.positive_rate,
            "n_locations_with_positive": self.n_locations_with_positive,
            "events_per_parameter": self.events_per_parameter,
            "missing_rate": self.missing_rate,
            "tier": self.tier,
        }


@dataclass
class ModelRouting:
    variant_name: str
    model_tier: Literal["reduced_per_variant", "pooled_sparse", "abstain"]
    reason: str
    quality_gate: Literal["PASS", "FAIL"] = "PASS"
    gate_metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "variant_name": self.variant_name,
            "model_tier": self.model_tier,
            "reason": self.reason,
            "quality_gate": self.quality_gate,
            "gate_metrics": self.gate_metrics,
        }


def compute_variant_support(
    variant_name: str,
    counts: np.ndarray,
    total_seq: np.ndarray,
    M_observed: np.ndarray,
    panel_shape: tuple[int, int],
    n_predictors: int,
) -> VariantSupport:
    counts = np.asarray(counts)
    M_observed = np.asarray(M_observed)

    n_observed = int(len(counts))
    n_positive = int(np.sum(counts > 0))
    n_zero = n_observed - n_positive
    positive_rate = n_positive / n_observed if n_observed > 0 else 0.0

    locations_with_positive: set[int] = set()
    if M_observed.ndim == 2:
        locs_pos, times_pos = np.where(M_observed == 1)
        if len(counts) == len(locs_pos):
            positive_mask_flat = counts > 0
            locations_with_positive = set(locs_pos[positive_mask_flat].tolist())

    n_locations_with_positive = int(len(locations_with_positive))
    events_per_parameter = n_positive / max(n_predictors, 1)

    n_total_cells = panel_shape[0] * panel_shape[1] if len(panel_shape) >= 2 else 0
    missing_rate = 1.0 - (n_observed / n_total_cells) if n_total_cells > 0 else 0.0

    return VariantSupport(
        variant_name=variant_name,
        n_observed=n_observed,
        n_zero=n_zero,
        n_positive=n_positive,
        positive_rate=positive_rate,
        n_locations_with_positive=n_locations_with_positive,
        events_per_parameter=events_per_parameter,
        missing_rate=missing_rate,
    )


def classify_support_tier(support: VariantSupport, routing_config: RoutingConfig) -> str:
    if support.n_positive == 0:
        return "abstain"
    if (
        support.n_positive > routing_config.high_support_n_positive
        and support.n_locations_with_positive >= routing_config.min_locations_with_positive
        and support.events_per_parameter >= routing_config.min_events_per_param
    ):
        return "high"
    if support.n_positive >= routing_config.medium_support_n_positive:
        return "medium"
    return "sparse"


def route_variant(support: VariantSupport, routing_config: RoutingConfig) -> ModelRouting:
    tier = classify_support_tier(support, routing_config)
    support.tier = tier

    if tier == "abstain":
        return ModelRouting(
            variant_name=support.variant_name,
            model_tier="abstain",
            reason=f"n_positive={support.n_positive} is zero; cannot fit any model",
        )

    if tier == "high":
        return ModelRouting(
            variant_name=support.variant_name,
            model_tier="reduced_per_variant",
            reason=(
                f"n_positive={support.n_positive} > {routing_config.high_support_n_positive}, "
                f"locations_with_positive={support.n_locations_with_positive}, "
                f"events/param={support.events_per_parameter:.1f}"
            ),
        )

    if tier == "medium":
        return ModelRouting(
            variant_name=support.variant_name,
            model_tier="reduced_per_variant",
            reason=(
                f"n_positive={support.n_positive} in "
                f"[{routing_config.medium_support_n_positive}, "
                f"{routing_config.high_support_n_positive}]; try reduced, fallback to pooled"
            ),
        )

    return ModelRouting(
        variant_name=support.variant_name,
        model_tier="pooled_sparse",
        reason=(
            f"n_positive={support.n_positive} < {routing_config.medium_support_n_positive} "
            f"or low events/param={support.events_per_parameter:.1f}"
        ),
    )


class QualityGate:
    def __init__(
        self,
        min_nonzero_recall: float = 0.95,
        collapse_max_fraction: float = 0.01,
    ):
        self.min_nonzero_recall = min_nonzero_recall
        self.collapse_max_fraction = collapse_max_fraction

    def check(
        self,
        fit_result,
        oof_metrics: dict,
        n_obs: int,
        n_predictors: int,
    ) -> tuple[bool, dict]:
        checks = {}

        converged = getattr(fit_result, "converged", False)
        params_finite = True
        if fit_result is not None:
            for attr in ("params_zi", "params_nb"):
                p = getattr(fit_result, attr, None)
                if p is not None and not np.all(np.isfinite(p)):
                    params_finite = False
            alpha = getattr(fit_result, "alpha", None)
            if alpha is not None and (not np.isfinite(alpha) or alpha <= 0):
                params_finite = False
        checks["converged_finite"] = bool(converged and params_finite)

        checks["rank_ok"] = bool(oof_metrics.get("rank_ok", True))
        checks["fold_coverage_ok"] = bool(oof_metrics.get("fold_coverage_ok", True))

        baseline_nll = oof_metrics.get("baseline_nll", np.inf)
        model_nll = oof_metrics.get("nll", np.inf)
        baseline_brier = oof_metrics.get("baseline_brier", np.inf)
        model_brier = oof_metrics.get("brier_score", np.inf)
        checks["nll_better_than_baseline"] = bool(
            np.isfinite(model_nll) and np.isfinite(model_brier)
            and model_nll <= baseline_nll + 1e-8
            and model_brier <= baseline_brier + 1e-8
        )

        recall = oof_metrics.get("recall", 0.0)
        checks["nonzero_recall"] = bool(recall >= self.min_nonzero_recall)

        frac_zero = oof_metrics.get("frac_predicted_zero", 0.0)
        frac_nonzero = oof_metrics.get("frac_predicted_nonzero", 0.0)
        checks["no_collapse"] = bool(
            frac_zero <= 1.0 - self.collapse_max_fraction
            and frac_nonzero >= self.collapse_max_fraction
        )

        checks["schema_match"] = bool(oof_metrics.get("schema_match", True))

        passed = all(checks.values())
        return passed, checks
