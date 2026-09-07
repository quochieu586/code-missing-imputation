"""Zero-state branch (Nhánh B) for missing imputation.

Implements ZINB-based zero-state initialization per Section 4.4.
"""

from .zinb import (
    ZINBConfig,
    ZINBFitResult,
    PosteriorProbs,
    ModelType,
    ReducedZINBModel,
    PooledSparseZINBModel,
    predict_zinb_probs,
)
from .encoder import (
    DesignEncoder,
    ZINBEncoderConfig,
)
from .routing import (
    RoutingConfig,
    VariantSupport,
    ModelRouting,
    compute_variant_support,
    route_variant,
    QualityGate,
)
from .posterior import (
    ZeroStatePosterior,
    build_zero_state_posterior,
    compute_global_day_indices,
)
from .sampler import (
    SampledState,
    STATE_STRUCTURAL_ZERO,
    STATE_SAMPLING_ZERO,
    STATE_NONZERO,
    STATE_UNCERTAIN,
    sample_states_three_way,
    apply_feasibility_projection_uncertain,
    run_seeded_sampling_three_way,
    save_sampled_states,
)
from .calibration import (
    CalibrationMetrics,
    generate_artificial_masks,
    compute_calibration_metrics,
    select_recall_constrained_threshold,
    run_oof_calibration,
    compute_baseline_metrics,
    compute_lag_lead_features,
    create_m_artificial_zero,
    create_m_artificial_mag,
)
from .masks import (
    ZeroStateMasks,
    create_target_masks_from_states,
    combine_variant_masks,
    save_combined_masks,
)
from .artifacts import (
    save_variant_support_report,
    save_model_routing_report,
    save_zero_state_metrics,
    save_design_schema,
    save_model_manifest,
    save_oof_predictions,
)

__all__ = [
    # zinb
    "ZINBConfig",
    "ZINBFitResult",
    "PosteriorProbs",
    "ModelType",
    "ReducedZINBModel",
    "PooledSparseZINBModel",
    "predict_zinb_probs",
    # encoder
    "DesignEncoder",
    "ZINBEncoderConfig",
    # routing
    "RoutingConfig",
    "VariantSupport",
    "ModelRouting",
    "compute_variant_support",
    "route_variant",
    "QualityGate",
    # posterior
    "ZeroStatePosterior",
    "build_zero_state_posterior",
    "compute_global_day_indices",
    # sampler
    "SampledState",
    "STATE_STRUCTURAL_ZERO",
    "STATE_SAMPLING_ZERO",
    "STATE_NONZERO",
    "STATE_UNCERTAIN",
    "sample_states_three_way",
    "apply_feasibility_projection_uncertain",
    "run_seeded_sampling_three_way",
    "save_sampled_states",
    # calibration
    "CalibrationMetrics",
    "generate_artificial_masks",
    "compute_calibration_metrics",
    "select_recall_constrained_threshold",
    "run_oof_calibration",
    "compute_baseline_metrics",
    "compute_lag_lead_features",
    "create_m_artificial_zero",
    "create_m_artificial_mag",
    # masks
    "ZeroStateMasks",
    "create_target_masks_from_states",
    "combine_variant_masks",
    "save_combined_masks",
    # artifacts
    "save_variant_support_report",
    "save_model_routing_report",
    "save_zero_state_metrics",
    "save_design_schema",
    "save_model_manifest",
    "save_oof_predictions",
]