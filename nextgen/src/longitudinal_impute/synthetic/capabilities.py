"""Fail-closed scope approved by the user; no substitutions of mechanisms."""


class NotReadyError(NotImplementedError):
    code = "NOT-READY"


_STATUS = {
    "synthetic_generator": ("IMPLEMENTED", "Domain-neutral benchmark assumptions; not a source-paper generative model."),
    "table1_design": ("IMPLEMENTED", "Five source designs; not post-imputation bias/coverage tables."),
    "mcar_declared_patterns": ("IMPLEMENTED", "Original ampute; probability is per candidate case."),
    "mar_observed_covariates": ("IMPLEMENTED", "N and entity_trait_01 only; zero target weights."),
    "fixed_wave": ("IMPLEMENTED", "One predeclared measurement wave, all target components."),
    "source_mnar": ("REFERENCE-ONLY", "Available only inside Table 1 reproduction."),
    "latent_p_mnar": ("NOT-READY", "No approved latent-p to sampled-count masking extension."),
    "random_temporal_blocks": ("NOT-READY", "No random start/length/component block algorithm approved."),
    "whole_component_within_entity": ("NOT-READY", "No whole-trajectory component generator approved."),
    "beta_strength_grid": ("NOT-READY", "Source rescales scores; no added logistic slope."),
    "full_cell_rate_grid": ("NOT-READY", "No independent target-cell-rate calibration approved."),
    "exact_realized_rate": ("NOT-READY", "No post-draw editing to enforce quotas."),
    "paper_inference_tables": ("NOT-READY", "PMM bias/coverage reproduction is outside Table 1 design scope."),
}


def capabilities():
    return {key: {"status": status, "reason": reason}
            for key, (status, reason) in _STATUS.items()}


def require_capability(name):
    status, reason = _STATUS.get(name, ("NOT-READY", "Unknown capability; no fallback permitted."))
    if status != "IMPLEMENTED":
        raise NotReadyError(f"NOT-READY: {name}: {reason}")
