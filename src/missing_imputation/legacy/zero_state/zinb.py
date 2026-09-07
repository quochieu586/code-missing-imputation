from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit
from scipy.stats import nbinom, poisson

from .encoder import DesignEncoder, ZINBEncoderConfig


class ModelType(Enum):
    ZINB = "zinb"
    ABSTAIN = "abstain"


@dataclass
class ZINBConfig:
    time_spline_df: int
    time_spline_type: str
    use_lag_lead: bool
    use_availability_indicators: bool
    center_spline: bool
    routing: dict
    l1_penalty_grid: list[float]
    tune_on_validation: bool
    max_starts: int
    max_iter: int
    convergence_tol: float
    artificial_mask_recipes: list[str]
    n_artificial_folds: int
    calibration_metrics: list[str]
    default_gate: str
    threshold_grid: list[float]
    min_nonzero_recall: float
    threshold_objective: str
    fit_seed: int
    sample_seeds: list[int]
    feasibility_mode: str
    save_posterior: bool
    save_masks: bool
    save_oof_predictions: bool
    save_design_schema: bool
    save_manifest: bool
    min_fold_coverage: float

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ZINBConfig":
        import yaml
        with Path(path).open() as f:
            cfg = yaml.safe_load(f)
        zs = cfg.get("zero_state", {}).get("zinb", {})
        return cls(
            time_spline_df=zs.get("time_spline_df", 4),
            time_spline_type=zs.get("time_spline_type", "natural"),
            use_lag_lead=zs.get("use_lag_lead", True),
            use_availability_indicators=zs.get("use_availability_indicators", True),
            center_spline=zs.get("center_spline", True),
            routing=zs.get("routing", {}),
            l1_penalty_grid=zs.get("l1_penalty_grid", [0.0, 0.01, 0.1, 1.0]),
            tune_on_validation=zs.get("tune_on_validation", True),
            max_starts=zs.get("max_starts", 5),
            max_iter=zs.get("max_iter", 500),
            convergence_tol=zs.get("convergence_tol", 1e-6),
            artificial_mask_recipes=zs.get("artificial_mask_recipes", ["random-cell", "empirical-pattern", "time-block"]),
            n_artificial_folds=zs.get("n_artificial_folds", 5),
            calibration_metrics=zs.get("calibration_metrics", ["nll", "brier", "calibration_curve", "precision", "recall", "f1", "ece"]),
            default_gate=zs.get("default_gate", "calibrated_threshold"),
            threshold_grid=zs.get("threshold_grid", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
            min_nonzero_recall=zs.get("min_nonzero_recall", 0.95),
            threshold_objective=zs.get("threshold_objective", "precision"),
            fit_seed=zs.get("fit_seed", 42),
            sample_seeds=zs.get("sample_seeds", [42, 123, 456, 789, 999]),
            feasibility_mode=zs.get("feasibility_mode", "priority_nonzero_uncertain_on_override"),
            save_posterior=zs.get("save_posterior", True),
            save_masks=zs.get("save_masks", True),
            save_oof_predictions=zs.get("save_oof_predictions", True),
            save_design_schema=zs.get("save_design_schema", True),
            save_manifest=zs.get("save_manifest", True),
            min_fold_coverage=zs.get("min_fold_coverage", 0.8),
        )

    def encoder_config(self) -> ZINBEncoderConfig:
        return ZINBEncoderConfig(
            time_spline_df=self.time_spline_df,
            time_spline_type=self.time_spline_type,
            use_lag_lead=self.use_lag_lead,
            use_availability_indicators=self.use_availability_indicators,
            center_spline=self.center_spline,
        )


@dataclass
class ZINBFitResult:
    variant_name: str
    model_type: ModelType
    params_zi: np.ndarray | None
    params_nb: np.ndarray | None
    alpha: float | None
    zi_predictor_names: list[str]
    nb_predictor_names: list[str]
    converged: bool
    n_observations: int
    n_iterations: int | None
    log_likelihood: float | None
    l1_penalty_used: float
    design_schema: dict = field(default_factory=dict)
    fallback_reason: str | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class PosteriorProbs:
    p_structural: np.ndarray
    p_sampling: np.ndarray
    p_nonzero: np.ndarray
    target_indices: np.ndarray


def fit_zinb_regularized(
    y: np.ndarray,
    X_zi: np.ndarray,
    X_nb: np.ndarray,
    offset: np.ndarray,
    alpha: float,
    max_iter: int,
    tol: float,
    start_params: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, bool, int | None, float]:
    try:
        model = sm.ZeroInflatedNegativeBinomialP(
            endog=y,
            exog=X_nb,
            exog_infl=X_zi,
            offset=offset,
            inflation="logit",
        )

        fit_kwargs = dict(
            alpha=alpha,
            L1_wt=1.0,
            maxiter=max_iter,
            tol=tol,
            disp=False,
        )
        if start_params is not None:
            fit_kwargs["start_params"] = start_params

        result = model.fit_regularized(**fit_kwargs)

        n_nb = X_nb.shape[1]
        n_zi = X_zi.shape[1]
        params_nb = result.params[:n_nb]
        params_zi = result.params[n_nb:n_nb + n_zi]
        alpha_nb = result.params[-1]

        n_iter = getattr(result, "iterations", None)
        if n_iter is None:
            retvals = getattr(result, "mle_retvals", None)
            if retvals is not None and isinstance(retvals, dict):
                n_iter = retvals.get("iterations", None)

        return params_zi, params_nb, alpha_nb, result.converged, n_iter, result.llf

    except Exception as e:
        raise RuntimeError(f"ZINB fit failed: {e}")


def predict_zinb_probs(
    X_zi: np.ndarray,
    X_nb: np.ndarray,
    offset: np.ndarray,
    params_zi: np.ndarray,
    params_nb: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    zi_linear = X_zi @ params_zi
    p_structural = expit(zi_linear)

    nb_linear = X_nb @ params_nb + offset
    mu = np.exp(np.clip(nb_linear, -30, 30))

    n_param = 1.0 / max(alpha, 1e-8)
    p_param = n_param / (n_param + mu)
    p_sampling_zero_marginal = nbinom.pmf(0, n_param, p_param)

    p_not_structural = 1 - p_structural
    p_sampling = p_not_structural * p_sampling_zero_marginal
    p_nonzero = p_not_structural * (1 - p_sampling_zero_marginal)

    return p_structural, p_sampling, p_nonzero


class ReducedZINBModel:
    """Reduced per-variant ZINB: no location FE, ~8-12 predictors."""

    def __init__(self, variant_name: str, config: ZINBConfig):
        self.variant_name = variant_name
        self.config = config
        self.encoder = DesignEncoder(config.encoder_config(), variant_name)
        self.fit_result: ZINBFitResult | None = None

    def fit(
        self,
        counts: np.ndarray,
        total_seq: np.ndarray,
        day_index: np.ndarray,
        lag_counts: np.ndarray,
        lag_avail: np.ndarray,
        lead_counts: np.ndarray,
        lead_avail: np.ndarray,
        validation_mask: np.ndarray | None = None,
    ) -> ZINBFitResult:
        self.encoder.fit(total_seq, day_index, lag_counts, lag_avail, lead_counts, lead_avail)
        X_zi, X_nb, offset = self.encoder.transform(
            total_seq, day_index, lag_counts, lag_avail, lead_counts, lead_avail
        )

        if not self.encoder.check_rank(X_zi, "X_zi") or not self.encoder.check_rank(X_nb, "X_nb"):
            self.fit_result = ZINBFitResult(
                variant_name=self.variant_name,
                model_type=ModelType.ABSTAIN,
                params_zi=None,
                params_nb=None,
                alpha=None,
                zi_predictor_names=self.encoder.schema["zi_predictor_names"],
                nb_predictor_names=self.encoder.schema["nb_predictor_names"],
                converged=False,
                n_observations=len(counts),
                n_iterations=None,
                log_likelihood=None,
                l1_penalty_used=self.config.l1_penalty_grid[0],
                design_schema=self.encoder.get_schema(),
                fallback_reason="Design matrix rank-deficient at fit; ABSTAIN",
            )
            return self.fit_result

        best_alpha = self.config.l1_penalty_grid[0]
        if self.config.tune_on_validation and validation_mask is not None:
            best_alpha = self._tune_l1_penalty(
                np.asarray(counts, dtype=np.float64), X_zi, X_nb, offset, validation_mask
            )

        best_result = None
        best_llf = -np.inf
        rng = np.random.default_rng(self.config.fit_seed)

        for start_attempt in range(self.config.max_starts):
            try:
                start_params = None
                if start_attempt > 0:
                    n_params = X_zi.shape[1] + X_nb.shape[1] + 1
                    start_params = rng.normal(0, 0.1, n_params)

                params_zi, params_nb, alpha_nb, converged, n_iter, llf = fit_zinb_regularized(
                    y=np.asarray(counts, dtype=np.float64),
                    X_zi=X_zi,
                    X_nb=X_nb,
                    offset=offset,
                    alpha=best_alpha,
                    max_iter=self.config.max_iter,
                    tol=self.config.convergence_tol,
                    start_params=start_params,
                )

                if converged and llf > best_llf:
                    best_llf = llf
                    best_result = ZINBFitResult(
                        variant_name=self.variant_name,
                        model_type=ModelType.ZINB,
                        params_zi=params_zi,
                        params_nb=params_nb,
                        alpha=alpha_nb,
                        zi_predictor_names=self.encoder.schema["zi_predictor_names"],
                        nb_predictor_names=self.encoder.schema["nb_predictor_names"],
                        converged=True,
                        n_observations=len(counts),
                        n_iterations=n_iter,
                        log_likelihood=llf,
                        l1_penalty_used=best_alpha,
                        design_schema=self.encoder.get_schema(),
                    )
            except Exception:
                continue

        if best_result is None:
            best_result = ZINBFitResult(
                variant_name=self.variant_name,
                model_type=ModelType.ABSTAIN,
                params_zi=None,
                params_nb=None,
                alpha=None,
                zi_predictor_names=self.encoder.schema.get("zi_predictor_names", []),
                nb_predictor_names=self.encoder.schema.get("nb_predictor_names", []),
                converged=False,
                n_observations=len(counts),
                n_iterations=None,
                log_likelihood=None,
                l1_penalty_used=best_alpha,
                design_schema=self.encoder.get_schema(),
                fallback_reason="ZINB failed to converge with all starting values; ABSTAIN",
            )

        self.fit_result = best_result
        return best_result

    def predict_proba(
        self,
        total_seq: np.ndarray,
        day_index: np.ndarray,
        lag_counts: np.ndarray,
        lag_avail: np.ndarray,
        lead_counts: np.ndarray,
        lead_avail: np.ndarray,
    ) -> PosteriorProbs:
        if self.fit_result is None or self.fit_result.model_type != ModelType.ZINB:
            raise RuntimeError("Model not fitted or ABSTAIN; cannot predict")

        X_zi, X_nb, offset = self.encoder.transform(
            total_seq, day_index, lag_counts, lag_avail, lead_counts, lead_avail
        )

        p_struct, p_samp, p_nonzero = predict_zinb_probs(
            X_zi, X_nb, offset,
            self.fit_result.params_zi,
            self.fit_result.params_nb,
            self.fit_result.alpha,
        )

        total = p_struct + p_samp + p_nonzero
        p_struct = p_struct / total
        p_samp = p_samp / total
        p_nonzero = p_nonzero / total

        return PosteriorProbs(
            p_structural=p_struct,
            p_sampling=p_samp,
            p_nonzero=p_nonzero,
            target_indices=np.arange(len(total_seq)),
        )

    def _tune_l1_penalty(
        self,
        counts: np.ndarray,
        X_zi: np.ndarray,
        X_nb: np.ndarray,
        offset: np.ndarray,
        validation_mask: np.ndarray,
    ) -> float:
        train_mask = ~validation_mask
        if train_mask.sum() == 0 or validation_mask.sum() == 0:
            return self.config.l1_penalty_grid[0]

        best_alpha = self.config.l1_penalty_grid[0]
        best_score = np.inf

        for alpha in self.config.l1_penalty_grid:
            try:
                params_zi, params_nb, alpha_nb, converged, _, _ = fit_zinb_regularized(
                    y=counts[train_mask],
                    X_zi=X_zi[train_mask],
                    X_nb=X_nb[train_mask],
                    offset=offset[train_mask],
                    alpha=alpha,
                    max_iter=self.config.max_iter,
                    tol=self.config.convergence_tol,
                )

                if not converged:
                    continue

                p_struct, p_samp, p_nonzero = predict_zinb_probs(
                    X_zi[validation_mask],
                    X_nb[validation_mask],
                    offset[validation_mask],
                    params_zi, params_nb, alpha_nb,
                )
                y_val = counts[validation_mask]
                y_binary = (y_val > 0).astype(np.float64)
                p_zero = np.clip(p_struct + p_samp, 1e-12, 1 - 1e-12)
                p_nz = np.clip(p_nonzero, 1e-12, 1 - 1e-12)
                score = -np.mean(y_binary * np.log(p_nz) + (1 - y_binary) * np.log(p_zero))

                if score < best_score:
                    best_score = score
                    best_alpha = alpha

            except Exception:
                continue

        return best_alpha


class PooledSparseZINBModel:
    """Pooled ZINB for sparse variants using K-1 treatment coding (D2 fix).

    Uses K-1 variant dummy columns with the first variant as reference level,
    so [intercept + K-1 dummies] is full column rank.
    """

    def __init__(self, variant_names: list[str], config: ZINBConfig):
        self.variant_names = list(variant_names)
        self.config = config
        self.encoder = DesignEncoder(config.encoder_config(), "pooled_sparse")
        self.fit_result: ZINBFitResult | None = None
        self._n_variants = len(self.variant_names)
        self._reference_variant = self.variant_names[0] if self.variant_names else ""

    def _build_variant_dummies(self, variant_codes: np.ndarray, n_obs: int) -> np.ndarray:
        n_effect_cols = max(self._n_variants - 1, 0)
        dummies = np.zeros((n_obs, n_effect_cols), dtype=np.float64)
        for k in range(1, self._n_variants):
            dummies[variant_codes == k, k - 1] = 1.0
        return dummies

    def fit(
        self,
        stacked_counts: np.ndarray,
        stacked_total_seq: np.ndarray,
        stacked_day_index: np.ndarray,
        stacked_lag_counts: np.ndarray,
        stacked_lag_avail: np.ndarray,
        stacked_lead_counts: np.ndarray,
        stacked_lead_avail: np.ndarray,
        variant_codes: np.ndarray,
        validation_mask: np.ndarray | None = None,
    ) -> ZINBFitResult:
        self.encoder.fit(
            stacked_total_seq, stacked_day_index, stacked_lag_counts,
            stacked_lag_avail, stacked_lead_counts, stacked_lead_avail,
        )
        X_zi_base, X_nb_base, offset = self.encoder.transform(
            stacked_total_seq, stacked_day_index, stacked_lag_counts,
            stacked_lag_avail, stacked_lead_counts, stacked_lead_avail,
        )

        variant_codes = np.asarray(variant_codes, dtype=np.int64)
        n_obs = len(stacked_counts)

        variant_dummies = self._build_variant_dummies(variant_codes, n_obs)

        X_zi = np.hstack([X_zi_base, variant_dummies])
        X_nb = np.hstack([X_nb_base, variant_dummies])

        n_effect_cols = max(self._n_variants - 1, 0)
        effect_names = [f"variant_effect_{self.variant_names[k]}" for k in range(1, self._n_variants)]
        zi_names = self.encoder.schema["zi_predictor_names"] + effect_names
        nb_names = self.encoder.schema["nb_predictor_names"] + effect_names

        if np.linalg.matrix_rank(X_zi) != X_zi.shape[1] or np.linalg.matrix_rank(X_nb) != X_nb.shape[1]:
            self.fit_result = ZINBFitResult(
                variant_name="pooled_sparse",
                model_type=ModelType.ABSTAIN,
                params_zi=None,
                params_nb=None,
                alpha=None,
                zi_predictor_names=zi_names,
                nb_predictor_names=nb_names,
                converged=False,
                n_observations=n_obs,
                n_iterations=None,
                log_likelihood=None,
                l1_penalty_used=self.config.l1_penalty_grid[0],
                design_schema=self.encoder.get_schema(),
                extra={"variant_names": self.variant_names, "reference_variant": self._reference_variant},
                fallback_reason="Pooled design matrix rank-deficient; ABSTAIN",
            )
            return self.fit_result

        best_alpha = self.config.l1_penalty_grid[0]
        if self.config.tune_on_validation and validation_mask is not None:
            best_alpha = self._tune_l1_penalty(
                np.asarray(stacked_counts, dtype=np.float64), X_zi, X_nb, offset, validation_mask
            )

        best_result = None
        best_llf = -np.inf
        rng = np.random.default_rng(self.config.fit_seed)

        for start_attempt in range(self.config.max_starts):
            try:
                start_params = None
                if start_attempt > 0:
                    n_params = X_zi.shape[1] + X_nb.shape[1] + 1
                    start_params = rng.normal(0, 0.1, n_params)

                params_zi, params_nb, alpha_nb, converged, n_iter, llf = fit_zinb_regularized(
                    y=np.asarray(stacked_counts, dtype=np.float64),
                    X_zi=X_zi,
                    X_nb=X_nb,
                    offset=offset,
                    alpha=best_alpha,
                    max_iter=self.config.max_iter,
                    tol=self.config.convergence_tol,
                    start_params=start_params,
                )

                if converged and llf > best_llf:
                    best_llf = llf
                    schema = self.encoder.get_schema()
                    schema["zi_predictor_names"] = zi_names
                    schema["nb_predictor_names"] = nb_names
                    schema["reference_variant"] = self._reference_variant
                    schema["variant_effect_names"] = effect_names
                    best_result = ZINBFitResult(
                        variant_name="pooled_sparse",
                        model_type=ModelType.ZINB,
                        params_zi=params_zi,
                        params_nb=params_nb,
                        alpha=alpha_nb,
                        zi_predictor_names=zi_names,
                        nb_predictor_names=nb_names,
                        converged=True,
                        n_observations=n_obs,
                        n_iterations=n_iter,
                        log_likelihood=llf,
                        l1_penalty_used=best_alpha,
                        design_schema=schema,
                        extra={"variant_names": self.variant_names, "reference_variant": self._reference_variant},
                    )
            except Exception:
                continue

        if best_result is None:
            best_result = ZINBFitResult(
                variant_name="pooled_sparse",
                model_type=ModelType.ABSTAIN,
                params_zi=None,
                params_nb=None,
                alpha=None,
                zi_predictor_names=zi_names,
                nb_predictor_names=nb_names,
                converged=False,
                n_observations=n_obs,
                n_iterations=None,
                log_likelihood=None,
                l1_penalty_used=best_alpha,
                design_schema=self.encoder.get_schema(),
                extra={"variant_names": self.variant_names, "reference_variant": self._reference_variant},
                fallback_reason="Pooled ZINB failed to converge; ABSTAIN",
            )

        self.fit_result = best_result
        return best_result

    def predict_proba_variant(
        self,
        variant_code: int,
        total_seq: np.ndarray,
        day_index: np.ndarray,
        lag_counts: np.ndarray,
        lag_avail: np.ndarray,
        lead_counts: np.ndarray,
        lead_avail: np.ndarray,
    ) -> PosteriorProbs:
        if self.fit_result is None or self.fit_result.model_type != ModelType.ZINB:
            raise RuntimeError("Model not fitted or ABSTAIN; cannot predict")

        n = len(total_seq)
        X_zi_base, X_nb_base, offset = self.encoder.transform(
            total_seq, day_index, lag_counts, lag_avail, lead_counts, lead_avail
        )

        variant_codes = np.full(n, variant_code, dtype=np.int64)
        variant_dummies = self._build_variant_dummies(variant_codes, n)

        X_zi = np.hstack([X_zi_base, variant_dummies])
        X_nb = np.hstack([X_nb_base, variant_dummies])

        p_struct, p_samp, p_nonzero = predict_zinb_probs(
            X_zi, X_nb, offset,
            self.fit_result.params_zi,
            self.fit_result.params_nb,
            self.fit_result.alpha,
        )

        total = p_struct + p_samp + p_nonzero
        p_struct = p_struct / total
        p_samp = p_samp / total
        p_nonzero = p_nonzero / total

        return PosteriorProbs(
            p_structural=p_struct,
            p_sampling=p_samp,
            p_nonzero=p_nonzero,
            target_indices=np.arange(n),
        )

    def _tune_l1_penalty(
        self,
        counts: np.ndarray,
        X_zi: np.ndarray,
        X_nb: np.ndarray,
        offset: np.ndarray,
        validation_mask: np.ndarray,
    ) -> float:
        train_mask = ~validation_mask
        if train_mask.sum() == 0 or validation_mask.sum() == 0:
            return self.config.l1_penalty_grid[0]

        best_alpha = self.config.l1_penalty_grid[0]
        best_score = np.inf

        for alpha in self.config.l1_penalty_grid:
            try:
                params_zi, params_nb, alpha_nb, converged, _, _ = fit_zinb_regularized(
                    y=counts[train_mask],
                    X_zi=X_zi[train_mask],
                    X_nb=X_nb[train_mask],
                    offset=offset[train_mask],
                    alpha=alpha,
                    max_iter=self.config.max_iter,
                    tol=self.config.convergence_tol,
                )

                if not converged:
                    continue

                p_struct, p_samp, p_nonzero = predict_zinb_probs(
                    X_zi[validation_mask],
                    X_nb[validation_mask],
                    offset[validation_mask],
                    params_zi, params_nb, alpha_nb,
                )
                y_val = counts[validation_mask]
                y_binary = (y_val > 0).astype(np.float64)
                p_zero = np.clip(p_struct + p_samp, 1e-12, 1 - 1e-12)
                p_nz = np.clip(p_nonzero, 1e-12, 1 - 1e-12)
                score = -np.mean(y_binary * np.log(p_nz) + (1 - y_binary) * np.log(p_zero))

                if score < best_score:
                    best_score = score
                    best_alpha = alpha

            except Exception:
                continue

        return best_alpha