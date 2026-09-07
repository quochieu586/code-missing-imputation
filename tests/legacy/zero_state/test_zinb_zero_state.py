"""Unit tests for ZINB zero-state branch (Section 4.4 updated plan)."""

import numpy as np
import pandas as pd
import pytest

from src.missing_imputation.zero_state.encoder import DesignEncoder, ZINBEncoderConfig
from src.missing_imputation.zero_state.routing import (
    RoutingConfig,
    VariantSupport,
    ModelRouting,
    compute_variant_support,
    route_variant,
    QualityGate,
)
from src.missing_imputation.zero_state.zinb import (
    ZINBConfig,
    ZINBFitResult,
    PosteriorProbs,
    ModelType,
    ReducedZINBModel,
    predict_zinb_probs,
)
from src.missing_imputation.zero_state.posterior import (
    ZeroStatePosterior,
)
from src.missing_imputation.zero_state.sampler import (
    SampledState,
    STATE_STRUCTURAL_ZERO,
    STATE_SAMPLING_ZERO,
    STATE_NONZERO,
    STATE_UNCERTAIN,
    sample_states_three_way,
    apply_feasibility_projection_uncertain,
)
from src.missing_imputation.zero_state.calibration import (
    compute_calibration_metrics,
    select_recall_constrained_threshold,
)
from src.missing_imputation.zero_state.masks import (
    ZeroStateMasks,
    create_target_masks_from_states,
    combine_variant_masks,
)


def _make_encoder_config():
    return ZINBEncoderConfig(
        time_spline_df=4,
        time_spline_type="natural",
        use_lag_lead=True,
        use_availability_indicators=True,
        center_spline=True,
    )


class TestDesignEncoder:

    def test_fit_transform_shapes(self):
        n = 100
        cfg = _make_encoder_config()
        enc = DesignEncoder(cfg, "test_variant")
        total_seq = np.random.randint(100, 1000, n).astype(np.float64)
        day_index = np.arange(n, dtype=np.float64)
        lag_counts = np.random.randint(0, 50, n).astype(np.float64)
        lag_avail = (np.random.rand(n) > 0.5).astype(np.float64)
        lead_counts = np.random.randint(0, 50, n).astype(np.float64)
        lead_avail = (np.random.rand(n) > 0.5).astype(np.float64)

        enc.fit(total_seq, day_index, lag_counts, lag_avail, lead_counts, lead_avail)
        X_zi, X_nb, offset = enc.transform(total_seq, day_index, lag_counts, lag_avail, lead_counts, lead_avail)

        assert X_zi.shape[0] == n
        assert X_nb.shape[0] == n
        assert len(offset) == n
        assert X_zi.shape[1] >= 7
        assert X_nb.shape[1] >= 6

    def test_no_location_fe_in_predictors(self):
        n = 50
        cfg = _make_encoder_config()
        enc = DesignEncoder(cfg, "test_variant")
        total_seq = np.full(n, 500.0)
        day_index = np.arange(n, dtype=np.float64)
        lag_counts = np.zeros(n)
        lag_avail = np.zeros(n)
        lead_counts = np.zeros(n)
        lead_avail = np.zeros(n)

        enc.fit(total_seq, day_index, lag_counts, lag_avail, lead_counts, lead_avail)
        X_zi, X_nb, offset = enc.transform(total_seq, day_index, lag_counts, lag_avail, lead_counts, lead_avail)

        zi_names = enc.schema["zi_predictor_names"]
        nb_names = enc.schema["nb_predictor_names"]
        assert not any("loc_" in name for name in zi_names)
        assert not any("loc_" in name for name in nb_names)

    def test_schema_roundtrip(self):
        n = 30
        cfg = _make_encoder_config()
        enc = DesignEncoder(cfg, "test_variant")
        total_seq = np.full(n, 200.0)
        day_index = np.arange(n, dtype=np.float64)
        zeros = np.zeros(n)

        enc.fit(total_seq, day_index, zeros, zeros, zeros, zeros)
        schema = enc.get_schema()

        enc2 = DesignEncoder.from_schema(schema)
        X_zi1, X_nb1, off1 = enc.transform(total_seq, day_index, zeros, zeros, zeros, zeros)
        X_zi2, X_nb2, off2 = enc2.transform(total_seq, day_index, zeros, zeros, zeros, zeros)

        assert np.allclose(X_zi1, X_zi2)
        assert np.allclose(X_nb1, X_nb2)
        assert np.allclose(off1, off2)

    def test_transform_before_fit_raises(self):
        cfg = _make_encoder_config()
        enc = DesignEncoder(cfg, "test_variant")
        n = 5
        with pytest.raises(AssertionError):
            enc.transform(
                np.ones(n), np.ones(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
            )


class TestRouting:

    def test_compute_variant_support(self):
        n_loc, n_time = 5, 20
        M_observed = np.ones((n_loc, n_time), dtype=np.uint8)
        counts = np.random.randint(0, 10, n_loc * n_time)
        total_seq = np.full(n_loc * n_time, 100)

        support = compute_variant_support(
            variant_name="test",
            counts=counts.astype(np.float64),
            total_seq=total_seq.astype(np.float64),
            M_observed=M_observed,
            panel_shape=(n_loc, n_time),
            n_predictors=10,
        )

        assert support.n_observed == n_loc * n_time
        assert support.n_positive == int(np.sum(counts > 0))
        assert support.n_zero == int(np.sum(counts == 0))
        assert 0 <= support.positive_rate <= 1

    def test_route_high_support(self):
        cfg = RoutingConfig(high_support_n_positive=500, medium_support_n_positive=200)
        support = VariantSupport(
            variant_name="test",
            n_observed=1000,
            n_zero=200,
            n_positive=800,
            positive_rate=0.8,
            n_locations_with_positive=50,
            events_per_parameter=80.0,
            missing_rate=0.1,
            tier="high",
        )
        routing = route_variant(support, cfg)
        assert routing.model_tier == "reduced_per_variant"

    def test_route_sparse_support(self):
        cfg = RoutingConfig(high_support_n_positive=500, medium_support_n_positive=200)
        support = VariantSupport(
            variant_name="test",
            n_observed=1000,
            n_zero=950,
            n_positive=50,
            positive_rate=0.05,
            n_locations_with_positive=3,
            events_per_parameter=5.0,
            missing_rate=0.5,
            tier="sparse",
        )
        routing = route_variant(support, cfg)
        assert routing.model_tier == "pooled_sparse"

    def test_quality_gate_pass(self):
        gate = QualityGate(min_nonzero_recall=0.95)
        fit_result = ZINBFitResult(
            variant_name="test",
            model_type=ModelType.ZINB,
            params_zi=np.zeros(5),
            params_nb=np.zeros(5),
            alpha=0.5,
            zi_predictor_names=["a"] * 5,
            nb_predictor_names=["b"] * 5,
            converged=True,
            n_observations=100,
            n_iterations=50,
            log_likelihood=-100.0,
            l1_penalty_used=0.1,
        )
        oof_metrics = {
            "nll": 0.3,
            "brier_score": 0.1,
            "recall": 0.97,
            "baseline_nll": 0.5,
            "baseline_brier": 0.2,
            "frac_predicted_zero": 0.4,
            "frac_predicted_nonzero": 0.6,
            "rank_ok": True,
            "schema_match": True,
        }
        passed, checks = gate.check(fit_result, oof_metrics, 100, 10)
        assert passed
        assert all(checks.values())

    def test_quality_gate_fail_low_recall(self):
        gate = QualityGate(min_nonzero_recall=0.95)
        fit_result = ZINBFitResult(
            variant_name="test",
            model_type=ModelType.ZINB,
            params_zi=np.zeros(5),
            params_nb=np.zeros(5),
            alpha=0.5,
            zi_predictor_names=["a"] * 5,
            nb_predictor_names=["b"] * 5,
            converged=True,
            n_observations=100,
            n_iterations=50,
            log_likelihood=-100.0,
            l1_penalty_used=0.1,
        )
        oof_metrics = {
            "nll": 0.3,
            "brier_score": 0.1,
            "recall": 0.80,
            "baseline_nll": 0.5,
            "baseline_brier": 0.2,
            "frac_predicted_zero": 0.4,
            "frac_predicted_nonzero": 0.6,
            "rank_ok": True,
            "schema_match": True,
        }
        passed, checks = gate.check(fit_result, oof_metrics, 100, 10)
        assert not passed
        assert not checks["nonzero_recall"]


class TestProbabilityPredictions:

    def test_zinb_probs_sum_to_one(self):
        n = 50
        p_zi = 3
        p_nb = 3
        X_zi = np.random.randn(n, p_zi)
        X_nb = np.random.randn(n, p_nb)
        offset = np.log(np.random.uniform(100, 1000, n))
        params_zi = np.random.randn(p_zi) * 0.5
        params_nb = np.random.randn(p_nb) * 0.5
        alpha = 0.5

        p_struct, p_samp, p_nonzero = predict_zinb_probs(
            X_zi, X_nb, offset, params_zi, params_nb, alpha
        )

        total = p_struct + p_samp + p_nonzero
        assert np.allclose(total, 1.0, atol=1e-4)
        assert np.all(p_struct >= -1e-10) and np.all(p_struct <= 1 + 1e-10)
        assert np.all(p_samp >= -1e-10) and np.all(p_samp <= 1 + 1e-10)
        assert np.all(p_nonzero >= -1e-10) and np.all(p_nonzero <= 1 + 1e-10)


class TestThreeStateSampling:

    def test_three_state_gate_nonzero(self):
        n = 100
        post = PosteriorProbs(
            p_structural=np.full(n, 0.05),
            p_sampling=np.full(n, 0.05),
            p_nonzero=np.full(n, 0.9),
            target_indices=np.arange(n),
        )
        states = sample_states_three_way(post, seed=42, threshold_nonzero=0.8, threshold_zero=0.9)
        assert np.all(states == STATE_NONZERO)

    def test_three_state_gate_confident_zero(self):
        n = 100
        post = PosteriorProbs(
            p_structural=np.full(n, 0.8),
            p_sampling=np.full(n, 0.15),
            p_nonzero=np.full(n, 0.05),
            target_indices=np.arange(n),
        )
        states = sample_states_three_way(post, seed=42, threshold_nonzero=0.5, threshold_zero=0.9)
        assert np.all(states == STATE_STRUCTURAL_ZERO)

    def test_three_state_gate_uncertain(self):
        n = 100
        post = PosteriorProbs(
            p_structural=np.full(n, 0.3),
            p_sampling=np.full(n, 0.3),
            p_nonzero=np.full(n, 0.4),
            target_indices=np.arange(n),
        )
        states = sample_states_three_way(post, seed=42, threshold_nonzero=0.8, threshold_zero=0.9)
        assert np.all(states == STATE_UNCERTAIN)

    def test_quality_fail_all_uncertain(self):
        n = 50
        post = PosteriorProbs(
            p_structural=np.full(n, 0.9),
            p_sampling=np.full(n, 0.05),
            p_nonzero=np.full(n, 0.05),
            target_indices=np.arange(n),
        )
        states = sample_states_three_way(post, seed=42, threshold_nonzero=0.5, threshold_zero=0.8, quality_pass=False)
        assert np.all(states == STATE_UNCERTAIN)


class TestFeasibilityProjectionUncertain:

    def test_override_to_uncertain_not_zero(self):
        n = 10
        states = np.full(n, STATE_NONZERO, dtype=np.uint8)
        p_nonzero = np.linspace(0.1, 1.0, n)
        total_seq = np.full(n, 10)
        other_counts = np.full(n, 10)

        feasible, overrides = apply_feasibility_projection_uncertain(
            states, p_nonzero, total_seq, other_counts
        )

        overridden = np.where(states != feasible)[0]
        for idx in overridden:
            assert feasible[idx] == STATE_UNCERTAIN
        assert len(overrides) == len(overridden)

    def test_no_override_when_budget_ok(self):
        n = 10
        states = np.full(n, STATE_NONZERO, dtype=np.uint8)
        p_nonzero = np.linspace(0.1, 1.0, n)
        total_seq = np.full(n, 100)
        other_counts = np.full(n, 50)

        feasible, overrides = apply_feasibility_projection_uncertain(
            states, p_nonzero, total_seq, other_counts
        )
        assert len(overrides) == 0
        assert np.all(feasible == STATE_NONZERO)


class TestCalibrationMetrics:

    def test_compute_calibration_metrics_perfect(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        p_nonzero = np.array([0.1, 0.2, 0.8, 0.9, 0.15, 0.85])
        p_zero = 1 - p_nonzero

        metrics = compute_calibration_metrics(y_true, p_nonzero, p_zero, threshold=0.5)

        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1 == 1.0
        assert metrics.brier_score < 0.1

    def test_recall_constrained_threshold(self):
        np.random.seed(42)
        n = 1000
        y_true = np.random.binomial(1, 0.3, n)
        p_nonzero = np.clip(0.3 + 0.4 * (y_true - 0.3) + np.random.normal(0, 0.1, n), 0.01, 0.99)

        threshold_grid = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
        best_threshold, metrics = select_recall_constrained_threshold(
            y_true, p_nonzero, threshold_grid, min_recall=0.95
        )

        assert best_threshold in threshold_grid
        assert metrics.recall >= 0.95 or best_threshold == threshold_grid[-1]


class TestMasks:

    def test_three_masks_partition(self):
        n_loc, n_time, n_feat = 3, 5, 2
        n_target = n_loc * n_time

        states_raw = np.zeros(n_target, dtype=np.uint8)
        states_raw[:4] = STATE_STRUCTURAL_ZERO
        states_raw[4:7] = STATE_SAMPLING_ZERO
        states_raw[7:11] = STATE_NONZERO
        states_raw[11:] = STATE_UNCERTAIN

        sampled = SampledState(
            sampled_state_raw=states_raw,
            sampled_state_feasible=states_raw.copy(),
            feasibility_overrides=[],
            seed=42,
            n_target=n_target,
        )

        M_target_3d = np.ones((n_loc, n_time, n_feat), dtype=np.uint8)

        class FakePanel:
            n_locations = n_loc
            n_times = n_time
            n_features = n_feat
            location_index = pd.Index([f"loc_{i}" for i in range(n_loc)])
            time_index = pd.date_range("2020-01-01", periods=n_time, freq="14D")
            feature_names = [f"var_{k}" for k in range(n_feat)]

        panel = FakePanel()
        quality_flags = {"var_0": True, "var_1": True}

        masks = create_target_masks_from_states(
            sampled_states={"var_0": sampled},
            panel=panel,
            M_target=M_target_3d,
            seed=42,
            quality_flags=quality_flags,
        )

        mask = masks["var_0"]
        ok, violations = mask.verify_partition()
        assert ok, f"Partition violations: {violations}"
        assert mask.M_target_zero.sum() > 0
        assert mask.M_target_nonzero.sum() > 0
        assert mask.M_target_uncertain.sum() > 0

    def test_combine_variant_masks(self):
        n_loc, n_time, n_feat = 5, 10, 3
        variant_masks = {}

        M_target_3d = np.ones((n_loc, n_time, n_feat), dtype=np.uint8)

        for f in range(n_feat):
            M_zero = np.zeros((n_loc, n_time, n_feat), dtype=np.uint8)
            M_nonzero = np.zeros((n_loc, n_time, n_feat), dtype=np.uint8)
            M_uncertain = np.zeros((n_loc, n_time, n_feat), dtype=np.uint8)
            mask = np.zeros((n_loc, n_time), dtype=bool)
            for i in range(n_loc):
                for j in range(n_time):
                    mask[i, j] = (i + j + f) % 3 == 0
            M_zero[mask, f] = 1
            M_nonzero[~mask, f] = 1
            M_target = M_zero | M_nonzero | M_uncertain

            variant_masks[f"var_{f}"] = ZeroStateMasks(
                M_target_zero=M_zero,
                M_target_nonzero=M_nonzero,
                M_target_uncertain=M_uncertain,
                M_target=M_target,
                location_index=pd.Index([f"loc_{i}" for i in range(n_loc)]),
                time_index=pd.date_range("2020-01-01", periods=n_time, freq="14D"),
                feature_names=[f"var_{k}" for k in range(n_feat)],
                seed=42,
                variant_name=f"var_{f}",
            )

        M_zero_global, M_nonzero_global, M_uncertain_global, M_target_global = combine_variant_masks(variant_masks, M_target_3d)

        assert M_zero_global.shape == (n_loc, n_time, n_feat)
        assert M_nonzero_global.shape == (n_loc, n_time, n_feat)
        assert M_uncertain_global.shape == (n_loc, n_time, n_feat)
        assert np.array_equal(M_target_global, M_zero_global | M_nonzero_global | M_uncertain_global)


class TestPosteriorVerification:

    def test_posterior_sum_to_one(self):
        n = 50
        p_s = np.random.rand(n)
        p_sa = np.random.rand(n)
        p_nz = np.random.rand(n)
        total = p_s + p_sa + p_nz
        variant_posteriors = {
            "var1": PosteriorProbs(
                p_structural=p_s / total,
                p_sampling=p_sa / total,
                p_nonzero=p_nz / total,
                target_indices=np.arange(n),
            ),
        }

        posterior = ZeroStatePosterior(
            variant_posteriors=variant_posteriors,
            location_index=pd.Index(["a"]),
            time_index=pd.date_range("2020-01-01", periods=1, freq="14D"),
            feature_names=["var1"],
        )

        ok, violations = posterior.verify_probabilities_sum_to_one()
        assert ok
        assert violations == 0

    def test_posterior_violations_detected(self):
        n = 10
        variant_posteriors = {
            "var1": PosteriorProbs(
                p_structural=np.full(n, 0.5),
                p_sampling=np.full(n, 0.5),
                p_nonzero=np.full(n, 0.5),
                target_indices=np.arange(n),
            ),
        }

        posterior = ZeroStatePosterior(
            variant_posteriors=variant_posteriors,
            location_index=pd.Index(["a"]),
            time_index=pd.date_range("2020-01-01", periods=1, freq="14D"),
            feature_names=["var1"],
        )

        ok, violations = posterior.verify_probabilities_sum_to_one(atol=1e-6)
        assert not ok
        assert violations == n


class TestCenteredSplineFullRank:

    @pytest.mark.parametrize("df", [3, 4, 5])
    def test_centered_spline_full_rank(self, df):
        n = 200
        cfg = ZINBEncoderConfig(
            time_spline_df=df,
            time_spline_type="natural",
            use_lag_lead=True,
            use_availability_indicators=True,
            center_spline=True,
        )
        enc = DesignEncoder(cfg, "test")
        total_seq = np.random.uniform(100, 1000, n)
        day_index = np.random.uniform(0, 500, n)
        lag_counts = np.random.randint(0, 20, n).astype(np.float64)
        lag_avail = np.ones(n)
        lead_counts = np.random.randint(0, 20, n).astype(np.float64)
        lead_avail = np.ones(n)

        enc.fit(total_seq, day_index, lag_counts, lag_avail, lead_counts, lead_avail)
        X_zi, X_nb, offset = enc.transform(total_seq, day_index, lag_counts, lag_avail, lead_counts, lead_avail)

        assert np.linalg.matrix_rank(X_zi) == X_zi.shape[1], f"ZI rank deficient: rank={np.linalg.matrix_rank(X_zi)}, cols={X_zi.shape[1]}"
        assert np.linalg.matrix_rank(X_nb) == X_nb.shape[1], f"NB rank deficient: rank={np.linalg.matrix_rank(X_nb)}, cols={X_nb.shape[1]}"

    def test_constant_lag_lead_no_rank_loss(self):
        n = 100
        cfg = _make_encoder_config()
        enc = DesignEncoder(cfg, "test")
        total_seq = np.full(n, 500.0)
        day_index = np.arange(n, dtype=np.float64)
        lag_counts = np.zeros(n)
        lag_avail = np.zeros(n)
        lead_counts = np.zeros(n)
        lead_avail = np.zeros(n)

        enc.fit(total_seq, day_index, lag_counts, lag_avail, lead_counts, lead_avail)
        X_zi, X_nb, offset = enc.transform(total_seq, day_index, lag_counts, lag_avail, lead_counts, lead_avail)

        assert np.linalg.matrix_rank(X_zi) == X_zi.shape[1]
        assert np.linalg.matrix_rank(X_nb) == X_nb.shape[1]

    def test_unseen_transform_preserves_schema(self):
        n_train, n_test = 100, 30
        cfg = _make_encoder_config()
        enc = DesignEncoder(cfg, "test")
        total_seq_train = np.random.uniform(100, 1000, n_train)
        day_index_train = np.random.uniform(0, 500, n_train)
        zeros_train = np.zeros(n_train)
        ones_train = np.ones(n_train)

        enc.fit(total_seq_train, day_index_train, zeros_train, ones_train, zeros_train, ones_train)

        total_seq_test = np.random.uniform(100, 1000, n_test)
        day_index_test = np.random.uniform(500, 600, n_test)
        zeros_test = np.zeros(n_test)
        ones_test = np.ones(n_test)

        X_zi_test, X_nb_test, offset_test = enc.transform(
            total_seq_test, day_index_test, zeros_test, ones_test, zeros_test, ones_test
        )
        assert X_zi_test.shape[1] == len(enc.schema["zi_predictor_names"])
        assert X_nb_test.shape[1] == len(enc.schema["nb_predictor_names"])
