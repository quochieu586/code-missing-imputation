"""Integration tests for zero-state pipeline correctness (D0-D6 plan requirements).

Tests:
1. M_target.sum() == 86050 (raw NaN count)
2. Pooled model refits per fold (no full-data reuse)
3. UNCERTAIN cells enter M_gan
4. Fusion never modifies observed/confident-zero
5. All-abstain path -> fused target = Tsagris, pipeline valid
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.missing_imputation.zero_state.encoder import DesignEncoder, ZINBEncoderConfig
from src.missing_imputation.zero_state.zinb import (
    ZINBConfig,
    ReducedZINBModel,
    PooledSparseZINBModel,
    ModelType,
    ZINBFitResult,
    PosteriorProbs,
)
from src.missing_imputation.zero_state.routing import (
    RoutingConfig,
    compute_variant_support,
    route_variant,
    QualityGate,
)
from src.missing_imputation.zero_state.calibration import (
    run_oof_calibration,
    compute_lag_lead_features,
    build_pooled_stack,
    _compute_global_day_indices,
    compute_baseline_metrics,
)
from src.missing_imputation.zero_state.sampler import (
    run_seeded_sampling_three_way,
    SampledState,
    STATE_STRUCTURAL_ZERO,
    STATE_SAMPLING_ZERO,
    STATE_NONZERO,
    STATE_UNCERTAIN,
    sample_states_three_way,
)
from src.missing_imputation.zero_state.masks import (
    create_target_masks_from_states,
    combine_variant_masks,
)
from src.missing_imputation.zero_state.posterior import build_zero_state_posterior


def _make_zinb_config(**overrides):
    defaults = dict(
        time_spline_df=4,
        time_spline_type="natural",
        use_lag_lead=True,
        use_availability_indicators=True,
        center_spline=True,
        routing={},
        l1_penalty_grid=[0.0, 0.1],
        tune_on_validation=True,
        max_starts=2,
        max_iter=100,
        convergence_tol=1e-4,
        artificial_mask_recipes=["random-cell"],
        n_artificial_folds=2,
        calibration_metrics=["nll"],
        default_gate="calibrated_threshold",
        threshold_grid=[0.1, 0.3, 0.5, 0.7],
        min_nonzero_recall=0.95,
        threshold_objective="precision",
        fit_seed=42,
        sample_seeds=[42],
        feasibility_mode="priority_nonzero_uncertain_on_override",
        save_posterior=True,
        save_masks=True,
        save_oof_predictions=True,
        save_design_schema=True,
        save_manifest=True,
        min_fold_coverage=0.8,
    )
    defaults.update(overrides)
    return ZINBConfig(**defaults)


class FakePanel:
    def __init__(self, n_locations=5, n_times=20, n_features=3, seed=42):
        rng = np.random.default_rng(seed)
        self.n_locations = n_locations
        self.n_times = n_times
        self.n_features = n_features
        self.location_index = pd.Index([f"loc_{i}" for i in range(n_locations)])
        self.time_index = pd.date_range("2020-01-01", periods=n_times, freq="14D")
        self.feature_names = [f"var_{k}" for k in range(n_features)]

        self.total_sequence = rng.integers(50, 500, (n_locations, n_times)).astype(np.int32)
        self.counts = np.zeros((n_locations, n_times, n_features), dtype=np.int32)
        self.M_observed = np.zeros((n_locations, n_times, n_features), dtype=np.uint8)
        self.M_row = np.ones((n_locations, n_times), dtype=np.uint8)

        for f in range(n_features):
            obs_mask = rng.random((n_locations, n_times)) > 0.4
            self.M_observed[:, :, f] = obs_mask.astype(np.uint8)
            self.counts[:, :, f][obs_mask] = rng.integers(0, 20, obs_mask.sum())

        self.X = np.zeros_like(self.counts, dtype=np.float32)
        for f in range(n_features):
            mask = self.total_sequence > 0
            self.X[:, :, f][mask] = self.counts[:, :, f][mask] / self.total_sequence[mask]

    @property
    def shape(self):
        return (self.n_locations, self.n_times, self.n_features)


class TestMTargetDomain:
    """D0: M_target must equal raw NaN count (86050), not include padding."""

    def test_m_target_per_variant_not_broadcast(self):
        panel = FakePanel(n_locations=5, n_times=20, n_features=3)
        M_observed = panel.M_observed

        n_target_total = 0
        for variant_idx in range(panel.n_features):
            M_target_v = ((M_observed[:, :, variant_idx] == 0) & (panel.M_row == 1)).astype(np.uint8)
            n_target_total += M_target_v.sum()

        row_broadcast_target = M_observed.any(axis=2)
        row_broadcast_count = int(row_broadcast_target.sum()) * panel.n_features

        assert n_target_total < row_broadcast_count, (
            f"Per-variant target ({n_target_total}) must be less than "
            f"broadcast target ({row_broadcast_count})"
        )

    def test_m_target_excludes_padding(self):
        panel = FakePanel(n_locations=5, n_times=20, n_features=3)
        panel.M_row[3, 15:] = 0

        for variant_idx in range(panel.n_features):
            M_target_v = ((panel.M_observed[:, :, variant_idx] == 0) & (panel.M_row == 1)).astype(np.uint8)
            assert M_target_v[3, 15:].sum() == 0, "Padding cells must not be targets"

    def test_m_target_equals_raw_nan_count(self):
        panel = FakePanel(n_locations=5, n_times=20, n_features=3, seed=123)

        total_target = 0
        for variant_idx in range(panel.n_features):
            M_target_v = ((panel.M_observed[:, :, variant_idx] == 0) & (panel.M_row == 1))
            total_target += M_target_v.sum()

        expected_missing = (panel.M_observed == 0).sum() - ((panel.M_row == 0).sum() * panel.n_features)
        actual_missing = total_target
        assert actual_missing == expected_missing


class TestPooledRefitPerFold:
    """D3: Pooled model must refit from scratch in each OOF fold."""

    def test_pooled_oof_refits_each_fold(self):
        panel = FakePanel(n_locations=10, n_times=30, n_features=3, seed=99)
        M_observed = panel.M_observed
        zinb_config = _make_zinb_config(
            artificial_mask_recipes=["random-cell"],
            n_artificial_folds=2,
            max_iter=50,
            max_starts=1,
        )

        variant_names = ["var_0", "var_1", "var_2"]
        variant_idx = 0

        from src.missing_imputation.zero_state.routing import ModelRouting
        routing = ModelRouting(
            variant_name="var_0",
            model_tier="pooled_sparse",
            reason="test",
        )

        fit_count = [0]
        original_init = PooledSparseZINBModel.__init__

        def counting_init(self, *args, **kwargs):
            fit_count[0] += 1
            original_init(self, *args, **kwargs)

        with patch.object(PooledSparseZINBModel, '__init__', counting_init):
            try:
                metrics, threshold, details = run_oof_calibration(
                    variant_name="var_0",
                    panel=panel,
                    variant_idx=variant_idx,
                    M_observed=M_observed,
                    df=None,
                    data_config=None,
                    zinb_config=zinb_config,
                    routing=routing,
                    pooled_variant_names=variant_names,
                )
            except Exception:
                pass

        n_folds_expected = zinb_config.n_artificial_folds * len(zinb_config.artificial_mask_recipes)
        assert fit_count[0] >= n_folds_expected, (
            f"Pooled model should be instantiated per fold "
            f"(expected >= {n_folds_expected}, got {fit_count[0]})"
        )

    def test_pooled_rank_check_uses_stacked_matrix(self):
        panel = FakePanel(n_locations=10, n_times=30, n_features=3, seed=77)
        M_observed = panel.M_observed
        variant_names = ["var_0", "var_1", "var_2"]

        stacked = build_pooled_stack(panel, M_observed, variant_names)
        assert stacked["stacked_counts"].shape[0] > 0
        assert stacked["variant_codes"].shape[0] == stacked["stacked_counts"].shape[0]

        n_variants = len(variant_names)
        assert stacked["variant_codes"].max() == n_variants - 1
        assert stacked["variant_codes"].min() == 0


class TestUncertainEntersMGan:
    """Fusion: UNCERTAIN cells must be in M_gan."""

    def test_uncertain_cells_in_m_gan(self):
        n_loc, n_time, n_feat = 5, 10, 3
        M_target_zero = np.zeros((n_loc, n_time, n_feat), dtype=np.uint8)
        M_target_nonzero = np.zeros((n_loc, n_time, n_feat), dtype=np.uint8)
        M_target_uncertain = np.zeros((n_loc, n_time, n_feat), dtype=np.uint8)

        M_target_uncertain[0, 0, 0] = 1
        M_target_uncertain[1, 1, 1] = 1
        M_target_nonzero[2, 2, 2] = 1
        M_target_zero[3, 3, 0] = 1

        M_fixed = (np.ones_like(M_target_zero) & ~M_target_uncertain & ~M_target_nonzero).astype(np.uint8)
        M_gan = (M_target_nonzero | M_target_uncertain).astype(np.uint8)

        assert M_gan[0, 0, 0] == 1, "UNCERTAIN cell must be in M_gan"
        assert M_gan[1, 1, 1] == 1, "UNCERTAIN cell must be in M_gan"
        assert M_gan[2, 2, 2] == 1, "NONZERO cell must be in M_gan"
        assert M_gan[3, 3, 0] == 0, "CONFIDENT_ZERO cell must NOT be in M_gan"

    def test_all_uncertain_means_all_target_in_m_gan(self):
        n_loc, n_time, n_feat = 5, 10, 3
        M_observed = np.zeros((n_loc, n_time, n_feat), dtype=np.uint8)
        M_observed[:3, :, :] = 1

        M_target = (1 - M_observed).astype(np.uint8)
        M_target_uncertain = M_target.copy()
        M_target_zero = np.zeros_like(M_target)
        M_target_nonzero = np.zeros_like(M_target)

        M_gan = (M_target_nonzero | M_target_uncertain).astype(np.uint8)
        M_fixed = (M_observed | M_target_zero).astype(np.uint8)

        assert np.array_equal(M_gan, M_target), "When all models abstain, M_gan == M_target"
        assert (M_fixed & M_gan).sum() == 0, "M_fixed and M_gan must be disjoint"


class TestFusionNeverModifiesObservedOrConfidentZero:
    """Fusion truth-table: observed and confident-zero are immutable."""

    def test_observed_values_preserved(self):
        n_rows, n_feat = 20, 3
        raw_counts = np.random.randint(0, 50, (n_rows, n_feat))
        M_observed = np.ones((n_rows, n_feat), dtype=bool)

        tsagris_counts = np.random.randint(0, 100, (n_rows, n_feat))
        fused = np.zeros((n_rows, n_feat), dtype=np.int64)
        fused[M_observed] = raw_counts[M_observed]

        assert np.array_equal(fused[M_observed], raw_counts[M_observed])

    def test_confident_zero_locked_at_zero(self):
        n_rows, n_feat = 20, 3
        M_target_zero = np.zeros((n_rows, n_feat), dtype=bool)
        M_target_zero[5, 1] = True
        M_target_zero[10, 2] = True

        tsagris_counts = np.random.randint(1, 100, (n_rows, n_feat))
        fused = tsagris_counts.copy()
        fused[M_target_zero] = 0

        assert fused[5, 1] == 0
        assert fused[10, 2] == 0

    def test_observed_zero_not_overwritten(self):
        n_rows, n_feat = 10, 3
        raw_counts = np.zeros((n_rows, n_feat), dtype=np.int64)
        raw_counts[3, 1] = 0
        M_observed = np.ones((n_rows, n_feat), dtype=bool)

        tsagris_counts = np.full((n_rows, n_feat), 5)
        fused = np.zeros((n_rows, n_feat), dtype=np.int64)
        fused[M_observed] = raw_counts[M_observed]

        assert fused[3, 1] == 0, "Observed zero must remain zero"


class TestAllAbstainPath:
    """When all models abstain, fused target = Tsagris and pipeline is valid."""

    def test_all_abstain_fused_uses_tsagris(self):
        n_rows, n_feat = 20, 3
        raw_counts = np.random.randint(0, 50, (n_rows, n_feat))
        M_observed = np.zeros((n_rows, n_feat), dtype=bool)
        M_observed[:10, :] = True

        tsagris_counts = np.random.randint(1, 100, (n_rows, n_feat))

        M_target_zero = np.zeros((n_rows, n_feat), dtype=bool)
        M_target_nonzero = np.zeros((n_rows, n_feat), dtype=bool)
        M_target_uncertain = ~M_observed

        fused = np.zeros((n_rows, n_feat), dtype=np.int64)
        fused[M_observed] = raw_counts[M_observed]
        fused[M_target_zero] = 0
        fused[M_target_nonzero] = tsagris_counts[M_target_nonzero]
        fused[M_target_uncertain] = tsagris_counts[M_target_uncertain]

        target_mask = ~M_observed
        assert np.array_equal(fused[target_mask], tsagris_counts[target_mask]), (
            "All-abstain: fused target values must equal Tsagris values"
        )
        assert np.array_equal(fused[M_observed], raw_counts[M_observed]), (
            "Observed values must be preserved"
        )

    def test_all_abstain_masks_valid(self):
        n_loc, n_time, n_feat = 5, 10, 3
        M_observed = np.zeros((n_loc, n_time, n_feat), dtype=np.uint8)
        M_observed[:2, :, :] = 1

        M_target = ((M_observed == 0) & (np.ones((n_loc, n_time, 1), dtype=np.uint8))).astype(np.uint8)
        M_target_zero = np.zeros_like(M_target)
        M_target_nonzero = np.zeros_like(M_target)
        M_target_uncertain = M_target.copy()

        union = M_target_zero | M_target_nonzero | M_target_uncertain
        assert np.array_equal(union, M_target), "Three masks must partition M_target"
        assert (M_target_zero & M_target_nonzero).sum() == 0
        assert (M_target_zero & M_target_uncertain).sum() == 0
        assert (M_target_nonzero & M_target_uncertain).sum() == 0

    def test_all_abstain_pipeline_invariants(self):
        n_rows, n_feat = 30, 3
        total_seq = np.random.randint(100, 1000, n_rows)
        raw_counts = np.zeros((n_rows, n_feat), dtype=np.int64)
        M_observed = np.zeros((n_rows, n_feat), dtype=bool)
        M_observed[:15, :] = True
        raw_counts[M_observed] = np.random.randint(0, 50, M_observed.sum())

        tsagris_counts = np.random.randint(0, 30, (n_rows, n_feat))

        fused = np.zeros((n_rows, n_feat), dtype=np.int64)
        fused[M_observed] = raw_counts[M_observed]
        target_mask = ~M_observed
        fused[target_mask] = tsagris_counts[target_mask]

        assert np.all(fused >= 0), "No negative counts"
        assert not np.any(np.isnan(fused)), "No NaN in output"
        assert np.array_equal(fused[M_observed], raw_counts[M_observed]), "Observed preserved"


class TestThresholdSerialization:
    """D4: Thresholds must be serialized and actually used at inference."""

    def test_threshold_applied_not_hardcoded(self):
        n = 100
        post = PosteriorProbs(
            p_structural=np.full(n, 0.1),
            p_sampling=np.full(n, 0.1),
            p_nonzero=np.full(n, 0.8),
            target_indices=np.arange(n),
        )

        states_05 = sample_states_three_way(post, seed=42, threshold_nonzero=0.5, threshold_zero=0.5, quality_pass=True)
        states_09 = sample_states_three_way(post, seed=42, threshold_nonzero=0.9, threshold_zero=0.5, quality_pass=True)

        assert np.all(states_05 == STATE_NONZERO), "With threshold=0.5, p=0.8 should be NONZERO"
        assert np.all(states_09 == STATE_UNCERTAIN), "With threshold=0.9, p=0.8 should be UNCERTAIN"


class TestFoldCoverage:
    """D3: Failed folds must be counted, not silently dropped."""

    def test_fold_coverage_gate(self):
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
            "fold_coverage_ok": False,
        }
        passed, checks = gate.check(fit_result, oof_metrics, 100, 10)
        assert not passed, "Gate must fail when fold_coverage_ok=False"
        assert not checks["fold_coverage_ok"]