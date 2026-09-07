from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.loader import load_config, load_covariants
from ..data.panel import build_panel, PanelTensor
from ..data.closure import (
    counts_to_proportions,
    proportions_to_counts_largest_remainder,
    compute_other,
    validate_closure,
)
from ..zero_state.zinb import (
    ReducedZINBModel,
    PooledSparseZINBModel,
    ZINBConfig,
    ModelType,
    ZINBFitResult,
)
from ..zero_state.encoder import ZINBEncoderConfig
from ..zero_state.routing import (
    RoutingConfig,
    compute_variant_support,
    route_variant,
    QualityGate,
)
from ..zero_state.calibration import (
    run_oof_calibration,
    compute_lag_lead_features,
    build_pooled_stack,
    _compute_global_day_indices,
)
from ..zero_state.posterior import build_zero_state_posterior
from ..zero_state.sampler import (
    run_seeded_sampling_three_way,
    save_sampled_states,
    STATE_NONZERO,
)
from ..zero_state.masks import (
    create_target_masks_from_states,
    combine_variant_masks,
    save_combined_masks,
)
from ..zero_state.artifacts import (
    save_variant_support_report,
    save_model_routing_report,
    save_zero_state_metrics,
    save_design_schema,
    save_model_manifest,
    save_oof_predictions,
)


@dataclass
class ZINBPipelineResult:
    dataset_01_zinb_path: str
    posterior_path: str
    masks_path: str
    sampled_states_path: str
    calibration_metrics: dict
    manifest: dict
    invariants_ok: bool
    stage_status: str


def run_zinb_pipeline(
    data_path: str | Path,
    config_path: str | Path,
    observed_mask_path: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    gate_mode: str = "calibrated_threshold",
) -> ZINBPipelineResult:
    project_root = Path(__file__).parents[3]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zinb_config = ZINBConfig.from_yaml(config_path)
    data_config = load_config(project_root / "configs" / "data.yaml")
    routing_config = RoutingConfig.from_zinb_config(zinb_config)

    df = load_covariants(data_path, data_config)
    variant_cols = data_config.variant_components

    mask_data = np.load(observed_mask_path, allow_pickle=True)
    M_observed = mask_data["M_observed"]

    panel = build_panel(df, data_config, variant_cols, freq_days=14)

    assert panel.M_observed.shape == M_observed.shape
    assert np.array_equal(panel.M_observed, M_observed)

    n_raw_nan = int(df[variant_cols].isna().values.sum())

    M_row_3d = np.broadcast_to(panel.M_row[:, :, None], panel.M_observed.shape)
    M_target = ((panel.M_observed == 0) & (M_row_3d == 1)).astype(np.uint8)
    assert int(M_target.sum()) == n_raw_nan, (
        f"M_target.sum()={int(M_target.sum())} != raw NaN count {n_raw_nan}; "
        "target domain must equal raw NaN variant cells (D0)"
    )

    print(f"Panel shape: {panel.shape}")
    print(f"Raw NaN target cells: {n_raw_nan}")

    encoder_cfg = ZINBEncoderConfig(
        time_spline_df=zinb_config.time_spline_df,
        time_spline_type=zinb_config.time_spline_type,
        use_lag_lead=zinb_config.use_lag_lead,
        use_availability_indicators=zinb_config.use_availability_indicators,
        center_spline=zinb_config.center_spline,
    )
    from ..zero_state.encoder import DesignEncoder
    _probe = DesignEncoder(encoder_cfg, "probe")
    dummy_n = 200
    rng_probe = np.random.default_rng(0)
    _probe.fit(
        total_seq=np.full(dummy_n, 100.0),
        day_index=np.linspace(0, 280, dummy_n),
        lag_counts=rng_probe.integers(0, 5, dummy_n).astype(float),
        lag_avail=np.ones(dummy_n),
        lead_counts=rng_probe.integers(0, 5, dummy_n).astype(float),
        lead_avail=np.ones(dummy_n),
    )
    n_predictors = len(_probe.schema["zi_predictor_names"]) + len(_probe.schema["nb_predictor_names"])

    print("\n=== Computing variant support ===")
    supports = []
    routings = []
    for variant_idx, variant_name in enumerate(variant_cols):
        obs_locs, obs_times = np.where(M_observed[:, :, variant_idx] == 1)
        counts = panel.counts[obs_locs, obs_times, variant_idx]
        total_seq_v = panel.total_sequence[obs_locs, obs_times]
        support = compute_variant_support(
            variant_name=variant_name,
            counts=counts,
            total_seq=total_seq_v,
            M_observed=M_observed[:, :, variant_idx],
            panel_shape=panel.shape[:2],
            n_predictors=n_predictors,
        )
        routing = route_variant(support, routing_config)
        supports.append(support)
        routings.append(routing)
        print(f"  {variant_name}: tier={support.tier}, route={routing.model_tier}")

    support_path = save_variant_support_report(supports, output_dir / "variant_support_report.csv")
    routing_path = save_model_routing_report(routings, output_dir / "model_routing.csv")

    print("\n=== Fitting models ===")
    fit_results: dict[str, ZINBFitResult | None] = {}
    fitted_models: dict = {}
    quality_flags: dict[str, bool] = {}
    thresholds: dict[str, float] = {}
    calibration_metrics: dict[str, dict] = {}
    oof_records: list[dict] = []

    quality_gate = QualityGate(min_nonzero_recall=zinb_config.min_nonzero_recall)

    pooled_variant_names = [
        supports[i].variant_name
        for i in range(len(variant_cols))
        if routings[i].model_tier == "pooled_sparse"
    ]

    pooled_model = None
    pooled_fit_result = None
    if pooled_variant_names:
        print(f"  Building pooled sparse model for {len(pooled_variant_names)} variants...")
        stacked = build_pooled_stack(panel, M_observed, pooled_variant_names)
        pooled_model = PooledSparseZINBModel(pooled_variant_names, zinb_config)
        pooled_fit_result = pooled_model.fit(**stacked)
        print(f"    Pooled fit: {pooled_fit_result.model_type.value}, converged={pooled_fit_result.converged}")

    def _finalize_variant(
        variant_name: str,
        routing,
        fit_result,
        model_entry,
        cal_metrics,
        threshold,
        details,
        counts,
    ) -> bool:
        baseline_nll = details.get("baseline_nll", np.inf)
        baseline_brier = details.get("baseline_brier", np.inf)
        oof_metrics = {
            "nll": cal_metrics.nll,
            "brier_score": cal_metrics.brier_score,
            "recall": cal_metrics.recall,
            "baseline_nll": baseline_nll,
            "baseline_brier": baseline_brier,
            "frac_nonzero_at_selected_threshold": details.get("frac_nonzero_at_selected_threshold", 0.0),
            "frac_nonzero_at_0_5": details.get("frac_nonzero_at_0_5", 0.0),
            "rank_ok": details.get("rank_ok", True),
            "fold_coverage_ok": details.get("fold_coverage_ok", False),
            "schema_match": details.get("schema_match", True),
        }
        passed, checks = quality_gate.check(fit_result, oof_metrics, len(counts), n_predictors)
        routing.quality_gate = "PASS" if passed else "FAIL"
        routing.gate_metrics = checks
        quality_flags[variant_name] = passed
        thresholds[variant_name] = float(threshold) if passed else 0.5
        calibration_metrics[variant_name] = {
            "status": "ok",
            "test_metrics": cal_metrics.to_dict(),
            "best_threshold": float(threshold),
            "threshold_used_at_inference": float(threshold) if passed else 0.5,
            "quality_gate": routing.quality_gate,
            "gate_checks": checks,
            "details": {k: v for k, v in details.items() if k != "oof_records"},
        }
        oof_records.extend(details.get("oof_records", []))
        return passed

    for variant_idx, variant_name in enumerate(variant_cols):
        routing = routings[variant_idx]

        if routing.model_tier == "abstain":
            fit_results[variant_name] = None
            quality_flags[variant_name] = False
            thresholds[variant_name] = 0.5
            calibration_metrics[variant_name] = {"status": "abstain", "reason": routing.reason}
            print(f"  {variant_name}: ABSTAIN (no fit)")
            continue

        print(f"  Fitting {variant_name} ({routing.model_tier})...")

        obs_locs, obs_times = np.where(M_observed[:, :, variant_idx] == 1)
        counts = panel.counts[obs_locs, obs_times, variant_idx].astype(np.float64)
        total_seq_v = panel.total_sequence[obs_locs, obs_times].astype(np.float64)
        day_index_v = _compute_global_day_indices(panel, obs_locs, obs_times)
        lag_counts, lag_avail, lead_counts, lead_avail = compute_lag_lead_features(
            panel.counts[:, :, variant_idx].astype(np.float64),
            M_observed[:, :, variant_idx],
            obs_locs, obs_times,
        )

        if routing.model_tier == "reduced_per_variant":
            model = ReducedZINBModel(variant_name, zinb_config)
            n_train_obs = len(counts)
            n_val_obs = max(1, n_train_obs // 5)
            validation_mask = np.zeros(n_train_obs, dtype=bool)
            validation_mask[-n_val_obs:] = True
            fit_result = model.fit(
                counts=counts,
                total_seq=total_seq_v,
                day_index=day_index_v,
                lag_counts=lag_counts,
                lag_avail=lag_avail,
                lead_counts=lead_counts,
                lead_avail=lead_avail,
                validation_mask=validation_mask,
            )
            fit_results[variant_name] = fit_result

            if fit_result.model_type == ModelType.ABSTAIN:
                routing.model_tier = "abstain"
                routing.reason = fit_result.fallback_reason or "reduced ZINB failed"
                quality_flags[variant_name] = False
                thresholds[variant_name] = 0.5
                calibration_metrics[variant_name] = {"status": "abstain", "reason": routing.reason}
                print(f"    {variant_name}: ABSTAIN")
                continue

            fitted_models[variant_name] = {"model": model, "fit_result": fit_result}

            metrics, threshold, details = run_oof_calibration(
                variant_name=variant_name,
                panel=panel,
                variant_idx=variant_idx,
                M_observed=M_observed,
                df=df,
                data_config=data_config,
                zinb_config=zinb_config,
                routing=routing,
                min_fold_coverage=zinb_config.min_fold_coverage,
            )

            passed = _finalize_variant(variant_name, routing, fit_result, fitted_models[variant_name], metrics, threshold, details, counts)
            print(f"    {variant_name}: gate={routing.quality_gate}, threshold={threshold:.3f}")

            if not passed:
                if supports[variant_idx].n_positive <= routing_config.high_support_n_positive:
                    print(f"    {variant_name}: quality gate FAIL, downgrading to pooled")
                    routing.model_tier = "pooled_sparse"
                    routing.reason += " -> downgraded to pooled after gate FAIL"

                    if variant_name not in pooled_variant_names:
                        pooled_variant_names.append(variant_name)
                        print(f"    Re-fitting pooled model to include {variant_name}...")
                        stacked = build_pooled_stack(panel, M_observed, pooled_variant_names)
                        pooled_model = PooledSparseZINBModel(pooled_variant_names, zinb_config)
                        pooled_fit_result = pooled_model.fit(**stacked)

                    if pooled_fit_result is not None and pooled_fit_result.model_type == ModelType.ZINB:
                        fitted_models[variant_name] = {"model": pooled_model, "fit_result": pooled_fit_result}
                        fit_results[variant_name] = pooled_fit_result

                        metrics, threshold, details = run_oof_calibration(
                            variant_name=variant_name,
                            panel=panel,
                            variant_idx=variant_idx,
                            M_observed=M_observed,
                            df=df,
                            data_config=data_config,
                            zinb_config=zinb_config,
                            routing=routing,
                            pooled_variant_names=list(pooled_variant_names),
                            min_fold_coverage=zinb_config.min_fold_coverage,
                        )
                        passed = _finalize_variant(variant_name, routing, pooled_fit_result, fitted_models[variant_name], metrics, threshold, details, counts)
                        print(f"    {variant_name}: pooled gate={routing.quality_gate}, threshold={threshold:.3f}")
                    else:
                        routing.model_tier = "abstain"
                        routing.reason = "pooled ZINB failed after downgrade -> ABSTAIN"
                        quality_flags[variant_name] = False
                        thresholds[variant_name] = 0.5
                        calibration_metrics[variant_name] = {"status": "abstain", "reason": routing.reason}
                        print(f"    {variant_name}: ABSTAIN (pooled re-fit failed)")

        else:
            if pooled_fit_result is not None and pooled_fit_result.model_type == ModelType.ZINB and variant_name in pooled_variant_names:
                fitted_models[variant_name] = {"model": pooled_model, "fit_result": pooled_fit_result}
                fit_results[variant_name] = pooled_fit_result

                metrics, threshold, details = run_oof_calibration(
                    variant_name=variant_name,
                    panel=panel,
                    variant_idx=variant_idx,
                    M_observed=M_observed,
                    df=df,
                    data_config=data_config,
                    zinb_config=zinb_config,
                    routing=routing,
                    pooled_variant_names=list(pooled_variant_names),
                    min_fold_coverage=zinb_config.min_fold_coverage,
                )
                _finalize_variant(variant_name, routing, pooled_fit_result, fitted_models[variant_name], metrics, threshold, details, counts)
                print(f"    {variant_name}: pooled gate={routing.quality_gate}, threshold={threshold:.3f}")
            else:
                routing.model_tier = "abstain"
                routing.reason = "pooled ZINB failed -> ABSTAIN"
                routing.quality_gate = "FAIL"
                quality_flags[variant_name] = False
                thresholds[variant_name] = 0.5
                fit_results[variant_name] = pooled_fit_result
                calibration_metrics[variant_name] = {"status": "abstain", "reason": routing.reason}
                print(f"    {variant_name}: ABSTAIN (pooled failed)")

    save_model_routing_report(routings, output_dir / "model_routing.csv")

    print("\n=== Computing posterior ===")
    posterior = build_zero_state_posterior(
        fitted_models=fitted_models,
        panel=panel,
        M_target=M_target,
        quality_flags=quality_flags,
        M_observed=M_observed,
    )
    ok, violations = posterior.verify_probabilities_sum_to_one()
    if not ok:
        print(f"  WARNING: {violations} probability sum violations")

    posterior_path = output_dir / "zero_state_posterior.npz"
    posterior.save(posterior_path)

    print("\n=== Three-state sampling ===")
    sampled_results = run_seeded_sampling_three_way(
        posteriors={
            v: p for v, p in posterior.variant_posteriors.items()
            if not np.any(np.isnan(p.p_nonzero))
        },
        quality_flags=quality_flags,
        panel=panel,
        M_target=M_target,
        seeds=zinb_config.sample_seeds,
        thresholds=thresholds,
        default_threshold=0.5,
    )

    sampled_states_path = output_dir / "sampled_states.npz"
    save_sampled_states(sampled_results, sampled_states_path)

    print("\n=== Creating target masks ===")
    primary_seed = zinb_config.sample_seeds[0]
    primary_states = sampled_results[primary_seed]

    from ..zero_state.sampler import SampledState, STATE_UNCERTAIN
    for variant_name in variant_cols:
        variant_idx = panel.feature_names.index(variant_name)
        n_target_variant = int(M_target[:, :, variant_idx].sum())
        locs_v, times_v = np.where(M_target[:, :, variant_idx] == 1)
        if variant_name not in primary_states:
            primary_states[variant_name] = SampledState(
                sampled_state_raw=np.full(n_target_variant, STATE_UNCERTAIN, dtype=np.uint8),
                sampled_state_feasible=np.full(n_target_variant, STATE_UNCERTAIN, dtype=np.uint8),
                seed=primary_seed,
                n_target=n_target_variant,
                n_uncertain=n_target_variant,
                target_locs=locs_v,
                target_times=times_v,
            )

    variant_masks = create_target_masks_from_states(
        sampled_states=primary_states,
        panel=panel,
        M_target=M_target,
        seed=primary_seed,
        quality_flags=quality_flags,
        use_feasible=True,
    )

    M_target_zero, M_target_nonzero, M_target_uncertain, M_target_combined = combine_variant_masks(variant_masks, M_target)

    for variant_name, mask in variant_masks.items():
        ok, violations = mask.verify_partition()
        if not ok:
            print(f"  WARNING: {variant_name} mask partition violations: {violations}")

    masks_path = output_dir / "zero_state_masks.npz"
    save_combined_masks(
        M_target_zero, M_target_nonzero, M_target_uncertain, M_target_combined,
        panel.location_index, panel.time_index, panel.feature_names,
        primary_seed, masks_path,
    )

    for name in ("M_target_zero", "M_target_nonzero", "M_target_uncertain"):
        arr = {"M_target_zero": M_target_zero, "M_target_nonzero": M_target_nonzero, "M_target_uncertain": M_target_uncertain}[name]
        np.savez_compressed(
            output_dir / f"{name}.npz",
            mask=arr,
            location_index=panel.location_index.to_numpy(),
            time_index=panel.time_index.to_numpy(),
            feature_names=np.array(panel.feature_names, dtype=object),
        )

    print("\n=== Building dataset_01_zinb (diagnostic) ===")
    dataset_01_path = _build_dataset_01_zinb(
        panel=panel,
        M_observed=M_observed,
        M_target_zero=M_target_zero,
        M_target_nonzero=M_target_nonzero,
        variant_cols=variant_cols,
        output_dir=output_dir,
        data_config=data_config,
    )

    print("\n=== Saving metrics and manifest ===")
    metrics_path = save_zero_state_metrics(calibration_metrics, output_dir / "zero_state_metrics.csv")

    schema_paths = []
    if zinb_config.save_design_schema:
        for variant_name, entry in fitted_models.items():
            model = entry["model"]
            fr = entry["fit_result"]
            schema = fr.design_schema if fr is not None and fr.design_schema else getattr(model.encoder, "schema", {})
            if schema:
                sp = save_design_schema(schema, output_dir / "design_schema" / f"{_sanitize_filename(variant_name)}.json")
                schema_paths.append(sp)

    if zinb_config.save_oof_predictions:
        save_oof_predictions(oof_records, output_dir / "oof_predictions.parquet")

    artifact_paths = [
        posterior_path, masks_path, sampled_states_path, dataset_01_path,
        support_path, routing_path, metrics_path,
    ] + schema_paths

    df_zinb = pd.read_csv(dataset_01_path)
    counts_zinb = df_zinb[variant_cols].values.astype(np.int64)
    total_seq_flat = df_zinb[data_config.total_sequence_col].values.astype(np.int64)
    other = df_zinb[data_config.other_col].values.astype(np.int64)

    M_row_flat = panel.M_row.reshape(-1)
    observed_mask_flat = (M_observed.reshape(-1, len(variant_cols)) & (M_row_flat[:, None] == 1)).astype(bool)
    # Filter to M_row==1 rows to match df_zinb (which is filtered by M_row)
    observed_mask_mrow = observed_mask_flat[M_row_flat == 1]

    closure_ok, n_viol = validate_closure(counts_zinb, total_seq_flat, other)
    observed_unchanged = np.array_equal(
        counts_zinb[observed_mask_mrow],
        panel.counts.reshape(-1, len(variant_cols))[M_row_flat == 1][observed_mask_mrow],
    )
    no_nan = not np.any(np.isnan(counts_zinb))
    non_negative = np.all(counts_zinb >= 0)
    target_count_ok = int(M_target.sum()) == n_raw_nan
    invariants_ok = closure_ok and observed_unchanged and no_nan and non_negative and target_count_ok

    stage_status = "SUCCESS" if invariants_ok else "FAILED"

    manifest = save_model_manifest(
        run_id=f"zinb_{int(time.time())}",
        variant_routings=routings,
        fit_results=fit_results,
        quality_flags=quality_flags,
        artifact_paths=artifact_paths,
        config_params={
            "time_spline_df": zinb_config.time_spline_df,
            "time_spline_type": zinb_config.time_spline_type,
            "center_spline": zinb_config.center_spline,
            "l1_penalty_grid": zinb_config.l1_penalty_grid,
            "min_nonzero_recall": zinb_config.min_nonzero_recall,
            "threshold_grid": zinb_config.threshold_grid,
            "sample_seeds": zinb_config.sample_seeds,
            "routing": routing_config.__dict__,
        },
        seeds=[seed] + zinb_config.sample_seeds,
        output_path=output_dir / "model_manifest.json",
    )
    manifest["stage"] = "zero_state"
    manifest["stage_status"] = stage_status
    manifest["thresholds_by_variant"] = {
        v: {"threshold": thresholds.get(v, 0.5), "quality_pass": quality_flags.get(v, False)}
        for v in variant_cols
    }
    manifest["target_invariants"] = {
        "raw_nan_count": n_raw_nan,
        "m_target_sum": int(M_target.sum()),
        "m_target_nonzero_sum": int(M_target_nonzero.sum()),
        "m_target_zero_sum": int(M_target_zero.sum()),
        "m_target_uncertain_sum": int(M_target_uncertain.sum()),
    }
    with (output_dir / "model_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, default=str)

    stage_manifest = {
        "stage": "zero_state",
        "status": stage_status,
        "run_id": manifest["run_id"],
        "invariants": manifest["target_invariants"],
    }
    with (output_dir / "stage_manifest.json").open("w") as f:
        json.dump(stage_manifest, f, indent=2)

    print(f"  Invariants: closure={closure_ok}, observed_unchanged={observed_unchanged}, "
          f"no_nan={no_nan}, non_negative={non_negative}, target_count_ok={target_count_ok}")
    print(f"  Stage status: {stage_status}")

    return ZINBPipelineResult(
        dataset_01_zinb_path=str(dataset_01_path),
        posterior_path=str(posterior_path),
        masks_path=str(masks_path),
        sampled_states_path=str(sampled_states_path),
        calibration_metrics=calibration_metrics,
        manifest=manifest,
        invariants_ok=invariants_ok,
        stage_status=stage_status,
    )


def _sanitize_filename(name: str) -> str:
    import re

    return re.sub(r'[:<>|?"/\\]', "_", name)


def _build_dataset_01_zinb(
    panel: PanelTensor,
    M_observed: np.ndarray,
    M_target_zero: np.ndarray,
    M_target_nonzero: np.ndarray,
    variant_cols: list[str],
    output_dir: Path,
    data_config,
) -> Path:
    n_locations, n_times, n_features = panel.shape
    counts = panel.counts.copy().astype(np.int64)

    for variant_idx in range(len(variant_cols)):
        variant_target_zero = M_target_zero[:, :, variant_idx].astype(bool)
        variant_target_nonzero = M_target_nonzero[:, :, variant_idx].astype(bool)

        counts[variant_target_zero, variant_idx] = 0

        nonzero_indices = np.where(variant_target_nonzero)
        if len(nonzero_indices[0]) > 0:
            counts[nonzero_indices[0], nonzero_indices[1], variant_idx] = np.maximum(
                counts[nonzero_indices[0], nonzero_indices[1], variant_idx], 1
            )

    total_seq_flat = panel.total_sequence.reshape(-1).astype(np.float64)
    counts_flat = counts.reshape(-1, n_features)
    M_observed_flat = M_observed.reshape(-1, n_features)

    proportions = counts_to_proportions(counts_flat, total_seq_flat)
    final_counts = proportions_to_counts_largest_remainder(
        proportions, total_seq_flat, M_observed_flat.astype(bool), panel.counts.reshape(-1, n_features)
    )

    other = compute_other(final_counts, total_seq_flat)
    closure_ok, n_viol = validate_closure(final_counts, total_seq_flat, other)
    if not closure_ok:
        print(f"  WARNING: {n_viol} closure violations after largest-remainder")

    df_out = pd.DataFrame({
        data_config.location_col: np.repeat(panel.location_index.to_numpy(), n_times),
        data_config.date_col: np.tile(panel.time_index.to_numpy(), n_locations),
        data_config.total_sequence_col: total_seq_flat.astype(np.int64),
    })

    for j, col in enumerate(variant_cols):
        df_out[col] = final_counts[:, j]

    df_out[data_config.other_col] = other

    M_row_flat = panel.M_row.reshape(-1)
    df_out = df_out[M_row_flat == 1].reset_index(drop=True)

    output_path = output_dir / "dataset_01_zinb.csv"
    df_out.to_csv(output_path, index=False)
    return output_path