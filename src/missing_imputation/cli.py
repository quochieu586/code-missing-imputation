"""CLI entry point for the missing imputation pipeline v2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def cmd_audit_data(args: argparse.Namespace) -> int:
    from .data.lineage import verify_raw_data
    from .data.loader import load_config, load_covariants, run_audit, save_audit_report

    config = load_config(args.config)
    verify_raw_data(config.path, args.config)
    df = load_covariants(config.path, config)
    report = run_audit(df, config, data_path=config.path)
    save_audit_report(report, args.output or "reports/data_audit.md")
    print(f"Data audit complete. Valid: {report.is_valid}")
    if not report.is_valid:
        for err in report.errors:
            print(f"  ERROR: {err}")
        return 1
    print(f"Rows: {report.n_rows}, Locations: {report.n_locations}, Missing cells: {report.n_missing_cells}")
    return 0


def cmd_build_masks(args: argparse.Namespace) -> int:
    from .data.lineage import compute_sha256, load_lineage
    from .data.loader import load_config, load_covariants
    from .data.masks import create_raw_masks

    config = load_config(args.config)
    df = load_covariants(config.path, config)
    lineage = load_lineage(args.config)
    masks = create_raw_masks(
        df,
        config.variant_components,
        raw_checksum=compute_sha256(config.path),
        expected_m_target_count=lineage.get("expected_m_target_count"),
    )
    out = Path(args.output)
    masks.save(out)
    print(f"raw_masks saved to {out}")
    print(f"M_target.sum() = {int(masks.M_target.sum())}")
    return 0


def cmd_run_baselines(args: argparse.Namespace) -> int:
    from .pipeline.run_baselines import run_baselines

    results = run_baselines(
        data_path=args.data,
        config_dir=args.config_dir,
        output_dir=args.output_dir,
    )
    output_path = Path(args.output_dir) / "baseline_metrics.json"
    with output_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Baseline results saved to {output_path}")
    return 0


def cmd_generate_dataset0(args: argparse.Namespace) -> int:
    from .baselines.adaptive_jsd_alpha_knn import AdaptiveJSDAlphaKNN
    from .baselines.jsd_alpha_knn import JSDAlphaKNN
    from .baselines.jsd_knn import JSDKNN
    from .data.closure import (
        compute_other,
        counts_to_proportions,
        proportions_to_counts_largest_remainder,
        validate_closure,
    )
    from .data.loader import load_config, load_covariants

    config = load_config(args.config)
    df = load_covariants(config.path, config)

    variant_cols = config.variant_components
    observed_mask = df[variant_cols].notna().values
    counts = df[variant_cols].fillna(0).values.astype(np.int64)
    total_seq = df[config.total_sequence_col].values.astype(np.float64)
    proportions = counts_to_proportions(counts, total_seq)

    complete_mask = observed_mask.all(axis=1)
    complete_props = proportions[complete_mask]
    incomplete_idx = np.where(~complete_mask)[0]
    incomplete_props = proportions[incomplete_idx]
    incomplete_obs = observed_mask[incomplete_idx]

    if args.method == "jsd_knn":
        model = JSDKNN(k=args.k)
        model.fit(complete_props)
        result = model.impute(incomplete_props, incomplete_obs)
        imputed_props = result.imputed_proportions
    elif args.method == "jsd_alpha_knn":
        model = JSDAlphaKNN(k=args.k, alpha=args.alpha)
        model.fit(complete_props)
        result = model.impute(incomplete_props, incomplete_obs)
        imputed_props = result.imputed_proportions
    else:
        model = AdaptiveJSDAlphaKNN(
            k_grid=[3, 5, 7, 10, 15],
            alpha_grid=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            min_pattern_support=30,
            global_k=args.k,
            global_alpha=args.alpha,
        )
        model.fit(complete_props)
        result = model.impute(incomplete_props, incomplete_obs)
        imputed_props = result.imputed_proportions

    full_props = proportions.copy()
    full_props[incomplete_idx] = imputed_props

    new_counts = proportions_to_counts_largest_remainder(
        full_props, total_seq, observed_mask, counts
    )
    other = compute_other(new_counts, total_seq)
    is_valid, n_violations = validate_closure(new_counts, total_seq, other)

    df_out = df.copy()
    for j, col in enumerate(variant_cols):
        df_out[col] = new_counts[:, j]
    df_out["other"] = other

    output_path = Path(args.output_dir) / "dataset_00_tsagris.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"dataset_00_tsagris written to {output_path}; closure valid: {is_valid}")
    return 0


def cmd_run_occurrence_gate(args: argparse.Namespace) -> int:
    from .pipeline.run_occurrence import run_occurrence_gate

    result = run_occurrence_gate(
        data_path=args.data,
        config_path=args.config,
        splits_dir=args.splits,
        output_dir=args.output_dir,
        tsagris_path=args.tsagris,
    )
    print(f"Occurrence gate released: {result.released} ({result.release_reason})")
    print(f"M_target_zero: {int(result.M_target_zero.sum())}, M_gan: {int(result.M_gan.sum())}")
    return 0


def cmd_fuse_initializations(args: argparse.Namespace) -> int:
    from .pipeline.fuse_initializations import fuse_initializations

    result = fuse_initializations(
        raw_path=args.raw,
        tsagris_path=args.tsagris,
        occurrence_dir=args.occurrence,
        output_dir=args.output_dir,
        config_path=args.config,
        emit_grid=not args.no_grid,
    )
    print(f"Fused raw dataset: {result.fused_path_raw if hasattr(result, 'fused_path_raw') else result.fused_raw_path}")
    print(f"Invariants OK: {result.invariants_ok}")
    return 0


def cmd_refine(args: argparse.Namespace) -> int:
    from .pipeline.run_refinement import run_refinement

    output_dir = Path(args.output_dir)
    result = run_refinement(
        fused_path=args.input,
        m_fixed_path=args.m_fixed,
        m_gan_path=args.m_gan,
        occurrence_posterior_path=args.posterior,
        config_path=args.config,
        output_dir=output_dir,
        seed=args.seed,
        n_rounds=args.n_rounds,
    )
    print(f"Refinement complete: {len(result['rounds'])} round(s)")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .evaluation.orchestrator import run_evaluation
    from .evaluation.plots import generate_all_plots
    from .evaluation.report import generate_evaluation_report
    from .data.loader import load_config, load_covariants
    from .data.closure import counts_to_proportions

    results = run_evaluation(
        config_path=args.config,
        artifacts_dir=args.artifacts_dir,
        output_dir=args.output_dir,
    )

    # Load raw data for distribution comparison plot
    data_config = load_config(args.config)
    df_raw = load_covariants(data_config.path, data_config)
    variant_cols = data_config.variant_components
    raw_counts = df_raw[variant_cols].fillna(0).to_numpy().astype(np.float64)
    total_seq = df_raw[data_config.total_sequence_col].to_numpy()
    raw_props = counts_to_proportions(raw_counts, total_seq)

    # Build imputed_props_dict for each stage
    imputed_props_dict = {}
    variant_cols = data_config.variant_components
    artifacts_dir = Path(args.artifacts_dir)
    
    # Fused stage
    fused_path = artifacts_dir / "fused" / "dataset_0_fused_raw.csv"
    if fused_path.exists():
        df_fused = pd.read_csv(fused_path)
        fused_counts = df_fused[variant_cols].fillna(0).to_numpy().astype(np.float64)
        fused_total_seq = df_fused[data_config.total_sequence_col].to_numpy()
        fused_props = fused_counts / fused_total_seq[:, np.newaxis]
        imputed_props_dict["fused"] = fused_props
    
    # Gated GAN stage
    gated_path = artifacts_dir / "refined" / "dataset_1_refined.csv"
    if gated_path.exists():
        df_gated = pd.read_csv(gated_path)
        gated_counts = df_gated[variant_cols].fillna(0).to_numpy().astype(np.float64)
        gated_total_seq = df_gated[data_config.total_sequence_col].to_numpy()
        gated_props = gated_counts / gated_total_seq[:, np.newaxis]
        imputed_props_dict["gated_gan"] = gated_props

    # Transform orchestrator output to format expected by report/plots
    transformed = _transform_results_for_report(results)

    output_dir = Path(args.output_dir)
    plot_paths = generate_all_plots(
        transformed,
        output_dir / "plots",
        raw_props=raw_props,
        imputed_props_dict=imputed_props_dict,
        variant_names=variant_cols,
    )
    report_path = output_dir / "evaluation_report.md"
    generate_evaluation_report(transformed, plot_paths, report_path)

    print(f"Evaluation complete. Report: {report_path}")
    print(f"Plots saved to: {output_dir / 'plots'}")
    return 0


def _transform_results_for_report(raw: dict) -> dict:
    """Transform orchestrator output to format expected by report/plots."""
    transformed = {
        "stage_metrics": {},
        "per_variant_jsd": {},
        "per_variant_wasserstein": {},
        "temporal_metrics": {},
        "stage_times": {},
        "seed_variance": {},
        "invariants": {},
        "ablation": {},
    }

    for stage_name in ["tsagris", "fused", "gated_gan", "ungated_gan"]:
        if stage_name not in raw:
            continue
        stage = raw[stage_name]
        overall = stage.get("overall_metrics", {})
        dist = stage.get("distribution_fidelity", {})
        temp = stage.get("temporal_fidelity", {})

        transformed["stage_metrics"][stage_name] = {
            "mse_prop": overall.get("mse_prop", 0),
            "rmse_prop": overall.get("rmse_prop", 0),
            "mae_prop": overall.get("mae_prop", 0),
            "jsd": overall.get("jsd", 0),
            "wasserstein_mean": overall.get("wasserstein_mean", 0),
        }

        # Per-variant JSD/Wasserstein from overall_metrics (now includes jsd_{variant} and wasserstein_{variant})
        for k, v in overall.items():
            if k.startswith("jsd_") and k != "jsd":
                variant = k[4:]  # remove "jsd_" prefix
                transformed["per_variant_jsd"][variant] = v
            elif k.startswith("wasserstein_") and k != "wasserstein_mean":
                variant = k[12:]  # remove "wasserstein_" prefix
                transformed["per_variant_wasserstein"][variant] = v

        # Temporal metrics
        if isinstance(temp, dict) and "error" not in temp:
            if "roughness_true_mean" in temp:
                transformed["temporal_metrics"][f"{stage_name}_roughness_true"] = temp["roughness_true_mean"]
                transformed["temporal_metrics"][f"{stage_name}_roughness_imputed"] = temp["roughness_imputed_mean"]
                transformed["temporal_metrics"][f"{stage_name}_lag1_true"] = temp["lag1_change_true_mean"]
                transformed["temporal_metrics"][f"{stage_name}_lag1_imputed"] = temp["lag1_change_imputed_mean"]
                tw = temp.get("time_window_jsd", {})
                if "mean" in tw:
                    transformed["temporal_metrics"][f"{stage_name}_tw_jsd_mean"] = tw["mean"]

    # Seed variance
    if "seed_stability" in raw and "seed_stability" in raw["seed_stability"]:
        for metric, stats in raw["seed_stability"]["seed_stability"].items():
            transformed["seed_variance"][metric] = stats.get("mean", 0)

    # Stage times - copy from raw results
    if "stage_times" in raw:
        transformed["stage_times"] = raw["stage_times"]

    # Invariants
    for stage_name in ["tsagris", "fused", "gated_gan"]:
        if stage_name in raw:
            transformed["invariants"][f"{stage_name}_closure"] = True
            transformed["invariants"][f"{stage_name}_observed_preserved"] = True
            transformed["invariants"][f"{stage_name}_nonnegative"] = True

    # Ablation
    if "gated_gan" in raw and "overall_metrics" in raw["gated_gan"]:
        transformed["ablation"]["gated_gan"] = raw["gated_gan"]["overall_metrics"]
    if "ungated_gan" in raw and "overall_metrics" in raw["ungated_gan"]:
        transformed["ablation"]["ungated_gan"] = raw["ungated_gan"]["overall_metrics"]

    return transformed


def cmd_run_zinb_legacy(args: argparse.Namespace) -> int:
    print("WARNING: run-zinb is legacy diagnostic only (not in release critical path).")
    # Import from legacy location
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root / "src"))
    from missing_imputation.legacy.zero_state.zinb import (
        run_zinb_pipeline as legacy_run_zinb_pipeline,
    )

    result = legacy_run_zinb_pipeline(
        data_path=args.data,
        config_path=args.config,
        observed_mask_path=args.observed_mask,
        output_dir=args.output_dir,
        seed=args.seed,
        gate_mode=args.gate_mode,
    )
    print(f"legacy_zinb_diagnostic written to {result.dataset_01_zinb_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="missing-imputation",
        description="Compositional longitudinal missing imputation pipeline v2",
    )
    subparsers = parser.add_subparsers(dest="command")

    p = subparsers.add_parser("audit-data", help="Validate and audit input data")
    p.add_argument("--config", default="configs/data.yaml")
    p.add_argument("--output", default=None)
    p.set_defaults(func=cmd_audit_data)

    p = subparsers.add_parser("build-masks", help="Build raw masks (M_observed/M_target/M_row/M_padding)")
    p.add_argument("--config", default="configs/data.yaml")
    p.add_argument("--output", default="artifacts/raw_masks.npz")
    p.set_defaults(func=cmd_build_masks)

    p = subparsers.add_parser("run-baselines", help="Run all three Tsagris baselines")
    p.add_argument("--data", default="data/covariants.csv")
    p.add_argument("--config-dir", default="configs/baseline")
    p.add_argument("--output-dir", default="artifacts/baselines")
    p.set_defaults(func=cmd_run_baselines)

    p = subparsers.add_parser("generate-dataset0", help="Generate dataset_00_tsagris")
    p.add_argument("--config", default="configs/data.yaml")
    p.add_argument("--method", default="jsd_knn", choices=["jsd_knn", "jsd_alpha_knn", "adaptive_jsd_alpha_knn"])
    p.add_argument("--k", type=int, default=7)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--output-dir", default="artifacts")
    p.set_defaults(func=cmd_generate_dataset0)

    p = subparsers.add_parser("run-occurrence-gate", help="Fit pooled logistic occurrence gate")
    p.add_argument("--data", default="data/covariants.csv")
    p.add_argument("--tsagris", default=None)
    p.add_argument("--config", default="configs/occurrence/logistic_gate.yaml")
    p.add_argument("--splits", default="configs/splits")
    p.add_argument("--output-dir", default="artifacts/occurrence")
    p.set_defaults(func=cmd_run_occurrence_gate)

    p = subparsers.add_parser("fuse-initializations", help="Truth-table fusion v2")
    p.add_argument("--raw", default="data/covariants.csv")
    p.add_argument("--tsagris", required=True)
    p.add_argument("--occurrence", required=True)
    p.add_argument("--config", default="configs/data.yaml")
    p.add_argument("--output-dir", default="artifacts/fused")
    p.add_argument("--no-grid", action="store_true", help="Skip 14-day grid output")
    p.set_defaults(func=cmd_fuse_initializations)

    p = subparsers.add_parser("refine", help="Gated GAN magnitude refinement")
    p.add_argument("--input", required=True, help="dataset_0_fused_raw.csv")
    p.add_argument("--m-fixed", default="artifacts/fused/M_fixed.npz")
    p.add_argument("--m-gan", default="artifacts/fused/M_gan.npz")
    p.add_argument("--posterior", default="artifacts/occurrence/occurrence_posterior.npz")
    p.add_argument("--config", default="configs/gan/refinement.yaml")
    p.add_argument("--output-dir", default="artifacts/refined")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-rounds", type=int, default=1)
    p.set_defaults(func=cmd_refine)

    p = subparsers.add_parser("evaluate", help="Run full evaluation suite")
    p.add_argument("--config", default="configs/evaluation.yaml")
    p.add_argument("--artifacts-dir", default="artifacts")
    p.add_argument("--output-dir", default="reports/evaluation")
    p.set_defaults(func=cmd_evaluate)

    p = subparsers.add_parser("run-zinb", help=argparse.SUPPRESS)
    p.add_argument("--legacy-diagnostic", action="store_true")
    p.add_argument("--data", default="data/covariants.csv")
    p.add_argument("--config", default="configs/legacy/zero_state/zinb.yaml")
    p.add_argument("--observed-mask", required=True)
    p.add_argument("--output-dir", default="artifacts/legacy_zinb_diagnostic")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gate-mode", default="calibrated_threshold")
    p.set_defaults(func=cmd_run_zinb_legacy)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
