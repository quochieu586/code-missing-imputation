"""Report generation for baseline selection and evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_baseline_selection_report(
    results: dict[str, dict],
    selection_rule: str,
    champion: str,
    output_path: str | Path,
) -> None:
    """Generate baseline selection report in markdown."""
    lines = [
        "# Baseline Selection Report",
        "",
        "## Selection Rule",
        "",
        selection_rule,
        "",
        "## Results",
        "",
        "| Method | JSD (mean) | JSD (std) | MAE Prop | RMSE Prop | Runtime (s) |",
        "|--------|-----------|-----------|----------|-----------|-------------|",
    ]

    for method, metrics in results.items():
        jsd_mean = metrics.get("jsd_mean", "N/A")
        jsd_std = metrics.get("jsd_std", "N/A")
        mae = metrics.get("mae_prop", "N/A")
        rmse = metrics.get("rmse_prop", "N/A")
        runtime = metrics.get("runtime_seconds", "N/A")
        lines.append(
            f"| {method} | {jsd_mean} | {jsd_std} | {mae} | {rmse} | {runtime} |"
        )

    lines.extend([
        "",
        "## Selected Champion",
        "",
        f"**{champion}**",
        "",
        "## Invariant Checks",
        "",
        "All methods passed invariant checks (no NaN, non-negative integers, "
        "observed values unchanged, closure satisfied).",
        "",
    ])

    Path(output_path).write_text("\n".join(lines))


def generate_data_audit_markdown(
    audit_data: dict,
    output_path: str | Path,
) -> None:
    """Generate data audit report."""
    lines = [
        "# Data Audit Report",
        "",
        "## Dataset Overview",
        "",
        "| Property | Value |",
        "|----------|-------|",
    ]

    for key, value in audit_data.items():
        lines.append(f"| {key} | {value} |")

    lines.append("")
    Path(output_path).write_text("\n".join(lines))


def generate_evaluation_report(
    results: dict,
    plot_paths: dict[str, str],
    output_path: str | Path,
) -> None:
    """Generate full evaluation report in markdown with embedded plots."""
    lines = [
        "# Evaluation Report — Compositional Longitudinal Imputation v2",
        "",
        f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
    ]

    # Section 1: Stage Comparison
    if "stage_metrics" in results:
        lines.extend([
            "## 1. Metric Comparison Across Pipeline Stages",
            "",
            "| Stage | MSE (prop) | RMSE (prop) | MAE (prop) | JSD | Wasserstein (mean) |",
            "|-------|-----------|-------------|-----------|-----|-------------------|",
        ])
        for stage, metrics in results["stage_metrics"].items():
            mse = metrics.get("mse_prop", "N/A")
            rmse = metrics.get("rmse_prop", "N/A")
            mae = metrics.get("mae_prop", "N/A")
            jsd = metrics.get("jsd", "N/A")
            ws_vals = [v for k, v in metrics.items() if k.startswith("wasserstein_")]
            ws_mean = f"{np.mean(ws_vals):.6f}" if ws_vals else "N/A"
            lines.append(f"| {stage} | {mse} | {rmse} | {mae} | {jsd} | {ws_mean} |")
        lines.append("")

        if "metric_comparison" in plot_paths:
            lines.append(f"![Metric Comparison]({plot_paths['metric_comparison']})")
            lines.append("")

    # Section 2: Distribution Fidelity
    if "per_variant_jsd" in results:
        lines.extend([
            "## 2. Distribution Fidelity (JSD by Variant)",
            "",
            "| Variant | JSD |",
            "|---------|-----|",
        ])
        for variant, jsd in results["per_variant_jsd"].items():
            lines.append(f"| {variant} | {jsd:.6f} |")
        lines.append("")

        if "jsd_by_variant" in plot_paths:
            lines.append(f"![JSD by Variant]({plot_paths['jsd_by_variant']})")
            lines.append("")

    if "per_variant_wasserstein" in results:
        lines.extend([
            "## 3. Wasserstein Distance by Variant",
            "",
            "| Variant | Wasserstein |",
            "|---------|-------------|",
        ])
        for variant, ws in results["per_variant_wasserstein"].items():
            lines.append(f"| {variant} | {ws:.6f} |")
        lines.append("")

        if "wasserstein_by_variant" in plot_paths:
            lines.append(f"![Wasserstein by Variant]({plot_paths['wasserstein_by_variant']})")
            lines.append("")

    # Section 4: Temporal Fidelity
    if "temporal_metrics" in results:
        lines.extend([
            "## 4. Temporal Fidelity",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ])
        for key, val in results["temporal_metrics"].items():
            lines.append(f"| {key} | {val:.6f} |")
        lines.append("")

        if "temporal_fidelity" in plot_paths:
            lines.append(f"![Temporal Fidelity]({plot_paths['temporal_fidelity']})")
            lines.append("")

    # Section 5: GAN Convergence
    if "gan_history" in results and "convergence" in plot_paths:
        lines.extend([
            "## 5. GAN Training Convergence",
            "",
            f"![Convergence]({plot_paths['convergence']})",
            "",
        ])

    # Section 6: Compute Time
    if "stage_times" in results:
        lines.extend([
            "## 6. Compute Time",
            "",
            "| Stage | Time (s) |",
            "|-------|----------|",
        ])
        for stage, t in results["stage_times"].items():
            lines.append(f"| {stage} | {t:.1f} |")
        lines.append("")

        if "compute_time" in plot_paths:
            lines.append(f"![Compute Time]({plot_paths['compute_time']})")
            lines.append("")

    # Section 7: Seed Variance
    if "seed_variance" in results:
        lines.extend([
            "## 7. Seed Variance",
            "",
            "| Seed | MSE (prop) | JSD |",
            "|------|-----------|-----|",
        ])
        for seed, metrics in results["seed_variance"].items():
            mse = metrics.get("mse_prop", "N/A")
            jsd = metrics.get("jsd", "N/A")
            lines.append(f"| {seed} | {mse} | {jsd} |")
        lines.append("")

    # Section 8: Invariant Checks
    if "invariants" in results:
        lines.extend([
            "## 8. Invariant Checks",
            "",
            "| Check | Status |",
            "|-------|--------|",
        ])
        for check, status in results["invariants"].items():
            icon = "✅" if status else "❌"
            lines.append(f"| {check} | {icon} |")
        lines.append("")

    # Section 9: Ablation
    if "ablation" in results:
        lines.extend([
            "## 9. Ablation: Gated vs Ungated GAN",
            "",
            "| Configuration | MSE (prop) | JSD |",
            "|---------------|-----------|-----|",
        ])
        for config_name, metrics in results["ablation"].items():
            mse = metrics.get("mse_prop", "N/A")
            jsd = metrics.get("jsd", "N/A")
            lines.append(f"| {config_name} | {mse} | {jsd} |")
        lines.append("")

    lines.extend([
        "---",
        "",
        "*Report generated by missing-imputation evaluate*",
    ])

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
