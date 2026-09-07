"""Matplotlib/seaborn plotting for evaluation reports."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


def _ensure_output_dir(output_dir: str | Path) -> Path:
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_convergence_history(
    history: list[dict],
    output_path: str | Path,
    title: str = "GAN Training Convergence",
) -> Path | None:
    if not HAS_PLOTTING:
        return None
    output_path = _ensure_output_dir(Path(output_path).parent) / Path(output_path).name

    epochs = [h.get("epoch", i) for i, h in enumerate(history)]
    g_loss = [h.get("g_loss", np.nan) for h in history]
    d_loss = [h.get("d_loss", np.nan) for h in history]
    recon_loss = [h.get("recon_loss", np.nan) for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, g_loss, label="Generator", color="tab:blue")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Generator Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, d_loss, label="Discriminator", color="tab:red")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Discriminator Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, recon_loss, label="Reconstruction", color="tab:green")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].set_title("Reconstruction Loss")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_calibration_curve(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    output_path: str | Path,
    n_bins: int = 10,
    title: str = "Calibration Curve (Occurrence Gate)",
) -> Path | None:
    if not HAS_PLOTTING:
        return None
    output_path = _ensure_output_dir(Path(output_path).parent) / Path(output_path).name

    bin_edges = np.linspace(0, 1, n_bins + 1)
    centers = []
    observed = []
    predicted = []
    counts = []

    for i in range(n_bins):
        mask = (p_pred >= bin_edges[i]) & (p_pred < bin_edges[i + 1])
        if mask.sum() > 0:
            centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            observed.append(y_true[mask].mean())
            predicted.append(p_pred[mask].mean())
            counts.append(mask.sum())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(predicted, observed, "o-", color="tab:blue", label="Model")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Frequency")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_metric_comparison(
    stage_metrics: dict[str, dict],
    metric_keys: list[str],
    output_path: str | Path,
    title: str = "Metric Comparison Across Pipeline Stages",
) -> Path | None:
    if not HAS_PLOTTING:
        return None
    output_path = _ensure_output_dir(Path(output_path).parent) / Path(output_path).name

    stages = list(stage_metrics.keys())
    data = []
    for stage in stages:
        for key in metric_keys:
            val = stage_metrics[stage].get(key, np.nan)
            data.append({"Stage": stage, "Metric": key, "Value": val})

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x="Stage", y="Value", hue="Metric", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Value")
    ax.legend(title="Metric")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_jsd_by_variant(
    per_variant_jsd: dict[str, float],
    output_path: str | Path,
    title: str = "JSD by Variant",
    zero_prevalence: dict[str, bool] | None = None,
) -> Path | None:
    if not HAS_PLOTTING:
        return None
    output_path = _ensure_output_dir(Path(output_path).parent) / Path(output_path).name

    variants = list(per_variant_jsd.keys())
    values = list(per_variant_jsd.values())

    fig, ax = plt.subplots(figsize=(10, max(6, len(variants) * 0.35)))
    bars = ax.barh(range(len(variants)), values, color="tab:blue", alpha=0.7)
    
    # Highlight zero-prevalence variants
    if zero_prevalence:
        for i, variant in enumerate(variants):
            if zero_prevalence.get(variant, False):
                # Find the bar and recolor
                for bar in ax.patches:
                    if bar.get_y() == i:
                        bar.set_color("tab:gray")
                        bar.set_alpha(0.5)
                        break

    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels(variants)
    ax.set_xlabel("JSD")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="x")
    
    # Add legend for zero-prevalence
    if zero_prevalence and any(zero_prevalence.values()):
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="tab:blue", alpha=0.7, label="Has data"),
            Patch(facecolor="tab:gray", alpha=0.5, label="Zero prevalence (both)")
        ]
        ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_wasserstein_by_variant(
    per_variant_ws: dict[str, float],
    output_path: str | Path,
    title: str = "Wasserstein Distance by Variant",
    zero_prevalence: dict[str, bool] | None = None,
) -> Path | None:
    if not HAS_PLOTTING:
        return None
    output_path = _ensure_output_dir(Path(output_path).parent) / Path(output_path).name

    variants = list(per_variant_ws.keys())
    values = list(per_variant_ws.values())

    fig, ax = plt.subplots(figsize=(10, max(6, len(variants) * 0.35)))
    bars = ax.barh(range(len(variants)), values, color="tab:orange", alpha=0.7)
    
    # Highlight zero-prevalence variants
    if zero_prevalence:
        for i, variant in enumerate(variants):
            if zero_prevalence.get(variant, False):
                for bar in ax.patches:
                    if bar.get_y() == i:
                        bar.set_color("tab:gray")
                        bar.set_alpha(0.5)
                        break

    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels(variants)
    ax.set_xlabel("Wasserstein Distance")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="x")
    
    # Add legend for zero-prevalence
    if zero_prevalence and any(zero_prevalence.values()):
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="tab:orange", alpha=0.7, label="Has data"),
            Patch(facecolor="tab:gray", alpha=0.5, label="Zero prevalence (both)")
        ]
        ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_seed_variance(
    seed_results: dict[int, dict],
    metric_key: str,
    output_path: str | Path,
    title: str = "Seed Variance",
) -> Path | None:
    if not HAS_PLOTTING:
        return None
    output_path = _ensure_output_dir(Path(output_path).parent) / Path(output_path).name

    seeds = sorted(seed_results.keys())
    values = [seed_results[s].get(metric_key, np.nan) for s in seeds]
    mean_val = np.nanmean(values)
    std_val = np.nanstd(values)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(s) for s in seeds], values, color="tab:purple", alpha=0.7)
    ax.axhline(mean_val, color="red", linestyle="--", label=f"Mean={mean_val:.4f}")
    ax.axhspan(mean_val - std_val, mean_val + std_val, alpha=0.1, color="red", label=f"±1σ={std_val:.4f}")
    ax.set_xlabel("Seed")
    ax.set_ylabel(metric_key)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_compute_time(
    stage_times: dict[str, float],
    output_path: str | Path,
    title: str = "Compute Time by Stage",
) -> Path | None:
    if not HAS_PLOTTING:
        return None
    output_path = _ensure_output_dir(Path(output_path).parent) / Path(output_path).name

    stages = list(stage_times.keys())
    times = list(stage_times.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(stages, times, color="tab:cyan", alpha=0.7)
    ax.set_xlabel("Time (seconds)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="x")

    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f"{t:.1f}s", va="center")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_temporal_fidelity(
    temporal_metrics: dict[str, float],
    output_path: str | Path,
    title: str = "Temporal Fidelity Metrics",
) -> Path | None:
    if not HAS_PLOTTING:
        return None
    output_path = _ensure_output_dir(Path(output_path).parent) / Path(output_path).name

    keys = list(temporal_metrics.keys())
    values = list(temporal_metrics.values())

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(keys, values, color="tab:green", alpha=0.7)
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_distribution_comparison(
    raw_props: np.ndarray,
    imputed_props: np.ndarray,
    variant_names: list[str],
    output_path: str | Path,
    title: str = "Distribution Comparison: Raw vs Imputed (Primary Dataset)",
) -> Path | None:
    """Plot distribution comparison between raw and imputed data for primary dataset.

    Creates violin plots comparing raw vs imputed proportions per variant.
    """
    if not HAS_PLOTTING:
        return None
    output_path = _ensure_output_dir(Path(output_path).parent) / Path(output_path).name

    n_variants = len(variant_names)
    n_cols = min(4, n_variants)
    n_rows = (n_variants + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for i, variant in enumerate(variant_names):
        ax = axes[i]
        raw_data = raw_props[:, i]
        imp_data = imputed_props[:, i]

        # Filter out zeros for better visualization
        raw_nonzero = raw_data[raw_data > 0]
        imp_nonzero = imputed_props[:, i][imputed_props[:, i] > 0]

        # Violin plot
        if len(raw_nonzero) > 0 and len(imp_nonzero) > 0:
            parts = ax.violinplot([raw_nonzero, imp_nonzero], positions=[1, 2], showmeans=True)
            parts['bodies'][0].set_facecolor('tab:blue')
            parts['bodies'][0].set_alpha(0.6)
            parts['bodies'][1].set_facecolor('tab:orange')
            parts['bodies'][1].set_alpha(0.6)
        elif len(raw_nonzero) > 0:
            ax.violinplot([raw_nonzero], positions=[1], showmeans=True)
            parts = ax.violinplot([raw_nonzero], positions=[1], showmeans=True)
            parts['bodies'][0].set_facecolor('tab:blue')
            parts['bodies'][0].set_alpha(0.6)
        elif len(imp_nonzero) > 0:
            ax.violinplot([imp_nonzero], positions=[1], showmeans=True)
            parts = ax.violinplot([imp_nonzero], positions=[1], showmeans=True)
            parts['bodies'][0].set_facecolor('tab:orange')
            parts['bodies'][0].set_alpha(0.6)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Raw', 'Imputed'])
        ax.set_ylabel('Proportion')
        ax.set_title(variant, fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

    # Hide unused subplots
    for i in range(n_variants, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_all_plots(
    results: dict,
    output_dir: str | Path,
    raw_props: np.ndarray | None = None,
    imputed_props_dict: dict[str, np.ndarray] | None = None,
    variant_names: list[str] | None = None,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = {}

    if "gan_history" in results:
        p = plot_convergence_history(
            results["gan_history"],
            output_dir / "convergence.png",
        )
        if p:
            plot_paths["convergence"] = str(p)

    if "stage_metrics" in results:
        p = plot_metric_comparison(
            results["stage_metrics"],
            ["mse_prop", "jsd", "mae_prop"],
            output_dir / "metric_comparison.png",
        )
        if p:
            plot_paths["metric_comparison"] = str(p)

    if "per_variant_jsd" in results:
        # Compute zero prevalence from distribution fidelity if available
        zero_prevalence = None
        if "tsagris" in results and "distribution_fidelity" in results["tsagris"]:
            dist_fid = results["tsagris"]["distribution_fidelity"]
            if "per_variant" in dist_fid:
                zero_prevalence = {}
                for variant, vdata in dist_fid["per_variant"].items():
                    # Zero prevalence if both true and imputed prevalence are near zero
                    true_zero = vdata.get("true_zero_prev", 0)
                    imp_zero = vdata.get("imputed_zero_prev", 0)
                    zero_prevalence[variant] = (true_zero > 0.95 and imp_zero > 0.95)

        p = plot_jsd_by_variant(
            results["per_variant_jsd"],
            output_dir / "jsd_by_variant.png",
            zero_prevalence=zero_prevalence,
        )
        if p:
            plot_paths["jsd_by_variant"] = str(p)

    if "per_variant_wasserstein" in results:
        zero_prevalence = None
        if "tsagris" in results and "distribution_fidelity" in results["tsagris"]:
            dist_fid = results["tsagris"]["distribution_fidelity"]
            if "per_variant" in dist_fid:
                zero_prevalence = {}
                for variant, vdata in dist_fid["per_variant"].items():
                    true_zero = vdata.get("true_zero_prev", 0)
                    imp_zero = vdata.get("imputed_zero_prev", 0)
                    zero_prevalence[variant] = (true_zero > 0.95 and imp_zero > 0.95)

        p = plot_wasserstein_by_variant(
            results["per_variant_wasserstein"],
            output_dir / "wasserstein_by_variant.png",
            zero_prevalence=zero_prevalence,
        )
        if p:
            plot_paths["wasserstein_by_variant"] = str(p)

    if "stage_times" in results:
        p = plot_compute_time(
            results["stage_times"],
            output_dir / "compute_time.png",
        )
        if p:
            plot_paths["compute_time"] = str(p)

    if "temporal_metrics" in results:
        p = plot_temporal_fidelity(
            results["temporal_metrics"],
            output_dir / "temporal_fidelity.png",
        )
        if p:
            plot_paths["temporal_fidelity"] = str(p)

    # Distribution comparison plot (raw vs imputed for primary dataset)
    if raw_props is not None and imputed_props_dict is not None and variant_names is not None:
        for stage_name, imputed_props in imputed_props_dict.items():
            p = plot_distribution_comparison(
                raw_props,
                imputed_props,
                variant_names,
                output_dir / f"distribution_comparison_{stage_name}.png",
                title=f"Distribution Comparison: Raw vs {stage_name} (Primary Dataset)",
            )
            if p:
                plot_paths[f"distribution_comparison_{stage_name}"] = str(p)

    return plot_paths