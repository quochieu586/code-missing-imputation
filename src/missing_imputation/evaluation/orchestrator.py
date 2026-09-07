"""Evaluation orchestrator: coordinates all evaluation stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..data.closure import counts_to_proportions
from ..data.loader import load_config, load_covariants
from .masking import EvaluationMask, create_all_evaluation_masks, evaluate_with_masks
from .metrics import (
    compute_all_metrics,
    count_metrics_by_bin,
    distribution_fidelity,
    jsd_metric,
    mae_proportions,
    mse_proportions,
    rmse_proportions,
    seed_stability,
    temporal_fidelity,
)


@dataclass
class StageResults:
    """Results for one pipeline stage."""
    stage_name: str
    metrics: dict
    per_variant_metrics: dict = field(default_factory=dict)
    temporal_metrics: dict = field(default_factory=dict)
    binned_metrics: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    primary_metric: str = "mse_proportions"
    secondary_metrics: list[str] = field(default_factory=lambda: ["jsd", "wasserstein", "rmse_proportions", "mae_proportions"])
    temporal_metrics: list[str] = field(default_factory=lambda: ["roughness", "lag1_change", "time_window_jsd"])
    n_bins: int = 5
    recipes: list[str] = field(default_factory=lambda: ["random-cell", "empirical-pattern", "time-block", "country-holdout"])
    test_fraction: float = 0.2
    use_same_splits: bool = True
    seeds: list[int] = field(default_factory=lambda: [42, 123, 456])
    ablation_compare: list[str] = field(default_factory=lambda: ["tsagris", "occurrence_gate", "fused", "ungated_gan", "gated_gan"])

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvaluationConfig:
        with Path(path).open() as f:
            cfg = yaml.safe_load(f)
        raw = cfg.get("evaluation", {})
        if "primary_metric" not in raw and isinstance(raw.get("metrics"), dict):
            raw["primary_metric"] = raw["metrics"].get("primary", "mse_proportions")
        if "secondary_metrics" not in raw and isinstance(raw.get("metrics"), dict):
            raw["secondary_metrics"] = raw["metrics"].get("secondary", [])
        if "n_bins" not in raw and isinstance(raw.get("bins"), dict):
            edges = raw["bins"].get("total_sequence_edges")
            if edges:
                raw["n_bins"] = max(len(edges) - 1, 1)
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in raw.items() if k in valid_fields}
        return cls(**filtered)


class EvaluationOrchestrator:
    """Orchestrates full pipeline evaluation with artificial masking."""

    def __init__(self, config: EvaluationConfig, artifacts_dir: str | Path):
        self.config = config
        self.artifacts_dir = Path(artifacts_dir)
        self.variant_cols = [
            "recombinant", "20A", "20B", "20C", "20E", "Beta", "Alpha", "Gamma",
            "Delta", "Kappa", "Epsilon", "Eta", "Iota", "Lambda", "Mu", "Omicron", "S:677",
        ]
        # Total sequence column name - will be set from data config
        self.total_seq_col = "total_sequence"

    def run_full_evaluation(self) -> dict:
        """Run complete evaluation across all pipeline stages."""
        import time
        stage_times = {}
        
        # Load raw data
        data_config = load_config("configs/data.yaml")
        self.total_seq_col = data_config.total_sequence_col
        df_raw = load_covariants(data_config.path, data_config)
        observed_mask = df_raw[self.variant_cols].notna().to_numpy()
        raw_counts = df_raw[self.variant_cols].fillna(0).to_numpy().astype(np.float64)
        total_seq = df_raw[data_config.total_sequence_col].to_numpy()
        raw_props = counts_to_proportions(raw_counts, total_seq)

        # Create evaluation masks (same for all stages) - use pre-computed split IDs
        splits_dir = Path("configs/splits")
        eval_masks = create_all_evaluation_masks(
            observed_mask, raw_counts, total_seq, df_raw,
            location_col=data_config.location_col,
            date_col=data_config.date_col,
            test_fraction=self.config.test_fraction,
            splits_dir=splits_dir,
        )

        # Load artifacts
        tsagris_path = self.artifacts_dir / "dataset_00_tsagris.csv"
        fused_path = self.artifacts_dir / "fused" / "dataset_0_fused_raw.csv"
        gated_path = self.artifacts_dir / "refined" / "dataset_1_refined.csv"

        results = {}
        results["stage_times"] = stage_times

        # 1. Evaluate Tsagris (dataset_00_tsagris)
        if tsagris_path.exists():
            start = time.time()
            results["tsagris"] = self._evaluate_stage(
                "tsagris", tsagris_path, df_raw, raw_props, raw_counts, total_seq,
                eval_masks, observed_mask
            )
            stage_times["tsagris"] = time.time() - start

        # 2. Evaluate Occurrence Gate (OOF metrics already in artifacts)
        occ_dir = self.artifacts_dir / "occurrence"
        if occ_dir.exists():
            start = time.time()
            results["occurrence_gate"] = self._evaluate_occurrence_gate(occ_dir)
            stage_times["occurrence_gate"] = time.time() - start

        # 3. Evaluate Fused (dataset_0_fused)
        if fused_path.exists():
            start = time.time()
            results["fused"] = self._evaluate_stage(
                "fused", fused_path, df_raw, raw_props, raw_counts, total_seq,
                eval_masks, observed_mask
            )
            stage_times["fused"] = time.time() - start

        # 4. Evaluate Gated GAN (dataset_1_refined)
        if gated_path.exists():
            start = time.time()
            results["gated_gan"] = self._evaluate_stage(
                "gated_gan", gated_path, df_raw, raw_props, raw_counts, total_seq,
                eval_masks, observed_mask
            )
            stage_times["gated_gan"] = time.time() - start

        # 5. Ablation: Ungated GAN (if available)
        ungated_dir = self.artifacts_dir / "refined_ungated"
        if ungated_dir.exists():
            ungated_path = ungated_dir / "dataset_1_refined.csv"
            if ungated_path.exists():
                start = time.time()
                results["ungated_gan"] = self._evaluate_stage(
                    "ungated_gan", ungated_path, df_raw, raw_props, raw_counts, total_seq,
                    eval_masks, observed_mask
                )
                stage_times["ungated_gan"] = time.time() - start

        # 6. Seed stability for GAN (if multiple seeds run)
        if len(self.config.seeds) > 1:
            start = time.time()
            results["seed_stability"] = self._evaluate_seed_stability()
            stage_times["seed_stability"] = time.time() - start

        return results

    def _evaluate_stage(
        self,
        stage_name: str,
        stage_path: Path,
        df_raw: pd.DataFrame,
        raw_props: np.ndarray,
        raw_counts: np.ndarray,
        total_seq: np.ndarray,
        eval_masks: dict[str, EvaluationMask],
        observed_mask: np.ndarray,
    ) -> dict:
        """Evaluate one pipeline stage against artificial masks."""
        df_stage = pd.read_csv(stage_path)
        stage_counts = df_stage[self.variant_cols].fillna(0).to_numpy().astype(np.float64)
        # Use stage's own total_sequence for proportions (may differ from raw)
        stage_total_seq = df_stage[self.total_seq_col].to_numpy()
        stage_props = counts_to_proportions(stage_counts, stage_total_seq)

        # Overall metrics on all cells
        overall_metrics = compute_all_metrics(
            raw_props, stage_props, raw_counts, stage_counts, self.variant_cols
        )

        # Per-recipe evaluation on artificial masked cells
        per_recipe = {}
        for recipe, mask in eval_masks.items():
            recipe_metrics = {}
            for metric_name, metric_fn in [
                ("mse", lambda t, p: mse_proportions(t, p)),
                ("mae", lambda t, p: mae_proportions(t, p)),
                ("rmse", lambda t, p: rmse_proportions(t, p)),
                ("jsd", lambda t, p: jsd_metric(t, p)),
            ]:
                recipe_metrics[metric_name] = evaluate_with_masks(
                    raw_props, stage_props, mask, metric_fn
                )
            per_recipe[recipe] = recipe_metrics

        # Distribution fidelity
        dist_fid = distribution_fidelity(raw_props, stage_props, self.variant_cols)

        # Temporal fidelity (need panel format)
        temporal = self._compute_temporal_fidelity(df_raw, df_stage, total_seq)

        # Count metrics by total_sequence bin
        binned = count_metrics_by_bin(raw_counts, stage_counts, total_seq, self.config.n_bins)

        return {
            "stage": stage_name,
            "overall_metrics": overall_metrics,
            "per_recipe": per_recipe,
            "distribution_fidelity": dist_fid,
            "temporal_fidelity": temporal,
            "binned_metrics": binned,
            "artifacts": {"input_path": str(stage_path)},
        }

    def _evaluate_occurrence_gate(self, occ_dir: Path) -> dict:
        """Evaluate occurrence gate using saved OOF predictions."""
        oof_path = occ_dir / "occurrence_oof_predictions.parquet"
        metrics_path = occ_dir / "occurrence_metrics.csv"
        thresholds_path = occ_dir / "occurrence_thresholds.csv"

        results = {"stage": "occurrence_gate"}

        if oof_path.exists():
            oof_df = pd.read_parquet(oof_path)
            # Aggregate by recipe
            for recipe in oof_df["recipe"].unique():
                sub = oof_df[oof_df["recipe"] == recipe]
                results[f"oof_{recipe}"] = {
                    "log_loss": float(-(sub["y_true"] * np.log(sub["p_calibrated"] + 1e-12) +
                                        (1 - sub["y_true"]) * np.log(1 - sub["p_calibrated"] + 1e-12)).mean()),
                    "brier": float(((sub["p_calibrated"] - sub["y_true"]) ** 2).mean()),
                    "ece": float(self._compute_ece(sub["y_true"], sub["p_calibrated"])),
                    "n_samples": len(sub),
                }

        if metrics_path.exists():
            df_metrics = pd.read_csv(metrics_path)
            # Filter out rows with NaN in key columns
            df_metrics = df_metrics.dropna(subset=["log_loss", "brier", "ece"])
            results["summary_metrics"] = df_metrics.to_dict("records")

        if thresholds_path.exists():
            df_thresholds = pd.read_csv(thresholds_path)
            # Filter out rows with NaN in key columns
            df_thresholds = df_thresholds.dropna(subset=["tau_zero", "status"])
            results["thresholds"] = df_thresholds.to_dict("records")

        return results

    def _compute_ece(self, y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10) -> float:
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n = len(y_true)
        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            mask = (p_pred >= lo) & (p_pred < hi) if i < n_bins - 1 else (p_pred >= lo) & (p_pred <= hi)
            if mask.sum() == 0:
                continue
            bin_acc = y_true[mask].mean()
            bin_conf = p_pred[mask].mean()
            ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
        return ece

    def _compute_temporal_fidelity(self, df_raw: pd.DataFrame, df_stage: pd.DataFrame, total_seq: np.ndarray) -> dict:
        """Compute temporal fidelity in panel format."""
        from ..data.panel import build_panel

        data_config = load_config("configs/data.yaml")
        panel_raw = build_panel(df_raw, data_config, self.variant_cols, freq_days=14)
        panel_stage = build_panel(df_stage, data_config, self.variant_cols, freq_days=14)

        # Only evaluate on original rows (M_row == 1)
        m_row = panel_raw.M_row.astype(bool)

        true_props = panel_raw.X[m_row]
        imputed_props = panel_stage.X[m_row]

        # Need same location-time structure
        if true_props.shape != imputed_props.shape:
            return {"error": "Panel shape mismatch"}

        # Reshape for temporal_fidelity: (n_loc, n_time, d) -> list of (n_time, d)
        n_loc = panel_raw.n_locations
        n_time = panel_raw.n_times
        d = panel_raw.n_features

        true_panel = np.zeros((n_loc, n_time, d))
        imputed_panel = np.zeros((n_loc, n_time, d))
        true_panel[m_row] = true_props
        imputed_panel[m_row] = imputed_props

        return temporal_fidelity(true_panel, imputed_panel)

    def _evaluate_seed_stability(self) -> dict:
        """Evaluate variance across seeds for GAN refinement."""
        seed_results = []
        for seed in self.config.seeds:
            seed_dir = self.artifacts_dir / f"refined_seed_{seed}"
            if not seed_dir.exists():
                continue
            path = seed_dir / "dataset_1_refined.csv"
            if not path.exists():
                continue

            data_config = load_config("configs/data.yaml")
            df_raw = load_covariants(data_config.path, data_config)
            df_seed = pd.read_csv(path)
            seed_counts = df_seed[self.variant_cols].fillna(0).to_numpy().astype(np.float64)
            total_seq = df_raw[data_config.total_sequence_col].to_numpy()
            seed_props = counts_to_proportions(seed_counts, total_seq)
            raw_counts = df_raw[self.variant_cols].fillna(0).to_numpy().astype(np.float64)
            raw_props = counts_to_proportions(raw_counts, total_seq)

            metrics = compute_all_metrics(raw_props, seed_props, raw_counts, seed_counts, self.variant_cols)
            seed_results.append(metrics)

        return {"seed_stability": seed_stability(seed_results)}


def run_evaluation(
    config_path: str | Path,
    artifacts_dir: str | Path,
    output_dir: str | Path,
) -> dict:
    """Main entry point for evaluation."""
    config = EvaluationConfig.from_yaml(config_path)
    orchestrator = EvaluationOrchestrator(config, artifacts_dir)
    results = orchestrator.run_full_evaluation()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results as YAML
    import json
    with (output_dir / "evaluation_results.json").open("w") as f:
        json.dump(results, f, indent=2, default=str)

    return results
