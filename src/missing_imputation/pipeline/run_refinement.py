"""Gated GAN refinement: dataset_i -> dataset_(i+1)."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..data.closure import compute_other, validate_closure
from ..tracking.manifest import create_run_manifest


def run_refinement(
    fused_path: str | Path,
    m_fixed_path: str | Path,
    m_gan_path: str | Path,
    occurrence_posterior_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    seed: int | None = None,
    n_rounds: int | None = None,
    w_gan_path: str | Path | None = None,
    data_config_path: str | Path = "configs/data.yaml",
    apply_soft_weights: bool = True,
) -> dict:
    """Run gated GAN refinement. seed and n_rounds fall back to the YAML config.

    apply_soft_weights=False skips the ZPGF weighting for the gated-vs-ungated
    ablation of plan S12.
    """
    project_root = Path(__file__).resolve().parents[3]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    with config_path.open(encoding="utf-8") as f:
        gan_cfg = yaml.safe_load(f).get("gan", {})

    # The YAML nests these sections; reading them flat off gan_cfg silently
    # discarded every configured value and always used the code defaults.
    arch_cfg = gan_cfg.get("architecture", {})
    training_cfg = gan_cfg.get("training", {})
    repr_cfg = gan_cfg.get("representation", {})
    refinement_cfg = gan_cfg.get("refinement", {})

    if seed is None:
        cfg_seeds = gan_cfg.get("seeds") or []
        seed = int(cfg_seeds[0]) if cfg_seeds else 42
    if n_rounds is None:
        n_rounds = int(refinement_cfg.get("n_rounds", 1))

    df = pd.read_csv(fused_path)
    df["date"] = pd.to_datetime(df["date"])
    M_fixed = np.load(m_fixed_path)["M_fixed"]
    M_gan = np.load(m_gan_path)["M_gan"]

    posterior = np.load(occurrence_posterior_path)
    p_nonzero = posterior["p_nonzero"]
    target_rows = posterior["target_row_indices"]
    target_vars = posterior["target_variant_indices"]

    p_nonzero_grid = np.zeros(M_gan.shape, dtype=np.float64)
    p_nonzero_grid[target_rows, target_vars] = p_nonzero

    # ZPGF soft weights from fusion. Absent -> weight 1.0, i.e. plain gated GAN.
    w_gan_grid = None
    if w_gan_path is not None and Path(w_gan_path).exists():
        w_gan_grid = np.load(w_gan_path)["w_gan"]
        if w_gan_grid.shape != M_gan.shape:
            raise ValueError(
                f"w_gan shape {w_gan_grid.shape} does not match M_gan {M_gan.shape}"
            )

    data_config_path = Path(data_config_path)
    if not data_config_path.is_absolute():
        data_config_path = project_root / data_config_path
    from ..data.loader import load_config

    variant_cols = load_config(data_config_path).variant_components
    feature_cols = variant_cols + ["other"]
    if M_gan.shape[1] != len(variant_cols):
        raise ValueError(
            f"M_gan has {M_gan.shape[1]} variant columns, config declares {len(variant_cols)}"
        )

    from ..gan.data import build_gan_panel
    from ..gan.postprocess import postprocess_gan_output
    from ..gan.train import TrainConfig, train_gan

    train_config = TrainConfig(
        epochs=int(training_cfg.get("epochs", 300)),
        lr=float(training_cfg.get("lr", 0.001)),
        d_steps=int(training_cfg.get("d_steps", 5)),
        g_steps=int(training_cfg.get("g_steps", 1)),
        patience=int(training_cfg.get("patience", 50)),
        artificial_mask_fraction=float(training_cfg.get("artificial_mask_fraction", 0.2)),
        seed=seed,
        device=gan_cfg.get("device", "cpu"),
        consistency_weight=float(training_cfg.get("consistency_weight", 1.0)),
        time_class_weight=float(training_cfg.get("time_class_weight", 1.0)),
        grad_clip=float(training_cfg.get("grad_clip", 1.0)),
        cnn_hidden1=int(arch_cfg.get("cnn_hidden1", 16)),
        cnn_hidden2=int(arch_cfg.get("cnn_hidden2", 8)),
        rnn_hidden=int(arch_cfg.get("rnn_hidden", 10)),
        lstm_hidden=int(arch_cfg.get("lstm_hidden", 10)),
    )

    results = []
    total_seq_flat = df["total_sequence"].to_numpy(dtype=np.int64)
    df_current = df

    for round_i in range(n_rounds):
        # Rebuild the panel from the previous round's output so rounds actually
        # chain dataset_i -> dataset_(i+1); reusing one panel would just retrain
        # on the same input and, with a fixed seed, reproduce the same result.
        dataset = build_gan_panel(
            df_current,
            M_fixed,
            M_gan,
            p_nonzero_rows=p_nonzero_grid,
            w_gan_rows=w_gan_grid,
            variant_cols=variant_cols,
            pseudo_count=repr_cfg.get("pseudo_count"),
        )
        round_config = replace(train_config, seed=train_config.seed + round_i)

        final_clr, train_info = train_gan(dataset, round_config)
        refined_counts = postprocess_gan_output(
            final_clr, dataset, apply_soft_weights=apply_soft_weights
        )

        # The residual "other" is expected to move as the variants change, so the
        # immutability check covers the locked variant cells only.
        fixed_mask_bool = dataset.M_fixed.astype(bool).copy()
        fixed_mask_bool[..., -1] = False
        invariants_ok = bool(
            np.array_equal(dataset.counts_raw[fixed_mask_bool], refined_counts[fixed_mask_bool])
        )

        n_locs, n_times, n_feats = refined_counts.shape
        flat_counts = np.zeros((len(df_current), n_feats), dtype=np.int64)
        for loc_i in range(n_locs):
            for t_i in range(n_times):
                row = int(dataset.row_index[loc_i, t_i])
                if row >= 0:
                    flat_counts[row] = refined_counts[loc_i, t_i]

        # "other" is the residual, recomputed from the variant columns.
        other = compute_other(flat_counts[:, :-1], total_seq_flat)
        closure_ok, n_viol = validate_closure(flat_counts[:, :-1], total_seq_flat, other)
        invariants_ok = invariants_ok and closure_ok

        df_round = df_current.copy()
        for j, col in enumerate(feature_cols):
            df_round[col] = flat_counts[:, j]
        df_round["other"] = other
        round_path = output_dir / f"dataset_{round_i + 1}_refined.csv"
        df_round.to_csv(round_path, index=False)
        df_current = df_round

        n_changed = int(
            (flat_counts[:, :-1] != df[variant_cols].to_numpy()).sum()
        )
        checkpoint = {
            "round": round_i + 1,
            "seed": round_config.seed,
            "best_recon": train_info["best_recon"],
            "epochs_run": train_info["epochs_run"],
            "invariants_ok": invariants_ok,
            "closure_violations": n_viol,
            "n_gan_cells": int(M_gan.sum()),
            "n_cells_changed_vs_fused": n_changed,
            "soft_weights_applied": apply_soft_weights,
            "output": str(round_path),
        }
        ckpt_path = output_dir / f"checkpoint_round_{round_i + 1}.json"
        with ckpt_path.open("w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2)

        results.append(checkpoint)

    manifest = create_run_manifest(
        run_id=f"refinement_{int(time.time())}",
        method="gated_gan_refinement_v2",
        params={
            "seed": seed,
            "n_rounds": n_rounds,
            "config": str(config_path),
            "train_config": asdict(train_config),
            "apply_soft_weights": apply_soft_weights,
            "w_gan_path": str(w_gan_path) if w_gan_path else None,
        },
        metrics={"rounds": results},
        data_path=fused_path,
        config_paths=[config_path],
        seeds=[seed],
        invariants_ok=all(r["invariants_ok"] for r in results),
        output_dir=output_dir / "manifests",
    )

    return {"rounds": results, "manifest": manifest}