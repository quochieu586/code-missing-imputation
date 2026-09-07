"""Gated GAN refinement: dataset_i -> dataset_(i+1)."""

from __future__ import annotations

import json
import time
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
    seed: int = 42,
    n_rounds: int = 1,
) -> dict:
    project_root = Path(__file__).resolve().parents[3]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    with config_path.open(encoding="utf-8") as f:
        gan_cfg = yaml.safe_load(f).get("gan", {})

    df = pd.read_csv(fused_path)
    df["date"] = pd.to_datetime(df["date"])
    M_fixed = np.load(m_fixed_path)["M_fixed"]
    M_gan = np.load(m_gan_path)["M_gan"]

    posterior = np.load(occurrence_posterior_path)
    p_nonzero = posterior["p_nonzero"]
    target_rows = posterior["target_row_indices"]
    target_vars = posterior["target_variant_indices"]

    p_nonzero_grid = np.zeros(M_gan.shape, dtype=np.float64)
    for k in range(len(target_rows)):
        p_nonzero_grid[target_rows[k], target_vars[k]] = p_nonzero[k]

    variant_cols = [
        "recombinant", "20A", "20B", "20C", "20E", "Beta", "Alpha", "Gamma",
        "Delta", "Kappa", "Epsilon", "Eta", "Iota", "Lambda", "Mu", "Omicron", "S:677",
    ]
    feature_cols = variant_cols + ["other"]

    from ..gan.data import build_gan_panel
    from ..gan.postprocess import postprocess_gan_output
    from ..gan.train import TrainConfig, train_gan

    dataset = build_gan_panel(
        df, M_fixed, M_gan, p_nonzero_flat=p_nonzero_grid.ravel(),
        variant_cols=variant_cols,
        pseudo_count=gan_cfg.get("pseudo_count"),
    )

    train_config = TrainConfig(
        epochs=gan_cfg.get("epochs", 300),
        lr=gan_cfg.get("lr", 0.001),
        d_steps=gan_cfg.get("d_steps", 5),
        g_steps=gan_cfg.get("g_steps", 1),
        patience=gan_cfg.get("patience", 50),
        artificial_mask_fraction=gan_cfg.get("artificial_mask_fraction", 0.2),
        seed=seed,
        device=gan_cfg.get("device", "cpu"),
    )

    results = []

    for round_i in range(n_rounds):
        final_clr, train_info = train_gan(dataset, train_config)
        refined_counts = postprocess_gan_output(final_clr, dataset)

        fixed_mask_bool = dataset.M_fixed.astype(bool)
        observed_before = dataset.counts_raw[fixed_mask_bool].copy()
        observed_after = refined_counts[fixed_mask_bool]
        invariants_ok = bool(np.array_equal(observed_before, observed_after))

        n_locs, n_times, n_feats = refined_counts.shape
        flat_counts = np.zeros((len(df), n_feats), dtype=np.int64)
        flat_idx = 0
        for loc_i in range(n_locs):
            for t_i in range(n_times):
                if dataset.M_row[loc_i, t_i] == 1:
                    flat_counts[flat_idx] = refined_counts[loc_i, t_i]
                    flat_idx += 1

        total_seq_flat = df["total_sequence"].to_numpy(dtype=np.int64)
        # "other" is the last column in flat_counts
        other = compute_other(flat_counts[:, :-1], total_seq_flat)
        closure_ok, n_viol = validate_closure(flat_counts[:, :-1], total_seq_flat, other)
        invariants_ok = invariants_ok and closure_ok

        df_round = df.copy()
        for j, col in enumerate(feature_cols):
            df_round[col] = flat_counts[:, j]
        df_round["other"] = other
        round_path = output_dir / f"dataset_{round_i + 1}_refined.csv"
        df_round.to_csv(round_path, index=False)

        checkpoint = {
            "round": round_i + 1,
            "seed": seed,
            "best_recon": train_info["best_recon"],
            "epochs_run": train_info["epochs_run"],
            "invariants_ok": invariants_ok,
            "closure_violations": n_viol,
            "output": str(round_path),
        }
        ckpt_path = output_dir / f"checkpoint_round_{round_i + 1}.json"
        with ckpt_path.open("w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2)

        results.append(checkpoint)

    manifest = create_run_manifest(
        run_id=f"refinement_{int(time.time())}",
        method="gated_gan_refinement_v2",
        params={"seed": seed, "n_rounds": n_rounds, "config": str(config_path)},
        metrics={"rounds": results},
        data_path=fused_path,
        config_paths=[config_path],
        seeds=[seed],
        invariants_ok=all(r["invariants_ok"] for r in results),
        output_dir=output_dir / "manifests",
    )

    return {"rounds": results, "manifest": manifest}