"""Pre-GAN readiness audit: everything the GAN stage consumes, steps 1-6.

Covers the plan's data contract (S3.2), the Tsagris warm-start (S4), the
occurrence gate (S5) and the fusion truth table (S6). Does not run or read GAN.

Run from the project root:  python scripts/audit_pregan.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from missing_imputation.data.closure import build_full_composition
from missing_imputation.data.lineage import compute_sha256, load_lineage
from missing_imputation.data.loader import load_config, load_covariants

checks: list[tuple[str, bool, str]] = []


def check(name, ok, detail=""):
    checks.append((name, bool(ok), str(detail)))


cfg = load_config("configs/data.yaml")
vc = cfg.variant_components
n_vars = len(vc)
lin = load_lineage("configs/data.yaml")

df = load_covariants(cfg.path, cfg)
counts18, obs18, props18, total = build_full_composition(df, vc, cfg.total_sequence_col)
M_obs = obs18[:, :n_vars]
M_target = ~M_obs

ts = pd.read_csv("artifacts/dataset_00_tsagris.csv")
fused = pd.read_csv("artifacts/fused/dataset_0_fused_raw.csv")
prov = pd.read_parquet("artifacts/fused/cell_provenance.parquet")
M_fixed = np.load("artifacts/fused/M_fixed.npz")["M_fixed"].astype(bool)
M_gan = np.load("artifacts/fused/M_gan.npz")["M_gan"].astype(bool)
w_gan = np.load("artifacts/fused/w_gan.npz")["w_gan"]
post = np.load("artifacts/occurrence/occurrence_posterior.npz")
M_tz = M_fixed & M_target

raw_counts = counts18[:, :n_vars]
ts_counts = ts[vc].to_numpy()
f = fused[vc].to_numpy()

# ---------------------------------------------------------------- 1. lineage
print("=" * 74)
print("1. RAW DATA AND MASKS (plan S3)")
print("=" * 74)
check("raw checksum matches frozen lineage",
      compute_sha256(cfg.path) == lin.get("raw_checksum"))
check("row count", len(df) == lin.get("expected_n_rows"), f"{len(df):,}")
check("location count", df[cfg.location_col].nunique() == lin.get("expected_n_locations"))
check("M_target.sum() == 86,050",
      int(M_target.sum()) == lin.get("expected_m_target_count"), f"{int(M_target.sum()):,}")
check("complete rows == 516",
      int(M_obs.all(axis=1).sum()) == lin.get("expected_n_complete_rows"))
check("M_observed AND M_target empty", not (M_obs & M_target).any())
check("M_observed OR M_target covers every cell", (M_obs | M_target).all())
check("variant order matches lineage", vc == lin.get("variant_order"))

# ------------------------------------------------------- 2. config actually read
print()
print("=" * 74)
print("2. CONFIG WIRING (L1) AND AUDIT HONESTY (L2)")
print("=" * 74)
import yaml
raw_yaml = yaml.safe_load(open("configs/data.yaml", encoding="utf-8"))
check("variant_components come from the YAML, not a hardcoded default",
      cfg.variant_components == raw_yaml["variant_components"])
check("validation rules reach DataConfig",
      cfg.validation == raw_yaml["validation"], f"{cfg.validation}")
check("freq_days reaches DataConfig", cfg.freq_days == raw_yaml["time_grid"]["freq_days"])
check("other_col reaches DataConfig", cfg.other_col == raw_yaml["derived"]["other_col"])

from missing_imputation.data.loader import run_audit
rep = run_audit(df, cfg, data_path=cfg.path)
# Corrupt a copy and confirm the audit now actually reports it
bad = df.copy()
bad.loc[bad.index[0], cfg.total_sequence_col] = 0
rep_bad = run_audit(bad, cfg)
check("audit reports clean data as valid", rep.is_valid, f"errors={rep.validation_errors}")
check("audit DETECTS an injected violation", not rep_bad.is_valid,
      f"errors={rep_bad.validation_errors}")

# --------------------------------------------------------- 3. Tsagris warm-start
print()
print("=" * 74)
print("3. TSAGRIS WARM-START (plan S4)  -- dataset_00_tsagris.csv")
print("=" * 74)
ts_other = ts[cfg.other_col].to_numpy()
check("observed counts unchanged", np.array_equal(ts_counts[M_obs], raw_counts[M_obs]))
check("no NaN", not ts[vc].isna().to_numpy().any())
check("non-negative", (ts_counts >= 0).all())
check("integer", np.array_equal(ts_counts, np.floor(ts_counts)))
check("closure sum(variants)+other == total_sequence",
      (ts_counts.sum(axis=1) + ts_other == total.astype(np.int64)).all())
check("other >= 0", (ts_other >= 0).all())
imputed_rows = ~M_obs.all(axis=1)
n_other_pos_imputed = int((ts_other[imputed_rows] > 0).sum())
check("residual survives on imputed rows (L3)", n_other_pos_imputed > 0,
      f"{n_other_pos_imputed:,}/{int(imputed_rows.sum()):,} imputed rows have other > 0")

metrics = json.load(open("artifacts/baselines/baseline_metrics.json", encoding="utf-8"))
distinct = all(
    not np.isclose(s["mse_empirical"], s["jsd_empirical"]) for s in metrics["scores"]
)
check("mse_empirical is an MSE, not a copy of the JSD (L5)", distinct,
      "; ".join(f"{s['method']}: mse={s['mse_empirical']:.6f} jsd={s['jsd_empirical']:.6f}"
                for s in metrics["scores"]))

pool_split = json.load(open("configs/splits/baseline_pool.json", encoding="utf-8"))
n_complete = int(M_obs.all(axis=1).sum())
check("baseline split IDs are in complete-pool index space (L8)",
      max(pool_split["time_block_test_rows"]) < n_complete
      and len(pool_split["fold_of_row"]) == n_complete,
      f"n_complete={n_complete}, max time-block idx={max(pool_split['time_block_test_rows'])}")
check("baseline split covers the full composition incl. residual",
      pool_split["n_composition_parts"] == n_vars + 1)

# ------------------------------------------------------------- 4. occurrence gate
print()
print("=" * 74)
print("4. OCCURRENCE GATE (plan S5)")
print("=" * 74)
occ_man = json.load(open("artifacts/occurrence/occurrence_model_manifest.json", encoding="utf-8"))
occ_cfg = yaml.safe_load(open("configs/occurrence/logistic_gate.yaml", encoding="utf-8"))["occurrence"]
c_grid = occ_cfg["model"]["C_grid"]
check("regularisation C was actually tuned, not stuck at C_grid[0] (L11)",
      occ_man["C"] != c_grid[0] or len(set(c_grid)) == 1,
      f"C={occ_man['C']}, grid={c_grid}")
check("release decision recorded", "released" in occ_man, f"released={occ_man['released']}")
check("posterior defined on every target cell",
      len(post["p_nonzero"]) == int(M_target.sum()), f"{len(post['p_nonzero']):,}")
p = post["p_nonzero"]
check("posterior is a probability", (p >= 0).all() and (p <= 1).all(),
      f"range [{p.min():.4f}, {p.max():.4f}]")
check("posterior is not collapsed to a single value", p.std() > 1e-3, f"std={p.std():.4f}")

oof = pd.read_csv("artifacts/occurrence/occurrence_oof_predictions.csv")
check("OOF predictions written for every recipe",
      oof["recipe"].nunique() >= 3, f"recipes={sorted(oof['recipe'].unique())}")
check("OOF covers all 17 variants", oof["variant"].nunique() == n_vars)

# ------------------------------------------------------------------- 5. fusion
print()
print("=" * 74)
print("5. FUSION TRUTH TABLE (plan S6, ZPGF gate)  -- dataset_0_fused_raw.csv")
print("=" * 74)
fu_other = fused[cfg.other_col].to_numpy()
check("observed cells == raw counts", np.array_equal(f[M_obs], raw_counts[M_obs]))
check("every M_target_zero cell is exactly 0", (f[M_tz] == 0).all(),
      f"{int((f[M_tz] != 0).sum())} violations")
check("closure sum(variants)+other == total_sequence",
      (f.sum(axis=1) + fu_other == total.astype(np.int64)).all())
check("non-negative integers", (f >= 0).all() and np.array_equal(f, np.floor(f)))
check("no NaN", not fused[vc].isna().to_numpy().any())
check("M_fixed AND M_gan empty", not (M_fixed & M_gan).any())
check("M_target_zero OR M_gan == M_target", np.array_equal(M_tz | M_gan, M_target))
check("M_gan is a subset of M_target", not (M_gan & ~M_target).any())
check("M_fixed == observed OR target_zero", np.array_equal(M_fixed, M_obs | M_tz))
check("residual survives on imputed rows (L16)",
      int((fu_other[imputed_rows] > 0).sum()) > 0,
      f"{int((fu_other[imputed_rows] > 0).sum()):,}/{int(imputed_rows.sum()):,}")

src = prov["source"].to_numpy().reshape(len(df), n_vars)
check("provenance RAW_OBSERVED == M_observed", np.array_equal(src == "RAW_OBSERVED", M_obs))
check("provenance TSAGRIS_WARM_START == M_gan (L15)",
      np.array_equal(src == "TSAGRIS_WARM_START", M_gan))
check("provenance OCCURRENCE_CONFIDENT_ZERO == M_target_zero",
      np.array_equal(src == "OCCURRENCE_CONFIDENT_ZERO", M_tz))
check("no stale cell_provenance.csv beside the parquet",
      not Path("artifacts/fused/cell_provenance.csv").exists())

# --------------------------------------------------------- 6. GAN input contract
print()
print("=" * 74)
print("6. GAN INPUT CONTRACT  (files the GAN stage will consume)")
print("=" * 74)
required = {
    "dataset_0_fused_raw.csv": "artifacts/fused/dataset_0_fused_raw.csv",
    "M_fixed.npz": "artifacts/fused/M_fixed.npz",
    "M_gan.npz": "artifacts/fused/M_gan.npz",
    "w_gan.npz": "artifacts/fused/w_gan.npz",
    "occurrence_posterior.npz": "artifacts/occurrence/occurrence_posterior.npz",
    "zpgf_gate_summary.csv": "artifacts/fused/zpgf_gate_summary.csv",
    "cell_provenance.parquet": "artifacts/fused/cell_provenance.parquet",
}
for label, path in required.items():
    check(f"exists: {label}", Path(path).exists(), path)

check("w_gan within [0, 1]", (w_gan >= 0).all() and (w_gan <= 1).all())
check("w_gan == 0 outside M_gan", (w_gan[~M_gan] == 0).all())
check("w_gan > 0 on every M_gan cell", (w_gan[M_gan] > 0).all())
check("every variant has GAN-editable cells",
      all(M_gan[:, j].sum() > 0 for j in range(n_vars)),
      f"{sum(M_gan[:, j].sum() > 0 for j in range(n_vars))}/{n_vars}")
check("fused row count matches raw", len(fused) == len(df))
check("mask shapes match the data grid",
      M_fixed.shape == (len(df), n_vars) and M_gan.shape == (len(df), n_vars))

# ----------------------------------------------------------------------- report
width = max(len(n) for n, _, _ in checks)
n_fail = 0
current = None
for name, ok, detail in checks:
    flag = "PASS" if ok else "FAIL"
    if not ok:
        n_fail += 1
    print(f"[{flag}] {name:<{width}}  {detail}")
print()
print(f"{len(checks) - n_fail}/{len(checks)} checks passed")

print()
print("=" * 74)
print("SUMMARY NUMBERS")
print("=" * 74)
print(f"  target cells                : {int(M_target.sum()):,}")
print(f"  hard-locked to zero         : {int(M_tz.sum()):,} ({100*M_tz.sum()/M_target.sum():.1f}%)")
print(f"  handed to the GAN (M_gan)   : {int(M_gan.sum()):,} ({100*M_gan.sum()/M_target.sum():.1f}%)")
print(f"  w_gan on M_gan              : mean {w_gan[M_gan].mean():.4f}, "
      f"min {w_gan[M_gan].min():.6f}, max {w_gan[M_gan].max():.4f}")
print(f"  rows with residual other>0  : {int((fu_other > 0).sum()):,}/{len(fused):,}")
raise SystemExit(1 if n_fail else 0)
