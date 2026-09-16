"""Comprehensive EDA inspecting the Stage A kNN-Aitchison imputation on covariants.csv.

Compares empirical findings with Hron (2009) paper assumptions, properties, and results.
"""
import sys, os
import pandas as pd
import numpy as np

# Ensure utf-8 output
sys.stdout.reconfigure(encoding='utf-8')

df_raw = pd.read_csv('data/covariants.csv')
df_imp = pd.read_csv('data/processed/X1_knn_imputed.csv')

variant_cols = [c for c in df_raw.columns if c not in ['location','date','total_sequence']]

print("=" * 60)
print("COMPREHENSIVE EDA: RAW DATA VS kNN-AITCHISON IMPUTATION (X1)")
print("=" * 60)

raw_mat = df_raw[variant_cols].values
imp_mat = df_imp[variant_cols].values
mask_missing = np.isnan(raw_mat)
mask_observed = ~mask_missing

n_rows, n_cols = raw_mat.shape
n_total = n_rows * n_cols
n_miss = mask_missing.sum()
n_obs = mask_observed.sum()

print(f"Dimensions: {n_rows} rows x {n_cols} variant columns")
print(f"Total cells: {n_total}")
print(f"Observed cells: {n_obs} ({n_obs / n_total * 100:.2f}%)")
print(f"Missing cells: {n_miss} ({n_miss / n_total * 100:.2f}%)")

# 1. Zero analysis
raw_obs = raw_mat[mask_observed]
n_zeros = (raw_obs == 0).sum()
pct_zeros = n_zeros / n_obs * 100
print(f"\n--- 1. ZERO INFLATION ANALYSIS ---")
print(f"Observed non-missing cells that are EXACTLY 0: {n_zeros:,} ({pct_zeros:.2f}%)")
print(f"Observed non-missing cells that are > 0: {(raw_obs > 0).sum():,} ({(raw_obs > 0).mean()*100:.2f}%)")

# 2. Imputed values distribution
imp_vals = imp_mat[mask_missing]
print(f"\n--- 2. IMPUTED VALUES DISTRIBUTION ---")
print(f"Min: {imp_vals.min():.6f}")
print(f"Max: {imp_vals.max():.2f}")
print(f"Mean: {imp_vals.mean():.4f}")
print(f"Median: {np.median(imp_vals):.4f}")
print(f"Std: {imp_vals.std():.4f}")

# Quantiles
qs = [0.01, 0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
q_vals = np.quantile(imp_vals, qs)
for q, v in zip(qs, q_vals):
    print(f"  Quantile {int(q*100):02d}%: {v:.4f}")

n_exact_half = (imp_vals == 0.5).sum()
print(f"Imputed values exactly equal to 0.5 (pseudo-count): {n_exact_half:,} ({n_exact_half / len(imp_vals) * 100:.2f}%)")
n_below_one = (imp_vals <= 1.0).sum()
print(f"Imputed values <= 1.0: {n_below_one:,} ({n_below_one / len(imp_vals) * 100:.2f}%)")

# 3. Sum of variants vs total_sequence
total_seq = df_raw['total_sequence'].values
raw_sums = np.nansum(raw_mat, axis=1)
imp_sums = np.sum(imp_mat, axis=1)

print(f"\n--- 3. TOTAL SUM & SCALE BEHAVIOR ---")
print(f"Total sequence min={total_seq.min()}, median={np.median(total_seq):.1f}, max={total_seq.max()}")
print(f"Raw variant sum median: {np.median(raw_sums):.1f}")
print(f"Imputed variant sum median: {np.median(imp_sums):.1f}")

ratio_imp = imp_sums / np.maximum(total_seq, 1)
print(f"Ratio (Imputed variant sum / total_seq):")
print(f"  Median: {np.median(ratio_imp):.4f}")
print(f"  Mean:   {np.mean(ratio_imp):.4f}")
print(f"  Max:    {np.max(ratio_imp):.4f}")
n_exceed = (imp_sums > total_seq).sum()
print(f"Cases where variant sum EXCEEDS total_sequence: {n_exceed:,} ({n_exceed / n_rows * 100:.2f}%)")

# 4. Temporal consistency check
print(f"\n--- 4. TEMPORAL VARIANT SUCCESSION CHECK ---")
# E.g. Delta should peak mid-2021, Omicron late-2021 to 2022, Alpha early-2021
df_imp['date'] = pd.to_datetime(df_imp['date'])
monthly = df_imp.groupby(df_imp['date'].dt.to_period('M'))[['Alpha','Delta','Omicron','20A','20B']].mean()
print("Monthly average imputed counts for major variants (sample):")
sample_dates = ['2020-06', '2020-12', '2021-04', '2021-09', '2022-02', '2022-08', '2023-01']
available_dates = [d for d in sample_dates if pd.Period(d, 'M') in monthly.index]
print(monthly.loc[[pd.Period(d, 'M') for d in available_dates]])

# 5. Check if neighbor matching pulled across locations or dates
print(f"\n--- 5. SPREAD OF IMPUTED VALUES ACROSS COLUMNS ---")
col_summary = []
for c in variant_cols:
    miss_c = mask_missing[:, variant_cols.index(c)]
    imp_c = imp_mat[miss_c, variant_cols.index(c)]
    obs_c = raw_mat[~miss_c, variant_cols.index(c)]
    col_summary.append({
        'Variant': c,
        'Obs_count': len(obs_c),
        'Miss_count': len(imp_c),
        'Obs_0_pct': (obs_c == 0).mean() * 100 if len(obs_c) > 0 else 0,
        'Imp_mean': imp_c.mean() if len(imp_c) > 0 else np.nan,
        'Imp_median': np.median(imp_c) if len(imp_c) > 0 else np.nan,
        'Imp_p95': np.quantile(imp_c, 0.95) if len(imp_c) > 0 else np.nan,
        'Obs_p95': np.quantile(obs_c, 0.95) if len(obs_c) > 0 else np.nan
    })
sum_df = pd.DataFrame(col_summary)
print(sum_df.to_string(index=False))
