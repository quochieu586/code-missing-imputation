import numpy as np
import pandas as pd

V = ['recombinant','20A','20B','20C','20E','Beta','Alpha','Gamma','Delta','Kappa','Epsilon','Eta','Iota','Lambda','Mu','Omicron','S:677']

df_raw = pd.read_csv('data/covariants.csv')
M_target = df_raw[V].isna().to_numpy()
mtz = np.load('artifacts/occurrence/M_target_zero.npz')['M_target_zero']
mgan = np.load('artifacts/occurrence/M_gan.npz')['M_gan']

print('== Claim 2: per-variant locking ==')
for j, v in enumerate(V):
    t = int(M_target[:, j].sum()); z = int(mtz[:, j].sum()); g = int(mgan[:, j].sum())
    pct = 100 * z / t if t else 0
    print(f'{v:12s} M_target={t:6d} locked0={z:6d} ({pct:5.1f}%) toGAN={g:5d}')

df_fused = pd.read_csv('artifacts/fused/dataset_0_fused_raw.csv')
df_gated = pd.read_csv('artifacts/refined/dataset_1_refined.csv')

F = df_fused[V].fillna(0).to_numpy()
G = df_gated[V].fillna(0).to_numpy()
print('== Claim 3 symptom: fused vs gated identical:', np.array_equal(F, G), ' n_diff:', int((F != G).sum()))

tot = df_raw['total_sequence'].to_numpy()
raw_props = df_raw[V].fillna(0).to_numpy() / tot[:, None]
fus_props = F / tot[:, None]
gat_props = G / tot[:, None]

from missing_imputation.evaluation.metrics import jsd_metric, mae_proportions, mse_proportions
for name, P in [('fused', fus_props), ('gated', gat_props)]:
    print(f'{name} full-matrix mse=%.17g jsd=%.17g mae=%.17g' % (mse_proportions(raw_props, P), jsd_metric(raw_props, P), mae_proportions(raw_props, P)))

obs = ~M_target
print('observed-only mae fused=%.17g gated=%.17g' % (mae_proportions(raw_props[obs], fus_props[obs]), mae_proportions(raw_props[obs], gat_props[obs])))