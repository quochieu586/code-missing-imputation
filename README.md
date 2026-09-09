# missing-imputation

Compositional longitudinal missing imputation for SARS-CoV-2 variant count data (covariants).

Pipeline: raw masks → Tsagris JSD-kNN baselines → `dataset_00_tsagris` → Occurrence Gate →
ZPGF fusion → `dataset_0_fused` → DeepMicroGen GAN magnitude refinement → `dataset_1_refined`.

The zero decision is a Zero-Preserving Gated Fusion (plan §18–19): fusion decides only the mask
(hard-lock vs hand to the GAN) and emits a soft weight `w_gan`; the weight multiplies the GAN's
magnitude in `gan/postprocess.py` — after the GAN runs, before closure projection.

## Install

```bash
pip install -e ".[test,gan]"
```

`gan` pulls in PyTorch, required by `refine`. CPU-only wheels:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Pipeline

```bash
missing-imputation audit-data      --config configs/data.yaml
missing-imputation build-masks     --config configs/data.yaml --output artifacts/raw_masks.npz
missing-imputation run-baselines   --data data/covariants.csv --config-dir configs/baseline \
                                   --output-dir artifacts/baselines
missing-imputation select-baseline --metrics artifacts/baselines/baseline_metrics.json
missing-imputation generate-dataset0 --config configs/data.yaml \
                                   --method adaptive_jsd_alpha_knn --k 5 --alpha 10.0 \
                                   --output-dir artifacts
missing-imputation run-occurrence-gate --data data/covariants.csv \
                                   --tsagris artifacts/dataset_00_tsagris.csv \
                                   --config configs/occurrence/logistic_gate.yaml \
                                   --output-dir artifacts/occurrence
missing-imputation fuse-initializations --raw data/covariants.csv \
                                   --tsagris artifacts/dataset_00_tsagris.csv \
                                   --occurrence artifacts/occurrence \
                                   --zpgf-config configs/fusion/zpgf.yaml \
                                   --output-dir artifacts/fused
missing-imputation refine          --input artifacts/fused/dataset_0_fused_raw.csv \
                                   --m-fixed artifacts/fused/M_fixed.npz \
                                   --m-gan artifacts/fused/M_gan.npz \
                                   --w-gan artifacts/fused/w_gan.npz \
                                   --posterior artifacts/occurrence/occurrence_posterior.npz \
                                   --output-dir artifacts/refined
missing-imputation evaluate        --artifacts-dir artifacts --output-dir reports/evaluation
```

`refine` takes `--seed` and `--n-rounds` from `configs/gan/refinement.yaml` when not passed.

### Ablations the evaluation suite picks up

`evaluate` reads these directories if they exist:

```bash
# gated vs ungated (plan §12)
missing-imputation refine ... --no-soft-weights --output-dir artifacts/refined_ungated

# seed variance (plan §13 risk register)
missing-imputation refine ... --seed 42  --output-dir artifacts/refined_seed_42
missing-imputation refine ... --seed 123 --output-dir artifacts/refined_seed_123
missing-imputation refine ... --seed 456 --output-dir artifacts/refined_seed_456
```

## Configs

| File | Controls |
|---|---|
| `configs/data.yaml` | data path, variant order, column names |
| `configs/baseline/*.yaml` | the three Tsagris baselines |
| `configs/occurrence/logistic_gate.yaml` | occurrence model, OOF recipes, release gate |
| `configs/fusion/zpgf.yaml` | CRC calibration, hard-lock floor and zero-lock cap |
| `configs/gan/refinement.yaml` | GAN architecture, training, seeds, rounds |
| `configs/evaluation.yaml` | evaluation recipes, seeds, ablation list |

## Tests

```bash
pytest
```
