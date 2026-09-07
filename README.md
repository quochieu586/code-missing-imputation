# missing-imputation

Compositional longitudinal missing imputation for SARS-CoV-2 variant count data (covariants).

Pipeline: Tsagris JSD-kNN baselines → `dataset_00_tsagris` → ZINB zero-state → `dataset_01_zinb` → Fusion → `dataset_0_fused` → DeepMicroGen GAN refinement.

## Install

```bash
pip install -e .[test]
```

## CLI

```bash
missing-imputation audit-data --config configs/data.yaml
missing-imputation run-baselines --data data/covariants.csv --config-dir configs/baseline --output-dir artifacts/baselines
missing-imputation select-baseline --metrics artifacts/baselines/baseline_metrics.json
missing-imputation generate-dataset0 --config configs/data.yaml --method jsd_knn --output-dir artifacts
```

## Tests

```bash
pytest
```