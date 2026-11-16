#!/bin/zsh
# Usage: run_all.sh <mask> ; the six frozen same_day_mean variants, rate 5%, seeds 0-4,
# --save-predictions. Baselines are read from tv2_step2/runs (same data and masks, checked).
set -e
cd "$(dirname $0)/.."
PYTHONPATH=src /opt/anaconda3/bin/python -m longitudinal_impute.run --data data/covariants.csv \
  --schema configs/baselines_covariants.toml --models configs/models_sameday_ablation.toml \
  --output tv2_step3/runs/ablation_$1 --mask $1 --mask-rate 0.05 --seeds 0 1 2 3 4 --save-predictions
