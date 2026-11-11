#!/bin/zsh
# Usage: run_all.sh <mask> ; the 7-model pass then the micoda_other pass (A18), rate 5%,
# with --save-predictions. whole_feature adds seed 109 (failure-policy case: 0 complete rows).
set -e
cd "$(dirname $0)/.."
PY=/opt/anaconda3/bin/python
export PYTHONPATH=nextgen/src
SEEDS=(0 1 2 3 4)
[[ $1 == whole_feature ]] && SEEDS+=(109)
$PY -m longitudinal_impute.run --data data/covariants.csv --schema nextgen/configs/baselines_covariants.toml \
  --models nextgen/configs/models_default.toml --output tv2_step2/runs/main_$1 --mask $1 --mask-rate 0.05 \
  --seeds $SEEDS --only hron tsagris micoda same_day_mean mean_fill marginal_sampler zero_fill --save-predictions || true
$PY -m longitudinal_impute.run --data data/covariants.csv --schema nextgen/configs/baselines_covariants.toml \
  --models nextgen/configs/models_micoda_other.toml --output tv2_step2/runs/micoda_other_$1 --mask $1 \
  --mask-rate 0.05 --seeds $SEEDS --save-predictions || true
