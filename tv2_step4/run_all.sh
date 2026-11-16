#!/bin/zsh
# Usage: run_all.sh <mask> ; the six new residual-rule variants (residual_<mask>), then the
# prop_other consistency runs (check_<mask>), rate 5%, seeds 0-4, --save-predictions.
# Current cells of the matrix are read from tv2_step2/runs and tv2_step3/runs.
set -e
cd "$(dirname $0)/.."
export PYTHONPATH=src
for pair in residual:models_residual_rule check:models_residual_check; do
  /opt/anaconda3/bin/python -m longitudinal_impute.run --data data/covariants.csv \
    --schema configs/baselines_covariants.toml --models configs/${pair#*:}.toml \
    --output tv2_step4/runs/${pair%%:*}_$1 --mask $1 --mask-rate 0.05 --seeds 0 1 2 3 4 --save-predictions
done
