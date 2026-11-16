#!/bin/zsh
# Usage: run_all.sh ; the three passes of step 2-4 (7 models, micoda_other, the cap variants) on the
# whole_feature_matched mask, rate 5%, mask seeds 0-4, --save-predictions, the frozen strata target.
set -e
cd "$(dirname $0)/.."
PY=/opt/anaconda3/bin/python
export PYTHONPATH=src
COMMON=(--data data/covariants.csv --schema configs/baselines_covariants.toml --mask whole_feature_matched
        --mask-target configs/mask_target_covid.json --mask-rate 0.05 --seeds 0 1 2 3 4 --save-predictions)
$PY -m longitudinal_impute.run $COMMON --models configs/models_default_a17.toml \
  --output tv2_step7/runs/main_whole_feature_matched \
  --only hron tsagris micoda same_day_mean mean_fill marginal_sampler zero_fill || true
$PY -m longitudinal_impute.run $COMMON --models configs/models_micoda_other.toml \
  --output tv2_step7/runs/micoda_other_whole_feature_matched || true
$PY -m longitudinal_impute.run $COMMON --models configs/models_residual_rule.toml \
  --output tv2_step7/runs/residual_whole_feature_matched --only hron_cap tsagris_cap micoda_other_cap || true
