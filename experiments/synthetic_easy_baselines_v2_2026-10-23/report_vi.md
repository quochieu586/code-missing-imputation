# Synthetic easy — baseline results

Dataset: `datasets/synthetic/wave_panel_v1/easy` (standard scale, seed 2026).
The benchmark used the two declared MCAR/MAR masks.

Protocol: entity-disjoint train/validation/test. Candidate parameters were selected by hidden-cell sample ratio MAE (X/N) on validation; each selected model was refit on train+validation and evaluated on test. Test results did not select parameters. Fallback was `train_mean`, degenerate policies were enabled, and no metadata covariates were passed to the baselines.

## Test metrics

| Mask | Baseline | Selected parameters | Hidden cells | Count MAE | Count RMSE | Sample ratio MAE | Sample ratio RMSE | Latent p MAE | Latent p RMSE | Fallback rows |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01_mcar_declared_patterns | hron_knn | `{"k": 10}` | 1680 | 12.4933 | 31.0557 | 0.0062688 | 0.0150398 | 0.00609235 | 0.0150488 | 0 |
| 01_mcar_declared_patterns | micoda_v2 | `{"W": 1e-06, "max_iter": 50}` | 1680 | 151.136 | 362.26 | 0.07421 | 0.167955 | 0.074052 | 0.167771 | 0 |
| 01_mcar_declared_patterns | tsagris_alpha | `{"alpha": 1.0, "k": 10}` | 1680 | 12.1695 | 29.6732 | 0.00601851 | 0.0140457 | 0.00599159 | 0.0143845 | 0 |
| 02_mar_declared_patterns | hron_knn | `{"k": 10}` | 1568 | 11.6604 | 28.7372 | 0.00687208 | 0.016277 | 0.00659672 | 0.0157323 | 0 |
| 02_mar_declared_patterns | micoda_v2 | `{"W": 1e-06, "max_iter": 50}` | 1568 | 127.151 | 291.367 | 0.0749776 | 0.166369 | 0.0747277 | 0.165765 | 0 |
| 02_mar_declared_patterns | tsagris_alpha | `{"alpha": 1.0, "k": 10}` | 1568 | 10.6959 | 25.9544 | 0.00631817 | 0.0148384 | 0.00603028 | 0.0144303 | 0 |

## Integrity checks

Per-run finite, nonnegative, observed-value preservation, and count-budget checks are in `summary.json`. Count, sample ratio (X/N), and latent p MAE/RMSE are distinct. Hidden test error by latent stratum (true zero, sampling zero, positive count) is also recorded there.

Candidate validation scores and failures, prediction arrays, and model diagnostics are retained under each mask directory. See `metrics.csv` for a compact spreadsheet-ready summary and `run_manifest.json` for provenance.
