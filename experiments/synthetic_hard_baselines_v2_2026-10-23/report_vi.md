# Synthetic hard — baseline results

Dataset: `C:\Users\admin\Tai_lieu\detaikhoahoc\COVID19\missing_impute\code-missing-imputation\datasets\synthetic\wave_panel_v2\hard` (scale standard, seed 2026).
The benchmark used the two declared MCAR/MAR masks.

Protocol: entity-disjoint train/validation/test. Candidate parameters were selected by hidden-cell sample ratio MAE (X/N) on validation; each selected model was refit on train+validation and evaluated on test. Test results did not select parameters. Fallback was `train_mean`, degenerate policies were enabled, and no metadata covariates were passed to the baselines.

## Test metrics

| Mask | Baseline | Selected parameters | Hidden cells | Count MAE | Count RMSE | Sample ratio MAE | Sample ratio RMSE | Latent p MAE | Latent p RMSE | Fallback rows |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01_mcar_declared_patterns | hron_knn | `{"k": 10}` | 1680 | 1.54254 | 5.14616 | 0.0165475 | 0.0420858 | 0.0188665 | 0.0470978 | 0 |
| 01_mcar_declared_patterns | micoda_v2 | `{"W": 1e-06, "max_iter": 50}` | 1680 | 8.68168 | 31.9966 | 0.0773084 | 0.187606 | 0.0764032 | 0.183244 | 0 |
| 01_mcar_declared_patterns | tsagris_alpha | `{"alpha": 0.5, "k": 10}` | 1680 | 2.48894 | 9.8687 | 0.0238439 | 0.0637033 | 0.0246947 | 0.0634111 | 0 |
| 02_mar_declared_patterns | hron_knn | `{"k": 10}` | 1472 | 1.16508 | 3.4003 | 0.020546 | 0.0574011 | 0.0230662 | 0.0589436 | 0 |
| 02_mar_declared_patterns | micoda_v2 | `{"W": 1e-06, "max_iter": 50}` | 1472 | 5.14817 | 15.5013 | 0.0762364 | 0.184607 | 0.0759994 | 0.180028 | 0 |
| 02_mar_declared_patterns | tsagris_alpha | `{"alpha": 0.5, "k": 10}` | 1472 | 1.46678 | 4.82961 | 0.0245861 | 0.0640205 | 0.0244079 | 0.0606796 | 0 |

## Integrity checks

Per-run finite, nonnegative, observed-value preservation, and count-budget checks are in `summary.json`. Count, sample ratio (X/N), and latent p MAE/RMSE are distinct. Hidden test error by latent stratum (true zero, sampling zero, positive count) is also recorded there.

Candidate validation scores and failures, prediction arrays, and model diagnostics are retained under each mask directory. See `metrics.csv` for a compact spreadsheet-ready summary and `run_manifest.json` for provenance.
