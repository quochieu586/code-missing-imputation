# Synthetic medium — baseline results

Dataset: `C:\Users\admin\Tai_lieu\detaikhoahoc\COVID19\missing_impute\code-missing-imputation\datasets\synthetic\wave_panel_v2\medium` (scale standard, seed 2026).
The benchmark used the two declared MCAR/MAR masks.

Protocol: entity-disjoint train/validation/test. Candidate parameters were selected by hidden-cell sample ratio MAE (X/N) on validation; each selected model was refit on train+validation and evaluated on test. Test results did not select parameters. Fallback was `train_mean`, degenerate policies were enabled, and no metadata covariates were passed to the baselines.

## Test metrics

| Mask | Baseline | Selected parameters | Hidden cells | Count MAE | Count RMSE | Sample ratio MAE | Sample ratio RMSE | Latent p MAE | Latent p RMSE | Fallback rows |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01_mcar_declared_patterns | hron_knn | `{"k": 10}` | 1680 | 4.39774 | 12.96 | 0.00899414 | 0.022106 | 0.00882018 | 0.0217163 | 0 |
| 01_mcar_declared_patterns | micoda_v2 | `{"W": 1e-06, "max_iter": 50}` | 1680 | 39.8176 | 111.915 | 0.0749276 | 0.173192 | 0.0750131 | 0.172613 | 0 |
| 01_mcar_declared_patterns | tsagris_alpha | `{"alpha": 1.0, "k": 10}` | 1680 | 5.56363 | 15.3233 | 0.0107429 | 0.0246671 | 0.0100165 | 0.0234585 | 0 |
| 02_mar_declared_patterns | hron_knn | `{"k": 10}` | 1576 | 3.82482 | 9.81339 | 0.0108018 | 0.0269421 | 0.0107797 | 0.0270257 | 0 |
| 02_mar_declared_patterns | micoda_v2 | `{"W": 1e-06, "max_iter": 50}` | 1576 | 27.6039 | 70.3967 | 0.0741292 | 0.168515 | 0.0741087 | 0.167686 | 0 |
| 02_mar_declared_patterns | tsagris_alpha | `{"alpha": 1.0, "k": 5}` | 1576 | 4.63629 | 12.0291 | 0.0125415 | 0.0303483 | 0.0122622 | 0.0300702 | 0 |

## Integrity checks

Per-run finite, nonnegative, observed-value preservation, and count-budget checks are in `summary.json`. Count, sample ratio (X/N), and latent p MAE/RMSE are distinct. Hidden test error by latent stratum (true zero, sampling zero, positive count) is also recorded there.

Candidate validation scores and failures, prediction arrays, and model diagnostics are retained under each mask directory. See `metrics.csv` for a compact spreadsheet-ready summary and `run_manifest.json` for provenance.
