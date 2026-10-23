# Baselines trên synthetic medium và hard (v2)

Chạy riêng từng profile bằng cùng candidate grid, seed, split, hai mask và giao thức như easy. Mỗi profile chạy đủ 30 ứng viên (3 + 3 + 9 cho mỗi mask); tất cả hoàn tất. Validation sample_ratio MAE (X/N) chọn tham số; refit train+validation; test chỉ dùng báo cáo.

| Profile | Mask | Baseline | Tham số chọn | Test count MAE | Test X/N MAE | Test latent p MAE | Test latent p RMSE | Hidden test cells |
|---|---|---|---|---:|---:|---:|---:|---:|
| medium | 01_mcar_declared_patterns | hron_knn | `{"k": 10}` | 4.397744 | 0.00899414 | 0.008820184 | 0.02171632 | 1680 |
| medium | 01_mcar_declared_patterns | micoda_v2 | `{"W": 1e-06, "max_iter": 50}` | 39.81765 | 0.07492763 | 0.0750131 | 0.1726134 | 1680 |
| medium | 01_mcar_declared_patterns | tsagris_alpha | `{"alpha": 1.0, "k": 10}` | 5.56363 | 0.01074295 | 0.01001647 | 0.02345845 | 1680 |
| medium | 02_mar_declared_patterns | hron_knn | `{"k": 10}` | 3.824821 | 0.0108018 | 0.0107797 | 0.02702574 | 1576 |
| medium | 02_mar_declared_patterns | micoda_v2 | `{"W": 1e-06, "max_iter": 50}` | 27.60385 | 0.07412921 | 0.07410873 | 0.1676861 | 1576 |
| medium | 02_mar_declared_patterns | tsagris_alpha | `{"alpha": 1.0, "k": 5}` | 4.636294 | 0.01254149 | 0.01226216 | 0.03007018 | 1576 |
| hard | 01_mcar_declared_patterns | hron_knn | `{"k": 10}` | 1.542539 | 0.01654746 | 0.01886646 | 0.04709779 | 1680 |
| hard | 01_mcar_declared_patterns | micoda_v2 | `{"W": 1e-06, "max_iter": 50}` | 8.681676 | 0.07730843 | 0.07640323 | 0.1832443 | 1680 |
| hard | 01_mcar_declared_patterns | tsagris_alpha | `{"alpha": 0.5, "k": 10}` | 2.48894 | 0.02384386 | 0.02469475 | 0.06341107 | 1680 |
| hard | 02_mar_declared_patterns | hron_knn | `{"k": 10}` | 1.165079 | 0.02054596 | 0.02306618 | 0.05894364 | 1472 |
| hard | 02_mar_declared_patterns | micoda_v2 | `{"W": 1e-06, "max_iter": 50}` | 5.14817 | 0.07623635 | 0.07599942 | 0.1800281 | 1472 |
| hard | 02_mar_declared_patterns | tsagris_alpha | `{"alpha": 0.5, "k": 10}` | 1.466781 | 0.02458606 | 0.02440791 | 0.06067959 | 1472 |

Prediction checks for both profiles: finite, nonnegative, observed values unchanged, and no count-budget violations. Detailed diagnostics, candidate records and prediction arrays are in each run directory listed in `medium_hard_results.json`.

Each profile uses one seed and one benchmark configuration; values describe these runs and do not establish a general difficulty ranking.
