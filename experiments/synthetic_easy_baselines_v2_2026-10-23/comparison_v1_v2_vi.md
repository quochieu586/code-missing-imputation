# So sánh baseline easy: dataset v1 và v2

**Kết quả:** không có thay đổi về prediction hoặc lỗi trên counts. Sáu prediction arrays và row IDs giống hệt nhau; tham số được chọn, validation X/N MAE, test count MAE/RMSE và test sample ratio MAE/RMSE giữ nguyên. V2 thêm latent-p MAE/RMSE trực tiếp vào result; các số đó trùng với lần rescore v1 đã lưu trước đó.

| Mask | Baseline | Tham số được chọn | Count MAE Δ | Count RMSE Δ | X/N MAE Δ | X/N RMSE Δ | Latent p MAE v1 → v2 | Latent p RMSE v1 → v2 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 01_mcar_declared_patterns | hron_knn | `{"k": 10}` | +0 | +0 | +0 | +0 | 0.0060923485 → 0.0060923485 | 0.01504878 → 0.01504878 |
| 01_mcar_declared_patterns | micoda_v2 | `{"W": 1e-06, "max_iter": 50}` | +0 | +0 | +0 | +0 | 0.074052022 → 0.074052022 | 0.16777093 → 0.16777093 |
| 01_mcar_declared_patterns | tsagris_alpha | `{"alpha": 1.0, "k": 10}` | +0 | +0 | +0 | +0 | 0.0059915929 → 0.0059915929 | 0.014384467 → 0.014384467 |
| 02_mar_declared_patterns | hron_knn | `{"k": 10}` | +0 | +0 | +0 | +0 | 0.006596724 → 0.006596724 | 0.015732323 → 0.015732323 |
| 02_mar_declared_patterns | micoda_v2 | `{"W": 1e-06, "max_iter": 50}` | +0 | +0 | +0 | +0 | 0.074727717 → 0.074727717 | 0.165765 → 0.165765 |
| 02_mar_declared_patterns | tsagris_alpha | `{"alpha": 1.0, "k": 10}` | +0 | +0 | +0 | +0 | 0.0060302812 → 0.0060302812 | 0.014430324 → 0.014430324 |

Cấu hình: easy/standard, seed 2026; entity-disjoint train/validation/test; cùng hai mask declared MCAR/MAR và cùng candidate grids. Chọn model bằng validation MAE của X/N, fit cuối trên train+validation, test chỉ dùng báo cáo. Các metric v1 về latent p lấy từ rescore prediction v1 đã lưu.

Đã đối chiếu 6 cặp (3 baseline × 2 masks). Prediction arrays/row IDs giống hệt nhau: **True**. Tham số và validation score giữ nguyên: **True**. Sai số count và X/N giữ nguyên: **True**.
