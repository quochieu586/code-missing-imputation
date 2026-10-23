# Benchmark offline: Hron, MICoDa, Tsagris

Trong kịch bản chính whole-feature, **tsagris có MAE c/N thấp nhất** trong ba cấu hình đã đăng ký: 0.050419 (5.042 điểm phần trăm), count MAE=30.2235. Kết luận này áp dụng cho artificial targets trên các ô vốn quan sát được; không phải accuracy đã biết của native missing.

Cả180 primary test jobs giữ nguyên observed counts, không âm và không vượt total_sequence. Các score được chọn bằng validation rồi chấm test tách location. MICoDa-V2 có giới hạn50 iterations; xem cờ hội tụ bên dưới trước khi diễn giải so sánh.

![MAE c/N theo masking và CI95](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/docs/benchmark_offline_2026-10-15/figures/primary_ratio_mae.png)

Dữ liệu: 11671 dòng, 150 địa điểm; 5-fold group theo location, 3 mask seeds, rate30%. Whole-feature là kịch bản chính. Validation chọn tham số trước test. Không có causal mask.

| Phương pháp | Kịch bản | Targets | MAE counts | MAE tỷ lệ/N | CI95 tỷ lệ (location bootstrap) | Fallback rows |
|---|---|---:|---:|---:|---|---:|
| hron | whole_feature | 117675 | 41.5933 | 0.069808 | [0.060303, 0.079879] | 45/35013 |
| hron | random_cell | 100834 | 53.5397 | 0.066664 | [0.060045, 0.074289] | 75/35013 |
| hron | time_block | 102822 | 174.9743 | 0.101858 | [0.095187, 0.109082] | 10701/35013 |
| hron | whole_row | 102822 | 141.4339 | 0.098455 | [0.092450, 0.104857] | 10701/35013 |
| micoda | whole_feature | 117675 | 62.5996 | 0.061203 | [0.055018, 0.068207] | 0/35013 |
| micoda | random_cell | 100834 | 78.8789 | 0.059165 | [0.054882, 0.063982] | 0/35013 |
| micoda | time_block | 102822 | 132.8183 | 0.081128 | [0.073190, 0.089793] | 0/35013 |
| micoda | whole_row | 102822 | 120.6099 | 0.076361 | [0.071151, 0.082353] | 0/35013 |
| tsagris | whole_feature | 117675 | 30.2235 | 0.050419 | [0.044770, 0.056763] | 0/35013 |
| tsagris | random_cell | 100834 | 41.6254 | 0.048731 | [0.043797, 0.054123] | 0/35013 |
| tsagris | time_block | 102822 | 133.6145 | 0.081997 | [0.073637, 0.091240] | 0/35013 |
| tsagris | whole_row | 102822 | 119.3472 | 0.077025 | [0.071663, 0.083135] | 0/35013 |

Targets là tổng các lần che qua3 seeds; không phải số ô độc lập. CI resample theo location, đã gộp các mask repetitions trong location. Ba phương pháp dùng đúng cùng split/mask/labels.

## Biến thể bổ sung (whole-feature)

| Method | Successful jobs | MAE tỷ lệ | Fallback |
|---|---:|---:|---:|
| tsagris_basic/whole_feature | 15/15 | 0.061918 | 0 |
| tsagris_global_cv/whole_feature | 15/15 | 0.060561 | 0 |
| tsagris_adaptive/whole_feature | 15/15 | 0.055656 | 0 |

- Fold 0, hron_ls: vượt wall budget — Exceeded preregistered 180s wall budget per supplementary fold/method.

- Fold 0, hron_lts: lỗi số học — CLR requires finite positive parts; use an explicit encoder zero policy.

- Fold 0, micoda_v1: vượt wall budget — Exceeded preregistered 180s wall budget per supplementary fold/method.

- Fold 1, hron_ls: vượt wall budget — Exceeded preregistered 180s wall budget per supplementary fold/method.

- Fold 1, hron_lts: lỗi số học — CLR requires finite positive parts; use an explicit encoder zero policy.

- Fold 1, micoda_v1: vượt wall budget — Exceeded preregistered 180s wall budget per supplementary fold/method.

- Fold 2, hron_ls: vượt wall budget — Exceeded preregistered 180s wall budget per supplementary fold/method.

- Fold 2, hron_lts: lỗi số học — CLR requires finite positive parts; use an explicit encoder zero policy.

- Fold 2, micoda_v1: vượt wall budget — Exceeded preregistered 180s wall budget per supplementary fold/method.

- Fold 3, hron_ls: vượt wall budget — Exceeded preregistered 180s wall budget per supplementary fold/method.

- Fold 3, hron_lts: lỗi số học — CLR requires finite positive parts; use an explicit encoder zero policy.

- Fold 3, micoda_v1: vượt wall budget — Exceeded preregistered 180s wall budget per supplementary fold/method.

- Fold 4, hron_ls: vượt wall budget — Exceeded preregistered 180s wall budget per supplementary fold/method.

- Fold 4, hron_lts: lỗi số học — CLR requires finite positive parts; use an explicit encoder zero policy.

- Fold 4, micoda_v1: vượt wall budget — Exceeded preregistered 180s wall budget per supplementary fold/method.

## Diễn giải và giới hạn

- Baselines là general/cross-sectional; offline cho phép ngữ cảnh tương lai, nhưng ba core này không tự học attention hoặc dynamics dọc chuỗi. Row/time-block che tất cả features khiến Hron dùng train-mean fallback; Tsagris dùng global donor policy; MICoDa dùng covariates.
- Metadata chính: median_age và new_cases_smoothed_per_million; fit encoder chỉ train. Hron/Tsagris core không dùng covariate regression; MICoDa có dùng. classification/Overall_score/group chưa được bật vì provenance chưa rõ.
- Counts zeros giữ nguyên; smoothing chỉ encoder. Other unknown sau khi che raw target. Output constraints đảm bảo tính hợp lệ, không tự chứng minh accuracy.
- Composition/coverage chỉ chấm trên rows có complete truth; conditional17 LR diagnostics dùng fixed additive1e-6 để xử lý zeros và có ghi support. Temporal/feature relationships mô tả vùng có truth; correlation pooled có thể phản ánh country mix và không phải quan hệ nhân quả.
- LTS raw/solver approximate và Hron Table1/software parity có giới hạn đã ghi trong nextgen/BASELINES_VI.md; không gọi benchmark này là tái lập tất cả paper tables.
- Biến thể bổ sung có wall budget 180s mỗi fold/method, đăng ký trước khi chạy test. Resource-limit là kết quả chi phí, không phải điểm accuracy và không được đưa vào ranking trên tập fold khác nhau.
- Native missing không có ground truth. CSV final refit dùng toàn bộ observed sau chọn cấu hình; không dùng nó để báo lại test accuracy.

## CSV hoàn chỉnh

- [hron](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/experiments/offline_baselines_2026-10-15_v2/final/hron/completed.csv): 11671 dòng, điền 86050 missing cells, observed changes=0, invalid rows=0.
- [micoda](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/experiments/offline_baselines_2026-10-15_v2/final/micoda/completed.csv): 11671 dòng, điền 86050 missing cells, observed changes=0, invalid rows=0.
- [tsagris](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/experiments/offline_baselines_2026-10-15_v2/final/tsagris/completed.csv): 11671 dòng, điền 86050 missing cells, observed changes=0, invalid rows=0.

Chi tiết strata, per-feature, location macro, paired bootstrap differences, temporal/feature diagnostics và failures: summary.json. Toàn bộ primary test prediction arrays được đọc lại và rescore; xem rescoring.json.


## Macro và strata cho whole-feature

| Họ | Location macro MAE tỷ lệ | Feature macro MAE tỷ lệ | Zero MAE | Positive MAE |
|---|---:|---:|---:|---:|
| hron | 0.105821 | 0.042532 | 0.010441 | 0.342811 |
| micoda | 0.090471 | 0.038075 | 0.032408 | 0.193617 |
| tsagris | 0.078239 | 0.031602 | 0.032355 | 0.133486 |

## Kiểm tra hội tụ và phạm vi kết luận

Các fits sau chưa đạt tolerance trước khi chạm max_iter đã đăng ký: micoda/fold0 (50 iterations), micoda/fold1 (50 iterations), micoda/fold2 (50 iterations), micoda/fold3 (50 iterations), micoda/fold4 (50 iterations). Đây là benchmark cấu hình có budget cố định; không kết luận đây là performance sau hội tụ đầy đủ.

Numerical failure, rank/domain not-applicable và resource-limit là các trạng thái khác nhau (xem supplementary_status.csv). Không dùng score của một vài fold thành ranking chung. Wall times gồm chạy pipeline/scoring/I/O trong điều kiện local có2 workers, không phải microbenchmark cô lập.

## Paired comparisons của whole-feature

| Chênh lệch MAE location-macro (first − second) | Điểm | CI95 |
|---|---:|---|
| hron-micoda | 0.015349 | [0.007936, 0.023518] |
| hron-tsagris | 0.027582 | [0.018548, 0.037508] |
| micoda-tsagris | 0.012233 | [0.008916, 0.016094] |

Chênh lệch âm ưu tiên method đứng trước. CI trên paired location errors; không suy ra quan hệ nhân quả hoặc hiệu quả trên native missing không có nhãn.

Tỷ lệ che whole-feature thực tế34,911% vì ceil feature count; random-cell gần30%, row/block30,505%. Time-block được chọn theo timepoints, không calendar duration cố định.

Đã lưu 450 proposal traces và 3 final traces. Các lần replay tái tạo constrained counts giống từng bit; predictions/selection/metrics gốc không bị thay đổi. Mỗi job có proposal_trace.npz gồm proposal trước repair, output cuối, Other và fallback flags.

Bảng CSV: primary_metrics.csv, per_feature_metrics.csv, selected_parameters.csv, supplementary_status.csv, paired_location_differences.csv.

## Quan hệ temporal/feature và coverage — whole-feature

| Họ | MAE temporal Δ(c/N) | MAE coverage | Conditional17 JSD | Error correlation feature |
|---|---:|---:|---:|---:|
| hron | 0.028273 | 0.124065 | 0.146290 | 0.098434 |
| micoda | 0.016257 | 0.142341 | 0.174244 | 0.187775 |
| tsagris | 0.024834 | 0.064959 | 0.085280 | 0.103300 |

Các diagnostics trên là trung bình per-job có trọng số support; không phải một covariance matrix pooled duy nhất. Full-truth composition chủ yếu được kiểm ở5 countries có complete data, không đại diện đầy đủ native missing tại150 locations. Support correlation có thể khác giữa methods do variance bằng0, cần đọc cột support trong relationship_diagnostics.csv; không ranking correlation chỉ từ một số thấp.

Bootstrap là mô tả OOS errors với location clusters và fits cố định, không refit toàn bộ models trong mỗi bootstrap và không phải uncertainty của từng giá trị imputed. So sánh có90train/30validation/30test locations trong mỗi outer fold, không phải mô tả chung chung80/20 cross-validation. Final CSV refit dùng100% native observations.

Tsagris có point error thấp nhất ở whole-feature/random-cell trong lưới này; MICoDa có point ratio MAE thấp hơn ở time-block/whole-row. Không có kết luận một method tốt nhất mọi masking scenario, và không có claim về độ chính xác tại các ô thiếu thật chưa biết ground truth. Các bước tiếp theo có giá trị là cải thiện ổn định/hội tụ của iterative baselines theo protocol mới đăng ký và so temporal models trên chính split/mask đã khóa, thay vì chọn mask dễ hơn sau khi xem test.
