# Metric và protocol đánh giá cho slide

## Kết quả đã có — không cần chờ synthetic

Nguồn accepted là `docs/benchmark_offline_2026-10-15/` và `experiments/offline_baselines_2026-10-15_v2/`. Kết quả này thuộc source/config đã khóa trong run đó; sơ đồ code hiện tại không tự chuyển kết quả sang phiên bản source mới.

11671 dòng, 150 locations, 17 target features. Mỗi outer fold có 90 train/30 validation/30 test locations; 5 folds, mask seeds 42/43/44. Tune theo validation whole-feature ratio MAE; test giữ cùng masks cho ba methods. Bốn scenario: whole-feature (primary), random-cell, time-block, whole-row. Nominal mask rate 30%; whole-feature thực tế 34.911% do làm tròn số feature; row/block 30.505%. Time-block theo timepoints, không phải độ dài lịch cố định.

Primary families: `hron_knn`, `micoda_v2`, `tsagris_alpha`. Metadata cấu hình gồm median_age và new_cases_smoothed_per_million; MICoDa có covariate regression, hai core còn lại không dùng covariates theo cách đó. Không diễn giải đây là ablation đã kiểm soát chỉ riêng thuật toán.

| Whole-feature | Pooled MAE c/N | Pooled RMSE c/N | MAE counts | Location-macro MAE c/N |
|---|---:|---:|---:|---:|
| Hron | 0.069808 | 0.238570 | 41.5933 | 0.105821 |
| MICoDa | 0.061203 | 0.199331 | 62.5996 | 0.090471 |
| Tsagris | 0.050419 | 0.182321 | 30.2235 | 0.078239 |

Các số hiển thị cần đối chiếu CSV bằng script trước dựng plot; không chép tay từ bảng làm nguồn tính. Ratio×100 cho đơn vị điểm phần trăm (pp), không gọi là % cải thiện tương đối. Primary whole-feature có 117675 lượt target qua ba seeds, không phải 117675 quan sát độc lập. Final CSV đã điền 86050 native missing cells nhưng không có truth để tính accuracy ở đó.

## Metric nào trả lời câu hỏi nào?

Ký hiệu: H là tập ô bị che vốn có observation; e_ij = predicted count − true count; N_i là total_sequence. Mọi metric accuracy chỉ tính đúng support có truth, không lẫn observed được copy nguyên vẹn.

| Nhóm / ưu tiên | Metric và đơn vị | Ý nghĩa | Domain và support / giới hạn |
|---|---|---|---|
| Primary | Pooled MAE c/N = mean_H |e_ij|/N_i | Sai số tỷ lệ trung bình trên các lượt target | N_i hợp lệ; ghi n targets và n locations; locations có nhiều ô nặng hơn |
| Secondary | Pooled RMSE c/N = sqrt(mean_H (e_ij/N_i)^2) | Nhạy với lỗi lớn | Không suy ra từ mean per-job RMSE; dùng sufficient stats/predictions đúng cách |
| Secondary | MAE counts, RMSE counts | Sai số số lượng, hữu ích cho scale/budget | Chịu ảnh hưởng mạnh sequencing depth; trục riêng với tỷ lệ |
| Fairness of aggregation | Location-macro MAE c/N | Mỗi location có trọng số bằng nhau | Gộp seeds trong location; không thay thế pooled mà không đổi tên estimand |
| Fairness of aggregation | Feature-macro MAE c/N | Mỗi feature có trọng số bằng nhau | Ghi số feature được hỗ trợ, tránh feature hiếm bị pooled score che khuất |
| Stratified recovery | Zero / positive / rare-positive MAE c/N | Phân biệt dự đoán zero tốt với khôi phục tín hiệu thực | Rare-positive hiện là 0<c/N<0.01; ghi n từng stratum, không đổi ngưỡng theo test |
| Temporal structure | MAE Δ(c/N); slope MAE/day | Khôi phục biến thiên giữa timepoints | Cùng entity; cả hai endpoints vốn observed; ít nhất một target; gap>0. Ghi pairs và gap. Không gọi lag-1 calendar khi gap không đều |
| Coverage | MAE tổng tỷ lệ 17 targets | Giữ coverage ngoài Other đúng đến đâu | Chỉ rows có full truth và có target; không ép tổng 17 parts=1 |
| Composition | Conditional17 JSD theo convention code | Sai hình dạng tương đối trên 17 parts | Rows có full truth, tổng hai compositions dương; conditional closure bỏ thông tin coverage, phải đọc cạnh coverage MAE |
| Composition — appendix | Variation matrix / ILR covariance Frobenius error | Giữ cấu trúc log-ratio | Full truth, scoring-only additive 1e-6, đủ rows; báo zero policy/support; không so tuyệt đối qua dimension/sample composition khác nhau |
| Relationship — descriptive | Feature correlation/covariance error, lag correlation error | Chẩn đoán quan hệ tuyến tính | Pairwise support≥10 và variance>0 theo code; support có thể khác giữa methods, pooled country mix có thể gây nhiễu; không nhân quả |
| Operational validity | Observed changes, nonfinite/negative, budget violations | Có đáp ứng contract counts không | Đây là validity, không phải accuracy. Bản accepted báo observed/negative/budget violations=0 |
| Operational stability | Fallback rate, convergence/iterations, terminal status | Accuracy được tạo ra bằng đường nào? | Báo denominator và lý do. Nonconverged/resource-limit/numerical_failure/not_applicable là các trạng thái khác nhau |
| Cost — descriptive | Wall time và peak memory nếu có đo thật | Chi phí của pipeline | Existing wall time gồm fit/predict/score/I/O, hai workers; chưa có peak memory thì để chưa đo, không suy từ kích thước file |

Không thêm MAPE khi truth có zero, không dùng một R² pooled thay recovery quality, không gọi một point estimate là proper multiple imputation. Không thêm downstream classification metric khi chưa có task/labels/protocol hợp lệ.

## Suy luận thống kê thực sự có thể bảo vệ

1. **Statistical unit là location.** Các timepoints, features và mask repetitions trong cùng location phụ thuộc nhau. Existing report gộp seeds trong location rồi bootstrap 2000 draws theo location; giữ nguyên clustering này. Không lấy SD của 15 fold×seed thành standard error của 15 thí nghiệm độc lập; folds có training sets chồng lấp.
2. **Chỉ rõ estimand.** Slide 19 dùng pooled MAE và pooled cluster CI. Slide 20 dùng paired location-macro differences và paired CI. Hai loại có trọng số khác nhau; không gắn paired macro CI vào pooled difference.
3. **Ghép cặp đúng.** Cùng location/mask/labels cho hai methods. Với whole-feature, các paired macro Δ hiện có: Hron−MICoDa=0.015349 (CI95 [0.007936,0.023518]); Hron−Tsagris=0.027582 ([0.018548,0.037508]); MICoDa−Tsagris=0.012233 ([0.008916,0.016094]). Dấu âm ưu tiên method đứng trước. Chuyển toàn bộ điểm và CI sang pp bằng nhân 100.
4. **Báo effect size và CI trước p-value.** Existing CI là mô tả fixed-fit OOS errors; không mô tả uncertainty do retraining, hyperparameter selection hoặc native missing. Không suy ý nghĩa từ việc marginal CI có chồng nhau hay không.
5. **Không giả xác nhận tiền đăng ký cho phân tích mới.** Current benchmark đã có primary scenario/selection rule đăng ký. Các diễn giải/so sánh bổ sung sau khi nhìn kết quả là exploratory. Ba CI ghép cặp hiện có không phải family-wise adjusted CI. Trên slide hiện có, ghi “CI95 mô tả; chưa hiệu chỉnh đa so sánh”, không gọi cả ba là confirmatory significance.
6. **Nếu sau này cần kiểm định xác nhận:** chốt trước primary contrast/endpoint và meaningful-error threshold với người dùng; cần threshold dựa domain chứ không chọn để đạt ý nghĩa. Nếu có nhiều contrasts, định nghĩa family, dùng kiểm định ghép cặp ở cluster level với giả định rõ và hiệu chỉnh Holm cho p-values, hoặc simultaneous bootstrap intervals phù hợp. Không gọi ordinary CI95 là Holm-adjusted CI. Không tự thêm p-values trong lượt plan.
7. **Không ép ranking diagnostics.** Full-truth composition chủ yếu thuộc 5 countries có complete data; correlation support có thể khác giữa methods. Ghi support, mẫu số, missingness mechanism trước so sánh. Kết quả time-block/whole-row chưa có paired effect evidence được trích ở đây, nên MICoDa chỉ được gọi là point estimate thấp hơn.

## Những caveat cần hiện ngay trên plot

- MICoDa-V2 chưa đạt tolerance sau 50 iterations ở cả 5 fits của accepted run.
- Hron fallback 45/35013 rows ở whole-feature, nhưng 10701/35013 ở row/block; không gọi mọi prediction là cùng donor mechanism.
- Hron-LS/MICoDa-V1 vượt supplementary budget 180s/fold-method; Hron-LTS lỗi số học. Không cho missing score bằng 0, không ranking trên subset folds thành công.
- Accuracy thuộc artificial hidden observations; native missing chưa có truth. Observed-lock và budget hợp lệ không chứng minh điền chính xác.
- Ba baseline cross-sectional; offline context không biến chúng thành temporal neural models.

## Synthetic — thiết kế đề xuất, CHƯA CHẠY

Mục tiêu: có full truth trên mọi ô, kiểm tra robustness khi temporal dynamics, zeros, depth, correlation và missingness thay đổi. Không chỉ nhân thêm số dòng hay tạo dữ liệu từ chính một baseline rồi tuyên bố baseline đó tốt nhất.

Generator đề xuất: latent logistic-normal compositions với cross-feature covariance biết trước và dynamics AR/change-point theo entity; multinomial hoặc Dirichlet-multinomial counts với sequencing depth riêng; Other là thành phần tường minh. Tách sampling zeros khỏi structural zeros nếu có generator cho loại sau. Dùng ít nhất một generator độc lập khác (ví dụ mixture/transition compositions) để kiểm độ phụ thuộc vào giả định generator đầu. Kiểm tổng counts, zero rate, autocorrelation, feature relationships, depth distribution trước benchmark.

| Trục | Sweep đề xuất ban đầu | Lý do |
|---|---|---|
| Missing rate | 10%, 30%, 50%, 70% | Độ bền khi ít observed anchors |
| Mask structure | cell / feature / row / time block | Tái dùng bốn scenario; không đồng nhất pattern với MCAR/MAR/MNAR |
| Mechanism | MCAR; MAR theo approved observed metadata; MNAR theo latent truth | MNAR dùng truth ở generator để sinh mask, tuyệt đối không lộ truth cho model |
| Dynamics | weak / strong persistence; abrupt transition | Đo khôi phục temporal structure và đổi regime |
| Sparsity / depth | zero rate và sequencing depth thấp/cao | Tách khó khăn log-ratio với counts scale |
| Relationship / shift | covariance yếu/mạnh; entity distribution shift | Đánh giá transfer, không chỉ same-distribution interpolation |

Không chạy full Cartesian grid ngay. Sau khi người dùng yêu cầu chạy: smoke generator → protocol audit → chọn anchor design và one-factor sensitivity, thêm ít tương tác có lý do → chốt compute budget → chạy. Independent generation seeds khác mask seeds; cùng synthetic realization và mask cho ba methods. Chọn số independent repetitions theo Monte Carlo standard error/CI stability của primary effect và tài nguyên sau pilot; không mặc định 3 mask seeds đủ power. Không tự đặt stopping rule sau khi thấy method muốn thắng.

Split theo synthetic entities, fit preprocessing/donor/metadata trên train, chọn tham số validation rồi test. Nested uncertainty: bootstrap independent generated datasets trước, location/entity clusters bên trong nếu cần; luôn giữ pairings giữa methods và mọi mask repetitions. Báo recovery trên formerly native missing vì synthetic có truth, nhưng diễn giải domain gap với real data. Mọi thay đổi solver/iteration budget/zero policy sau accepted real-data cần experiment ID và protocol mới.

Figure slots sẽ bổ sung: (1) error vs missing rate với CI; (2) per-mechanism paired-effect forest; (3) temporal/relationship recovery; (4) convergence/fallback/runtime. Khi chưa có dữ liệu chỉ hiện sơ đồ thiết kế, không vẽ line/CI minh họa trông như kết quả.
