# EDA cho baseline trên easy, medium và hard

Các biểu đồ dùng test split bị che của cùng các run v2, standard scale, seed 2026. MCAR và MAR được tách riêng. Métrics theo ô chỉ tính hidden cells. Chỉ có một seed/profile, nên đây là mô tả các run này, không phải ước lượng ổn định qua nhiều lần lặp.

## Những gì biểu đồ cho thấy

- Sample-ratio MAE (X/N) và latent-p MAE tăng khi chuyển sang profile hard với cả Hron và Tsagris. MICoDa v2 ở khoảng 7.4–7.6% error qua cả ba profile trong lần chạy này; đó là một quan sát benchmark cụ thể, chưa đủ để kết luận chung về phương pháp.
- Count MAE nhỏ đi ở medium/hard vì tổng N nhỏ hơn nhiều. Vì vậy hàng count dùng trục log và không nên đọc là hard dễ impute hơn. Sample-ratio và latent-p MAE cho so sánh theo tỷ lệ rõ hơn.
- Phân tầng cho thấy sai số lớn nhất ở positive-count cells, nhất là MICoDa trong run này; sampling-zero/true-zero có mức sai số khác nhiều. Xem riêng hình phân tầng.

## Vì sao chạy khoảng nửa phút

Mỗi profile có cùng shape **60 entities × 120 thời điểm × 12 targets**. Train/validation/test là entity-disjoint; candidate chỉ chạy trên train 4,320 rows và validation 1,440 rows. Easy/medium/hard thay đổi phân phối/độ lớn N và cấu hình latent, không làm ma trận lớn hơn.

Mỗi profile chạy 30 validation candidates: 3 Hron k, 3 MICoDa W, 9 Tsagris k×alpha, cho hai masks; sau đó 6 model được fit lại trên train+validation và predict test. Đây là các numerical cores CPU gọn trên 12 chiều, không có GPU training, neural network, posterior sampling, hoặc hàng trăm replicates. Mean một lượt candidate gồm fit và validation prediction thường dưới 1.1 giây; toàn run gồm đọc/kiểm tra bundle, chấm, lưu predictions/diagnostics/report nên mất 28–36 giây trên máy này.

| Profile | Toàn run | Hron kNN, mean/candidate | MICoDa v2, mean/candidate | Tsagris alpha-kNN, mean/candidate |
|---|---:|---:|---:|---:|
| easy | 28.4 s | 0.73 s | 0.46 s | 0.77 s |
| medium | 34.7 s | 0.89 s | 0.78 s | 0.86 s |
| hard | 36.2 s | 0.86 s | 1.04 s | 0.82 s |

## Mức độ bám paper

Có: các **core** là các biến thể paper-derived đã triển khai trong repo. Run này không chạy toàn bộ phương pháp hay tái tạo thí nghiệm gốc của các paper:

- Hron chỉ chạy `HronKNN`, với k = 3/5/10; không chạy iterative LS/LTS.
- MICoDa chạy v2 ALR core với W = 1e-6/1e-4/1e-3, tối đa 50 vòng; không phải multiple-imputation uncertainty study.
- Tsagris chạy `TsagrisKNN` với alpha mean, chọn k/alpha bằng validation grid. Không chạy `TsagrisAdaptive` repeated leave-N-out CV của core khác trong repo.
- Cả ba được dùng qua `CountBaselineAdapter`: counts/N, một phần residual “Other”, khóa observed values và enforce count budget; fallback train_mean và degenerate policies được bật. Đó là protocol adaptation cho benchmark count, không tuyên bố parity với pipeline thí nghiệm nguyên bản.

Vì vậy thời gian ngắn là bình thường cho phạm vi này, còn con số runtime không chứng minh full-paper reproduction hoặc hiệu quả ngoài dataset/profile/seed đang xét.

## Tệp

- `profile_test_errors.png`: test MAE theo profile, hai masks và ba baseline.
- `latent_error_by_stratum.png`: latent-p MAE trong true-zero, sampling-zero, positive-count cells.
- `runtime_by_profile.png`: thời gian candidate và toàn suite.
- `test_metrics.csv`, `candidate_timing.csv`, `run_timing.csv`: dữ liệu dùng vẽ.
- `plot_eda.R`: script vẽ bằng base R, không cần package bổ sung.
