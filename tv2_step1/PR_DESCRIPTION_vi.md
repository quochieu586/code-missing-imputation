# Step 1: baseline trung bình cùng ngày + ba mức sàn, đọc `time` như ngày thật

## Mục đích

Đặt mốc so sánh trước khi làm model chính:
- **Mức sàn:** biết điểm của một model "không học gì".
- **Baseline đơn giản dùng thời gian:** trung bình cùng ngày, cùng châu lục.

Không sửa `evaluate.py`. Hron/Tsagris/MICoDa không đổi: so với `main`, output giống từng byte trên cùng cấu hình.

## Thay đổi

- **`data.py`:** cột `time` được đọc thành `days`/`week`.
  - Ngày ISO hoặc theo `time_format` thành số ngày; dạng số (mốc thăm khám) giữ nguyên giá trị; chữ khác thành nhãn.
  - Ngày sai hoặc trộn định dạng thì báo lỗi kèm dòng CSV.
  - Thêm khóa schema tùy chọn `group_col`/`group_map`. COVID dùng châu lục qua `configs/covid_location_continent.csv`.
- **Baseline mới** (mỗi model một file trong `baselines/`):
  - `same_day_mean`: trung bình c/N có trọng số N, cùng nhóm, trong ±28 ngày. Nới cửa sổ ×2 tối đa 2 lần khi có ít hơn 20 ô, sau đó dùng trung bình toàn cục. Số ô ở mỗi mức được ghi trong `info_`. **Bật mặc định.**
  - `zero_fill`, `mean_fill`, `marginal_sampler`: mức sàn, `enabled = false`. marginal_sampler bị co theo phần dư nên không đảm bảo Wasserstein ≈ 0.
- **`run.py`:** đăng ký 4 model. Model cần ngữ cảnh nhận N, ngày, nhóm, entity (không nhận counts). Thêm khóa `enabled = false` và cờ `--only NAME ...`.
- **Config:** `models_default.toml` (4 bảng mới); `baselines_covariants.toml` (`time_format`, `group_col`, `group_map`); `models_micoda_other.toml` (MICoDa `other = true` cho lượt so sánh riêng).
- **Test:** 177 test trong `tests` (115 cũ + 62 mới) và 8 test cho `tv2_step1/summarize_runs.py`. Có test "ô bị che = 1e100 không đổi output", tái lập theo seed, tham chiếu same_day_mean tính tay.
- **Docs:** README, mục mới trong `THREE_BASELINE_CONTRACT_VI.md`, báo cáo `tv2_step1/REPORT_vi.md`, bảng `tv2_step1/results_multiseed.md`, manifest `tv2_step1/results_manifest.json`; sửa theo review: `tv2_step1/FIXES_PR4_vi.md`.

## Kết quả chính

COVID đầy đủ, mask `whole_feature` 5%, seed 0–4, nhóm ô do model quyết định. mean ± sd, trong đó sd là độ lệch chuẩn mẫu qua 5 seed, **không phải khoảng tin cậy**. Đây là lượt che giả (masked pilot), không phải đánh giá trên ô thiếu thật:

| Model | MAE tỷ lệ | MSE tỷ lệ | MAE count |
|---|---:|---:|---:|
| same_day_mean | **0.0351 ± 0.0066** | **0.0142 ± 0.0036** | **29.4 ± 26.8** |
| hron | 0.0875 ± 0.0105 | 0.0660 ± 0.0092 | 74.6 ± 113.7 |
| micoda_other (A18/R2) | 0.1075 ± 0.0093 | 0.0684 ± 0.0078 | 188.7 ± 205.5 |
| tsagris | 0.1079 ± 0.0209 | 0.0521 ± 0.0095 | 33.2 ± 14.7 |
| mean_fill (sàn) | 0.1354 ± 0.0145 | 0.0704 ± 0.0084 | 218.2 ± 186.8 |
| marginal_sampler (sàn) | 0.1354 ± 0.0120 | 0.1107 ± 0.0122 | 225.1 ± 200.7 |
| micoda (A17/R1, mặc định) | 0.1406 ± 0.0099 | 0.0823 ± 0.0094 | 236.2 ± 188.5 |
| zero_fill (sàn) | 0.1693 ± 0.0219 | 0.1525 ± 0.0230 | 287.1 ± 211.3 |

- same_day_mean thắng Hron 5/5 seed về MAE và MSE tỷ lệ, ở cả hai mask: nó **tốt hơn các baseline trong lượt che giả này**, chưa phải tốt nhất trên ô thiếu thật. Lợi thế đến từ bốn yếu tố cùng lúc: thời gian, châu lục, trung bình có trọng số N và quy tắc co phần dư. Chưa tách được đóng góp của từng yếu tố; ablation để step sau.
- Hron thắng Tsagris về MAE tỷ lệ 5/5; Tsagris thắng Hron về MSE tỷ lệ 5/5 ở cả hai mask, nhưng về MAE count chỉ 2/5 (whole_feature) và 0/5 (random_cell).
- **MICoDa A17/R1 thua mean_fill** trên whole_feature (MAE thua 4/5 seed, MSE thua 5/5). A18/R2 hơn mean_fill về MAE, còn MSE gần như hòa.
- **Hạn chế về support:** 25,5% mọi ô thiếu thật phải dùng trung bình toàn cục của same_day_mean, so với 1,4–9% ô che thuộc nhóm model (whole_feature). Hai tỷ lệ khác mẫu số nên không so trực tiếp; đây là cảnh báo, không phải phép đo độ chính xác trên ô thiếu thật.

### Thắng/thua theo seed (sinh tự động từ `results_multiseed.csv`)

<!-- auto:pairs:start -->
**whole_feature** (thắng = sai số nhỏ hơn trên nhóm ô model; Δ = trung bình A − B; × = trung bình tỷ số A / B theo seed)

| A vs B | MAE tỷ lệ | MSE tỷ lệ | MAE count | MSE count |
|---|---|---|---|---|
| same_day_mean vs hron | 5/5, Δ -0.05247, ×0.41 | 5/5, Δ -0.0518, ×0.224 | 4/5, Δ -45.16, ×0.637 | 4/5, Δ -4.95e+06, ×0.592 |
| tsagris vs hron | 0/5, Δ +0.02036, ×1.23 | 5/5, Δ -0.01396, ×0.787 | 2/5, Δ -41.33, ×0.979 | 3/5, Δ -5.229e+06, ×1.11 |
| micoda_other vs mean_fill | 5/5, Δ -0.02785, ×0.796 | 4/5, Δ -0.001956, ×0.973 | 4/5, Δ -29.55, ×0.815 | 2/5, Δ +1.3e+06, ×0.938 |
| micoda vs mean_fill | 1/5, Δ +0.005246, ×1.05 | 0/5, Δ +0.01192, ×1.18 | 1/5, Δ +17.94, ×1.33 | 2/5, Δ -6.169e+06, ×1.56 |
| micoda_other vs micoda | 5/5, Δ -0.03309, ×0.765 | 5/5, Δ -0.01387, ×0.833 | 4/5, Δ -47.49, ×0.708 | 4/5, Δ +7.469e+06, ×1.22 |

**random_cell** (thắng = sai số nhỏ hơn trên nhóm ô model; Δ = trung bình A − B; × = trung bình tỷ số A / B theo seed)

| A vs B | MAE tỷ lệ | MSE tỷ lệ | MAE count | MSE count |
|---|---|---|---|---|
| same_day_mean vs hron | 5/5, Δ -0.05825, ×0.348 | 5/5, Δ -0.05021, ×0.181 | 5/5, Δ -3.929, ×0.825 | 1/5, Δ +2.976e+04, ×1.72 |
| tsagris vs hron | 0/5, Δ +0.0107, ×1.12 | 5/5, Δ -0.01304, ×0.789 | 0/5, Δ +3.633, ×1.17 | 0/5, Δ +1.357e+04, ×1.51 |
| micoda_other vs mean_fill | 5/5, Δ -0.02931, ×0.786 | 1/5, Δ +0.0003484, ×1.01 | 5/5, Δ -42.84, ×0.709 | 4/5, Δ -8.011e+05, ×0.681 |
| micoda vs mean_fill | 5/5, Δ -0.005373, ×0.961 | 0/5, Δ +0.01111, ×1.16 | 1/5, Δ +15.33, ×1.11 | 3/5, Δ -7.884e+05, ×0.874 |
| micoda_other vs micoda | 5/5, Δ -0.02394, ×0.818 | 5/5, Δ -0.01076, ×0.866 | 5/5, Δ -58.17, ×0.644 | 3/5, Δ -1.267e+04, ×1.02 |

Dạng câu:

- whole_feature: same_day_mean thắng hron ở MAE tỷ lệ 5/5, MSE tỷ lệ 5/5, MAE count 4/5, MSE count 4/5 seed.
- random_cell: same_day_mean thắng hron ở MAE tỷ lệ 5/5, MSE tỷ lệ 5/5, MAE count 5/5, MSE count 1/5 seed.
- whole_feature: tsagris thắng hron ở MAE tỷ lệ 0/5, MSE tỷ lệ 5/5, MAE count 2/5, MSE count 3/5 seed.
- random_cell: tsagris thắng hron ở MAE tỷ lệ 0/5, MSE tỷ lệ 5/5, MAE count 0/5, MSE count 0/5 seed.
- whole_feature: micoda_other thắng mean_fill ở MAE tỷ lệ 5/5, MSE tỷ lệ 4/5, MAE count 4/5, MSE count 2/5 seed.
- random_cell: micoda_other thắng mean_fill ở MAE tỷ lệ 5/5, MSE tỷ lệ 1/5, MAE count 5/5, MSE count 4/5 seed.
- whole_feature: micoda thắng mean_fill ở MAE tỷ lệ 1/5, MSE tỷ lệ 0/5, MAE count 1/5, MSE count 2/5 seed.
- random_cell: micoda thắng mean_fill ở MAE tỷ lệ 5/5, MSE tỷ lệ 0/5, MAE count 1/5, MSE count 3/5 seed.
- whole_feature: micoda_other thắng micoda ở MAE tỷ lệ 5/5, MSE tỷ lệ 5/5, MAE count 4/5, MSE count 4/5 seed.
- random_cell: micoda_other thắng micoda ở MAE tỷ lệ 5/5, MSE tỷ lệ 5/5, MAE count 5/5, MSE count 3/5 seed.
<!-- auto:pairs:end -->

## Cần reviewer xem

1. **Quy tắc phần dư** của 4 model mới: khi tổng đề xuất vượt R thì co lại cho bằng R, phần còn lại vào Other. Cách này khác Hron/Tsagris (dự đoán cả Other rồi chia R theo tỷ lệ).
2. **Bật `same_day_mean` mặc định** nên lượt chạy mặc định có thêm một model. Output của Hron/Tsagris/MICoDa vẫn không đổi.
3. **Cơ chế `enabled` + `--only`** thay cho việc đổi nghĩa `--models`.
4. **Map châu lục** theo quy ước Our World in Data (Turkey/Georgia thuộc Asia, Russia thuộc Europe, Caribbean thuộc North America).
5. **mean_fill** dùng trung bình không trọng số; mức toàn cục của same_day_mean dùng trọng số N và lấy trên mọi nhóm. Khi `exclude_same_location = true` mà không còn ô nào thì báo lỗi, không có fallback ngầm.
6. **Time không phải ngày:** dạng số xét cùng mốc rồi ±1, ±2; dạng nhãn chỉ xét cùng nhãn.
7. **MSE count của MICoDa A18** không khớp ghi chú D08 (×1,22 ở whole_feature, ×1,02 ở random_cell, so với ~×4,5 trong D08). **Chưa kiểm chứng độc lập**, chưa điều tra. Số này lấy từ các run đã kiểm tra bằng manifest và nằm trong CSV.

## Để lại cho step 2

- Tên biến thể trong `run.py` (ví dụ `[same_day_mean_wide]`, `[micoda_other]`).
- `evaluate.py`: bootstrap cho so sánh ghép cặp, chọn metric chính, thống kê mức cửa sổ trên ô bị che, Wasserstein cho lượt masked.
- Ablation same_day_mean (thời gian, châu lục, trọng số N, quy tắc co phần dư).
- Độ nhạy của same_day_mean theo tham số, và cách giảm tỷ lệ rơi về toàn cục trên ô thiếu thật.
- Điều tra MICoDa: vì sao A17 thua mean_fill, và MSE count của A18.
- Chạy thử với dữ liệu microbiome thật.
