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
  - `zero_fill`, `mean_fill`, `marginal_sampler`: mức sàn, `enabled = false`.
- **`run.py`:** đăng ký 4 model. Model cần ngữ cảnh nhận N, ngày, nhóm, entity (không nhận counts). Thêm khóa `enabled = false` và cờ `--only NAME ...`.
- **Config:** `models_default.toml` (4 bảng mới); `baselines_covariants.toml` (`time_format`, `group_col`, `group_map`); `models_micoda_other.toml` (MICoDa `other = true` cho lượt so sánh riêng).
- **Test:** 156 pass (115 cũ + 41 mới). Có test "ô bị che = 1e100 không đổi output", tái lập theo seed, tham chiếu same_day_mean tính tay.
- **Docs:** README, mục mới trong `THREE_BASELINE_CONTRACT_VI.md`, báo cáo `tv2_step1/REPORT_vi.md`, bảng `tv2_step1/results_multiseed.md`.

## Kết quả chính

COVID đầy đủ, mask `whole_feature` 5%, seed 0–4, nhóm ô do model quyết định; mean ± sd:

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

- same_day_mean thắng Hron 5/5 seed (cả MAE và MSE, cả hai mask).
- Hron thắng Tsagris về MAE 5/5 nhưng thua về MSE 5/5.
- **MICoDa A17/R1 thua mean_fill** trên whole_feature (MAE thua 4/5 seed, MSE thua 5/5). A18/R2 hơn mean_fill về MAE, còn MSE gần như hòa.
- Hạn chế: 25.5% ô thiếu thật phải dùng trung bình toàn cục của same_day_mean, trong khi ô bị che chỉ 0.7–9%. Vì vậy điểm masked lạc quan hơn thực tế.

## Cần reviewer xem

1. **Quy tắc phần dư** của 4 model mới: khi tổng đề xuất vượt R thì co lại cho bằng R, phần còn lại vào Other. Cách này khác Hron/Tsagris (dự đoán cả Other rồi chia R theo tỷ lệ).
2. **Bật `same_day_mean` mặc định** nên lượt chạy mặc định có thêm một model. Output của Hron/Tsagris/MICoDa vẫn không đổi.
3. **Cơ chế `enabled` + `--only`** thay cho việc đổi nghĩa `--models`.
4. **Map châu lục** theo quy ước Our World in Data (Turkey/Georgia thuộc Asia, Russia thuộc Europe, Caribbean thuộc North America).
5. **mean_fill** dùng trung bình không trọng số; mức toàn cục của same_day_mean dùng trọng số N và lấy trên mọi nhóm. Khi `exclude_same_location = true` mà không còn ô nào thì báo lỗi, không có fallback ngầm.
6. **Time không phải ngày:** dạng số xét cùng mốc rồi ±1, ±2; dạng nhãn chỉ xét cùng nhãn.
7. **MSE count của MICoDa A18** không khớp ghi chú D08 (×1.02 ở random_cell, so với ~×4.5 trong D08). Chưa điều tra.

## Để lại cho step 2

- Tên biến thể trong `run.py` (ví dụ `[same_day_mean_wide]`, `[micoda_other]`).
- `evaluate.py`: bootstrap cho so sánh ghép cặp, chọn metric chính, thống kê mức cửa sổ trên ô bị che, Wasserstein cho lượt masked.
- Độ nhạy của same_day_mean theo tham số, và cách giảm tỷ lệ rơi về toàn cục trên ô thiếu thật.
- Điều tra MICoDa: vì sao A17 thua mean_fill, và MSE count của A18.
- Chạy thử với dữ liệu microbiome thật.
