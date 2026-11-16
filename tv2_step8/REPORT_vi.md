# Step 8: đóng băng evaluation v1 và chạy lượt xác nhận (chỉ baseline)

Branch `step8-eval-v1-confirm` (từ `step7-matched-mask`). Không có `iterative_net` hay `framework/`. Không sửa baseline, không đổi metric, diagnostic, ngưỡng, use case, mask hay cấu hình model so với step 2–7. Mọi bảng số trong khối `auto:*` do `evaluate_v1.py` sinh, không sửa tay; bảng đầy đủ ở [v1_results.md](v1_results.md).

Đây là **lượt xác nhận**: chạy một lần trên mask seed 200–209, chưa step nào dùng. Kết luận chỉ nói về **biến thiên theo mask trên một bảng cố định**. Không kết luận nào là độ chính xác trên ô thiếu thật, và không kết luận nào tổng quát hóa sang nước mới.

## 1. File đã sửa/thêm

| File | Nội dung |
|---|---|
| `tv2_step8/eval_v1.toml` | Toàn bộ lựa chọn của v1 và giả thuyết H1–H9. |
| `tv2_step8/v1.py` (157 dòng) | Đọc config (khóa lạ hoặc thiếu thì dừng), kiểm seed chưa dùng, sign test chính xác, chính sách fail, chấm giả thuyết. |
| `tv2_step8/evaluate_v1.py` (403 dòng) | Bốn lệnh `check` / `masks` / `run` / `evaluate`. Mọi phép chấm gọi `tier01`, `tier2`, `tier3`, `matched`. |
| `tv2_step8/test_v1.py` | 12 test. |
| `tv2_step6/tier3.py` | `load_mask` lấy seed của ca từ tham số (mặc định 0–4). Output step 6 và step 7 sinh lại giống từng byte. |
| `tv2_step8/masks_manifest.json` | Hash của 30 mask xác nhận, sinh trước khi chạy model. |
| `tv2_step8/v1_results.md`, `v1_pairs.csv`, `v1_cases.json`, `v1_manifest.json` | Output sinh tự động. Thư mục run và bản chạy lại Tầng 2 không commit; hash nằm trong manifest. |

## 2. Evaluation v1

**Tóm tắt `eval_v1.toml`:**

- **Dữ liệu:** COVID đầy đủ, rate 0.05.
- **Mask:**
  - `whole_feature` (chính);
  - `whole_feature_matched` (sensitivity bắt buộc, π của step 7);
  - `random_cell` (sensitivity).
- **Seed:** 200–209. Script từ chối chạy nếu seed nào trong dải này có trong config hoặc manifest của step 1–7.
- **Model:**
  - model thật: hron, hron_cap, tsagris, micoda, micoda_other, same_day_mean;
  - đối thủ giả: mean_fill, zero_fill, marginal_sampler;
  - chỉ ở Tầng 1: tsagris_cap, micoda_other_cap.
  
  Ba lượt chạy như step 2–7: `main`, `micoda_other`, `residual`.
- **Tầng 0:** báo mọi run đã lên lịch. Model fail ở > 2/10 seed của một mask thì mọi giả thuyết liên quan tới nó ở mask đó là "không đủ support". Ghép cặp chỉ trên seed chung, có nhãn "có điều kiện".
- **Tầng 1:**
  - metric chính: MAE tỷ lệ trên ô `model_cell`, cell-weighted trong từng seed;
  - metric phụ: MSE tỷ lệ, MAE count, series-macro, các nhóm theo đáp án, theo châu lục và nhóm N (ngưỡng của step 5, tầng < 30 ô thì NA);
  - ghép cặp A − B theo seed: mean, sd, số seed thắng, sign test chính xác hai phía (bỏ hòa) làm thống kê mô tả. Không bootstrap.
- **Tầng 2:** D1/D2/D3 của step 5, chạy lại và phải giống từng byte.
- **Tầng 3:** đặc tả step 6 không đổi, 10 ca mỗi mask. "Cả ba" là đạt nếu cả ba mask đạt, không đạt nếu một mask không đạt, còn lại là không đủ support.
- **Cách dùng diagnostic** (nhãn, không phải logic):
  - D3 là cờ;
  - D1 chỉ cảnh báo điền thừa, đọc so với hàng truth;
  - D2 chỉ mô tả, không dùng để xếp hạng.

**Giả thuyết:** chín giả thuyết của brief, quyết định trên mask chính.
- H1–H4, H6: A có metric thấp hơn B, xác nhận khi Δ trung bình < 0 và A thắng ≥ 8/10 seed. H2 phải đạt với từng model thật còn lại.
- H5: micoda thắng mean_fill ≤ 5/10 seed.
- H7–H9: kết luận Tầng 3 trên mask chính.

**Commit:**
- `996feba` đóng băng Phần A + B, trước khi sinh mask;
- `c13308b` sinh mask và commit hash;
- `9954fd2` sửa bug Tầng 3 phát hiện sau bước 3 (mục 6);
- `4b8c065` sửa phần trình bày của bảng 8 (mục 6);
- `4535924` commit số của lượt chạy thứ hai.

Các run của lượt chạy thứ hai được chạy từ commit `6303a6d`, git tree sạch.

## 3. Kiểm tra

<!-- auto:checks:start -->
Seed xác nhận 200–209: không xuất hiện trong 29 file config / manifest của step 1–7 (`v1.check_unused`).
Mask: 30 mask đã lưu khớp `masks_manifest.json`; F2 đạt ở cả ba mask; long file khớp `summary.json` (tier01.load_run).
<!-- auto:checks:end -->

- **Test:** 346 pass với `-W error::RuntimeWarning`:
  - nextgen 303;
  - script 43, trong đó 12 test mới của step 8 và 1 test hồi quy mới cho bug ở mục 6.
- **Test mới kiểm:**
  - `eval_v1.toml` đọc đúng; khóa lạ hoặc thiếu (ở top level, ở section, trong một giả thuyết) đều dừng script trước khi ghi `masks_manifest.json`;
  - seed trùng với config hoặc manifest giả của step 1–7 bị từ chối;
  - sign test chính xác tính tay, kể cả khi có hòa (hòa bị bỏ);
  - "không đủ support" khi model fail > 2/10 seed (2 seed fail vẫn được chấm);
  - chấm giả thuyết trên toy: xác nhận / không xác nhận / không đủ support, H2 nhiều cặp, H5 "không tốt hơn", Tầng 3, quy tắc ba mask;
  - Tầng 2 chạy qua `evaluate_v1.py` giống từng byte output step 5;
  - lệnh `run` sinh đúng 9 lượt theo config.
- **Seed chưa dùng:** tập seed tìm thấy trong 29 file của step 1–7 là {0, 1, 2, 3, 4, 109}.
- **Tầng 2:** khớp step 5 ở cả 5 file (bảng 5). CSV native của 11 model trong lượt chạy step 8 giống từng byte với CSV step 2/4 mà step 5 đã dùng (cờ `rerun_identical` của manifest).
- **Tái lập:**
  - `evaluate_v1.py` chạy hai lần (hai thư mục output) cho 4 file output giống từng byte;
  - hai lượt chạy model giống nhau ở 135/144 file (CSV completed, long file, mask, `errors.log`); 9 file khác nhau đều là `summary.csv`, chỉ vì có cột thời gian chạy.

## 4. Bảng

### Bảng 7: giả thuyết H1–H9

<!-- auto:hypotheses:start -->
| ID | Giả thuyết | wf (chính) | wfm | rc | Chi tiết ở mask chính |
|---|---|---|---|---|---|
| H1 | same_day_mean has a lower MAE of c / N than hron | **xác nhận** | xác nhận | xác nhận | vs hron: Δ -0.0683, thắng 8/8, p 0.008 |
| H2 | same_day_mean has a lower MAE of c / N than every other real model | **xác nhận** | xác nhận | xác nhận | vs hron: Δ -0.0683, thắng 8/8, p 0.008; vs hron_cap: Δ -0.0682, thắng 8/8, p 0.008; vs tsagris: Δ -0.0738, thắng 8/8, p 0.008; vs micoda: Δ -0.1075, thắng 8/8, p 0.008; vs micoda_other: Δ -0.0870, thắng 8/8, p 0.008 |
| H3 | hron has a lower MAE of c / N than tsagris | **không xác nhận** | xác nhận | xác nhận | vs tsagris: Δ -0.0054, thắng 6/8, p 0.289 |
| H4 | tsagris has a lower MSE of c / N than hron | **xác nhận** | xác nhận | xác nhận | vs hron: Δ -0.0205, thắng 8/8, p 0.008 |
| H5 | micoda (A17) does not have a lower MAE of c / N than mean_fill | **không xác nhận** | không xác nhận | không xác nhận | vs mean_fill: Δ -0.0154, thắng 8/8, p 0.008 |
| H6 | micoda_other has a lower MAE of c / N than micoda | **xác nhận** | xác nhận | xác nhận | vs micoda: Δ -0.0204, thắng 8/8, p 0.008 |
| H7 | tier 3: D3 passes U3 | **xác nhận** | xác nhận | xác nhận | - |
| H8 | tier 3: D1 passes U1, over-filling direction | **xác nhận** | xác nhận | xác nhận | - |
| H9 | tier 3: D1 does not pass U1, under-filling direction | **xác nhận** | xác nhận | xác nhận | - |
<!-- auto:hypotheses:end -->

### Bảng 1: Tầng 0

<!-- auto:status:start -->
| Model | Lượt | Mask | Lên lịch | completed | partial | failed | missing | Run không ok |
|---|---|---|---:|---:|---:|---:|---:|---|
| hron | native | none | 3 | 3 | 0 | 0 | 0 | - |
| tsagris | native | none | 3 | 3 | 0 | 0 | 0 | - |
| micoda | native | none | 3 | 3 | 0 | 0 | 0 | - |
| same_day_mean | native | none | 3 | 3 | 0 | 0 | 0 | - |
| mean_fill | native | none | 3 | 3 | 0 | 0 | 0 | - |
| marginal_sampler | native | none | 3 | 3 | 0 | 0 | 0 | - |
| zero_fill | native | none | 3 | 3 | 0 | 0 | 0 | - |
| hron | masked | whole_feature | 10 | 8 | 0 | 2 | 0 | 200: failed (baseline_error); 203: failed (baseline_error) |
| tsagris | masked | whole_feature | 10 | 8 | 0 | 2 | 0 | 200: failed (baseline_error); 203: failed (baseline_error) |
| micoda | masked | whole_feature | 10 | 8 | 0 | 2 | 0 | 200: failed (insufficient_fit); 203: failed (insufficient_fit) |
| same_day_mean | masked | whole_feature | 10 | 10 | 0 | 0 | 0 | - |
| mean_fill | masked | whole_feature | 10 | 10 | 0 | 0 | 0 | - |
| marginal_sampler | masked | whole_feature | 10 | 10 | 0 | 0 | 0 | - |
| zero_fill | masked | whole_feature | 10 | 10 | 0 | 0 | 0 | - |
| micoda_other | native | none | 3 | 3 | 0 | 0 | 0 | - |
| micoda_other | masked | whole_feature | 10 | 8 | 0 | 2 | 0 | 200: failed (insufficient_fit); 203: failed (insufficient_fit) |
| hron_cap | native | none | 3 | 3 | 0 | 0 | 0 | - |
| tsagris_cap | native | none | 3 | 3 | 0 | 0 | 0 | - |
| micoda_other_cap | native | none | 3 | 3 | 0 | 0 | 0 | - |
| hron_cap | masked | whole_feature | 10 | 8 | 0 | 2 | 0 | 200: failed (baseline_error); 203: failed (baseline_error) |
| tsagris_cap | masked | whole_feature | 10 | 8 | 0 | 2 | 0 | 200: failed (baseline_error); 203: failed (baseline_error) |
| micoda_other_cap | masked | whole_feature | 10 | 8 | 0 | 2 | 0 | 200: failed (insufficient_fit); 203: failed (insufficient_fit) |
| hron | masked | whole_feature_matched | 10 | 10 | 0 | 0 | 0 | - |
| tsagris | masked | whole_feature_matched | 10 | 10 | 0 | 0 | 0 | - |
| micoda | masked | whole_feature_matched | 10 | 10 | 0 | 0 | 0 | - |
| same_day_mean | masked | whole_feature_matched | 10 | 10 | 0 | 0 | 0 | - |
| mean_fill | masked | whole_feature_matched | 10 | 10 | 0 | 0 | 0 | - |
| marginal_sampler | masked | whole_feature_matched | 10 | 10 | 0 | 0 | 0 | - |
| zero_fill | masked | whole_feature_matched | 10 | 10 | 0 | 0 | 0 | - |
| micoda_other | masked | whole_feature_matched | 10 | 10 | 0 | 0 | 0 | - |
| hron_cap | masked | whole_feature_matched | 10 | 10 | 0 | 0 | 0 | - |
| tsagris_cap | masked | whole_feature_matched | 10 | 10 | 0 | 0 | 0 | - |
| micoda_other_cap | masked | whole_feature_matched | 10 | 10 | 0 | 0 | 0 | - |
| hron | masked | random_cell | 10 | 10 | 0 | 0 | 0 | - |
| tsagris | masked | random_cell | 10 | 10 | 0 | 0 | 0 | - |
| micoda | masked | random_cell | 10 | 10 | 0 | 0 | 0 | - |
| same_day_mean | masked | random_cell | 10 | 10 | 0 | 0 | 0 | - |
| mean_fill | masked | random_cell | 10 | 10 | 0 | 0 | 0 | - |
| marginal_sampler | masked | random_cell | 10 | 10 | 0 | 0 | 0 | - |
| zero_fill | masked | random_cell | 10 | 10 | 0 | 0 | 0 | - |
| micoda_other | masked | random_cell | 10 | 10 | 0 | 0 | 0 | - |
| hron_cap | masked | random_cell | 10 | 10 | 0 | 0 | 0 | - |
| tsagris_cap | masked | random_cell | 10 | 10 | 0 | 0 | 0 | - |
| micoda_other_cap | masked | random_cell | 10 | 10 | 0 | 0 | 0 | - |
<!-- auto:status:end -->

### Bảng 2: Tầng 1

<!-- auto:tier1:start -->
**whole_feature**

Tập seed chung (mọi model completed, coverage = 1): 201, 202, 204, 205, 206, 207, 208, 209.
Seed 200 bị loại: hron failed (baseline_error); hron_cap failed (baseline_error); tsagris failed (baseline_error); micoda failed (insufficient_fit); micoda_other failed (insufficient_fit); tsagris_cap failed (baseline_error); micoda_other_cap failed (insufficient_fit).
Seed 203 bị loại: hron failed (baseline_error); hron_cap failed (baseline_error); tsagris failed (baseline_error); micoda failed (insufficient_fit); micoda_other failed (insufficient_fit); tsagris_cap failed (baseline_error); micoda_other_cap failed (insufficient_fit).

Sắp theo MAE tỷ lệ (nhóm ô model) trung bình trên tập seed chung. mean ± sd (ddof=1) qua seed; sd không phải khoảng tin cậy. **exploratory**.

| Model | MAE tỷ lệ | MSE tỷ lệ | MAE count | MAE tỷ lệ series-macro | MAE tỷ lệ, đáp án = 0 | MAE tỷ lệ, đáp án > 0 | MAE tỷ lệ, ô model đáp án = 0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| same_day_mean | 0.0298 ± 0.0070 | 0.0105 ± 0.0033 | 16.8 ± 11.1 | 0.0446 ± 0.0104 | 0.0031 ± 0.0014 | 0.0641 ± 0.0145 | 0.0083 ± 0.0038 |
| hron_cap | 0.0981 ± 0.0138 | 0.0725 ± 0.0126 | 28.0 ± 16.0 | 0.1062 ± 0.0109 | 0.0111 ± 0.0032 | 0.2084 ± 0.0354 | 0.0296 ± 0.0082 |
| hron | 0.0982 ± 0.0141 | 0.0725 ± 0.0126 | 24.8 ± 13.1 | 0.1061 ± 0.0106 | 0.0127 ± 0.0035 | 0.2019 ± 0.0357 | 0.0339 ± 0.0090 |
| tsagris | 0.1036 ± 0.0134 | 0.0519 ± 0.0084 | 27.6 ± 14.0 | 0.1163 ± 0.0138 | 0.0300 ± 0.0040 | 0.1412 ± 0.0286 | 0.0808 ± 0.0143 |
| tsagris_cap | 0.1036 ± 0.0134 | 0.0519 ± 0.0084 | 27.6 ± 14.0 | 0.1163 ± 0.0138 | 0.0300 ± 0.0040 | 0.1412 ± 0.0286 | 0.0808 ± 0.0143 |
| micoda_other | 0.1169 ± 0.0139 | 0.0745 ± 0.0105 | 88.4 ± 111.4 | 0.1230 ± 0.0136 | 0.0297 ± 0.0051 | 0.1762 ± 0.0271 | 0.0797 ± 0.0136 |
| micoda_other_cap | 0.1169 ± 0.0139 | 0.0745 ± 0.0105 | 88.4 ± 111.4 | 0.1230 ± 0.0136 | 0.0297 ± 0.0051 | 0.1762 ± 0.0271 | 0.0797 ± 0.0136 |
| micoda | 0.1373 ± 0.0155 | 0.0842 ± 0.0116 | 85.2 ± 64.9 | 0.1454 ± 0.0134 | 0.0483 ± 0.0053 | 0.1494 ± 0.0250 | 0.1297 ± 0.0173 |
| marginal_sampler | 0.1524 ± 0.0135 | 0.1253 ± 0.0118 | 123.6 ± 91.1 | 0.1436 ± 0.0142 | 0.0214 ± 0.0027 | 0.3048 ± 0.0149 | 0.0573 ± 0.0076 |
| mean_fill | 0.1527 ± 0.0141 | 0.0766 ± 0.0085 | 123.3 ± 94.0 | 0.1431 ± 0.0131 | 0.0249 ± 0.0038 | 0.2909 ± 0.0205 | 0.0666 ± 0.0091 |
| zero_fill | 0.1989 ± 0.0193 | 0.1814 ± 0.0184 | 191.1 ± 109.3 | 0.1674 ± 0.0195 | 0.0000 ± 0.0000 | 0.5179 ± 0.0379 | 0.0000 ± 0.0000 |

**whole_feature_matched**

Tập seed chung (mọi model completed, coverage = 1): 200, 201, 202, 203, 204, 205, 206, 207, 208, 209.

Sắp theo MAE tỷ lệ (nhóm ô model) trung bình trên tập seed chung. mean ± sd (ddof=1) qua seed; sd không phải khoảng tin cậy. **exploratory**.

| Model | MAE tỷ lệ | MSE tỷ lệ | MAE count | MAE tỷ lệ series-macro | MAE tỷ lệ, đáp án = 0 | MAE tỷ lệ, đáp án > 0 | MAE tỷ lệ, ô model đáp án = 0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| same_day_mean | 0.0363 ± 0.0052 | 0.0133 ± 0.0032 | 12.3 ± 9.5 | 0.0489 ± 0.0080 | 0.0046 ± 0.0010 | 0.0790 ± 0.0116 | 0.0118 ± 0.0029 |
| hron_cap | 0.1139 ± 0.0251 | 0.0855 ± 0.0203 | 19.6 ± 10.9 | 0.1069 ± 0.0206 | 0.0141 ± 0.0049 | 0.2499 ± 0.0451 | 0.0359 ± 0.0127 |
| hron | 0.1156 ± 0.0249 | 0.0866 ± 0.0199 | 18.6 ± 8.6 | 0.1089 ± 0.0201 | 0.0157 ± 0.0052 | 0.2476 ± 0.0441 | 0.0401 ± 0.0134 |
| micoda_other | 0.1230 ± 0.0244 | 0.0779 ± 0.0203 | 43.7 ± 24.7 | 0.1172 ± 0.0228 | 0.0322 ± 0.0092 | 0.1953 ± 0.0303 | 0.0824 ± 0.0243 |
| micoda_other_cap | 0.1230 ± 0.0244 | 0.0779 ± 0.0203 | 43.7 ± 24.7 | 0.1172 ± 0.0228 | 0.0322 ± 0.0092 | 0.1953 ± 0.0303 | 0.0824 ± 0.0243 |
| tsagris | 0.1260 ± 0.0201 | 0.0591 ± 0.0114 | 18.7 ± 6.8 | 0.1261 ± 0.0169 | 0.0355 ± 0.0074 | 0.1882 ± 0.0218 | 0.0908 ± 0.0193 |
| tsagris_cap | 0.1260 ± 0.0201 | 0.0591 ± 0.0114 | 18.7 ± 6.8 | 0.1261 ± 0.0169 | 0.0355 ± 0.0074 | 0.1882 ± 0.0218 | 0.0908 ± 0.0193 |
| micoda | 0.1326 ± 0.0242 | 0.0803 ± 0.0207 | 45.6 ± 23.4 | 0.1301 ± 0.0214 | 0.0445 ± 0.0096 | 0.1669 ± 0.0302 | 0.1138 ± 0.0257 |
| mean_fill | 0.1634 ± 0.0272 | 0.0804 ± 0.0150 | 66.9 ± 34.8 | 0.1423 ± 0.0236 | 0.0287 ± 0.0072 | 0.3208 ± 0.0330 | 0.0733 ± 0.0187 |
| marginal_sampler | 0.1642 ± 0.0305 | 0.1353 ± 0.0280 | 64.1 ± 30.0 | 0.1443 ± 0.0262 | 0.0238 ± 0.0069 | 0.3445 ± 0.0413 | 0.0608 ± 0.0178 |
| zero_fill | 0.2079 ± 0.0376 | 0.1906 ± 0.0366 | 103.5 ± 65.7 | 0.1690 ± 0.0299 | 0.0000 ± 0.0000 | 0.5694 ± 0.0695 | 0.0000 ± 0.0000 |

**random_cell**

Tập seed chung (mọi model completed, coverage = 1): 200, 201, 202, 203, 204, 205, 206, 207, 208, 209.

Sắp theo MAE tỷ lệ (nhóm ô model) trung bình trên tập seed chung. mean ± sd (ddof=1) qua seed; sd không phải khoảng tin cậy. **exploratory**.

| Model | MAE tỷ lệ | MSE tỷ lệ | MAE count | MAE tỷ lệ series-macro | MAE tỷ lệ, đáp án = 0 | MAE tỷ lệ, đáp án > 0 | MAE tỷ lệ, ô model đáp án = 0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| same_day_mean | 0.0298 ± 0.0017 | 0.0105 ± 0.0007 | 22.0 ± 6.5 | 0.0357 ± 0.0025 | 0.0035 ± 0.0005 | 0.0668 ± 0.0037 | 0.0090 ± 0.0012 |
| hron_cap | 0.0873 ± 0.0038 | 0.0608 ± 0.0039 | 36.7 ± 9.0 | 0.0875 ± 0.0048 | 0.0100 ± 0.0011 | 0.1966 ± 0.0091 | 0.0260 ± 0.0029 |
| hron | 0.0893 ± 0.0042 | 0.0622 ± 0.0040 | 30.1 ± 7.9 | 0.0898 ± 0.0055 | 0.0113 ± 0.0011 | 0.1961 ± 0.0089 | 0.0295 ± 0.0029 |
| tsagris | 0.0986 ± 0.0029 | 0.0462 ± 0.0023 | 29.9 ± 8.2 | 0.1036 ± 0.0044 | 0.0286 ± 0.0015 | 0.1420 ± 0.0087 | 0.0743 ± 0.0043 |
| tsagris_cap | 0.0986 ± 0.0029 | 0.0462 ± 0.0023 | 29.9 ± 8.2 | 0.1036 ± 0.0044 | 0.0286 ± 0.0015 | 0.1420 ± 0.0087 | 0.0743 ± 0.0043 |
| micoda_other | 0.1062 ± 0.0048 | 0.0674 ± 0.0044 | 115.9 ± 38.1 | 0.1069 ± 0.0064 | 0.0235 ± 0.0021 | 0.1865 ± 0.0076 | 0.0611 ± 0.0055 |
| micoda_other_cap | 0.1062 ± 0.0048 | 0.0674 ± 0.0044 | 115.9 ± 38.1 | 0.1069 ± 0.0064 | 0.0235 ± 0.0021 | 0.1865 ± 0.0076 | 0.0611 ± 0.0055 |
| micoda | 0.1293 ± 0.0044 | 0.0768 ± 0.0042 | 162.7 ± 19.5 | 0.1302 ± 0.0063 | 0.0426 ± 0.0015 | 0.1621 ± 0.0070 | 0.1108 ± 0.0044 |
| mean_fill | 0.1372 ± 0.0025 | 0.0694 ± 0.0024 | 164.8 ± 44.0 | 0.1283 ± 0.0047 | 0.0214 ± 0.0013 | 0.2827 ± 0.0059 | 0.0556 ± 0.0034 |
| marginal_sampler | 0.1378 ± 0.0059 | 0.1129 ± 0.0058 | 163.3 ± 45.4 | 0.1287 ± 0.0072 | 0.0170 ± 0.0022 | 0.3046 ± 0.0105 | 0.0443 ± 0.0060 |
| zero_fill | 0.1740 ± 0.0036 | 0.1575 ± 0.0038 | 249.8 ± 60.4 | 0.1505 ± 0.0051 | 0.0000 ± 0.0000 | 0.4845 ± 0.0112 | 0.0000 ± 0.0000 |
<!-- auto:tier1:end -->

### Bảng 3: ghép cặp kèm sign test

<!-- auto:pairs:start -->
**whole_feature**. Δ = A − B theo seed; thắng = A nhỏ hơn; p = sign test chính xác hai phía (bỏ hòa), mô tả biến thiên mask trên bảng cố định, không phải CI theo nước.

| A vs B | metric | n seed | A thắng | hòa | Δ mean ± sd | sign p | ghi chú |
|---|---|---:|---:|---:|---|---:|---|
| same_day_mean vs hron | mae_prop | 8 | 8 | 0 | -0.06833 ± 0.01787 | 0.0078 | có điều kiện: 200: hron failed (baseline_error); 203: hron failed (baseline_error) |
| same_day_mean vs hron_cap | mae_prop | 8 | 8 | 0 | -0.06823 ± 0.01725 | 0.0078 | có điều kiện: 200: hron_cap failed (baseline_error); 203: hron_cap failed (baseline_error) |
| same_day_mean vs tsagris | mae_prop | 8 | 8 | 0 | -0.07377 ± 0.01025 | 0.0078 | có điều kiện: 200: tsagris failed (baseline_error); 203: tsagris failed (baseline_error) |
| same_day_mean vs micoda | mae_prop | 8 | 8 | 0 | -0.10745 ± 0.01389 | 0.0078 | có điều kiện: 200: micoda failed (insufficient_fit); 203: micoda failed (insufficient_fit) |
| same_day_mean vs micoda_other | mae_prop | 8 | 8 | 0 | -0.08702 ± 0.01088 | 0.0078 | có điều kiện: 200: micoda_other failed (insufficient_fit); 203: micoda_other failed (insufficient_fit) |
| hron vs tsagris | mae_prop | 8 | 6 | 0 | -0.00543 ± 0.01636 | 0.2891 | có điều kiện: 200: hron failed (baseline_error), tsagris failed (baseline_error); 203: hron failed (baseline_error), tsagris failed (baseline_error) |
| micoda vs mean_fill | mae_prop | 8 | 8 | 0 | -0.01544 ± 0.01236 | 0.0078 | có điều kiện: 200: micoda failed (insufficient_fit); 203: micoda failed (insufficient_fit) |
| micoda_other vs micoda | mae_prop | 8 | 8 | 0 | -0.02043 ± 0.00898 | 0.0078 | có điều kiện: 200: micoda_other failed (insufficient_fit), micoda failed (insufficient_fit); 203: micoda_other failed (insufficient_fit), micoda failed (insufficient_fit) |
| micoda_other vs mean_fill | mae_prop | 8 | 8 | 0 | -0.03587 ± 0.00667 | 0.0078 | có điều kiện: 200: micoda_other failed (insufficient_fit); 203: micoda_other failed (insufficient_fit) |
| hron vs mean_fill | mae_prop | 8 | 8 | 0 | -0.05456 ± 0.01684 | 0.0078 | có điều kiện: 200: hron failed (baseline_error); 203: hron failed (baseline_error) |
| tsagris vs hron | mse_prop | 8 | 8 | 0 | -0.02055 ± 0.01695 | 0.0078 | có điều kiện: 200: tsagris failed (baseline_error), hron failed (baseline_error); 203: tsagris failed (baseline_error), hron failed (baseline_error) |

**whole_feature_matched**. Δ = A − B theo seed; thắng = A nhỏ hơn; p = sign test chính xác hai phía (bỏ hòa), mô tả biến thiên mask trên bảng cố định, không phải CI theo nước.

| A vs B | metric | n seed | A thắng | hòa | Δ mean ± sd | sign p | ghi chú |
|---|---|---:|---:|---:|---|---:|---|
| same_day_mean vs hron | mae_prop | 10 | 10 | 0 | -0.07935 ± 0.02171 | 0.0020 | - |
| same_day_mean vs hron_cap | mae_prop | 10 | 10 | 0 | -0.07757 ± 0.02178 | 0.0020 | - |
| same_day_mean vs tsagris | mae_prop | 10 | 10 | 0 | -0.08973 ± 0.01794 | 0.0020 | - |
| same_day_mean vs micoda | mae_prop | 10 | 10 | 0 | -0.09628 ± 0.02026 | 0.0020 | - |
| same_day_mean vs micoda_other | mae_prop | 10 | 10 | 0 | -0.08672 ± 0.02039 | 0.0020 | - |
| hron vs tsagris | mae_prop | 10 | 9 | 0 | -0.01038 ± 0.01189 | 0.0215 | - |
| micoda vs mean_fill | mae_prop | 10 | 10 | 0 | -0.03081 ± 0.01373 | 0.0020 | - |
| micoda_other vs micoda | mae_prop | 10 | 10 | 0 | -0.00956 ± 0.00452 | 0.0020 | - |
| micoda_other vs mean_fill | mae_prop | 10 | 10 | 0 | -0.04037 ± 0.01006 | 0.0020 | - |
| hron vs mean_fill | mae_prop | 10 | 10 | 0 | -0.04775 ± 0.01321 | 0.0020 | - |
| tsagris vs hron | mse_prop | 10 | 10 | 0 | -0.02750 ± 0.01082 | 0.0020 | - |

**random_cell**. Δ = A − B theo seed; thắng = A nhỏ hơn; p = sign test chính xác hai phía (bỏ hòa), mô tả biến thiên mask trên bảng cố định, không phải CI theo nước.

| A vs B | metric | n seed | A thắng | hòa | Δ mean ± sd | sign p | ghi chú |
|---|---|---:|---:|---:|---|---:|---|
| same_day_mean vs hron | mae_prop | 10 | 10 | 0 | -0.05958 ± 0.00385 | 0.0020 | - |
| same_day_mean vs hron_cap | mae_prop | 10 | 10 | 0 | -0.05755 ± 0.00358 | 0.0020 | - |
| same_day_mean vs tsagris | mae_prop | 10 | 10 | 0 | -0.06882 ± 0.00218 | 0.0020 | - |
| same_day_mean vs micoda | mae_prop | 10 | 10 | 0 | -0.09951 ± 0.00363 | 0.0020 | - |
| same_day_mean vs micoda_other | mae_prop | 10 | 10 | 0 | -0.07642 ± 0.00392 | 0.0020 | - |
| hron vs tsagris | mae_prop | 10 | 10 | 0 | -0.00924 ± 0.00366 | 0.0020 | - |
| micoda vs mean_fill | mae_prop | 10 | 10 | 0 | -0.00791 ± 0.00272 | 0.0020 | - |
| micoda_other vs micoda | mae_prop | 10 | 10 | 0 | -0.02308 ± 0.00192 | 0.0020 | - |
| micoda_other vs mean_fill | mae_prop | 10 | 10 | 0 | -0.03100 ± 0.00282 | 0.0020 | - |
| hron vs mean_fill | mae_prop | 10 | 10 | 0 | -0.04784 ± 0.00410 | 0.0020 | - |
| tsagris vs hron | mse_prop | 10 | 10 | 0 | -0.01601 ± 0.00366 | 0.0020 | - |
<!-- auto:pairs:end -->

### Bảng 4: Tầng 1 theo châu lục và nhóm N

<!-- auto:strata:start -->
**MAE tỷ lệ (ô model) theo châu lục**: mean ± sd qua seed có ≥ 30 ô (n khi thiếu).

| Model | Mask | Africa | Asia | Europe | North America | Oceania | South America |
|---|---|---:|---:|---:|---:|---:|---:|
| hron | wf | 0.1796 ± 0.0461 (n = 8) | 0.1052 ± 0.0285 (n = 8) | 0.0720 ± 0.0138 (n = 8) | 0.0854 ± 0.0882 (n = 8) | 0.0336 ± 0.0340 (n = 6) | 0.1214 ± 0.0481 (n = 8) |
| hron | wfm | 0.1283 ± 0.0296 | 0.1007 ± 0.0369 | 0.0923 ± 0.0438 | 0.2113 ± 0.1179 | 0.0597 ± 0.0773 (n = 9) | 0.0866 ± 0.0763 |
| hron_cap | wf | 0.1741 ± 0.0459 (n = 8) | 0.1001 ± 0.0265 (n = 8) | 0.0758 ± 0.0141 (n = 8) | 0.0855 ± 0.0864 (n = 8) | 0.0331 ± 0.0225 (n = 6) | 0.1219 ± 0.0462 (n = 8) |
| hron_cap | wfm | 0.1246 ± 0.0302 | 0.0980 ± 0.0363 | 0.0953 ± 0.0432 | 0.2088 ± 0.1202 | 0.0541 ± 0.0772 (n = 9) | 0.0876 ± 0.0729 |
| tsagris | wf | 0.2212 ± 0.0504 (n = 8) | 0.1193 ± 0.0259 (n = 8) | 0.0672 ± 0.0108 (n = 8) | 0.0772 ± 0.0470 (n = 8) | 0.0473 ± 0.0456 (n = 6) | 0.1294 ± 0.0366 (n = 8) |
| tsagris | wfm | 0.1663 ± 0.0335 | 0.1203 ± 0.0223 | 0.0730 ± 0.0227 | 0.1485 ± 0.0697 | 0.0681 ± 0.0878 (n = 9) | 0.1026 ± 0.0509 |
| micoda | wf | 0.2038 ± 0.0425 (n = 8) | 0.1347 ± 0.0296 (n = 8) | 0.1182 ± 0.0189 (n = 8) | 0.1597 ± 0.0608 (n = 8) | 0.0613 ± 0.0209 (n = 6) | 0.1504 ± 0.0164 (n = 8) |
| micoda | wfm | 0.1478 ± 0.0353 | 0.1268 ± 0.0304 | 0.1185 ± 0.0331 | 0.1451 ± 0.0728 | 0.1000 ± 0.0865 (n = 9) | 0.1141 ± 0.0412 |
| micoda_other | wf | 0.2024 ± 0.0432 (n = 8) | 0.1224 ± 0.0315 (n = 8) | 0.1015 ± 0.0164 (n = 8) | 0.0794 ± 0.0579 (n = 8) | 0.0396 ± 0.0263 (n = 6) | 0.1250 ± 0.0295 (n = 8) |
| micoda_other | wfm | 0.1435 ± 0.0374 | 0.1204 ± 0.0331 | 0.1021 ± 0.0316 | 0.1313 ± 0.0842 | 0.0638 ± 0.0825 (n = 9) | 0.0905 ± 0.0457 |
| same_day_mean | wf | 0.0733 ± 0.0424 | 0.0313 ± 0.0178 | 0.0218 ± 0.0048 | 0.0349 ± 0.0198 | 0.0268 ± 0.0236 (n = 8) | 0.0300 ± 0.0106 |
| same_day_mean | wfm | 0.0569 ± 0.0089 | 0.0261 ± 0.0103 | 0.0248 ± 0.0125 | 0.0335 ± 0.0166 | 0.0407 ± 0.0417 (n = 9) | 0.0317 ± 0.0258 |
| mean_fill | wf | 0.2323 ± 0.0482 | 0.1612 ± 0.0505 | 0.1366 ± 0.0396 | 0.1102 ± 0.0740 | 0.0363 ± 0.0351 (n = 8) | 0.1321 ± 0.0588 |
| mean_fill | wfm | 0.1812 ± 0.0492 | 0.1563 ± 0.0415 | 0.1498 ± 0.0428 | 0.1984 ± 0.1185 | 0.0621 ± 0.0799 (n = 9) | 0.1100 ± 0.0688 |
| zero_fill | wf | 0.2577 ± 0.0829 | 0.1992 ± 0.0721 | 0.1913 ± 0.0670 | 0.1341 ± 0.1120 | 0.0415 ± 0.0572 (n = 8) | 0.1525 ± 0.0901 |
| zero_fill | wfm | 0.2103 ± 0.0697 | 0.1953 ± 0.0602 | 0.2150 ± 0.0816 | 0.2935 ± 0.1977 | 0.0572 ± 0.0764 (n = 9) | 0.1281 ± 0.1077 |
| marginal_sampler | wf | 0.2342 ± 0.0497 | 0.1642 ± 0.0507 | 0.1359 ± 0.0390 | 0.1047 ± 0.0677 | 0.0375 ± 0.0421 (n = 8) | 0.1328 ± 0.0576 |
| marginal_sampler | wfm | 0.1836 ± 0.0475 | 0.1580 ± 0.0450 | 0.1459 ± 0.0411 | 0.2062 ± 0.1258 | 0.0576 ± 0.0757 (n = 9) | 0.1118 ± 0.0683 |
| tsagris_cap | wf | 0.2212 ± 0.0504 (n = 8) | 0.1193 ± 0.0259 (n = 8) | 0.0672 ± 0.0108 (n = 8) | 0.0772 ± 0.0470 (n = 8) | 0.0473 ± 0.0456 (n = 6) | 0.1294 ± 0.0366 (n = 8) |
| tsagris_cap | wfm | 0.1663 ± 0.0335 | 0.1203 ± 0.0223 | 0.0730 ± 0.0227 | 0.1485 ± 0.0697 | 0.0681 ± 0.0878 (n = 9) | 0.1026 ± 0.0509 |
| micoda_other_cap | wf | 0.2024 ± 0.0432 (n = 8) | 0.1224 ± 0.0315 (n = 8) | 0.1015 ± 0.0164 (n = 8) | 0.0794 ± 0.0579 (n = 8) | 0.0396 ± 0.0263 (n = 6) | 0.1250 ± 0.0295 (n = 8) |
| micoda_other_cap | wfm | 0.1435 ± 0.0374 | 0.1204 ± 0.0331 | 0.1021 ± 0.0316 | 0.1313 ± 0.0842 | 0.0638 ± 0.0825 (n = 9) | 0.0905 ± 0.0457 |

same_day_mean − hron theo châu lục: Δ mean ± sd (thắng / số seed, sign p).

| Tầng | wf | wfm |
|---|---|---|
| Africa | -0.1063 ± 0.0324 (8/8, p 0.008) | -0.0714 ± 0.0254 (10/10, p 0.002) |
| Asia | -0.0810 ± 0.0264 (8/8, p 0.008) | -0.0746 ± 0.0328 (10/10, p 0.002) |
| Europe | -0.0498 ± 0.0131 (8/8, p 0.008) | -0.0675 ± 0.0457 (10/10, p 0.002) |
| North America | -0.0525 ± 0.0892 (6/8, p 0.289) | -0.1778 ± 0.1207 (10/10, p 0.002) |
| Oceania | +0.0004 ± 0.0355 (3/6, p 1.000) | -0.0189 ± 0.0447 (7/9, p 0.180) |
| South America | -0.0909 ± 0.0519 (8/8, p 0.008) | -0.0549 ± 0.0675 (6/10, p 0.754) |

**MAE tỷ lệ (ô model) theo nhóm N**: mean ± sd qua seed có ≥ 30 ô (n khi thiếu).

| Model | Mask | N1 | N2 | N3 |
|---|---|---:|---:|---:|
| hron | wf | 0.1877 ± 0.0344 (n = 8) | 0.1157 ± 0.0192 (n = 8) | 0.0415 ± 0.0082 (n = 8) |
| hron | wfm | 0.1661 ± 0.0292 | 0.1246 ± 0.0280 | 0.0595 ± 0.0201 |
| hron_cap | wf | 0.1870 ± 0.0340 (n = 8) | 0.1130 ± 0.0185 (n = 8) | 0.0432 ± 0.0111 (n = 8) |
| hron_cap | wfm | 0.1644 ± 0.0297 | 0.1211 ± 0.0283 | 0.0593 ± 0.0208 |
| tsagris | wf | 0.2022 ± 0.0160 (n = 8) | 0.1249 ± 0.0157 (n = 8) | 0.0413 ± 0.0148 (n = 8) |
| tsagris | wfm | 0.1900 ± 0.0201 | 0.1333 ± 0.0214 | 0.0589 ± 0.0144 |
| micoda | wf | 0.1930 ± 0.0142 (n = 8) | 0.1647 ± 0.0282 (n = 8) | 0.0934 ± 0.0226 (n = 8) |
| micoda | wfm | 0.1596 ± 0.0157 | 0.1384 ± 0.0312 | 0.1027 ± 0.0301 |
| micoda_other | wf | 0.1750 ± 0.0135 (n = 8) | 0.1356 ± 0.0238 (n = 8) | 0.0773 ± 0.0243 (n = 8) |
| micoda_other | wfm | 0.1500 ± 0.0180 | 0.1238 ± 0.0301 | 0.0981 ± 0.0329 |
| same_day_mean | wf | 0.0535 ± 0.0154 | 0.0354 ± 0.0086 | 0.0187 ± 0.0083 |
| same_day_mean | wfm | 0.0469 ± 0.0077 | 0.0374 ± 0.0068 | 0.0258 ± 0.0098 |
| mean_fill | wf | 0.2138 ± 0.0174 | 0.1607 ± 0.0180 | 0.1095 ± 0.0282 |
| mean_fill | wfm | 0.2043 ± 0.0285 | 0.1599 ± 0.0292 | 0.1296 ± 0.0388 |
| zero_fill | wf | 0.2457 ± 0.0453 | 0.1799 ± 0.0183 | 0.1645 ± 0.0499 |
| zero_fill | wfm | 0.2566 ± 0.0491 | 0.1957 ± 0.0364 | 0.1763 ± 0.0591 |
| marginal_sampler | wf | 0.2169 ± 0.0234 | 0.1619 ± 0.0158 | 0.1058 ± 0.0257 |
| marginal_sampler | wfm | 0.2038 ± 0.0324 | 0.1624 ± 0.0325 | 0.1295 ± 0.0407 |
| tsagris_cap | wf | 0.2022 ± 0.0160 (n = 8) | 0.1249 ± 0.0157 (n = 8) | 0.0413 ± 0.0148 (n = 8) |
| tsagris_cap | wfm | 0.1900 ± 0.0201 | 0.1333 ± 0.0214 | 0.0589 ± 0.0144 |
| micoda_other_cap | wf | 0.1750 ± 0.0135 (n = 8) | 0.1356 ± 0.0238 (n = 8) | 0.0773 ± 0.0243 (n = 8) |
| micoda_other_cap | wfm | 0.1500 ± 0.0180 | 0.1238 ± 0.0301 | 0.0981 ± 0.0329 |

same_day_mean − hron theo nhóm N: Δ mean ± sd (thắng / số seed, sign p).

| Tầng | wf | wfm |
|---|---|---|
| N1 | -0.1354 ± 0.0461 (8/8, p 0.008) | -0.1191 ± 0.0333 (10/10, p 0.002) |
| N2 | -0.0833 ± 0.0206 (8/8, p 0.008) | -0.0872 ± 0.0264 (10/10, p 0.002) |
| N3 | -0.0233 ± 0.0037 (8/8, p 0.008) | -0.0337 ± 0.0152 (10/10, p 0.002) |
<!-- auto:strata:end -->

### Bảng 5: Tầng 2 native

<!-- auto:tier2:start -->
Lượt native chạy lại qua `evaluate_v1.py` (`tier2.summarize`, CSV native của step 8 làm lượt chạy lại); so từng byte với output step 5:

| File | giống step 5 |
|---|:---:|
| tier2_targets.csv | True |
| tier2_strata.csv | True |
| tier2_support.csv | True |
| tier2_results.md | True |
| tier2_manifest.json | True |
<!-- auto:tier2:end -->

### Bảng 6: Tầng 3 theo mask

<!-- auto:tier3:start -->
| Diagnostic | Use case | wf | wfm | rc | cả ba |
|---|---|---|---|---|---|
| D1 | U1 thiếu | không đạt | không đạt | không đạt | **không đạt** |
| D1 | U1 thừa | đạt | đạt | đạt | **đạt** |
| D2 (chỉ mô tả) | U2 | không đạt | không đạt | đạt | **không đạt** |
| D3 | U3 | đạt | đạt | đạt | **đạt** |
| D1 c/N < 0.001 | U1 thiếu | không đạt | không đạt | không đạt | **không đạt** |
| D1 c/N < 0.001 | U1 thừa | đạt | đạt | đạt | **đạt** |
| D1 R-pooled | U1 thiếu | không đạt | không đạt | không đạt | **không đạt** |
| D1 R-pooled | U1 thừa | đạt | đạt | đạt | **đạt** |
| D2 R-pooled (chỉ mô tả) | U2 | không đạt | đạt | đạt | **không đạt** |
<!-- auto:tier3:end -->

### Bảng 8: so với exploratory

<!-- auto:exploratory:start -->
Exploratory: seed 0–4 (đã nhìn ở step 2–7). Xác nhận: seed 200–209. Δ = A − B, thắng = A nhỏ hơn.

| A vs B | metric | wf 0–4 | wf 200–209 | wfm 0–4 | wfm 200–209 | rc 0–4 | rc 200–209 |
|---|---|---|---|---|---|---|---|
| same_day_mean vs hron | mae_prop | -0.0525 (5/5) | -0.0683 (8/8) | -0.0681 (5/5) | -0.0794 (10/10) | -0.0582 (5/5) | -0.0596 (10/10) |
| same_day_mean vs hron_cap | mae_prop | -0.0518 (5/5) | -0.0682 (8/8) | -0.0669 (5/5) | -0.0776 (10/10) | -0.0570 (5/5) | -0.0575 (10/10) |
| same_day_mean vs tsagris | mae_prop | -0.0728 (5/5) | -0.0738 (8/8) | -0.0775 (5/5) | -0.0897 (10/10) | -0.0689 (5/5) | -0.0688 (10/10) |
| same_day_mean vs micoda | mae_prop | -0.1055 (5/5) | -0.1075 (8/8) | -0.0901 (5/5) | -0.0963 (10/10) | -0.1008 (5/5) | -0.0995 (10/10) |
| same_day_mean vs micoda_other | mae_prop | -0.0724 (5/5) | -0.0870 (8/8) | -0.0760 (5/5) | -0.0867 (10/10) | -0.0768 (5/5) | -0.0764 (10/10) |
| hron vs tsagris | mae_prop | -0.0204 (5/5) | -0.0054 (6/8) | -0.0093 (4/5) | -0.0104 (9/10) | -0.0107 (5/5) | -0.0092 (10/10) |
| tsagris vs hron | mse_prop | -0.0140 (5/5) | -0.0205 (8/8) | -0.0181 (5/5) | -0.0275 (10/10) | -0.0130 (5/5) | -0.0160 (10/10) |
| micoda vs mean_fill | mae_prop | +0.0052 (1/5) | -0.0154 (8/8) | -0.0184 (4/5) | -0.0308 (10/10) | -0.0054 (5/5) | -0.0079 (10/10) |
| micoda_other vs micoda | mae_prop | -0.0331 (5/5) | -0.0204 (8/8) | -0.0140 (4/5) | -0.0096 (10/10) | -0.0239 (5/5) | -0.0231 (10/10) |

MAE tỷ lệ trên seed chung của mọi model trong từng mask và dải seed (F07).

| Model | wf MAE 0–4 | wf MAE 200–209 | wfm MAE 0–4 | wfm MAE 200–209 | rc MAE 0–4 | rc MAE 200–209 |
|---|---:|---:|---:|---:|---:|---:|
| hron | 0.0875 ± 0.0105 (5) | 0.0982 ± 0.0141 (8) | 0.1072 ± 0.0230 (5) | 0.1156 ± 0.0249 (10) | 0.0892 ± 0.0028 (5) | 0.0893 ± 0.0042 (10) |
| hron_cap | 0.0869 ± 0.0095 (5) | 0.0981 ± 0.0138 (8) | 0.1060 ± 0.0228 (5) | 0.1139 ± 0.0251 (10) | 0.0879 ± 0.0025 (5) | 0.0873 ± 0.0038 (10) |
| tsagris | 0.1079 ± 0.0209 (5) | 0.1036 ± 0.0134 (8) | 0.1165 ± 0.0160 (5) | 0.1260 ± 0.0201 (10) | 0.0999 ± 0.0024 (5) | 0.0986 ± 0.0029 (10) |
| micoda | 0.1406 ± 0.0099 (5) | 0.1373 ± 0.0155 (8) | 0.1291 ± 0.0163 (5) | 0.1326 ± 0.0242 (10) | 0.1317 ± 0.0018 (5) | 0.1293 ± 0.0044 (10) |
| micoda_other | 0.1075 ± 0.0093 (5) | 0.1169 ± 0.0139 (8) | 0.1151 ± 0.0137 (5) | 0.1230 ± 0.0244 (10) | 0.1078 ± 0.0008 (5) | 0.1062 ± 0.0048 (10) |
| same_day_mean | 0.0351 ± 0.0066 (5) | 0.0298 ± 0.0070 (8) | 0.0391 ± 0.0088 (5) | 0.0363 ± 0.0052 (10) | 0.0310 ± 0.0027 (5) | 0.0298 ± 0.0017 (10) |
| mean_fill | 0.1354 ± 0.0145 (5) | 0.1527 ± 0.0141 (8) | 0.1475 ± 0.0223 (5) | 0.1634 ± 0.0272 (10) | 0.1371 ± 0.0016 (5) | 0.1372 ± 0.0025 (10) |
| zero_fill | 0.1693 ± 0.0219 (5) | 0.1989 ± 0.0193 (8) | 0.1828 ± 0.0411 (5) | 0.2079 ± 0.0376 (10) | 0.1716 ± 0.0035 (5) | 0.1740 ± 0.0036 (10) |
| marginal_sampler | 0.1354 ± 0.0120 (5) | 0.1524 ± 0.0135 (8) | 0.1482 ± 0.0231 (5) | 0.1642 ± 0.0305 (10) | 0.1362 ± 0.0072 (5) | 0.1378 ± 0.0059 (10) |
| tsagris_cap | 0.1079 ± 0.0209 (5) | 0.1036 ± 0.0134 (8) | 0.1165 ± 0.0160 (5) | 0.1260 ± 0.0201 (10) | 0.0999 ± 0.0024 (5) | 0.0986 ± 0.0029 (10) |
| micoda_other_cap | 0.1075 ± 0.0093 (5) | 0.1169 ± 0.0139 (8) | 0.1151 ± 0.0137 (5) | 0.1230 ± 0.0244 (10) | 0.1078 ± 0.0008 (5) | 0.1062 ± 0.0048 (10) |

Tầng 3: exploratory gộp 10 ca (step 6: wf + rc; step 7: wfm + rc), v1 theo từng mask 10 ca.

| Diagnostic | Use case | step 6 wf+rc | step 7 wfm+rc | v1 wf | v1 wfm | v1 rc |
|---|---|---|---|---|---|---|
| D1 | U1 thiếu | không đạt | không đạt | không đạt | không đạt | không đạt |
| D1 | U1 thừa | đạt | đạt | đạt | đạt | đạt |
| D2 (chỉ mô tả) | U2 | đạt | không đạt | không đạt | không đạt | đạt |
| D3 | U3 | đạt | đạt | đạt | đạt | đạt |
| D1 c/N < 0.001 | U1 thiếu | không đạt | không đạt | không đạt | không đạt | không đạt |
| D1 c/N < 0.001 | U1 thừa | đạt | đạt | đạt | đạt | đạt |
| D1 R-pooled | U1 thiếu | không đạt | không đạt | không đạt | không đạt | không đạt |
| D1 R-pooled | U1 thừa | đạt | đạt | đạt | đạt | đạt |
| D2 R-pooled (chỉ mô tả) | U2 | không đạt | đạt | không đạt | đạt | đạt |
<!-- auto:exploratory:end -->

## 5. Kết quả xác nhận

Mọi kết luận dưới đây chỉ nói về **biến thiên theo mask** (10 seed chưa dùng) trên **một bảng COVID cố định**. Chúng không phải độ chính xác trên ô thiếu thật, không phải CI theo nước, và không tổng quát hóa sang nước mới. Sign test là thống kê mô tả.

**Tầng 0:** ở `whole_feature`, seed 200 và 203 xóa mọi dòng đầy đủ (0/516 dòng còn lại, giống seed 109 ở step 2). hron, tsagris, MICoDa (A17 và A18) và các biến thể cap fail ở đúng 2/10 seed (`baseline_error`, `insufficient_fit`). Theo chính sách đã khai báo (> 2 mới là không đủ support), giả thuyết vẫn được chấm, có điều kiện trên 8 seed chung. Hai mask sensitivity không có run nào fail.

| ID | Kết quả (wf) | Diễn giải có giới hạn |
|---|---|---|
| H1 | **xác nhận** | same_day_mean có MAE thấp hơn hron ở 8/8 seed chung, Δ −0.068. Cùng kết quả ở wfm (10/10) và rc (10/10). |
| H2 | **xác nhận** | same_day_mean thắng cả 5 model thật còn lại ở 8/8 seed (Δ từ −0.068 đến −0.108); wfm và rc cũng vậy. |
| H3 | **không xác nhận** | hron chỉ thắng tsagris 6/8 seed, Δ −0.005, nhỏ hơn nhiều so với exploratory (−0.020, 5/5). Ở wfm (9/10) và rc (10/10) thì đạt. Thứ tự hron < tsagris về MAE không vững trên mask chính. |
| H4 | **xác nhận** | tsagris có MSE thấp hơn hron ở 8/8 seed (Δ −0.021). Cùng với H3: hai metric xếp hai model ngược nhau. |
| H5 | **không xác nhận** | micoda (A17) thắng mean_fill ở 8/8 seed (Δ −0.015). Exploratory wf cho điều ngược lại (thắng 1/5, Δ +0.005); giả thuyết được viết theo kết quả đó, và lượt mới không giữ nó. |
| H6 | **xác nhận** | micoda_other thắng micoda ở 8/8 seed (Δ −0.020); wfm và rc 10/10. |
| H7 | **xác nhận** | D3 đạt U3 ở cả ba mask (chênh ≥ 0.2 ở 10/10 ca mỗi mask). |
| H8 | **xác nhận** | D1 đạt U1 chiều điền thừa ở cả ba mask, nhưng vẫn sát ngưỡng (min đúng 0.80 ở cả ba). |
| H9 | **xác nhận** | D1 không đạt U1 chiều điền thiếu ở cả ba mask (TB wf 0.65, wfm 0.75, rc 0.67). |

**Thêm, ngoài giả thuyết:**
- **D2** (chỉ mô tả trong v1) không đạt U2 ở wf và wfm, đạt ở rc. Regret ≤ 0.005 ở 10/10 ca mỗi mask, vì same_day_mean luôn được chọn. Điều này khớp với step 7: tỷ lệ khớp của D2 không bền theo mask.
- **same_day_mean thắng hron theo tầng** ở wf, 8/8 seed với: Châu Phi Δ −0.106, Châu Á −0.081, Châu Âu, Nam Mỹ, N1 −0.135, N2, N3.
- **Hai chỗ same_day_mean không thắng rõ:**
  - Bắc Mỹ: wf thắng 6/8, wfm 10/10;
  - Châu Đại Dương: wf 3/6, Δ +0.000; wfm 7/9.
- **Ở wf, Tầng 3 chỉ có tỷ lệ ở 8 ca.** Ở seed 200 và 203 chỉ còn same_day_mean trong số model thật, nên không có cặp nào để so.

## 6. Sai lệch so với kế hoạch

1. **Bug Tầng 3 sau bước 3** (`9954fd2`).
   - **Bug:** `tier3.one_case` luôn đưa cả 6 model thật vào `common_targets`. Khi một model thật fail trong một ca (wf seed 200, 203), lần chạy `evaluate_v1.py` đầu tiên dừng vì `KeyError: 'hron'`, và không sinh bảng nào. Traceback nằm ở `attempt1_evaluate.txt`.
   - **Sửa:** chỉ lấy target chung trên các model có mặt. Cặp có model fail thành NA, đúng như đặc tả step 6 đã khai báo. Có test hồi quy. Output step 6 và step 7 sinh lại giống từng byte.
   - **Chạy lại theo C.5:** lượt chạy model thứ nhất được giữ ở `tv2_step8/runs_attempt1/` (không commit). Toàn bộ bước 3–4 được chạy lại. Output model của hai lượt giống từng byte (mục 3).
   - **Ảnh hưởng:** lượt thứ nhất không có kết quả nào, nên không có số "trước sửa" để so; kết quả trong báo cáo là của lượt thứ hai.
2. **Hai sửa phần trình bày bảng 8, sau khi thấy số** (`4b8c065`). Không đổi giả thuyết, verdict hay cặp nào.
   - Mỗi cặp giả thuyết giờ chỉ in một lần (trước đó same_day_mean vs hron lặp vì nằm trong cả H1 và H2).
   - Trung bình MAE theo model giờ lấy trên seed chung của mọi model (F07). Bản trước lấy trên các seed riêng của từng model, khiến ở wf có model tính trên 10 seed, có model trên 8.
3. **`tv2_step6/tier3.py` được sửa hai lần** (`88caf34` nhận seed qua tham số, `9954fd2` sửa bug). Output step 6 và 7 không đổi, nhưng file không còn đúng như lúc các step đó commit số.

## 7. Giới hạn còn lại của evaluation v1

- **Coverage R-local:** ô thiếu thật chỉ có 0.56, ô che có 0.90. Kết luận Tầng 3 đúng cho vùng có support.
- **Omicron và Delta:** chiếm phần lớn ô che nhưng không có ô thiếu thật nào mà model phải quyết định. Còn 20C và S:677 thì ngược lại. Không mask nào khớp theo target.
- **Châu Đại Dương:** ít ô, sd lớn; same_day_mean không thắng rõ ở đó.
- **D2 không ổn định** giữa các mask, nên v1 chỉ dùng D2 để mô tả.
- **D1:** chỉ cảnh báo điền thừa, và luôn sát ngưỡng 0.80. D1 không cảnh báo được điền thiếu.
- **Seed xóa hết dòng đầy đủ** (wf seed 200, 203, 109) làm Hron, Tsagris và MICoDa fail. Ghép cặp vì thế có điều kiện.
- **Một bảng, một quốc gia-thời gian cố định:** không có biến thiên theo nước, không có CI.
- **Tầng 4 chưa làm.**

## 8. Đưa một model mới vào v1

1. **Thêm model** vào `run.py` (`MODELS`) và một bảng `[tên]` trong file models toml. Nếu cần chạy riêng, thêm một pass vào `[runs.*]`. Sau đó cập nhật `[models]` của một bản config mới, ví dụ `eval_v1_<model>.toml`.
2. **Không bao giờ tune trên seed 200–209**, kể cả nhìn kết quả từng seed để chọn tham số. Mọi tuning dùng seed 0–4 hoặc tập kiểm định nội bộ của model.
3. **Chốt cấu hình trước khi chạy:** commit file models toml và config mới trước khi chạy `evaluate_v1.py run`. Đây là đường đóng băng của v1, giống Phần C.
4. **Chạy:**
   - `PYTHONPATH=src python tv2_step8/evaluate_v1.py check`;
   - `run` cho các lượt của model mới (thư mục mới, không ghi đè);
   - `evaluate`.
   
   `masks_manifest.json` đã có, nên mask không được sinh lại.
5. **So với baseline** bằng cùng cặp và cùng sign test, trên seed chung. Một kết quả dùng seed 200–209 mà đã được nhìn trong lúc phát triển model thì phải ghi nhãn exploratory.
