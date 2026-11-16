# Step 6: Tầng 3, kiểm định diagnostic Tầng 2 trên ô che giả (exploratory)

Branch `step6-tier3-calibration` (từ `step5-tier2-native`). Không có `iterative_net` hay `framework/`. Không sửa baseline, không chạy seed mới. Định nghĩa D1/D2/D3 giữ nguyên như step 5. Mọi bảng số trong khối `auto:*` do `tier3.py` sinh, không sửa tay; bảng đầy đủ ở [tier3_results.md](tier3_results.md).

Mọi kết luận dưới đây có dạng: **"trong 10 ca (whole_feature, random_cell × seed 0–4) và support đã kiểm, diagnostic X làm được / chưa làm được việc Y"**. Không kết luận nào là độ chính xác trên ô thiếu thật (F01).

## 1. File đã sửa/thêm

| File | Nội dung |
|---|---|
| `src/longitudinal_impute/tier3_calibration.py` (207 dòng) | Use case U1–U4, ngưỡng đạt, sensitivity, so cặp trong từng ca, regret, precision của cờ D3, kiểm đối thủ giả, D1 với ngưỡng theo tỷ lệ. Commit trước khi tính trên output nào. |
| `tests/test_tier3_calibration.py` | 13 test mới. |
| `README.md` | Một đoạn về Tầng 3. |
| `tv2_step6/tier3.py` (430 dòng), `test_tier3.py` (3 test) | Kiểm hash, đọc long file, dựng bảng. Đọc run và regression check qua `tier01.load_run` và `validate_pair`; diagnostic qua `tier2_diagnostics`; sai số Tầng 1 qua `evaluate.cell_errors`. Không có bộ chấm điểm thứ hai. |
| `tv2_step6/tier3_cases.json`, `tier3_results.md`, `tier3_manifest.json` | Output sinh tự động. |

## 2. Đặc tả (đóng băng trước khi tính)

**Dữ liệu:**
- 10 ca: whole_feature và random_cell, rate 0.05, mask seed 0–4. Không dùng seed 109.
- Dùng lại long file của step 2 (`main_*`, `micoda_other_*`) và step 4 (`residual_*`).
- **Model thật:** hron, hron_cap, tsagris, micoda (A17), micoda_other, same_day_mean.
- **Đối thủ giả:** mean_fill, zero_fill, marginal_sampler.
- **Bỏ:** tsagris_cap (trùng tsagris: `cap` ≡ `prop_other`, step 4–5) và micoda_other_cap (gần trùng micoda_other), để không nhân đôi cặp.

**Calibration:**
- E = ô che thuộc nhóm `model_cell`; V = O \ H (`calibration_cells`).
- D1, D2 (R-local chính, R-pooled phụ) và D3 theo đúng định nghĩa step 5; T_j của D3 chỉ lấy từ V.
- **Hàng truth:** D1/D2 của chính đáp án trên E, như thể truth là một model.

**Quy một ca về một số:**
- Lấy giá trị theo target (hoặc khoảng cách tới giá trị của truth), trung bình có trọng số theo số ô mà giá trị đó dùng (R-local: ô có support; R-pooled: ô E).
- Chỉ dùng **tập target chung**: những target mà truth và cả 6 model thật đều có giá trị.
- Sai số Tầng 1 của một diagnostic là `evaluate.cell_errors` trên **đúng các ô diagnostic đó chấm**.

**Use case và cách kiểm:**

| Use case | Diagnostic | So trong từng ca, mọi cặp model thật (15 cặp) |
|---|---|---|
| U1 thiếu | D1 − D1_truth | dấu Δ(D1 − D1_truth) so với dấu Δ MAE tỷ lệ trên ô đáp án > 0 |
| U1 thừa | D1 − D1_truth | dấu −Δ(D1 − D1_truth) so với dấu Δ MAE tỷ lệ trên ô đáp án = 0 |
| U2 | \|D2 − D2_truth\| | dấu Δ so với dấu Δ MAE tỷ lệ; thêm regret = MAE của model được chọn (giá trị nhỏ nhất) − MAE nhỏ nhất của ca |
| U3 | D3 | trên ô E được 6 model thật điền ≥ 0.5 count: tỷ lệ đáp án 0 khi bị cờ, trừ tỷ lệ đó khi không bị cờ |
| U4 | D1, D2, D3 | đối thủ giả có bị xếp kém hơn **mọi** model thật không (báo kết quả, không có ngưỡng) |

**Quy tắc so cặp:**
- |Δ MAE| < 0.001 là **hòa**: báo riêng, không vào tỷ lệ.
- Δ diagnostic = 0 tính là không khớp.
- Cặp có giá trị thiếu là NA: báo riêng, không vào tỷ lệ.

**Ngưỡng đạt** (thực dụng, không phải chuẩn phổ quát):
- **U1** (từng chiều) **và U2:** tỷ lệ khớp trung bình ≥ 0.80 **trong từng mask**, và không ca nào < 0.5. U2 thêm điều kiện regret ≤ 0.005 ở ≥ 9/10 ca.
- **U3:** chênh ≥ 0.2 ở ≥ 9/10 ca.
- **Không đủ support:** một mask có dưới 4/5 ca có tỷ lệ (U1, U2), hoặc dưới 9 ca có chênh (U3).

**Sensitivity**, báo cạnh kết quả chính, không thay kết quả chính:
- D1 với zero là c/N < 0.001 (U1);
- R-pooled thay R-local (U1, U2).

**Commit:**
- `bf4f096` đóng băng đặc tả, trước khi tính trên output nào.
- `26ec4a8` thêm script. Tôi chạy thử script một lần vào thư mục tạm trước commit này.
- `11bfcc9` sửa phần trình bày, không đổi định nghĩa nào:
  - tiêu đề bảng 7 chứa `|` làm vỡ markdown;
  - ghi số ca khi có ca NA;
  - thêm dòng **tham chiếu MAE Tầng 1** vào bảng 7. Dòng này được thêm **sau khi thấy số**, để đọc kết quả đối thủ giả. Nó không phải diagnostic và không vào phần đạt/không đạt.
- `b6f8799` commit số.

## 3. Kiểm tra

- **Test:** 322 pass với `-W error::RuntimeWarning`:
  - nextgen 295 (282 cũ, 13 mới);
  - script 27 (24 cũ, 3 mới).
- **Test mới kiểm:**
  - đổi đáp án của ô trong H thì mọi output diagnostic không đổi từng byte (R-local, R-pooled, T_j); không ô H nào vào reference;
  - hàng truth chỉ đọc đáp án trên E, và một model dự đoán đúng đáp án cho đúng kết quả của hàng truth;
  - so cặp, hòa theo ngưỡng 0.001, cặp NA, Δ diagnostic = 0, regret trên toy tính tay;
  - toy kiểu E-X2: tương quan pooled > 0.9 nhưng tỷ lệ khớp trong từng ca = 0, nên không đạt;
  - precision U3 tính tay, NA khi không có cờ; T_j chỉ từ V;
  - ngưỡng c/N < 0.001 đúng biên (0.001 là dương);
  - quy tắc đạt theo từng mask, regret, U3;
  - chạy hai lần cùng input cho kết quả giống từng byte.
- **Tái lập:** `tier3.py` chạy hai lần (hai thư mục output) cho 3 file output giống từng byte.

<!-- auto:check:start -->
whole_feature: 2220 giá trị long file khớp summary.json (tier01.load_run); mask seed 0–4 giống nhau ở 3 thư mục run và bằng `make_mask`; model loại bỏ: tsagris_cap (trùng tsagris (`cap` ≡ `prop_other`, step 4–5)), micoda_other_cap (gần trùng micoda_other (step 4–5)).
random_cell: 2100 giá trị long file khớp summary.json (tier01.load_run); mask seed 0–4 giống nhau ở 3 thư mục run và bằng `make_mask`; model loại bỏ: tsagris_cap (trùng tsagris (`cap` ≡ `prop_other`, step 4–5)), micoda_other_cap (gần trùng micoda_other (step 4–5)).
<!-- auto:check:end -->

F2 (`validate_pair`) đạt ở cả hai mask. Bảng model trong config của 3 thư mục run khớp file toml hiện tại; data, schema, group map khớp, và các run được chạy từ git tree sạch.

## 4. Bảng

### Bảng 10: tóm tắt

<!-- auto:summary:start -->
| Diagnostic | Use case | Kết quả (10 ca, 2 mask) |
|---|---|---|
| D1 | U1 thiếu (cảnh báo điền thiếu) | **không đạt** |
| D1 | U1 thừa (cảnh báo điền thừa) | **đạt** |
| D2 | U2 proxy xếp hạng MAE | **đạt** |
| D3 | U3 cờ ngoài support | **đạt** |
| D1, ngưỡng c/N < 0.001 (sensitivity) | U1 thiếu | **không đạt** |
| D1, ngưỡng c/N < 0.001 (sensitivity) | U1 thừa | **đạt** |
| D1 R-pooled (sensitivity) | U1 thiếu | **không đạt** |
| D1 R-pooled (sensitivity) | U1 thừa | **đạt** |
| D2 R-pooled (sensitivity) | U2 | **không đạt** |
| D1, D2, D3 | U4 đối thủ giả | báo kết quả, không có ngưỡng (bảng 7) |
<!-- auto:summary:end -->

### Bảng 1: support của E

<!-- auto:support:start -->
| Mask | Seed | ô H | ô E (model) | ô E có support R-local | coverage | target chung D1 / D2 (R-local) | ô chấm D1 / D2 |
|---|---:|---:|---:|---:|---:|---|---|
| whole_feature | 0 | 5625 | 2595 | 2552 | 0.983 | 16 / 13 | 2552 / 2400 |
| whole_feature | 1 | 5705 | 2729 | 2399 | 0.879 | 15 / 11 | 2399 / 2158 |
| whole_feature | 2 | 5637 | 2522 | 2247 | 0.891 | 15 / 14 | 2247 / 2198 |
| whole_feature | 3 | 5696 | 2504 | 2218 | 0.886 | 14 / 11 | 2218 / 2066 |
| whole_feature | 4 | 5646 | 2406 | 2142 | 0.890 | 17 / 15 | 2142 / 2062 |
| random_cell | 0 | 5579 | 2851 | 2629 | 0.922 | 17 / 17 | 2629 / 2629 |
| random_cell | 1 | 5534 | 2740 | 2520 | 0.920 | 17 / 17 | 2520 / 2520 |
| random_cell | 2 | 5604 | 2762 | 2532 | 0.917 | 17 / 17 | 2532 / 2532 |
| random_cell | 3 | 5673 | 2832 | 2603 | 0.919 | 17 / 17 | 2603 / 2603 |
| random_cell | 4 | 5546 | 2750 | 2519 | 0.916 | 17 / 16 | 2519 / 2497 |
<!-- auto:support:end -->

### Bảng 2: D1, D2, D3 calibration, gồm hàng truth

<!-- auto:diag:start -->
**whole_feature** (mean ± sd qua 5 seed; D1, D2 = TB trọng số qua target; khoảng cách = trên target chung)

| Model | D1 local | D2 local | D1 pooled | D2 pooled | D3 ô | D1 − D1_truth (local) | \|D2 − D2_truth\| (local) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **truth** | -0.050 ± 0.009 | 0.0583 ± 0.0183 | -0.142 ± 0.023 | 0.0574 ± 0.0184 | 1 ± 1 | 0 | 0 |
| hron | -0.015 ± 0.025 | 0.1002 ± 0.0142 | -0.120 ± 0.033 | 0.0833 ± 0.0146 | 48 ± 13 | +0.035 ± 0.018 | 0.0439 ± 0.0290 |
| hron_cap | +0.031 ± 0.029 | 0.1055 ± 0.0185 | -0.067 ± 0.025 | 0.0837 ± 0.0167 | 38 ± 14 | +0.082 ± 0.020 | 0.0493 ± 0.0348 |
| tsagris | -0.180 ± 0.044 | 0.1363 ± 0.0401 | -0.284 ± 0.054 | 0.1192 ± 0.0308 | 193 ± 46 | -0.130 ± 0.038 | 0.0802 ± 0.0583 |
| micoda | -0.382 ± 0.035 | 0.1836 ± 0.0216 | -0.506 ± 0.029 | 0.1645 ± 0.0115 | 453 ± 84 | -0.332 ± 0.034 | 0.1309 ± 0.0399 |
| micoda_other | -0.174 ± 0.039 | 0.2089 ± 0.0272 | -0.287 ± 0.035 | 0.1794 ± 0.0180 | 327 ± 64 | -0.124 ± 0.034 | 0.1594 ± 0.0445 |
| same_day_mean | -0.076 ± 0.024 | 0.0766 ± 0.0086 | -0.185 ± 0.037 | 0.0684 ± 0.0075 | 12 ± 9 | -0.026 ± 0.021 | 0.0230 ± 0.0164 |
| mean_fill | -0.345 ± 0.048 | 0.2472 ± 0.0293 | -0.452 ± 0.048 | 0.2173 ± 0.0196 | 337 ± 66 | -0.295 ± 0.043 | 0.1997 ± 0.0486 |
| zero_fill | +0.320 ± 0.024 | NA | +0.204 ± 0.018 | NA | 0 ± 0 | +0.371 ± 0.021 | NA |
| marginal_sampler | +0.139 ± 0.020 | 0.1140 ± 0.0170 | +0.033 ± 0.006 | 0.0928 ± 0.0085 | 106 ± 18 | +0.189 ± 0.019 | 0.0823 ± 0.0160 (n = 2) |

**random_cell** (mean ± sd qua 5 seed; D1, D2 = TB trọng số qua target; khoảng cách = trên target chung)

| Model | D1 local | D2 local | D1 pooled | D2 pooled | D3 ô | D1 − D1_truth (local) | \|D2 − D2_truth\| (local) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **truth** | -0.051 ± 0.004 | 0.0312 ± 0.0028 | -0.137 ± 0.006 | 0.0297 ± 0.0050 | 1 ± 1 | 0 | 0 |
| hron | -0.055 ± 0.010 | 0.0990 ± 0.0069 | -0.146 ± 0.009 | 0.0840 ± 0.0054 | 74 ± 4 | -0.005 ± 0.011 | 0.0681 ± 0.0058 |
| hron_cap | -0.006 ± 0.004 | 0.1093 ± 0.0060 | -0.094 ± 0.004 | 0.0930 ± 0.0072 | 59 ± 3 | +0.044 ± 0.005 | 0.0784 ± 0.0071 |
| tsagris | -0.201 ± 0.010 | 0.1154 ± 0.0097 | -0.294 ± 0.012 | 0.1024 ± 0.0088 | 236 ± 12 | -0.151 ± 0.012 | 0.0846 ± 0.0097 |
| micoda | -0.384 ± 0.009 | 0.1615 ± 0.0016 | -0.489 ± 0.006 | 0.1403 ± 0.0034 | 501 ± 26 | -0.334 ± 0.012 | 0.1305 ± 0.0023 |
| micoda_other | -0.199 ± 0.012 | 0.1974 ± 0.0014 | -0.296 ± 0.010 | 0.1747 ± 0.0030 | 368 ± 23 | -0.148 ± 0.014 | 0.1666 ± 0.0023 |
| same_day_mean | -0.093 ± 0.007 | 0.0703 ± 0.0028 | -0.177 ± 0.011 | 0.0577 ± 0.0032 | 4 ± 1 | -0.043 ± 0.006 | 0.0394 ± 0.0025 |
| mean_fill | -0.388 ± 0.006 | 0.2411 ± 0.0024 | -0.478 ± 0.007 | 0.2185 ± 0.0031 | 418 ± 17 | -0.338 ± 0.008 | 0.2104 ± 0.0029 |
| zero_fill | +0.319 ± 0.003 | NA | +0.218 ± 0.003 | NA | 0 ± 0 | +0.370 ± 0.006 | NA |
| marginal_sampler | +0.124 ± 0.008 | 0.1126 ± 0.0121 | +0.030 ± 0.007 | 0.0971 ± 0.0067 | 136 ± 16 | +0.174 ± 0.008 | 0.0719 ± 0.0015 (n = 2) |
<!-- auto:diag:end -->

### Bảng 3: độ lệch của reference (D1_truth, D2_truth theo target)

<!-- auto:truth:start -->
Giá trị của chính đáp án trên E so với reference (mean qua 5 seed). D1_truth ≠ 0 và D2_truth > 0 là độ lệch sẵn có của reference khỏi phân phối thật của ô che.

| Target | wf D1t local | wf D2t local | wf D1t pooled | wf D2t pooled | rc D1t local | rc D2t local | rc D1t pooled | rc D2t pooled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| recombinant | -0.212 | 0.0361 | -0.153 | 0.0272 | -0.205 | 0.0083 | -0.193 | 0.0092 |
| 20A | +0.010 | 0.0970 | -0.166 | 0.0996 | -0.024 | 0.0383 | -0.158 | 0.0505 |
| 20B | -0.037 | 0.1221 | -0.068 | 0.0912 | -0.021 | 0.0301 | -0.115 | 0.0272 |
| 20C | -0.004 | 0.0205 | -0.030 | 0.0237 | -0.005 | 0.0183 | -0.078 | 0.0168 |
| 20E | +0.028 | 0.0822 | -0.195 | 0.0821 | +0.001 | 0.0348 | -0.134 | 0.0357 |
| Beta | -0.155 | 0.1351 | -0.330 | 0.2055 | -0.049 | 0.0406 | -0.164 | 0.0620 |
| Alpha | -0.035 | 0.0690 | -0.236 | 0.0779 | -0.056 | 0.0302 | -0.222 | 0.0444 |
| Gamma | -0.085 | 0.0919 | -0.195 | 0.1401 | -0.053 | 0.0375 | -0.156 | 0.0362 |
| Delta | -0.104 | 0.1005 | -0.285 | 0.0546 | -0.071 | 0.0575 | -0.248 | 0.0247 |
| Kappa | +0.026 | 0.0403 | -0.018 | 0.0428 | -0.016 | 0.0173 | -0.053 | 0.0172 |
| Epsilon | +0.016 | 0.0341 | -0.048 | 0.0369 | -0.038 | 0.0410 | -0.089 | 0.0338 |
| Eta | -0.032 | 0.0719 | -0.109 | 0.0812 | -0.020 | 0.0558 | -0.106 | 0.0498 |
| Iota | +0.009 | 0.0095 | -0.065 | 0.0451 | -0.016 | 0.0170 | -0.057 | 0.0274 |
| Lambda | -0.030 | 0.0526 | -0.113 | 0.0699 | -0.032 | 0.0534 | -0.082 | 0.0411 |
| Mu | -0.009 | 0.1697 | -0.097 | 0.1363 | -0.034 | 0.0401 | -0.077 | 0.0352 |
| Omicron | -0.011 | 0.0115 | -0.087 | 0.0074 | -0.010 | 0.0120 | -0.069 | 0.0079 |
| S:677 | -0.005 | 0.0361 | -0.029 | 0.0560 | -0.061 | 0.0828 | -0.067 | 0.0789 |
<!-- auto:truth:end -->

### Bảng 4: U1

<!-- auto:u1:start -->
| Mask | Seed | thiếu: khớp (khớp/quyết định) | thừa: khớp |
|---|---:|---|---|
| whole_feature | 0 | 0.43 (6/14, hòa 1) | 0.80 (12/15) |
| whole_feature | 1 | 0.64 (9/14, hòa 1) | 0.80 (12/15) |
| whole_feature | 2 | 0.73 (11/15) | 0.87 (13/15) |
| whole_feature | 3 | 0.60 (9/15) | 0.87 (13/15) |
| whole_feature | 4 | 0.67 (10/15) | 0.87 (13/15) |
| random_cell | 0 | 0.67 (10/15) | 0.80 (12/15) |
| random_cell | 1 | 0.64 (9/14, hòa 1) | 0.80 (12/15) |
| random_cell | 2 | 0.67 (10/15) | 0.80 (12/15) |
| random_cell | 3 | 0.71 (10/14, hòa 1) | 0.87 (13/15) |
| random_cell | 4 | 0.67 (10/15) | 0.87 (13/15) |
- U1 thiếu (D1 cao ↔ MAE ô đáp án > 0): whole_feature: TB 0.61, min 0.43 (5 ca); random_cell: TB 0.67, min 0.64 (5 ca) → **không đạt**.
- U1 thừa (D1 thấp ↔ MAE ô đáp án = 0): whole_feature: TB 0.84, min 0.80 (5 ca); random_cell: TB 0.83, min 0.80 (5 ca) → **đạt**.
<!-- auto:u1:end -->

### Bảng 5: U2

<!-- auto:u2:start -->
| Mask | Seed | khớp \|D2 − D2t\| | chọn → regret | khớp D2 thô | regret thô | khớp, MAE toàn ô model | regret, MAE toàn ô model |
|---|---:|---|---|---|---:|---|---:|
| whole_feature | 0 | 0.87 (13/15) | same_day_mean → 0.0000 | 0.87 (13/15) | 0.0000 | 0.87 (13/15) | 0.0000 |
| whole_feature | 1 | 0.80 (12/15) | same_day_mean → 0.0000 | 0.80 (12/15) | 0.0000 | 0.80 (12/15) | 0.0000 |
| whole_feature | 2 | 0.93 (13/14, hòa 1) | same_day_mean → 0.0000 | 0.93 (13/14, hòa 1) | 0.0000 | 0.93 (13/14, hòa 1) | 0.0000 |
| whole_feature | 3 | 0.93 (14/15) | same_day_mean → 0.0000 | 0.93 (14/15) | 0.0000 | 0.93 (14/15) | 0.0000 |
| whole_feature | 4 | 0.79 (11/14, hòa 1) | hron_cap → 0.0457 | 0.93 (13/14, hòa 1) | 0.0000 | 0.79 (11/14, hòa 1) | 0.0393 |
| random_cell | 0 | 0.93 (13/14, hòa 1) | same_day_mean → 0.0000 | 0.93 (13/14, hòa 1) | 0.0000 | 0.93 (13/14, hòa 1) | 0.0000 |
| random_cell | 1 | 0.87 (13/15) | same_day_mean → 0.0000 | 0.87 (13/15) | 0.0000 | 0.87 (13/15) | 0.0000 |
| random_cell | 2 | 0.86 (12/14, hòa 1) | same_day_mean → 0.0000 | 0.86 (12/14, hòa 1) | 0.0000 | 0.86 (12/14, hòa 1) | 0.0000 |
| random_cell | 3 | 0.87 (13/15) | same_day_mean → 0.0000 | 0.87 (13/15) | 0.0000 | 0.87 (13/15) | 0.0000 |
| random_cell | 4 | 0.80 (12/15) | same_day_mean → 0.0000 | 0.80 (12/15) | 0.0000 | 0.80 (12/15) | 0.0000 |
- U2: whole_feature: TB 0.86, min 0.79 (5 ca); random_cell: TB 0.86, min 0.80 (5 ca); regret ≤ 0.005 ở 9/10 ca → **đạt**.
<!-- auto:u2:end -->

### Bảng 6: U3

<!-- auto:u3:start -->
| Mask | Seed | ô dương bị cờ | tỷ lệ đáp án 0 khi bị cờ | ô dương không cờ | khi không cờ | chênh |
|---|---:|---:|---:|---:|---:|---:|
| whole_feature | 0 | 914 | 1.000 | 6564 | 0.298 | +0.702 |
| whole_feature | 1 | 1403 | 0.998 | 6362 | 0.337 | +0.661 |
| whole_feature | 2 | 1032 | 0.997 | 5812 | 0.276 | +0.721 |
| whole_feature | 3 | 1043 | 1.000 | 5075 | 0.321 | +0.679 |
| whole_feature | 4 | 965 | 1.000 | 5010 | 0.334 | +0.666 |
| random_cell | 0 | 1308 | 0.998 | 6647 | 0.261 | +0.737 |
| random_cell | 1 | 1281 | 0.995 | 6505 | 0.282 | +0.713 |
| random_cell | 2 | 1257 | 1.000 | 6527 | 0.284 | +0.716 |
| random_cell | 3 | 1191 | 1.000 | 6584 | 0.276 | +0.724 |
| random_cell | 4 | 1177 | 1.000 | 6564 | 0.288 | +0.712 |
- U3: chênh ≥ 0.2 ở 10/10 ca → **đạt**.
<!-- auto:u3:end -->

### Bảng 7: U4, đối thủ giả

<!-- auto:u4:start -->
Số ca diagnostic xếp đối thủ giả kém hơn **mọi** model thật (trên số ca có giá trị), R-local.

| Đối thủ giả | Mask | \|D1 − D1_truth\| | \|D1\| thô | \|D2 − D2_truth\| | D2 thô | D3: ô cờ / ô E | *tham chiếu: MAE Tầng 1, toàn ô model* |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| mean_fill | whole_feature | 1/5 | 1/5 | 5/5 | 5/5 | 0/5 | 1/5 |
| mean_fill | random_cell | 4/5 | 4/5 | 5/5 | 5/5 | 0/5 | 5/5 |
| zero_fill | whole_feature | 4/5 | 0/5 | NA | NA | 0/5 | 5/5 |
| zero_fill | random_cell | 5/5 | 0/5 | NA | NA | 0/5 | 5/5 |
| marginal_sampler | whole_feature | 0/5 | 0/5 | 0/2 | 0/2 | 0/5 | 2/5 |
| marginal_sampler | random_cell | 0/5 | 0/5 | 0/2 | 0/2 | 0/5 | 2/5 |

Giá trị trung bình qua 5 seed (R-local, trên target chung của model thật):

| Model | Mask | \|D1 − D1_truth\| | \|D1\| thô | \|D2 − D2_truth\| | D2 thô | D3: ô cờ / ô E | *tham chiếu: MAE Tầng 1, toàn ô model* |
|---|---|---:|---:|---:|---:|---:|---:|
| hron | whole_feature | 0.0350 ± 0.0177 | 0.0855 ± 0.0134 | 0.0439 ± 0.0290 | 0.1023 ± 0.0170 | 0.0190 ± 0.0049 | 0.0875 ± 0.0105 |
| hron | random_cell | 0.0092 ± 0.0071 | 0.0801 ± 0.0046 | 0.0681 ± 0.0058 | 0.0990 ± 0.0069 | 0.0267 ± 0.0017 | 0.0892 ± 0.0028 |
| hron_cap | whole_feature | 0.0815 ± 0.0200 | 0.0892 ± 0.0092 | 0.0493 ± 0.0348 | 0.1078 ± 0.0205 | 0.0147 ± 0.0049 | 0.0869 ± 0.0095 |
| hron_cap | random_cell | 0.0445 ± 0.0052 | 0.0692 ± 0.0058 | 0.0784 ± 0.0071 | 0.1093 ± 0.0060 | 0.0213 ± 0.0014 | 0.0879 ± 0.0025 |
| tsagris | whole_feature | 0.1303 ± 0.0377 | 0.2004 ± 0.0447 | 0.0802 ± 0.0583 | 0.1387 ± 0.0417 | 0.0752 ± 0.0148 | 0.1079 ± 0.0209 |
| tsagris | random_cell | 0.1506 ± 0.0122 | 0.2045 ± 0.0108 | 0.0846 ± 0.0097 | 0.1155 ± 0.0097 | 0.0846 ± 0.0050 | 0.0999 ± 0.0024 |
| micoda | whole_feature | 0.3318 ± 0.0338 | 0.3852 ± 0.0397 | 0.1309 ± 0.0399 | 0.1893 ± 0.0235 | 0.1772 ± 0.0278 | 0.1406 ± 0.0099 |
| micoda | random_cell | 0.3339 ± 0.0125 | 0.3844 ± 0.0094 | 0.1305 ± 0.0023 | 0.1614 ± 0.0018 | 0.1798 ± 0.0087 | 0.1317 ± 0.0018 |
| micoda_other | whole_feature | 0.1244 ± 0.0343 | 0.2060 ± 0.0368 | 0.1594 ± 0.0445 | 0.2178 ± 0.0313 | 0.1276 ± 0.0201 | 0.1075 ± 0.0093 |
| micoda_other | random_cell | 0.1484 ± 0.0138 | 0.1991 ± 0.0116 | 0.1666 ± 0.0023 | 0.1976 ± 0.0010 | 0.1322 ± 0.0077 | 0.1078 ± 0.0008 |
| same_day_mean | whole_feature | 0.0263 ± 0.0211 | 0.0907 ± 0.0165 | 0.0230 ± 0.0164 | 0.0771 ± 0.0089 | 0.0049 ± 0.0040 | 0.0351 ± 0.0066 |
| same_day_mean | random_cell | 0.0428 ± 0.0059 | 0.0933 ± 0.0070 | 0.0394 ± 0.0025 | 0.0703 ± 0.0028 | 0.0013 ± 0.0005 | 0.0310 ± 0.0027 |
| mean_fill | whole_feature | 0.2951 ± 0.0434 | 0.3521 ± 0.0515 | 0.1997 ± 0.0486 | 0.2581 ± 0.0332 | 0.1316 ± 0.0204 | 0.1354 ± 0.0145 |
| mean_fill | random_cell | 0.3376 ± 0.0080 | 0.3882 ± 0.0059 | 0.2104 ± 0.0029 | 0.2414 ± 0.0019 | 0.1501 ± 0.0048 | 0.1371 ± 0.0016 |
| zero_fill | whole_feature | 0.3705 ± 0.0210 | 0.3205 ± 0.0242 | NA | NA | 0.0000 ± 0.0000 | 0.1693 ± 0.0219 |
| zero_fill | random_cell | 0.3700 ± 0.0055 | 0.3194 ± 0.0034 | NA | NA | 0.0000 ± 0.0000 | 0.1716 ± 0.0035 |
| marginal_sampler | whole_feature | 0.1890 ± 0.0194 | 0.1402 ± 0.0198 | 0.0823 ± 0.0160 (n = 2) | 0.1256 ± 0.0285 (n = 2) | 0.0416 ± 0.0068 | 0.1354 ± 0.0120 |
| marginal_sampler | random_cell | 0.1745 ± 0.0083 | 0.1239 ± 0.0080 | 0.0719 ± 0.0015 (n = 2) | 0.1008 ± 0.0051 (n = 2) | 0.0487 ± 0.0052 | 0.1362 ± 0.0072 |
<!-- auto:u4:end -->

### Bảng 8: sensitivity

<!-- auto:sensitivity:start -->
**D1 với ngưỡng zero c/N < 0.001 (R-local)**

| Mask | Seed | thiếu: khớp (khớp/quyết định) | thừa: khớp |
|---|---:|---|---|
| whole_feature | 0 | 0.36 (5/14, hòa 1) | 0.73 (11/15) |
| whole_feature | 1 | 0.64 (9/14, hòa 1) | 0.80 (12/15) |
| whole_feature | 2 | 0.73 (11/15) | 0.87 (13/15) |
| whole_feature | 3 | 0.60 (9/15) | 0.87 (13/15) |
| whole_feature | 4 | 0.60 (9/15) | 0.80 (12/15) |
| random_cell | 0 | 0.67 (10/15) | 0.80 (12/15) |
| random_cell | 1 | 0.64 (9/14, hòa 1) | 0.80 (12/15) |
| random_cell | 2 | 0.67 (10/15) | 0.80 (12/15) |
| random_cell | 3 | 0.64 (9/14, hòa 1) | 0.80 (12/15) |
| random_cell | 4 | 0.60 (9/15) | 0.80 (12/15) |
- U1 thiếu (D1 cao ↔ MAE ô đáp án > 0): whole_feature: TB 0.59, min 0.36 (5 ca); random_cell: TB 0.64, min 0.60 (5 ca) → **không đạt**.
- U1 thừa (D1 thấp ↔ MAE ô đáp án = 0): whole_feature: TB 0.81, min 0.73 (5 ca); random_cell: TB 0.80, min 0.80 (5 ca) → **đạt**.

**D1 với R-pooled**

| Mask | Seed | thiếu: khớp (khớp/quyết định) | thừa: khớp |
|---|---:|---|---|
| whole_feature | 0 | 0.43 (6/14, hòa 1) | 0.80 (12/15) |
| whole_feature | 1 | 0.64 (9/14, hòa 1) | 0.80 (12/15) |
| whole_feature | 2 | 0.73 (11/15) | 0.87 (13/15) |
| whole_feature | 3 | 0.57 (8/14, hòa 1) | 0.80 (12/15) |
| whole_feature | 4 | 0.71 (10/14, hòa 1) | 0.87 (13/15) |
| random_cell | 0 | 0.67 (10/15) | 0.80 (12/15) |
| random_cell | 1 | 0.67 (10/15) | 0.80 (12/15) |
| random_cell | 2 | 0.67 (10/15) | 0.80 (12/15) |
| random_cell | 3 | 0.64 (9/14, hòa 1) | 0.80 (12/15) |
| random_cell | 4 | 0.67 (10/15) | 0.87 (13/15) |
- U1 thiếu (D1 cao ↔ MAE ô đáp án > 0): whole_feature: TB 0.62, min 0.43 (5 ca); random_cell: TB 0.66, min 0.64 (5 ca) → **không đạt**.
- U1 thừa (D1 thấp ↔ MAE ô đáp án = 0): whole_feature: TB 0.83, min 0.80 (5 ca); random_cell: TB 0.81, min 0.80 (5 ca) → **đạt**.

**D2 với R-pooled**

| Mask | Seed | khớp \|D2 − D2t\| | chọn → regret | khớp D2 thô | regret thô | khớp, MAE toàn ô model | regret, MAE toàn ô model |
|---|---:|---|---|---|---:|---|---:|
| whole_feature | 0 | 0.87 (13/15) | same_day_mean → 0.0000 | 0.87 (13/15) | 0.0000 | 0.87 (13/15) | 0.0000 |
| whole_feature | 1 | 0.80 (12/15) | same_day_mean → 0.0000 | 0.80 (12/15) | 0.0000 | 0.80 (12/15) | 0.0000 |
| whole_feature | 2 | 0.93 (13/14, hòa 1) | same_day_mean → 0.0000 | 0.93 (13/14, hòa 1) | 0.0000 | 0.93 (13/14, hòa 1) | 0.0000 |
| whole_feature | 3 | 0.80 (12/15) | hron → 0.0382 | 0.80 (12/15) | 0.0382 | 0.80 (12/15) | 0.0366 |
| whole_feature | 4 | 0.79 (11/14, hòa 1) | hron_cap → 0.0403 | 1.00 (14/14, hòa 1) | 0.0000 | 0.79 (11/14, hòa 1) | 0.0393 |
| random_cell | 0 | 0.93 (13/14, hòa 1) | same_day_mean → 0.0000 | 0.93 (13/14, hòa 1) | 0.0000 | 0.93 (13/14, hòa 1) | 0.0000 |
| random_cell | 1 | 0.87 (13/15) | same_day_mean → 0.0000 | 0.87 (13/15) | 0.0000 | 0.87 (13/15) | 0.0000 |
| random_cell | 2 | 0.93 (13/14, hòa 1) | same_day_mean → 0.0000 | 0.93 (13/14, hòa 1) | 0.0000 | 0.93 (13/14, hòa 1) | 0.0000 |
| random_cell | 3 | 0.87 (13/15) | same_day_mean → 0.0000 | 0.87 (13/15) | 0.0000 | 0.87 (13/15) | 0.0000 |
| random_cell | 4 | 0.87 (13/15) | same_day_mean → 0.0000 | 0.87 (13/15) | 0.0000 | 0.87 (13/15) | 0.0000 |
- U2: whole_feature: TB 0.84, min 0.79 (5 ca); random_cell: TB 0.89, min 0.87 (5 ca); regret ≤ 0.005 ở 8/10 ca → **không đạt**.
<!-- auto:sensitivity:end -->

### Bảng 9: chuyển giao, masked E so với native M_m

<!-- auto:transfer:start -->
| Tập ô | coverage R-local | Africa | Asia | Europe | North America | Oceania | South America | N1 (N cắt native) | N2 (N cắt native) | N3 (N cắt native) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| native M_m | 0.557 | 0.318 | 0.274 | 0.197 | 0.085 | 0.029 | 0.098 | 0.455 | 0.347 | 0.198 |
| masked E, whole_feature (TB 5 seed) | 0.906 | 0.163 | 0.231 | 0.381 | 0.119 | 0.021 | 0.086 | 0.269 | 0.289 | 0.442 |
| masked E, random_cell (TB 5 seed) | 0.919 | 0.129 | 0.210 | 0.407 | 0.131 | 0.027 | 0.096 | 0.241 | 0.275 | 0.484 |

Tỷ lệ 0 theo target: native = tỷ lệ zero quan sát (ô O); masked = tỷ lệ đáp án 0 trên E (TB 5 seed).

| Target | native O | wf E | rc E |
|---|---:|---:|---:|
| recombinant | 0.747 | 0.591 | 0.553 |
| 20A | 0.881 | 0.714 | 0.723 |
| 20B | 0.920 | 0.852 | 0.805 |
| 20C | 0.950 | 0.919 | 0.872 |
| 20E | 0.895 | 0.700 | 0.761 |
| Beta | 0.889 | 0.560 | 0.726 |
| Alpha | 0.833 | 0.596 | 0.611 |
| Gamma | 0.882 | 0.687 | 0.726 |
| Delta | 0.745 | 0.461 | 0.498 |
| Kappa | 0.961 | 0.942 | 0.908 |
| Epsilon | 0.939 | 0.890 | 0.850 |
| Eta | 0.928 | 0.819 | 0.823 |
| Iota | 0.952 | 0.887 | 0.895 |
| Lambda | 0.927 | 0.815 | 0.845 |
| Mu | 0.919 | 0.821 | 0.842 |
| Omicron | 0.418 | 0.331 | 0.348 |
| S:677 | 0.959 | 0.928 | 0.892 |
<!-- auto:transfer:end -->

## 5. Kết luận (trong 10 ca và support đã kiểm)

1. **D1, cảnh báo điền thiếu (U1 thiếu): không đạt.**
   - Tỷ lệ khớp TB 0.61 (wf, min 0.43) và 0.67 (rc).
   - Ba cặp sai **ở cả 10/10 ca**: same_day_mean với tsagris, micoda, micoda_other. same_day_mean có D1 cao hơn (ít ô dương hơn) nhưng MAE trên ô đáp án dương thấp hơn nhiều.
   - Sai số trên ô đáp án dương do **độ lớn** giá trị điền quyết định, không do tỷ lệ zero; D1 không thấy điều đó.
   - Sensitivity (ngưỡng c/N, R-pooled) cho cùng kết luận.
2. **D1, cảnh báo điền thừa (U1 thừa): đạt, nhưng sát ngưỡng.**
   - TB 0.84 (wf) và 0.83 (rc); min 0.80 ở cả hai mask.
   - Sai có hệ thống ở hron và hron_cap so với same_day_mean (10/10 và 9/10 ca). hron "có nhiều zero hơn" theo ngưỡng count < 0.5, nhưng MAE trên ô đáp án 0 lại cao hơn: các giá trị lẻ < 0.5 count vẫn là sai số tỷ lệ.
   - Hai sensitivity cũng đạt; với ngưỡng c/N, min của wf còn 0.73.
3. **D2, proxy xếp hạng theo MAE (U2): đạt với R-local.**
   - TB 0.86 ở cả hai mask, min 0.79.
   - Regret ≤ 0.005 ở 9/10 ca: diagnostic chọn same_day_mean, model có MAE nhỏ nhất, ở 9 ca. Ca còn lại (wf seed 4) chọn hron_cap, regret 0.046.
   - Sai **ở cả 10/10 ca** ở cặp micoda với micoda_other: D2 chỉ nhìn phần dương nên không thấy lỗi điền thừa trên ô đáp án 0 của A17.
   - **Với R-pooled thì không đạt** (regret ≤ 0.005 chỉ ở 8/10 ca).
   - D2 thô (không trừ truth) cho tỷ lệ khớp ngang hoặc cao hơn, nên hàng truth không làm U2 tốt lên.
   - Phần lớn 15 cặp là cặp dễ phân biệt. Kết quả này chưa cho biết D2 có tách được hai model gần nhau hay không.
4. **D3, cờ ngoài support (U3): đạt.**
   - Ô dương bị cờ có đáp án 0 ở 99.5–100% trường hợp, so với 26–34% khi không bị cờ. Chênh từ +0.66 đến +0.74 ở 10/10 ca.
   - Số ô bị cờ ít: 900–1 400 trên mỗi ca, gộp 6 model.
5. **U4, đối thủ giả** (R-local, không có ngưỡng):
   - **mean_fill:** \|D2 − D2_truth\| xếp nó kém hơn mọi model thật ở 10/10 ca. \|D1 − D1_truth\| chỉ 1/5 (wf) và 4/5 (rc).
   - **zero_fill:** \|D1 − D1_truth\| bắt được ở 9/10 ca; \|D1\| thô bắt được 0/10. D2 không định nghĩa (không có ô dương).
   - **marginal_sampler:** không diagnostic nào xếp nó kém hơn mọi model thật (0/5 mỗi mask; D2 chỉ có giá trị ở 2/5 ca).
   - **D3** không bắt được đối thủ giả nào: zero_fill không có ô dương, nên cũng không bao giờ bị cờ.
   - **Dòng tham chiếu MAE Tầng 1** (không phải diagnostic) cũng chỉ xếp marginal_sampler kém hơn mọi model thật ở 2/5 ca mỗi mask, và mean_fill ở 1/5 (wf). Lý do: MICoDa A17 có MAE ngang hoặc cao hơn hai đối thủ giả này.
6. **Giải thích kết quả marginal_sampler ở step 5.**
   - D1 R-pooled thô của nó trên ô che là +0.033, gần 0, giống step 5 native (+0.023). Nhưng **chính đáp án** có D1 R-pooled là −0.142: ô che có ít zero hơn reference.
   - Vì vậy "giống reference" không có nghĩa là "giống đáp án". Một sampler chép phân phối quan sát sẽ trông tốt theo D1 thô.
   - Đo bằng khoảng cách tới truth, D1 của nó (0.19 wf / 0.17 rc) xa hơn hron, hron_cap, same_day_mean, tsagris và micoda_other. Nó chỉ gần hơn micoda (0.33), ở cả hai mask.
   - D1/D2 là diagnostic phân phối, nên không thấy việc gán sai giá trị cho từng ô.
   - Tóm lại, ở step 5 marginal_sampler trông hợp lý vì hai lý do: diagnostic thô thưởng cho việc chép reference, và reference lệch khỏi phân phối thật của ô thiếu.
7. **Reference lệch khỏi phân phối thật** (bảng 3):
   - D1_truth R-local TB −0.050 (wf) và −0.051 (rc); R-pooled −0.142 và −0.137. Reference có nhiều zero hơn ô che, và R-pooled lệch gần gấp ba R-local.
   - Lệch lớn ở recombinant (R-local −0.21), Beta (R-pooled −0.33 wf) và Delta (R-pooled −0.29 wf).
   - D2_truth > 0 ở mọi target (R-local TB 0.058 wf, 0.031 rc).

## 6. Điều gì mang sang được Tầng 2 native

Bảng 9 cho thấy ô che E và ô thiếu thật M_m khác nhau ở **mọi** trục đã khai báo:
- **coverage R-local:** 0.91–0.92 (E) so với 0.56 (M_m);
- **châu lục:** Châu Âu chiếm 0.38–0.41 của E nhưng 0.20 của M_m; Châu Phi 0.13–0.16 so với 0.32;
- **nhóm N:** N3 chiếm 0.44–0.48 của E nhưng 0.20 của M_m; N1 0.24–0.27 so với 0.46;
- **tỷ lệ 0 theo target:** tỷ lệ đáp án 0 trên E thấp hơn tỷ lệ zero quan sát ở mọi target. Ví dụ Beta 0.56–0.73 so với 0.89, Delta 0.46–0.50 so với 0.75. Hai số không cùng loại: E đã loại ô ép về 0.

Vì vậy:
- **Có thể mang sang, kèm giả định:**
  - cờ D3 đánh dấu ô điền dương ngoài khoảng quan sát mà gần như luôn sai trên ô che;
  - D1 dùng để cảnh báo điền thừa.
  
  Cả hai đúng trong support chủ yếu là Châu Âu và N lớn. Trên native, phần lớn ô thiếu nằm ở Châu Phi, Châu Á và N nhỏ, nơi Tầng 3 có ít ca kiểm.
- **Không mang sang được:**
  - D1 làm cảnh báo điền thiếu (không đạt ngay trên ô che);
  - mọi số thô so với reference;
  - mọi kết luận ở phần R-local không có support của native (44% M_m, toàn bộ Châu Đại Dương).
- **Mang sang có điều kiện:** D2 làm proxy chọn model. Nó đạt với R-local trên ô che, nhưng native chỉ có 56% M_m có support. Nó cũng không thấy lỗi điền thừa trên ô zero, nên phải đọc cùng D1.

Không có câu nào ở đây là độ chính xác native.

## 7. Quyết định thiết kế chưa chắc

1. **Một số cho mỗi (ca, model)**, tính trên tập target chung. Cách khác là so cặp theo từng (ca, target), sẽ cho nhiều cặp hơn và kết quả có thể khác.
2. **Sai số Tầng 1 chỉ tính trên các ô diagnostic chấm** (cùng tập ô). Với U2, bản tính trên toàn ô model của `tier01` cho cùng kết luận từng ca, trừ regret ở wf seed 4 (0.039 so với 0.046).
3. **Trong U1, khoảng cách tới truth không đổi gì.** Δ(D1 − D1_truth) giữa hai model bằng ΔD1, vì D1_truth là chung. Hàng truth chỉ có tác dụng khi so tuyệt đối (U4, bảng 2).
4. **Δ diagnostic = 0 tính là không khớp.** Đây là lựa chọn thận trọng; trên COVID không có cặp nào như vậy sau khi bỏ tsagris_cap.
5. **"≥ 0.80 ở cả hai mask"** được hiểu là trung bình của từng mask, tức chặt hơn trung bình gộp 10 ca.
6. **U3 gộp ô của 6 model thật.** Số theo từng model nằm trong `tier3_cases.json`.
7. **15 cặp của 6 model, phần lớn dễ phân biệt.** Ngưỡng hòa 0.001 chỉ loại vài cặp. Review 9.1 cảnh báo rằng agreement cao có thể chỉ phản ánh các cặp dễ.
8. **10 ca không độc lập**: cùng một bảng, và seed 0–4 đã được nhìn ở các step trước. Mọi kết quả là exploratory.
9. **Dòng tham chiếu MAE Tầng 1 ở bảng 7** được thêm sau khi thấy số (mục 2).

## 8. Việc để lại

- **D3: giữ** làm cờ soát ô. Cần kiểm thêm ở support giống native: Châu Phi, N nhỏ, ô gần mép T_j.
- **D1:**
  - **giữ** cho cảnh báo điền thừa, nhưng biên mỏng;
  - **bỏ use case điền thiếu**, hoặc định nghĩa lại bằng một đại lượng nhìn được độ lớn (ví dụ phần khối lượng thiếu trên ô dương), ở lượt sau;
  - ngưỡng count < 0.5 gộp giá trị lẻ thành zero. Sensitivity theo tỷ lệ không cải thiện, nên chưa có lý do đổi.
- **D2: giữ** làm proxy chọn model, chỉ với R-local, và luôn đọc cùng D1 (cặp micoda/micoda_other). Bỏ bản R-pooled cho U2.
- **Đối thủ giả:** không diagnostic nào bắt được marginal_sampler. Nếu cần bắt, phải có diagnostic nhìn được việc gán sai từng ô (ví dụ phụ thuộc thời gian trong chuỗi). Việc này nằm ngoài phạm vi step này.
- **Điều kiện cho một lượt xác nhận trên seed mới:**
  - khai báo trước bộ seed (ví dụ 5–9);
  - giữ nguyên `tier3_calibration.py` (hash trong manifest), cùng 6 model thật, cùng ngưỡng;
  - chỉ kiểm các cặp diagnostic × use case đã đạt ở đây (D1 thừa, D2 R-local, D3);
  - thêm một mask có support gần native hơn (Châu Phi, N nhỏ), khai báo trước;
  - không chạy lại nếu kết quả không đạt.
- **Còn mở:** so cặp theo từng target, ngưỡng hòa tương đối, bootstrap/CI, Tầng 4.
