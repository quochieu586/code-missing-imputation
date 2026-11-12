# Step 8: evaluation v1, lượt xác nhận (seed 200–209)

Sinh bởi `evaluate_v1.py`; không sửa tay.

## 7. Giả thuyết

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

## Kiểm tra

Seed xác nhận 200–209: không xuất hiện trong 29 file config / manifest của step 1–7 (`v1.check_unused`).
Mask: 30 mask đã lưu khớp `masks_manifest.json`; F2 đạt ở cả ba mask; long file khớp `summary.json` (tier01.load_run).

## 1. Tầng 0

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

## 2. Tầng 1

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


## 3. Ghép cặp

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


## 4. Tầng 1 theo tầng

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


## 5. Tầng 2

Lượt native chạy lại qua `evaluate_v1.py` (`tier2.summarize`, CSV native của step 8 làm lượt chạy lại); so từng byte với output step 5:

| File | giống step 5 |
|---|:---:|
| tier2_targets.csv | True |
| tier2_strata.csv | True |
| tier2_support.csv | True |
| tier2_results.md | True |
| tier2_manifest.json | True |

## 6. Tầng 3

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

## 6. Tầng 3 chi tiết whole_feature

| Mask | Seed | thiếu: khớp (khớp/quyết định) | thừa: khớp |
|---|---:|---|---|
| whole_feature | 200 | NA | NA |
| whole_feature | 201 | 0.73 (11/15) | 0.80 (12/15) |
| whole_feature | 202 | 0.60 (9/15) | 0.80 (12/15) |
| whole_feature | 203 | NA | NA |
| whole_feature | 204 | 0.43 (6/14, hòa 1) | 0.80 (12/15) |
| whole_feature | 205 | 0.60 (9/15) | 0.87 (13/15) |
| whole_feature | 206 | 0.67 (10/15) | 0.86 (12/14, hòa 1) |
| whole_feature | 207 | 0.80 (12/15) | 0.87 (13/15) |
| whole_feature | 208 | 0.67 (10/15) | 0.80 (12/15) |
| whole_feature | 209 | 0.67 (10/15) | 0.87 (13/15) |
- U1 thiếu (D1 cao ↔ MAE ô đáp án > 0): whole_feature: TB 0.65, min 0.43 (8 ca) → **không đạt**.
- U1 thừa (D1 thấp ↔ MAE ô đáp án = 0): whole_feature: TB 0.83, min 0.80 (8 ca) → **đạt**.

| Mask | Seed | khớp \|D2 − D2t\| | chọn → regret | khớp D2 thô | regret thô | khớp, MAE toàn ô model | regret, MAE toàn ô model |
|---|---:|---|---|---|---:|---|---:|
| whole_feature | 200 | NA | same_day_mean → 0.0000 | NA | 0.0000 | NA | 0.0000 |
| whole_feature | 201 | 0.47 (7/15) | same_day_mean → 0.0000 | 0.47 (7/15) | 0.0000 | 0.47 (7/15) | 0.0000 |
| whole_feature | 202 | 0.93 (14/15) | same_day_mean → 0.0000 | 0.93 (14/15) | 0.0000 | 0.93 (14/15) | 0.0000 |
| whole_feature | 203 | NA | same_day_mean → 0.0000 | NA | 0.0000 | NA | 0.0000 |
| whole_feature | 204 | 0.87 (13/15) | same_day_mean → 0.0000 | 0.87 (13/15) | 0.0000 | 0.87 (13/15) | 0.0000 |
| whole_feature | 205 | 0.93 (13/14, hòa 1) | same_day_mean → 0.0000 | 0.93 (13/14, hòa 1) | 0.0000 | 0.93 (13/14, hòa 1) | 0.0000 |
| whole_feature | 206 | 0.80 (12/15) | same_day_mean → 0.0000 | 0.80 (12/15) | 0.0000 | 0.80 (12/15) | 0.0000 |
| whole_feature | 207 | 0.93 (13/14, hòa 1) | same_day_mean → 0.0000 | 0.93 (13/14, hòa 1) | 0.0000 | 0.93 (13/14, hòa 1) | 0.0000 |
| whole_feature | 208 | 0.80 (12/15) | same_day_mean → 0.0000 | 0.80 (12/15) | 0.0000 | 0.80 (12/15) | 0.0000 |
| whole_feature | 209 | 0.92 (11/12, hòa 3) | same_day_mean → 0.0000 | 0.92 (11/12, hòa 3) | 0.0000 | 0.93 (13/14, hòa 1) | 0.0000 |
- U2: whole_feature: TB 0.83, min 0.47 (8 ca); regret ≤ 0.005 ở 10/10 ca → **không đạt**.

| Mask | Seed | ô dương bị cờ | tỷ lệ đáp án 0 khi bị cờ | ô dương không cờ | khi không cờ | chênh |
|---|---:|---:|---:|---:|---:|---:|
| whole_feature | 200 | 17 | 1.000 | 1017 | 0.198 | +0.802 |
| whole_feature | 201 | 1404 | 1.000 | 7093 | 0.259 | +0.741 |
| whole_feature | 202 | 1145 | 0.998 | 6560 | 0.223 | +0.775 |
| whole_feature | 203 | 19 | 1.000 | 938 | 0.212 | +0.788 |
| whole_feature | 204 | 1119 | 1.000 | 6036 | 0.287 | +0.713 |
| whole_feature | 205 | 1112 | 0.995 | 6237 | 0.216 | +0.778 |
| whole_feature | 206 | 1469 | 0.965 | 6364 | 0.210 | +0.756 |
| whole_feature | 207 | 1278 | 1.000 | 7489 | 0.268 | +0.732 |
| whole_feature | 208 | 1185 | 1.000 | 7063 | 0.254 | +0.746 |
| whole_feature | 209 | 1184 | 1.000 | 6770 | 0.308 | +0.692 |
- U3: chênh ≥ 0.2 ở 10/10 ca → **đạt**.

Số ca diagnostic xếp đối thủ giả kém hơn **mọi** model thật (trên số ca có giá trị), R-local.

| Đối thủ giả | Mask | \|D1 − D1_truth\| | \|D1\| thô | \|D2 − D2_truth\| | D2 thô | D3: ô cờ / ô E | *tham chiếu: MAE Tầng 1, toàn ô model* |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| mean_fill | whole_feature | 5/8 | 5/8 | 6/6 | 6/6 | 0/8 | 8/8 |
| zero_fill | whole_feature | 8/8 | 3/8 | NA | NA | 0/8 | 8/8 |
| marginal_sampler | whole_feature | 0/8 | 0/8 | 0/4 | 0/4 | 0/8 | 8/8 |

Giá trị trung bình qua 5 seed (R-local, trên target chung của model thật):

| Model | Mask | \|D1 − D1_truth\| | \|D1\| thô | \|D2 − D2_truth\| | D2 thô | D3: ô cờ / ô E | *tham chiếu: MAE Tầng 1, toàn ô model* |
|---|---|---:|---:|---:|---:|---:|---:|
| hron | whole_feature | 0.0215 ± 0.0141 (n = 8) | 0.0983 ± 0.0127 (n = 8) | 0.0543 ± 0.0161 (n = 8) | 0.1001 ± 0.0152 (n = 8) | 0.0270 ± 0.0097 (n = 8) | 0.0982 ± 0.0141 (n = 8) |
| hron_cap | whole_feature | 0.0759 ± 0.0215 (n = 8) | 0.0960 ± 0.0172 (n = 8) | 0.0578 ± 0.0115 (n = 8) | 0.1036 ± 0.0127 (n = 8) | 0.0199 ± 0.0066 (n = 8) | 0.0981 ± 0.0138 (n = 8) |
| tsagris | whole_feature | 0.1275 ± 0.0238 (n = 8) | 0.1955 ± 0.0100 (n = 8) | 0.0736 ± 0.0180 (n = 8) | 0.1193 ± 0.0132 (n = 8) | 0.0799 ± 0.0085 (n = 8) | 0.1036 ± 0.0134 (n = 8) |
| micoda | whole_feature | 0.3145 ± 0.0265 (n = 8) | 0.3680 ± 0.0363 (n = 8) | 0.1273 ± 0.0130 (n = 8) | 0.1731 ± 0.0198 (n = 8) | 0.1794 ± 0.0209 (n = 8) | 0.1373 ± 0.0155 (n = 8) |
| micoda_other | whole_feature | 0.1323 ± 0.0222 (n = 8) | 0.2119 ± 0.0343 (n = 8) | 0.1514 ± 0.0122 (n = 8) | 0.1972 ± 0.0160 (n = 8) | 0.1356 ± 0.0207 (n = 8) | 0.1169 ± 0.0139 (n = 8) |
| same_day_mean | whole_feature | 0.0296 ± 0.0151 | 0.0903 ± 0.0139 | 0.0230 ± 0.0091 | 0.0676 ± 0.0087 | 0.0029 ± 0.0025 | 0.0318 ± 0.0075 |
| mean_fill | whole_feature | 0.3144 ± 0.0374 | 0.3710 ± 0.0379 | 0.2174 ± 0.0164 (n = 7) | 0.2614 ± 0.0206 (n = 7) | 0.1517 ± 0.0200 | 0.1505 ± 0.0152 |
| zero_fill | whole_feature | 0.3892 ± 0.0272 | 0.3381 ± 0.0265 | NA | NA | 0.0000 ± 0.0000 | 0.1904 ± 0.0280 |
| marginal_sampler | whole_feature | 0.1799 ± 0.0204 | 0.1309 ± 0.0155 | 0.0626 ± 0.0190 (n = 5) | 0.1080 ± 0.0256 (n = 5) | 0.0534 ± 0.0088 | 0.1498 ± 0.0154 |

**D1 với ngưỡng zero c/N < 0.001 (R-local)**

| Mask | Seed | thiếu: khớp (khớp/quyết định) | thừa: khớp |
|---|---:|---|---|
| whole_feature | 200 | NA | NA |
| whole_feature | 201 | 0.80 (12/15) | 0.87 (13/15) |
| whole_feature | 202 | 0.60 (9/15) | 0.80 (12/15) |
| whole_feature | 203 | NA | NA |
| whole_feature | 204 | 0.36 (5/14, hòa 1) | 0.87 (13/15) |
| whole_feature | 205 | 0.53 (8/15) | 0.80 (12/15) |
| whole_feature | 206 | 0.67 (10/15) | 0.86 (12/14, hòa 1) |
| whole_feature | 207 | 0.73 (11/15) | 0.80 (12/15) |
| whole_feature | 208 | 0.67 (10/15) | 0.80 (12/15) |
| whole_feature | 209 | 0.67 (10/15) | 0.87 (13/15) |
- U1 thiếu (D1 cao ↔ MAE ô đáp án > 0): whole_feature: TB 0.63, min 0.36 (8 ca) → **không đạt**.
- U1 thừa (D1 thấp ↔ MAE ô đáp án = 0): whole_feature: TB 0.83, min 0.80 (8 ca) → **đạt**.

**D1 với R-pooled**

| Mask | Seed | thiếu: khớp (khớp/quyết định) | thừa: khớp |
|---|---:|---|---|
| whole_feature | 200 | NA | NA |
| whole_feature | 201 | 0.73 (11/15) | 0.80 (12/15) |
| whole_feature | 202 | 0.60 (9/15) | 0.80 (12/15) |
| whole_feature | 203 | NA | NA |
| whole_feature | 204 | 0.38 (5/13, hòa 2) | 0.80 (12/15) |
| whole_feature | 205 | 0.60 (9/15) | 0.87 (13/15) |
| whole_feature | 206 | 0.67 (10/15) | 0.80 (12/15) |
| whole_feature | 207 | 0.80 (12/15) | 0.87 (13/15) |
| whole_feature | 208 | 0.67 (10/15) | 0.80 (12/15) |
| whole_feature | 209 | 0.67 (10/15) | 0.86 (12/14, hòa 1) |
- U1 thiếu (D1 cao ↔ MAE ô đáp án > 0): whole_feature: TB 0.64, min 0.38 (8 ca) → **không đạt**.
- U1 thừa (D1 thấp ↔ MAE ô đáp án = 0): whole_feature: TB 0.82, min 0.80 (8 ca) → **đạt**.

**D2 với R-pooled**

| Mask | Seed | khớp \|D2 − D2t\| | chọn → regret | khớp D2 thô | regret thô | khớp, MAE toàn ô model | regret, MAE toàn ô model |
|---|---:|---|---|---|---:|---|---:|
| whole_feature | 200 | NA | same_day_mean → 0.0000 | NA | 0.0000 | NA | 0.0000 |
| whole_feature | 201 | 0.47 (7/15) | same_day_mean → 0.0000 | 0.47 (7/15) | 0.0000 | 0.47 (7/15) | 0.0000 |
| whole_feature | 202 | 0.93 (14/15) | same_day_mean → 0.0000 | 0.93 (14/15) | 0.0000 | 0.93 (14/15) | 0.0000 |
| whole_feature | 203 | NA | same_day_mean → 0.0000 | NA | 0.0000 | NA | 0.0000 |
| whole_feature | 204 | 0.87 (13/15) | same_day_mean → 0.0000 | 0.87 (13/15) | 0.0000 | 0.87 (13/15) | 0.0000 |
| whole_feature | 205 | 1.00 (14/14, hòa 1) | same_day_mean → 0.0000 | 1.00 (14/14, hòa 1) | 0.0000 | 1.00 (14/14, hòa 1) | 0.0000 |
| whole_feature | 206 | 0.73 (11/15) | same_day_mean → 0.0000 | 0.73 (11/15) | 0.0000 | 0.73 (11/15) | 0.0000 |
| whole_feature | 207 | 0.93 (13/14, hòa 1) | same_day_mean → 0.0000 | 0.93 (13/14, hòa 1) | 0.0000 | 0.93 (13/14, hòa 1) | 0.0000 |
| whole_feature | 208 | 0.80 (12/15) | same_day_mean → 0.0000 | 0.80 (12/15) | 0.0000 | 0.80 (12/15) | 0.0000 |
| whole_feature | 209 | 0.93 (13/14, hòa 1) | same_day_mean → 0.0000 | 0.93 (13/14, hòa 1) | 0.0000 | 0.93 (13/14, hòa 1) | 0.0000 |
- U2: whole_feature: TB 0.83, min 0.47 (8 ca); regret ≤ 0.005 ở 10/10 ca → **không đạt**.

## 6. Tầng 3 chi tiết whole_feature_matched

| Mask | Seed | thiếu: khớp (khớp/quyết định) | thừa: khớp |
|---|---:|---|---|
| whole_feature_matched | 200 | 0.79 (11/14, hòa 1) | 0.87 (13/15) |
| whole_feature_matched | 201 | 0.60 (9/15) | 0.87 (13/15) |
| whole_feature_matched | 202 | 0.73 (11/15) | 0.80 (12/15) |
| whole_feature_matched | 203 | 0.73 (11/15) | 0.87 (13/15) |
| whole_feature_matched | 204 | 0.80 (12/15) | 0.80 (12/15) |
| whole_feature_matched | 205 | 0.80 (12/15) | 0.86 (12/14, hòa 1) |
| whole_feature_matched | 206 | 0.71 (10/14, hòa 1) | 0.80 (12/15) |
| whole_feature_matched | 207 | 0.71 (10/14, hòa 1) | 0.80 (12/15) |
| whole_feature_matched | 208 | 0.79 (11/14, hòa 1) | 0.87 (13/15) |
| whole_feature_matched | 209 | 0.80 (12/15) | 0.87 (13/15) |
- U1 thiếu (D1 cao ↔ MAE ô đáp án > 0): whole_feature_matched: TB 0.75, min 0.60 (10 ca) → **không đạt**.
- U1 thừa (D1 thấp ↔ MAE ô đáp án = 0): whole_feature_matched: TB 0.84, min 0.80 (10 ca) → **đạt**.

| Mask | Seed | khớp \|D2 − D2t\| | chọn → regret | khớp D2 thô | regret thô | khớp, MAE toàn ô model | regret, MAE toàn ô model |
|---|---:|---|---|---|---:|---|---:|
| whole_feature_matched | 200 | 0.87 (13/15) | same_day_mean → 0.0000 | 0.87 (13/15) | 0.0000 | 0.86 (12/14, hòa 1) | 0.0000 |
| whole_feature_matched | 201 | 0.80 (12/15) | same_day_mean → 0.0000 | 0.80 (12/15) | 0.0000 | 0.80 (12/15) | 0.0000 |
| whole_feature_matched | 202 | 0.73 (11/15) | same_day_mean → 0.0000 | 0.73 (11/15) | 0.0000 | 0.79 (11/14, hòa 1) | 0.0000 |
| whole_feature_matched | 203 | 0.93 (13/14, hòa 1) | same_day_mean → 0.0000 | 0.93 (13/14, hòa 1) | 0.0000 | 0.93 (13/14, hòa 1) | 0.0000 |
| whole_feature_matched | 204 | 0.64 (9/14, hòa 1) | same_day_mean → 0.0000 | 0.64 (9/14, hòa 1) | 0.0000 | 0.69 (9/13, hòa 2) | 0.0000 |
| whole_feature_matched | 205 | 0.93 (13/14, hòa 1) | same_day_mean → 0.0000 | 0.93 (13/14, hòa 1) | 0.0000 | 0.86 (12/14, hòa 1) | 0.0000 |
| whole_feature_matched | 206 | 0.67 (10/15) | same_day_mean → 0.0000 | 0.67 (10/15) | 0.0000 | 0.67 (10/15) | 0.0000 |
| whole_feature_matched | 207 | 0.86 (12/14, hòa 1) | same_day_mean → 0.0000 | 0.86 (12/14, hòa 1) | 0.0000 | 0.86 (12/14, hòa 1) | 0.0000 |
| whole_feature_matched | 208 | 0.50 (7/14, hòa 1) | same_day_mean → 0.0000 | 0.50 (7/14, hòa 1) | 0.0000 | 0.47 (7/15) | 0.0000 |
| whole_feature_matched | 209 | 1.00 (13/13, hòa 2) | same_day_mean → 0.0000 | 1.00 (13/13, hòa 2) | 0.0000 | 0.92 (12/13, hòa 2) | 0.0000 |
- U2: whole_feature_matched: TB 0.79, min 0.50 (10 ca); regret ≤ 0.005 ở 10/10 ca → **không đạt**.

| Mask | Seed | ô dương bị cờ | tỷ lệ đáp án 0 khi bị cờ | ô dương không cờ | khi không cờ | chênh |
|---|---:|---:|---:|---:|---:|---:|
| whole_feature_matched | 200 | 1118 | 1.000 | 6049 | 0.315 | +0.685 |
| whole_feature_matched | 201 | 1267 | 1.000 | 7303 | 0.328 | +0.672 |
| whole_feature_matched | 202 | 1016 | 0.997 | 6979 | 0.271 | +0.726 |
| whole_feature_matched | 203 | 1466 | 1.000 | 6892 | 0.242 | +0.758 |
| whole_feature_matched | 204 | 1380 | 1.000 | 6741 | 0.223 | +0.777 |
| whole_feature_matched | 205 | 1293 | 1.000 | 5831 | 0.277 | +0.723 |
| whole_feature_matched | 206 | 1064 | 0.992 | 6114 | 0.325 | +0.667 |
| whole_feature_matched | 207 | 1312 | 0.996 | 6634 | 0.258 | +0.738 |
| whole_feature_matched | 208 | 1522 | 1.000 | 7198 | 0.216 | +0.784 |
| whole_feature_matched | 209 | 1173 | 1.000 | 5934 | 0.272 | +0.728 |
- U3: chênh ≥ 0.2 ở 10/10 ca → **đạt**.

Số ca diagnostic xếp đối thủ giả kém hơn **mọi** model thật (trên số ca có giá trị), R-local.

| Đối thủ giả | Mask | \|D1 − D1_truth\| | \|D1\| thô | \|D2 − D2_truth\| | D2 thô | D3: ô cờ / ô E | *tham chiếu: MAE Tầng 1, toàn ô model* |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| mean_fill | whole_feature_matched | 8/10 | 8/10 | 9/9 | 9/9 | 1/10 | 10/10 |
| zero_fill | whole_feature_matched | 10/10 | 5/10 | NA | NA | 0/10 | 10/10 |
| marginal_sampler | whole_feature_matched | 0/10 | 0/10 | 0/9 | 0/9 | 0/10 | 10/10 |

Giá trị trung bình qua 5 seed (R-local, trên target chung của model thật):

| Model | Mask | \|D1 − D1_truth\| | \|D1\| thô | \|D2 − D2_truth\| | D2 thô | D3: ô cờ / ô E | *tham chiếu: MAE Tầng 1, toàn ô model* |
|---|---|---:|---:|---:|---:|---:|---:|
| hron | whole_feature_matched | 0.0302 ± 0.0127 | 0.0833 ± 0.0168 | 0.0737 ± 0.0353 | 0.1247 ± 0.0293 | 0.0270 ± 0.0093 | 0.1156 ± 0.0249 |
| hron_cap | whole_feature_matched | 0.0634 ± 0.0232 | 0.0864 ± 0.0159 | 0.0826 ± 0.0378 | 0.1336 ± 0.0314 | 0.0235 ± 0.0092 | 0.1139 ± 0.0251 |
| tsagris | whole_feature_matched | 0.1645 ± 0.0369 | 0.2273 ± 0.0344 | 0.1185 ± 0.0256 | 0.1696 ± 0.0189 | 0.0905 ± 0.0154 | 0.1260 ± 0.0201 |
| micoda | whole_feature_matched | 0.2764 ± 0.0382 | 0.3231 ± 0.0442 | 0.1598 ± 0.0249 | 0.2109 ± 0.0230 | 0.1637 ± 0.0133 | 0.1326 ± 0.0242 |
| micoda_other | whole_feature_matched | 0.1440 ± 0.0298 | 0.2189 ± 0.0221 | 0.1844 ± 0.0290 | 0.2355 ± 0.0258 | 0.1310 ± 0.0154 | 0.1230 ± 0.0244 |
| same_day_mean | whole_feature_matched | 0.0327 ± 0.0233 | 0.0878 ± 0.0143 | 0.0380 ± 0.0127 | 0.0890 ± 0.0158 | 0.0054 ± 0.0070 | 0.0363 ± 0.0052 |
| mean_fill | whole_feature_matched | 0.2893 ± 0.0379 | 0.3430 ± 0.0345 | 0.2425 ± 0.0390 (n = 9) | 0.2942 ± 0.0294 (n = 9) | 0.1417 ± 0.0184 | 0.1634 ± 0.0272 |
| zero_fill | whole_feature_matched | 0.3772 ± 0.0307 | 0.3336 ± 0.0341 | NA | NA | 0.0000 ± 0.0000 | 0.2079 ± 0.0376 |
| marginal_sampler | whole_feature_matched | 0.1601 ± 0.0230 | 0.1168 ± 0.0196 | 0.0728 ± 0.0201 (n = 9) | 0.1246 ± 0.0221 (n = 9) | 0.0539 ± 0.0119 | 0.1642 ± 0.0305 |

**D1 với ngưỡng zero c/N < 0.001 (R-local)**

| Mask | Seed | thiếu: khớp (khớp/quyết định) | thừa: khớp |
|---|---:|---|---|
| whole_feature_matched | 200 | 0.71 (10/14, hòa 1) | 0.80 (12/15) |
| whole_feature_matched | 201 | 0.67 (10/15) | 0.80 (12/15) |
| whole_feature_matched | 202 | 0.73 (11/15) | 0.80 (12/15) |
| whole_feature_matched | 203 | 0.67 (10/15) | 0.80 (12/15) |
| whole_feature_matched | 204 | 0.73 (11/15) | 0.87 (13/15) |
| whole_feature_matched | 205 | 0.73 (11/15) | 0.86 (12/14, hòa 1) |
| whole_feature_matched | 206 | 0.71 (10/14, hòa 1) | 0.80 (12/15) |
| whole_feature_matched | 207 | 0.71 (10/14, hòa 1) | 0.80 (12/15) |
| whole_feature_matched | 208 | 0.79 (11/14, hòa 1) | 0.87 (13/15) |
| whole_feature_matched | 209 | 0.73 (11/15) | 0.80 (12/15) |
- U1 thiếu (D1 cao ↔ MAE ô đáp án > 0): whole_feature_matched: TB 0.72, min 0.67 (10 ca) → **không đạt**.
- U1 thừa (D1 thấp ↔ MAE ô đáp án = 0): whole_feature_matched: TB 0.82, min 0.80 (10 ca) → **đạt**.

**D1 với R-pooled**

| Mask | Seed | thiếu: khớp (khớp/quyết định) | thừa: khớp |
|---|---:|---|---|
| whole_feature_matched | 200 | 0.73 (11/15) | 0.87 (13/15) |
| whole_feature_matched | 201 | 0.57 (8/14, hòa 1) | 0.87 (13/15) |
| whole_feature_matched | 202 | 0.73 (11/15) | 0.80 (12/15) |
| whole_feature_matched | 203 | 0.73 (11/15) | 0.87 (13/15) |
| whole_feature_matched | 204 | 0.80 (12/15) | 0.80 (12/15) |
| whole_feature_matched | 205 | 0.80 (12/15) | 0.86 (12/14, hòa 1) |
| whole_feature_matched | 206 | 0.71 (10/14, hòa 1) | 0.80 (12/15) |
| whole_feature_matched | 207 | 0.71 (10/14, hòa 1) | 0.80 (12/15) |
| whole_feature_matched | 208 | 0.79 (11/14, hòa 1) | 0.87 (13/15) |
| whole_feature_matched | 209 | 0.80 (12/15) | 0.87 (13/15) |
- U1 thiếu (D1 cao ↔ MAE ô đáp án > 0): whole_feature_matched: TB 0.74, min 0.57 (10 ca) → **không đạt**.
- U1 thừa (D1 thấp ↔ MAE ô đáp án = 0): whole_feature_matched: TB 0.84, min 0.80 (10 ca) → **đạt**.

**D2 với R-pooled**

| Mask | Seed | khớp \|D2 − D2t\| | chọn → regret | khớp D2 thô | regret thô | khớp, MAE toàn ô model | regret, MAE toàn ô model |
|---|---:|---|---|---|---:|---|---:|
| whole_feature_matched | 200 | 0.86 (12/14, hòa 1) | same_day_mean → 0.0000 | 0.86 (12/14, hòa 1) | 0.0000 | 0.86 (12/14, hòa 1) | 0.0000 |
| whole_feature_matched | 201 | 0.80 (12/15) | same_day_mean → 0.0000 | 0.80 (12/15) | 0.0000 | 0.80 (12/15) | 0.0000 |
| whole_feature_matched | 202 | 0.86 (12/14, hòa 1) | same_day_mean → 0.0000 | 0.86 (12/14, hòa 1) | 0.0000 | 0.86 (12/14, hòa 1) | 0.0000 |
| whole_feature_matched | 203 | 0.93 (13/14, hòa 1) | same_day_mean → 0.0000 | 0.93 (13/14, hòa 1) | 0.0000 | 0.93 (13/14, hòa 1) | 0.0000 |
| whole_feature_matched | 204 | 0.69 (9/13, hòa 2) | same_day_mean → 0.0000 | 0.69 (9/13, hòa 2) | 0.0000 | 0.69 (9/13, hòa 2) | 0.0000 |
| whole_feature_matched | 205 | 0.86 (12/14, hòa 1) | same_day_mean → 0.0000 | 0.86 (12/14, hòa 1) | 0.0000 | 0.86 (12/14, hòa 1) | 0.0000 |
| whole_feature_matched | 206 | 0.67 (10/15) | same_day_mean → 0.0000 | 0.67 (10/15) | 0.0000 | 0.67 (10/15) | 0.0000 |
| whole_feature_matched | 207 | 0.86 (12/14, hòa 1) | same_day_mean → 0.0000 | 0.86 (12/14, hòa 1) | 0.0000 | 0.86 (12/14, hòa 1) | 0.0000 |
| whole_feature_matched | 208 | 0.50 (7/14, hòa 1) | same_day_mean → 0.0000 | 0.50 (7/14, hòa 1) | 0.0000 | 0.47 (7/15) | 0.0000 |
| whole_feature_matched | 209 | 1.00 (11/11, hòa 4) | same_day_mean → 0.0000 | 1.00 (11/11, hòa 4) | 0.0000 | 0.92 (12/13, hòa 2) | 0.0000 |
- U2: whole_feature_matched: TB 0.80, min 0.50 (10 ca); regret ≤ 0.005 ở 10/10 ca → **đạt**.

## 6. Tầng 3 chi tiết random_cell

| Mask | Seed | thiếu: khớp (khớp/quyết định) | thừa: khớp |
|---|---:|---|---|
| random_cell | 200 | 0.60 (9/15) | 0.80 (12/15) |
| random_cell | 201 | 0.73 (11/15) | 0.87 (13/15) |
| random_cell | 202 | 0.60 (9/15) | 0.86 (12/14, hòa 1) |
| random_cell | 203 | 0.73 (11/15) | 0.87 (13/15) |
| random_cell | 204 | 0.73 (11/15) | 0.87 (13/15) |
| random_cell | 205 | 0.60 (9/15) | 0.87 (13/15) |
| random_cell | 206 | 0.67 (10/15) | 0.87 (13/15) |
| random_cell | 207 | 0.73 (11/15) | 0.80 (12/15) |
| random_cell | 208 | 0.67 (8/12, hòa 3) | 0.87 (13/15) |
| random_cell | 209 | 0.62 (8/13, hòa 2) | 0.87 (13/15) |
- U1 thiếu (D1 cao ↔ MAE ô đáp án > 0): random_cell: TB 0.67, min 0.60 (10 ca) → **không đạt**.
- U1 thừa (D1 thấp ↔ MAE ô đáp án = 0): random_cell: TB 0.85, min 0.80 (10 ca) → **đạt**.

| Mask | Seed | khớp \|D2 − D2t\| | chọn → regret | khớp D2 thô | regret thô | khớp, MAE toàn ô model | regret, MAE toàn ô model |
|---|---:|---|---|---|---:|---|---:|
| random_cell | 200 | 0.93 (14/15) | same_day_mean → 0.0000 | 0.93 (14/15) | 0.0000 | 0.93 (14/15) | 0.0000 |
| random_cell | 201 | 0.87 (13/15) | same_day_mean → 0.0000 | 0.87 (13/15) | 0.0000 | 0.87 (13/15) | 0.0000 |
| random_cell | 202 | 0.93 (14/15) | same_day_mean → 0.0000 | 0.93 (14/15) | 0.0000 | 0.93 (14/15) | 0.0000 |
| random_cell | 203 | 0.80 (12/15) | same_day_mean → 0.0000 | 0.80 (12/15) | 0.0000 | 0.80 (12/15) | 0.0000 |
| random_cell | 204 | 0.73 (11/15) | same_day_mean → 0.0000 | 0.73 (11/15) | 0.0000 | 0.73 (11/15) | 0.0000 |
| random_cell | 205 | 0.93 (13/14, hòa 1) | same_day_mean → 0.0000 | 0.93 (13/14, hòa 1) | 0.0000 | 0.93 (13/14, hòa 1) | 0.0000 |
| random_cell | 206 | 0.87 (13/15) | same_day_mean → 0.0000 | 0.87 (13/15) | 0.0000 | 0.87 (13/15) | 0.0000 |
| random_cell | 207 | 0.93 (13/14, hòa 1) | same_day_mean → 0.0000 | 0.93 (13/14, hòa 1) | 0.0000 | 0.93 (13/14, hòa 1) | 0.0000 |
| random_cell | 208 | 0.87 (13/15) | same_day_mean → 0.0000 | 0.87 (13/15) | 0.0000 | 0.87 (13/15) | 0.0000 |
| random_cell | 209 | 0.87 (13/15) | same_day_mean → 0.0000 | 0.87 (13/15) | 0.0000 | 0.87 (13/15) | 0.0000 |
- U2: random_cell: TB 0.87, min 0.73 (10 ca); regret ≤ 0.005 ở 10/10 ca → **đạt**.

| Mask | Seed | ô dương bị cờ | tỷ lệ đáp án 0 khi bị cờ | ô dương không cờ | khi không cờ | chênh |
|---|---:|---:|---:|---:|---:|---:|
| random_cell | 200 | 1274 | 0.998 | 6522 | 0.274 | +0.723 |
| random_cell | 201 | 1155 | 1.000 | 6662 | 0.277 | +0.723 |
| random_cell | 202 | 1176 | 1.000 | 6549 | 0.282 | +0.718 |
| random_cell | 203 | 1299 | 0.995 | 6522 | 0.260 | +0.735 |
| random_cell | 204 | 1163 | 0.997 | 6600 | 0.276 | +0.721 |
| random_cell | 205 | 1463 | 0.996 | 6508 | 0.256 | +0.740 |
| random_cell | 206 | 1189 | 0.997 | 6784 | 0.269 | +0.728 |
| random_cell | 207 | 1195 | 0.997 | 6716 | 0.284 | +0.713 |
| random_cell | 208 | 1104 | 0.995 | 6798 | 0.294 | +0.700 |
| random_cell | 209 | 1236 | 0.999 | 6610 | 0.281 | +0.718 |
- U3: chênh ≥ 0.2 ở 10/10 ca → **đạt**.

Số ca diagnostic xếp đối thủ giả kém hơn **mọi** model thật (trên số ca có giá trị), R-local.

| Đối thủ giả | Mask | \|D1 − D1_truth\| | \|D1\| thô | \|D2 − D2_truth\| | D2 thô | D3: ô cờ / ô E | *tham chiếu: MAE Tầng 1, toàn ô model* |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| mean_fill | random_cell | 4/10 | 4/10 | 10/10 | 10/10 | 0/10 | 10/10 |
| zero_fill | random_cell | 10/10 | 0/10 | NA | NA | 0/10 | 10/10 |
| marginal_sampler | random_cell | 0/10 | 0/10 | 0/5 | 0/5 | 0/10 | 9/10 |

Giá trị trung bình qua 5 seed (R-local, trên target chung của model thật):

| Model | Mask | \|D1 − D1_truth\| | \|D1\| thô | \|D2 − D2_truth\| | D2 thô | D3: ô cờ / ô E | *tham chiếu: MAE Tầng 1, toàn ô model* |
|---|---|---:|---:|---:|---:|---:|---:|
| hron | random_cell | 0.0075 ± 0.0049 | 0.0748 ± 0.0071 | 0.0708 ± 0.0059 | 0.1046 ± 0.0070 | 0.0250 ± 0.0030 | 0.0893 ± 0.0042 |
| hron_cap | random_cell | 0.0505 ± 0.0104 | 0.0675 ± 0.0044 | 0.0754 ± 0.0087 | 0.1092 ± 0.0106 | 0.0186 ± 0.0022 | 0.0873 ± 0.0038 |
| tsagris | random_cell | 0.1524 ± 0.0088 | 0.2055 ± 0.0091 | 0.0857 ± 0.0071 | 0.1196 ± 0.0048 | 0.0826 ± 0.0071 | 0.0986 ± 0.0029 |
| micoda | random_cell | 0.3389 ± 0.0106 | 0.3873 ± 0.0088 | 0.1337 ± 0.0079 | 0.1676 ± 0.0058 | 0.1808 ± 0.0171 | 0.1293 ± 0.0044 |
| micoda_other | random_cell | 0.1452 ± 0.0128 | 0.1940 ± 0.0112 | 0.1688 ± 0.0078 | 0.2026 ± 0.0050 | 0.1325 ± 0.0139 | 0.1062 ± 0.0048 |
| same_day_mean | random_cell | 0.0462 ± 0.0089 | 0.0949 ± 0.0070 | 0.0374 ± 0.0058 | 0.0712 ± 0.0039 | 0.0011 ± 0.0006 | 0.0298 ± 0.0017 |
| mean_fill | random_cell | 0.3362 ± 0.0134 | 0.3847 ± 0.0104 | 0.2106 ± 0.0070 | 0.2444 ± 0.0043 | 0.1485 ± 0.0102 | 0.1372 ± 0.0025 |
| zero_fill | random_cell | 0.3744 ± 0.0076 | 0.3261 ± 0.0038 | NA | NA | 0.0000 ± 0.0000 | 0.1740 ± 0.0036 |
| marginal_sampler | random_cell | 0.1847 ± 0.0059 | 0.1364 ± 0.0098 | 0.0735 ± 0.0075 (n = 5) | 0.1096 ± 0.0067 (n = 5) | 0.0450 ± 0.0050 | 0.1378 ± 0.0059 |

**D1 với ngưỡng zero c/N < 0.001 (R-local)**

| Mask | Seed | thiếu: khớp (khớp/quyết định) | thừa: khớp |
|---|---:|---|---|
| random_cell | 200 | 0.60 (9/15) | 0.80 (12/15) |
| random_cell | 201 | 0.67 (10/15) | 0.80 (12/15) |
| random_cell | 202 | 0.60 (9/15) | 0.86 (12/14, hòa 1) |
| random_cell | 203 | 0.67 (10/15) | 0.80 (12/15) |
| random_cell | 204 | 0.67 (10/15) | 0.80 (12/15) |
| random_cell | 205 | 0.53 (8/15) | 0.80 (12/15) |
| random_cell | 206 | 0.60 (9/15) | 0.80 (12/15) |
| random_cell | 207 | 0.73 (11/15) | 0.80 (12/15) |
| random_cell | 208 | 0.58 (7/12, hòa 3) | 0.80 (12/15) |
| random_cell | 209 | 0.54 (7/13, hòa 2) | 0.80 (12/15) |
- U1 thiếu (D1 cao ↔ MAE ô đáp án > 0): random_cell: TB 0.62, min 0.53 (10 ca) → **không đạt**.
- U1 thừa (D1 thấp ↔ MAE ô đáp án = 0): random_cell: TB 0.81, min 0.80 (10 ca) → **đạt**.

**D1 với R-pooled**

| Mask | Seed | thiếu: khớp (khớp/quyết định) | thừa: khớp |
|---|---:|---|---|
| random_cell | 200 | 0.60 (9/15) | 0.80 (12/15) |
| random_cell | 201 | 0.73 (11/15) | 0.87 (13/15) |
| random_cell | 202 | 0.57 (8/14, hòa 1) | 0.86 (12/14, hòa 1) |
| random_cell | 203 | 0.73 (11/15) | 0.87 (13/15) |
| random_cell | 204 | 0.67 (10/15) | 0.80 (12/15) |
| random_cell | 205 | 0.53 (8/15) | 0.80 (12/15) |
| random_cell | 206 | 0.67 (10/15) | 0.87 (13/15) |
| random_cell | 207 | 0.73 (11/15) | 0.80 (12/15) |
| random_cell | 208 | 0.57 (8/14, hòa 1) | 0.87 (13/15) |
| random_cell | 209 | 0.69 (9/13, hòa 2) | 0.87 (13/15) |
- U1 thiếu (D1 cao ↔ MAE ô đáp án > 0): random_cell: TB 0.65, min 0.53 (10 ca) → **không đạt**.
- U1 thừa (D1 thấp ↔ MAE ô đáp án = 0): random_cell: TB 0.84, min 0.80 (10 ca) → **đạt**.

**D2 với R-pooled**

| Mask | Seed | khớp \|D2 − D2t\| | chọn → regret | khớp D2 thô | regret thô | khớp, MAE toàn ô model | regret, MAE toàn ô model |
|---|---:|---|---|---|---:|---|---:|
| random_cell | 200 | 0.87 (13/15) | same_day_mean → 0.0000 | 0.87 (13/15) | 0.0000 | 0.87 (13/15) | 0.0000 |
| random_cell | 201 | 0.87 (13/15) | same_day_mean → 0.0000 | 0.87 (13/15) | 0.0000 | 0.87 (13/15) | 0.0000 |
| random_cell | 202 | 0.93 (14/15) | same_day_mean → 0.0000 | 0.93 (14/15) | 0.0000 | 0.93 (14/15) | 0.0000 |
| random_cell | 203 | 0.80 (12/15) | same_day_mean → 0.0000 | 0.80 (12/15) | 0.0000 | 0.80 (12/15) | 0.0000 |
| random_cell | 204 | 0.80 (12/15) | same_day_mean → 0.0000 | 0.80 (12/15) | 0.0000 | 0.80 (12/15) | 0.0000 |
| random_cell | 205 | 0.93 (13/14, hòa 1) | same_day_mean → 0.0000 | 0.93 (13/14, hòa 1) | 0.0000 | 0.93 (13/14, hòa 1) | 0.0000 |
| random_cell | 206 | 0.87 (13/15) | same_day_mean → 0.0000 | 0.87 (13/15) | 0.0000 | 0.87 (13/15) | 0.0000 |
| random_cell | 207 | 0.93 (13/14, hòa 1) | same_day_mean → 0.0000 | 0.93 (13/14, hòa 1) | 0.0000 | 0.93 (13/14, hòa 1) | 0.0000 |
| random_cell | 208 | 0.93 (14/15) | same_day_mean → 0.0000 | 0.93 (14/15) | 0.0000 | 0.93 (14/15) | 0.0000 |
| random_cell | 209 | 0.87 (13/15) | same_day_mean → 0.0000 | 0.87 (13/15) | 0.0000 | 0.87 (13/15) | 0.0000 |
- U2: random_cell: TB 0.88, min 0.80 (10 ca); regret ≤ 0.005 ở 10/10 ca → **đạt**.

## 8. So với exploratory

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

