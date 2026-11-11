# Step 4: so sánh theo quy tắc phần dư (COVID đầy đủ, che 5%, seed 0–4) — **exploratory**

Sinh bởi `residual.py` (chấm điểm bằng `tier01`); không sửa tay. Metric chính: MAE tỷ lệ, cell-weighted, nhóm ô model. mean ± sd (ddof=1) qua seed, không phải khoảng tin cậy.

## whole_feature (chính)

### Composition được chia (xác minh p)

| Biến thể | Quy tắc | Tổng dòng của composition trước khi đóng (min / median / max) | Dòng bị co/chuẩn hóa / dòng có ô thiếu |
|---|---|---|---|
| hron_cap | cap | 0.0002775 / 1 / 3.474e+05 | 37546 / 57570 |
| hron_rescale | rescale | 0.0002775 / 1 / 3.474e+05 | 57570 / 57570 |
| tsagris_cap | cap | 1 / 1 / 1 | 1 / 57570 |
| tsagris_rescale | rescale | 1 / 1 / 1 | 18933 / 57570 |
| micoda_other_cap | cap | 1 / 1 / 1 | 36496 / 57570 |
| micoda_other_rescale | rescale | 1 / 1 / 1 | 57570 / 57570 |
| hron_prop_check | prop_other | 0.0002775 / 1 / 3.474e+05 | - |
| tsagris_prop_check | prop_other | 1 / 1 / 1 | - |
| micoda_other_prop_check | prop_other | 1 / 1 / 1 | - |

### Consistency check

- hron_prop_check: 28309 / 28309 ô che giống hệt, chênh lớn nhất 0 count; CSV native giống từng byte (chênh lớn nhất 0)
- tsagris_prop_check: 28309 / 28309 ô che giống hệt, chênh lớn nhất 0 count; CSV native giống từng byte (chênh lớn nhất 0)
- micoda_other_prop_check: 22923 / 28309 ô che giống hệt, chênh lớn nhất 7.28e-12 count; CSV native khác (chênh lớn nhất 2.84e-14)
- tsagris_cap_vs_prop: 24638 / 28309 ô che giống hệt, chênh lớn nhất 2.91e-11 count

### Bảng 1: quy tắc `prop_other`

Tập seed chung (mọi model completed, coverage = 1): 0, 1, 2, 3, 4.

Sắp theo MAE tỷ lệ (nhóm ô model) trung bình trên tập seed chung. mean ± sd (ddof=1) qua seed; sd không phải khoảng tin cậy. **exploratory**.

| Model | MAE tỷ lệ | MSE tỷ lệ | MAE count | MAE tỷ lệ series-macro | MAE tỷ lệ, đáp án = 0 | MAE tỷ lệ, đáp án > 0 | MAE tỷ lệ, ô model đáp án = 0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| hron_prop | 0.0875 ± 0.0105 | 0.0660 ± 0.0092 | 74.6 ± 113.7 | 0.1003 ± 0.0135 | 0.0103 ± 0.0027 | 0.1978 ± 0.0195 | 0.0294 ± 0.0065 |
| micoda_other_prop | 0.1075 ± 0.0093 | 0.0684 ± 0.0078 | 188.7 ± 205.5 | 0.1175 ± 0.0137 | 0.0194 ± 0.0036 | 0.2056 ± 0.0195 | 0.0554 ± 0.0095 |
| tsagris_prop | 0.1079 ± 0.0209 | 0.0521 ± 0.0095 | 33.2 ± 14.7 | 0.1207 ± 0.0207 | 0.0260 ± 0.0075 | 0.1719 ± 0.0458 | 0.0745 ± 0.0214 |

### Bảng 1: quy tắc `cap`

Tập seed chung (mọi model completed, coverage = 1): 0, 1, 2, 3, 4.

Sắp theo MAE tỷ lệ (nhóm ô model) trung bình trên tập seed chung. mean ± sd (ddof=1) qua seed; sd không phải khoảng tin cậy. **exploratory**.

| Model | MAE tỷ lệ | MSE tỷ lệ | MAE count | MAE tỷ lệ series-macro | MAE tỷ lệ, đáp án = 0 | MAE tỷ lệ, đáp án > 0 | MAE tỷ lệ, ô model đáp án = 0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| sdm_cap | 0.0351 ± 0.0066 | 0.0142 ± 0.0036 | 29.4 ± 26.8 | 0.0495 ± 0.0095 | 0.0033 ± 0.0014 | 0.0844 ± 0.0181 | 0.0094 ± 0.0040 |
| hron_cap | 0.0869 ± 0.0095 | 0.0658 ± 0.0087 | 81.5 ± 115.2 | 0.0993 ± 0.0128 | 0.0091 ± 0.0028 | 0.2029 ± 0.0175 | 0.0258 ± 0.0070 |
| micoda_other_cap | 0.1075 ± 0.0093 | 0.0684 ± 0.0078 | 188.7 ± 205.5 | 0.1175 ± 0.0137 | 0.0194 ± 0.0036 | 0.2056 ± 0.0195 | 0.0554 ± 0.0095 |
| tsagris_cap | 0.1079 ± 0.0209 | 0.0521 ± 0.0095 | 33.2 ± 14.7 | 0.1207 ± 0.0207 | 0.0260 ± 0.0075 | 0.1719 ± 0.0458 | 0.0745 ± 0.0214 |
| mean_fill_cap | 0.1354 ± 0.0145 | 0.0704 ± 0.0084 | 218.2 ± 186.8 | 0.1324 ± 0.0162 | 0.0178 ± 0.0037 | 0.2950 ± 0.0154 | 0.0508 ± 0.0092 |

### Bảng 1: quy tắc `rescale`

Tập seed chung (mọi model completed, coverage = 1): 0, 1, 2, 3, 4.

Sắp theo MAE tỷ lệ (nhóm ô model) trung bình trên tập seed chung. mean ± sd (ddof=1) qua seed; sd không phải khoảng tin cậy. **exploratory**.

| Model | MAE tỷ lệ | MSE tỷ lệ | MAE count | MAE tỷ lệ series-macro | MAE tỷ lệ, đáp án = 0 | MAE tỷ lệ, đáp án > 0 | MAE tỷ lệ, ô model đáp án = 0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| sdm_rescale | 0.0877 ± 0.0169 | 0.0497 ± 0.0119 | 83.8 ± 56.3 | 0.1047 ± 0.0183 | 0.0307 ± 0.0043 | 0.0874 ± 0.0253 | 0.0883 ± 0.0149 |
| micoda_rescale | 0.1406 ± 0.0099 | 0.0823 ± 0.0094 | 236.2 ± 188.5 | 0.1476 ± 0.0161 | 0.0411 ± 0.0051 | 0.1834 ± 0.0261 | 0.1177 ± 0.0116 |
| hron_rescale | 0.1428 ± 0.0100 | 0.0842 ± 0.0099 | 157.0 ± 154.3 | 0.1548 ± 0.0133 | 0.0451 ± 0.0058 | 0.1694 ± 0.0107 | 0.1290 ± 0.0134 |
| tsagris_rescale | 0.1615 ± 0.0124 | 0.1045 ± 0.0130 | 86.0 ± 49.2 | 0.1792 ± 0.0172 | 0.0626 ± 0.0097 | 0.1304 ± 0.0118 | 0.1788 ± 0.0231 |
| micoda_other_rescale | 0.1637 ± 0.0121 | 0.0845 ± 0.0110 | 165.4 ± 103.7 | 0.1727 ± 0.0143 | 0.0629 ± 0.0087 | 0.1330 ± 0.0191 | 0.1799 ± 0.0203 |

### Bảng 2: ghép cặp trong cùng quy tắc

| A vs B | MAE tỷ lệ | MSE tỷ lệ | MAE count | MAE tỷ lệ series-macro | MAE tỷ lệ, đáp án = 0 | MAE tỷ lệ, đáp án > 0 | MAE tỷ lệ, ô model đáp án = 0 |
|---|---|---|---|---|---|---|---|
| sdm_cap vs hron_cap | -0.05184 ± 0.014 (5/5) | -0.05154 ± 0.011 (5/5) | -52.1 ± 90 (5/5) | -0.04987 ± 0.016 (5/5) | -0.005813 ± 0.0028 (5/5) | -0.1185 ± 0.029 (5/5) | -0.01643 ± 0.0072 (5/5) |
| sdm_cap vs tsagris_cap | -0.07283 ± 0.025 (5/5) | -0.03784 ± 0.012 (5/5) | -3.827 ± 12 (4/5) | -0.07121 ± 0.023 (5/5) | -0.02275 ± 0.0075 (5/5) | -0.08752 ± 0.059 (5/5) | -0.06513 ± 0.021 (5/5) |
| hron_cap vs tsagris_cap | -0.02099 ± 0.014 (5/5) | +0.0137 ± 0.0054 (0/5) | +48.28 ± 1e+02 (2/5) | -0.02133 ± 0.0084 (5/5) | -0.01693 ± 0.0065 (5/5) | +0.03098 ± 0.032 (1/5) | -0.0487 ± 0.019 (5/5) |
| micoda_other_cap vs hron_cap | +0.0206 ± 0.0042 (0/5) | +0.002656 ± 0.003 (0/5) | +107.2 ± 95 (0/5) | +0.01812 ± 0.0035 (0/5) | +0.01027 ± 0.0038 (0/5) | +0.002676 ± 0.023 (3/5) | +0.02955 ± 0.011 (0/5) |
| sdm_rescale vs hron_rescale | -0.05511 ± 0.017 (5/5) | -0.0345 ± 0.017 (5/5) | -73.12 ± 1.1e+02 (5/5) | -0.05013 ± 0.02 (5/5) | -0.01441 ± 0.0076 (5/5) | -0.08204 ± 0.018 (5/5) | -0.04063 ± 0.019 (5/5) |
| sdm_rescale vs tsagris_rescale | -0.0738 ± 0.024 (5/5) | -0.05481 ± 0.018 (5/5) | -2.198 ± 9.4 (2/5) | -0.07457 ± 0.021 (5/5) | -0.03189 ± 0.012 (5/5) | -0.04301 ± 0.016 (5/5) | -0.09052 ± 0.03 (5/5) |
| hron_rescale vs tsagris_rescale | -0.01869 ± 0.0081 (5/5) | -0.02031 ± 0.0035 (5/5) | +70.93 ± 1.1e+02 (0/5) | -0.02443 ± 0.0072 (5/5) | -0.01748 ± 0.0056 (5/5) | +0.03903 ± 0.012 (0/5) | -0.04988 ± 0.015 (5/5) |
| micoda_other_rescale vs hron_rescale | +0.02087 ± 0.0055 (0/5) | +0.0003432 ± 0.0037 (2/5) | +8.395 ± 60 (1/5) | +0.01793 ± 0.0038 (0/5) | +0.0178 ± 0.0044 (0/5) | -0.03645 ± 0.024 (5/5) | +0.05088 ± 0.012 (0/5) |

### Bảng 3: ghép cặp trong cùng model

| A vs B | MAE tỷ lệ | MSE tỷ lệ | MAE count | MAE tỷ lệ series-macro | MAE tỷ lệ, đáp án = 0 | MAE tỷ lệ, đáp án > 0 | MAE tỷ lệ, ô model đáp án = 0 |
|---|---|---|---|---|---|---|---|
| hron_cap vs hron_prop | -0.0006277 ± 0.0019 (4/5) | -0.0002593 ± 0.0014 (3/5) | +6.943 ± 8.3 (1/5) | -0.0009751 ± 0.0022 (4/5) | -0.001232 ± 0.00016 (5/5) | +0.005113 ± 0.0057 (1/5) | -0.003542 ± 0.00056 (5/5) |
| hron_rescale vs hron_prop | +0.05527 ± 0.013 (0/5) | +0.01812 ± 0.0056 (0/5) | +82.39 ± 60 (0/5) | +0.0545 ± 0.0083 (0/5) | +0.03478 ± 0.0036 (0/5) | -0.02837 ± 0.029 (5/5) | +0.09961 ± 0.0093 (0/5) |
| tsagris_cap vs tsagris_prop | +0 ± 9.8e-18 (1/5, hòa 3) | -2.776e-18 ± 6.2e-18 (1/5, hòa 4) | +4.263e-15 ± 1.1e-14 (1/5, hòa 2) | +0 ± 0 (0/5, hòa 5) | +2.082e-18 ± 3.1e-18 (0/5, hòa 3) | -1.11e-17 ± 1.5e-17 (2/5, hòa 3) | +0 ± 0 (0/5, hòa 5) |
| tsagris_rescale vs tsagris_prop | +0.05361 ± 0.013 (0/5) | +0.0524 ± 0.0077 (0/5) | +52.8 ± 37 (0/5) | +0.05857 ± 0.013 (0/5) | +0.03655 ± 0.0071 (0/5) | -0.04154 ± 0.052 (5/5) | +0.1043 ± 0.017 (0/5) |
| micoda_other_cap vs micoda_other_prop | +9.429e-07 ± 4.2e-07 (0/5) | -4.005e-07 ± 2.4e-07 (5/5) | +0.002264 ± 0.0028 (1/5) | +3.971e-07 ± 2.5e-07 (0/5) | -5.181e-07 ± 2e-07 (5/5) | +5.585e-06 ± 2.3e-06 (0/5) | -1.491e-06 ± 5.9e-07 (5/5) |
| micoda_other_rescale vs micoda_other_prop | +0.05616 ± 0.0071 (0/5) | +0.01607 ± 0.0074 (0/5) | -23.33 ± 1.1e+02 (2/5) | +0.05529 ± 0.0076 (0/5) | +0.04353 ± 0.0058 (0/5) | -0.07261 ± 0.015 (5/5) | +0.1245 ± 0.013 (0/5) |

### Bảng 4: tách khoảng cách (MAE tỷ lệ; mô tả, không phải phân rã chặt)

| Đường tách | Thành phần | Δ mean ± sd | âm / 5 seed | % khoảng cách |
|---|---|---|---:|---:|
| — | khoảng cách sdm_cap − hron_prop | -0.0525 ± 0.0158 | 5 | 100% |
| cap | thuật toán @cap | -0.0518 ± 0.0144 | 5 | +99% |
| cap | quy tắc hron (cap − prop_other) | -0.0006 ± 0.0019 | 4 | +1% |
| rescale | quy tắc same_day_mean (cap − rescale) | -0.0526 ± 0.0114 | 5 | +100% |
| rescale | thuật toán @rescale | -0.0551 ± 0.0175 | 5 | +105% |
| rescale | quy tắc hron (rescale − prop_other) | +0.0553 ± 0.0127 | 0 | -105% |

| Đường tách | Thành phần | Δ mean ± sd | âm / 5 seed | % khoảng cách |
|---|---|---|---:|---:|
| — | khoảng cách sdm_cap − tsagris_prop | -0.0728 ± 0.0251 | 5 | 100% |
| cap | thuật toán @cap | -0.0728 ± 0.0251 | 5 | +100% |
| cap | quy tắc tsagris (cap − prop_other) | +0.0000 ± 0.0000 | 1 | -0% |
| rescale | quy tắc same_day_mean (cap − rescale) | -0.0526 ± 0.0114 | 5 | +72% |
| rescale | thuật toán @rescale | -0.0738 ± 0.0240 | 5 | +101% |
| rescale | quy tắc tsagris (rescale − prop_other) | +0.0536 ± 0.0126 | 0 | -74% |

### Bảng 5: theo nhóm đáp án

| Biến thể | Model | Quy tắc | MAE tỷ lệ, đáp án = 0 | MAE tỷ lệ, đáp án > 0 | MAE tỷ lệ, ô model đáp án = 0 |
|---|---|---|---:|---:|---:|
| hron_prop | hron | prop_other | 0.0103 ± 0.0027 | 0.1978 ± 0.0195 | 0.0294 ± 0.0065 |
| hron_cap | hron | cap | 0.0091 ± 0.0028 | 0.2029 ± 0.0175 | 0.0258 ± 0.0070 |
| hron_rescale | hron | rescale | 0.0451 ± 0.0058 | 0.1694 ± 0.0107 | 0.1290 ± 0.0134 |
| tsagris_prop | tsagris | prop_other | 0.0260 ± 0.0075 | 0.1719 ± 0.0458 | 0.0745 ± 0.0214 |
| tsagris_cap | tsagris | cap | 0.0260 ± 0.0075 | 0.1719 ± 0.0458 | 0.0745 ± 0.0214 |
| tsagris_rescale | tsagris | rescale | 0.0626 ± 0.0097 | 0.1304 ± 0.0118 | 0.1788 ± 0.0231 |
| micoda_other_prop | micoda_other | prop_other | 0.0194 ± 0.0036 | 0.2056 ± 0.0195 | 0.0554 ± 0.0095 |
| micoda_other_cap | micoda_other | cap | 0.0194 ± 0.0036 | 0.2056 ± 0.0195 | 0.0554 ± 0.0095 |
| micoda_other_rescale | micoda_other | rescale | 0.0629 ± 0.0087 | 0.1330 ± 0.0191 | 0.1799 ± 0.0203 |
| micoda_rescale | micoda (A17) | rescale | 0.0411 ± 0.0051 | 0.1834 ± 0.0261 | 0.1177 ± 0.0116 |
| sdm_cap | same_day_mean | cap | 0.0033 ± 0.0014 | 0.0844 ± 0.0181 | 0.0094 ± 0.0040 |
| sdm_rescale | same_day_mean | rescale | 0.0307 ± 0.0043 | 0.0874 ± 0.0253 | 0.0883 ± 0.0149 |
| mean_fill_cap | mean_fill | cap | 0.0178 ± 0.0037 | 0.2950 ± 0.0154 | 0.0508 ± 0.0092 |

### Bảng 6: Other ngầm

Dòng đầy đủ trước khi che: 516, thuộc 5 nước. Dòng dùng / số nước theo seed: seed 0: 407 / 4, seed 1: 436 / 4, seed 2: 218 / 2, seed 3: 407 / 4, seed 4: 327 / 3

| Biến thể | Quy tắc | MAE tỷ lệ Other ngầm | Sai số có dấu (ngầm − thật) |
|---|---|---:|---:|
| hron_prop | prop_other | 0.0123 ± 0.0102 | 0.0038 ± 0.0122 |
| hron_cap | cap | 0.0131 ± 0.0112 | 0.0052 ± 0.0114 |
| hron_rescale | rescale | 0.1349 ± 0.0262 | -0.1349 ± 0.0262 |
| tsagris_prop | prop_other | 0.0164 ± 0.0149 | -0.0127 ± 0.0137 |
| tsagris_cap | cap | 0.0164 ± 0.0149 | -0.0127 ± 0.0137 |
| tsagris_rescale | rescale | 0.0875 ± 0.0296 | -0.0873 ± 0.0297 |
| micoda_other_prop | prop_other | 0.0348 ± 0.0323 | 0.0101 ± 0.0208 |
| micoda_other_cap | cap | 0.0348 ± 0.0323 | 0.0101 ± 0.0208 |
| micoda_other_rescale | rescale | 0.1349 ± 0.0262 | -0.1349 ± 0.0262 |
| micoda_rescale | rescale | 0.1349 ± 0.0262 | -0.1349 ± 0.0262 |
| sdm_cap | cap | 0.0060 ± 0.0050 | 0.0007 ± 0.0031 |
| sdm_rescale | rescale | 0.0597 ± 0.0449 | -0.0597 ± 0.0449 |
| mean_fill_cap | cap | 0.0532 ± 0.0665 | 0.0289 ± 0.0438 |

### Bảng 7: series-macro so với cell-weighted

**prop_other**
Seed chung: 0, 1, 2, 3, 4. Cell-weighted: hron_prop < micoda_other_prop < tsagris_prop.
Series-macro: hron_prop < micoda_other_prop < tsagris_prop.
Thứ hạng **không đổi**.

**cap**
Seed chung: 0, 1, 2, 3, 4. Cell-weighted: sdm_cap < hron_cap < micoda_other_cap < tsagris_cap < mean_fill_cap.
Series-macro: sdm_cap < hron_cap < micoda_other_cap < tsagris_cap < mean_fill_cap.
Thứ hạng **không đổi**.

**rescale**
Seed chung: 0, 1, 2, 3, 4. Cell-weighted: sdm_rescale < micoda_rescale < hron_rescale < tsagris_rescale < micoda_other_rescale.
Series-macro: sdm_rescale < micoda_rescale < hron_rescale < micoda_other_rescale < tsagris_rescale.
- Đổi chỗ tsagris_rescale / micoda_other_rescale: cell-weighted tsagris_rescale thắng 4/5 (Δ -0.00218); series-macro tsagris_rescale thắng 1/5 (Δ +0.00650).

**mọi biến thể**
Seed chung: 0, 1, 2, 3, 4. Cell-weighted: sdm_cap < hron_cap < hron_prop < sdm_rescale < micoda_other_prop < micoda_other_cap < tsagris_cap < tsagris_prop < mean_fill_cap < micoda_rescale < hron_rescale < tsagris_rescale < micoda_other_rescale.
Series-macro: sdm_cap < hron_cap < hron_prop < sdm_rescale < micoda_other_prop < micoda_other_cap < tsagris_cap < tsagris_prop < mean_fill_cap < micoda_rescale < hron_rescale < micoda_other_rescale < tsagris_rescale.
- Đổi chỗ tsagris_rescale / micoda_other_rescale: cell-weighted tsagris_rescale thắng 4/5 (Δ -0.00218); series-macro tsagris_rescale thắng 1/5 (Δ +0.00650).

## random_cell (sensitivity)

### Composition được chia (xác minh p)

| Biến thể | Quy tắc | Tổng dòng của composition trước khi đóng (min / median / max) | Dòng bị co/chuẩn hóa / dòng có ô thiếu |
|---|---|---|---|
| hron_cap | cap | 2.335e-06 / 1 / 3.257e+05 | 37025 / 57292 |
| hron_rescale | rescale | 2.335e-06 / 1 / 3.257e+05 | 57292 / 57292 |
| tsagris_cap | cap | 1 / 1 / 1 | 18 / 57292 |
| tsagris_rescale | rescale | 1 / 1 / 1 | 19394 / 57292 |
| micoda_other_cap | cap | 1 / 1 / 1 | 35928 / 57292 |
| micoda_other_rescale | rescale | 1 / 1 / 1 | 57292 / 57292 |
| hron_prop_check | prop_other | 2.335e-06 / 1 / 3.257e+05 | - |
| tsagris_prop_check | prop_other | 1 / 1 / 1 | - |
| micoda_other_prop_check | prop_other | 1 / 1 / 1 | - |

### Consistency check

- hron_prop_check: 27936 / 27936 ô che giống hệt, chênh lớn nhất 0 count; CSV native giống từng byte (chênh lớn nhất 0)
- tsagris_prop_check: 27936 / 27936 ô che giống hệt, chênh lớn nhất 0 count; CSV native giống từng byte (chênh lớn nhất 0)
- micoda_other_prop_check: 22155 / 27936 ô che giống hệt, chênh lớn nhất 2.91e-11 count; CSV native khác (chênh lớn nhất 2.84e-14)
- tsagris_cap_vs_prop: 24192 / 27936 ô che giống hệt, chênh lớn nhất 2.91e-11 count

### Bảng 1: quy tắc `prop_other`

Tập seed chung (mọi model completed, coverage = 1): 0, 1, 2, 3, 4.

Sắp theo MAE tỷ lệ (nhóm ô model) trung bình trên tập seed chung. mean ± sd (ddof=1) qua seed; sd không phải khoảng tin cậy. **exploratory**.

| Model | MAE tỷ lệ | MSE tỷ lệ | MAE count | MAE tỷ lệ series-macro | MAE tỷ lệ, đáp án = 0 | MAE tỷ lệ, đáp án > 0 | MAE tỷ lệ, ô model đáp án = 0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| hron_prop | 0.0892 ± 0.0028 | 0.0613 ± 0.0027 | 22.9 ± 4.0 | 0.0909 ± 0.0057 | 0.0123 ± 0.0008 | 0.1943 ± 0.0061 | 0.0315 ± 0.0024 |
| tsagris_prop | 0.0999 ± 0.0024 | 0.0482 ± 0.0017 | 26.6 ± 3.6 | 0.1059 ± 0.0047 | 0.0305 ± 0.0025 | 0.1395 ± 0.0149 | 0.0781 ± 0.0071 |
| micoda_other_prop | 0.1078 ± 0.0008 | 0.0695 ± 0.0013 | 109.6 ± 37.3 | 0.1075 ± 0.0049 | 0.0256 ± 0.0006 | 0.1851 ± 0.0025 | 0.0653 ± 0.0017 |

### Bảng 1: quy tắc `cap`

Tập seed chung (mọi model completed, coverage = 1): 0, 1, 2, 3, 4.

Sắp theo MAE tỷ lệ (nhóm ô model) trung bình trên tập seed chung. mean ± sd (ddof=1) qua seed; sd không phải khoảng tin cậy. **exploratory**.

| Model | MAE tỷ lệ | MSE tỷ lệ | MAE count | MAE tỷ lệ series-macro | MAE tỷ lệ, đáp án = 0 | MAE tỷ lệ, đáp án > 0 | MAE tỷ lệ, ô model đáp án = 0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| sdm_cap | 0.0310 ± 0.0027 | 0.0110 ± 0.0015 | 19.0 ± 4.8 | 0.0376 ± 0.0041 | 0.0036 ± 0.0002 | 0.0704 ± 0.0077 | 0.0093 ± 0.0005 |
| hron_cap | 0.0879 ± 0.0025 | 0.0604 ± 0.0022 | 32.9 ± 15.0 | 0.0891 ± 0.0064 | 0.0113 ± 0.0008 | 0.1955 ± 0.0057 | 0.0288 ± 0.0024 |
| tsagris_cap | 0.0999 ± 0.0024 | 0.0482 ± 0.0017 | 26.6 ± 3.6 | 0.1059 ± 0.0047 | 0.0305 ± 0.0025 | 0.1395 ± 0.0149 | 0.0781 ± 0.0071 |
| micoda_other_cap | 0.1078 ± 0.0008 | 0.0695 ± 0.0013 | 109.6 ± 37.3 | 0.1075 ± 0.0049 | 0.0256 ± 0.0006 | 0.1851 ± 0.0025 | 0.0653 ± 0.0017 |
| mean_fill_cap | 0.1371 ± 0.0016 | 0.0692 ± 0.0007 | 152.4 ± 33.5 | 0.1277 ± 0.0041 | 0.0226 ± 0.0004 | 0.2817 ± 0.0045 | 0.0577 ± 0.0012 |

### Bảng 1: quy tắc `rescale`

Tập seed chung (mọi model completed, coverage = 1): 0, 1, 2, 3, 4.

Sắp theo MAE tỷ lệ (nhóm ô model) trung bình trên tập seed chung. mean ± sd (ddof=1) qua seed; sd không phải khoảng tin cậy. **exploratory**.

| Model | MAE tỷ lệ | MSE tỷ lệ | MAE count | MAE tỷ lệ series-macro | MAE tỷ lệ, đáp án = 0 | MAE tỷ lệ, đáp án > 0 | MAE tỷ lệ, ô model đáp án = 0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| sdm_rescale | 0.0773 ± 0.0019 | 0.0453 ± 0.0020 | 76.3 ± 12.5 | 0.0881 ± 0.0028 | 0.0295 ± 0.0013 | 0.0805 ± 0.0085 | 0.0755 ± 0.0045 |
| hron_rescale | 0.1305 ± 0.0036 | 0.0800 ± 0.0038 | 111.7 ± 11.2 | 0.1352 ± 0.0031 | 0.0482 ± 0.0014 | 0.1437 ± 0.0083 | 0.1232 ± 0.0052 |
| micoda_rescale | 0.1317 ± 0.0018 | 0.0803 ± 0.0024 | 167.7 ± 38.9 | 0.1312 ± 0.0031 | 0.0446 ± 0.0011 | 0.1641 ± 0.0041 | 0.1139 ± 0.0042 |
| tsagris_rescale | 0.1510 ± 0.0040 | 0.0975 ± 0.0039 | 89.3 ± 14.9 | 0.1618 ± 0.0068 | 0.0671 ± 0.0017 | 0.1137 ± 0.0109 | 0.1715 ± 0.0058 |
| micoda_other_rescale | 0.1567 ± 0.0020 | 0.0859 ± 0.0017 | 141.3 ± 18.5 | 0.1580 ± 0.0022 | 0.0693 ± 0.0016 | 0.1191 ± 0.0067 | 0.1772 ± 0.0056 |

### Bảng 2: ghép cặp trong cùng quy tắc

| A vs B | MAE tỷ lệ | MSE tỷ lệ | MAE count | MAE tỷ lệ series-macro | MAE tỷ lệ, đáp án = 0 | MAE tỷ lệ, đáp án > 0 | MAE tỷ lệ, ô model đáp án = 0 |
|---|---|---|---|---|---|---|---|
| sdm_cap vs hron_cap | -0.05696 ± 0.0044 (5/5) | -0.04937 ± 0.0033 (5/5) | -13.87 ± 14 (5/5) | -0.05155 ± 0.0056 (5/5) | -0.007608 ± 0.00099 (5/5) | -0.1251 ± 0.0088 (5/5) | -0.01947 ± 0.0027 (5/5) |
| sdm_cap vs tsagris_cap | -0.06894 ± 0.0015 (5/5) | -0.03718 ± 0.0015 (5/5) | -7.562 ± 3.3 (5/5) | -0.06833 ± 0.003 (5/5) | -0.02689 ± 0.0026 (5/5) | -0.06908 ± 0.012 (5/5) | -0.0688 ± 0.0072 (5/5) |
| hron_cap vs tsagris_cap | -0.01198 ± 0.0044 (5/5) | +0.01219 ± 0.0033 (0/5) | +6.311 ± 13 (2/5) | -0.01678 ± 0.0057 (5/5) | -0.01928 ± 0.0024 (5/5) | +0.05606 ± 0.012 (0/5) | -0.04933 ± 0.0064 (5/5) |
| micoda_other_cap vs hron_cap | +0.01989 ± 0.0031 (0/5) | +0.009089 ± 0.0026 (0/5) | +76.7 ± 36 (0/5) | +0.01837 ± 0.0022 (0/5) | +0.0143 ± 0.0013 (0/5) | -0.0104 ± 0.0044 (5/5) | +0.03654 ± 0.0032 (0/5) |
| sdm_rescale vs hron_rescale | -0.05325 ± 0.0021 (5/5) | -0.03473 ± 0.0031 (5/5) | -35.33 ± 4.9 (5/5) | -0.04707 ± 0.0022 (5/5) | -0.01869 ± 0.00078 (5/5) | -0.06321 ± 0.005 (5/5) | -0.04778 ± 0.002 (5/5) |
| sdm_rescale vs tsagris_rescale | -0.07373 ± 0.0033 (5/5) | -0.05221 ± 0.003 (5/5) | -12.95 ± 4.4 (5/5) | -0.07371 ± 0.0042 (5/5) | -0.03755 ± 0.0017 (5/5) | -0.03317 ± 0.0026 (5/5) | -0.09601 ± 0.0041 (5/5) |
| hron_rescale vs tsagris_rescale | -0.02048 ± 0.0048 (5/5) | -0.01748 ± 0.0046 (5/5) | +22.39 ± 6.4 (0/5) | -0.02665 ± 0.0048 (5/5) | -0.01887 ± 0.0018 (5/5) | +0.03003 ± 0.007 (0/5) | -0.04823 ± 0.0045 (5/5) |
| micoda_other_rescale vs hron_rescale | +0.02614 ± 0.0042 (0/5) | +0.005906 ± 0.0046 (1/5) | +29.63 ± 21 (0/5) | +0.02284 ± 0.002 (0/5) | +0.02112 ± 0.0018 (0/5) | -0.0246 ± 0.0049 (5/5) | +0.054 ± 0.0045 (0/5) |

### Bảng 3: ghép cặp trong cùng model

| A vs B | MAE tỷ lệ | MSE tỷ lệ | MAE count | MAE tỷ lệ series-macro | MAE tỷ lệ, đáp án = 0 | MAE tỷ lệ, đáp án > 0 | MAE tỷ lệ, ô model đáp án = 0 |
|---|---|---|---|---|---|---|---|
| hron_cap vs hron_prop | -0.001287 ± 0.00087 (5/5) | -0.0008421 ± 0.0006 (5/5) | +9.944 ± 13 (0/5) | -0.001724 ± 0.0011 (5/5) | -0.001041 ± 6.2e-05 (5/5) | +0.001207 ± 0.0023 (2/5) | -0.002663 ± 0.00019 (5/5) |
| hron_rescale vs hron_prop | +0.04131 ± 0.0033 (0/5) | +0.01876 ± 0.0031 (0/5) | +88.74 ± 8.8 (0/5) | +0.0443 ± 0.0044 (0/5) | +0.03589 ± 0.0012 (0/5) | -0.05058 ± 0.0042 (5/5) | +0.09179 ± 0.0039 (0/5) |
| tsagris_cap vs tsagris_prop | -5.551e-18 ± 1.2e-17 (1/5, hòa 4) | +0 ± 0 (0/5, hòa 5) | +7.105e-16 ± 7.3e-15 (1/5, hòa 2) | -2.776e-18 ± 6.2e-18 (1/5, hòa 4) | +0 ± 0 (0/5, hòa 5) | -5.551e-18 ± 1.2e-17 (1/5, hòa 4) | +5.551e-18 ± 7.6e-18 (0/5, hòa 3) |
| tsagris_rescale vs tsagris_prop | +0.05109 ± 0.0019 (0/5) | +0.04927 ± 0.0024 (0/5) | +62.72 ± 13 (0/5) | +0.05589 ± 0.0038 (0/5) | +0.03652 ± 0.0026 (0/5) | -0.02576 ± 0.012 (5/5) | +0.09335 ± 0.0063 (0/5) |
| micoda_other_cap vs micoda_other_prop | +9.795e-07 ± 5.8e-08 (0/5) | -6.29e-07 ± 6.9e-08 (5/5) | +0.002055 ± 0.0012 (0/5) | +5.445e-07 ± 2.2e-08 (0/5) | -6.524e-07 ± 1.7e-08 (5/5) | +5.799e-06 ± 1.3e-07 (0/5) | -1.668e-06 ± 4.2e-08 (5/5) |
| micoda_other_rescale vs micoda_other_prop | +0.04885 ± 0.0016 (0/5) | +0.01642 ± 0.0012 (0/5) | +31.73 ± 22 (1/5) | +0.0505 ± 0.0045 (0/5) | +0.04376 ± 0.0012 (0/5) | -0.06598 ± 0.0047 (5/5) | +0.1119 ± 0.0041 (0/5) |

### Bảng 4: tách khoảng cách (MAE tỷ lệ; mô tả, không phải phân rã chặt)

| Đường tách | Thành phần | Δ mean ± sd | âm / 5 seed | % khoảng cách |
|---|---|---|---:|---:|
| — | khoảng cách sdm_cap − hron_prop | -0.0582 ± 0.0049 | 5 | 100% |
| cap | thuật toán @cap | -0.0570 ± 0.0044 | 5 | +98% |
| cap | quy tắc hron (cap − prop_other) | -0.0013 ± 0.0009 | 5 | +2% |
| rescale | quy tắc same_day_mean (cap − rescale) | -0.0463 ± 0.0034 | 5 | +79% |
| rescale | thuật toán @rescale | -0.0533 ± 0.0021 | 5 | +91% |
| rescale | quy tắc hron (rescale − prop_other) | +0.0413 ± 0.0033 | 0 | -71% |

| Đường tách | Thành phần | Δ mean ± sd | âm / 5 seed | % khoảng cách |
|---|---|---|---:|---:|
| — | khoảng cách sdm_cap − tsagris_prop | -0.0689 ± 0.0015 | 5 | 100% |
| cap | thuật toán @cap | -0.0689 ± 0.0015 | 5 | +100% |
| cap | quy tắc tsagris (cap − prop_other) | -0.0000 ± 0.0000 | 1 | +0% |
| rescale | quy tắc same_day_mean (cap − rescale) | -0.0463 ± 0.0034 | 5 | +67% |
| rescale | thuật toán @rescale | -0.0737 ± 0.0033 | 5 | +107% |
| rescale | quy tắc tsagris (rescale − prop_other) | +0.0511 ± 0.0019 | 0 | -74% |

### Bảng 5: theo nhóm đáp án

| Biến thể | Model | Quy tắc | MAE tỷ lệ, đáp án = 0 | MAE tỷ lệ, đáp án > 0 | MAE tỷ lệ, ô model đáp án = 0 |
|---|---|---|---:|---:|---:|
| hron_prop | hron | prop_other | 0.0123 ± 0.0008 | 0.1943 ± 0.0061 | 0.0315 ± 0.0024 |
| hron_cap | hron | cap | 0.0113 ± 0.0008 | 0.1955 ± 0.0057 | 0.0288 ± 0.0024 |
| hron_rescale | hron | rescale | 0.0482 ± 0.0014 | 0.1437 ± 0.0083 | 0.1232 ± 0.0052 |
| tsagris_prop | tsagris | prop_other | 0.0305 ± 0.0025 | 0.1395 ± 0.0149 | 0.0781 ± 0.0071 |
| tsagris_cap | tsagris | cap | 0.0305 ± 0.0025 | 0.1395 ± 0.0149 | 0.0781 ± 0.0071 |
| tsagris_rescale | tsagris | rescale | 0.0671 ± 0.0017 | 0.1137 ± 0.0109 | 0.1715 ± 0.0058 |
| micoda_other_prop | micoda_other | prop_other | 0.0256 ± 0.0006 | 0.1851 ± 0.0025 | 0.0653 ± 0.0017 |
| micoda_other_cap | micoda_other | cap | 0.0256 ± 0.0006 | 0.1851 ± 0.0025 | 0.0653 ± 0.0017 |
| micoda_other_rescale | micoda_other | rescale | 0.0693 ± 0.0016 | 0.1191 ± 0.0067 | 0.1772 ± 0.0056 |
| micoda_rescale | micoda (A17) | rescale | 0.0446 ± 0.0011 | 0.1641 ± 0.0041 | 0.1139 ± 0.0042 |
| sdm_cap | same_day_mean | cap | 0.0036 ± 0.0002 | 0.0704 ± 0.0077 | 0.0093 ± 0.0005 |
| sdm_rescale | same_day_mean | rescale | 0.0295 ± 0.0013 | 0.0805 ± 0.0085 | 0.0755 ± 0.0045 |
| mean_fill_cap | mean_fill | cap | 0.0226 ± 0.0004 | 0.2817 ± 0.0045 | 0.0577 ± 0.0012 |

### Bảng 6: Other ngầm

Dòng đầy đủ trước khi che: 516, thuộc 5 nước. Dòng dùng / số nước theo seed: seed 0: 299 / 5, seed 1: 307 / 5, seed 2: 309 / 5, seed 3: 294 / 5, seed 4: 308 / 5

| Biến thể | Quy tắc | MAE tỷ lệ Other ngầm | Sai số có dấu (ngầm − thật) |
|---|---|---:|---:|
| hron_prop | prop_other | 0.0098 ± 0.0023 | -0.0005 ± 0.0041 |
| hron_cap | cap | 0.0114 ± 0.0041 | 0.0021 ± 0.0047 |
| hron_rescale | rescale | 0.1504 ± 0.0179 | -0.1504 ± 0.0179 |
| tsagris_prop | prop_other | 0.0190 ± 0.0028 | -0.0147 ± 0.0038 |
| tsagris_cap | cap | 0.0190 ± 0.0028 | -0.0147 ± 0.0038 |
| tsagris_rescale | rescale | 0.1108 ± 0.0112 | -0.1107 ± 0.0112 |
| micoda_other_prop | prop_other | 0.0379 ± 0.0075 | 0.0082 ± 0.0051 |
| micoda_other_cap | cap | 0.0379 ± 0.0075 | 0.0083 ± 0.0051 |
| micoda_other_rescale | rescale | 0.1504 ± 0.0179 | -0.1504 ± 0.0179 |
| micoda_rescale | rescale | 0.1504 ± 0.0179 | -0.1504 ± 0.0179 |
| sdm_cap | cap | 0.0058 ± 0.0019 | 0.0016 ± 0.0021 |
| sdm_rescale | rescale | 0.0731 ± 0.0070 | -0.0731 ± 0.0070 |
| mean_fill_cap | cap | 0.0501 ± 0.0069 | 0.0228 ± 0.0048 |

### Bảng 7: series-macro so với cell-weighted

**prop_other**
Seed chung: 0, 1, 2, 3, 4. Cell-weighted: hron_prop < tsagris_prop < micoda_other_prop.
Series-macro: hron_prop < tsagris_prop < micoda_other_prop.
Thứ hạng **không đổi**.

**cap**
Seed chung: 0, 1, 2, 3, 4. Cell-weighted: sdm_cap < hron_cap < tsagris_cap < micoda_other_cap < mean_fill_cap.
Series-macro: sdm_cap < hron_cap < tsagris_cap < micoda_other_cap < mean_fill_cap.
Thứ hạng **không đổi**.

**rescale**
Seed chung: 0, 1, 2, 3, 4. Cell-weighted: sdm_rescale < hron_rescale < micoda_rescale < tsagris_rescale < micoda_other_rescale.
Series-macro: sdm_rescale < micoda_rescale < hron_rescale < micoda_other_rescale < tsagris_rescale.
- Đổi chỗ hron_rescale / micoda_rescale: cell-weighted hron_rescale thắng 3/5 (Δ -0.00123); series-macro hron_rescale thắng 0/5 (Δ +0.00398).
- Đổi chỗ tsagris_rescale / micoda_other_rescale: cell-weighted tsagris_rescale thắng 5/5 (Δ -0.00566); series-macro tsagris_rescale thắng 2/5 (Δ +0.00380).

**mọi biến thể**
Seed chung: 0, 1, 2, 3, 4. Cell-weighted: sdm_cap < sdm_rescale < hron_cap < hron_prop < tsagris_cap < tsagris_prop < micoda_other_prop < micoda_other_cap < hron_rescale < micoda_rescale < mean_fill_cap < tsagris_rescale < micoda_other_rescale.
Series-macro: sdm_cap < sdm_rescale < hron_cap < hron_prop < tsagris_cap < tsagris_prop < micoda_other_prop < micoda_other_cap < mean_fill_cap < micoda_rescale < hron_rescale < micoda_other_rescale < tsagris_rescale.
- Đổi chỗ hron_rescale / micoda_rescale: cell-weighted hron_rescale thắng 3/5 (Δ -0.00123); series-macro hron_rescale thắng 0/5 (Δ +0.00398).
- Đổi chỗ hron_rescale / mean_fill_cap: cell-weighted hron_rescale thắng 5/5 (Δ -0.00660); series-macro hron_rescale thắng 0/5 (Δ +0.00745).
- Đổi chỗ micoda_rescale / mean_fill_cap: cell-weighted micoda_rescale thắng 5/5 (Δ -0.00537); series-macro micoda_rescale thắng 0/5 (Δ +0.00348).
- Đổi chỗ tsagris_rescale / micoda_other_rescale: cell-weighted tsagris_rescale thắng 5/5 (Δ -0.00566); series-macro tsagris_rescale thắng 2/5 (Δ +0.00380).

