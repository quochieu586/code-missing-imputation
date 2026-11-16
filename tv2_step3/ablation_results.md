# Step 3: ablation same_day_mean (COVID đầy đủ, che 5%, seed 0–4) — **exploratory**

Sinh bởi `ablation.py` (chấm điểm bằng `tier01`); không sửa tay. Metric chính: MAE tỷ lệ, cell-weighted, nhóm ô model. mean ± sd (ddof=1) qua seed, không phải khoảng tin cậy.

## whole_feature (chính)

### Trạng thái

| Model | Lượt | Mask | Lên lịch | completed | partial | failed | missing | Run không ok |
|---|---|---|---:|---:|---:|---:|---:|---|
| sdm_full | native | none | 1 | 1 | 0 | 0 | 0 | - |
| sdm_no_time | native | none | 1 | 1 | 0 | 0 | 0 | - |
| sdm_no_group | native | none | 1 | 1 | 0 | 0 | 0 | - |
| sdm_unweighted | native | none | 1 | 1 | 0 | 0 | 0 | - |
| sdm_rescale | native | none | 1 | 1 | 0 | 0 | 0 | - |
| sdm_none | native | none | 1 | 1 | 0 | 0 | 0 | - |
| sdm_full | masked | whole_feature | 5 | 5 | 0 | 0 | 0 | - |
| sdm_no_time | masked | whole_feature | 5 | 5 | 0 | 0 | 0 | - |
| sdm_no_group | masked | whole_feature | 5 | 5 | 0 | 0 | 0 | - |
| sdm_unweighted | masked | whole_feature | 5 | 5 | 0 | 0 | 0 | - |
| sdm_rescale | masked | whole_feature | 5 | 5 | 0 | 0 | 0 | - |
| sdm_none | masked | whole_feature | 5 | 5 | 0 | 0 | 0 | - |

### Bảng metric

Tập seed chung (mọi model completed, coverage = 1): 0, 1, 2, 3, 4.

Sắp theo MAE tỷ lệ (nhóm ô model) trung bình trên tập seed chung. mean ± sd (ddof=1) qua seed; sd không phải khoảng tin cậy. **exploratory**.

| Model | MAE tỷ lệ | MSE tỷ lệ | MAE count | MAE tỷ lệ series-macro | MAE tỷ lệ, đáp án = 0 | MAE tỷ lệ, đáp án > 0 | MAE tỷ lệ, ô model đáp án = 0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| sdm_unweighted | 0.0349 ± 0.0056 | 0.0120 ± 0.0030 | 42.4 ± 53.6 | 0.0487 ± 0.0084 | 0.0034 ± 0.0014 | 0.0830 ± 0.0131 | 0.0097 ± 0.0042 |
| sdm_full | 0.0351 ± 0.0066 | 0.0142 ± 0.0036 | 29.4 ± 26.8 | 0.0495 ± 0.0095 | 0.0033 ± 0.0014 | 0.0844 ± 0.0181 | 0.0094 ± 0.0040 |
| sdm_no_group | 0.0378 ± 0.0071 | 0.0152 ± 0.0045 | 28.5 ± 17.7 | 0.0542 ± 0.0081 | 0.0040 ± 0.0009 | 0.0884 ± 0.0228 | 0.0116 ± 0.0028 |
| hron | 0.0875 ± 0.0105 | 0.0660 ± 0.0092 | 74.6 ± 113.7 | 0.1003 ± 0.0135 | 0.0103 ± 0.0027 | 0.1978 ± 0.0195 | 0.0294 ± 0.0065 |
| sdm_rescale | 0.0877 ± 0.0169 | 0.0497 ± 0.0119 | 83.8 ± 56.3 | 0.1047 ± 0.0183 | 0.0307 ± 0.0043 | 0.0874 ± 0.0253 | 0.0883 ± 0.0149 |
| sdm_no_time | 0.1353 ± 0.0173 | 0.0682 ± 0.0078 | 207.1 ± 178.8 | 0.1298 ± 0.0167 | 0.0194 ± 0.0050 | 0.2862 ± 0.0210 | 0.0554 ± 0.0127 |
| mean_fill | 0.1354 ± 0.0145 | 0.0704 ± 0.0084 | 218.2 ± 186.8 | 0.1324 ± 0.0162 | 0.0178 ± 0.0037 | 0.2950 ± 0.0154 | 0.0508 ± 0.0092 |
| sdm_none | 0.1354 ± 0.0145 | 0.0704 ± 0.0084 | 218.2 ± 186.8 | 0.1324 ± 0.0162 | 0.0178 ± 0.0037 | 0.2950 ± 0.0154 | 0.0508 ± 0.0092 |

### Biến thể vs sdm_full (mọi metric)

Ô: Δ = A − B (mean ± sd), (A thắng / n seed). Thắng = sai số nhỏ hơn.

| A vs B | MAE tỷ lệ | MSE tỷ lệ | MAE count | MAE tỷ lệ series-macro | MAE tỷ lệ, đáp án = 0 | MAE tỷ lệ, đáp án > 0 | MAE tỷ lệ, ô model đáp án = 0 |
|---|---|---|---|---|---|---|---|
| sdm_no_time vs sdm_full | +0.1002 ± 0.022 (0/5) | +0.05392 ± 0.011 (0/5) | +177.7 ± 1.6e+02 (0/5) | +0.08038 ± 0.022 (0/5) | +0.01617 ± 0.0046 (0/5) | +0.2018 ± 0.037 (0/5) | +0.04606 ± 0.012 (0/5) |
| sdm_no_group vs sdm_full | +0.002756 ± 0.0034 (2/5) | +0.0009273 ± 0.0017 (1/5) | -0.8997 ± 10 (2/5) | +0.004693 ± 0.0045 (1/5) | +0.0007633 ± 0.001 (1/5) | +0.004013 ± 0.0064 (1/5) | +0.002224 ± 0.003 (1/5) |
| sdm_unweighted vs sdm_full | -0.0001953 ± 0.002 (3/5) | -0.002246 ± 0.001 (5/5) | +13.04 ± 29 (1/5) | -0.0007989 ± 0.0024 (3/5) | +9.949e-05 ± 0.00037 (1/5) | -0.001442 ± 0.0055 (2/5) | +0.0003169 ± 0.001 (1/5) |
| sdm_rescale vs sdm_full | +0.05264 ± 0.011 (0/5) | +0.03542 ± 0.0085 (0/5) | +54.43 ± 35 (0/5) | +0.05521 ± 0.011 (0/5) | +0.02741 ± 0.0036 (0/5) | +0.002966 ± 0.0094 (1/5) | +0.07894 ± 0.013 (0/5) |
| sdm_none vs sdm_full | +0.1003 ± 0.019 (0/5) | +0.05615 ± 0.011 (0/5) | +188.8 ± 1.6e+02 (0/5) | +0.08299 ± 0.021 (0/5) | +0.01453 ± 0.0035 (0/5) | +0.2105 ± 0.032 (0/5) | +0.04145 ± 0.0087 (0/5) |
| sdm_none vs mean_fill | +0 ± 0 (0/5, hòa 5) | +0 ± 0 (0/5, hòa 5) | +0 ± 0 (0/5, hòa 5) | +0 ± 0 (0/5, hòa 5) | +0 ± 0 (0/5, hòa 5) | +0 ± 0 (0/5, hòa 5) | +0 ± 0 (0/5, hòa 5) |
| sdm_full vs hron | -0.05247 ± 0.016 (5/5) | -0.0518 ± 0.012 (5/5) | -45.16 ± 89 (4/5) | -0.05085 ± 0.018 (5/5) | -0.007045 ± 0.0028 (5/5) | -0.1134 ± 0.033 (5/5) | -0.01997 ± 0.0069 (5/5) |

### Đóng góp (mô tả, không cộng tính)

| Yếu tố tắt | MAE tỷ lệ | MSE tỷ lệ | MAE count | MAE tỷ lệ series-macro | MAE tỷ lệ, đáp án = 0 | MAE tỷ lệ, đáp án > 0 | MAE tỷ lệ, ô model đáp án = 0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| sdm_none − sdm_full (mẫu số) | +0.1003 | +0.05615 | +188.8 | +0.08299 | +0.01453 | +0.2105 | +0.04145 |
| T thời gian (sdm_no_time) | +100% (Δ +0.1) | +96% (Δ +0.0539) | +94% (Δ +178) | +97% (Δ +0.0804) | +111% (Δ +0.0162) | +96% (Δ +0.202) | +111% (Δ +0.0461) |
| G châu lục (sdm_no_group) | +3% (Δ +0.00276) | +2% (Δ +0.000927) | -0% (Δ -0.9) | +6% (Δ +0.00469) | +5% (Δ +0.000763) | +2% (Δ +0.00401) | +5% (Δ +0.00222) |
| W trọng số N (sdm_unweighted) | -0% (Δ -0.000195) | -4% (Δ -0.00225) | +7% (Δ +13) | -1% (Δ -0.000799) | +1% (Δ +9.95e-05) | -1% (Δ -0.00144) | +1% (Δ +0.000317) |
| R phần dư (sdm_rescale) | +52% (Δ +0.0526) | +63% (Δ +0.0354) | +29% (Δ +54.4) | +67% (Δ +0.0552) | +189% (Δ +0.0274) | +1% (Δ +0.00297) | +190% (Δ +0.0789) |

### Mức cửa sổ trên ô che thuộc nhóm model (gốc / nới 1 / nới 2 / toàn cục)

| Biến thể | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | Tổng: toàn cục / ô model |
|---|---|---|---|---|---|---:|
| sdm_full | 2552 / 6 / 0 / 37 | 2399 / 149 / 66 / 115 | 2247 / 106 / 53 / 116 | 2218 / 121 / 26 / 139 | 2142 / 24 / 23 / 217 | 624 / 12756 (4.9%) |
| sdm_no_group | 2595 / 0 / 0 / 0 | 2729 / 0 / 0 / 0 | 2522 / 0 / 0 / 0 | 2504 / 0 / 0 / 0 | 2406 / 0 / 0 / 0 | 0 / 12756 (0.0%) |
| sdm_unweighted | 2552 / 6 / 0 / 37 | 2399 / 149 / 66 / 115 | 2247 / 106 / 53 / 116 | 2218 / 121 / 26 / 139 | 2142 / 24 / 23 / 217 | 624 / 12756 (4.9%) |
| sdm_rescale | 2552 / 6 / 0 / 37 | 2399 / 149 / 66 / 115 | 2247 / 106 / 53 / 116 | 2218 / 121 / 26 / 139 | 2142 / 24 / 23 / 217 | 624 / 12756 (4.9%) |

Ô thiếu thật (lượt native), mức toàn cục:
- sdm_full: 21946 / 86050 (25.5%); window 51136, widened_1 4912, widened_2 8056, global 21946
- sdm_no_group: 0 / 86050 (0.0%); window 86050, widened_1 0, widened_2 0, global 0

### Series-macro so với cell-weighted

Seed chung: 0, 1, 2, 3, 4. Cell-weighted: sdm_unweighted < sdm_full < sdm_no_group < hron < sdm_rescale < sdm_no_time < mean_fill < sdm_none.
Series-macro: sdm_unweighted < sdm_full < sdm_no_group < hron < sdm_rescale < sdm_no_time < mean_fill < sdm_none.
Thứ hạng **không đổi**.

sdm_none vs mean_fill, whole_feature: 28309 / 28309 ô che dự đoán giống hệt; chênh lớn nhất 0 (count), 0 (tỷ lệ).

Regression check: 2070 giá trị khớp summary.json, sai khác lớn nhất 0. sdm_full = same_day_mean step 2: CSV native giống từng byte, 28309 dòng ô che giống hệt.

## random_cell (sensitivity)

### Trạng thái

| Model | Lượt | Mask | Lên lịch | completed | partial | failed | missing | Run không ok |
|---|---|---|---:|---:|---:|---:|---:|---|
| sdm_full | native | none | 1 | 1 | 0 | 0 | 0 | - |
| sdm_no_time | native | none | 1 | 1 | 0 | 0 | 0 | - |
| sdm_no_group | native | none | 1 | 1 | 0 | 0 | 0 | - |
| sdm_unweighted | native | none | 1 | 1 | 0 | 0 | 0 | - |
| sdm_rescale | native | none | 1 | 1 | 0 | 0 | 0 | - |
| sdm_none | native | none | 1 | 1 | 0 | 0 | 0 | - |
| sdm_full | masked | random_cell | 5 | 5 | 0 | 0 | 0 | - |
| sdm_no_time | masked | random_cell | 5 | 5 | 0 | 0 | 0 | - |
| sdm_no_group | masked | random_cell | 5 | 5 | 0 | 0 | 0 | - |
| sdm_unweighted | masked | random_cell | 5 | 5 | 0 | 0 | 0 | - |
| sdm_rescale | masked | random_cell | 5 | 5 | 0 | 0 | 0 | - |
| sdm_none | masked | random_cell | 5 | 5 | 0 | 0 | 0 | - |

### Bảng metric

Tập seed chung (mọi model completed, coverage = 1): 0, 1, 2, 3, 4.

Sắp theo MAE tỷ lệ (nhóm ô model) trung bình trên tập seed chung. mean ± sd (ddof=1) qua seed; sd không phải khoảng tin cậy. **exploratory**.

| Model | MAE tỷ lệ | MSE tỷ lệ | MAE count | MAE tỷ lệ series-macro | MAE tỷ lệ, đáp án = 0 | MAE tỷ lệ, đáp án > 0 | MAE tỷ lệ, ô model đáp án = 0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| sdm_unweighted | 0.0304 ± 0.0024 | 0.0096 ± 0.0012 | 26.6 ± 6.2 | 0.0372 ± 0.0038 | 0.0033 ± 0.0003 | 0.0703 ± 0.0072 | 0.0086 ± 0.0006 |
| sdm_full | 0.0310 ± 0.0027 | 0.0110 ± 0.0015 | 19.0 ± 4.8 | 0.0376 ± 0.0041 | 0.0036 ± 0.0002 | 0.0704 ± 0.0077 | 0.0093 ± 0.0005 |
| sdm_no_group | 0.0339 ± 0.0024 | 0.0128 ± 0.0015 | 28.4 ± 6.5 | 0.0411 ± 0.0043 | 0.0040 ± 0.0003 | 0.0770 ± 0.0074 | 0.0103 ± 0.0007 |
| sdm_rescale | 0.0773 ± 0.0019 | 0.0453 ± 0.0020 | 76.3 ± 12.5 | 0.0881 ± 0.0028 | 0.0295 ± 0.0013 | 0.0805 ± 0.0085 | 0.0755 ± 0.0045 |
| hron | 0.0892 ± 0.0028 | 0.0613 ± 0.0027 | 22.9 ± 4.0 | 0.0909 ± 0.0057 | 0.0123 ± 0.0008 | 0.1943 ± 0.0061 | 0.0315 ± 0.0024 |
| sdm_no_time | 0.1364 ± 0.0019 | 0.0666 ± 0.0005 | 142.3 ± 31.6 | 0.1271 ± 0.0045 | 0.0251 ± 0.0005 | 0.2682 ± 0.0054 | 0.0641 ± 0.0013 |
| mean_fill | 0.1371 ± 0.0016 | 0.0692 ± 0.0007 | 152.4 ± 33.5 | 0.1277 ± 0.0041 | 0.0226 ± 0.0004 | 0.2817 ± 0.0045 | 0.0577 ± 0.0012 |
| sdm_none | 0.1371 ± 0.0016 | 0.0692 ± 0.0007 | 152.4 ± 33.5 | 0.1277 ± 0.0041 | 0.0226 ± 0.0004 | 0.2817 ± 0.0045 | 0.0577 ± 0.0012 |

### Biến thể vs sdm_full (mọi metric)

Ô: Δ = A − B (mean ± sd), (A thắng / n seed). Thắng = sai số nhỏ hơn.

| A vs B | MAE tỷ lệ | MSE tỷ lệ | MAE count | MAE tỷ lệ series-macro | MAE tỷ lệ, đáp án = 0 | MAE tỷ lệ, đáp án > 0 | MAE tỷ lệ, ô model đáp án = 0 |
|---|---|---|---|---|---|---|---|
| sdm_no_time vs sdm_full | +0.1055 ± 0.0022 (0/5) | +0.05552 ± 0.0013 (0/5) | +123.3 ± 28 (0/5) | +0.08952 ± 0.0028 (0/5) | +0.02142 ± 0.0004 (0/5) | +0.1978 ± 0.0061 (0/5) | +0.05478 ± 0.0013 (0/5) |
| sdm_no_group vs sdm_full | +0.002969 ± 0.0011 (0/5) | +0.001747 ± 0.00062 (0/5) | +9.365 ± 5.4 (0/5) | +0.003469 ± 0.0016 (0/5) | +0.0003829 ± 0.0003 (0/5) | +0.00657 ± 0.003 (0/5) | +0.0009872 ± 0.00078 (0/5) |
| sdm_unweighted vs sdm_full | -0.0005375 ± 0.001 (4/5) | -0.001477 ± 0.00058 (5/5) | +7.557 ± 3.5 (0/5) | -0.000411 ± 0.0012 (3/5) | -0.0002996 ± 0.0001 (5/5) | -0.0001141 ± 0.0025 (3/5) | -0.0007671 ± 0.00027 (5/5) |
| sdm_rescale vs sdm_full | +0.0463 ± 0.0034 (0/5) | +0.03424 ± 0.0025 (0/5) | +57.33 ± 13 (0/5) | +0.05051 ± 0.004 (0/5) | +0.02585 ± 0.0015 (0/5) | +0.01015 ± 0.0079 (1/5) | +0.06615 ± 0.0047 (0/5) |
| sdm_none vs sdm_full | +0.1062 ± 0.0017 (0/5) | +0.05811 ± 0.00096 (0/5) | +133.4 ± 30 (0/5) | +0.09012 ± 0.0029 (0/5) | +0.01892 ± 0.00037 (0/5) | +0.2113 ± 0.0044 (0/5) | +0.04837 ± 0.0013 (0/5) |
| sdm_none vs mean_fill | +0 ± 0 (0/5, hòa 5) | +0 ± 0 (0/5, hòa 5) | +0 ± 0 (0/5, hòa 5) | +0 ± 0 (0/5, hòa 5) | +0 ± 0 (0/5, hòa 5) | +0 ± 0 (0/5, hòa 5) | +0 ± 0 (0/5, hòa 5) |
| sdm_full vs hron | -0.05825 ± 0.0049 (5/5) | -0.05021 ± 0.0037 (5/5) | -3.929 ± 3.3 (5/5) | -0.05328 ± 0.0052 (5/5) | -0.008649 ± 0.001 (5/5) | -0.1239 ± 0.0099 (5/5) | -0.02214 ± 0.0028 (5/5) |

### Đóng góp (mô tả, không cộng tính)

| Yếu tố tắt | MAE tỷ lệ | MSE tỷ lệ | MAE count | MAE tỷ lệ series-macro | MAE tỷ lệ, đáp án = 0 | MAE tỷ lệ, đáp án > 0 | MAE tỷ lệ, ô model đáp án = 0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| sdm_none − sdm_full (mẫu số) | +0.1062 | +0.05811 | +133.4 | +0.09012 | +0.01892 | +0.2113 | +0.04837 |
| T thời gian (sdm_no_time) | +99% (Δ +0.105) | +96% (Δ +0.0555) | +92% (Δ +123) | +99% (Δ +0.0895) | +113% (Δ +0.0214) | +94% (Δ +0.198) | +113% (Δ +0.0548) |
| G châu lục (sdm_no_group) | +3% (Δ +0.00297) | +3% (Δ +0.00175) | +7% (Δ +9.36) | +4% (Δ +0.00347) | +2% (Δ +0.000383) | +3% (Δ +0.00657) | +2% (Δ +0.000987) |
| W trọng số N (sdm_unweighted) | -1% (Δ -0.000538) | -3% (Δ -0.00148) | +6% (Δ +7.56) | -0% (Δ -0.000411) | -2% (Δ -0.0003) | -0% (Δ -0.000114) | -2% (Δ -0.000767) |
| R phần dư (sdm_rescale) | +44% (Δ +0.0463) | +59% (Δ +0.0342) | +43% (Δ +57.3) | +56% (Δ +0.0505) | +137% (Δ +0.0259) | +5% (Δ +0.0101) | +137% (Δ +0.0661) |

### Mức cửa sổ trên ô che thuộc nhóm model (gốc / nới 1 / nới 2 / toàn cục)

| Biến thể | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | Tổng: toàn cục / ô model |
|---|---|---|---|---|---|---:|
| sdm_full | 2661 / 104 / 63 / 23 | 2560 / 93 / 68 / 19 | 2566 / 92 / 76 / 28 | 2651 / 102 / 60 / 19 | 2570 / 97 / 61 / 22 | 111 / 13935 (0.8%) |
| sdm_no_group | 2851 / 0 / 0 / 0 | 2740 / 0 / 0 / 0 | 2762 / 0 / 0 / 0 | 2832 / 0 / 0 / 0 | 2750 / 0 / 0 / 0 | 0 / 13935 (0.0%) |
| sdm_unweighted | 2661 / 104 / 63 / 23 | 2560 / 93 / 68 / 19 | 2566 / 92 / 76 / 28 | 2651 / 102 / 60 / 19 | 2570 / 97 / 61 / 22 | 111 / 13935 (0.8%) |
| sdm_rescale | 2661 / 104 / 63 / 23 | 2560 / 93 / 68 / 19 | 2566 / 92 / 76 / 28 | 2651 / 102 / 60 / 19 | 2570 / 97 / 61 / 22 | 111 / 13935 (0.8%) |

Ô thiếu thật (lượt native), mức toàn cục:
- sdm_full: 21946 / 86050 (25.5%); window 51136, widened_1 4912, widened_2 8056, global 21946
- sdm_no_group: 0 / 86050 (0.0%); window 86050, widened_1 0, widened_2 0, global 0

### Series-macro so với cell-weighted

Seed chung: 0, 1, 2, 3, 4. Cell-weighted: sdm_unweighted < sdm_full < sdm_no_group < sdm_rescale < hron < sdm_no_time < mean_fill < sdm_none.
Series-macro: sdm_unweighted < sdm_full < sdm_no_group < sdm_rescale < hron < sdm_no_time < mean_fill < sdm_none.
Thứ hạng **không đổi**.

sdm_none vs mean_fill, random_cell: 27936 / 27936 ô che dự đoán giống hệt; chênh lớn nhất 0 (count), 0 (tỷ lệ).

Regression check: 1950 giá trị khớp summary.json, sai khác lớn nhất 0. sdm_full = same_day_mean step 2: CSV native giống từng byte, 27936 dòng ô che giống hệt.

