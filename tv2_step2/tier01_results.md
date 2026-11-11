# Step 2: Tầng 0–1 (COVID đầy đủ, che 5%) — **exploratory**

Sinh bởi `tier01.py`; không sửa tay. Metric chính: MAE tỷ lệ trên nhóm ô `model`, cell-weighted trong từng seed. Seed 0–4 đã được nhìn và chỉnh trước đó.

## whole_feature (chính)

### Tầng 0: reliability (mọi run đã lên lịch)

| Model | Lượt | Mask | Lên lịch | completed | partial | failed | missing | Run không ok |
|---|---|---|---:|---:|---:|---:|---:|---|
| hron | native | none | 1 | 1 | 0 | 0 | 0 | - |
| tsagris | native | none | 1 | 1 | 0 | 0 | 0 | - |
| micoda | native | none | 1 | 1 | 0 | 0 | 0 | - |
| same_day_mean | native | none | 1 | 1 | 0 | 0 | 0 | - |
| mean_fill | native | none | 1 | 1 | 0 | 0 | 0 | - |
| marginal_sampler | native | none | 1 | 1 | 0 | 0 | 0 | - |
| zero_fill | native | none | 1 | 1 | 0 | 0 | 0 | - |
| hron | masked | whole_feature | 6 | 5 | 0 | 1 | 0 | 109: failed (baseline_error) |
| tsagris | masked | whole_feature | 6 | 5 | 0 | 1 | 0 | 109: failed (baseline_error) |
| micoda | masked | whole_feature | 6 | 5 | 0 | 1 | 0 | 109: failed (insufficient_fit) |
| same_day_mean | masked | whole_feature | 6 | 6 | 0 | 0 | 0 | - |
| mean_fill | masked | whole_feature | 6 | 6 | 0 | 0 | 0 | - |
| marginal_sampler | masked | whole_feature | 6 | 6 | 0 | 0 | 0 | - |
| zero_fill | masked | whole_feature | 6 | 6 | 0 | 0 | 0 | - |
| micoda_other | native | none | 1 | 1 | 0 | 0 | 0 | - |
| micoda_other | masked | whole_feature | 6 | 5 | 0 | 1 | 0 | 109: failed (insufficient_fit) |

### Tầng 1: bảng metric

Tập seed chung (mọi model completed, coverage = 1): 0, 1, 2, 3, 4.
Seed 109 bị loại: hron failed (baseline_error); tsagris failed (baseline_error); micoda failed (insufficient_fit); micoda_other failed (insufficient_fit).

Sắp theo MAE tỷ lệ (nhóm ô model) trung bình trên tập seed chung. mean ± sd (ddof=1) qua seed; sd không phải khoảng tin cậy. **exploratory**.

| Model | MAE tỷ lệ | MSE tỷ lệ | MAE count | MAE tỷ lệ series-macro | MAE tỷ lệ, đáp án = 0 | MAE tỷ lệ, đáp án > 0 | MAE tỷ lệ, ô model đáp án = 0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| same_day_mean | 0.0351 ± 0.0066 | 0.0142 ± 0.0036 | 29.4 ± 26.8 | 0.0495 ± 0.0095 | 0.0033 ± 0.0014 | 0.0844 ± 0.0181 | 0.0094 ± 0.0040 |
| hron | 0.0875 ± 0.0105 | 0.0660 ± 0.0092 | 74.6 ± 113.7 | 0.1003 ± 0.0135 | 0.0103 ± 0.0027 | 0.1978 ± 0.0195 | 0.0294 ± 0.0065 |
| micoda_other | 0.1075 ± 0.0093 | 0.0684 ± 0.0078 | 188.7 ± 205.5 | 0.1175 ± 0.0137 | 0.0194 ± 0.0036 | 0.2056 ± 0.0195 | 0.0554 ± 0.0095 |
| tsagris | 0.1079 ± 0.0209 | 0.0521 ± 0.0095 | 33.2 ± 14.7 | 0.1207 ± 0.0207 | 0.0260 ± 0.0075 | 0.1719 ± 0.0458 | 0.0745 ± 0.0214 |
| mean_fill | 0.1354 ± 0.0145 | 0.0704 ± 0.0084 | 218.2 ± 186.8 | 0.1324 ± 0.0162 | 0.0178 ± 0.0037 | 0.2950 ± 0.0154 | 0.0508 ± 0.0092 |
| marginal_sampler | 0.1354 ± 0.0120 | 0.1107 ± 0.0122 | 225.1 ± 200.7 | 0.1333 ± 0.0151 | 0.0144 ± 0.0019 | 0.3130 ± 0.0142 | 0.0411 ± 0.0037 |
| micoda | 0.1406 ± 0.0099 | 0.0823 ± 0.0094 | 236.2 ± 188.5 | 0.1476 ± 0.0161 | 0.0411 ± 0.0051 | 0.1834 ± 0.0261 | 0.1177 ± 0.0116 |
| zero_fill | 0.1693 ± 0.0219 | 0.1525 ± 0.0230 | 287.1 ± 211.3 | 0.1498 ± 0.0241 | 0.0000 ± 0.0000 | 0.4886 ± 0.0405 | 0.0000 ± 0.0000 |

### Ghép cặp, mọi cặp (MAE tỷ lệ)

**whole_feature** (có điều kiện trên tập seed chung của từng cặp; thắng = MAE tỷ lệ nhóm model nhỏ hơn; Δ = A − B, mean ± sd ddof=1)

| A vs B | n seed | A thắng | Δ MAE tỷ lệ | Seed loại (lý do) |
|---|---:|---:|---|---|
| same_day_mean vs hron | 5 | 5/5 | -0.05247 ± 0.01578 | 109: hron failed (baseline_error) |
| same_day_mean vs micoda_other | 5 | 5/5 | -0.07245 ± 0.01355 | 109: micoda_other failed (insufficient_fit) |
| same_day_mean vs tsagris | 5 | 5/5 | -0.07283 ± 0.02515 | 109: tsagris failed (baseline_error) |
| same_day_mean vs mean_fill | 6 | 6/6 | -0.10059 ± 0.01691 | - |
| same_day_mean vs marginal_sampler | 6 | 6/6 | -0.10068 ± 0.01522 | - |
| same_day_mean vs micoda | 5 | 5/5 | -0.10554 ± 0.01238 | 109: micoda failed (insufficient_fit) |
| same_day_mean vs zero_fill | 6 | 6/6 | -0.13628 ± 0.02401 | - |
| hron vs micoda_other | 5 | 5/5 | -0.01998 ± 0.00474 | 109: hron failed (baseline_error), micoda_other failed (insufficient_fit) |
| hron vs tsagris | 5 | 5/5 | -0.02036 ± 0.01274 | 109: hron failed (baseline_error), tsagris failed (baseline_error) |
| hron vs mean_fill | 5 | 5/5 | -0.04782 ± 0.00853 | 109: hron failed (baseline_error) |
| hron vs marginal_sampler | 5 | 5/5 | -0.04783 ± 0.00749 | 109: hron failed (baseline_error) |
| hron vs micoda | 5 | 5/5 | -0.05307 ± 0.00769 | 109: hron failed (baseline_error), micoda failed (insufficient_fit) |
| hron vs zero_fill | 5 | 5/5 | -0.08177 ± 0.01596 | 109: hron failed (baseline_error) |
| micoda_other vs tsagris | 5 | 2/5 | -0.00038 ± 0.01360 | 109: micoda_other failed (insufficient_fit), tsagris failed (baseline_error) |
| micoda_other vs mean_fill | 5 | 5/5 | -0.02785 ± 0.00699 | 109: micoda_other failed (insufficient_fit) |
| micoda_other vs marginal_sampler | 5 | 5/5 | -0.02785 ± 0.00567 | 109: micoda_other failed (insufficient_fit) |
| micoda_other vs micoda | 5 | 5/5 | -0.03309 ± 0.00681 | 109: micoda_other failed (insufficient_fit), micoda failed (insufficient_fit) |
| micoda_other vs zero_fill | 5 | 5/5 | -0.06180 ± 0.01493 | 109: micoda_other failed (insufficient_fit) |
| tsagris vs mean_fill | 5 | 5/5 | -0.02746 ± 0.01542 | 109: tsagris failed (baseline_error) |
| tsagris vs marginal_sampler | 5 | 5/5 | -0.02747 ± 0.01411 | 109: tsagris failed (baseline_error) |
| tsagris vs micoda | 5 | 5/5 | -0.03271 ± 0.01402 | 109: tsagris failed (baseline_error), micoda failed (insufficient_fit) |
| tsagris vs zero_fill | 5 | 5/5 | -0.06141 ± 0.01933 | 109: tsagris failed (baseline_error) |
| mean_fill vs marginal_sampler | 6 | 3/6 | -0.00009 ± 0.00427 | - |
| mean_fill vs micoda | 5 | 4/5 | -0.00525 ± 0.01350 | 109: micoda failed (insufficient_fit) |
| mean_fill vs zero_fill | 6 | 6/6 | -0.03568 ± 0.00842 | - |
| marginal_sampler vs micoda | 5 | 3/5 | -0.00524 ± 0.01214 | 109: micoda failed (insufficient_fit) |
| marginal_sampler vs zero_fill | 6 | 6/6 | -0.03559 ± 0.01052 | - |
| micoda vs zero_fill | 5 | 5/5 | -0.02870 ± 0.02139 | 109: micoda failed (insufficient_fit) |

### Series-macro so với cell-weighted

Seed chung: 0, 1, 2, 3, 4. Cell-weighted: same_day_mean < hron < micoda_other < tsagris < mean_fill < marginal_sampler < micoda < zero_fill.
Series-macro: same_day_mean < hron < micoda_other < tsagris < mean_fill < marginal_sampler < micoda < zero_fill.
Thứ hạng **không đổi**.

Regression check: 1320 giá trị khớp summary.json, sai khác lớn nhất 0 (ngưỡng 1e-12).

## random_cell (sensitivity)

### Tầng 0: reliability (mọi run đã lên lịch)

| Model | Lượt | Mask | Lên lịch | completed | partial | failed | missing | Run không ok |
|---|---|---|---:|---:|---:|---:|---:|---|
| hron | native | none | 1 | 1 | 0 | 0 | 0 | - |
| tsagris | native | none | 1 | 1 | 0 | 0 | 0 | - |
| micoda | native | none | 1 | 1 | 0 | 0 | 0 | - |
| same_day_mean | native | none | 1 | 1 | 0 | 0 | 0 | - |
| mean_fill | native | none | 1 | 1 | 0 | 0 | 0 | - |
| marginal_sampler | native | none | 1 | 1 | 0 | 0 | 0 | - |
| zero_fill | native | none | 1 | 1 | 0 | 0 | 0 | - |
| hron | masked | random_cell | 5 | 5 | 0 | 0 | 0 | - |
| tsagris | masked | random_cell | 5 | 5 | 0 | 0 | 0 | - |
| micoda | masked | random_cell | 5 | 5 | 0 | 0 | 0 | - |
| same_day_mean | masked | random_cell | 5 | 5 | 0 | 0 | 0 | - |
| mean_fill | masked | random_cell | 5 | 5 | 0 | 0 | 0 | - |
| marginal_sampler | masked | random_cell | 5 | 5 | 0 | 0 | 0 | - |
| zero_fill | masked | random_cell | 5 | 5 | 0 | 0 | 0 | - |
| micoda_other | native | none | 1 | 1 | 0 | 0 | 0 | - |
| micoda_other | masked | random_cell | 5 | 5 | 0 | 0 | 0 | - |

### Tầng 1: bảng metric

Tập seed chung (mọi model completed, coverage = 1): 0, 1, 2, 3, 4.

Sắp theo MAE tỷ lệ (nhóm ô model) trung bình trên tập seed chung. mean ± sd (ddof=1) qua seed; sd không phải khoảng tin cậy. **exploratory**.

| Model | MAE tỷ lệ | MSE tỷ lệ | MAE count | MAE tỷ lệ series-macro | MAE tỷ lệ, đáp án = 0 | MAE tỷ lệ, đáp án > 0 | MAE tỷ lệ, ô model đáp án = 0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| same_day_mean | 0.0310 ± 0.0027 | 0.0110 ± 0.0015 | 19.0 ± 4.8 | 0.0376 ± 0.0041 | 0.0036 ± 0.0002 | 0.0704 ± 0.0077 | 0.0093 ± 0.0005 |
| hron | 0.0892 ± 0.0028 | 0.0613 ± 0.0027 | 22.9 ± 4.0 | 0.0909 ± 0.0057 | 0.0123 ± 0.0008 | 0.1943 ± 0.0061 | 0.0315 ± 0.0024 |
| tsagris | 0.0999 ± 0.0024 | 0.0482 ± 0.0017 | 26.6 ± 3.6 | 0.1059 ± 0.0047 | 0.0305 ± 0.0025 | 0.1395 ± 0.0149 | 0.0781 ± 0.0071 |
| micoda_other | 0.1078 ± 0.0008 | 0.0695 ± 0.0013 | 109.6 ± 37.3 | 0.1075 ± 0.0049 | 0.0256 ± 0.0006 | 0.1851 ± 0.0025 | 0.0653 ± 0.0017 |
| micoda | 0.1317 ± 0.0018 | 0.0803 ± 0.0024 | 167.7 ± 38.9 | 0.1312 ± 0.0031 | 0.0446 ± 0.0011 | 0.1641 ± 0.0041 | 0.1139 ± 0.0042 |
| marginal_sampler | 0.1362 ± 0.0072 | 0.1112 ± 0.0069 | 143.4 ± 39.8 | 0.1262 ± 0.0082 | 0.0185 ± 0.0019 | 0.2983 ± 0.0115 | 0.0472 ± 0.0044 |
| mean_fill | 0.1371 ± 0.0016 | 0.0692 ± 0.0007 | 152.4 ± 33.5 | 0.1277 ± 0.0041 | 0.0226 ± 0.0004 | 0.2817 ± 0.0045 | 0.0577 ± 0.0012 |
| zero_fill | 0.1716 ± 0.0035 | 0.1551 ± 0.0031 | 226.8 ± 52.7 | 0.1473 ± 0.0040 | 0.0000 ± 0.0000 | 0.4840 ± 0.0115 | 0.0000 ± 0.0000 |

### Ghép cặp, mọi cặp (MAE tỷ lệ)

**random_cell** (có điều kiện trên tập seed chung của từng cặp; thắng = MAE tỷ lệ nhóm model nhỏ hơn; Δ = A − B, mean ± sd ddof=1)

| A vs B | n seed | A thắng | Δ MAE tỷ lệ | Seed loại (lý do) |
|---|---:|---:|---|---|
| same_day_mean vs hron | 5 | 5/5 | -0.05825 ± 0.00489 | - |
| same_day_mean vs tsagris | 5 | 5/5 | -0.06894 ± 0.00150 | - |
| same_day_mean vs micoda_other | 5 | 5/5 | -0.07685 ± 0.00214 | - |
| same_day_mean vs micoda | 5 | 5/5 | -0.10078 ± 0.00346 | - |
| same_day_mean vs marginal_sampler | 5 | 5/5 | -0.10527 ± 0.00644 | - |
| same_day_mean vs mean_fill | 5 | 5/5 | -0.10616 ± 0.00166 | - |
| same_day_mean vs zero_fill | 5 | 5/5 | -0.14061 ± 0.00279 | - |
| hron vs tsagris | 5 | 5/5 | -0.01070 ± 0.00465 | - |
| hron vs micoda_other | 5 | 5/5 | -0.01860 ± 0.00350 | - |
| hron vs micoda | 5 | 5/5 | -0.04254 ± 0.00396 | - |
| hron vs marginal_sampler | 5 | 5/5 | -0.04703 ± 0.00789 | - |
| hron vs mean_fill | 5 | 5/5 | -0.04791 ± 0.00405 | - |
| hron vs zero_fill | 5 | 5/5 | -0.08236 ± 0.00551 | - |
| tsagris vs micoda_other | 5 | 5/5 | -0.00790 ± 0.00211 | - |
| tsagris vs micoda | 5 | 5/5 | -0.03184 ± 0.00268 | - |
| tsagris vs marginal_sampler | 5 | 5/5 | -0.03633 ± 0.00729 | - |
| tsagris vs mean_fill | 5 | 5/5 | -0.03721 ± 0.00134 | - |
| tsagris vs zero_fill | 5 | 5/5 | -0.07167 ± 0.00244 | - |
| micoda_other vs micoda | 5 | 5/5 | -0.02394 ± 0.00184 | - |
| micoda_other vs marginal_sampler | 5 | 5/5 | -0.02843 ± 0.00714 | - |
| micoda_other vs mean_fill | 5 | 5/5 | -0.02931 ± 0.00139 | - |
| micoda_other vs zero_fill | 5 | 5/5 | -0.06377 ± 0.00339 | - |
| micoda vs marginal_sampler | 5 | 2/5 | -0.00449 ± 0.00881 | - |
| micoda vs mean_fill | 5 | 5/5 | -0.00537 ± 0.00256 | - |
| micoda vs zero_fill | 5 | 5/5 | -0.03983 ± 0.00426 | - |
| marginal_sampler vs mean_fill | 5 | 3/5 | -0.00088 ± 0.00645 | - |
| marginal_sampler vs zero_fill | 5 | 5/5 | -0.03534 ± 0.00573 | - |
| mean_fill vs zero_fill | 5 | 5/5 | -0.03446 ± 0.00203 | - |

### Series-macro so với cell-weighted

Seed chung: 0, 1, 2, 3, 4. Cell-weighted: same_day_mean < hron < tsagris < micoda_other < micoda < marginal_sampler < mean_fill < zero_fill.
Series-macro: same_day_mean < hron < tsagris < micoda_other < marginal_sampler < mean_fill < micoda < zero_fill.
- Đổi chỗ micoda / marginal_sampler: cell-weighted micoda thắng 2/5 (Δ -0.00449); series-macro micoda thắng 1/5 (Δ +0.00495).
- Đổi chỗ micoda / mean_fill: cell-weighted micoda thắng 5/5 (Δ -0.00537); series-macro micoda thắng 0/5 (Δ +0.00348).

Regression check: 1200 giá trị khớp summary.json, sai khác lớn nhất 0 (ngưỡng 1e-12).

## Seed 109 (whole_feature, ca thử chính sách failure)

Seed 109 (whole_feature): 5626 ô che, 2726 ép về 0, 2900 ô model; 0 dòng đầy đủ sau khi che.

| Model | status | coverage | mã lỗi | thông điệp | hội tụ (MICoDa) | MAE tỷ lệ (ô model) |
|---|---|---:|---|---|---|---:|
| hron | failed | 0.000 | baseline_error | BaselineError: row 0, feature 0: no donor observed on O_i and j | - | NA |
| tsagris | failed | 0.000 | baseline_error | BaselineError: Tsagris needs at least one complete donor row | - | NA |
| micoda | failed | 0.000 | insufficient_fit | insufficient_fit: Complete-case initialization needs more than one row; context={} | - | NA |
| same_day_mean | completed | 1.000 | - | - | - | 0.0217 |
| mean_fill | completed | 1.000 | - | - | - | 0.1238 |
| marginal_sampler | completed | 1.000 | - | - | - | 0.1243 |
| zero_fill | completed | 1.000 | - | - | - | 0.1682 |
| micoda_other | failed | 0.000 | insufficient_fit | insufficient_fit: Complete-case initialization needs more than one row; context={} | - | NA |

