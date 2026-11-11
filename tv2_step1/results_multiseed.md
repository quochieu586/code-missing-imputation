# Step 1: kết quả nhiều seed (COVID đầy đủ, che 5%, seed 0–4)

Nhóm `model` = ô bị che mà phần dư của dòng > 0 (evaluate.forced_zero). Sắp theo MAE tỷ lệ trung bình. mean ± sd (ddof=1) qua 5 seed: độ lệch chuẩn mẫu, không phải khoảng tin cậy. Nguồn: results_manifest.json.

## Mask `whole_feature` (chính)

Trạng thái: mọi lượt ok

### Nhóm ô model

| Model | ok/5 | MAE tỷ lệ | MSE tỷ lệ | MAE count | Số ô model (seed 0–4) |
|---|---:|---:|---:|---:|---|
| same_day_mean | 5 | 0.0351 ± 0.0066 | 0.0142 ± 0.0036 | 29.4 ± 26.8 | 2595/2729/2522/2504/2406 |
| hron | 5 | 0.0875 ± 0.0105 | 0.0660 ± 0.0092 | 74.6 ± 113.7 | 2595/2729/2522/2504/2406 |
| micoda_other | 5 | 0.1075 ± 0.0093 | 0.0684 ± 0.0078 | 188.7 ± 205.5 | 2595/2729/2522/2504/2406 |
| tsagris | 5 | 0.1079 ± 0.0209 | 0.0521 ± 0.0095 | 33.2 ± 14.7 | 2595/2729/2522/2504/2406 |
| mean_fill | 5 | 0.1354 ± 0.0145 | 0.0704 ± 0.0084 | 218.2 ± 186.8 | 2595/2729/2522/2504/2406 |
| marginal_sampler | 5 | 0.1354 ± 0.0120 | 0.1107 ± 0.0122 | 225.1 ± 200.7 | 2595/2729/2522/2504/2406 |
| micoda | 5 | 0.1406 ± 0.0099 | 0.0823 ± 0.0094 | 236.2 ± 188.5 | 2595/2729/2522/2504/2406 |
| zero_fill | 5 | 0.1693 ± 0.0219 | 0.1525 ± 0.0230 | 287.1 ± 211.3 | 2595/2729/2522/2504/2406 |

### MAE tỷ lệ theo đáp án

| Model | MAE tỷ lệ, đáp án = 0 (truth_zero) | đáp án = 0, ô model (model_truth_zero) | đáp án > 0 (truth_positive) | Số ô 0 / ô model 0 / ô > 0 (seed 0) |
|---|---:|---:|---:|---|
| hron | 0.0103 ± 0.0027 | 0.0294 ± 0.0065 | 0.1978 ± 0.0195 | 4633 / 1603 / 992 |
| tsagris | 0.0260 ± 0.0075 | 0.0745 ± 0.0214 | 0.1719 ± 0.0458 | 4633 / 1603 / 992 |
| same_day_mean | 0.0033 ± 0.0014 | 0.0094 ± 0.0040 | 0.0844 ± 0.0181 | 4633 / 1603 / 992 |

### same_day_mean: mức cửa sổ trên ô bị che

| Seed | Ô che (nhóm model): gốc / nới 1 / nới 2 / toàn cục | Mọi ô che: gốc / nới 1 / nới 2 / toàn cục |
|---|---|---|
| 0 | 2552 / 6 / 0 / 37 (toàn cục 1.4%) | 5450 / 41 / 23 / 111 (toàn cục 2.0%) |
| 1 | 2399 / 149 / 66 / 115 (toàn cục 4.2%) | 4945 / 328 / 141 / 291 (toàn cục 5.1%) |
| 2 | 2247 / 106 / 53 / 116 (toàn cục 4.6%) | 4832 / 341 / 157 / 307 (toàn cục 5.4%) |
| 3 | 2218 / 121 / 26 / 139 (toàn cục 5.6%) | 4925 / 237 / 99 / 435 (toàn cục 7.6%) |
| 4 | 2142 / 24 / 23 / 217 (toàn cục 9.0%) | 5012 / 62 / 77 / 495 (toàn cục 8.8%) |

Ô thiếu thật (lượt native): window 51136, widened_1 4912, widened_2 8056, global 21946; toàn cục 25.5% trên 86050 ô.

## Mask `random_cell` (tham khảo)

Trạng thái: mọi lượt ok

### Nhóm ô model

| Model | ok/5 | MAE tỷ lệ | MSE tỷ lệ | MAE count | Số ô model (seed 0–4) |
|---|---:|---:|---:|---:|---|
| same_day_mean | 5 | 0.0310 ± 0.0027 | 0.0110 ± 0.0015 | 19.0 ± 4.8 | 2851/2740/2762/2832/2750 |
| hron | 5 | 0.0892 ± 0.0028 | 0.0613 ± 0.0027 | 22.9 ± 4.0 | 2851/2740/2762/2832/2750 |
| tsagris | 5 | 0.0999 ± 0.0024 | 0.0482 ± 0.0017 | 26.6 ± 3.6 | 2851/2740/2762/2832/2750 |
| micoda_other | 5 | 0.1078 ± 0.0008 | 0.0695 ± 0.0013 | 109.6 ± 37.3 | 2851/2740/2762/2832/2750 |
| micoda | 5 | 0.1317 ± 0.0018 | 0.0803 ± 0.0024 | 167.7 ± 38.9 | 2851/2740/2762/2832/2750 |
| marginal_sampler | 5 | 0.1362 ± 0.0072 | 0.1112 ± 0.0069 | 143.4 ± 39.8 | 2851/2740/2762/2832/2750 |
| mean_fill | 5 | 0.1371 ± 0.0016 | 0.0692 ± 0.0007 | 152.4 ± 33.5 | 2851/2740/2762/2832/2750 |
| zero_fill | 5 | 0.1716 ± 0.0035 | 0.1551 ± 0.0031 | 226.8 ± 52.7 | 2851/2740/2762/2832/2750 |

### MAE tỷ lệ theo đáp án

| Model | MAE tỷ lệ, đáp án = 0 (truth_zero) | đáp án = 0, ô model (model_truth_zero) | đáp án > 0 (truth_positive) | Số ô 0 / ô model 0 / ô > 0 (seed 0) |
|---|---:|---:|---:|---|
| hron | 0.0123 ± 0.0008 | 0.0315 ± 0.0024 | 0.1943 ± 0.0061 | 4548 / 1820 / 1031 |
| tsagris | 0.0305 ± 0.0025 | 0.0781 ± 0.0071 | 0.1395 ± 0.0149 | 4548 / 1820 / 1031 |
| same_day_mean | 0.0036 ± 0.0002 | 0.0093 ± 0.0005 | 0.0704 ± 0.0077 | 4548 / 1820 / 1031 |

### same_day_mean: mức cửa sổ trên ô bị che

| Seed | Ô che (nhóm model): gốc / nới 1 / nới 2 / toàn cục | Mọi ô che: gốc / nới 1 / nới 2 / toàn cục |
|---|---|---|
| 0 | 2661 / 104 / 63 / 23 (toàn cục 0.8%) | 5146 / 212 / 165 / 56 (toàn cục 1.0%) |
| 1 | 2560 / 93 / 68 / 19 (toàn cục 0.7%) | 5109 / 219 / 165 / 41 (toàn cục 0.7%) |
| 2 | 2566 / 92 / 76 / 28 (toàn cục 1.0%) | 5193 / 195 / 162 / 54 (toàn cục 1.0%) |
| 3 | 2651 / 102 / 60 / 19 (toàn cục 0.7%) | 5246 / 213 / 166 / 48 (toàn cục 0.8%) |
| 4 | 2570 / 97 / 61 / 22 (toàn cục 0.8%) | 5126 / 226 / 158 / 36 (toàn cục 0.6%) |

Ô thiếu thật (lượt native): window 51136, widened_1 4912, widened_2 8056, global 21946; toàn cục 25.5% trên 86050 ô.

## Thắng/thua theo seed (ghép cặp, cùng mask; sinh từ results_multiseed.csv)

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
