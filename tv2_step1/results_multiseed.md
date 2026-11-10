# Step 1: kết quả nhiều seed (COVID đầy đủ, che 5%, seed 0–4)

Nhóm `model` = ô bị che mà phần dư của dòng > 0 (evaluate.forced_zero). Sắp theo MAE tỷ lệ trung bình. mean ± sd (ddof=1) qua 5 seed.

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

### Thắng/thua theo seed (ghép cặp, cùng mask; thắng = sai số nhỏ hơn)

| Cặp (A vs B) | MAE tỷ lệ: A thắng | hiệu TB (A − B) | MSE tỷ lệ: A thắng | hiệu TB (A − B) |
|---|---:|---:|---:|---:|
| same_day_mean vs hron | 5/5 seed | -0.0525 | 5/5 seed | -0.0518 |
| hron vs tsagris | 5/5 seed | -0.0204 | 0/5 seed | +0.0140 |
| micoda_other vs mean_fill | 5/5 seed | -0.0278 | 4/5 seed | -0.0020 |
| micoda vs mean_fill | 1/5 seed | +0.0052 | 0/5 seed | +0.0119 |

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

### Thắng/thua theo seed (ghép cặp, cùng mask; thắng = sai số nhỏ hơn)

| Cặp (A vs B) | MAE tỷ lệ: A thắng | hiệu TB (A − B) | MSE tỷ lệ: A thắng | hiệu TB (A − B) |
|---|---:|---:|---:|---:|
| same_day_mean vs hron | 5/5 seed | -0.0582 | 5/5 seed | -0.0502 |
| hron vs tsagris | 5/5 seed | -0.0107 | 0/5 seed | +0.0130 |
| micoda_other vs mean_fill | 5/5 seed | -0.0293 | 1/5 seed | +0.0003 |
| micoda vs mean_fill | 5/5 seed | -0.0054 | 0/5 seed | +0.0111 |

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

