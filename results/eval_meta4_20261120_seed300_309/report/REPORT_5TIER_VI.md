# Đánh giá 5 tầng: MainModel (metadata 4 trường) vs baseline — exploratory

Seed 300–309, tỷ lệ che 5%, 6 model: `hron`, `tsagris`, `micoda`, `same_day_mean`, `main`, `main_nometa`. Điểm tính bằng `longitudinal_impute.evaluation` (tier0, tier1, tier2), không viết lại metric. Seed đã được nhìn trong lúc phát triển, nên đây là exploratory, không phải confirmatory.

## Tầng 0 — hợp lệ

| Model | Mask | Lên lịch | Completed | Partial | Failed | Missing |
|---|---|---:|---:|---:|---:|---:|
| hron | whole_feature | 10 | 10 | 0 | 0 | 0 |
| tsagris | whole_feature | 10 | 10 | 0 | 0 | 0 |
| micoda | whole_feature | 10 | 10 | 0 | 0 | 0 |
| same_day_mean | whole_feature | 10 | 10 | 0 | 0 | 0 |
| main | whole_feature | 10 | 10 | 0 | 0 | 0 |
| main_nometa | whole_feature | 10 | 10 | 0 | 0 | 0 |
| hron | random_cell | 10 | 10 | 0 | 0 | 0 |
| tsagris | random_cell | 10 | 10 | 0 | 0 | 0 |
| micoda | random_cell | 10 | 10 | 0 | 0 | 0 |
| same_day_mean | random_cell | 10 | 10 | 0 | 0 | 0 |
| main | random_cell | 10 | 10 | 0 | 0 | 0 |
| main_nometa | random_cell | 10 | 10 | 0 | 0 | 0 |
| hron | whole_feature_matched | 10 | 10 | 0 | 0 | 0 |
| tsagris | whole_feature_matched | 10 | 10 | 0 | 0 | 0 |
| micoda | whole_feature_matched | 10 | 10 | 0 | 0 | 0 |
| same_day_mean | whole_feature_matched | 10 | 10 | 0 | 0 | 0 |
| main | whole_feature_matched | 10 | 10 | 0 | 0 | 0 |
| main_nometa | whole_feature_matched | 10 | 10 | 0 | 0 | 0 |

## Tầng 1 — che giả có đáp án

MAE / MSE của tỷ lệ c/N trên ô model quyết định (bỏ ô bị ép 0), mean ± sd qua seed chung. Thấp hơn là tốt hơn.

| Model | MAE whole_feature | MAE random_cell | MAE whole_feature_matched | MSE whole_feature | MSE random_cell | MSE whole_feature_matched |
|---|---:|---:|---:|---:|---:|---:|
| `hron` | 0.0897 ± 0.0130 | 0.0906 ± 0.0034 | 0.1147 ± 0.0172 | 0.0637 ± 0.0091 | 0.0632 ± 0.0024 | 0.0890 ± 0.0163 |
| `tsagris` | 0.1009 ± 0.0104 | 0.0993 ± 0.0030 | 0.1219 ± 0.0147 | 0.0484 ± 0.0048 | 0.0468 ± 0.0026 | 0.0590 ± 0.0090 |
| `micoda` | 0.1043 ± 0.0091 | 0.1069 ± 0.0025 | 0.1185 ± 0.0143 | 0.0665 ± 0.0068 | 0.0683 ± 0.0024 | 0.0775 ± 0.0124 |
| `same_day_mean` | 0.0311 ± 0.0057 | 0.0304 ± 0.0020 | 0.0364 ± 0.0058 | 0.0112 ± 0.0025 | 0.0107 ± 0.0013 | 0.0144 ± 0.0037 |
| `main` | 0.0471 ± 0.0101 | 0.0237 ± 0.0018 | 0.0581 ± 0.0136 | 0.0233 ± 0.0076 | 0.0100 ± 0.0014 | 0.0317 ± 0.0102 |
| `main_nometa` | 0.0516 ± 0.0140 | 0.0233 ± 0.0029 | 0.0644 ± 0.0202 | 0.0246 ± 0.0077 | 0.0098 ± 0.0023 | 0.0370 ± 0.0174 |

**So cặp** (Δ = A − B trung bình qua seed; thắng = seed A thấp hơn; sign test hai phía, mô tả biến thiên mask):

| A | B | Metric | whole_feature | random_cell | whole_feature_matched |
|---|---|---|---|---|---|
| `main` | `hron` | mae_prop | -0.0426, 9/10, p=0.0215 | -0.0670, 10/10, p=0.00195 | -0.0566, 10/10, p=0.00195 |
| `main` | `hron` | mse_prop | -0.0404, 10/10, p=0.00195 | -0.0532, 10/10, p=0.00195 | -0.0573, 10/10, p=0.00195 |
| `main` | `tsagris` | mae_prop | -0.0538, 10/10, p=0.00195 | -0.0757, 10/10, p=0.00195 | -0.0638, 10/10, p=0.00195 |
| `main` | `tsagris` | mse_prop | -0.0252, 10/10, p=0.00195 | -0.0368, 10/10, p=0.00195 | -0.0273, 10/10, p=0.00195 |
| `main` | `micoda` | mae_prop | -0.0573, 10/10, p=0.00195 | -0.0832, 10/10, p=0.00195 | -0.0604, 10/10, p=0.00195 |
| `main` | `micoda` | mse_prop | -0.0432, 10/10, p=0.00195 | -0.0583, 10/10, p=0.00195 | -0.0458, 10/10, p=0.00195 |
| `main` | `same_day_mean` | mae_prop | +0.0160, 0/10, p=0.00195 | -0.0067, 10/10, p=0.00195 | +0.0218, 0/10, p=0.00195 |
| `main` | `same_day_mean` | mse_prop | +0.0121, 0/10, p=0.00195 | -0.0007, 6/10, p=0.754 | +0.0173, 0/10, p=0.00195 |
| `main` | `main_nometa` | mae_prop | -0.0045, 6/10, p=0.754 | +0.0003, 4/10, p=0.754 | -0.0062, 4/10, p=0.754 |
| `main` | `main_nometa` | mse_prop | -0.0013, 6/10, p=0.754 | +0.0002, 4/10, p=0.754 | -0.0053, 3/10, p=0.344 |
| `main_nometa` | `same_day_mean` | mae_prop | +0.0205, 0/10, p=0.00195 | -0.0071, 10/10, p=0.00195 | +0.0280, 1/10, p=0.0215 |
| `main_nometa` | `same_day_mean` | mse_prop | +0.0134, 0/10, p=0.00195 | -0.0009, 8/10, p=0.109 | +0.0226, 0/10, p=0.00195 |

**MAE theo châu lục (whole_feature, mean qua seed)**:

| Model | Africa | Asia | Europe | North America | Oceania | South America |
|---|---:|---:|---:|---:|---:|---:|
| `hron` | 0.1658 | 0.1114 | 0.0572 | 0.0720 | 0.0879 | 0.0691 |
| `tsagris` | 0.1842 | 0.1294 | 0.0626 | 0.0887 | 0.1121 | 0.0848 |
| `micoda` | 0.1745 | 0.1247 | 0.0800 | 0.0942 | 0.1200 | 0.0724 |
| `same_day_mean` | 0.0711 | 0.0308 | 0.0209 | 0.0289 | 0.0699 | 0.0198 |
| `main` | 0.1048 | 0.0430 | 0.0282 | 0.0540 | 0.0575 | 0.0605 |
| `main_nometa` | 0.0870 | 0.0489 | 0.0326 | 0.0687 | 0.0510 | 0.0672 |

## Tầng 2 — ô thiếu thật (mô tả, không phải độ chính xác)

D1 = chênh tỷ lệ ô bằng 0 so với reference; D2 = Wasserstein W1 trên tỷ lệ c/N phần dương; D3 = ô điền > 0 ngoài khoảng thời gian có dữ liệu của biến thể. R-local là reference chính.

| Model | D1 local (đều / trọng số) | D2 local (đều / trọng số) | D1 pooled (đều / trọng số) | D2 pooled (đều / trọng số) | D3 ô | D3 khối lượng |
|---|---|---|---|---|---:|---:|
| hron | +0.054 / +0.044 | 0.0485 / 0.0433 | -0.009 / +0.001 | 0.0524 / 0.0393 | 102 | 6.947 |
| tsagris | -0.052 / -0.028 | 0.0646 / 0.0486 | -0.120 / -0.056 | 0.0910 / 0.0536 | 589 | 18.441 |
| micoda | -0.055 / -0.066 | 0.0652 / 0.0553 | -0.116 / -0.111 | 0.0985 / 0.0681 | 1259 | 24.028 |
| same_day_mean | +0.066 / +0.061 | 0.0565 / 0.0471 | +0.054 / +0.018 | 0.0782 / 0.0595 | 83 | 0.362 |
| main | -0.005 / -0.010 | 0.0591 / 0.0530 | +0.004 / -0.038 | 0.0694 / 0.0580 | 433 | 72.654 |
| main_nometa | -0.062 / -0.072 | 0.0598 / 0.0500 | -0.053 / -0.104 | 0.0768 / 0.0608 | 793 | 106.732 |

## Tầng 3 / Tầng 4

- Tầng 3 (U1–U4) kiểm định chính các chẩn đoán D1–D3 trên seed 0–4, không phụ thuộc model; không chạy lại.
- Tầng 4 (downstream) module chưa hiện thực.

