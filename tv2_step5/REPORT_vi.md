# Step 5: Tầng 2, diagnostic trên ô thiếu thật (native), chỉ baseline (exploratory)

Branch `step5-tier2-native` (từ `step4-residual-rule`). Không có `iterative_net` hay phần nào của `framework/`. Không sửa thuật toán baseline, không đổi mặc định. Mọi bảng số trong khối `auto:*` do `tier2.py` sinh, không sửa tay; bảng đầy đủ ở [tier2_results.md](tier2_results.md).

Tầng 2 chỉ **mô tả**: trên 86 050 ô thiếu thật không có đáp án, mọi số dưới đây đo mức giống một reference lấy từ ô quan sát. Không số nào là độ chính xác, không bảng nào là xếp hạng, và không có điểm tổng hợp.

## 1. File đã sửa/thêm

| File | Nội dung |
|---|---|
| `nextgen/src/longitudinal_impute/tier2_diagnostics.py` (277 dòng) | Hằng số và hàm D1/D2/D3, R-pooled, R-local, phân tầng, tổng hợp qua target, quy tắc bảng 6. `native_cells` và `calibration_cells` dựng hai mask E/V. Commit trước khi tính trên output nào. |
| `nextgen/tests/test_tier2_diagnostics.py` | 13 test mới. |
| `nextgen/README.md` | Một đoạn về diagnostic Tầng 2. |
| `tv2_step5/tier2.py` (408 dòng), `test_tier2.py` (3 test) | Kiểm hash, đọc output native, dựng bảng. Phần hash, `update_report` và CSV gọi `tier01`. |
| `tv2_step5/tier2_targets.csv`, `tier2_strata.csv`, `tier2_support.csv`, `tier2_results.md`, `tier2_manifest.json` | Output sinh tự động. |

Không chạy model mới: dùng lại CSV native của step 2 (`main_*`, `micoda_other_*`) và step 4 (`residual_*`).

## 2. Đặc tả từng diagnostic

Ký hiệu: M là ô thiếu thật; M_f ⊂ M là ô ép về 0 (dòng có tổng quan sát = N); M_m = M \ M_f; O là ô quan sát. Mọi hàm nhận hai mask tường minh: **E** (ô được mô tả; native là M_m) và **V** (ô reference; native là O). Hàm không tự đọc `~isfinite(counts)`. Đơn vị: tỷ lệ p = c/N của từng ô, tính theo từng target j. "Zero" là count điền **< 0.5** (dấu `<`, nên 0.5 là dương).

**Reference cho D1/D2:**
- **R-pooled:** mọi ô V của target j, trọng số đều (cùng tinh thần `wasserstein_scores`).
- **R-local (chính):** với mỗi ô e = (ℓ, t, j) ∈ E, S_e = các ô V của j ở location ℓ' ≠ ℓ cùng châu lục, với |ngày(t') − ngày(t)| ≤ 28. Ô e **có support** khi |S_e| ≥ 20 và S_e trải trên ≥ 3 location. Không nới cửa sổ, không fallback toàn cục. Reference của j là hỗn hợp

  R_j = (1/n_s) · Σ_{e có support} (1/|S_e|) · Σ_{r ∈ S_e} δ(p_r),

  tức mỗi ô r ∈ S_e mang trọng số 1/|S_e|, nên mỗi ô thiếu có support đóng góp tổng trọng số bằng 1. Một ô V có thể nằm trong nhiều S_e. D1/D2 bản R-local chỉ dùng các ô E có support.

| | D1: chênh tỷ lệ zero | D2: W1 phần dương | D3: cờ ngoài support thời gian |
|---|---|---|---|
| Định nghĩa | gap_j = Pr̂(zero \| E của j) − Pr_ref(zero \| reference của j), có dấu | W1(p của ô E dương, p của ô reference dương), reference có trọng số ở R-local | T_j = [ngày đầu, ngày cuối] có count ≥ 0.5 trong V, ở bất kỳ location nào, tính cả hai đầu. Đếm ô E điền ≥ 0.5 ở ngày ngoài T_j và tổng p của chúng |
| Reference | R-local (chính) và R-pooled | R-local (chính) và R-pooled | Ô V dương của j, toàn bảng |
| Miền ô | M_m (R-local: phần có support) | như D1 | M_m |
| Đơn vị | hiệu hai tỷ lệ, trong [−1, 1] | tỷ lệ c/N | số ô; tổng tỷ lệ |
| Support | coverage R-local theo target, châu lục, nhóm N; số location và số ô O trung bình | như D1, cộng số ô dương mỗi phía | số ô M_m ngoài T_j |
| NA | không có ô E (hoặc không ô nào có support) | một phía không có ô dương: NA, không phải 0 | target không có ô dương trong V: T_j rỗng, mọi ô điền dương đều bị cờ |
| Cách đọc bị cấm | "model sai/đúng", "model A tốt hơn B" | như D1; không cộng D1 với D2 | "sai chắc chắn", "biến thể chưa xuất hiện" |
| Quyết định hỗ trợ | đánh dấu model × target có dấu hiệu sụp về 0 (gap ≫ 0) hoặc điền dương tràn lan (gap ≪ 0) cần xem lại trước khi dùng bảng native | đánh dấu cỡ giá trị dương điền lệch thang so với quan sát lân cận | danh sách ô cần soát trước khi dùng cho phân tích phụ thuộc thời điểm xuất hiện |

**Phân tầng** (D1, D2 R-local, D3):
- theo châu lục;
- theo nhóm N: tam phân vị của N trên các dòng có ô M_m, với N1 ≤ q1 < N2 ≤ q2 < N3.

Với D1/D2, một ô stratum × target có dưới 30 ô có support thì báo NA. Với D3, cả tầng (mọi target) có dưới 30 ô M_m thì báo NA.

**Tổng hợp qua target:** báo cả trung bình đều và trung bình có trọng số theo số ô đã dùng (R-local: ô có support; R-pooled: ô M_m). Target NA bị bỏ và được đếm.

**Bảng 6:** "đổi dấu" khi tích hai giá trị < 0. "Thay đổi lớn" khi |D1 local − D1 pooled| ≥ 0.10; ngưỡng này được chốt trong code trước khi chạy.

**Commit:**
- `b284c3b` đóng băng đặc tả (hàm, hằng số, test), trước khi tính trên output nào;
- `580f15e` thêm script;
- `922fafe` sửa hai lỗi trình bày gặp khi chạy lần đầu, không đổi định nghĩa nào: đọc CSV completed (xem mục 8.8) và mẫu số của dòng đối thủ giả;
- `6e70a8a` commit số.

## 3. Kiểm tra

- **Test:** 306 pass với `-W error::RuntimeWarning`:
  - nextgen 282 (269 cũ, 13 mới);
  - script 24 (21 cũ, 3 mới).
- **Test mới kiểm:**
  - D1 và D2 khớp giá trị tính tay; W1 có trọng số bằng hàm `wasserstein` có sẵn khi trọng số đều và khi nhân bản mẫu;
  - D2 = NA khi một phía không có ô dương; tầng dưới 30 ô hoặc rỗng là NA;
  - R-local loại location của chính ô, loại châu lục khác và ô cách 29 ngày (28 ngày vẫn được dùng); ngưỡng 20 ô / 3 location; tổng trọng số mỗi ô = 1; D1 R-local bằng trung bình các tỷ lệ zero cục bộ;
  - helper calibration:
    - đổi giá trị trong E thì Pr_ref, số ô dương reference và T_j không đổi;
    - ô ngoài E ∪ V (kể cả NaN hay 1e6) không đổi kết quả từng byte;
    - NaN trong E hoặc V thì báo lỗi;
    - E, V chồng nhau thì báo lỗi;
  - ô forced-zero không vào M_m, ở cả native lẫn calibration;
  - T_j đúng, ngày bằng đầu/cuối nằm trong support;
  - cùng input cho kết quả giống từng byte;
  - script dừng khi bảng model trong config lệch file toml, hoặc CSV native ở thư mục random_cell khác byte.
- **Chạy lại:** `tier2.py` chạy hai lần (hai thư mục output) cho 5 file output giống từng byte.

Kiểm hash trước khi tái dùng (lệch thì script dừng):
- data sha256 và schema;
- group map sha256 và số dòng resolve được;
- bảng tham số của model so với file toml hiện tại;
- git sạch lúc chạy;
- trạng thái native `ok`;
- CSV native giống từng byte ở thư mục random_cell (lượt native không phụ thuộc mask);
- ô quan sát không đổi (`check_completed`).

Ngoài ra, tôi chạy lại lượt native của cả 11 model bằng code hiện tại (`run.py --mask none`, thư mục tạm) và so: cả 11 CSV giống từng byte.

<!-- auto:check:start -->
data sha256 `dbdce6030721`, schema `0d5745559671`, group map `5e0547a924b2`, tier2_diagnostics.py `3750c72202ef`.

| Model | Thư mục run | commit của run | CSV native sha256 | cùng byte ở random_cell | chạy lại = cùng byte |
|---|---|---|---|:---:|:---:|
| hron | `tv2_step2/runs/main_whole_feature` | `7ef9146` | `81ed55c426c0` | True | True |
| tsagris | `tv2_step2/runs/main_whole_feature` | `7ef9146` | `ab0e46fbf46d` | True | True |
| micoda | `tv2_step2/runs/main_whole_feature` | `7ef9146` | `8d5ac70b1fc9` | True | True |
| micoda_other | `tv2_step2/runs/micoda_other_whole_feature` | `8978d72` | `f4d2261fa433` | True | True |
| same_day_mean | `tv2_step2/runs/main_whole_feature` | `7ef9146` | `e9f684d65368` | True | True |
| mean_fill | `tv2_step2/runs/main_whole_feature` | `7ef9146` | `b881cbc1b787` | True | True |
| zero_fill | `tv2_step2/runs/main_whole_feature` | `7ef9146` | `0f48ede78260` | True | True |
| marginal_sampler | `tv2_step2/runs/main_whole_feature` | `7ef9146` | `735261e2cf93` | True | True |
| hron_cap | `tv2_step4/runs/residual_whole_feature` | `fefd4d0` | `8b7f251db1c3` | True | True |
| tsagris_cap | `tv2_step4/runs/residual_whole_feature` | `fefd4d0` | `1a82bcb4bc4c` | True | True |
| micoda_other_cap | `tv2_step4/runs/residual_whole_feature` | `fefd4d0` | `7976a9e4a63b` | True | True |
<!-- auto:check:end -->

## 4. Bảng

### Bảng 1: support

<!-- auto:support:start -->
Ô thiếu thật M = 86050; ép về 0 M_f = 60595 (70.42%); model quyết định M_m = 25455 (29.58%); ô quan sát O = 112357. R-local có support: 14185 / 25455 ô M_m (coverage 55.73%). Ngưỡng nhóm N (tam phân vị trên dòng có ô M_m): N1 ≤ 17 < N2 ≤ 149 < N3.

| Target | ô O | ô M_m | có support | coverage | TB ô O trong reference | TB location | T_j (ngày đầu – cuối có count dương) | ô M_m ngoài T_j |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| recombinant | 9713 | 660 | 626 | 0.948 | 64.4 | 14.8 | 2020-07-06 – 2024-07-29 | 90 |
| 20A | 7661 | 1152 | 978 | 0.849 | 60.9 | 13.7 | 2020-04-27 – 2023-04-10 | 3 |
| 20B | 4545 | 2137 | 1287 | 0.602 | 49.3 | 10.5 | 2020-07-06 – 2022-07-18 | 300 |
| 20C | 2220 | 2870 | 519 | 0.181 | 82.2 | 16.9 | 2020-07-06 – 2021-09-13 | 591 |
| 20E | 7598 | 1147 | 1083 | 0.944 | 58.7 | 13.2 | 2020-04-27 – 2022-09-26 | 8 |
| Beta | 8605 | 752 | 706 | 0.939 | 79.7 | 18.1 | 2020-08-03 – 2023-02-13 | 139 |
| Alpha | 11325 | 68 | 39 | 0.574 | 73.1 | 17.7 | 2020-09-14 – 2022-12-05 | 20 |
| Gamma | 6732 | 1435 | 650 | 0.453 | 79.8 | 17.3 | 2020-10-12 – 2023-04-10 | 463 |
| Delta | 11656 | 0 | 0 | NA | NA | NA | 2020-10-12 – 2024-04-08 | 0 |
| Kappa | 4650 | 2085 | 1591 | 0.763 | 57.7 | 12.5 | 2020-11-23 – 2022-04-25 | 841 |
| Epsilon | 4087 | 2322 | 1527 | 0.658 | 47.9 | 10.0 | 2020-08-03 – 2022-09-26 | 429 |
| Eta | 6778 | 1247 | 750 | 0.601 | 86.5 | 19.4 | 2020-12-07 – 2022-12-05 | 550 |
| Iota | 4168 | 2233 | 1430 | 0.640 | 47.1 | 9.8 | 2020-11-09 – 2022-10-10 | 840 |
| Lambda | 3915 | 2297 | 660 | 0.287 | 74.0 | 15.4 | 2020-10-26 – 2022-05-23 | 823 |
| Mu | 4751 | 2067 | 1199 | 0.580 | 53.3 | 11.4 | 2021-01-04 – 2024-05-06 | 1005 |
| Omicron | 11658 | 4 | 0 | 0.000 | NA | NA | 2021-08-30 – 2024-07-29 | 4 |
| S:677 | 2295 | 2979 | 1140 | 0.383 | 30.9 | 6.5 | 2020-07-20 – 2022-09-26 | 467 |

| Tầng | ô M_m | có support | coverage | số target có ≥ 30 ô có support |
|---|---:|---:|---:|---:|
| Africa | 8082 | 2070 | 0.256 | 6 |
| Asia | 6962 | 4519 | 0.649 | 11 |
| Europe | 5020 | 4996 | 0.995 | 13 |
| North America | 2151 | 1526 | 0.709 | 12 |
| Oceania | 742 | 0 | 0.000 | 0 |
| South America | 2498 | 1074 | 0.430 | 8 |
| N1 | 11583 | 6334 | 0.547 | 15 |
| N2 | 8836 | 4643 | 0.525 | 14 |
| N3 | 5036 | 3208 | 0.637 | 13 |
<!-- auto:support:end -->

### Bảng 2: D1 (model × target)

<!-- auto:d1:start -->
**D1 R-local (chính)** = Pr̂(zero, ô M_m) − Pr_ref(zero). Dòng đầu: Pr_ref(zero) của reference.

| Model | recombinant | 20A | 20B | 20C | 20E | Beta | Alpha | Gamma | Delta | Kappa | Epsilon | Eta | Iota | Lambda | Mu | Omicron | S:677 | TB đều | TB trọng số (số ô) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| *Pr_ref(zero)* | 0.992 | 0.820 | 0.833 | 0.804 | 0.840 | 0.811 | 0.799 | 0.871 | NA | 0.910 | 0.873 | 0.785 | 0.896 | 0.878 | 0.927 | NA | 0.937 | | |
| hron | -0.040 | +0.068 | +0.073 | +0.129 | +0.077 | +0.075 | +0.124 | +0.015 | NA | +0.041 | +0.022 | +0.129 | +0.039 | +0.088 | -0.023 | NA | -0.010 | +0.054 | +0.044 (15 target) |
| tsagris | -0.149 | -0.197 | +0.004 | +0.164 | -0.236 | +0.115 | -0.517 | -0.135 | NA | +0.031 | -0.058 | +0.113 | +0.016 | +0.106 | -0.050 | NA | +0.008 | -0.052 | -0.028 (15 target) |
| micoda | -0.508 | -0.173 | -0.324 | -0.624 | -0.186 | -0.038 | -0.004 | -0.223 | NA | -0.255 | -0.391 | +0.004 | -0.358 | -0.394 | -0.425 | NA | -0.828 | -0.315 | -0.340 (15 target) |
| micoda_other | -0.359 | -0.022 | -0.028 | -0.029 | -0.016 | +0.090 | +0.073 | -0.071 | NA | -0.023 | -0.081 | +0.163 | -0.071 | -0.053 | -0.160 | NA | -0.236 | -0.055 | -0.066 (15 target) |
| same_day_mean | +0.006 | +0.095 | +0.093 | +0.139 | +0.110 | +0.027 | +0.098 | +0.074 | NA | +0.005 | +0.083 | +0.071 | +0.042 | +0.070 | +0.022 | NA | +0.054 | +0.066 | +0.061 (15 target) |
| mean_fill | -0.117 | -0.169 | -0.019 | +0.009 | -0.126 | -0.029 | -0.132 | -0.171 | NA | +0.050 | -0.100 | +0.088 | -0.018 | -0.172 | -0.233 | NA | -0.304 | -0.096 | -0.092 (15 target) |
| zero_fill | +0.008 | +0.180 | +0.167 | +0.196 | +0.160 | +0.189 | +0.201 | +0.129 | NA | +0.090 | +0.127 | +0.215 | +0.104 | +0.122 | +0.073 | NA | +0.063 | +0.135 | +0.126 (15 target) |
| marginal_sampler | -0.083 | +0.099 | +0.133 | +0.177 | +0.114 | +0.143 | +0.150 | +0.077 | NA | +0.080 | +0.108 | +0.177 | +0.090 | +0.087 | +0.034 | NA | +0.047 | +0.096 | +0.091 (15 target) |
| hron_cap | -0.002 | +0.105 | +0.136 | +0.154 | +0.090 | +0.128 | +0.098 | +0.057 | NA | +0.081 | +0.085 | +0.171 | +0.093 | +0.110 | +0.053 | NA | +0.044 | +0.094 | +0.091 (15 target) |
| tsagris_cap | -0.149 | -0.197 | +0.004 | +0.164 | -0.236 | +0.115 | -0.517 | -0.135 | NA | +0.031 | -0.058 | +0.113 | +0.016 | +0.106 | -0.050 | NA | +0.008 | -0.052 | -0.028 (15 target) |
| micoda_other_cap | -0.359 | -0.022 | -0.028 | -0.029 | -0.016 | +0.090 | +0.073 | -0.071 | NA | -0.023 | -0.081 | +0.163 | -0.071 | -0.053 | -0.160 | NA | -0.236 | -0.055 | -0.066 (15 target) |

**D1 R-pooled** = Pr̂(zero, ô M_m) − Pr_ref(zero). Dòng đầu: Pr_ref(zero) của reference.

| Model | recombinant | 20A | 20B | 20C | 20E | Beta | Alpha | Gamma | Delta | Kappa | Epsilon | Eta | Iota | Lambda | Mu | Omicron | S:677 | TB đều | TB trọng số (số ô) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| *Pr_ref(zero)* | 0.747 | 0.881 | 0.920 | 0.950 | 0.895 | 0.889 | 0.833 | 0.882 | NA | 0.961 | 0.939 | 0.928 | 0.952 | 0.927 | 0.919 | 0.418 | 0.959 | | |
| hron | +0.208 | +0.013 | +0.005 | -0.002 | +0.026 | +0.004 | +0.079 | +0.003 | NA | -0.011 | -0.031 | -0.001 | -0.011 | +0.011 | -0.004 | -0.418 | -0.021 | -0.009 | +0.001 (16 target) |
| tsagris | +0.088 | -0.288 | -0.078 | +0.005 | -0.306 | +0.037 | -0.612 | -0.150 | NA | -0.041 | -0.105 | -0.055 | -0.028 | +0.042 | -0.009 | -0.418 | -0.010 | -0.120 | -0.056 (16 target) |
| micoda | -0.262 | -0.260 | -0.385 | -0.779 | -0.247 | -0.122 | -0.142 | -0.250 | NA | -0.348 | -0.424 | -0.201 | -0.383 | -0.426 | -0.386 | -0.418 | -0.787 | -0.364 | -0.440 (16 target) |
| micoda_other | -0.115 | -0.104 | -0.092 | -0.168 | -0.076 | +0.002 | -0.068 | -0.073 | NA | -0.103 | -0.111 | -0.028 | -0.098 | -0.103 | -0.117 | -0.418 | -0.184 | -0.116 | -0.111 (16 target) |
| same_day_mean | +0.252 | +0.043 | +0.021 | +0.026 | +0.052 | -0.050 | -0.083 | -0.011 | NA | -0.045 | -0.005 | -0.016 | -0.013 | +0.046 | +0.035 | +0.582 | +0.031 | +0.054 | +0.018 (16 target) |
| mean_fill | +0.126 | -0.227 | -0.065 | -0.065 | -0.180 | -0.113 | -0.245 | -0.200 | NA | -0.006 | -0.105 | -0.083 | -0.040 | -0.142 | -0.174 | -0.418 | -0.139 | -0.130 | -0.104 (16 target) |
| zero_fill | +0.253 | +0.119 | +0.080 | +0.050 | +0.105 | +0.111 | +0.167 | +0.118 | NA | +0.039 | +0.061 | +0.072 | +0.048 | +0.073 | +0.081 | +0.582 | +0.041 | +0.125 | +0.074 (16 target) |
| marginal_sampler | +0.163 | +0.039 | +0.049 | +0.031 | +0.059 | +0.063 | +0.064 | +0.066 | NA | +0.029 | +0.042 | +0.042 | +0.033 | +0.044 | +0.043 | -0.418 | +0.028 | +0.023 | +0.044 (16 target) |
| hron_cap | +0.244 | +0.046 | +0.053 | +0.038 | +0.039 | +0.053 | +0.064 | +0.056 | NA | +0.031 | +0.021 | +0.034 | +0.037 | +0.059 | +0.060 | -0.418 | +0.023 | +0.028 | +0.046 (16 target) |
| tsagris_cap | +0.088 | -0.288 | -0.078 | +0.005 | -0.306 | +0.037 | -0.612 | -0.150 | NA | -0.041 | -0.105 | -0.055 | -0.028 | +0.042 | -0.009 | -0.418 | -0.010 | -0.120 | -0.056 (16 target) |
| micoda_other_cap | -0.115 | -0.104 | -0.092 | -0.168 | -0.076 | +0.002 | -0.068 | -0.073 | NA | -0.103 | -0.111 | -0.028 | -0.098 | -0.103 | -0.117 | -0.418 | -0.184 | -0.116 | -0.111 (16 target) |
<!-- auto:d1:end -->

### Bảng 3: D2 (model × target)

<!-- auto:d2:start -->
**D2 R-local (chính)** = W1 trên tỷ lệ c/N, phần dương (count ≥ 0.5). NA = một phía không có ô dương.

| Model | recombinant | 20A | 20B | 20C | 20E | Beta | Alpha | Gamma | Delta | Kappa | Epsilon | Eta | Iota | Lambda | Mu | Omicron | S:677 | TB đều | TB trọng số (số ô) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hron | 0.0608 | 0.0482 | 0.0316 | 0.0144 | 0.1212 | 0.0840 | 0.1035 | 0.0634 | NA | 0.0351 | 0.0510 | 0.0321 | 0.0132 | 0.0253 | 0.0218 | NA | 0.0226 | 0.0485 | 0.0433 (15 target) |
| tsagris | 0.0746 | 0.1024 | 0.0291 | 0.0241 | 0.1118 | 0.1688 | 0.1998 | 0.0608 | NA | 0.0067 | 0.0196 | 0.0458 | 0.0209 | 0.0444 | 0.0326 | NA | 0.0283 | 0.0646 | 0.0486 (15 target) |
| micoda | 0.0986 | 0.0891 | 0.0563 | 0.1510 | 0.0741 | 0.1797 | 0.1261 | 0.0419 | NA | 0.0308 | 0.0480 | 0.0692 | 0.0320 | 0.0495 | 0.0548 | NA | 0.1529 | 0.0836 | 0.0727 (15 target) |
| micoda_other | 0.0833 | 0.0978 | 0.0498 | 0.0223 | 0.0865 | 0.1906 | 0.1346 | 0.0287 | NA | 0.0256 | 0.0243 | 0.0714 | 0.0296 | 0.0445 | 0.0598 | NA | 0.0295 | 0.0652 | 0.0553 (15 target) |
| same_day_mean | 0.0710 | 0.0878 | 0.0404 | 0.0199 | 0.0408 | 0.1702 | 0.1258 | 0.0194 | NA | 0.0186 | 0.0253 | 0.0559 | 0.0203 | 0.0461 | 0.0574 | NA | 0.0478 | 0.0565 | 0.0471 (15 target) |
| mean_fill | 0.0715 | 0.0990 | 0.0509 | 0.0284 | 0.0883 | 0.1912 | 0.1314 | 0.0275 | NA | 0.0274 | 0.0268 | 0.0733 | 0.0323 | 0.0476 | 0.0614 | NA | 0.0330 | 0.0660 | 0.0567 (15 target) |
| zero_fill | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA (0 target) |
| marginal_sampler | 0.0616 | 0.1128 | 0.1437 | 0.0167 | 0.0801 | 0.1193 | 0.2047 | 0.0932 | NA | 0.0472 | 0.0360 | 0.1985 | 0.0111 | 0.0406 | 0.0960 | NA | 0.0199 | 0.0854 | 0.0734 (15 target) |
| hron_cap | 0.0943 | 0.0611 | 0.0485 | 0.0158 | 0.1171 | 0.1346 | 0.0884 | 0.0251 | NA | 0.0435 | 0.0401 | 0.0544 | 0.0170 | 0.0810 | 0.0371 | NA | 0.0284 | 0.0591 | 0.0534 (15 target) |
| tsagris_cap | 0.0746 | 0.1024 | 0.0291 | 0.0241 | 0.1118 | 0.1688 | 0.1998 | 0.0608 | NA | 0.0067 | 0.0196 | 0.0458 | 0.0209 | 0.0444 | 0.0326 | NA | 0.0283 | 0.0646 | 0.0486 (15 target) |
| micoda_other_cap | 0.0833 | 0.0978 | 0.0498 | 0.0223 | 0.0865 | 0.1906 | 0.1346 | 0.0287 | NA | 0.0256 | 0.0243 | 0.0714 | 0.0296 | 0.0445 | 0.0598 | NA | 0.0295 | 0.0652 | 0.0553 (15 target) |

**D2 R-pooled** = W1 trên tỷ lệ c/N, phần dương (count ≥ 0.5). NA = một phía không có ô dương.

| Model | recombinant | 20A | 20B | 20C | 20E | Beta | Alpha | Gamma | Delta | Kappa | Epsilon | Eta | Iota | Lambda | Mu | Omicron | S:677 | TB đều | TB trọng số (số ô) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hron | 0.0357 | 0.0784 | 0.0295 | 0.0086 | 0.0716 | 0.0652 | 0.1751 | 0.0568 | NA | 0.0285 | 0.0584 | 0.0392 | 0.0471 | 0.0303 | 0.0363 | 0.0506 | 0.0271 | 0.0524 | 0.0393 (16 target) |
| tsagris | 0.0324 | 0.1451 | 0.0355 | 0.0235 | 0.1365 | 0.1423 | 0.2814 | 0.0982 | NA | 0.0086 | 0.0279 | 0.0634 | 0.0198 | 0.0628 | 0.0652 | 0.2726 | 0.0413 | 0.0910 | 0.0536 (16 target) |
| micoda | 0.0653 | 0.1453 | 0.0588 | 0.2051 | 0.1279 | 0.1532 | 0.3254 | 0.1419 | NA | 0.0290 | 0.0386 | 0.0815 | 0.0282 | 0.0673 | 0.0775 | 0.0845 | 0.1551 | 0.1115 | 0.0984 (16 target) |
| micoda_other | 0.0437 | 0.1668 | 0.0556 | 0.0222 | 0.1480 | 0.1632 | 0.3346 | 0.1493 | NA | 0.0212 | 0.0373 | 0.0883 | 0.0290 | 0.0742 | 0.0921 | 0.1058 | 0.0453 | 0.0985 | 0.0681 (16 target) |
| same_day_mean | 0.0331 | 0.1599 | 0.0486 | 0.0236 | 0.0909 | 0.1598 | 0.1777 | 0.1528 | NA | 0.0128 | 0.0301 | 0.0679 | 0.0194 | 0.0610 | 0.0947 | NA | 0.0405 | 0.0782 | 0.0595 (15 target) |
| mean_fill | 0.0315 | 0.1677 | 0.0575 | 0.0280 | 0.1498 | 0.1638 | 0.3247 | 0.1508 | NA | 0.0215 | 0.0413 | 0.0901 | 0.0313 | 0.0772 | 0.0952 | 0.4274 | 0.0485 | 0.1191 | 0.0705 (16 target) |
| zero_fill | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA (0 target) |
| marginal_sampler | 0.0369 | 0.0556 | 0.1100 | 0.0150 | 0.0369 | 0.1885 | 0.1663 | 0.0610 | NA | 0.0573 | 0.0434 | 0.1440 | 0.0372 | 0.0537 | 0.0528 | 0.0918 | 0.0276 | 0.0736 | 0.0569 (16 target) |
| hron_cap | 0.0642 | 0.0789 | 0.0342 | 0.0140 | 0.0681 | 0.1077 | 0.2111 | 0.0846 | NA | 0.0421 | 0.0281 | 0.0225 | 0.0125 | 0.0217 | 0.0624 | 0.0506 | 0.0318 | 0.0584 | 0.0402 (16 target) |
| tsagris_cap | 0.0324 | 0.1451 | 0.0355 | 0.0235 | 0.1365 | 0.1423 | 0.2814 | 0.0982 | NA | 0.0086 | 0.0279 | 0.0634 | 0.0198 | 0.0628 | 0.0652 | 0.2726 | 0.0413 | 0.0910 | 0.0536 (16 target) |
| micoda_other_cap | 0.0437 | 0.1668 | 0.0556 | 0.0222 | 0.1480 | 0.1632 | 0.3346 | 0.1493 | NA | 0.0212 | 0.0373 | 0.0883 | 0.0290 | 0.0742 | 0.0921 | 0.1058 | 0.0453 | 0.0985 | 0.0681 (16 target) |
<!-- auto:d2:end -->

### Bảng 4: D3 (model × target)

<!-- auto:d3:start -->
Số ô M_m được điền dương (≥ 0.5 count) ở ngày ngoài T_j; dòng đầu: số ô M_m ngoài T_j (mọi model).

| Model | recombinant | 20A | 20B | 20C | 20E | Beta | Alpha | Gamma | Delta | Kappa | Epsilon | Eta | Iota | Lambda | Mu | Omicron | S:677 | tổng ô | tổng khối lượng tỷ lệ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| *ô M_m ngoài T_j* | 90 | 3 | 300 | 591 | 8 | 139 | 20 | 463 | 0 | 841 | 429 | 550 | 840 | 823 | 1005 | 4 | 467 | 6573 | |
| hron | 0 | 0 | 5 | 46 | 0 | 0 | 0 | 2 | 0 | 7 | 4 | 2 | 2 | 10 | 13 | 4 | 7 | 102 | 6.947 |
| tsagris | 32 | 0 | 56 | 58 | 0 | 2 | 19 | 120 | 0 | 14 | 109 | 14 | 70 | 10 | 37 | 4 | 44 | 589 | 18.441 |
| micoda | 52 | 0 | 168 | 462 | 0 | 32 | 7 | 200 | 0 | 331 | 236 | 151 | 391 | 441 | 537 | 4 | 390 | 3402 | 453.734 |
| micoda_other | 46 | 0 | 73 | 108 | 0 | 15 | 6 | 111 | 0 | 113 | 87 | 52 | 129 | 150 | 245 | 4 | 120 | 1259 | 24.028 |
| same_day_mean | 0 | 0 | 1 | 8 | 0 | 1 | 0 | 26 | 0 | 8 | 13 | 0 | 14 | 0 | 10 | 0 | 2 | 83 | 0.362 |
| mean_fill | 9 | 0 | 33 | 48 | 0 | 18 | 7 | 93 | 0 | 11 | 20 | 37 | 27 | 97 | 177 | 4 | 14 | 595 | 7.263 |
| zero_fill | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.000 |
| marginal_sampler | 2 | 0 | 7 | 13 | 0 | 10 | 4 | 21 | 0 | 7 | 4 | 13 | 10 | 18 | 37 | 4 | 1 | 151 | 36.690 |
| hron_cap | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 4 | 0 | 8 | 4.508 |
| tsagris_cap | 32 | 0 | 56 | 58 | 0 | 2 | 19 | 120 | 0 | 14 | 109 | 14 | 70 | 10 | 37 | 4 | 44 | 589 | 18.441 |
| micoda_other_cap | 46 | 0 | 73 | 108 | 0 | 15 | 6 | 111 | 0 | 113 | 87 | 52 | 129 | 150 | 245 | 4 | 120 | 1259 | 24.027 |
<!-- auto:d3:end -->

### Bảng 5: phân tầng (R-local)

<!-- auto:strata:start -->
**D1 R-local theo châu lục**: TB đều / TB trọng số qua các target có ≥ 30 ô có support (số target trong ngoặc).

| Model | Africa | Asia | Europe | North America | Oceania | South America |
|---|---:|---:|---:|---:|---:|---:|
| hron | +0.059 / +0.030 (6) | +0.004 / -0.010 (11) | +0.099 / +0.061 (13) | +0.160 / +0.173 (12) | NA / NA (0) | +0.043 / +0.016 (8) |
| tsagris | -0.036 / -0.083 (6) | -0.101 / -0.111 (11) | +0.062 / +0.040 (13) | +0.119 / +0.105 (12) | NA / NA (0) | -0.027 / -0.070 (8) |
| micoda | -0.164 / -0.221 (6) | -0.277 / -0.326 (11) | -0.278 / -0.464 (13) | -0.167 / -0.175 (12) | NA / NA (0) | -0.298 / -0.322 (8) |
| micoda_other | -0.012 / -0.060 (6) | -0.094 / -0.107 (11) | -0.000 / -0.097 (13) | +0.133 / +0.142 (12) | NA / NA (0) | -0.051 / -0.070 (8) |
| same_day_mean | -0.051 / -0.016 (6) | +0.049 / +0.048 (11) | +0.078 / +0.070 (13) | +0.147 / +0.181 (12) | NA / NA (0) | +0.033 / +0.040 (8) |
| mean_fill | -0.007 / -0.034 (6) | -0.083 / -0.097 (11) | -0.065 / -0.165 (13) | +0.114 / +0.112 (12) | NA / NA (0) | -0.138 / -0.154 (8) |
| zero_fill | +0.136 / +0.103 (6) | +0.096 / +0.088 (11) | +0.160 / +0.121 (13) | +0.242 / +0.257 (12) | NA / NA (0) | +0.157 / +0.151 (8) |
| marginal_sampler | +0.081 / +0.048 (6) | +0.060 / +0.057 (11) | +0.125 / +0.092 (13) | +0.209 / +0.225 (12) | NA / NA (0) | +0.114 / +0.110 (8) |
| hron_cap | +0.093 / +0.065 (6) | +0.054 / +0.053 (11) | +0.126 / +0.097 (13) | +0.202 / +0.219 (12) | NA / NA (0) | +0.092 / +0.083 (8) |
| tsagris_cap | -0.036 / -0.083 (6) | -0.101 / -0.111 (11) | +0.062 / +0.040 (13) | +0.119 / +0.105 (12) | NA / NA (0) | -0.027 / -0.070 (8) |
| micoda_other_cap | -0.012 / -0.060 (6) | -0.094 / -0.107 (11) | -0.000 / -0.097 (13) | +0.133 / +0.142 (12) | NA / NA (0) | -0.051 / -0.070 (8) |

**D2 R-local theo châu lục**: TB đều / TB trọng số qua các target có ≥ 30 ô có support (số target trong ngoặc).

| Model | Africa | Asia | Europe | North America | Oceania | South America |
|---|---:|---:|---:|---:|---:|---:|
| hron | 0.101 / 0.076 (6) | 0.086 / 0.067 (11) | 0.050 / 0.027 (12) | 0.071 / 0.072 (12) | NA / NA (0) | 0.112 / 0.090 (8) |
| tsagris | 0.160 / 0.111 (6) | 0.053 / 0.043 (11) | 0.039 / 0.021 (13) | 0.070 / 0.069 (12) | NA / NA (0) | 0.079 / 0.060 (7) |
| micoda | 0.184 / 0.133 (6) | 0.058 / 0.051 (11) | 0.071 / 0.083 (13) | 0.068 / 0.067 (12) | NA / NA (0) | 0.083 / 0.063 (8) |
| micoda_other | 0.194 / 0.140 (6) | 0.042 / 0.033 (11) | 0.030 / 0.013 (13) | 0.054 / 0.045 (12) | NA / NA (0) | 0.087 / 0.060 (8) |
| same_day_mean | 0.161 / 0.136 (5) | 0.040 / 0.030 (9) | 0.024 / 0.013 (10) | 0.067 / 0.056 (9) | NA / NA (0) | 0.105 / 0.067 (4) |
| mean_fill | 0.218 / 0.164 (5) | 0.038 / 0.031 (11) | 0.026 / 0.011 (13) | 0.063 / 0.053 (8) | NA / NA (0) | 0.091 / 0.065 (8) |
| zero_fill | NA / NA (0) | NA / NA (0) | NA / NA (0) | NA / NA (0) | NA / NA (0) | NA / NA (0) |
| marginal_sampler | 0.051 / 0.048 (6) | 0.134 / 0.119 (11) | 0.104 / 0.068 (13) | 0.157 / 0.175 (10) | NA / NA (0) | 0.139 / 0.118 (8) |
| hron_cap | 0.126 / 0.098 (6) | 0.083 / 0.059 (10) | 0.027 / 0.013 (12) | 0.080 / 0.085 (10) | NA / NA (0) | 0.121 / 0.095 (8) |
| tsagris_cap | 0.160 / 0.111 (6) | 0.053 / 0.043 (11) | 0.039 / 0.021 (13) | 0.070 / 0.069 (12) | NA / NA (0) | 0.079 / 0.060 (7) |
| micoda_other_cap | 0.194 / 0.140 (6) | 0.042 / 0.033 (11) | 0.030 / 0.013 (13) | 0.054 / 0.045 (12) | NA / NA (0) | 0.087 / 0.060 (8) |

**D3 theo châu lục**: số ô cờ / khối lượng tỷ lệ (NA khi tầng có < 30 ô M_m).

| Model | Africa | Asia | Europe | North America | Oceania | South America |
|---|---:|---:|---:|---:|---:|---:|
| hron | 16 / 0.995 | 50 / 1.639 | 14 / 0.047 | 1 / 0.001 | 5 / 4.001 | 16 / 0.264 |
| tsagris | 184 / 6.015 | 220 / 3.721 | 76 / 1.269 | 23 / 0.188 | 18 / 4.220 | 68 / 3.027 |
| micoda | 990 / 94.405 | 1031 / 117.515 | 778 / 148.076 | 154 / 27.830 | 86 / 16.083 | 363 / 49.826 |
| micoda_other | 337 / 6.750 | 409 / 6.655 | 271 / 3.502 | 52 / 0.826 | 40 / 4.287 | 150 / 2.007 |
| same_day_mean | 61 / 0.350 | 5 / 0.001 | 0 / 0.000 | 0 / 0.000 | 6 / 0.007 | 11 / 0.003 |
| mean_fill | 155 / 1.898 | 205 / 1.438 | 126 / 0.869 | 20 / 0.149 | 28 / 2.531 | 61 / 0.379 |
| zero_fill | 0 / 0.000 | 0 / 0.000 | 0 / 0.000 | 0 / 0.000 | 0 / 0.000 | 0 / 0.000 |
| marginal_sampler | 46 / 11.873 | 45 / 8.124 | 26 / 5.364 | 6 / 0.740 | 11 / 6.674 | 17 / 3.916 |
| hron_cap | 1 / 0.001 | 2 / 0.477 | 1 / 0.030 | 0 / 0.000 | 4 / 4.000 | 0 / 0.000 |
| tsagris_cap | 184 / 6.015 | 220 / 3.721 | 76 / 1.269 | 23 / 0.188 | 18 / 4.220 | 68 / 3.027 |
| micoda_other_cap | 337 / 6.750 | 409 / 6.655 | 271 / 3.502 | 52 / 0.826 | 40 / 4.287 | 150 / 2.007 |

**D1 R-local theo nhóm N**: TB đều / TB trọng số qua các target có ≥ 30 ô có support (số target trong ngoặc).

| Model | N1 | N2 | N3 |
|---|---:|---:|---:|
| hron | +0.107 / +0.095 (15) | +0.051 / +0.045 (14) | -0.058 / -0.057 (13) |
| tsagris | +0.016 / +0.041 (15) | -0.058 / -0.071 (14) | -0.109 / -0.099 (13) |
| micoda | -0.112 / -0.089 (15) | -0.529 / -0.525 (14) | -0.459 / -0.568 (13) |
| micoda_other | +0.113 / +0.107 (15) | -0.152 / -0.148 (14) | -0.229 / -0.284 (13) |
| same_day_mean | +0.108 / +0.099 (15) | +0.064 / +0.063 (14) | -0.046 / -0.019 (13) |
| mean_fill | +0.113 / +0.117 (15) | -0.082 / -0.067 (14) | -0.550 / -0.539 (13) |
| zero_fill | +0.128 / +0.118 (15) | +0.147 / +0.140 (14) | +0.144 / +0.120 (13) |
| marginal_sampler | +0.105 / +0.098 (15) | +0.099 / +0.097 (14) | +0.083 / +0.071 (13) |
| hron_cap | +0.114 / +0.104 (15) | +0.096 / +0.092 (14) | +0.054 / +0.065 (13) |
| tsagris_cap | +0.016 / +0.041 (15) | -0.058 / -0.071 (14) | -0.109 / -0.099 (13) |
| micoda_other_cap | +0.113 / +0.107 (15) | -0.152 / -0.148 (14) | -0.229 / -0.284 (13) |

**D2 R-local theo nhóm N**: TB đều / TB trọng số qua các target có ≥ 30 ô có support (số target trong ngoặc).

| Model | N1 | N2 | N3 |
|---|---:|---:|---:|
| hron | 0.136 / 0.137 (13) | 0.067 / 0.068 (14) | 0.034 / 0.025 (13) |
| tsagris | 0.118 / 0.110 (12) | 0.062 / 0.057 (14) | 0.035 / 0.021 (13) |
| micoda | 0.111 / 0.096 (15) | 0.099 / 0.084 (14) | 0.044 / 0.043 (13) |
| micoda_other | 0.098 / 0.073 (2) | 0.068 / 0.063 (14) | 0.032 / 0.022 (13) |
| same_day_mean | 0.113 / 0.113 (12) | 0.056 / 0.054 (14) | 0.033 / 0.024 (12) |
| mean_fill | 0.126 / 0.126 (1) | 0.089 / 0.085 (9) | 0.033 / 0.023 (13) |
| zero_fill | NA / NA (0) | NA / NA (0) | NA / NA (0) |
| marginal_sampler | 0.187 / 0.189 (15) | 0.083 / 0.083 (13) | 0.044 / 0.033 (13) |
| hron_cap | 0.137 / 0.134 (14) | 0.066 / 0.065 (14) | 0.033 / 0.024 (13) |
| tsagris_cap | 0.118 / 0.110 (12) | 0.062 / 0.057 (14) | 0.035 / 0.021 (13) |
| micoda_other_cap | 0.098 / 0.073 (2) | 0.068 / 0.063 (14) | 0.032 / 0.022 (13) |

**D3 theo nhóm N**: số ô cờ / khối lượng tỷ lệ (NA khi tầng có < 30 ô M_m).

| Model | N1 | N2 | N3 |
|---|---:|---:|---:|
| hron | 12 / 5.343 | 20 / 0.983 | 70 / 0.620 |
| tsagris | 39 / 8.099 | 342 / 8.006 | 208 / 2.336 |
| micoda | 1136 / 175.497 | 1795 / 226.446 | 471 / 51.791 |
| micoda_other | 31 / 5.331 | 892 / 14.294 | 336 / 4.402 |
| same_day_mean | 0 / 0.000 | 23 / 0.177 | 60 / 0.185 |
| mean_fill | 6 / 2.329 | 269 / 3.324 | 320 / 1.610 |
| zero_fill | 0 / 0.000 | 0 / 0.000 | 0 / 0.000 |
| marginal_sampler | 48 / 20.394 | 76 / 14.154 | 27 / 2.142 |
| hron_cap | 5 / 4.473 | 1 / 0.030 | 2 / 0.005 |
| tsagris_cap | 39 / 8.099 | 342 / 8.006 | 208 / 2.336 |
| micoda_other_cap | 31 / 5.331 | 892 / 14.293 | 336 / 4.402 |
<!-- auto:strata:end -->

### Bảng 6: R-local so với R-pooled (D1)

<!-- auto:shift:start -->
Ký hiệu: **±** đổi dấu, **L** |D1 local − D1 pooled| ≥ 0.1, · không có cả hai. Cột Δref = Pr_ref(zero) local − pooled (chung mọi model).

| Target | Δref | hron | tsagris | micoda | micoda_other | same_day_mean | mean_fill | zero_fill | marginal_sampler | hron_cap | tsagris_cap | micoda_other_cap |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| recombinant | +0.246 | ±L | ±L | L | L | L | ±L | L | ±L | ±L | ±L | L |
| 20A | -0.061 | · | · | · | · | · | · | · | · | · | · | · |
| 20B | -0.087 | · | ± | · | · | · | · | · | · | · | ± | · |
| 20C | -0.146 | ±L | L | L | L | L | ± | L | L | L | L | L |
| 20E | -0.055 | · | · | · | · | · | · | · | · | · | · | · |
| Beta | -0.078 | · | · | · | · | ± | · | · | · | · | · | · |
| Alpha | -0.034 | · | · | L | ±L | ±L | L | · | · | · | · | ±L |
| Gamma | -0.011 | · | · | · | · | ± | · | · | · | · | · | · |
| Delta | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| Kappa | -0.051 | ± | ± | · | · | ± | ± | · | · | · | ± | · |
| Epsilon | -0.066 | ± | · | · | · | ± | · | · | · | · | · | · |
| Eta | -0.142 | ±L | ±L | ±L | ±L | ± | ±L | L | L | L | ±L | ±L |
| Iota | -0.056 | ± | ± | · | · | ± | · | · | · | · | ± | · |
| Lambda | -0.049 | · | · | · | · | · | · | · | · | · | · | · |
| Mu | +0.008 | · | · | · | · | · | · | · | · | · | · | · |
| Omicron | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| S:677 | -0.022 | · | ± | · | · | · | L | · | · | · | ± | · |

Số target đổi dấu / thay đổi lớn (trên số target có cả hai giá trị): hron 6 / 3 (/15); tsagris 6 / 3 (/15); micoda 1 / 4 (/15); micoda_other 2 / 4 (/15); same_day_mean 7 / 3 (/15); mean_fill 4 / 4 (/15); zero_fill 0 / 3 (/15); marginal_sampler 1 / 3 (/15); hron_cap 1 / 3 (/15); tsagris_cap 6 / 3 (/15); micoda_other_cap 2 / 4 (/15).
<!-- auto:shift:end -->

### Bảng 7: đối thủ giả làm mốc đọc (không phải cổng)

<!-- auto:decoys:start -->
- zero_fill: D1 pooled = 1 − Pr_ref(zero) ở 16/16 target có D1 (mọi ô M_m điền 0). D1 TB đều R-local +0.135, R-pooled +0.125. D1 lớn nhất (hoặc bằng) trong các model ở 15/15 target (R-local), 16/16 (R-pooled). D2 = NA ở 15/15 (R-local), 16/16 (R-pooled) target có D1. Target không có D1 (không có ô M_m, hoặc không ô nào có support): Delta, Omicron (R-local); Delta (R-pooled).
- marginal_sampler: D1 R-pooled TB đều +0.023, trọng số +0.044; |D1 pooled| ≤ 0.05 ở 10/16 target có D1. D2 R-pooled TB đều 0.0736, trọng số 0.0569. R-local: D1 +0.096 / +0.091, D2 0.0854 / 0.0734 (đều / trọng số).
<!-- auto:decoys:end -->

### Tổng hợp (cùng số với bảng 2–4, gom một chỗ)

<!-- auto:overview:start -->
| Model | D1 local (đều / trọng số) | D2 local (đều / trọng số) | D1 pooled (đều / trọng số) | D2 pooled (đều / trọng số) | D3 ô | D3 khối lượng |
|---|---|---|---|---|---:|---:|
| hron | +0.054 / +0.044 | 0.0485 / 0.0433 | -0.009 / +0.001 | 0.0524 / 0.0393 | 102 | 6.947 |
| tsagris | -0.052 / -0.028 | 0.0646 / 0.0486 | -0.120 / -0.056 | 0.0910 / 0.0536 | 589 | 18.441 |
| micoda | -0.315 / -0.340 | 0.0836 / 0.0727 | -0.364 / -0.440 | 0.1115 / 0.0984 | 3402 | 453.734 |
| micoda_other | -0.055 / -0.066 | 0.0652 / 0.0553 | -0.116 / -0.111 | 0.0985 / 0.0681 | 1259 | 24.028 |
| same_day_mean | +0.066 / +0.061 | 0.0565 / 0.0471 | +0.054 / +0.018 | 0.0782 / 0.0595 | 83 | 0.362 |
| mean_fill | -0.096 / -0.092 | 0.0660 / 0.0567 | -0.130 / -0.104 | 0.1191 / 0.0705 | 595 | 7.263 |
| zero_fill | +0.135 / +0.126 | NA / NA | +0.125 / +0.074 | NA / NA | 0 | 0.000 |
| marginal_sampler | +0.096 / +0.091 | 0.0854 / 0.0734 | +0.023 / +0.044 | 0.0736 / 0.0569 | 151 | 36.690 |
| hron_cap | +0.094 / +0.091 | 0.0591 / 0.0534 | +0.028 / +0.046 | 0.0584 / 0.0402 | 8 | 4.508 |
| tsagris_cap | -0.052 / -0.028 | 0.0646 / 0.0486 | -0.120 / -0.056 | 0.0910 / 0.0536 | 589 | 18.441 |
| micoda_other_cap | -0.055 / -0.066 | 0.0652 / 0.0553 | -0.116 / -0.111 | 0.0985 / 0.0681 | 1259 | 24.027 |
<!-- auto:overview:end -->

## 5. Mô tả, không xếp hạng

Coverage R-local của toàn bộ M_m là **55.73%** (14 185 / 25 455 ô). Câu nào dùng R-local chỉ nói về phần ô có support đó.

1. **Điền dương tràn lan: MICoDa A17.**
   - D1 âm ở 14/15 target có D1 R-local: TB trọng số −0.340 (coverage 55.7%); R-pooled −0.440 (coverage 100%).
   - Lớn nhất ở S:677: −0.828 R-local, coverage 0.383. Tiếp theo là 20C: −0.624, coverage 0.181.
   - D3 cờ 3 402 ô (khối lượng 453.7), gấp hơn 2.5 lần model kế tiếp.
   - Khớp với cơ chế R1 (`rescale`) đã thấy ở step 4: dồn toàn bộ phần dư vào ô thiếu.
2. **Điền dương nhiều hơn reference ở mức vừa:** micoda_other, mean_fill, tsagris có D1 R-local TB trọng số −0.066, −0.092, −0.028.
   - Riêng tsagris ở Alpha là −0.517, nhưng chỉ dựa trên 39 ô có support (coverage 0.574 của 68 ô M_m).
   - micoda_other cờ D3 1 259 ô; tsagris và mean_fill khoảng 590.
3. **Nhiều zero hơn reference cục bộ, nhưng không sụp về 0 hoàn toàn:** hron, same_day_mean, hron_cap, marginal_sampler có D1 R-local dương, TB trọng số +0.044 đến +0.091.
   - Mốc sụp hoàn toàn là zero_fill: +0.126.
   - hron_cap (+0.091) lệch về zero nhiều hơn hron (+0.044).
   - Hai model có ít cờ D3 nhất là same_day_mean (83 ô, khối lượng 0.36) và hron_cap (8 ô).
4. **Dấu của D1 phụ thuộc mạnh vào nhóm N** (bảng 5; coverage N1 0.547, N2 0.525, N3 0.637).
   - Ở N1 (N ≤ 17), mọi model trừ MICoDa A17 có D1 dương (TB đều +0.016 đến +0.128), vì giá trị điền lẻ < 0.5 count tính là zero.
   - Ở N3 (N > 149), mọi model trừ zero_fill, marginal_sampler, hron_cap có D1 âm (TB trọng số): MICoDa A17 −0.568, mean_fill −0.539, micoda_other −0.284, tsagris −0.099, hron −0.057, same_day_mean −0.019.
   - Một D1 gộp qua mọi N vì vậy trộn hai hiện tượng ngược chiều.
5. **Theo châu lục:**
   - Bắc Mỹ (coverage 0.709): mọi model trừ MICoDa A17 có D1 dương (TB đều và trọng số từ +0.105 đến +0.257).
   - Châu Âu (coverage 0.995): MICoDa A17 âm nhất, −0.464.
   - Châu Phi (coverage 0.256) và Nam Mỹ (0.430): chỉ 6 và 8 target đủ 30 ô.
6. **Support thấp ở đâu:**
   - **Theo châu lục:** Châu Đại Dương coverage **0**, nên mọi số R-local ở đó là NA. Châu Phi 25.6%, Nam Mỹ 43.0%.
   - **Theo target:** 20C 18.1%, Lambda 28.7%, S:677 38.3%, Gamma 45.3%. Omicron có 0/4 ô có support. Delta không có ô M_m nào.
   - Hệ quả: các target có nhiều ô M_m nhất (S:677 2 979, 20C 2 870) lại có coverage thấp nhất, nên R-local đại diện kém cho đúng phần lớn ô cần mô tả.
7. **D3 nằm ở các target có ô M_m sau thời kỳ lưu hành.**
   - 6 573 ô M_m (25.8%) nằm ngoài T_j, nhiều nhất ở Mu, Kappa, Iota, Lambda.
   - Cờ ở Omicron đến từ 4 dòng của Fiji (N = 1, 15 target thiếu, trước 2021-08-30). Vì N = 1, mỗi ô điền dương góp gần trọn khối lượng 1. 9/11 model điền dương ở đó (trừ same_day_mean và zero_fill). Phần lớn khối lượng D3 của hron_cap đến từ 4 ô này (4.0 / 4.5).
8. **Hai cặp gần trùng:** tsagris_cap trùng tsagris; micoda_other_cap gần trùng micoda_other (khác ở chữ số thứ 3 khối lượng D3). Đúng như step 4: hai model này cho composition đóng trên D+1, nên `cap` ≈ `prop_other`.
9. **Hai reference cho kết quả khác nhau ở một số target** (bảng 6).
   - **recombinant:** Pr_ref(zero) R-local 0.992 so với R-pooled 0.747. Ô M_m của recombinant gần các ô quan sát zero, còn ô dương nằm ở thời kỳ khác. Cả 11 model đổi ≥ 0.10, và 6 model đổi dấu.
   - **20C và Eta:** Δref là −0.146 và −0.142, và mọi model đổi ≥ 0.10 ở ít nhất một trong hai target.
   - Ở 12 target còn lại có số, thay đổi ≥ 0.10 chỉ có ở Alpha (5 model) và S:677 (mean_fill). Đổi dấu xuất hiện rải rác ở các target có |D1| nhỏ: Kappa, Iota, Epsilon, Beta, Gamma, 20B, S:677.

## 6. So với Tầng 1

Chỉ nêu chỗ hai tầng gợi ý khác nhau, không kết luận tầng nào đúng. Tầng 1 đo sai số trên ô che giả có đáp án, Tầng 2 đo mức giống reference trên ô thiếu thật: hai đối tượng đo khác nhau.

<!-- auto:tier1:start -->
Tầng 1: trung bình trên seed chung của 11 model (whole_feature: 0, 1, 2, 3, 4; random_cell: 0, 1, 2, 3, 4), lấy từ tier01_runs.csv và residual_runs.csv đã commit. Nhóm ô model; z = đáp án 0, + = đáp án dương. Tầng 2: native, R-local, TB trọng số.

| Model | T1 wf MAE | T1 wf MAE z | T1 wf MAE + | T1 rc MAE | T1 rc MAE z | T1 rc MAE + | T2 D1 | T2 D2 | T2 D3 ô |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hron | 0.0875 | 0.0294 | 0.1978 | 0.0892 | 0.0315 | 0.1943 | +0.044 | 0.0433 | 102 |
| tsagris | 0.1079 | 0.0745 | 0.1719 | 0.0999 | 0.0781 | 0.1395 | -0.028 | 0.0486 | 589 |
| micoda | 0.1406 | 0.1177 | 0.1834 | 0.1317 | 0.1139 | 0.1641 | -0.340 | 0.0727 | 3402 |
| micoda_other | 0.1075 | 0.0554 | 0.2056 | 0.1078 | 0.0653 | 0.1851 | -0.066 | 0.0553 | 1259 |
| same_day_mean | 0.0351 | 0.0094 | 0.0844 | 0.0310 | 0.0093 | 0.0704 | +0.061 | 0.0471 | 83 |
| mean_fill | 0.1354 | 0.0508 | 0.2950 | 0.1371 | 0.0577 | 0.2817 | -0.092 | 0.0567 | 595 |
| zero_fill | 0.1693 | 0.0000 | 0.4886 | 0.1716 | 0.0000 | 0.4840 | +0.126 | NA | 0 |
| marginal_sampler | 0.1354 | 0.0411 | 0.3130 | 0.1362 | 0.0472 | 0.2983 | +0.091 | 0.0734 | 151 |
| hron_cap | 0.0869 | 0.0258 | 0.2029 | 0.0879 | 0.0288 | 0.1955 | +0.091 | 0.0534 | 8 |
| tsagris_cap | 0.1079 | 0.0745 | 0.1719 | 0.0999 | 0.0781 | 0.1395 | -0.028 | 0.0486 | 589 |
| micoda_other_cap | 0.1075 | 0.0554 | 0.2056 | 0.1078 | 0.0653 | 0.1851 | -0.066 | 0.0553 | 1259 |
<!-- auto:tier1:end -->

1. **marginal_sampler trông "hợp lý" ở Tầng 2 nhưng kém ở Tầng 1.**
   - Nó lấy mẫu từ phân phối quan sát, nên D1 R-pooled gần 0 (+0.023 / +0.044; |D1| ≤ 0.05 ở 10/16 target). D2 R-pooled (0.0569) cùng cỡ hron (0.0393) và tsagris (0.0536).
   - Ở Tầng 1, MAE của nó (0.1354 wf) ngang mean_fill (0.1354) và kém xa hron (0.0875).
   - Vậy D1/D2 không phân biệt được một mồi nhử đúng phân phối với model sát từng ô. Đây là điều Tầng 3 phải đo.
2. **hron và hron_cap gần trùng ở Tầng 1** (MAE wf 0.0875 so với 0.0869). Ở Tầng 2 lại khác: D1 R-local +0.044 so với +0.091, và D3 102 ô so với 8.
3. **same_day_mean có MAE Tầng 1 thấp nhất** (0.0351 wf), kể cả trên ô đáp án dương (0.0844). Nhưng ở Tầng 2, D1 R-local là +0.061 (nhiều zero hơn reference cục bộ), và D2 R-local (0.0471) không nhỏ hơn hron (0.0433).
4. **tsagris và micoda_other gần nhau ở Tầng 1** (MAE wf 0.1079 so với 0.1075), nhưng D3 khác hơn hai lần (589 so với 1 259 ô).
5. **Chỗ hai tầng cùng hướng:**
   - MICoDa A17 có MAE trên ô model đáp án 0 cao nhất ở Tầng 1 (0.1177 wf), và D1 âm nhất ở Tầng 2. Cả hai cùng chỉ việc điền dương nơi lẽ ra là 0.
   - zero_fill có MAE z = 0 ở Tầng 1 và D1 lớn nhất ở Tầng 2.

## 7. Giới hạn

- **Không có đáp án** trên ô thiếu thật. Mọi số là mức giống reference từ ô quan sát, không phải độ chính xác. Hai thế giới (missing hay Other nhận phần dư) cho cùng input (review 4.2).
- **R-local giả định các nước cùng châu lục, trong ±28 ngày, trao đổi được với nhau.** Reference có thể thưởng cho việc giống nước láng giềng thay vì giống nước được điền. Reference của một ô trung bình đến từ 6.5–19.4 location tùy target (bảng 1), với nhiều mốc lặp; ≥ 20 ô không phải 20 mẫu độc lập.
- **Observed zero không đồng nghĩa biến thể vắng mặt**: có thể chỉ là không lấy mẫu được. T_j là khoảng có count dương trong bảng, không phải mốc xuất hiện hay biến mất thật.
- **Ngưỡng zero theo count gắn với N**: với N nhỏ, giá trị điền lẻ dễ thành zero. Bảng 5 cho thấy điều này đổi dấu D1.
- **Coverage R-local chỉ 55.7%.** Châu Đại Dương không có ô nào có support. Các target nhiều ô M_m nhất lại có coverage thấp nhất.
- **Một lượt native cho mỗi model**: không có độ dao động theo seed, không CI.

## 8. Quyết định thiết kế chưa chắc

1. **Zero theo count (< 0.5), không theo tỷ lệ.** Có ý nghĩa "ít hơn nửa mẫu", nhưng phụ thuộc N (mục 5.4). Một ngưỡng theo tỷ lệ sẽ cho kết quả khác.
2. **Trọng số R-local**: mỗi ô có support mang tổng trọng số 1, và một ô V có thể được đếm nhiều lần. D2 so với hỗn hợp R_j điều kiện trên phần dương, không lấy trung bình các W1 cục bộ, nên ô thiếu có reference toàn zero không góp gì vào D2.
3. **Trọng số đều theo ô (c/N), không theo N.** Review 8.1 nêu rằng N-weighted không mặc nhiên đúng hơn.
4. **Cửa sổ ±28 ngày tính cả hai đầu.** Loại mọi ô của chính location (kể cả ngày khác), dù review 8.1 cho rằng dùng chuỗi của chính nước không phải leakage.
5. **Bảng theo target không có ngưỡng ô tối thiểu**, chỉ tầng có ngưỡng 30. Omicron (4 ô M_m) và Alpha (39 ô có support) vì vậy vẫn có số.
6. **T_j tính trên toàn bảng**, không theo châu lục.
7. **Ngưỡng 0.10 của bảng 6 là quy ước.** Đã chốt trước khi chạy.
8. **Đọc CSV completed không qua `load_csv`.** `load_csv` từ chối dòng có tổng điền vượt N một sai số làm tròn (Bulgaria, dòng 796), trong khi `check_completed` chấp nhận (1e-10 tương đối). Script so từng cột không phải target với data rồi để `check_completed` kiểm.

## 9. Việc để lại cho Tầng 3

- Chạy D1/D2/D3 trên lượt masked bằng `calibration_cells`:
  - E = ô che thuộc nhóm model;
  - V = O \ H;
  - so thứ tự hoặc dấu cặp model trong từng mask × seed × target với Tầng 1;
  - báo near-tie và selection regret.
- Kiểm hành vi trên đối thủ giả: marginal_sampler hiện qua được D1/D2 R-pooled (mục 6.1). Cần biết R-local trên lượt masked có tách được nó hay không.
- Coverage R-local trên H: whole_feature xóa cả chuỗi, nên support của location khác sẽ khác native.
- Phân tầng N cho cả Tầng 1, để biết hiệu ứng N của D1 có thật trên ô có đáp án hay không.
- Cân nhắc ngưỡng ô tối thiểu theo target, ngưỡng zero theo tỷ lệ, và reference leave-country-out. Đây là việc của step sau, không đổi định nghĩa ở step này.
- Mang sang từ các step trước: bootstrap/CI, JSD/ACF (chưa có use case), Tầng 4.
