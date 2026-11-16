# Step 7: mask khớp support của ô thiếu thật, kiểm lại Tầng 0, 1, 3 (exploratory)

Branch `step7-matched-mask` (từ `step6-tier3-calibration`). Không có `iterative_net` hay `framework/`. Không sửa baseline, không đổi định nghĩa metric, diagnostic, use case hay ngưỡng nào, không dùng seed ≥ 100. Mọi bảng số trong khối `auto:*` do `matched.py` sinh, không sửa tay; bảng đầy đủ ở [matched_results.md](matched_results.md).

`whole_feature_matched` vẫn là **ô quan sát được che lại**: đây là sensitivity, không phải native validation (review 7.2). Không kết quả nào dưới đây là độ chính xác trên ô thiếu thật.

## 1. File đã sửa/thêm

| File | Nội dung |
|---|---|
| `src/longitudinal_impute/matched_mask.py` (126 dòng) | Tầng, π native, trọng số, lấy mẫu có trọng số không hoàn lại, TV. |
| `src/longitudinal_impute/data.py` | `make_mask(..., target=None)` thêm kiểu `whole_feature_matched`. Vòng che chuỗi, guard và quy tắc dừng dùng chung với `whole_feature`; `whole_feature` vẫn gọi `rng.permutation` như cũ. |
| `src/longitudinal_impute/run.py` | `--mask whole_feature_matched --mask-target JSON`. Đường dẫn, sha256 và nội dung target ghi vào `config.json`. |
| `configs/mask_target_covid.json` | π đóng băng: 18 tầng, ngưỡng N 17 / 149 đọc từ output step 5. |
| `tests/test_matched_mask.py` | 8 test mới. |
| `tv2_step6/tier3.py` | `load_mask` nhận thư mục run, seed và target; các bảng lặp theo các mask được đưa vào. Output step 6 sinh lại giống từng byte. |
| `tv2_step7/matched.py` (323 dòng), `test_matched.py` (3 test), `run_all.sh` | Tính π, dựng bảng 1–6, chạy ba lượt model. |
| `tv2_step7/matched_*.csv`, `matched_results.md`, `matched_manifest.json`, `matched_cases.json` | Output sinh tự động. Thư mục run không commit; hash nằm trong manifest. |

## 2. Đặc tả mask matched

- **Tầng:** châu lục (6) × nhóm N (3) = 18 tầng. N1: N ≤ 17; N2: 17 < N ≤ 149; N3: N > 149. Ngưỡng **đọc từ output step 5**, không tính lại.
- **π:** tỷ lệ ô M_m native theo tầng, tính một lần, lưu trong `mask_target_covid.json`. M_m dùng cách tách forced-zero (tổng quan sát của dòng = N), tức chỉ đọc tổng của ô quan sát, không đọc ô che nào.
- **Ứng viên:** mọi cặp (location, target) có ô quan sát, che trọn chuỗi quan sát của cặp (như `whole_feature`). n_s,k là số ô của ứng viên s ở tầng k; p_k là tỷ lệ ô của mọi ứng viên ở tầng k.
- **Trọng số:** w_s = Σ_k n_s,k · π_k / p_k, trên các tầng của π có p_k > 0. Tầng có p_k = 0 bị bỏ, phần khối lượng π của nó được ghi là "không với tới".
- **Lấy mẫu:** không hoàn lại, xác suất tỷ lệ với w_s. Dùng khóa Efraimidis–Spirakis log(u)/w_s, với u = 1 − uniform từ `default_rng(seed)`; ứng viên có w_s = 0 không bao giờ được chọn. Sau đó áp đúng quy tắc của `whole_feature`: bỏ chuỗi làm một dòng mất mọi target visible (guard, có đếm), và dừng khi đã che ≥ 5% ô quan sát.
- **Firewall:** việc chọn chỉ đọc location, châu lục, N và vị trí ô quan sát; không đọc giá trị count nào.
- **Khai báo trước khi chạy** (docstring `matched.py`):
  - Tầng 1 theo tầng: `cell_errors` trên ô che nhóm model của tầng; tầng < 30 ô trong một seed thì NA ở seed đó.
  - Tầng 3 giữ nguyên đặc tả step 6. Bộ 10 ca là `whole_feature_matched` (5) cộng `random_cell` (5 ca của step 6, tính lại và kiểm khớp JSON đã commit), để quy tắc "trong từng mask" và "≥ 9/10 ca" giữ nguyên nghĩa.
- **Commit:**
  - `3bc59a9` đóng băng mask (hàm, runner, test), trước khi sinh mask nào;
  - `a575ad6` sửa `tier3.py` (output step 6 không đổi);
  - `2c21c8c` thêm script và các khai báo, trước khi sinh mask;
  - `a9d7e90` đóng băng π;
  - `53e6a8a` sửa phần trình bày (nhãn tầng chứa `|`) và thêm provenance của run vào manifest; không đổi định nghĩa nào;
  - `0f495b7` commit số. Các run được chạy từ commit `a9d7e90`, git tree sạch.

## 3. Kiểm tra

- **Test:** 333 pass với `-W error::RuntimeWarning`:
  - nextgen 303 (295 cũ, 8 mới);
  - script 30 (27 cũ, 3 mới).
- **Test mới kiểm:**
  - mask chỉ chọn ô quan sát, che trọn chuỗi, đạt rate theo quy tắc dừng; cùng seed cho cùng mask;
  - firewall: đổi count (giữ N, ngày, location, vị trí NaN) thì mask không đổi, trên toy lẫn trên COVID thật;
  - guard: không dòng nào mất mọi target visible, và số chuỗi bị bỏ được đếm;
  - trọng số đúng công thức trên toy tính tay; tầng p_k = 0 bị bỏ và được ghi lại; w = 0 không bao giờ được chọn;
  - trên fixture lệch, TV của mask matched nhỏ hơn `whole_feature` ít nhất 0.3;
  - ngưỡng N đọc từ output step 5 (17 / 149), không tính lại;
  - runner: `--mask-target` bắt buộc cho mask matched và chỉ cho nó, sha256 nằm trong `config.json`;
  - `whole_feature` và `random_cell` cho mask giống hash của step 1 ở seed 0–4.
- **Tái lập:** `matched.py` chạy hai lần (hai thư mục output) cho 5 file output giống từng byte.
- **Output cũ không đổi:** step 6 sinh lại giống từng byte sau khi sửa `tier3.py`. 5 ca random_cell tính lại khớp `tv2_step6/tier3_cases.json`; script dừng nếu không khớp.
- **Kiểm hash trước khi đọc run** (qua `tier3.load_mask`):
  - F2 giữa `main_*` và `micoda_other_*`;
  - data, schema, group map và bảng model khớp file toml;
  - git sạch lúc chạy;
  - mask đã lưu giống nhau ở 3 thư mục run và bằng `make_mask`;
  - long file khớp `summary.json` (1 650 giá trị);
  - sha256 của target trong `config.json` của cả 3 run bằng sha của `mask_target_covid.json`.

## 4. Bảng

### Bảng 1: khớp phân bố

<!-- auto:match:start -->
π: ô M_m native theo 18 tầng (châu lục × nhóm N, cắt [17.0, 149.0]; từ `mask_target_covid.json`). Phân bố đạt được: trên ô che thuộc nhóm model, TB 5 seed.

| Tầng | π | wf | wfm | p_k (ô ứng viên) |
|---|---:|---:|---:|---:|
| Africa · N1 | 0.168 | 0.085 | 0.176 | 0.073 |
| Africa · N2 | 0.121 | 0.061 | 0.146 | 0.042 |
| Africa · N3 | 0.029 | 0.016 | 0.052 | 0.009 |
| Asia · N1 | 0.113 | 0.057 | 0.069 | 0.055 |
| Asia · N2 | 0.103 | 0.074 | 0.082 | 0.064 |
| Asia · N3 | 0.058 | 0.100 | 0.105 | 0.089 |
| Europe · N1 | 0.070 | 0.060 | 0.029 | 0.056 |
| Europe · N2 | 0.051 | 0.080 | 0.044 | 0.089 |
| Europe · N3 | 0.076 | 0.240 | 0.128 | 0.253 |
| North America · N1 | 0.050 | 0.037 | 0.020 | 0.051 |
| North America · N2 | 0.030 | 0.033 | 0.021 | 0.047 |
| North America · N3 | 0.004 | 0.049 | 0.011 | 0.049 |
| Oceania · N1 | 0.014 | 0.007 | 0.010 | 0.003 |
| Oceania · N2 | 0.010 | 0.009 | 0.014 | 0.005 |
| Oceania · N3 | 0.005 | 0.005 | 0.029 | 0.020 |
| South America · N1 | 0.041 | 0.024 | 0.025 | 0.027 |
| South America · N2 | 0.032 | 0.030 | 0.022 | 0.031 |
| South America · N3 | 0.026 | 0.032 | 0.016 | 0.037 |

| Mask | TV tới π (mean ± sd) | coverage R-local trên E | ứng viên bị guard bỏ / seed | cặp được che / seed | đạt rate |
|---|---|---|---|---|---|
| whole_feature | 0.313 ± 0.015 | 0.906 ± 0.044 | 0, 0, 0, 0, 0 | 67, 64, 65, 68, 66 | True, True, True, True, True |
| whole_feature_matched | 0.206 ± 0.043 | 0.902 ± 0.027 | 0, 0, 0, 0, 0 | 66, 64, 65, 65, 64 | True, True, True, True, True |

Khối lượng π không với tới (tầng của π không có ô ứng viên nào): 0.0000 (không). Cặp ứng viên có w = 0: 0 / 1298.

Phân bố theo target (chỉ báo, không khớp): tỷ lệ ô của từng target.

| Target | native M_m | wf E | wfm E |
|---|---:|---:|---:|
| recombinant | 0.026 | 0.070 | 0.136 |
| 20A | 0.045 | 0.068 | 0.057 |
| 20B | 0.084 | 0.023 | 0.014 |
| 20C | 0.113 | 0.030 | 0.003 |
| 20E | 0.045 | 0.049 | 0.040 |
| Beta | 0.030 | 0.054 | 0.096 |
| Alpha | 0.003 | 0.081 | 0.098 |
| Gamma | 0.056 | 0.057 | 0.015 |
| Delta | 0.000 | 0.118 | 0.113 |
| Kappa | 0.082 | 0.024 | 0.040 |
| Epsilon | 0.091 | 0.045 | 0.023 |
| Eta | 0.049 | 0.067 | 0.074 |
| Iota | 0.088 | 0.053 | 0.017 |
| Lambda | 0.090 | 0.029 | 0.029 |
| Mu | 0.081 | 0.030 | 0.037 |
| Omicron | 0.000 | 0.165 | 0.195 |
| S:677 | 0.117 | 0.035 | 0.014 |
<!-- auto:match:end -->

### Bảng 2: Tầng 0 trên `whole_feature_matched`

<!-- auto:status:start -->
| Model | Lượt | Mask | Lên lịch | completed | partial | failed | missing | Run không ok |
|---|---|---|---:|---:|---:|---:|---:|---|
| hron | native | none | 1 | 1 | 0 | 0 | 0 | - |
| tsagris | native | none | 1 | 1 | 0 | 0 | 0 | - |
| micoda | native | none | 1 | 1 | 0 | 0 | 0 | - |
| same_day_mean | native | none | 1 | 1 | 0 | 0 | 0 | - |
| mean_fill | native | none | 1 | 1 | 0 | 0 | 0 | - |
| marginal_sampler | native | none | 1 | 1 | 0 | 0 | 0 | - |
| zero_fill | native | none | 1 | 1 | 0 | 0 | 0 | - |
| hron | masked | whole_feature_matched | 5 | 5 | 0 | 0 | 0 | - |
| tsagris | masked | whole_feature_matched | 5 | 5 | 0 | 0 | 0 | - |
| micoda | masked | whole_feature_matched | 5 | 5 | 0 | 0 | 0 | - |
| same_day_mean | masked | whole_feature_matched | 5 | 5 | 0 | 0 | 0 | - |
| mean_fill | masked | whole_feature_matched | 5 | 5 | 0 | 0 | 0 | - |
| marginal_sampler | masked | whole_feature_matched | 5 | 5 | 0 | 0 | 0 | - |
| zero_fill | masked | whole_feature_matched | 5 | 5 | 0 | 0 | 0 | - |
| micoda_other | native | none | 1 | 1 | 0 | 0 | 0 | - |
| micoda_other | masked | whole_feature_matched | 5 | 5 | 0 | 0 | 0 | - |
| hron_cap | native | none | 1 | 1 | 0 | 0 | 0 | - |
| tsagris_cap | native | none | 1 | 1 | 0 | 0 | 0 | - |
| micoda_other_cap | native | none | 1 | 1 | 0 | 0 | 0 | - |
| hron_cap | masked | whole_feature_matched | 5 | 5 | 0 | 0 | 0 | - |
| tsagris_cap | masked | whole_feature_matched | 5 | 5 | 0 | 0 | 0 | - |
| micoda_other_cap | masked | whole_feature_matched | 5 | 5 | 0 | 0 | 0 | - |

Seed 0 (whole_feature_matched): 5692 ô che, 2915 ép về 0, 2777 ô model; 298 dòng đầy đủ sau khi che.
Seed 1 (whole_feature_matched): 5695 ô che, 2813 ép về 0, 2882 ô model; 189 dòng đầy đủ sau khi che.
Seed 2 (whole_feature_matched): 5701 ô che, 2812 ép về 0, 2889 ô model; 327 dòng đầy đủ sau khi che.
Seed 3 (whole_feature_matched): 5643 ô che, 2317 ép về 0, 3326 ô model; 298 dòng đầy đủ sau khi che.
Seed 4 (whole_feature_matched): 5622 ô che, 2949 ép về 0, 2673 ô model; 407 dòng đầy đủ sau khi che.
<!-- auto:status:end -->

### Bảng 3: Tầng 1, so thứ hạng với `whole_feature`

<!-- auto:tier1:start -->
**whole_feature**

Tập seed chung (mọi model completed, coverage = 1): 0, 1, 2, 3, 4.

Sắp theo MAE tỷ lệ (nhóm ô model) trung bình trên tập seed chung. mean ± sd (ddof=1) qua seed; sd không phải khoảng tin cậy. **exploratory**.

| Model | MAE tỷ lệ | MSE tỷ lệ | MAE count | MAE tỷ lệ series-macro | MAE tỷ lệ, đáp án = 0 | MAE tỷ lệ, đáp án > 0 | MAE tỷ lệ, ô model đáp án = 0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| same_day_mean | 0.0351 ± 0.0066 | 0.0142 ± 0.0036 | 29.4 ± 26.8 | 0.0495 ± 0.0095 | 0.0033 ± 0.0014 | 0.0844 ± 0.0181 | 0.0094 ± 0.0040 |
| hron_cap | 0.0869 ± 0.0095 | 0.0658 ± 0.0087 | 81.5 ± 115.2 | 0.0993 ± 0.0128 | 0.0091 ± 0.0028 | 0.2029 ± 0.0175 | 0.0258 ± 0.0070 |
| hron | 0.0875 ± 0.0105 | 0.0660 ± 0.0092 | 74.6 ± 113.7 | 0.1003 ± 0.0135 | 0.0103 ± 0.0027 | 0.1978 ± 0.0195 | 0.0294 ± 0.0065 |
| micoda_other | 0.1075 ± 0.0093 | 0.0684 ± 0.0078 | 188.7 ± 205.5 | 0.1175 ± 0.0137 | 0.0194 ± 0.0036 | 0.2056 ± 0.0195 | 0.0554 ± 0.0095 |
| micoda_other_cap | 0.1075 ± 0.0093 | 0.0684 ± 0.0078 | 188.7 ± 205.5 | 0.1175 ± 0.0137 | 0.0194 ± 0.0036 | 0.2056 ± 0.0195 | 0.0554 ± 0.0095 |
| tsagris_cap | 0.1079 ± 0.0209 | 0.0521 ± 0.0095 | 33.2 ± 14.7 | 0.1207 ± 0.0207 | 0.0260 ± 0.0075 | 0.1719 ± 0.0458 | 0.0745 ± 0.0214 |
| tsagris | 0.1079 ± 0.0209 | 0.0521 ± 0.0095 | 33.2 ± 14.7 | 0.1207 ± 0.0207 | 0.0260 ± 0.0075 | 0.1719 ± 0.0458 | 0.0745 ± 0.0214 |
| mean_fill | 0.1354 ± 0.0145 | 0.0704 ± 0.0084 | 218.2 ± 186.8 | 0.1324 ± 0.0162 | 0.0178 ± 0.0037 | 0.2950 ± 0.0154 | 0.0508 ± 0.0092 |
| marginal_sampler | 0.1354 ± 0.0120 | 0.1107 ± 0.0122 | 225.1 ± 200.7 | 0.1333 ± 0.0151 | 0.0144 ± 0.0019 | 0.3130 ± 0.0142 | 0.0411 ± 0.0037 |
| micoda | 0.1406 ± 0.0099 | 0.0823 ± 0.0094 | 236.2 ± 188.5 | 0.1476 ± 0.0161 | 0.0411 ± 0.0051 | 0.1834 ± 0.0261 | 0.1177 ± 0.0116 |
| zero_fill | 0.1693 ± 0.0219 | 0.1525 ± 0.0230 | 287.1 ± 211.3 | 0.1498 ± 0.0241 | 0.0000 ± 0.0000 | 0.4886 ± 0.0405 | 0.0000 ± 0.0000 |

**whole_feature**, cặp chính (có điều kiện trên tập seed chung của từng cặp; thắng = MAE tỷ lệ nhóm model nhỏ hơn; Δ = A − B, mean ± sd ddof=1)

| A vs B | n seed | A thắng | Δ MAE tỷ lệ | Seed loại (lý do) |
|---|---:|---:|---|---|
| same_day_mean vs hron | 5 | 5/5 | -0.05247 ± 0.01578 | - |
| tsagris vs hron | 5 | 0/5 | +0.02036 ± 0.01274 | - |
| micoda_other vs mean_fill | 5 | 5/5 | -0.02785 ± 0.00699 | - |
| micoda vs mean_fill | 5 | 1/5 | +0.00525 ± 0.01350 | - |
| micoda_other vs micoda | 5 | 5/5 | -0.03309 ± 0.00681 | - |
| hron vs mean_fill | 5 | 5/5 | -0.04782 ± 0.00853 | - |
| same_day_mean vs hron_cap | 5 | 5/5 | -0.05184 ± 0.01437 | - |

**whole_feature_matched**

Tập seed chung (mọi model completed, coverage = 1): 0, 1, 2, 3, 4.

Sắp theo MAE tỷ lệ (nhóm ô model) trung bình trên tập seed chung. mean ± sd (ddof=1) qua seed; sd không phải khoảng tin cậy. **exploratory**.

| Model | MAE tỷ lệ | MSE tỷ lệ | MAE count | MAE tỷ lệ series-macro | MAE tỷ lệ, đáp án = 0 | MAE tỷ lệ, đáp án > 0 | MAE tỷ lệ, ô model đáp án = 0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| same_day_mean | 0.0391 ± 0.0088 | 0.0150 ± 0.0053 | 23.1 ± 15.4 | 0.0550 ± 0.0104 | 0.0054 ± 0.0014 | 0.0847 ± 0.0197 | 0.0136 ± 0.0040 |
| hron_cap | 0.1060 ± 0.0228 | 0.0719 ± 0.0129 | 23.8 ± 8.5 | 0.1080 ± 0.0114 | 0.0148 ± 0.0048 | 0.2287 ± 0.0435 | 0.0365 ± 0.0092 |
| hron | 0.1072 ± 0.0230 | 0.0726 ± 0.0128 | 21.0 ± 6.8 | 0.1095 ± 0.0110 | 0.0163 ± 0.0049 | 0.2252 ± 0.0427 | 0.0401 ± 0.0095 |
| micoda_other | 0.1151 ± 0.0137 | 0.0728 ± 0.0101 | 66.5 ± 42.9 | 0.1192 ± 0.0110 | 0.0301 ± 0.0102 | 0.1912 ± 0.0291 | 0.0744 ± 0.0224 |
| micoda_other_cap | 0.1151 ± 0.0137 | 0.0728 ± 0.0101 | 66.5 ± 42.9 | 0.1192 ± 0.0110 | 0.0301 ± 0.0102 | 0.1912 ± 0.0291 | 0.0744 ± 0.0224 |
| tsagris | 0.1165 ± 0.0160 | 0.0545 ± 0.0097 | 22.6 ± 11.3 | 0.1218 ± 0.0080 | 0.0324 ± 0.0080 | 0.1817 ± 0.0209 | 0.0807 ± 0.0211 |
| tsagris_cap | 0.1165 ± 0.0160 | 0.0545 ± 0.0097 | 22.6 ± 11.3 | 0.1218 ± 0.0080 | 0.0324 ± 0.0080 | 0.1817 ± 0.0209 | 0.0807 ± 0.0211 |
| micoda | 0.1291 ± 0.0163 | 0.0791 ± 0.0131 | 60.1 ± 43.1 | 0.1350 ± 0.0149 | 0.0447 ± 0.0097 | 0.1656 ± 0.0374 | 0.1108 ± 0.0218 |
| mean_fill | 0.1475 ± 0.0223 | 0.0725 ± 0.0093 | 98.4 ± 103.3 | 0.1401 ± 0.0113 | 0.0269 ± 0.0072 | 0.2918 ± 0.0151 | 0.0665 ± 0.0155 |
| marginal_sampler | 0.1482 ± 0.0231 | 0.1201 ± 0.0206 | 110.3 ± 120.0 | 0.1414 ± 0.0120 | 0.0217 ± 0.0055 | 0.3158 ± 0.0204 | 0.0535 ± 0.0111 |
| zero_fill | 0.1828 ± 0.0411 | 0.1651 ± 0.0400 | 157.7 ± 215.4 | 0.1629 ± 0.0240 | 0.0000 ± 0.0000 | 0.5027 ± 0.0677 | 0.0000 ± 0.0000 |

**whole_feature_matched**, cặp chính (có điều kiện trên tập seed chung của từng cặp; thắng = MAE tỷ lệ nhóm model nhỏ hơn; Δ = A − B, mean ± sd ddof=1)

| A vs B | n seed | A thắng | Δ MAE tỷ lệ | Seed loại (lý do) |
|---|---:|---:|---|---|
| same_day_mean vs hron | 5 | 5/5 | -0.06814 ± 0.01933 | - |
| tsagris vs hron | 5 | 1/5 | +0.00933 ± 0.00991 | - |
| micoda_other vs mean_fill | 5 | 5/5 | -0.03241 ± 0.01412 | - |
| micoda vs mean_fill | 5 | 4/5 | -0.01839 ± 0.02189 | - |
| micoda_other vs micoda | 5 | 4/5 | -0.01402 ± 0.00810 | - |
| hron vs mean_fill | 5 | 5/5 | -0.04032 ± 0.00947 | - |
| same_day_mean vs hron_cap | 5 | 5/5 | -0.06695 ± 0.01926 | - |

Thứ tự whole_feature: same_day_mean < hron_cap < hron < micoda_other < micoda_other_cap < tsagris_cap < tsagris < mean_fill < marginal_sampler < micoda < zero_fill.
Thứ tự whole_feature_matched: same_day_mean < hron_cap < hron < micoda_other < micoda_other_cap < tsagris < tsagris_cap < micoda < mean_fill < marginal_sampler < zero_fill.
Cặp đổi chỗ: tsagris_cap / tsagris; mean_fill / micoda; marginal_sampler / micoda.
<!-- auto:tier1:end -->

### Bảng 4: Tầng 1 theo tầng

<!-- auto:strata:start -->
**MAE tỷ lệ (ô model) theo châu lục** — mean ± sd qua seed có ≥ 30 ô; n = số seed khi thiếu.

| Model | Mask | Africa | Asia | Europe | North America | Oceania | South America |
|---|---|---:|---:|---:|---:|---:|---:|
| hron | wf | 0.1458 ± 0.0530 | 0.0787 ± 0.0473 | 0.0766 ± 0.0156 | 0.0812 ± 0.0413 | 0.0670 ± 0.0950 (n = 4) | 0.0590 ± 0.0342 |
| hron | wfm | 0.1509 ± 0.0477 | 0.0826 ± 0.0268 | 0.0726 ± 0.0450 | 0.1172 ± 0.0688 | 0.0849 ± 0.0990 | 0.0671 ± 0.0730 |
| tsagris | wf | 0.1964 ± 0.0576 | 0.1050 ± 0.0579 | 0.0750 ± 0.0130 | 0.0889 ± 0.0377 | 0.0802 ± 0.0803 (n = 4) | 0.1140 ± 0.0692 |
| tsagris | wfm | 0.1695 ± 0.0285 | 0.0938 ± 0.0177 | 0.0586 ± 0.0304 | 0.1242 ± 0.0515 | 0.0991 ± 0.1174 | 0.0890 ± 0.0720 |
| micoda | wf | 0.1734 ± 0.0455 | 0.1318 ± 0.0384 | 0.1268 ± 0.0106 | 0.1805 ± 0.0382 | 0.0925 ± 0.0941 (n = 4) | 0.1183 ± 0.0273 |
| micoda | wfm | 0.1543 ± 0.0325 | 0.1128 ± 0.0198 | 0.1066 ± 0.0289 | 0.1667 ± 0.0785 | 0.1392 ± 0.0908 | 0.1066 ± 0.0604 |
| micoda_other | wf | 0.1700 ± 0.0477 | 0.1017 ± 0.0467 | 0.0987 ± 0.0192 | 0.0719 ± 0.0243 | 0.0653 ± 0.0964 (n = 4) | 0.0882 ± 0.0402 |
| micoda_other | wfm | 0.1542 ± 0.0345 | 0.0939 ± 0.0171 | 0.0800 ± 0.0550 | 0.1149 ± 0.0750 | 0.1312 ± 0.0931 | 0.0758 ± 0.0682 |
| same_day_mean | wf | 0.0652 ± 0.0511 | 0.0281 ± 0.0129 | 0.0290 ± 0.0105 | 0.0387 ± 0.0182 | 0.0185 ± 0.0105 (n = 4) | 0.0233 ± 0.0127 |
| same_day_mean | wfm | 0.0491 ± 0.0100 | 0.0317 ± 0.0136 | 0.0231 ± 0.0055 | 0.0431 ± 0.0276 | 0.0877 ± 0.0674 | 0.0283 ± 0.0194 |
| mean_fill | wf | 0.2051 ± 0.0534 | 0.1142 ± 0.0495 | 0.1377 ± 0.0229 | 0.1009 ± 0.0483 | 0.0638 ± 0.0911 (n = 4) | 0.0911 ± 0.0433 |
| mean_fill | wfm | 0.1953 ± 0.0422 | 0.1123 ± 0.0322 | 0.1180 ± 0.0840 | 0.1497 ± 0.0850 | 0.1433 ± 0.0949 | 0.0896 ± 0.0933 |
| zero_fill | wf | 0.2360 ± 0.0675 | 0.1295 ± 0.0676 | 0.1924 ± 0.0261 | 0.1396 ± 0.1065 | 0.0552 ± 0.0831 (n = 4) | 0.0788 ± 0.0394 |
| zero_fill | wfm | 0.2266 ± 0.0598 | 0.1387 ± 0.0684 | 0.1738 ± 0.1264 | 0.1994 ± 0.1424 | 0.1614 ± 0.1044 | 0.0960 ± 0.1296 |
| marginal_sampler | wf | 0.2064 ± 0.0480 | 0.1161 ± 0.0478 | 0.1339 ± 0.0227 | 0.1042 ± 0.0509 | 0.0677 ± 0.0972 (n = 4) | 0.0974 ± 0.0521 |
| marginal_sampler | wfm | 0.1939 ± 0.0470 | 0.1150 ± 0.0312 | 0.1215 ± 0.0902 | 0.1491 ± 0.0837 | 0.1510 ± 0.1013 | 0.0824 ± 0.0920 |
| hron_cap | wf | 0.1421 ± 0.0538 | 0.0731 ± 0.0460 | 0.0795 ± 0.0165 | 0.0790 ± 0.0415 | 0.0617 ± 0.0991 (n = 4) | 0.0636 ± 0.0373 |
| hron_cap | wfm | 0.1486 ± 0.0485 | 0.0814 ± 0.0271 | 0.0715 ± 0.0434 | 0.1160 ± 0.0730 | 0.0896 ± 0.1097 | 0.0717 ± 0.0737 |
| tsagris_cap | wf | 0.1964 ± 0.0576 | 0.1050 ± 0.0579 | 0.0750 ± 0.0130 | 0.0889 ± 0.0377 | 0.0802 ± 0.0803 (n = 4) | 0.1140 ± 0.0692 |
| tsagris_cap | wfm | 0.1695 ± 0.0285 | 0.0938 ± 0.0177 | 0.0586 ± 0.0304 | 0.1242 ± 0.0515 | 0.0991 ± 0.1174 | 0.0890 ± 0.0720 |
| micoda_other_cap | wf | 0.1700 ± 0.0477 | 0.1017 ± 0.0467 | 0.0987 ± 0.0192 | 0.0719 ± 0.0243 | 0.0653 ± 0.0964 (n = 4) | 0.0882 ± 0.0402 |
| micoda_other_cap | wfm | 0.1542 ± 0.0345 | 0.0939 ± 0.0171 | 0.0800 ± 0.0550 | 0.1149 ± 0.0750 | 0.1312 ± 0.0931 | 0.0758 ± 0.0682 |

same_day_mean − hron theo châu lục (Δ MAE, mean ± sd; số seed same_day_mean thắng / số seed có cả hai).

| Tầng | wf | wfm |
|---|---|---|
| Africa | -0.0806 ± 0.0413 (5/5) | -0.1018 ± 0.0425 (5/5) |
| Asia | -0.0506 ± 0.0357 (5/5) | -0.0509 ± 0.0244 (5/5) |
| Europe | -0.0476 ± 0.0092 (5/5) | -0.0495 ± 0.0447 (4/5) |
| North America | -0.0425 ± 0.0506 (5/5) | -0.0741 ± 0.0842 (4/5) |
| Oceania | -0.0485 ± 0.0903 (3/4) | +0.0028 ± 0.0632 (2/5) |
| South America | -0.0357 ± 0.0242 (5/5) | -0.0388 ± 0.0746 (3/5) |

**MAE tỷ lệ (ô model) theo nhóm N** — mean ± sd qua seed có ≥ 30 ô; n = số seed khi thiếu.

| Model | Mask | N1 | N2 | N3 |
|---|---|---:|---:|---:|
| hron | wf | 0.1541 ± 0.0268 | 0.1112 ± 0.0143 | 0.0315 ± 0.0092 |
| hron | wfm | 0.1550 ± 0.0439 | 0.1259 ± 0.0296 | 0.0423 ± 0.0074 |
| tsagris | wf | 0.2031 ± 0.0425 | 0.1242 ± 0.0251 | 0.0390 ± 0.0099 |
| tsagris | wfm | 0.1788 ± 0.0246 | 0.1307 ± 0.0258 | 0.0419 ± 0.0073 |
| micoda | wf | 0.1776 ± 0.0168 | 0.1581 ± 0.0286 | 0.1068 ± 0.0133 |
| micoda | wfm | 0.1498 ± 0.0338 | 0.1531 ± 0.0325 | 0.0868 ± 0.0141 |
| micoda_other | wf | 0.1602 ± 0.0207 | 0.1246 ± 0.0232 | 0.0638 ± 0.0127 |
| micoda_other | wfm | 0.1394 ± 0.0352 | 0.1363 ± 0.0307 | 0.0714 ± 0.0179 |
| same_day_mean | wf | 0.0513 ± 0.0148 | 0.0494 ± 0.0034 | 0.0156 ± 0.0059 |
| same_day_mean | wfm | 0.0476 ± 0.0095 | 0.0478 ± 0.0147 | 0.0222 ± 0.0067 |
| mean_fill | wf | 0.1962 ± 0.0448 | 0.1474 ± 0.0242 | 0.0899 ± 0.0235 |
| mean_fill | wfm | 0.1902 ± 0.0314 | 0.1629 ± 0.0313 | 0.0910 ± 0.0300 |
| zero_fill | wf | 0.2275 ± 0.0796 | 0.1735 ± 0.0269 | 0.1301 ± 0.0381 |
| zero_fill | wfm | 0.2354 ± 0.0326 | 0.1913 ± 0.0525 | 0.1229 ± 0.0530 |
| marginal_sampler | wf | 0.2007 ± 0.0455 | 0.1467 ± 0.0201 | 0.0874 ± 0.0212 |
| marginal_sampler | wfm | 0.1902 ± 0.0323 | 0.1656 ± 0.0314 | 0.0902 ± 0.0269 |
| hron_cap | wf | 0.1522 ± 0.0278 | 0.1078 ± 0.0142 | 0.0332 ± 0.0092 |
| hron_cap | wfm | 0.1540 ± 0.0441 | 0.1234 ± 0.0311 | 0.0420 ± 0.0059 |
| tsagris_cap | wf | 0.2031 ± 0.0425 | 0.1242 ± 0.0251 | 0.0390 ± 0.0099 |
| tsagris_cap | wfm | 0.1788 ± 0.0246 | 0.1307 ± 0.0258 | 0.0419 ± 0.0073 |
| micoda_other_cap | wf | 0.1602 ± 0.0207 | 0.1246 ± 0.0232 | 0.0638 ± 0.0127 |
| micoda_other_cap | wfm | 0.1394 ± 0.0352 | 0.1363 ± 0.0307 | 0.0714 ± 0.0179 |

same_day_mean − hron theo nhóm N (Δ MAE, mean ± sd; số seed same_day_mean thắng / số seed có cả hai).

| Tầng | wf | wfm |
|---|---|---|
| N1 | -0.1028 ± 0.0283 (5/5) | -0.1074 ± 0.0458 (5/5) |
| N2 | -0.0618 ± 0.0168 (5/5) | -0.0781 ± 0.0196 (5/5) |
| N3 | -0.0159 ± 0.0115 (5/5) | -0.0201 ± 0.0058 (5/5) |
<!-- auto:strata:end -->

### Bảng 5: Tầng 3, step 6 và step 7 cạnh nhau

<!-- auto:tier3:start -->
| Diagnostic | Use case | step 6: wf + rc | step 7: wfm + rc |
|---|---|---|---|
| D1 | U1 thiếu | không đạt | **không đạt** |
| D1 | U1 thừa | đạt | **đạt** |
| D2 | U2 | đạt | **không đạt** |
| D3 | U3 | đạt | **đạt** |
| D1 c/N < 0.001 | U1 thiếu | không đạt | **không đạt** |
| D1 c/N < 0.001 | U1 thừa | đạt | **đạt** |
| D1 R-pooled | U1 thiếu | không đạt | **không đạt** |
| D1 R-pooled | U1 thừa | đạt | **đạt** |
| D2 R-pooled | U2 | không đạt | **đạt** |
<!-- auto:tier3:end -->

Chi tiết từng ca (wfm + rc):

<!-- auto:tier3_detail:start -->
| Mask | Seed | thiếu: khớp (khớp/quyết định) | thừa: khớp |
|---|---:|---|---|
| whole_feature_matched | 0 | 0.33 (5/15) | 0.87 (13/15) |
| whole_feature_matched | 1 | 0.80 (12/15) | 0.87 (13/15) |
| whole_feature_matched | 2 | 0.73 (11/15) | 0.80 (12/15) |
| whole_feature_matched | 3 | 0.73 (11/15) | 0.80 (12/15) |
| whole_feature_matched | 4 | 0.80 (12/15) | 0.87 (13/15) |
| random_cell | 0 | 0.67 (10/15) | 0.80 (12/15) |
| random_cell | 1 | 0.64 (9/14, hòa 1) | 0.80 (12/15) |
| random_cell | 2 | 0.67 (10/15) | 0.80 (12/15) |
| random_cell | 3 | 0.71 (10/14, hòa 1) | 0.87 (13/15) |
| random_cell | 4 | 0.67 (10/15) | 0.87 (13/15) |
- U1 thiếu (D1 cao ↔ MAE ô đáp án > 0): whole_feature_matched: TB 0.68, min 0.33 (5 ca); random_cell: TB 0.67, min 0.64 (5 ca) → **không đạt**.
- U1 thừa (D1 thấp ↔ MAE ô đáp án = 0): whole_feature_matched: TB 0.84, min 0.80 (5 ca); random_cell: TB 0.83, min 0.80 (5 ca) → **đạt**.

| Mask | Seed | khớp \|D2 − D2t\| | chọn → regret | khớp D2 thô | regret thô | khớp, MAE toàn ô model | regret, MAE toàn ô model |
|---|---:|---|---|---|---:|---|---:|
| whole_feature_matched | 0 | 0.93 (13/14, hòa 1) | same_day_mean → 0.0000 | 0.93 (13/14, hòa 1) | 0.0000 | 0.93 (13/14, hòa 1) | 0.0000 |
| whole_feature_matched | 1 | 0.69 (9/13, hòa 2) | same_day_mean → 0.0000 | 0.69 (9/13, hòa 2) | 0.0000 | 0.75 (9/12, hòa 3) | 0.0000 |
| whole_feature_matched | 2 | 0.53 (8/15) | same_day_mean → 0.0000 | 0.53 (8/15) | 0.0000 | 0.71 (10/14, hòa 1) | 0.0000 |
| whole_feature_matched | 3 | 0.67 (6/9, hòa 6) | same_day_mean → 0.0000 | 0.67 (6/9, hòa 6) | 0.0000 | 1.00 (7/7, hòa 8) | 0.0000 |
| whole_feature_matched | 4 | 0.80 (12/15) | same_day_mean → 0.0000 | 0.80 (12/15) | 0.0000 | 0.80 (12/15) | 0.0000 |
| random_cell | 0 | 0.93 (13/14, hòa 1) | same_day_mean → 0.0000 | 0.93 (13/14, hòa 1) | 0.0000 | 0.93 (13/14, hòa 1) | 0.0000 |
| random_cell | 1 | 0.87 (13/15) | same_day_mean → 0.0000 | 0.87 (13/15) | 0.0000 | 0.87 (13/15) | 0.0000 |
| random_cell | 2 | 0.86 (12/14, hòa 1) | same_day_mean → 0.0000 | 0.86 (12/14, hòa 1) | 0.0000 | 0.86 (12/14, hòa 1) | 0.0000 |
| random_cell | 3 | 0.87 (13/15) | same_day_mean → 0.0000 | 0.87 (13/15) | 0.0000 | 0.87 (13/15) | 0.0000 |
| random_cell | 4 | 0.80 (12/15) | same_day_mean → 0.0000 | 0.80 (12/15) | 0.0000 | 0.80 (12/15) | 0.0000 |
- U2: whole_feature_matched: TB 0.72, min 0.53 (5 ca); random_cell: TB 0.86, min 0.80 (5 ca); regret ≤ 0.005 ở 10/10 ca → **không đạt**.

| Mask | Seed | ô dương bị cờ | tỷ lệ đáp án 0 khi bị cờ | ô dương không cờ | khi không cờ | chênh |
|---|---:|---:|---:|---:|---:|---:|
| whole_feature_matched | 0 | 889 | 1.000 | 5892 | 0.327 | +0.673 |
| whole_feature_matched | 1 | 1487 | 1.000 | 6365 | 0.199 | +0.801 |
| whole_feature_matched | 2 | 1166 | 0.997 | 6721 | 0.216 | +0.780 |
| whole_feature_matched | 3 | 1418 | 1.000 | 7657 | 0.201 | +0.799 |
| whole_feature_matched | 4 | 953 | 1.000 | 5684 | 0.344 | +0.656 |
| random_cell | 0 | 1308 | 0.998 | 6647 | 0.261 | +0.737 |
| random_cell | 1 | 1281 | 0.995 | 6505 | 0.282 | +0.713 |
| random_cell | 2 | 1257 | 1.000 | 6527 | 0.284 | +0.716 |
| random_cell | 3 | 1191 | 1.000 | 6584 | 0.276 | +0.724 |
| random_cell | 4 | 1177 | 1.000 | 6564 | 0.288 | +0.712 |
- U3: chênh ≥ 0.2 ở 10/10 ca → **đạt**.

Số ca diagnostic xếp đối thủ giả kém hơn **mọi** model thật (trên số ca có giá trị), R-local.

| Đối thủ giả | Mask | \|D1 − D1_truth\| | \|D1\| thô | \|D2 − D2_truth\| | D2 thô | D3: ô cờ / ô E | *tham chiếu: MAE Tầng 1, toàn ô model* |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| mean_fill | whole_feature_matched | 4/5 | 4/5 | 4/4 | 4/4 | 1/5 | 4/5 |
| zero_fill | whole_feature_matched | 5/5 | 2/5 | NA | NA | 0/5 | 5/5 |
| marginal_sampler | whole_feature_matched | 0/5 | 0/5 | 0/3 | 0/3 | 0/5 | 4/5 |

Giá trị trung bình qua 5 seed (R-local, trên target chung của model thật):

| Model | Mask | \|D1 − D1_truth\| | \|D1\| thô | \|D2 − D2_truth\| | D2 thô | D3: ô cờ / ô E | *tham chiếu: MAE Tầng 1, toàn ô model* |
|---|---|---:|---:|---:|---:|---:|---:|
| hron | whole_feature_matched | 0.0227 ± 0.0196 | 0.0968 ± 0.0186 | 0.0819 ± 0.0390 | 0.1458 ± 0.0300 | 0.0274 ± 0.0071 | 0.1072 ± 0.0230 |
| hron_cap | whole_feature_matched | 0.0531 ± 0.0209 | 0.0981 ± 0.0243 | 0.0947 ± 0.0469 | 0.1587 ± 0.0381 | 0.0257 ± 0.0087 | 0.1060 ± 0.0228 |
| tsagris | whole_feature_matched | 0.1327 ± 0.0371 | 0.2212 ± 0.0238 | 0.1105 ± 0.0337 | 0.1744 ± 0.0235 | 0.0821 ± 0.0160 | 0.1165 ± 0.0160 |
| micoda | whole_feature_matched | 0.2459 ± 0.0559 | 0.3074 ± 0.0548 | 0.1283 ± 0.0183 | 0.1922 ± 0.0133 | 0.1527 ± 0.0277 | 0.1291 ± 0.0163 |
| micoda_other | whole_feature_matched | 0.0894 ± 0.0246 | 0.1820 ± 0.0186 | 0.1641 ± 0.0218 | 0.2280 ± 0.0141 | 0.1119 ± 0.0223 | 0.1151 ± 0.0137 |
| same_day_mean | whole_feature_matched | 0.0338 ± 0.0154 | 0.0933 ± 0.0172 | 0.0316 ± 0.0118 | 0.0955 ± 0.0077 | 0.0048 ± 0.0058 | 0.0391 ± 0.0088 |
| mean_fill | whole_feature_matched | 0.2774 ± 0.0227 | 0.3456 ± 0.0169 | 0.2278 ± 0.0371 (n = 4) | 0.2891 ± 0.0255 (n = 4) | 0.1292 ± 0.0236 | 0.1475 ± 0.0223 |
| zero_fill | whole_feature_matched | 0.3687 ± 0.0404 | 0.3162 ± 0.0350 | NA | NA | 0.0000 ± 0.0000 | 0.1828 ± 0.0411 |
| marginal_sampler | whole_feature_matched | 0.1734 ± 0.0245 | 0.1221 ± 0.0126 | 0.0600 ± 0.0225 (n = 3) | 0.1201 ± 0.0071 (n = 3) | 0.0454 ± 0.0104 | 0.1482 ± 0.0231 |
<!-- auto:tier3_detail:end -->

### Bảng 6: chuyển giao, masked E so với native M_m

<!-- auto:transfer:start -->
| Tập ô | coverage R-local | Africa | Asia | Europe | North America | Oceania | South America | N1 (N cắt native) | N2 (N cắt native) | N3 (N cắt native) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| native M_m | 0.557 | 0.318 | 0.274 | 0.197 | 0.085 | 0.029 | 0.098 | 0.455 | 0.347 | 0.198 |
| masked E, whole_feature (TB 5 seed) | 0.906 | 0.163 | 0.231 | 0.381 | 0.119 | 0.021 | 0.086 | 0.269 | 0.289 | 0.442 |
| masked E, random_cell (TB 5 seed) | 0.919 | 0.129 | 0.210 | 0.407 | 0.131 | 0.027 | 0.096 | 0.241 | 0.275 | 0.484 |
| masked E, whole_feature_matched (TB 5 seed) | 0.902 | 0.374 | 0.256 | 0.202 | 0.052 | 0.053 | 0.063 | 0.330 | 0.329 | 0.342 |

Tỷ lệ 0 theo target: native = tỷ lệ zero quan sát (ô O); masked = tỷ lệ đáp án 0 trên E (TB 5 seed).

| Target | native O | wf E | rc E | wfm E |
|---|---:|---:|---:|---:|
| recombinant | 0.747 | 0.591 | 0.553 | 0.551 |
| 20A | 0.881 | 0.714 | 0.723 | 0.747 |
| 20B | 0.920 | 0.852 | 0.805 | 0.768 |
| 20C | 0.950 | 0.919 | 0.872 | 0.947 |
| 20E | 0.895 | 0.700 | 0.761 | 0.795 |
| Beta | 0.889 | 0.560 | 0.726 | 0.742 |
| Alpha | 0.833 | 0.596 | 0.611 | 0.669 |
| Gamma | 0.882 | 0.687 | 0.726 | 0.644 |
| Delta | 0.745 | 0.461 | 0.498 | 0.531 |
| Kappa | 0.961 | 0.942 | 0.908 | 0.935 |
| Epsilon | 0.939 | 0.890 | 0.850 | 0.889 |
| Eta | 0.928 | 0.819 | 0.823 | 0.805 |
| Iota | 0.952 | 0.887 | 0.895 | 0.946 |
| Lambda | 0.927 | 0.815 | 0.845 | 0.875 |
| Mu | 0.919 | 0.821 | 0.842 | 0.884 |
| Omicron | 0.418 | 0.331 | 0.348 | 0.338 |
| S:677 | 0.959 | 0.928 | 0.892 | 0.958 |
<!-- auto:transfer:end -->

## 5. Kết luận có giới hạn (exploratory; sd không phải CI)

1. **Mask matched khớp π tốt hơn nhưng chưa khớp hẳn.**
   - TV tới π giảm từ 0.313 ± 0.015 (`whole_feature`) xuống 0.206 ± 0.043.
   - Không tầng nào không với tới được (khối lượng 0), không ứng viên nào có w = 0, guard không bỏ chuỗi nào, và cả 5 seed đều đạt rate.
   - Châu Phi tăng từ 0.16 lên 0.37 (native 0.32), Châu Âu giảm từ 0.38 xuống 0.20 (native 0.20).
   - Nhóm N chỉ dịch một phần: N1 0.27 → 0.33 (native 0.46), N3 0.44 → 0.34 (native 0.20).
   - Phần thiếu N1 sinh ra ngay ở ô che (34–38% ô che nằm ở N1), không phải do tách forced-zero: tỷ lệ ô model trong ô che gần đều giữa các nhóm N (0.44–0.65). Mask chọn **trọn chuỗi**, mà một chuỗi trải qua nhiều nhóm N theo thời gian, nên không nhắm được riêng các dòng N nhỏ. Đây là giải thích hợp lý, chưa kiểm riêng.
2. **Hai trục mask matched không đổi được:**
   - **Coverage R-local:** 0.90 trên E, gần như cũ (0.91), trong khi native là 0.56. Che 5% ô quan sát không tái tạo được việc cả một target thiếu ở nhiều nước cùng lúc, nên các location lân cận vẫn còn ô quan sát.
   - **Phân bố theo target** (không khớp theo target, theo đặc tả): ô che tập trung ở Omicron (0.195) và Delta (0.113), hai target **không có ô M_m native nào**. Ngược lại, 20C (native 0.113) và S:677 (0.117) chỉ chiếm 0.003 và 0.014 của ô che.
3. **Tầng 0:** cả 33 run (11 model × 5 seed, cộng native) đều completed với coverage 1. Không model nào fail. Trái với dự đoán, mask matched để lại **nhiều** dòng đầy đủ hơn (189–407 trên 516, so với 80–298 của `whole_feature`). Nó ít chọn chuỗi ở Châu Âu N lớn, nơi có các nước có dòng đầy đủ.
4. **Thứ hạng Tầng 1 giữ ở đầu bảng.**
   - Bốn vị trí đầu không đổi: same_day_mean < hron_cap < hron < micoda_other.
   - same_day_mean thắng hron 5/5 seed, Δ MAE −0.068 ± 0.019 (whole_feature: −0.052 ± 0.016).
   - Ba cặp đổi chỗ, đều ở cuối bảng: micoda (A17) lên trên mean_fill và marginal_sampler; tsagris với tsagris_cap (trùng nhau).
   - MAE mọi model tăng nhẹ: same_day_mean 0.0351 → 0.0391, hron 0.0875 → 0.1072.
5. **same_day_mean vẫn dẫn ở Châu Phi, Châu Á và N nhỏ** (bảng 4, wfm):
   - Châu Phi Δ −0.102 (5/5), Châu Á −0.051 (5/5), N1 −0.107 (5/5), N2 −0.078 (5/5), N3 −0.020 (5/5).
   - Không dẫn rõ ở Châu Đại Dương (+0.003, thắng 2/5; ít ô, sd lớn) và yếu hơn ở Nam Mỹ (3/5), Bắc Mỹ và Châu Âu (4/5).
6. **Tầng 3 (10 ca = wfm + rc, đặc tả step 6 không đổi):**
   - **Còn đạt:**
     - D1 cảnh báo điền thừa: TB 0.84 trên wfm, min 0.80;
     - D3 cờ ngoài support: chênh +0.66 đến +0.80, 10/10 ca;
     - các sensitivity của D1 thừa.
   - **Vẫn không đạt:** D1 cảnh báo điền thiếu. Trên wfm, TB 0.68, có ca chỉ 0.33.
   - **Đổi kết luận:**
     - **D2/U2 R-local chuyển sang không đạt:** wfm TB 0.72, min 0.53, nhiều cặp hòa (6 cặp ở seed 3), dù regret đạt 10/10 ca (luôn chọn same_day_mean).
     - **D2/U2 R-pooled chuyển sang đạt:** TB 0.84, regret 10/10.
     - Kết luận về D2 làm proxy xếp hạng vì thế **không bền theo mask**. Chỉ phần "chọn được model tốt nhất" (regret) đứng vững ở cả hai bộ ca, và đó là trường hợp model tốt nhất cách xa các model còn lại.
   - **Đối thủ giả (U4, wfm):**
     - zero_fill bị \|D1 − D1_truth\| bắt 5/5;
     - mean_fill bị D2 bắt 4/4 ca có giá trị, D1 bắt 4/5;
     - marginal_sampler vẫn không bị diagnostic nào bắt (0/5), trong khi MAE Tầng 1 xếp nó kém hơn mọi model thật ở 4/5 ca.

## 6. Hệ quả cho bộ evaluation v1

- **Dùng `whole_feature_matched` làm sensitivity bắt buộc, chưa làm mask chính.**
  - Nó sửa được phần lệch châu lục và một phần lệch N.
  - Nó không sửa được hai trục quan trọng nhất cho Tầng 2/3: coverage R-local và phân bố theo target.
  - Kết luận Tầng 1 chính (same_day_mean dẫn) đứng vững trên nó, kể cả trong các tầng khó. Còn một kết luận Tầng 3 (D2/U2) đổi chiều, nên mọi kết luận Tầng 3 cần báo trên cả hai mask.
- **Giữ `whole_feature` làm mask chính:** nó đơn giản, đã có từ step 1, và không phụ thuộc π. Báo `whole_feature_matched` cạnh nó.
- **Không bỏ mask matched:** đây là mask duy nhất hiện có với Châu Phi chiếm tỷ trọng gần native, và kết luận same_day_mean > hron ở N1 / Châu Phi dựa vào nó.

## 7. Quyết định thiết kế chưa chắc

1. **Trọng số theo chuỗi, chọn trọn chuỗi.** Một chuỗi trải qua nhiều nhóm N, nên N1 vẫn thiếu. Che theo đoạn chuỗi sẽ khớp N tốt hơn, nhưng đó là kiểu mask khác với `whole_feature`.
2. **Lấy mẫu tỷ lệ với w_s, không theo hạn ngạch.** Kết quả khớp π theo kỳ vọng, không khớp từng seed (sd của TV 0.043).
3. **π đo trên M_m native, còn khớp đo trên ô che nhóm model.** Hai tập dùng cùng quy tắc forced-zero, nhưng ô che là ô quan sát được.
4. **Không khớp theo target** (theo đặc tả). Đây là trục lệch lớn nhất còn lại: Omicron và Delta chiếm 31% ô che nhưng 0% M_m.
5. **Bộ 10 ca của Tầng 3 dùng lại 5 ca random_cell của step 6**, nên hai cột của bảng 5 có chung một nửa số ca.
6. **Tầng 1 theo tầng tính trên ô model của từng tầng, theo từng seed.** Tầng < 30 ô thì NA. Châu Đại Dương có sd lớn.
7. **`tv2_step6/tier3.py` được sửa** để nhận thư mục run và target. Output step 6 không đổi, nhưng file đó không còn đúng như lúc step 6 commit số.

## 8. Việc để lại cho step 8 (lượt xác nhận)

- **Khai báo trước:**
  - seed ≥ 100 (ví dụ 100–104);
  - mask: `whole_feature` (chính) và `whole_feature_matched` (sensitivity), cùng `mask_target_covid.json` (sha trong manifest);
  - cùng 11 model, cấu hình y hệt.
- **Chỉ kiểm các kết luận đã đứng ở cả step 6 và step 7:**
  - Tầng 1: same_day_mean thắng hron, toàn bảng và ở Châu Phi / Châu Á / N1;
  - Tầng 3: D1 cảnh báo điền thừa, D3 cờ ngoài support, và regret của D2 (không kiểm tỷ lệ khớp của D2).
- **Không chạy thêm** nếu kết quả không đạt. Ghi đúng như vậy.
- **Ngoài phạm vi step 8, cần quyết định riêng:**
  - một mask khớp theo target hoặc theo pattern thiếu đồng thời giữa các nước, để tiến gần coverage native;
  - random_cell matched;
  - che theo đoạn chuỗi.
