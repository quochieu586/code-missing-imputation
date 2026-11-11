# Step 3: sửa nhỏ step 2 + ablation same_day_mean (exploratory)

Branch `step3-sameday-ablation` (từ `step2-tier01-eval`). Không đổi thuật toán Hron/Tsagris/MICoDa, không đổi mặc định của model nào. Mọi bảng số trong các khối `auto:*` do `ablation.py` sinh, không sửa tay. Bảng đầy đủ: [ablation_results.md](ablation_results.md).

## 1. Phần A: sửa từ review step 2

1. **Bảng metric và bảng ghép cặp của step 2 chỉ dùng seed khai báo trước (0–4).** Seed 109 chỉ còn ở bảng trạng thái và bảng reliability (`tier01.UNDECLARED`); cột MAE cũng đã bỏ khỏi bảng seed 109. Các bảng step 2 đã được sinh lại và [tv2_step2/REPORT_vi.md](../tv2_step2/REPORT_vi.md) có thêm mục *Sửa sau review*. Chỉ các cặp trước đây có n = 6 thay đổi (6/6 → 5/5; mean_fill vs marginal_sampler 3/6 → 2/5, Δ ≈ 0). **Không kết luận nào đổi.** Có test riêng cho trường hợp này.
2. **Bảng tổng hợp cũ của `run.py`:** README có ghi chú rằng bảng in ra, `aggregate` và `evaluate.summarize` lấy trung bình trên các seed mà từng model chạy xong, nên không dùng để xếp hạng; xếp hạng dùng `tier01.py`. `run.py` in thêm một dòng text ngay sau bảng, format bảng giữ nguyên.

## 2. File đã sửa/thêm

| File | Nội dung |
|---|---|
| `nextgen/src/longitudinal_impute/baselines/same_day_mean.py` | Thêm `use_time`, `use_group`, `weighting`, `residual_rule`. `info_` ghi lại cả bốn tham số. Mặc định giống từng byte bản trước. |
| `nextgen/src/longitudinal_impute/baselines/common.py` | `fill_within_residual(..., rule='cap'\|'rescale')`. Mặc định `cap` như cũ. |
| `nextgen/src/longitudinal_impute/run.py` | Khóa tùy chọn `model = "<tên đã đăng ký>"` (`model_kind`, `build`). `config.json` có thêm `resolved_models` (bảng → model, class, tham số khai báo, tham số hiệu lực). Thêm dòng ghi chú sau bảng in. |
| `nextgen/configs/models_sameday_ablation.toml` | Sáu biến thể, đóng băng trước khi chạy. |
| `nextgen/README.md` | Ghi chú bảng tổng hợp (A2), khóa `model`, các công tắc ablation. |
| `nextgen/tests/test_sameday_ablation.py` | 28 test mới. |
| `tv2_step2/tier01.py`, `test_tier01.py` | `UNDECLARED` (A1), tách hàm `load_run`; thêm 1 test. Bảng step 2 sinh lại. |
| `tv2_step3/run_all.sh`, `ablation.py` (263 dòng), `test_ablation.py` (4 test) | Chạy ablation và dựng bảng. Phần chấm điểm, regression check và ghép cặp đều gọi `tier01`. |
| `tv2_step3/ablation_status.csv`, `ablation_runs.csv`, `ablation_pairs.csv`, `ablation_results.md`, `ablation_manifest.json` | Output sinh tự động. File dự đoán không commit, chỉ commit hash trong manifest. |

## 3. Thiết kế ablation (đóng băng trước khi chạy)

Thêm tham số vào `SameDayMean`. Giá trị mặc định phải tái lập **giống từng byte** output hiện tại.

| Tham số | Mặc định (full) | Giá trị tắt |
|---|---|---|
| `use_time` | `true` | `false`: bỏ cửa sổ thời gian, lấy mọi ngày trong nhóm, không nới |
| `use_group` | `true` | `false`: một nhóm duy nhất cho mọi location |
| `weighting` | `"n"` | `"none"`: trung bình không trọng số của c/N |
| `residual_rule` | `"cap"` | `"rescale"`: luôn chuẩn hóa đề xuất của các ô thiếu cho tổng đúng bằng R, Other = 0 khi có ô thiếu nhận khối lượng (kiểu R1). Nếu tổng đề xuất = 0 thì giữ quy tắc hiện có, không chia 0 ngầm |

Các biến thể chạy (mọi tham số khác giữ mặc định, `window_days=28`, `min_obs=20`, `max_widen=2`, `exclude_same_location=false`):
1. `sdm_full`: mặc định.
2. `sdm_no_time`
3. `sdm_no_group`
4. `sdm_unweighted`
5. `sdm_rescale`
6. `sdm_none`: tắt cả T, G, W, giữ `cap`. So với `mean_fill` để thấy khác biệt chỉ do quy tắc phần dư và định nghĩa trung bình. Ghi rõ hai cái không hoàn toàn trùng định nghĩa.

Đây là ablation bỏ từng yếu tố (one-factor-off) quanh `sdm_full`, cộng một điểm "tắt hết". Không chạy đủ 16 tổ hợp. Không thêm biến thể sau khi xem kết quả. Nếu thấy cần thêm, ghi vào mục "để lại cho step sau".

Quy ước đánh giá giữ như step 2: metric chính là MAE tỷ lệ, cell-weighted, trên nhóm ô `model`. whole_feature là kịch bản chính, random_cell là sensitivity, rate 0.05, seed 0–4, mọi kết quả đều **exploratory**.

## 4. Kiểm tra

1. **Test:** 245 pass với `-W error::RuntimeWarning`: nextgen 227 (199 cũ + 28 mới), `tv2_step1` 8, `tv2_step2` 6, `tv2_step3` 4. Các test mới gồm:
   - **Mặc định giống từng byte bản trước:** hash output tính bằng code ở commit c982e25 (trước khi sửa), trên `covid_small.csv` (ba mask × ba bộ tham số) và `microbiome_small.csv` (time dạng số). Kiểm cả khi không truyền tham số lẫn khi truyền tường minh bốn giá trị mặc định.
   - Từng công tắc cho đúng giá trị tính tay trên một toy 5 dòng, mỗi biến thể một ca.
   - `rescale`: trên fixture, tổng các ô thiếu bằng đúng phần dư R của dòng (rtol 1e-12). Đề xuất có tổng 0, hoặc R = 0, không sinh NaN và không chia cho 0.
   - Khóa `model`: alias chạy đúng class, tên output đúng; kết quả bằng đúng việc gọi class trực tiếp. Tên model sai, kiểu sai, hoặc bảng thiếu khóa `model` thì báo lỗi trước mọi output.
   - Đặt ô hidden = 1e100 không đổi output, với cả 6 biến thể × 2 mask, trên dữ liệu có 2 nhóm.
   - Script: từ chối ghép khi mask khác nhau, hoặc khi `sdm_full` khác `same_day_mean` step 2; khi đó không ghi output nào. Phần trăm đóng góp được kiểm bằng tính tay.
2. **Mask:** với cả hai mask, mask seed 0–4 của lượt ablation giống mask step 2. Data, schema, rate và group map cũng cùng hash. Script kiểm điều này tự động trước khi ghép (`check_same_masks`, theo kiểu F2). Hash nằm trong [ablation_manifest.json](ablation_manifest.json).
3. **Regression check:** `sdm_full` giống từng byte `same_day_mean` của step 2: CSV native giống hệt, và 28 309 (whole_feature) / 27 936 (random_cell) dòng ô che trong file long giống hệt. Metric tính từ file long khớp `summary.json` (sai khác lớn nhất 0) cho cả lượt ablation lẫn lượt step 2.
4. **Mức cửa sổ:** chạy lại các biến thể có cửa sổ trên mask đã lưu. Dự đoán ở ô che khớp file long trước khi đếm mức.
5. **Nguồn gốc:** ablation chạy ở commit 5ca70d8 (`uncommitted_changes = false`). Baseline (hron, mean_fill) đọc từ output step 2, sau khi đã kiểm cùng mask và data.

## 5. Biến thể so với full, và bảng đóng góp

### whole_feature (chính)

<!-- auto:versus_whole_feature:start -->
**whole_feature**, biến thể vs sdm_full (có điều kiện trên tập seed chung của từng cặp; thắng = MAE tỷ lệ nhóm model nhỏ hơn; Δ = A − B, mean ± sd ddof=1)

| A vs B | n seed | A thắng | Δ MAE tỷ lệ | Seed loại (lý do) |
|---|---:|---:|---|---|
| sdm_no_time vs sdm_full | 5 | 0/5 | +0.10025 ± 0.02169 | - |
| sdm_no_group vs sdm_full | 5 | 2/5 | +0.00276 ± 0.00344 | - |
| sdm_unweighted vs sdm_full | 5 | 3/5 | -0.00020 ± 0.00198 | - |
| sdm_rescale vs sdm_full | 5 | 0/5 | +0.05264 ± 0.01137 | - |
| sdm_none vs sdm_full | 5 | 0/5 | +0.10029 ± 0.01889 | - |
| sdm_none vs mean_fill | 5 | 0/5, hòa 5 | +0.00000 ± 0.00000 | - |
| sdm_full vs hron | 5 | 5/5 | -0.05247 ± 0.01578 | - |
<!-- auto:versus_whole_feature:end -->

Mọi metric (bảng chính của step này):

<!-- auto:wide_whole_feature:start -->
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
<!-- auto:wide_whole_feature:end -->

Đóng góp = (metric biến thể − metric full) / (metric `sdm_none` − metric full), tính trên trung bình qua seed 0–4. Đây là **mô tả, không phải phân rã cộng tính**: ablation one-factor-off không cộng được, nên tổng các dòng không phải 100%.

<!-- auto:share_whole_feature:start -->
| Yếu tố tắt | MAE tỷ lệ | MSE tỷ lệ | MAE count | MAE tỷ lệ series-macro | MAE tỷ lệ, đáp án = 0 | MAE tỷ lệ, đáp án > 0 | MAE tỷ lệ, ô model đáp án = 0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| sdm_none − sdm_full (mẫu số) | +0.1003 | +0.05615 | +188.8 | +0.08299 | +0.01453 | +0.2105 | +0.04145 |
| T thời gian (sdm_no_time) | +100% (Δ +0.1) | +96% (Δ +0.0539) | +94% (Δ +178) | +97% (Δ +0.0804) | +111% (Δ +0.0162) | +96% (Δ +0.202) | +111% (Δ +0.0461) |
| G châu lục (sdm_no_group) | +3% (Δ +0.00276) | +2% (Δ +0.000927) | -0% (Δ -0.9) | +6% (Δ +0.00469) | +5% (Δ +0.000763) | +2% (Δ +0.00401) | +5% (Δ +0.00222) |
| W trọng số N (sdm_unweighted) | -0% (Δ -0.000195) | -4% (Δ -0.00225) | +7% (Δ +13) | -1% (Δ -0.000799) | +1% (Δ +9.95e-05) | -1% (Δ -0.00144) | +1% (Δ +0.000317) |
| R phần dư (sdm_rescale) | +52% (Δ +0.0526) | +63% (Δ +0.0354) | +29% (Δ +54.4) | +67% (Δ +0.0552) | +189% (Δ +0.0274) | +1% (Δ +0.00297) | +190% (Δ +0.0789) |
<!-- auto:share_whole_feature:end -->

### random_cell (sensitivity)

<!-- auto:versus_random_cell:start -->
**random_cell**, biến thể vs sdm_full (có điều kiện trên tập seed chung của từng cặp; thắng = MAE tỷ lệ nhóm model nhỏ hơn; Δ = A − B, mean ± sd ddof=1)

| A vs B | n seed | A thắng | Δ MAE tỷ lệ | Seed loại (lý do) |
|---|---:|---:|---|---|
| sdm_no_time vs sdm_full | 5 | 0/5 | +0.10548 ± 0.00223 | - |
| sdm_no_group vs sdm_full | 5 | 0/5 | +0.00297 ± 0.00105 | - |
| sdm_unweighted vs sdm_full | 5 | 4/5 | -0.00054 ± 0.00101 | - |
| sdm_rescale vs sdm_full | 5 | 0/5 | +0.04630 ± 0.00343 | - |
| sdm_none vs sdm_full | 5 | 0/5 | +0.10616 ± 0.00166 | - |
| sdm_none vs mean_fill | 5 | 0/5, hòa 5 | +0.00000 ± 0.00000 | - |
| sdm_full vs hron | 5 | 5/5 | -0.05825 ± 0.00489 | - |
<!-- auto:versus_random_cell:end -->

<!-- auto:wide_random_cell:start -->
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
<!-- auto:wide_random_cell:end -->

<!-- auto:share_random_cell:start -->
| Yếu tố tắt | MAE tỷ lệ | MSE tỷ lệ | MAE count | MAE tỷ lệ series-macro | MAE tỷ lệ, đáp án = 0 | MAE tỷ lệ, đáp án > 0 | MAE tỷ lệ, ô model đáp án = 0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| sdm_none − sdm_full (mẫu số) | +0.1062 | +0.05811 | +133.4 | +0.09012 | +0.01892 | +0.2113 | +0.04837 |
| T thời gian (sdm_no_time) | +99% (Δ +0.105) | +96% (Δ +0.0555) | +92% (Δ +123) | +99% (Δ +0.0895) | +113% (Δ +0.0214) | +94% (Δ +0.198) | +113% (Δ +0.0548) |
| G châu lục (sdm_no_group) | +3% (Δ +0.00297) | +3% (Δ +0.00175) | +7% (Δ +9.36) | +4% (Δ +0.00347) | +2% (Δ +0.000383) | +3% (Δ +0.00657) | +2% (Δ +0.000987) |
| W trọng số N (sdm_unweighted) | -1% (Δ -0.000538) | -3% (Δ -0.00148) | +6% (Δ +7.56) | -0% (Δ -0.000411) | -2% (Δ -0.0003) | -0% (Δ -0.000114) | -2% (Δ -0.000767) |
| R phần dư (sdm_rescale) | +44% (Δ +0.0463) | +59% (Δ +0.0342) | +43% (Δ +57.3) | +56% (Δ +0.0505) | +137% (Δ +0.0259) | +5% (Δ +0.0101) | +137% (Δ +0.0661) |
<!-- auto:share_random_cell:end -->

### sdm_none so với mean_fill

<!-- auto:none_whole_feature:start -->
sdm_none vs mean_fill, whole_feature: 28309 / 28309 ô che dự đoán giống hệt; chênh lớn nhất 0 (count), 0 (tỷ lệ).
<!-- auto:none_whole_feature:end -->

<!-- auto:none_random_cell:start -->
sdm_none vs mean_fill, random_cell: 27936 / 27936 ô che dự đoán giống hệt; chênh lớn nhất 0 (count), 0 (tỷ lệ).
<!-- auto:none_random_cell:end -->

## 6. Mức cửa sổ (chỉ ô che thuộc nhóm model)

Mỗi ô ghi: gốc / nới 1 / nới 2 / toàn cục. Chỉ gồm các biến thể có dùng cửa sổ thời gian. Số ô ở từng mức phụ thuộc vào số ô visible, không phụ thuộc trọng số hay quy tắc phần dư, nên `sdm_unweighted` và `sdm_rescale` có số giống `sdm_full`.

**whole_feature**

<!-- auto:levels_whole_feature:start -->
| Biến thể | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | Tổng: toàn cục / ô model |
|---|---|---|---|---|---|---:|
| sdm_full | 2552 / 6 / 0 / 37 | 2399 / 149 / 66 / 115 | 2247 / 106 / 53 / 116 | 2218 / 121 / 26 / 139 | 2142 / 24 / 23 / 217 | 624 / 12756 (4.9%) |
| sdm_no_group | 2595 / 0 / 0 / 0 | 2729 / 0 / 0 / 0 | 2522 / 0 / 0 / 0 | 2504 / 0 / 0 / 0 | 2406 / 0 / 0 / 0 | 0 / 12756 (0.0%) |
| sdm_unweighted | 2552 / 6 / 0 / 37 | 2399 / 149 / 66 / 115 | 2247 / 106 / 53 / 116 | 2218 / 121 / 26 / 139 | 2142 / 24 / 23 / 217 | 624 / 12756 (4.9%) |
| sdm_rescale | 2552 / 6 / 0 / 37 | 2399 / 149 / 66 / 115 | 2247 / 106 / 53 / 116 | 2218 / 121 / 26 / 139 | 2142 / 24 / 23 / 217 | 624 / 12756 (4.9%) |

Ô thiếu thật (lượt native), mức toàn cục:
- sdm_full: 21946 / 86050 (25.5%); window 51136, widened_1 4912, widened_2 8056, global 21946
- sdm_no_group: 0 / 86050 (0.0%); window 86050, widened_1 0, widened_2 0, global 0
<!-- auto:levels_whole_feature:end -->

**random_cell**

<!-- auto:levels_random_cell:start -->
| Biến thể | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | Tổng: toàn cục / ô model |
|---|---|---|---|---|---|---:|
| sdm_full | 2661 / 104 / 63 / 23 | 2560 / 93 / 68 / 19 | 2566 / 92 / 76 / 28 | 2651 / 102 / 60 / 19 | 2570 / 97 / 61 / 22 | 111 / 13935 (0.8%) |
| sdm_no_group | 2851 / 0 / 0 / 0 | 2740 / 0 / 0 / 0 | 2762 / 0 / 0 / 0 | 2832 / 0 / 0 / 0 | 2750 / 0 / 0 / 0 | 0 / 13935 (0.0%) |
| sdm_unweighted | 2661 / 104 / 63 / 23 | 2560 / 93 / 68 / 19 | 2566 / 92 / 76 / 28 | 2651 / 102 / 60 / 19 | 2570 / 97 / 61 / 22 | 111 / 13935 (0.8%) |
| sdm_rescale | 2661 / 104 / 63 / 23 | 2560 / 93 / 68 / 19 | 2566 / 92 / 76 / 28 | 2651 / 102 / 60 / 19 | 2570 / 97 / 61 / 22 | 111 / 13935 (0.8%) |

Ô thiếu thật (lượt native), mức toàn cục:
- sdm_full: 21946 / 86050 (25.5%); window 51136, widened_1 4912, widened_2 8056, global 21946
- sdm_no_group: 0 / 86050 (0.0%); window 86050, widened_1 0, widened_2 0, global 0
<!-- auto:levels_random_cell:end -->

## 7. Kết luận có giới hạn

Exploratory: masked pilot trên seed 0–4 (đã được nhìn ở step 1–2), dựng lại bảng cố định. sd là độ lệch chuẩn mẫu qua 5 seed, **không phải khoảng tin cậy**. Chưa có bootstrap.

1. **Thời gian (T) là yếu tố quan trọng nhất.** Bỏ cửa sổ thời gian làm MAE tỷ lệ tăng khoảng 0.10 ở cả hai mask, thua cả 5/5 seed, tương đương ~100% khoảng cách full → none. `sdm_no_time` gần như về mức `mean_fill` (whole_feature 0.1353 so với 0.1354). Không có thời gian thì same_day_mean chỉ còn ngang mức sàn.
2. **Quy tắc phần dư (R) đứng thứ hai, và tác động gần như dồn vào ô có đáp án 0.**
   - Đổi `cap` sang `rescale` làm MAE tăng 0.053 (whole_feature) và 0.046 (random_cell), thua 0/5 ở cả hai mask.
   - Trên ô model có đáp án 0, sai số tăng mạnh. Trên ô đáp án > 0 gần như không đổi: thắng 1/5 ở cả hai mask, Δ +0.003 và +0.010.
   - Diễn giải (chưa kiểm chứng riêng): `cap` để Other giữ phần dư, nên ô có đáp án 0 nhận giá trị nhỏ; `rescale` ép toàn bộ R vào các ô thiếu.
   - Khi dùng `rescale`, same_day_mean xuống ngang Hron ở whole_feature (0.0877 so với 0.0875).
3. **Châu lục (G) đóng góp nhỏ:** khoảng +0.003 MAE (~3%). Ở random_cell, full thắng 5/5. Ở whole_feature thì không ổn định: `sdm_no_group` thắng 2/5. MAE count trung bình của nó thấp hơn một chút (Δ −0.9), nhưng chỉ thắng 2/5 seed.
   - Bỏ nhóm thì không ô nào phải rơi về trung bình toàn cục: 0% so với 25,5% trên ô thiếu thật, và 4,9% trên ô che model ở whole_feature. Đổi lại, độ chính xác trên ô che hơi kém đi.
   - Đây là dữ kiện về support, không phải về độ chính xác trên ô thiếu thật.
4. **Trọng số N (W) gần như không ảnh hưởng MAE tỷ lệ, và có lợi cho thang count.**
   - `sdm_unweighted` có MAE tỷ lệ thấp hơn rất ít (Δ −0.0002 và −0.0005; thắng 3/5 và 4/5) và MSE tỷ lệ thấp hơn 5/5 ở cả hai mask.
   - Ngược lại, MAE count của nó cao hơn (full thắng 4/5 và 5/5).
   - Vì vậy `sdm_unweighted` đứng đầu bảng theo trung bình MAE, nhưng khoảng cách nằm trong độ dao động giữa các seed. Đây là quan sát sau khi xem kết quả, nên **không đổi mặc định**.
5. **Series-macro so với cell-weighted:** thứ hạng của 8 model (6 biến thể, hron, mean_fill) **không đổi** ở cả hai mask.
6. **`sdm_none` so với `mean_fill` không nói được gì về quy tắc phần dư.**
   - Hai cái cho dự đoán giống hệt trên mọi ô che ở cả hai mask (28 309 và 27 936 ô, chênh lệch 0).
   - Về định nghĩa hai cái không trùng hoàn toàn. `sdm_none` tính bằng tổng tiền tố trên một "cửa sổ" phủ cả bảng, có ngưỡng `min_obs` rồi mới về toàn cục, và cắt đề xuất về [0, 1]. `mean_fill` dùng `np.nanmean`. Trên COVID, mọi target đều có ≥ 20 ô visible nên cả hai cùng tính trung bình không trọng số của c/N trên toàn bảng, rồi cùng qua `fill_within_residual` với quy tắc `cap`.
   - Vậy cặp này không tách được gì. Tác động của quy tắc phần dư được đo bằng `sdm_rescale` so với `sdm_full` (mục 2).

## 8. Quyết định thiết kế chưa chắc

1. **Bảng đóng góp chỉ mang tính mô tả.** Ở whole_feature, T (~100%) cộng R (~52%) đã vượt 100%: các yếu tố tương tác, và one-factor-off không đo được tương tác. Ngoài ra `sdm_none` vẫn giữ `cap`, nên mẫu số không chứa R; phần trăm của R là so với một khoảng cách mà R không nằm trong đó.
2. **`use_time=false`** nghĩa là một cửa sổ phủ mọi ngày trong nhóm, vẫn áp `min_obs` rồi về trung bình toàn cục, không nới. **`weighting="none"`** đổi cả mức trung bình toàn cục sang không trọng số.
3. **`rescale`** áp cho mọi ô thiếu của dòng, gồm cả ô thiếu thật trong lượt masked, không riêng ô bị che. Ô che và ô thiếu thật trong cùng một dòng chia nhau R.
4. **Không mở rộng `tier01.py`** (đã 418 dòng; thêm ablation sẽ vượt 500). Tôi viết `tv2_step3/ablation.py`, chỉ để ghép bảng. Script này gọi `tier01.load_run`, `pair`, `ranking`, `main_markdown`, `macro_markdown`; điểm số vẫn từ `cells.score_predictions`, nên không có bộ chấm điểm thứ hai. Phần mới chỉ gồm kiểm tra mask/byte, bảng đóng góp và mức cửa sổ.
5. **Mức cửa sổ lấy từ việc chạy lại model** trên mask đã lưu, có kiểm tra với file long, thay vì lưu `levels_` khi chạy. Tôi không đổi định dạng file long của step 2.
6. **`resolved_models.params`** chỉ chứa tham số khai báo trong TOML; `effective_params` chứa thêm mặc định của constructor. Nếu sau này mặc định trong code đổi, `effective_params` của các run cũ vẫn giữ giá trị lúc chạy.
7. **Cặp `sdm_none` vs `mean_fill`** cho kết quả suy biến (mục 7.6). Thiết kế đã đóng băng nên tôi không thêm biến thể để sửa (xem mục 9).

## 9. Việc để lại cho step sau

- **Seed xác nhận mới, cố định trước.** Đặc biệt để kiểm lợi thế nhỏ của `sdm_unweighted` (MAE, MSE tỷ lệ), trước khi bàn đến việc đổi mặc định.
- **Tương tác giữa các yếu tố:** một vài tổ hợp hai yếu tố (T×R, G×R), hoặc đủ 16 tổ hợp nếu cần phân rã. Một biến thể kiểu "chỉ có R, không T/G/W, `rescale`" để so với `mean_fill`, vì cặp `sdm_none`/`mean_fill` hiện không tách được quy tắc phần dư.
- **Kiểm chứng giả thuyết về R:** sai số của `rescale` dồn vào ô có đáp án 0; thử tách theo việc dòng có Other > 0 hay không.
- Sensitivity theo `window_days`, `min_obs`, `exclude_same_location`, theo hướng giảm tỷ lệ 25,5% rơi về trung bình toàn cục trên ô thiếu thật (ví dụ dùng nhóm khi đủ ô, cả bảng khi thiếu).
- Mang sang từ step 2: bootstrap/CI, diagnostic Tầng 2–4, mã lỗi cụ thể hơn cho Hron/Tsagris, chính sách khi whole_feature xóa hết dòng đầy đủ, MSE count của MICoDa A18.
