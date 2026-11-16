# Step 2: Evaluation Tầng 0–1 tối thiểu (exploratory)

> **Ghi chú sau khi rebase lên main mới (PR #5):** micoda trong các step này = A17 (other=false); micoda_other = A18. Trên main sau PR #5, micoda mặc định = A18. Để chạy lại đúng A17, dùng `configs/models_default_a17.toml` (bảng `[micoda]` ghi tường minh `other = false`); tên model trong các bảng kết quả giữ nguyên như lúc chạy. Đường dẫn đã đổi theo cấu trúc mới (`nextgen/src` → `src`, `nextgen/configs` → `configs`, `nextgen/tests` → `tests`); hash commit nhắc trong báo cáo là hash trước khi rebase; bảng ánh xạ cũ → mới ở `tv2_step8/HASH_MAP.md` (PR step 5–8).

Branch `step2-tier01-eval` (từ `step1-decoys-sameday` 14818f3). Không đổi thuật toán của model nào. Mọi bảng số trong các khối `auto:*` do `tier01.py` sinh, không sửa tay. Bảng đầy đủ (mọi cặp, cả random_cell): [tier01_results.md](tier01_results.md).

## 1. File đã sửa/thêm

| File | Nội dung |
|---|---|
| `src/longitudinal_impute/evaluate.py` | Tách `cell_errors(truth, pred, N, forced)` để `masked_errors` và phần chấm từ file long dùng chung một định nghĩa (định nghĩa không đổi, test cũ pass). Thêm `GROUPS`, `METRICS`, `series_macro_mae`. Nhóm rỗng → `None` (NA). |
| `src/longitudinal_impute/cells.py` (mới) | File long: `prediction_rows`, `write_predictions` (csv.gz, float dạng `repr` nên đọc lại không mất chính xác, header gzip cố định nên cùng nội dung thì cùng hash), `read_predictions`, `score_predictions`. |
| `src/longitudinal_impute/run.py` | Thêm cờ `--save-predictions` (opt-in, không bật thì output như cũ). Run failed/partial ghi thêm `error_code`. Mask stats ghi thêm `forced_zero_cells` và `model_cells` (tính bằng `evaluate.forced_zero`); `native_cells` cho lượt native. |
| `tests/test_tier01_eval.py` | 22 test mới. |
| `tv2_step1/summarize_runs.py` | `validate_pair` (F2) nhận thêm danh sách seed, mặc định vẫn là 0–4. |
| `tv2_step2/run_all.sh` | Chạy 7 model rồi lượt riêng `micoda_other` (A18), rate 0.05, `--save-predictions`; whole_feature thêm seed 109. |
| `tv2_step2/tier01.py`, `test_tier01.py` | Script tổng hợp Tầng 0–1 (408 dòng) và 5 test. |
| `tv2_step2/tier01_status.csv`, `tier01_runs.csv`, `tier01_pairs.csv`, `tier01_results.md`, `tier01_manifest.json` | Output sinh tự động. File dự đoán `predictions_<mask>.csv.gz` (0,5–3,6 MB) không commit; chỉ commit hash trong manifest. |

## 2. Quy ước đã chốt cho step này

- **Mục tiêu đánh giá:** dựng lại bảng cố định (retrospective), theo tỷ lệ. Không tuyên bố tổng quát hóa sang nước mới.
- **Metric chính:** MAE tỷ lệ trên nhóm ô `model`, cell-weighted trong từng seed. So model bằng hiệu ghép cặp A−B theo seed trên cùng mask, báo trung bình, sample SD (ddof=1) và số seed thắng.
- **Metric phụ cố định:** MSE tỷ lệ, MAE count, series-macro MAE tỷ lệ (trung bình MAE trong từng cặp location×target bị che, rồi trung bình đều qua chuỗi), MAE tỷ lệ theo nhóm `truth_zero` / `truth_positive` / `model_truth_zero`. Không tạo composite score. Không tính RMSE.
- **Mask:** whole_feature là kịch bản chính, random_cell là sensitivity. Rate 0.05.
- **Seed:** 0–4 đã được nhìn và chỉnh nên mọi kết quả đều gắn nhãn **exploratory**. Thêm seed 109 (whole_feature) làm ca thử chính sách failure.
- **Budget:** giữ nguyên. MICoDa A17 mặc định, A18 (`micoda_other`) là sensitivity có tên riêng. Không đổi R1/R2.

Cách áp dụng: một run chỉ được tính khi `completed` và coverage = 1. Bảng metric lấy trung bình trên **tập seed chung của mọi model**. Mỗi cặp so trên **tập seed chung của riêng cặp đó**. Bảng reliability đếm mọi run đã lên lịch. Series-macro tính trên nhóm ô `model`, mỗi chuỗi là một cặp (location, target) có ít nhất một ô model.

## 3. Kiểm tra

1. **Test:** 212 pass với `-W error::RuntimeWarning`: nextgen 199 (177 cũ + 22 mới), `tv2_step1` 8 (cũ), `tv2_step2` 5 (mới). Các test được yêu cầu:
   - đổi giá trị ô hidden (đặt về 0) thì input model không đổi: so mọi đối số của `fit`/`predict`, và với MICoDa là input của lõi `MICoDa.fit`; 8 model × 2 mask. Kiểm tra ngược bằng cách sửa tạm `model_input` cho lộ ô hidden: 12/16 ca fail, đúng như mong đợi. 4 ca còn lại là MICoDa, vốn đi qua adapter riêng chứ không qua `model_input`. Đã hoàn nguyên.
   - nhóm rỗng trả NA, không trả 0 (`cell_errors`, `series_macro_mae`, cặp không có seed chung);
   - micro và series-macro trên toy tính tay: 0.2 và 0.3;
   - toy F07: A = [0.10, 0.80], B = [0.20, fail] → cặp chỉ dùng seed 0, A thắng 1/1; bảng xếp hạng đặt A trước B;
   - run failed và run lên lịch mà không có kết quả (`missing`) vẫn có mặt trong bảng trạng thái;
   - metric từ file long khớp `masked_errors`/`summary.json` (fixture, hai mask, mọi nhóm, ngưỡng 1e-12); file long giống từng byte giữa hai lần chạy.
2. **Regression check bắt buộc** (trong `tier01.py`; lệch thì dừng trước khi ghi bất kỳ output nào): whole_feature 1320 giá trị, random_cell 1200 giá trị. Sai khác lớn nhất **0**, ngưỡng 1e-12.
3. **Tái dùng output step 1:** không tái dùng được, vì step 1 không lưu dự đoán trên ô che (chỉ có `summary.json` và CSV của lượt native). Tôi đã chạy lại. Mask seed 0–4 có cùng hash với `tv2_step1/results_manifest.json`. Status và `errors` của cả 40 run mỗi mask trùng khớp hoàn toàn với `summary.json` của step 1. Nói cách khác, số liệu 5 seed của step 1 được tái lập đúng hoàn toàn.
4. **Nguồn gốc:** lượt main chạy ở commit 7ef9146, lượt micoda_other ở 8978d72. `nextgen/` giống hệt giữa hai commit (`git diff` rỗng) và cả hai lượt đều có `uncommitted_changes = false`. F2 (`validate_pair`) đã kiểm tra rồi mới ghép hai lượt. Hash của data, schema, mask, map, models, config.json, summary.json và file dự đoán nằm trong [tier01_manifest.json](tier01_manifest.json).

## 4. Bảng trạng thái và reliability

Bảng trạng thái từng run: [tier01_status.csv](tier01_status.csv) (cột `status`, `coverage`, `error_code`, `error_message`, `converged`, `termination_reason`, `iterations`, `hidden_cells`, `forced_zero_cells`, `model_cells`, `complete_rows_after`).

**whole_feature** (mọi run đã lên lịch):

<!-- auto:status_whole_feature:start -->
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
<!-- auto:status_whole_feature:end -->

**random_cell:** chi tiết trong [tier01_results.md](tier01_results.md).

<!-- auto:status_random_cell:start -->
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
<!-- auto:status_random_cell:end -->

### Seed 109

<!-- auto:seed109:start -->
Seed 109 (whole_feature): 5626 ô che, 2726 ép về 0, 2900 ô model; 0 dòng đầy đủ sau khi che.

| Model | status | coverage | mã lỗi | thông điệp | hội tụ (MICoDa) |
|---|---|---:|---|---|---|
| hron | failed | 0.000 | baseline_error | BaselineError: row 0, feature 0: no donor observed on O_i and j | - |
| tsagris | failed | 0.000 | baseline_error | BaselineError: Tsagris needs at least one complete donor row | - |
| micoda | failed | 0.000 | insufficient_fit | insufficient_fit: Complete-case initialization needs more than one row; context={} | - |
| same_day_mean | completed | 1.000 | - | - | - |
| mean_fill | completed | 1.000 | - | - | - |
| marginal_sampler | completed | 1.000 | - | - | - |
| zero_fill | completed | 1.000 | - | - | - |
| micoda_other | failed | 0.000 | insufficient_fit | insufficient_fit: Complete-case initialization needs more than one row; context={} | - |
<!-- auto:seed109:end -->

- Seed 109 không còn dòng đầy đủ nào sau khi che. Tsagris và MICoDa (cả A17 lẫn A18) fail như dự kiến. ⚠️ **Hron cũng fail** (`no donor observed on O_i and j`), điều mà đề bài không nêu trước.
- same_day_mean và ba mức sàn chạy xong. Seed này chỉ có trong bảng trạng thái và reliability (xem *Sửa sau review*).
- Hron và Tsagris báo mã `baseline_error`, tức mã của lớp gốc, kém cụ thể hơn `insufficient_data`. Không sửa vì đó là code model.
- Mọi run MICoDa completed đều `converged = True` (`termination_reason = tolerance`, 45–71 vòng).

## 5. Bảng metric chính và ghép cặp

### whole_feature (chính)

<!-- auto:main_whole_feature:start -->
Tập seed chung (mọi model completed, coverage = 1): 0, 1, 2, 3, 4.
Seed không khai báo trước (109): chỉ có trong bảng trạng thái và reliability, không vào bảng metric hay hiệu ghép cặp.

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
<!-- auto:main_whole_feature:end -->

### random_cell (sensitivity)

<!-- auto:main_random_cell:start -->
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
<!-- auto:main_random_cell:end -->

Vì seed 0–4 và các run trùng với step 1, số liệu trên tập seed 0–4 trùng với bảng step 1. Điểm mới là mọi model đều được tính trên **cùng một tập seed**.

### Ghép cặp (khối thắng/thua, sinh tự động)

<!-- auto:pairs_whole_feature:start -->
**whole_feature**, cặp chính (có điều kiện trên tập seed chung của từng cặp; thắng = MAE tỷ lệ nhóm model nhỏ hơn; Δ = A − B, mean ± sd ddof=1)

| A vs B | n seed | A thắng | Δ MAE tỷ lệ | Seed loại (lý do) |
|---|---:|---:|---|---|
| same_day_mean vs hron | 5 | 5/5 | -0.05247 ± 0.01578 | - |
| tsagris vs hron | 5 | 0/5 | +0.02036 ± 0.01274 | - |
| micoda_other vs mean_fill | 5 | 5/5 | -0.02785 ± 0.00699 | - |
| micoda vs mean_fill | 5 | 1/5 | +0.00525 ± 0.01350 | - |
| micoda_other vs micoda | 5 | 5/5 | -0.03309 ± 0.00681 | - |
| hron vs mean_fill | 5 | 5/5 | -0.04782 ± 0.00853 | - |
<!-- auto:pairs_whole_feature:end -->

<!-- auto:pairs_random_cell:start -->
**random_cell**, cặp chính (có điều kiện trên tập seed chung của từng cặp; thắng = MAE tỷ lệ nhóm model nhỏ hơn; Δ = A − B, mean ± sd ddof=1)

| A vs B | n seed | A thắng | Δ MAE tỷ lệ | Seed loại (lý do) |
|---|---:|---:|---|---|
| same_day_mean vs hron | 5 | 5/5 | -0.05825 ± 0.00489 | - |
| tsagris vs hron | 5 | 0/5 | +0.01070 ± 0.00465 | - |
| micoda_other vs mean_fill | 5 | 5/5 | -0.02931 ± 0.00139 | - |
| micoda vs mean_fill | 5 | 5/5 | -0.00537 ± 0.00256 | - |
| micoda_other vs micoda | 5 | 5/5 | -0.02394 ± 0.00184 | - |
| hron vs mean_fill | 5 | 5/5 | -0.04791 ± 0.00405 | - |
<!-- auto:pairs_random_cell:end -->

Mọi cặp (28 cặp × 7 metric × 2 mask) nằm trong [tier01_pairs.csv](tier01_pairs.csv). Bảng 28 cặp theo metric chính nằm trong [tier01_results.md](tier01_results.md).

## 6. Series-macro so với cell-weighted

**whole_feature**

<!-- auto:macro_whole_feature:start -->
Seed chung: 0, 1, 2, 3, 4. Cell-weighted: same_day_mean < hron < micoda_other < tsagris < mean_fill < marginal_sampler < micoda < zero_fill.
Series-macro: same_day_mean < hron < micoda_other < tsagris < mean_fill < marginal_sampler < micoda < zero_fill.
Thứ hạng **không đổi**.
<!-- auto:macro_whole_feature:end -->

**random_cell**

<!-- auto:macro_random_cell:start -->
Seed chung: 0, 1, 2, 3, 4. Cell-weighted: same_day_mean < hron < tsagris < micoda_other < micoda < marginal_sampler < mean_fill < zero_fill.
Series-macro: same_day_mean < hron < tsagris < micoda_other < marginal_sampler < mean_fill < micoda < zero_fill.
- Đổi chỗ micoda / marginal_sampler: cell-weighted micoda thắng 2/5 (Δ -0.00449); series-macro micoda thắng 1/5 (Δ +0.00495).
- Đổi chỗ micoda / mean_fill: cell-weighted micoda thắng 5/5 (Δ -0.00537); series-macro micoda thắng 0/5 (Δ +0.00348).
<!-- auto:macro_random_cell:end -->

Đọc: hai lần đổi chỗ đều nằm trong nhóm sát mức sàn (|Δ| ≈ 0.003–0.005). Ở random_cell, MICoDa A17 thắng mean_fill 5/5 khi tính cell-weighted nhưng thua 5/5 khi tính series-macro. Vì vậy lợi thế của A17 so với mức sàn không bền theo cách gộp. Ba hạng đầu (same_day_mean, hron, rồi tsagris hoặc micoda_other) không đổi.

## 7. Kết luận có giới hạn

Mọi điều dưới đây là **exploratory**: masked pilot trên seed 0–4 đã được nhìn trước, dựng lại bảng cố định, không suy ra cho ô thiếu thật hay cho nước mới. sd là độ lệch chuẩn mẫu qua seed, **không phải khoảng tin cậy**. Chưa có bootstrap.

1. **same_day_mean đứng đầu trên metric chính ở cả hai mask.** Nó thắng mọi model khác ở cả 5 seed khai báo trước (5/5). Lợi thế vẫn đến từ bốn yếu tố cùng lúc như ở step 1; chưa có ablation.
2. **Hron xếp trên Tsagris theo metric chính** (MAE tỷ lệ: 5/5 seed ở cả hai mask). Theo MSE tỷ lệ thì ngược lại (xem step 1 và `tier01_pairs.csv`). Thứ hạng này là hệ quả của việc chốt MAE làm metric chính, không phải bằng chứng Hron tốt hơn về mọi mặt.
3. **micoda_other và tsagris không phân định được ở whole_feature** (2/5, Δ −0.0004). Ở random_cell, tsagris thắng 5/5.
4. **MICoDa A17 không vượt chắc mức sàn:** ở whole_feature, mean_fill thắng nó 4/5. Ở random_cell, nó thắng mean_fill 5/5 theo cell-weighted nhưng thua 5/5 theo series-macro. mean_fill và marginal_sampler gần như hòa.
5. **Chính sách failure (F07):** với seed 109, 4 trong 8 model fail. Bảng xếp hạng không còn thưởng cho model fail: trước đây, trung bình riêng phần success của Hron tính trên 5 seed, còn same_day_mean trên 6. Mức fail này là tính chất của một seed được chọn có chủ đích, không phải ước lượng tần suất fail.

## 8. Quyết định thiết kế chưa chắc

1. **Tập seed chung cho bảng metric là giao của cả 8 model.** Chỉ cần một model fail là seed đó bị loại cho mọi model. Hiện bảng metric dùng đúng 0–4. Cách thay thế là xếp hạng bằng số cặp thắng.
2. **Mỗi cặp dùng tập seed chung riêng** trong các seed khai báo trước (0–4). Bản đầu của report này có dùng seed 109 cho các cặp không fail ở seed đó; đã bỏ (xem *Sửa sau review*).
3. **Cột cờ nhóm `model` được đặt tên `model_cell`**, vì cột `model` đã là tên model.
4. **Bất biến input chỉ được kiểm ở đầu vào của phần fit.** Adapter MICoDa (`impute_counts`) vẫn nhận `data.counts` đầy đủ cùng `hidden` và tự che bên trong (thiết kế D07 có sẵn). Test kiểm tra input của lõi sau khi che, không kiểm chính adapter.
5. **`evaluate.summarize` và bảng in của `run.py`** vẫn lấy trung bình trên các seed success của từng model. Tôi giữ nguyên để không phá output và test cũ. Các bảng này không sắp thứ hạng và step 2 không dùng chúng, nhưng vẫn có thể bị đọc nhầm (ví dụ hron 5/1).
6. **Coverage của lượt native:** 1.0 nếu completed. Nếu partial thì lấy từ `resolved_cells / missing_cells` của MICoDa. Các trường hợp còn lại là NA.
7. Series-macro chỉ tính trên nhóm `model`. Ở random_cell phần lớn chuỗi chỉ có 1–2 ô, nên series-macro gần với trung bình theo location hơn là theo chuỗi.
8. File dự đoán xác định theo byte (header gzip cố định). Điều này đã được kiểm trên fixture; chưa chạy lại COVID hai lần để so hash.

## Sửa sau review (step 3, phần A)

Bảng metric và mọi hiệu ghép cặp giờ chỉ dùng seed khai báo trước (0–4). Seed 109 được chọn vì nó gây fail, nên chỉ còn trong bảng trạng thái và reliability; cột MAE cũng đã bỏ khỏi bảng seed 109. Các bảng trên đã được sinh lại. Chỉ các cặp trước đây có n = 6 thay đổi: same_day_mean với ba mức sàn và các mức sàn với nhau. Chúng đổi từ 6/6 thành 5/5, còn mean_fill vs marginal_sampler đổi từ 3/6 thành 2/5 (Δ ≈ 0 ở cả hai). Bảng metric chính vốn đã chỉ dùng 0–4 nên không đổi. **Không kết luận nào đổi.**

## 9. Việc để lại cho step sau

- Seed mới, chọn và cố định **trước** khi xem kết quả, dùng cho bảng xác nhận. Hiện mọi kết quả đều exploratory.
- Chính sách khi mask whole_feature xóa hết dòng đầy đủ: mask generator phải giữ lại dòng đầy đủ, hay model phải có fallback? Đây là việc cần quyết định, không thuộc đánh giá.
- Mã lỗi cụ thể hơn cho Hron/Tsagris (`baseline_error` → `insufficient_data`); bỏ hoặc đổi tên trung bình success-only trong `run.py`.
- Bootstrap/CI, WD/JSD/F1, diagnostic Tầng 2–4, ablation same_day_mean (đều nằm ngoài phạm vi step này).
- Từ step 1: MSE count bất thường của MICoDa A18; tên biến thể trong `run.py`.
