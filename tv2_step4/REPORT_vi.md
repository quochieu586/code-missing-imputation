# Step 4: so sánh công bằng theo quy tắc phần dư (exploratory)

> **Ghi chú sau khi rebase lên main mới (PR #5):** micoda trong các step này = A17 (other=false); micoda_other = A18. Trên main sau PR #5, micoda mặc định = A18. Để chạy lại đúng A17, dùng `configs/models_default_a17.toml` (bảng `[micoda]` ghi tường minh `other = false`); tên model trong các bảng kết quả giữ nguyên như lúc chạy. Đường dẫn đã đổi theo cấu trúc mới (`nextgen/src` → `src`, `nextgen/configs` → `configs`, `nextgen/tests` → `tests`); hash commit nhắc trong báo cáo là hash trước khi rebase; bảng ánh xạ cũ → mới ở `tv2_step8/HASH_MAP.md` (PR step 5–8).

Branch `step4-residual-rule` (từ `step3-sameday-ablation`). Không sửa thuật toán bên trong Hron/Tsagris/MICoDa/same_day_mean; không đổi mặc định của model nào; A18 vẫn chỉ là sensitivity (D08 không đổi). Mọi bảng số trong khối `auto:*` do `residual.py` sinh, không sửa tay. Bảng đầy đủ cả hai mask: [residual_results.md](residual_results.md).

## 1. File đã sửa/thêm

| File | Nội dung |
|---|---|
| `src/longitudinal_impute/data.py` | `apply_residual_rule(data, composition, hidden, rule)`: hàm dùng chung cho ba quy tắc, chỉ ghép `fill_within_residual` với `restore_counts`, không thêm tolerance mới. Chỉ đọc các dòng có ô thiếu. |
| `src/longitudinal_impute/run.py` | Khóa tùy chọn `residual_rule` cho `hron`, `tsagris`, `micoda` có `other = true`, xử lý ở tầng restore. Quy tắc không áp dụng được thì báo lỗi kèm lý do trước mọi side effect. `resolved_models` ghi thêm `runner_residual_rule`. |
| `src/longitudinal_impute/baselines/micoda.py` | Chỉ thêm thuộc tính `proposals_`, lưu composition của lõi để runner dùng. Không đổi phép tính nào; nhánh R2 cũ vẫn chạy cho mặc định. |
| `src/longitudinal_impute/evaluate.py` | `implied_other_errors`: sai số của Other ngầm trên các dòng đầy đủ trước khi che. |
| `configs/models_residual_rule.toml`, `models_residual_check.toml` | Sáu biến thể mới (đóng băng trước khi chạy) và ba lượt kiểm tra `prop_other`. |
| `README.md` | Mục `residual_rule`. |
| `tests/test_residual_rule.py` | 42 test mới. |
| `tv2_step4/run_all.sh`, `residual.py` (283 dòng), `test_residual.py` (3 test) | Chạy và dựng bảng. Phần đọc run, chấm điểm, regression check và ghép cặp đều gọi `tier01` và `ablation`. |
| `tv2_step4/residual_runs.csv`, `residual_pairs.csv`, `residual_results.md`, `residual_manifest.json` | Output sinh tự động. File dự đoán không commit; hash nằm trong manifest. |

## 2. Ba quy tắc và composition p của từng model

**Định nghĩa (đóng băng):** Với một dòng: N là tổng, V là tổng count quan sát của target, R = N − V ≥ 0, U là tập target thiếu. Model cho composition dự đoán p (đóng về 1 trên D+1 phần nếu có Other, trên D phần nếu không), đề xuất q_j = N·p_j.
- **`prop_other`:** chia R theo tỷ lệ giữa {q_j, j∈U} và q_Other. Cần dự đoán Other.
- **`cap`:** nếu Σ_U q_j > R thì co các q_j về tổng R và Other = 0; ngược lại giữ q_j, Other = R − Σ_U q_j. Không dùng q_Other.
- **`rescale`:** chuẩn hóa {q_j, j∈U} cho tổng đúng bằng R, Other = 0. Nếu Σ_U q_j = 0 thì giữ quy tắc hiện có của code, không chia 0 ngầm.

`prop_other` và `rescale` chỉ dùng **tỷ lệ** giữa các phần, nên không phụ thuộc thang của p. Chỉ `cap` phụ thuộc thang tuyệt đối, vì nó so Σ_U q_j với R.

**p cụ thể của từng model** (đọc code, rồi đo trên COVID ở bảng bên dưới):

| Model | p là gì | Đóng trên | Hệ quả |
|---|---|---|---|
| hron | Output `predict` trên D+1 phần: ô quan sát là c/N (ô 0 đã thay bằng δ = min dương / 2), ô thiếu là `median(donor_lj / scale_l) · scale_i`. Median adjust (Eq. 7) đưa giá trị điền về thang của các phần quan sát của chính dòng i, nhưng **không ràng buộc tổng dòng**. | Không đóng. Tổng dòng trên COVID từ 0.0003 (random_cell: 2.3e-6) đến 3.5e5, median 1. Với `cap`/`rescale`, hàm chung đóng dòng về 1 trên D+1 trước khi chia. | `prop_other` và `rescale` không bị ảnh hưởng bởi việc đóng. Kết quả `cap` của Hron phụ thuộc vào lựa chọn đóng này. |
| tsagris | Output `predict` trên D+1: ô quan sát là c/N, ô thiếu (gồm cả Other) = (1 − Σ quan sát) · trung bình Fréchet chuẩn hóa. | Đã đóng về 1. Đo được tổng = 1. | Σ_U q_j + q_Other = R, nên `cap` không bao giờ bị kích hoạt (chỉ do làm tròn) và **`cap` ≡ `prop_other`**. |
| micoda_other (A18) | Composition của lõi MICoDa trên D+1 (`proposals_`), trong không gian model: ô quan sát thay 0 bằng δ, dòng thiếu có thể bị co để chừa headroom; ô thiếu = phần dư × softmax(ALR). | Đã đóng về 1. NaN ở các dòng đầy đủ toàn 0 bị loại khỏi lõi; các dòng này không có ô thiếu. | Phần quan sát chỉ lệch V/N rất ít (δ, headroom), nên `cap` ≈ `prop_other` (lệch khoảng 1e-6 MAE). |
| micoda (A17) | Composition của lõi trên D phần, không có Other. | Đóng trên D. | **Không áp dụng** `prop_other` và `cap`: model không dự đoán phần Other. R1 hiện tại chính là `rescale`, với một ngoại lệ là dòng chỉ thiếu một ô thì gán thẳng R (cùng kết quả). |
| same_day_mean | Đề xuất c/N của từng ô thiếu, không có Other. | Không đóng. | **Không áp dụng** `prop_other`. `cap` = `sdm_full`, `rescale` = `sdm_rescale` (tham số của chính model, step 3); không đi qua hàm chung. |

Composition mà mỗi biến thể mới đã chia, đo trên COVID:

<!-- auto:p_whole_feature:start -->
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
<!-- auto:p_whole_feature:end -->

Cột cuối đếm số dòng có Σ_U q_j > R (với `cap`) hoặc số dòng được chuẩn hóa (với `rescale`), trên tổng số dòng có ô thiếu qua 5 seed. **Số này gồm cả các dòng có R = 0** (ô bị ép về 0, mọi đề xuất dương đều "vượt" R), nên không đọc nó như tần suất `cap` thực sự cắt khối lượng.

## 3. Kiểm tra

1. **Test:** 290 pass với `-W error::RuntimeWarning`: nextgen 269 (227 cũ + 42 mới), `tv2_step1` 8, `tv2_step2` 6, `tv2_step3` 4, `tv2_step4` 3. Các test mới gồm:
   - Ba quy tắc trên toy tính tay: Σ_U q < R, = R, > R, = 0, R = 0, hai ô thiếu, p chưa đóng.
   - Mọi biến thể mới giữ ô quan sát, không âm, Σ ≤ N, qua `check_completed`.
   - **Không có khóa thì giống từng byte bản cũ:** hash tính bằng code ở commit 6948635 cho cả 8 model × 3 mask trên fixture.
   - Quy tắc không áp dụng được (same_day_mean với `prop_other`, A17 với mọi quy tắc, mean_fill, tên quy tắc sai) báo lỗi trước mọi output.
   - Đặt ô hidden = 1e100 không đổi output, cho 6 biến thể mới × 3 mask.
   - Other ngầm: toy tính tay, dòng không đầy đủ bị loại, nhóm rỗng trả NA, và khớp với cách tính trực tiếp từ bảng đã điền.
   - Dòng đầy đủ toàn 0 với MICoDa A18, với cả ba quy tắc.
2. **Consistency check** (`prop_other` qua hàm chung so với output hiện tại):

<!-- auto:consistency_whole_feature:start -->
- hron_prop_check: 28309 / 28309 ô che giống hệt, chênh lớn nhất 0 count; CSV native giống từng byte (chênh lớn nhất 0)
- tsagris_prop_check: 28309 / 28309 ô che giống hệt, chênh lớn nhất 0 count; CSV native giống từng byte (chênh lớn nhất 0)
- micoda_other_prop_check: 22923 / 28309 ô che giống hệt, chênh lớn nhất 7.28e-12 count; CSV native khác (chênh lớn nhất 2.84e-14)
- tsagris_cap_vs_prop: 24638 / 28309 ô che giống hệt, chênh lớn nhất 2.91e-11 count
<!-- auto:consistency_whole_feature:end -->

   Hron và Tsagris khớp từng byte, vì nhánh `prop_other` gọi đúng `restore_counts` như trước. MICoDa A18 chỉ khớp đến sai số làm tròn (lớn nhất 2.9e-11 count), vì nhánh R2 của adapter chia bằng `_allocate_budget` và có bước sửa `nextafter`, còn hàm chung dùng `restore_counts`. Kết quả random_cell giống như vậy (xem [residual_results.md](residual_results.md)). Cặp Tsagris `cap` vs `prop_other` cũng chỉ lệch ở mức làm tròn, đúng như phân tích ở mục 2. same_day_mean `cap` không đi qua đường mới: nó là `sdm_full`, đã kiểm giống từng byte với step 2 trong step 3.
3. **Lượt check đầu tiên của MICoDa A18 fail.** Composition để NaN ở các dòng bị loại, và nhánh `prop_other` khi đó đưa cả ma trận vào `restore_counts`. Đã sửa ở 6a558fa, thêm test, rồi chạy lại riêng lượt check. Lượt fail được giữ lại trong `runs/failed_first_attempt/` (không commit). Các lượt `residual_*` (cap/rescale) chạy ở fefd4d0; mã nguồn giữa hai commit chỉ khác đúng nhánh `prop_other` (`git diff` 4 dòng), nên không chạy lại.
4. **Hash mask:** cả 5 thư mục mỗi mask (step 2 main, step 2 micoda_other, step 3 ablation, step 4 residual, step 4 check) có cùng data, schema, rate, group map và mask seed 0–4 với step 2 main. Script kiểm điều này trước khi ghép (theo kiểu F2). Regression check của mọi thư mục cho sai khác lớn nhất 0. Chi tiết trong [residual_manifest.json](residual_manifest.json).

## 4. Bảng

### Bảng 1: xếp hạng trong từng quy tắc (whole_feature)

`prop_other`:

<!-- auto:rule_prop_other_whole_feature:start -->
Tập seed chung (mọi model completed, coverage = 1): 0, 1, 2, 3, 4.

Sắp theo MAE tỷ lệ (nhóm ô model) trung bình trên tập seed chung. mean ± sd (ddof=1) qua seed; sd không phải khoảng tin cậy. **exploratory**.

| Model | MAE tỷ lệ | MSE tỷ lệ | MAE count | MAE tỷ lệ series-macro | MAE tỷ lệ, đáp án = 0 | MAE tỷ lệ, đáp án > 0 | MAE tỷ lệ, ô model đáp án = 0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| hron_prop | 0.0875 ± 0.0105 | 0.0660 ± 0.0092 | 74.6 ± 113.7 | 0.1003 ± 0.0135 | 0.0103 ± 0.0027 | 0.1978 ± 0.0195 | 0.0294 ± 0.0065 |
| micoda_other_prop | 0.1075 ± 0.0093 | 0.0684 ± 0.0078 | 188.7 ± 205.5 | 0.1175 ± 0.0137 | 0.0194 ± 0.0036 | 0.2056 ± 0.0195 | 0.0554 ± 0.0095 |
| tsagris_prop | 0.1079 ± 0.0209 | 0.0521 ± 0.0095 | 33.2 ± 14.7 | 0.1207 ± 0.0207 | 0.0260 ± 0.0075 | 0.1719 ± 0.0458 | 0.0745 ± 0.0214 |
<!-- auto:rule_prop_other_whole_feature:end -->

`cap`:

<!-- auto:rule_cap_whole_feature:start -->
Tập seed chung (mọi model completed, coverage = 1): 0, 1, 2, 3, 4.

Sắp theo MAE tỷ lệ (nhóm ô model) trung bình trên tập seed chung. mean ± sd (ddof=1) qua seed; sd không phải khoảng tin cậy. **exploratory**.

| Model | MAE tỷ lệ | MSE tỷ lệ | MAE count | MAE tỷ lệ series-macro | MAE tỷ lệ, đáp án = 0 | MAE tỷ lệ, đáp án > 0 | MAE tỷ lệ, ô model đáp án = 0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| sdm_cap | 0.0351 ± 0.0066 | 0.0142 ± 0.0036 | 29.4 ± 26.8 | 0.0495 ± 0.0095 | 0.0033 ± 0.0014 | 0.0844 ± 0.0181 | 0.0094 ± 0.0040 |
| hron_cap | 0.0869 ± 0.0095 | 0.0658 ± 0.0087 | 81.5 ± 115.2 | 0.0993 ± 0.0128 | 0.0091 ± 0.0028 | 0.2029 ± 0.0175 | 0.0258 ± 0.0070 |
| micoda_other_cap | 0.1075 ± 0.0093 | 0.0684 ± 0.0078 | 188.7 ± 205.5 | 0.1175 ± 0.0137 | 0.0194 ± 0.0036 | 0.2056 ± 0.0195 | 0.0554 ± 0.0095 |
| tsagris_cap | 0.1079 ± 0.0209 | 0.0521 ± 0.0095 | 33.2 ± 14.7 | 0.1207 ± 0.0207 | 0.0260 ± 0.0075 | 0.1719 ± 0.0458 | 0.0745 ± 0.0214 |
| mean_fill_cap | 0.1354 ± 0.0145 | 0.0704 ± 0.0084 | 218.2 ± 186.8 | 0.1324 ± 0.0162 | 0.0178 ± 0.0037 | 0.2950 ± 0.0154 | 0.0508 ± 0.0092 |
<!-- auto:rule_cap_whole_feature:end -->

`rescale`:

<!-- auto:rule_rescale_whole_feature:start -->
Tập seed chung (mọi model completed, coverage = 1): 0, 1, 2, 3, 4.

Sắp theo MAE tỷ lệ (nhóm ô model) trung bình trên tập seed chung. mean ± sd (ddof=1) qua seed; sd không phải khoảng tin cậy. **exploratory**.

| Model | MAE tỷ lệ | MSE tỷ lệ | MAE count | MAE tỷ lệ series-macro | MAE tỷ lệ, đáp án = 0 | MAE tỷ lệ, đáp án > 0 | MAE tỷ lệ, ô model đáp án = 0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| sdm_rescale | 0.0877 ± 0.0169 | 0.0497 ± 0.0119 | 83.8 ± 56.3 | 0.1047 ± 0.0183 | 0.0307 ± 0.0043 | 0.0874 ± 0.0253 | 0.0883 ± 0.0149 |
| micoda_rescale | 0.1406 ± 0.0099 | 0.0823 ± 0.0094 | 236.2 ± 188.5 | 0.1476 ± 0.0161 | 0.0411 ± 0.0051 | 0.1834 ± 0.0261 | 0.1177 ± 0.0116 |
| hron_rescale | 0.1428 ± 0.0100 | 0.0842 ± 0.0099 | 157.0 ± 154.3 | 0.1548 ± 0.0133 | 0.0451 ± 0.0058 | 0.1694 ± 0.0107 | 0.1290 ± 0.0134 |
| tsagris_rescale | 0.1615 ± 0.0124 | 0.1045 ± 0.0130 | 86.0 ± 49.2 | 0.1792 ± 0.0172 | 0.0626 ± 0.0097 | 0.1304 ± 0.0118 | 0.1788 ± 0.0231 |
| micoda_other_rescale | 0.1637 ± 0.0121 | 0.0845 ± 0.0110 | 165.4 ± 103.7 | 0.1727 ± 0.0143 | 0.0629 ± 0.0087 | 0.1330 ± 0.0191 | 0.1799 ± 0.0203 |
<!-- auto:rule_rescale_whole_feature:end -->

random_cell, `cap` và `rescale`:

<!-- auto:rule_cap_random_cell:start -->
Tập seed chung (mọi model completed, coverage = 1): 0, 1, 2, 3, 4.

Sắp theo MAE tỷ lệ (nhóm ô model) trung bình trên tập seed chung. mean ± sd (ddof=1) qua seed; sd không phải khoảng tin cậy. **exploratory**.

| Model | MAE tỷ lệ | MSE tỷ lệ | MAE count | MAE tỷ lệ series-macro | MAE tỷ lệ, đáp án = 0 | MAE tỷ lệ, đáp án > 0 | MAE tỷ lệ, ô model đáp án = 0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| sdm_cap | 0.0310 ± 0.0027 | 0.0110 ± 0.0015 | 19.0 ± 4.8 | 0.0376 ± 0.0041 | 0.0036 ± 0.0002 | 0.0704 ± 0.0077 | 0.0093 ± 0.0005 |
| hron_cap | 0.0879 ± 0.0025 | 0.0604 ± 0.0022 | 32.9 ± 15.0 | 0.0891 ± 0.0064 | 0.0113 ± 0.0008 | 0.1955 ± 0.0057 | 0.0288 ± 0.0024 |
| tsagris_cap | 0.0999 ± 0.0024 | 0.0482 ± 0.0017 | 26.6 ± 3.6 | 0.1059 ± 0.0047 | 0.0305 ± 0.0025 | 0.1395 ± 0.0149 | 0.0781 ± 0.0071 |
| micoda_other_cap | 0.1078 ± 0.0008 | 0.0695 ± 0.0013 | 109.6 ± 37.3 | 0.1075 ± 0.0049 | 0.0256 ± 0.0006 | 0.1851 ± 0.0025 | 0.0653 ± 0.0017 |
| mean_fill_cap | 0.1371 ± 0.0016 | 0.0692 ± 0.0007 | 152.4 ± 33.5 | 0.1277 ± 0.0041 | 0.0226 ± 0.0004 | 0.2817 ± 0.0045 | 0.0577 ± 0.0012 |
<!-- auto:rule_cap_random_cell:end -->

<!-- auto:rule_rescale_random_cell:start -->
Tập seed chung (mọi model completed, coverage = 1): 0, 1, 2, 3, 4.

Sắp theo MAE tỷ lệ (nhóm ô model) trung bình trên tập seed chung. mean ± sd (ddof=1) qua seed; sd không phải khoảng tin cậy. **exploratory**.

| Model | MAE tỷ lệ | MSE tỷ lệ | MAE count | MAE tỷ lệ series-macro | MAE tỷ lệ, đáp án = 0 | MAE tỷ lệ, đáp án > 0 | MAE tỷ lệ, ô model đáp án = 0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| sdm_rescale | 0.0773 ± 0.0019 | 0.0453 ± 0.0020 | 76.3 ± 12.5 | 0.0881 ± 0.0028 | 0.0295 ± 0.0013 | 0.0805 ± 0.0085 | 0.0755 ± 0.0045 |
| hron_rescale | 0.1305 ± 0.0036 | 0.0800 ± 0.0038 | 111.7 ± 11.2 | 0.1352 ± 0.0031 | 0.0482 ± 0.0014 | 0.1437 ± 0.0083 | 0.1232 ± 0.0052 |
| micoda_rescale | 0.1317 ± 0.0018 | 0.0803 ± 0.0024 | 167.7 ± 38.9 | 0.1312 ± 0.0031 | 0.0446 ± 0.0011 | 0.1641 ± 0.0041 | 0.1139 ± 0.0042 |
| tsagris_rescale | 0.1510 ± 0.0040 | 0.0975 ± 0.0039 | 89.3 ± 14.9 | 0.1618 ± 0.0068 | 0.0671 ± 0.0017 | 0.1137 ± 0.0109 | 0.1715 ± 0.0058 |
| micoda_other_rescale | 0.1567 ± 0.0020 | 0.0859 ± 0.0017 | 141.3 ± 18.5 | 0.1580 ± 0.0022 | 0.0693 ± 0.0016 | 0.1191 ± 0.0067 | 0.1772 ± 0.0056 |
<!-- auto:rule_rescale_random_cell:end -->

### Bảng 2: ghép cặp trong cùng quy tắc (whole_feature)

Ô: Δ = A − B (mean ± sd), (A thắng / 5 seed). Thắng = sai số nhỏ hơn.

<!-- auto:within_rule_whole_feature:start -->
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
<!-- auto:within_rule_whole_feature:end -->

### Bảng 3: ghép cặp trong cùng model

**whole_feature**

<!-- auto:within_model_whole_feature:start -->
| A vs B | MAE tỷ lệ | MSE tỷ lệ | MAE count | MAE tỷ lệ series-macro | MAE tỷ lệ, đáp án = 0 | MAE tỷ lệ, đáp án > 0 | MAE tỷ lệ, ô model đáp án = 0 |
|---|---|---|---|---|---|---|---|
| hron_cap vs hron_prop | -0.0006277 ± 0.0019 (4/5) | -0.0002593 ± 0.0014 (3/5) | +6.943 ± 8.3 (1/5) | -0.0009751 ± 0.0022 (4/5) | -0.001232 ± 0.00016 (5/5) | +0.005113 ± 0.0057 (1/5) | -0.003542 ± 0.00056 (5/5) |
| hron_rescale vs hron_prop | +0.05527 ± 0.013 (0/5) | +0.01812 ± 0.0056 (0/5) | +82.39 ± 60 (0/5) | +0.0545 ± 0.0083 (0/5) | +0.03478 ± 0.0036 (0/5) | -0.02837 ± 0.029 (5/5) | +0.09961 ± 0.0093 (0/5) |
| tsagris_cap vs tsagris_prop | +0 ± 9.8e-18 (1/5, hòa 3) | -2.776e-18 ± 6.2e-18 (1/5, hòa 4) | +4.263e-15 ± 1.1e-14 (1/5, hòa 2) | +0 ± 0 (0/5, hòa 5) | +2.082e-18 ± 3.1e-18 (0/5, hòa 3) | -1.11e-17 ± 1.5e-17 (2/5, hòa 3) | +0 ± 0 (0/5, hòa 5) |
| tsagris_rescale vs tsagris_prop | +0.05361 ± 0.013 (0/5) | +0.0524 ± 0.0077 (0/5) | +52.8 ± 37 (0/5) | +0.05857 ± 0.013 (0/5) | +0.03655 ± 0.0071 (0/5) | -0.04154 ± 0.052 (5/5) | +0.1043 ± 0.017 (0/5) |
| micoda_other_cap vs micoda_other_prop | +9.429e-07 ± 4.2e-07 (0/5) | -4.005e-07 ± 2.4e-07 (5/5) | +0.002264 ± 0.0028 (1/5) | +3.971e-07 ± 2.5e-07 (0/5) | -5.181e-07 ± 2e-07 (5/5) | +5.585e-06 ± 2.3e-06 (0/5) | -1.491e-06 ± 5.9e-07 (5/5) |
| micoda_other_rescale vs micoda_other_prop | +0.05616 ± 0.0071 (0/5) | +0.01607 ± 0.0074 (0/5) | -23.33 ± 1.1e+02 (2/5) | +0.05529 ± 0.0076 (0/5) | +0.04353 ± 0.0058 (0/5) | -0.07261 ± 0.015 (5/5) | +0.1245 ± 0.013 (0/5) |
<!-- auto:within_model_whole_feature:end -->

**random_cell**

<!-- auto:within_model_random_cell:start -->
| A vs B | MAE tỷ lệ | MSE tỷ lệ | MAE count | MAE tỷ lệ series-macro | MAE tỷ lệ, đáp án = 0 | MAE tỷ lệ, đáp án > 0 | MAE tỷ lệ, ô model đáp án = 0 |
|---|---|---|---|---|---|---|---|
| hron_cap vs hron_prop | -0.001287 ± 0.00087 (5/5) | -0.0008421 ± 0.0006 (5/5) | +9.944 ± 13 (0/5) | -0.001724 ± 0.0011 (5/5) | -0.001041 ± 6.2e-05 (5/5) | +0.001207 ± 0.0023 (2/5) | -0.002663 ± 0.00019 (5/5) |
| hron_rescale vs hron_prop | +0.04131 ± 0.0033 (0/5) | +0.01876 ± 0.0031 (0/5) | +88.74 ± 8.8 (0/5) | +0.0443 ± 0.0044 (0/5) | +0.03589 ± 0.0012 (0/5) | -0.05058 ± 0.0042 (5/5) | +0.09179 ± 0.0039 (0/5) |
| tsagris_cap vs tsagris_prop | -5.551e-18 ± 1.2e-17 (1/5, hòa 4) | +0 ± 0 (0/5, hòa 5) | +7.105e-16 ± 7.3e-15 (1/5, hòa 2) | -2.776e-18 ± 6.2e-18 (1/5, hòa 4) | +0 ± 0 (0/5, hòa 5) | -5.551e-18 ± 1.2e-17 (1/5, hòa 4) | +5.551e-18 ± 7.6e-18 (0/5, hòa 3) |
| tsagris_rescale vs tsagris_prop | +0.05109 ± 0.0019 (0/5) | +0.04927 ± 0.0024 (0/5) | +62.72 ± 13 (0/5) | +0.05589 ± 0.0038 (0/5) | +0.03652 ± 0.0026 (0/5) | -0.02576 ± 0.012 (5/5) | +0.09335 ± 0.0063 (0/5) |
| micoda_other_cap vs micoda_other_prop | +9.795e-07 ± 5.8e-08 (0/5) | -6.29e-07 ± 6.9e-08 (5/5) | +0.002055 ± 0.0012 (0/5) | +5.445e-07 ± 2.2e-08 (0/5) | -6.524e-07 ± 1.7e-08 (5/5) | +5.799e-06 ± 1.3e-07 (0/5) | -1.668e-06 ± 4.2e-08 (5/5) |
| micoda_other_rescale vs micoda_other_prop | +0.04885 ± 0.0016 (0/5) | +0.01642 ± 0.0012 (0/5) | +31.73 ± 22 (1/5) | +0.0505 ± 0.0045 (0/5) | +0.04376 ± 0.0012 (0/5) | -0.06598 ± 0.0047 (5/5) | +0.1119 ± 0.0041 (0/5) |
<!-- auto:within_model_random_cell:end -->

### Bảng 4: tách khoảng cách (MAE tỷ lệ; mô tả, không phải phân rã chặt)

Các thành phần được tính theo từng seed rồi lấy trung bình, nên trong mỗi đường tách chúng cộng lại đúng bằng khoảng cách. Hai đường tách cho hai câu chuyện khác nhau, vì có tương tác giữa thuật toán và quy tắc.

**whole_feature**

<!-- auto:gap_whole_feature:start -->
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
<!-- auto:gap_whole_feature:end -->

**random_cell**

<!-- auto:gap_random_cell:start -->
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
<!-- auto:gap_random_cell:end -->

### Bảng 5: theo nhóm đáp án (whole_feature)

<!-- auto:truth_whole_feature:start -->
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
<!-- auto:truth_whole_feature:end -->

### Bảng 6: Other ngầm

Chỉ tính trên các dòng đầy đủ trước khi che và có ít nhất một ô bị che; ở đó Other thật đã biết. **Chỉ có 5 nước**, và mỗi seed whole_feature chỉ dùng được 2–4 nước, nên **không suy rộng ra 150 nước**.

**whole_feature**

<!-- auto:other_whole_feature:start -->
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
<!-- auto:other_whole_feature:end -->

**random_cell**

<!-- auto:other_random_cell:start -->
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
<!-- auto:other_random_cell:end -->

Với `rescale`, Other ngầm bằng 0 trên mọi dòng có đề xuất dương. Vì thế sai số có dấu bằng −(Other thật), và giống hệt nhau giữa hron, micoda_other và A17 (−0.1349 ở whole_feature, −0.1504 ở random_cell). Suy ra Other thật trung bình trên các dòng này khoảng 0.13–0.15. `sdm_rescale` có sai số nhỏ hơn vì những dòng có đề xuất bằng 0 vẫn giữ Other = R.

### Bảng 7: series-macro so với cell-weighted

**whole_feature**

<!-- auto:macro_whole_feature:start -->
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
<!-- auto:macro_whole_feature:end -->

**random_cell**

<!-- auto:macro_random_cell:start -->
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
<!-- auto:macro_random_cell:end -->

Trong `prop_other` và `cap`, thứ hạng không đổi ở cả hai mask. Các lần đổi chỗ chỉ xảy ra giữa những biến thể `rescale` sát nhau, và giữa `rescale` với `mean_fill` (|Δ| ≤ 0.0075).

## 5. Kết luận có giới hạn

Exploratory: masked pilot, seed 0–4 đã được nhìn ở các step trước, dựng lại bảng cố định. sd là độ lệch chuẩn mẫu qua 5 seed, **không phải khoảng tin cậy**. Other ngầm chỉ dựa trên 5 nước.

1. **Thứ hạng khi mọi model dùng cùng một quy tắc:**
   - same_day_mean đứng đầu ở cả hai quy tắc chạy được (`cap`, `rescale`), cả hai mask. Nó thắng Hron và Tsagris 5/5 seed ở mỗi quy tắc (MAE count với Tsagris là 4/5 ở `cap` và 2/5 ở `rescale` trên whole_feature).
   - Hron đứng thứ hai ở `cap` và thắng Tsagris 5/5 về MAE. Ở `rescale` trên whole_feature, Hron chỉ xếp sau A17 theo trung bình (0.1428 so với 0.1406; cặp này chưa ghép).
   - Thứ tự Hron > Tsagris về MAE tỷ lệ giữ nguyên ở mọi quy tắc. Về MSE tỷ lệ thì Tsagris vẫn hơn Hron ở `cap`, như ở step 1.
2. **Khoảng cách same_day_mean − Hron gần như hoàn toàn do thuật toán.**
   - Theo đường `cap`: thuật toán chiếm 99% (whole_feature) và 98% (random_cell); phần quy tắc của Hron (cap thay cho prop_other) chỉ 1–2%. Với Tsagris, thuật toán chiếm 100%, vì `cap` ≡ `prop_other`.
   - Theo đường `rescale`: hai model cùng mất khoảng như nhau khi chuyển sang `rescale`, nên khoảng cách thuật toán còn nguyên. `sdm_rescale` thắng `hron_rescale` 5/5, Δ −0.055 và −0.053.
   - Câu "same_day_mean dùng `rescale` chỉ còn ngang Hron" ở step 3 là so hai quy tắc khác nhau. Khi cùng quy tắc, khoảng cách không thu hẹp.
3. **Quy tắc nào hợp với dữ liệu này hơn:**
   - **`rescale` kém nhất với mọi model.** So với `prop_other`, MAE tăng 0.041–0.056, thua 0/5 seed với cả Hron, Tsagris và A18 ở cả hai mask.
   - Thiệt hại nằm ở các ô model có đáp án 0 (+0.09 đến +0.12). Ô có đáp án > 0 lại tốt hơn 5/5, vì `rescale` dồn khối lượng vào ô thiếu.
   - Other ngầm sai khoảng 0.13–0.15 với `rescale`, so với 0.006–0.038 với `prop_other`/`cap`. Dữ liệu này có phần Other đáng kể, và một quy tắc ép Other = 0 thì sai có hệ thống.
   - Giữa `cap` và `prop_other`: trùng nhau với Tsagris; lệch khoảng 1e-6 với A18; với Hron, `cap` tốt hơn rất ít về MAE tỷ lệ (Δ −0.0006, thắng 4/5; random_cell −0.0013, 5/5) nhưng kém hơn về MAE count (1/5, 0/5).
   - Hệ quả cho MICoDa: R1 của A17 chính là `rescale`, nên A17 gánh đúng khoản phạt này. Khi cùng `rescale`, lõi A17 còn có MAE trung bình thấp hơn lõi A18 (0.1406 so với 0.1637; random_cell 0.1317 so với 0.1567; cặp này chưa ghép). Vậy lợi thế của A18 ở step 1–2 đến từ quy tắc (R2), không phải từ lõi. Đây là sensitivity, không phải quyết định đổi D08.

## 6. Quyết định thiết kế chưa chắc

1. **Hron không cho composition đóng.** Tổng dòng trải từ 1e-6 đến 3.5e5, và tôi đóng dòng về 1 trên D+1 trước `cap`. Nếu dùng thẳng giá trị điền (thang c/N của dòng) thì kết quả `cap` của Hron sẽ khác. `prop_other` và `rescale` không bị ảnh hưởng.
2. **Khóa `residual_rule` có hai cách xử lý:** ở runner (Hron/Tsagris/A18), và là tham số của chính model (same_day_mean, step 3). Cùng tên, cùng định nghĩa `cap`/`rescale`; `prop_other` chỉ có ở runner.
3. **MICoDa A18 với quy tắc mới:** adapter vẫn chạy nhánh R2 của nó rồi bỏ kết quả; runner chia lại từ `proposals_`. Trạng thái R2 gốc được ghi trong `info.adapter_status`. Nếu nhánh R2 trả partial nhưng đường mới chia được thì run vẫn tính là completed. Trên COVID không xảy ra (mọi run R2 đều completed).
4. **Phải sửa adapter MICoDa** (thêm `proposals_`) vì `restore` nằm bên trong `impute_counts`. Không có phép tính nào thay đổi; mặc định giống từng byte. Consistency check cho thấy đường `restore_counts` tái lập R2 đến 3e-11 count, là bằng chứng cho việc hợp nhất S2 (audit) nếu sau này A18 được thăng hạng.
5. **Cột "dòng bị co/chuẩn hóa"** gồm cả các dòng có R = 0, nên tách riêng dòng R > 0 cần một lần chạy mới.
6. **Bảng 4** chỉ tách MAE tỷ lệ. Các metric khác của cùng các cặp nằm trong `residual_pairs.csv`, nhưng chưa có bảng tách riêng.
7. **Nhãn `space` của run fail** trong lượt check đầu là `17_targets` cho alias MICoDa có `other = true`, vì nhánh exception gọi `space(model_kind)` không có instance (hành vi cũ, chỉ ảnh hưởng nhãn). Lượt chạy lại không fail nên không gặp.

## 7. Gợi ý cho main model (chỉ dựa trên kết quả trên, không đổi code)

- **Dự đoán phần Other một cách tường minh, hoặc ít nhất không ép toàn bộ R vào target thiếu.** Trên dữ liệu này `rescale`/R1 làm MAE tăng khoảng 0.05 với mọi model đã thử, và sai Other ngầm khoảng 0.14.
- **Cho output là composition đóng trên D+1.** Khi đó `cap` và `prop_other` gần như trùng nhau (như Tsagris và A18), nên kết quả không còn phụ thuộc lựa chọn giữa hai quy tắc này.
- **So main model với baseline dưới cùng một quy tắc phần dư**, và báo kèm Other ngầm như một diagnostic. Mốc cần vượt là same_day_mean (`cap`), không phải Hron.

## 8. Việc để lại cho step sau

- Seed xác nhận mới, cố định trước, cho các kết luận ở mục 5.
- Ghép cặp A17 với A18 ở `rescale`, và A17 với Hron ở `rescale` (hiện chỉ có trung bình).
- Tách khoảng cách cho MSE tỷ lệ và MAE count; đếm riêng các dòng R > 0 bị `cap` cắt.
- Thử Hron với p chưa đóng (thang c/N) cho `cap`, để biết lựa chọn đóng ảnh hưởng bao nhiêu.
- Other ngầm trên nhiều nước hơn: cần dữ liệu có Other đã biết, ví dụ một bảng có cột Other thật.
- Hợp nhất nhánh R2 của adapter vào `restore_counts` (audit S2) nếu A18 được xét thăng hạng.
- Mang sang từ các step trước: bootstrap/CI, diagnostic Tầng 2–4, mã lỗi cụ thể hơn cho Hron/Tsagris, chính sách khi whole_feature xóa hết dòng đầy đủ.
