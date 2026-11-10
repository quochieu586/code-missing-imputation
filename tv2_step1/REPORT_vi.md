# Step 1: đối thủ giả + trung bình cùng ngày

Branch `step1-decoys-sameday` (từ `main` a5a918a). Không sửa `evaluate.py`; Hron/Tsagris/MICoDa không đổi hành vi.

Phần đầu (đến mục *Quyết định thiết kế*) là lượt kiểm tra một seed khi thêm code. Phần **Hoàn thiện step 1** ở cuối có kết quả 5 seed, MICoDa `other = true` và kết luận.

## File đã sửa/thêm

| File | Nội dung |
|---|---|
| `nextgen/src/longitudinal_impute/data.py` | `Dataset` thêm `days`, `week`, `time_kind`, `groups`. Ngày ISO (hoặc `time_format`) thành số ngày kể từ ngày nhỏ nhất, `week = days // 7`. Time dạng số (mốc 1, 2, 3) giữ nguyên giá trị, chữ khác thành nhãn. Ngày sai, hoặc trộn ngày ISO với giá trị khác, báo lỗi kèm dòng CSV. Thêm khóa schema `group_col`, `group_map`. |
| `baselines/common.py` | `fill_within_residual`: nếu tổng đề xuất của dòng vượt phần dư R thì co lại cho bằng R, phần còn lại vào Other. `observed_columns`: kiểm tra input. |
| `baselines/zero_fill.py`, `mean_fill.py`, `marginal_sampler.py` | Ba đối thủ giả (docstring ghi rõ đây là mức sàn). |
| `baselines/same_day_mean.py` | Trung bình cùng ngày, có nới cửa sổ và ghi nhận mức (`info_['cells_by_level']`, `levels_`). |
| `run.py` | Đăng ký 4 model. Model có `needs_context` nhận N, days, time_kind, groups, entities (không bao giờ nhận counts). Thêm khóa `enabled = false` và cờ `--only`. |
| `configs/models_default.toml` | Thêm 4 bảng. Sau phần hoàn thiện: `same_day_mean` chạy mặc định; ba mức sàn `enabled = false`. |
| `configs/models_micoda_other.toml` | MICoDa `other = true` (A18/R2) cho lượt riêng `micoda_other`. |
| `configs/baselines_covariants.toml` | Thêm `time_format`, `group_col = "continent"`, `group_map`. |
| `configs/covid_location_continent.csv` | Map 150 location → 6 châu lục. |
| `nextgen/README.md`, `docs/THREE_BASELINE_CONTRACT_VI.md` | Khóa schema mới, `enabled`, `--only`; mục same_day_mean và ba mức sàn. |
| `tv2_step1/summarize_runs.py`, `results_multiseed.md/.csv` | Script và bảng tổng hợp 5 seed (output đầy đủ trong `tv2_step1/runs/`, không commit). |
| `tests/test_time_and_groups.py`, `tests/test_floor_and_same_day.py` | 41 test mới. |

## Kiểm tra

1. **Test:** 156 pass (115 cũ + 41 mới), kể cả khi coi RuntimeWarning là lỗi. Test mới gồm: parse ngày, lỗi định dạng, time dạng số/nhãn/không có, group từ cột hoặc từ map, Dataset không có group_col. Với 4 model: giữ nguyên ô quan sát, không âm, tổng ≤ N, cùng seed cùng kết quả, đặt ô bị che = 1e100 không đổi output (cả whole_feature và random_cell). Ngoài ra có tham chiếu same_day_mean tính tay cho từng mức (cửa sổ, nới, toàn cục, loại cùng location) và cách chọn model `--only`/`enabled`.
2. **COVID đầy đủ**, mask `whole_feature`, seed 0, tỷ lệ 5% (5625 ô bị che, trong đó 2595 ô `model` và 3030 ô ép về 0). Cả 7 model chạy xong, `check_completed` pass, không có `errors.log`.

| Model | MAE tỷ lệ (model) | MSE tỷ lệ (model) | MAE count (model) | MAE tỷ lệ (all) |
|---|---:|---:|---:|---:|
| **same_day_mean** | **0.0333** | **0.0136** | 73.3 | **0.0154** |
| hron | 0.0903 | 0.0697 | 277.7 | 0.0417 |
| tsagris | 0.1291 | 0.0613 | **56.3** | 0.0596 |
| mean_fill (sàn) | 0.1414 | 0.0776 | 523.5 | 0.0652 |
| micoda (A17/R1) | 0.1454 | 0.0880 | 551.6 | 0.0671 |
| marginal_sampler (sàn) | 0.1456 | 0.1199 | 558.9 | 0.0672 |
| zero_fill (sàn) | 0.1798 | 0.1630 | 589.4 | 0.0830 |

   - same_day_mean, Hron và Tsagris tốt hơn cả ba mức sàn trên MAE/MSE tỷ lệ. Riêng Tsagris có MAE count thấp hơn same_day_mean.
   - ⚠️ **MICoDa thua mean_fill** trên cả MAE tỷ lệ (0.1454 so với 0.1414), MSE tỷ lệ (0.0880 so với 0.0776) và MAE count. MAE tỷ lệ của MICoDa cũng ngang marginal_sampler. Tôi không chỉnh gì để số đẹp hơn. Đây là MICoDa mặc định A17/R1, vốn đã biết có thiên lệch điền cao (README, D08).
   - same_day_mean tốt hơn Hron gần 3 lần trên MAE tỷ lệ. Nghĩa là thông tin "cùng thời điểm, cùng châu lục" rất mạnh, mà Hron/Tsagris/MICoDa đều không dùng thời gian.
   - Kết quả chỉ có **một seed**, chưa có độ lệch giữa các seed.
3. **Hron/Tsagris/MICoDa không đổi:** so với lượt chạy trên `main` cùng cấu hình, các file `*_completed.csv` và mask giống từng byte, `summary.json` (native và masked, trừ `seconds`) bằng nhau.
4. **Số ô same_day_mean rơi vào từng mức** (`window_days=28`, `min_obs=20`, `max_widen=2`; các cửa sổ ±28/±56/±112 ngày; 6 nhóm):

| Tập ô | Cửa sổ gốc | Nới 1 lần | Nới 2 lần | Toàn cục |
|---|---:|---:|---:|---:|
| Ô bị che, nhóm model (2595) | 2552 | 6 | 0 | 37 |
| Mọi ô bị che (5625) | 5450 | 41 | 23 | 111 |
| Mọi ô thiếu, lượt masked (91675) | 56507 | 4960 | 5637 | 24571 |
| Mọi ô thiếu, lượt native | 51136 | 4912 | 8056 | 21946 |

   Ô thiếu thật rơi về mức toàn cục nhiều (25.5% ở lượt native; 26.8% trên mọi ô thiếu của lượt masked), có thể do các ngày đầu/cuối chuỗi và nhóm nhỏ (Oceania 6 location). Với seed 0, ô bị che gần như luôn nằm trong cửa sổ gốc (qua 5 seed: xem bên dưới). `summary.json` hiện ghi `cells_by_level` trên mọi ô thiếu. Bảng theo riêng ô bị che được tính từ `levels_`.

## Quyết định thiết kế chưa chắc

1. **Quy tắc chia phần dư:** đề xuất của các ô thiếu nhân N. Nếu tổng vượt R thì co tỷ lệ cho bằng R, Other = 0. Nếu không vượt thì Other nhận phần còn lại. Hron/Tsagris thì khác: Other cũng được dự đoán như một thành phần rồi mới chia R theo tỷ lệ. Hai cách cho kết quả khác nhau khi tổng đề xuất nhỏ hơn R.
2. **mean_fill** dùng trung bình *không trọng số* của c/N. Mức toàn cục của same_day_mean dùng trung bình *có trọng số N*. Hai cách này không hoàn toàn cùng định nghĩa "trung bình toàn cục".
3. **Mức toàn cục** của same_day_mean lấy trên cả bảng (mọi nhóm), không chỉ trong nhóm. Khi `exclude_same_location=true` thì mức toàn cục cũng loại location đó. Nếu không còn ô nào thì báo `InsufficientDataError`, không có fallback ngầm. Điều này xảy ra được trên dữ liệu nhỏ (có test).
4. `min_obs` đếm số ô quan sát của biến thể j trong cửa sổ, không đếm số location khác nhau.
5. **`exclude_same_location` mặc định `false`.** Với whole_feature, chuỗi (location, j) đã bị che toàn bộ nên cờ này không đổi kết quả trên ô bị che. Nó chỉ có tác dụng với random_cell và ô thiếu thật.
6. **Time không phải ngày:** dạng số thì xét cùng giá trị trước, rồi nới ±1, ±2, ±4 (bỏ qua `window_days`). Dạng nhãn chỉ xét cùng nhãn, không nới. Không có cột time thì lấy cả nhóm.
7. `week = days // 7` tính từ ngày nhỏ nhất, không phải tuần ISO. Hiện chưa model nào dùng.
8. **Tự nhận dạng ngày:** mọi giá trị dạng YYYY-MM-DD thì coi là ngày. Trộn ISO với giá trị khác thì báo lỗi. Schema COVID khai báo `time_format` tường minh.
9. **Châu lục** theo quy ước Our World in Data: Turkey, Georgia thuộc Asia; Russia thuộc Europe; vùng Caribbean thuộc North America. Đây là file cấu hình, không hard-code trong code.
10. **Chọn model:** dùng `enabled = false` cùng cờ `--only`, không đổi nghĩa của `--models` (vẫn là đường dẫn TOML). Yêu cầu "thêm section trong baselines_covariants.toml" được hiểu là thêm khóa vào `[dataset]`, vì file schema chỉ có bảng đó.
11. **Seed của marginal_sampler** cố định theo config, không gắn với seed của mask.
12. Lượt chạy này dùng `/opt/anaconda3` (Python 3.13, numpy 2.1.3), vì `.venv` trong repo là venv Windows.

## Hoàn thiện step 1: 5 seed, MICoDa `other = true`

COVID đầy đủ, che 5%, seed 0–4, hai kiểu che. Model trong `src/` không đổi giữa các lượt (commit 7b0ccbe và 12fd81d giống nhau ở `nextgen/src`). Tất cả 80 lượt masked và 16 lượt native đều `completed`/`ok`, không có `errors.log`; seed 0 trùng khớp lượt một-seed ở trên. Bảng đầy đủ: [results_multiseed.md](results_multiseed.md) (sinh bởi `summarize_runs.py`).

### whole_feature (chính): nhóm ô model, mean ± sd qua 5 seed

| Model | MAE tỷ lệ | MSE tỷ lệ | MAE count |
|---|---:|---:|---:|
| same_day_mean | **0.0351 ± 0.0066** | **0.0142 ± 0.0036** | **29.4 ± 26.8** |
| hron | 0.0875 ± 0.0105 | 0.0660 ± 0.0092 | 74.6 ± 113.7 |
| micoda_other (A18/R2) | 0.1075 ± 0.0093 | 0.0684 ± 0.0078 | 188.7 ± 205.5 |
| tsagris | 0.1079 ± 0.0209 | 0.0521 ± 0.0095 | 33.2 ± 14.7 |
| mean_fill (sàn) | 0.1354 ± 0.0145 | 0.0704 ± 0.0084 | 218.2 ± 186.8 |
| marginal_sampler (sàn) | 0.1354 ± 0.0120 | 0.1107 ± 0.0122 | 225.1 ± 200.7 |
| micoda (A17/R1) | 0.1406 ± 0.0099 | 0.0823 ± 0.0094 | 236.2 ± 188.5 |
| zero_fill (sàn) | 0.1693 ± 0.0219 | 0.1525 ± 0.0230 | 287.1 ± 211.3 |

Số ô model theo seed: 2595 / 2729 / 2522 / 2504 / 2406.

random_cell (tham khảo) xếp hạng gần giống: same_day_mean 0.0310, hron 0.0892, tsagris 0.0999, micoda_other 0.1078, micoda 0.1317, marginal_sampler 0.1362, mean_fill 0.1371, zero_fill 0.1716 (MAE tỷ lệ). Khác biệt: ở random_cell MICoDa A17 thắng mean_fill về MAE.

### Thắng/thua theo seed (ghép cặp, cùng mask; thắng = sai số nhỏ hơn trên nhóm model)

| Cặp | whole_feature, MAE tỷ lệ | whole_feature, MSE tỷ lệ | random_cell, MAE tỷ lệ | random_cell, MSE tỷ lệ |
|---|---|---|---|---|
| same_day_mean vs hron | thắng 5/5, −0.0525 | thắng 5/5, −0.0518 | thắng 5/5, −0.0582 | thắng 5/5, −0.0502 |
| hron vs tsagris | thắng 5/5, −0.0204 | thắng 0/5, +0.0140 | thắng 5/5, −0.0107 | thắng 0/5, +0.0130 |
| micoda_other vs mean_fill | thắng 5/5, −0.0278 | thắng 4/5, −0.0020 | thắng 5/5, −0.0293 | thắng 1/5, +0.0003 |
| micoda vs mean_fill | thắng 1/5, +0.0052 | thắng 0/5, +0.0119 | thắng 5/5, −0.0054 | thắng 0/5, +0.0111 |

Số sau dấu phẩy là hiệu trung bình A − B (âm = A tốt hơn). Chưa có bootstrap.

### MAE tỷ lệ theo đáp án (whole_feature, mean ± sd)

| Model | đáp án = 0 (`truth_zero`) | đáp án = 0, ô model (`model_truth_zero`) | đáp án > 0 (`truth_positive`) |
|---|---:|---:|---:|
| hron | 0.0103 ± 0.0027 | 0.0294 ± 0.0065 | 0.1978 ± 0.0195 |
| tsagris | 0.0260 ± 0.0075 | 0.0745 ± 0.0214 | 0.1719 ± 0.0458 |
| same_day_mean | 0.0033 ± 0.0014 | 0.0094 ± 0.0040 | 0.0844 ± 0.0181 |

Seed 0: 4633 ô đáp án 0 (1603 trong nhóm model, phần còn lại là ô ép về 0), 992 ô đáp án > 0. Ô đáp án > 0 luôn thuộc nhóm model. Sai số tập trung ở ô > 0: MAE ở ô > 0 lớn gấp ~2 lần (Tsagris), ~7 lần (Hron), ~9 lần (same_day_mean) ô = 0 thuộc nhóm model.

### MICoDa `other = true` (micoda_other)

- Chạy bằng `nextgen/configs/models_micoda_other.toml`, lượt riêng, không sửa `run.py`. Mọi lượt đều `completed`, nên đã qua `check_completed`.
- **Còn thua mean_fill không?** Về MAE tỷ lệ: không, thắng 5/5 seed ở cả hai mask (hiệu ~ −0.028). Về MSE tỷ lệ: gần như **hòa** (whole_feature thắng 4/5, hiệu −0.0020; random_cell thắng 1/5, hiệu +0.0003). Tức là chỉ nhỉnh hơn mức sàn một chút.
- **So với micoda mặc định (A17/R1):** tốt hơn 5/5 seed ở cả hai mask trên MAE tỷ lệ (whole_feature −0.033, tức ×0.77; random_cell −0.024) và MSE tỷ lệ (−0.014, ×0.83; −0.011). MAE count giảm ~30–35%.
- ⚠️ **Bất thường cần báo:** MSE count của micoda_other trên whole_feature tốt hơn ở 4/5 seed, nhưng trung bình lại *cao hơn* (+7.5e6, tỷ số trung bình ×1.22), do một seed rất xấu (đuôi nặng ở các dòng N lớn). Trên random_cell tỷ số MSE count chỉ ×1.02. Con số này **không khớp** với ghi chú D08 ("random-cell MSE count ~×4.5"). Tôi chưa điều tra nguyên nhân, có thể khác thiết lập lúc chốt D08. Không sửa gì.

### same_day_mean: tỷ lệ ô dùng trung bình toàn cục

| Tập ô | whole_feature (seed 0–4) | random_cell (seed 0–4) |
|---|---|---|
| Ô che, nhóm model | 1.4% / 4.2% / 4.6% / 5.6% / 9.0% | 0.7–1.0% |
| Ô thiếu thật (native) | 25.5% (21 946 / 86 050) | như bên trái |

Câu "~1–2%" trong bản nháp contract đã được sửa theo bảng này.

### Kết luận

1. **same_day_mean là baseline mạnh nhất**, thắng Hron 5/5 seed ở cả hai mask, cả MAE lẫn MSE tỷ lệ (MAE giảm ~60%). Nó đã được bật mặc định; model chính phải thắng nó, không chỉ thắng Hron/Tsagris.
2. **Hron và Tsagris không phân thắng bại:** Hron tốt hơn về MAE tỷ lệ (5/5), Tsagris tốt hơn về MSE tỷ lệ và MAE count (5/5). Chọn metric chính sẽ quyết định thứ hạng, đây là việc của step 2.
3. **MICoDa sát mức sàn:** bản mặc định A17/R1 thua mean_fill trên whole_feature (MAE thua 4/5 seed, MSE thua 5/5). Bản A18/R2 cải thiện rõ so với A17 nhưng chỉ hơn mean_fill về MAE; MSE gần như bằng.
4. **Điểm masked của same_day_mean lạc quan hơn thực tế:** ô thiếu thật dùng trung bình toàn cục 25.5%, còn ô bị che chỉ 0.7–9%.
5. **MAE count do vài dòng N lớn chi phối** (sd lớn hơn trung bình ở whole_feature), nên xếp hạng nên dựa trên tỷ lệ, còn count chỉ tham khảo.

## Để lại cho step 2

- **Tên biến thể** kiểu `[same_day_mean_wide]`, `[micoda_other]`: `run.py` map tên bảng thẳng vào class nên chưa hỗ trợ (step 1 phải chạy micoda_other bằng lượt riêng). Có thể thêm khóa `model = "same_day_mean"`.
- `evaluate.py`: bootstrap/khoảng tin cậy cho so sánh ghép cặp; thống kê mức nới cửa sổ chỉ trên ô bị che trong `summary`; Wasserstein cho lượt masked (marginal_sampler được thiết kế để thắng ở đó); chọn metric chính (MAE hay MSE tỷ lệ) vì Hron/Tsagris đổi thứ hạng theo metric.
- Độ nhạy của same_day_mean theo `window_days`, `min_obs`, `exclude_same_location`, và cách giảm tỷ lệ 25.5% rơi về toàn cục trên ô thiếu thật.
- Điều tra MSE count của MICoDa A18 (seed xấu ở whole_feature; không khớp D08) và vì sao A17/R1 thua mean_fill.
- Chạy thử với dữ liệu microbiome thật (time dạng số) và quyết định cách nới cửa sổ cho dạng đó.
