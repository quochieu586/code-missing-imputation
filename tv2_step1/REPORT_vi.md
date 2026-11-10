# Step 1: đối thủ giả + trung bình cùng ngày

Branch `step1-decoys-sameday` (từ `main` a5a918a). Không sửa `evaluate.py`; Hron/Tsagris/MICoDa không đổi hành vi.

## File đã sửa/thêm

| File | Nội dung |
|---|---|
| `nextgen/src/longitudinal_impute/data.py` | `Dataset` thêm `days`, `week`, `time_kind`, `groups`. Ngày ISO (hoặc `time_format`) thành số ngày kể từ ngày nhỏ nhất, `week = days // 7`. Time dạng số (mốc 1, 2, 3) giữ nguyên giá trị, chữ khác thành nhãn. Ngày sai, hoặc trộn ngày ISO với giá trị khác, báo lỗi kèm dòng CSV. Thêm khóa schema `group_col`, `group_map`. |
| `baselines/common.py` | `fill_within_residual`: nếu tổng đề xuất của dòng vượt phần dư R thì co lại cho bằng R, phần còn lại vào Other. `observed_columns`: kiểm tra input. |
| `baselines/zero_fill.py`, `mean_fill.py`, `marginal_sampler.py` | Ba đối thủ giả (docstring ghi rõ đây là mức sàn). |
| `baselines/same_day_mean.py` | Trung bình cùng ngày, có nới cửa sổ và ghi nhận mức (`info_['cells_by_level']`, `levels_`). |
| `run.py` | Đăng ký 4 model. Model có `needs_context` nhận N, days, time_kind, groups, entities (không bao giờ nhận counts). Thêm khóa `enabled = false` và cờ `--only`. |
| `configs/models_default.toml` | Thêm 4 bảng, tất cả `enabled = false`, nên lượt chạy mặc định không đổi. |
| `configs/baselines_covariants.toml` | Thêm `time_format`, `group_col = "continent"`, `group_map`. |
| `configs/covid_location_continent.csv` | Map 150 location → 6 châu lục. |
| `nextgen/README.md` | Mô tả khóa schema mới, `enabled`, `--only`. |
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

   Ô thiếu thật rơi về mức toàn cục nhiều (~27%), có thể do các ngày đầu/cuối chuỗi và nhóm nhỏ (Oceania 6 location). Ô bị che thì gần như luôn nằm trong cửa sổ gốc. `summary.json` hiện ghi `cells_by_level` trên mọi ô thiếu. Bảng theo riêng ô bị che được tính từ `levels_`.

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
10. **Giữ mặc định không đổi:** dùng `enabled = false` cùng cờ `--only`, không đổi nghĩa của `--models` (vẫn là đường dẫn TOML). Yêu cầu "thêm section trong baselines_covariants.toml" được hiểu là thêm khóa vào `[dataset]`, vì file schema chỉ có bảng đó.
11. **Seed của marginal_sampler** cố định theo config, không gắn với seed của mask.
12. Lượt chạy này dùng `/opt/anaconda3` (Python 3.13, numpy 2.1.3), vì `.venv` trong repo là venv Windows.

## Để lại cho step 2

- **Tên biến thể** kiểu `[same_day_mean_wide]`: `run.py` đang map tên bảng thẳng vào class, nên chưa hỗ trợ. Có thể thêm khóa `model = "same_day_mean"`. Step 1 chưa làm.
- `evaluate.py`: thêm thống kê mức nới cửa sổ chỉ trên ô bị che vào `summary`, và Wasserstein cho lượt masked (marginal_sampler được thiết kế để thắng ở đó).
- Chạy nhiều seed (0–4), cả mask random_cell, và độ nhạy theo `window_days`, `min_obs`, `exclude_same_location`.
- Tìm hiểu vì sao MICoDa A17/R1 thua mean_fill (có thể do thiên lệch R1).
- Chạy thử với dữ liệu microbiome thật (time dạng số) và quyết định cách nới cửa sổ cho dạng đó.
- Từ nay mọi model chính phải thắng same_day_mean, không chỉ thắng Hron/Tsagris.
