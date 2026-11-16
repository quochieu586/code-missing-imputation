# Sửa theo review PR #4 (TV1, head 0800e35)

Branch `step1-decoys-sameday`. `main` chưa tiến (a5a918a), nên không cần rebase. Không sửa `evaluate.py`; thuật toán Hron/Tsagris/MICoDa giữ nguyên; kết quả số của same_day_mean trên COVID giữ nguyên.

## Tổng quan kiểm tra

- **Test:** `PYTHONPATH=src python -W error::RuntimeWarning -m pytest tests tv2_step1` cho **185 passed**: 177 trong `tests` (156 trước review + 21 mới) và 8 cho `tv2_step1/summarize_runs.py`.
- **Chạy lại cả 4 run** (main + micoda_other, whole_feature + random_cell, seed 0–4) sau F1/P3/F3. So với run trước review:
  - mọi `*_completed.csv` và `mask_*.npy` giống từng byte;
  - `summary.json` bằng nhau, trừ `seconds`;
  - `summary.csv` chỉ khác ở cột `seconds`.
- **Bảng kết quả:** 80 dòng của `results_multiseed.csv` giữ nguyên giá trị mọi cột cũ, chỉ thêm cột `mse_count_*`. Riêng same_day_mean whole_feature seed 0–4: MAE tỷ lệ, MSE tỷ lệ và MAE count giống hệt CSV cũ.
- **Nguồn gốc run:** run main ghi commit a5807df (cây sạch). Run micoda_other ghi d1e391b với `uncommitted_changes = true`; lúc đó chỉ có docstring của `marginal_sampler.py` và contract doc chưa commit (về sau nằm trong d8cdb39), không đụng code của MICoDa.

## F1: bộ nhớ của same_day_mean (commit e698e77)

- **Sửa:** thay `tables['group'][:, self.groups_[rows]]` bằng `table[:, g, hi] - table[:, g, lo]` với `g = self.groups_[rows]`; làm tương tự cho bảng pair (`exclude_same_location`). Mỗi dòng giờ chỉ đọc hai endpoint.
- **Số đo** (`tracemalloc`, peak trong `predict()`, n dòng với n ngày khác nhau, một nhóm, nửa số dòng thiếu target a):

| n | Trước (exclude=false) | Sau | Trước (exclude=true, 10 entity) | Sau |
|---:|---:|---:|---:|---:|
| 1.000 | 11,87 MiB | 0,24 MiB | 23,76 MiB | 0,67 MiB |
| 2.000 | 46,62 MiB | 0,46 MiB | 93,29 MiB | 1,33 MiB |
| 4.000 | 184,78 MiB | 0,91 MiB | 369,67 MiB | 2,65 MiB |

- **Test** (`tests/test_floor_and_same_day.py`):
  - `test_same_day_mean_matches_a_direct_donor_oracle`: 12 case ngẫu nhiên (3 nhóm, 40 mốc ngày, 8 entity, cả hai giá trị `exclude_same_location`, đủ các mức nới). Output và `levels_` khớp với một oracle vòng lặp trực tiếp trên donor.
  - `test_same_day_mean_predict_memory_does_not_grow_with_rows_times_days`: 4.000 dòng, 4.000 ngày, **ngưỡng 16 MiB** cho cả hai chế độ.
  - Test bộ nhớ fail trên code trước F1; oracle pass cả trước lẫn sau (kết quả không đổi).
- **COVID:** chạy lại same_day_mean whole_feature seed 0–4. `same_day_mean_completed.csv` giống từng byte; `errors` và `info` của lượt masked giống hệt.

## F2: ghép run trong summarize_runs.py (commit 9857243, kết quả 2a2c24f)

- **Sửa:** trước khi ghép, `validate_pair()` kiểm tra:
  - cùng `data_sha256`, schema, kiểu mask, rate, seeds và group map;
  - `np.array_equal` từng mask đã lưu;
  - run main có MICoDa `other = false`, run micoda_other chỉ có MICoDa `other = true`.
  
  Sai bất kỳ điểm nào thì báo `ValueError` và **không ghi file nào**. Script tự ghi file output, không còn dùng `> file` (cách cũ để lại file rỗng khi lỗi).
- **Manifest** `tv2_step1/results_manifest.json`: mỗi run gồm commit git, hash data/schema, kiểu mask, rate, seeds, sha256 từng mask, tham số model và hash, group map (đường dẫn tương đối theo repo, sha256, số dòng map, số nhóm).
- **Test** (`tv2_step1/test_summarize_runs.py`, run giả):
  - cặp hợp lệ ghi đủ 3 file;
  - lệch mask, lệch data hash, `other = false`, lệch rate, run main có `other = true`: đều báo lỗi và thư mục output không được tạo.
- **MSE count:** đã thêm vào CSV (cột `mse_count_*`), lấy từ các run đã qua kiểm tra này.

## F3: nguồn gốc group map (commit a5807df)

- **Sửa:**
  - `load_csv` hash đúng những byte đã parse và lưu `Dataset.group_map = {path (đã resolve), sha256, entries, rows_resolved, groups}`; nếu nhóm lấy từ cột CSV hoặc không có `group_col` thì là `None`.
  - `run.py` ghi trường này vào `config.json` (`null` nếu không dùng map).
  - `summarize_runs.py` không tính lại bảng mức cửa sổ khi hash data hoặc hash map hiện tại khác `config.json`, hoặc khi run cũ chưa có trường này.
- **Test** (`tests/test_time_and_groups.py`):
  - provenance bằng hash của file;
  - chỉ đổi nội dung map thì sha256 trong `config.json` đổi, đường dẫn giữ nguyên;
  - không dùng map thì ghi `null`.
- **COVID:** map `configs/covid_location_continent.csv` có sha256 `5e0547a9…`, 150 dòng, 6 nhóm, đã resolve cho 11.671 dòng.

## F4: sắp xếp bảng (commit bf14cc7)

- **Sửa:** bỏ `metric(...) or np.inf`. Khóa sắp xếp mới là (thiếu seed nào không, trung bình các giá trị hữu hạn, tên model), kiểm tra `None`/NaN tường minh. Kết quả: 0,0 đứng trước giá trị dương; model có seed lỗi/thiếu đứng cuối.
- **Test:** `test_zero_error_sorts_first_and_incomplete_models_last` (perfect 0,0, worse 0,1, một seed lỗi, toàn bộ lỗi).

## F5: số thắng/thua (commit d1e391b, tài liệu efa456c)

- **Sửa:**
  - `pairs_from_csv()` **đọc lại** `results_multiseed.csv` và đếm, theo từng mask và từng metric (MAE/MSE tỷ lệ, MAE/MSE count), số seed A thắng B, số hòa, hiệu trung bình A − B và tỷ số trung bình A/B.
  - `pair_markdown()` sinh bảng và câu, rồi script điền khối này vào REPORT và PR giữa `<!-- auto:pairs:start/end -->`. Không còn số k/5 nào gõ tay trong bảng.
  - Các cặp: same_day_mean/hron, tsagris/hron, micoda_other/mean_fill, micoda/mean_fill, micoda_other/micoda.
- **Đã sửa câu sai:** Tsagris thắng Hron về MSE tỷ lệ 5/5 ở cả hai mask; về MAE count chỉ 2/5 (whole_feature) và 0/5 (random_cell). Mọi câu k/5 còn lại trong REPORT/PR đã được đối chiếu với khối tự sinh và đều khớp.
- **Test:** `test_win_counts_come_from_the_csv_per_metric` (thắng theo seed khác với điều trung bình gợi ý; mỗi metric tính riêng; khối giữa marker được thay đúng).

## P3: guard số học (commit c3ea968)

- **Lựa chọn: raise lỗi rõ ràng, không rơi xuống mức fallback kế tiếp.** Fallback sẽ âm thầm đổi sang một ước lượng khác cho đúng những ô có vấn đề số học, và repo đã có nguyên tắc "không fallback ngầm".
- **Sửa:**
  - Miền input hỗ trợ: tổng N của mọi dòng ≤ 2^53 (~9,0e15; COVID 1,5e7), để tổng tiền tố của count nguyên là chính xác. Ngoài miền này `fit` báo `BaselineInputError`.
  - Lớp guard thứ hai trong `predict`: tổng/mẫu số cửa sổ hoặc toàn cục không hữu hạn, hoặc mẫu số ≤ 0, thì báo `NumericalError` thay vì điền 0.
  - Ghi miền hỗ trợ và hai case ngoài phạm vi trong docstring và contract.
- **Test:**
  - Hai input của TV1 (hai donor N = 1e308; donor N = 1e20 ngoài cửa sổ cạnh donor N = 100 trong cửa sổ) bị từ chối, cả khi RuntimeWarning bị bỏ qua lẫn khi coi là lỗi.
  - Khi bỏ qua bước kiểm tra của `fit`, guard trong `predict` báo `NumericalError` cho cả hai.
  - 6 test này đều fail trên code trước P3.
- **COVID:** output không đổi.

## Diễn giải (commit d8cdb39)

REPORT, PR description và contract nay ghi rõ:
- same_day_mean "tốt hơn các baseline trong lượt che giả (masked pilot) này", không phải tốt nhất trên ô thiếu thật.
- sd là độ lệch chuẩn mẫu qua 5 seed, không phải khoảng tin cậy.
- Lợi thế đến từ bốn yếu tố cùng lúc (thời gian, châu lục, trọng số N, quy tắc co phần dư); ablation để step sau.
- 25,5% (mọi ô thiếu thật) và 1,4–9% (ô che thuộc nhóm model) khác mẫu số, không so trực tiếp.
- marginal_sampler bị co theo phần dư nên không đảm bảo Wasserstein ≈ 0 (chỉ sửa docstring).
- Tỷ số MSE count ×1,22 / ×1,02 so với D08 là "chưa kiểm chứng độc lập". Số này nay nằm trong CSV, lấy từ các run đã kiểm tra bằng manifest.
