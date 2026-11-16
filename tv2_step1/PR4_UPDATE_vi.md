# Cập nhật PR #4: step 1 (đối thủ giả + same_day_mean) trên main mới

PR này thay nội dung PR #4 (`step1-decoys-sameday`). Base: `main` sau PR #5 (`53dbf7b`). Đây là PR đầu của chuỗi ba PR xếp chồng:

1. **PR #4 (PR này):** step 1.
2. `eval/pr-tier01-ablation-residual`: step 2–4, base là `step1-decoys-sameday`.
3. `eval/pr-tier23-v1`: step 5–8, base là `eval/pr-tier01-ablation-residual`.

## Mục đích

Thêm một baseline có dùng thông tin thời gian và ba đối thủ giả làm mức sàn, đúng hai việc ưu tiên cao nhất của đề xuất evaluation 5 tầng.

## Phạm vi

- `same_day_mean`: trung bình c/N có trọng số N, cùng châu lục, trong cửa sổ ±28 ngày.
- Ba mức sàn: `zero_fill`, `mean_fill`, `marginal_sampler`.
- Đọc cột thời gian thành ngày thật, thêm cột nhóm (châu lục). Thêm `enabled = false` và `--only` cho `run.py`.
- Sửa F1–F5 và P3 theo review PR #4, chi tiết ở [FIXES_PR4_vi.md](FIXES_PR4_vi.md):
  - F1: `same_day_mean` chỉ đọc hai đầu cửa sổ, không chép cả trục thời gian;
  - F2: từ chối ghép hai lượt chạy khác dữ liệu, mask hoặc cấu hình;
  - F3: ghi và kiểm group map thực dùng;
  - F4: sắp bảng khi có metric 0.0;
  - F5: số thắng/thua sinh từ CSV;
  - P3: từ chối total ngoài miền tính được.
- Không đổi Hron, Tsagris, MICoDa.

**Rebase lên main mới.** Thứ tự và nội dung commit giữ 1:1; message bỏ trailer, email tác giả thống nhất. Xung đột được giải như sau:

- `run.py` của main được viết lại hoàn toàn. Mỗi thay đổi của commit cũ được chuyển vào file mới; `MainModel`, các stage và `paired_summary` giữ nguyên.
- `data.py` giữ trường `times` của main và các trường thời gian/nhóm của PR này.

Commit cuối `chore: paths after repository restructure` làm các việc sau:

- đổi đường dẫn `nextgen/...` sang gốc repo;
- thêm `configs/models_default_a17.toml`: trên main mới MICoDa mặc định là A18, nên muốn chạy lại step 1 đúng A17 phải ghi tường minh `other = false`;
- thêm `same_day_mean` vào `configs/models_all.toml`.

Chi tiết từng xung đột: `tv2_step8/REBASE_REPORT_vi.md` (PR thứ ba).

## Kết quả chính (masked pilot, exploratory)

COVID đầy đủ, che 5% `whole_feature`, seed 0–4. Chỉ số là MAE tỷ lệ trên ô do model quyết định, viết mean ± sd. sd là độ lệch chuẩn mẫu, không phải khoảng tin cậy.

| Model | MAE tỷ lệ |
|---|---:|
| same_day_mean | 0.0351 ± 0.0066 |
| hron | 0.0875 ± 0.0105 |
| micoda_other (A18) | 0.1075 ± 0.0093 |
| tsagris | 0.1079 ± 0.0209 |
| mean_fill (sàn) | 0.1354 ± 0.0145 |
| micoda (A17) | 0.1406 ± 0.0099 |
| zero_fill (sàn) | 0.1693 ± 0.0219 |

**Kiểm tra sau rebase:**

- 329 test pass với `-W error::RuntimeWarning` (`tests/` và `tv2_step1/`).
- Chạy lại lượt native của mọi model và lượt masked `whole_feature` seed 0: CSV và file dự đoán giống từng byte run cũ.
- `summarize_runs.py` sinh lại bảng step 1 từ run có sẵn, giống từng byte.

## Giới hạn

- Kết quả là một bảng cố định, seed 0–4 đã được nhìn khi phát triển, nên chỉ là exploratory. Đây không phải độ chính xác trên ô thiếu thật.
- Lợi thế của `same_day_mean` đến cùng lúc từ thời gian, châu lục, trọng số N và quy tắc phần dư. Phần tách từng yếu tố (ablation) nằm ở PR thứ hai.
- Trong các bảng của step 1–8, `micoda` = A17 (`other = false`) và `micoda_other` = A18. Trên main sau PR #5, `micoda` mặc định là A18. Tên trong bảng kết quả giữ nguyên như lúc chạy.

## Chưa làm so với đề xuất

- Chỉ số F1 cho câu hỏi có/vắng mặt.
- Sai số theo thời gian (sai phân, tương quan quỹ đạo).
- Kiểu che `mnar` và `pattern_transplant`.
- JSD composition theo tuần ở Tầng 2.
- Tầng 4.
