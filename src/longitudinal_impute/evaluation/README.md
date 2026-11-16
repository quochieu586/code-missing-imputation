# Evaluation v1 (5 tầng)

Module đánh giá các phương pháp điền bảng count. Cấu hình đóng băng và ảnh chụp đầu vào nằm ở `configs/evaluation/`.

| Tầng | File | Đã hiện thực |
|---|---|---|
| 0 Hợp lệ | `tier0.py` | Trạng thái mọi run đã lên lịch (kể cả run fail), reliability, ghép run có kiểm tra (F2), chính sách fail |
| 1 Che giả có đáp án | `tier1.py`, `metrics.py`, `masks.py` | MAE/MSE tỷ lệ trên ô do model quyết định; ghép cặp trên seed chung (F07); sign test; theo châu lục và nhóm N; mask `whole_feature`, `random_cell`, `whole_feature_matched` |
| 2 Ô thiếu thật | `tier2.py` | D1 (chênh tỷ lệ 0), D2 (W1 phần dương), D3 (cờ ngoài support thời gian). Đây là mô tả, không phải độ chính xác |
| 3 Kiểm định Tầng 2 | `tier3.py` | U1–U4 trên ô che giả, ngưỡng đạt khai báo trước, so trong từng ca |
| 4 Downstream | — | Chưa làm |

`protocol.py` đọc `eval_v1.toml`, kiểm hash đóng băng, kiểm seed chưa dùng (`used_seeds.json`) và chấm giả thuyết. `pipeline.py` thực hiện các lệnh.

## Chạy

```bash
PYTHONPATH=src python -m longitudinal_impute.evaluation check --config configs/evaluation/eval_v1.toml
PYTHONPATH=src python -m longitudinal_impute.evaluation evaluate --config configs/evaluation/eval_v1.toml \
  --runs <run xác nhận> --exploratory-runs <run seed 0–4> [<thư mục khác>] --out <thư mục output>
```

- **Thư mục run:** `--runs` chứa `<pass>_<mask>` của lượt xác nhận (pass `main`, `micoda_other`, `residual`). `--exploratory-runs` chứa các run seed 0–4 cùng tên; lệnh `evaluate` dùng chúng để chạy lại Tầng 2 native và so với `configs/evaluation/step5_tier2/`.
- **Output:** `v1_results.md`, `v1_pairs.csv`, `v1_cases.json`, `v1_manifest.json`.
- **`masks` và `run`:** chỉ dùng một lần cho một bộ seed mới. Mask của seed 200–209 đã sinh; không chạy lại.
- **Kiểm tra chạy trước:** mọi lệnh đều từ chối khi `eval_v1.toml` lệch bản đóng băng (sha256 trong `masks_manifest.json`, sau khi bỏ khối amendment), hoặc khi seed xác nhận đã dùng ở step 1–7.

## Kết quả lượt xác nhận (seed 200–209, mask chính `whole_feature`)

| ID | Giả thuyết | Kết quả |
|---|---|---|
| H1, H2 | same_day_mean có MAE c/N thấp hơn Hron và mọi model thật khác | xác nhận, 8/8 seed chung |
| H3 | Hron < Tsagris theo MAE | không xác nhận (6/8) |
| H4 | Tsagris < Hron theo MSE | xác nhận (8/8) |
| H5 | MICoDa A17 không thắng mean_fill | không xác nhận (A17 thắng 8/8) |
| H6 | micoda_other < micoda | xác nhận (8/8) |
| H7, H8, H9 | D3 đạt U3; D1 đạt U1 chiều điền thừa; D1 không đạt chiều điền thiếu | xác nhận ở cả 3 mask |

**Giới hạn:**
- Kết quả chỉ nói về biến thiên theo mask trên một bảng cố định. Đây không phải độ chính xác trên ô thiếu thật, cũng không phải khoảng tin cậy theo nước.
- Seed 200 và 203 làm Hron, Tsagris và MICoDa fail, nên so sánh ở mask chính có điều kiện trên 8 seed.
- D2 không ổn định giữa các mask, nên chỉ dùng để mô tả. `marginal_sampler` không bị diagnostic nào bắt.
- Coverage R-local: 0,56 trên ô thiếu thật, 0,90 trên ô che.
- Trong các bảng, `micoda` = A17 và `micoda_other` = A18; chạy lại A17 bằng `configs/models_default_a17.toml`.

## Thêm một model mới

1. Đăng ký model trong `run.py` (`MODELS`) và thêm bảng `[tên]` vào một file models toml.
2. Tạo một config mới (ví dụ `eval_v1_<model>.toml`) liệt kê model và lượt chạy của nó, rồi commit trước khi chạy.
3. Chạy `run` vào thư mục mới, rồi chạy `evaluate`.
4. Tuning chỉ dùng seed 0–4, không bao giờ dùng seed 200–209. Kết quả trên 200–209 đã được nhìn trong lúc phát triển thì phải ghi nhãn exploratory.

## Kết quả đầy đủ và phân tích chạy một lần

`git checkout evaluation-v1-results` chứa báo cáo, bảng và manifest của step 1–8. Các phân tích chạy một lần cũng chỉ còn ở tag này:
- tổng hợp step 1;
- ablation `same_day_mean` (step 3);
- quy tắc phần dư (step 4);
- bảng riêng của step 6 và step 7;
- các script `run_all.sh`.
