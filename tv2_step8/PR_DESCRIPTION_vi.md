# PR: Tầng 2–3, mask matched và evaluation v1 xác nhận (step 5–8)

Branch `eval/pr-tier23-v1`. **Base:** `eval/pr-tier01-ablation-residual` (PR step 2–4), branch đó lại chồng lên `step1-decoys-sameday` (PR #4). Đây là PR cuối trong chuỗi ba PR xếp chồng.

## Mục đích

Hoàn thành Tầng 2 (diagnostic trên ô thiếu thật) và Tầng 3 (kiểm định diagnostic trên ô che có đáp án) của đề xuất evaluation 5 tầng. Sau đó đóng băng evaluation v1 và chạy một lượt xác nhận trên seed chưa dùng.

## Phạm vi

- **Step 5, Tầng 2 (`tier2_diagnostics.py`):**
  - D1: chênh tỷ lệ zero;
  - D2: W1 trên phần dương;
  - D3: cờ ngoài support thời gian.
  Reference R-local là các nước cùng châu lục, trong ±28 ngày; R-pooled là sensitivity. Chỉ dùng để mô tả, không xếp hạng.
- **Step 6, Tầng 3 (`tier3_calibration.py`):** use case U1–U4 với ngưỡng khai báo trước. So sánh luôn trong từng ca (mask, seed), có cả hàng truth và đối thủ giả.
- **Step 7:** mask `whole_feature_matched` (`matched_mask.py`) khớp phân bố châu lục × nhóm N của ô thiếu thật; π được đóng băng trước khi sinh mask.
- **Step 8:**
  - `eval_v1.toml` đóng băng cấu hình và giả thuyết H1–H9 trước khi sinh mask;
  - lượt xác nhận chạy một lần trên seed 200–209 với 3 mask (`whole_feature` chính, `whole_feature_matched`, `random_cell`).
- Không đổi baseline, metric, ngưỡng, use case hay mask so với step 2–7.

**Sau rebase lên main mới (PR #5).** Commit `chore: paths after repository restructure` làm các việc sau:

- đổi đường dẫn;
- lượt main đọc `configs/models_default_a17.toml`;
- `tier2.recorded_table()` so bảng `[micoda]` của run cũ (thời mặc định A17) với file A17;
- `eval_v1.toml` được sửa trong một khối amendment. `v1.frozen_bytes()` đảo lại đúng các thay đổi đó, nên kiểm tra đóng băng vẫn so với sha256 trong `masks_manifest.json`;
- `tier2_manifest.json` được sinh lại: 7 trường `models_file`.

Chi tiết: [REBASE_REPORT_vi.md](REBASE_REPORT_vi.md); ánh xạ hash cũ → mới: [HASH_MAP.md](HASH_MAP.md).

## Kết quả chính (lượt xác nhận, seed 200–209)

Mọi kết luận nói về **biến thiên theo mask trên một bảng COVID cố định**. Đây không phải độ chính xác trên ô thiếu thật, không phải khoảng tin cậy theo nước, và không tổng quát hóa sang nước mới.

| ID | Giả thuyết | Mask chính (wf) |
|---|---|---|
| H1, H2 | `same_day_mean` có MAE c/N thấp hơn Hron và mọi model thật khác | **xác nhận**, thắng 8/8 seed chung (Δ −0.068 đến −0.108); wfm và rc 10/10 |
| H3 | Hron < Tsagris theo MAE | **không xác nhận** (6/8, Δ −0.005); wfm và rc thì đạt |
| H4 | Tsagris < Hron theo MSE | **xác nhận** (8/8): hai metric xếp hai model ngược nhau |
| H5 | MICoDa A17 không thắng `mean_fill` | **không xác nhận**: A17 thắng 8/8 (Δ −0.015), ngược với exploratory |
| H6 | micoda_other < micoda | **xác nhận** (8/8) |
| H7 | D3 đạt U3 (cờ ngoài support) | **xác nhận** ở cả 3 mask |
| H8 | D1 đạt U1 chiều điền thừa | **xác nhận**, nhưng sát ngưỡng (min 0.80) |
| H9 | D1 không đạt U1 chiều điền thiếu | **xác nhận** |

Ngoài giả thuyết:

- D2 không ổn định giữa các mask (không đạt ở wf và wfm, đạt ở rc), nên v1 chỉ dùng D2 để mô tả.
- Đối thủ giả `marginal_sampler` không bị D1, D2 hay D3 bắt ở ca nào trong cả 3 mask. Trong khi đó, MAE Tầng 1 xếp nó kém hơn mọi model thật ở 8/8 (wf), 10/10 (wfm) và 9/10 (rc) ca. Đây đúng là kiểu gian lận khớp phân phối mà đề xuất đã cảnh báo.

**Kiểm tra sau rebase:**

- 492 test pass, 0 skip.
- `evaluate_v1.py check` và `evaluate` trên các run xác nhận có sẵn cho 4 file output giống từng byte bản đã commit.
- Bảng step 5–7 sinh lại giống từng byte.
- Chạy lại native và masked seed 0 cho CSV và long file giống hệt run cũ.

## Giới hạn

- Ở `whole_feature`, seed 200 và 203 xóa hết dòng đầy đủ, làm Hron, Tsagris và MICoDa (A17, A18, các biến thể cap) fail. So sánh ở mask chính vì vậy **có điều kiện** trên 8 seed chung. Hai mask sensitivity không có run nào fail.
- So sánh MICoDa A17 với `mean_fill` đổi chiều giữa seed 0–4 (`mean_fill` thắng 4/5) và seed 200–209 (A17 thắng 8/8). Không nên dựa vào cặp này.
- Coverage R-local trên ô thiếu thật chỉ 0.56, so với 0.90 trên ô che. Kết luận Tầng 3 chỉ đúng cho vùng có support.
- Omicron và Delta chiếm phần lớn ô che nhưng không có ô thiếu thật nào do model quyết định; không mask nào khớp theo target.
- Châu Đại Dương ít ô, sd lớn; ở đó `same_day_mean` không thắng rõ.
- Một bảng cố định, không bootstrap theo nước, không khoảng tin cậy.
- Trong các bảng của step 1–8, `micoda` = A17 (`other = false`) và `micoda_other` = A18. Trên main sau PR #5, `micoda` mặc định là A18.

## Chưa làm so với đề xuất

- Chỉ số F1 cho câu hỏi có/vắng mặt.
- Sai số theo thời gian (sai phân, tương quan quỹ đạo).
- Kiểu che `mnar` và `pattern_transplant`.
- JSD composition theo tuần ở Tầng 2.
- Tầng 4 (downstream).
- Mask khớp theo target hoặc theo pattern thiếu đồng thời giữa các nước.
