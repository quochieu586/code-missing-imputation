# PR: evaluation Tầng 0–1, ablation same_day_mean, quy tắc phần dư (step 2–4)

Branch `eval/pr-tier01-ablation-residual`. **Base:** `step1-decoys-sameday` (PR #4). PR này là PR thứ hai trong chuỗi ba PR xếp chồng; `eval/pr-tier23-v1` (step 5–8) chồng lên PR này.

## Mục đích

Dựng Tầng 0 (kiểm hợp lệ) và Tầng 1 (che giả có đáp án) của đề xuất evaluation 5 tầng. Sau đó trả lời hai câu hỏi:

- Lợi thế của `same_day_mean` đến từ đâu?
- So sánh có công bằng không khi các model chia phần dư khác nhau?

## Phạm vi

- **Step 2 (Tầng 0–1):**
  - Định nghĩa sai số dùng chung `evaluate.cell_errors`, thêm MAE series-macro.
  - File dự đoán dạng long theo từng ô che (`--save-predictions`, `cells.py`), có mã lỗi.
  - Bảng trạng thái giữ mọi run đã lên lịch, kể cả run fail.
  - Ghép cặp trên tập seed chung (F07): `tv2_step2/tier01.py`.
- **Step 3:** ablation `same_day_mean` với bốn công tắc `use_time`, `use_group`, `weighting`, `residual_rule`; mặc định giữ nguyên từng byte. Thêm khóa `model = "..."` để đặt tên biến thể trong file models.
- **Step 4:** khóa `residual_rule` (`prop_other`, `cap`, `rescale`) cho Hron, Tsagris và MICoDa A18 ở tầng restore. Thêm sai số của Other ngầm.
- Không đổi thuật toán của Hron, Tsagris, MICoDa. Không đổi mặc định của model nào.

**Sau rebase lên main mới (PR #5):**

- Commit `chore: paths after repository restructure`:
  - đổi đường dẫn;
  - `run_all.sh` của step 2 chạy `configs/models_default_a17.toml`;
  - `check_rule` lấy mặc định `other` của `MICoDaCovid`, là A18 sau PR #5;
  - các test từng giả định MICoDa mặc định A17 nay ghi A17 tường minh.
- Commit `test: MICoDa golden ... within rtol 1e-9`:
  - test hash từng byte của `test_residual_rule.py` giữ so byte cho model không lặp;
  - MICoDa và micoda_other so với mảng golden bằng `assert_allclose(rtol=1e-9, atol=1e-9)`, kèm status khớp chính xác. MICoDa lặp đến hội tụ, nên các bit cuối phụ thuộc nền tảng (khác trên Linux với numpy 2.5.5).

## Kết quả chính (exploratory, seed 0–4, che 5%)

1. **Tầng 1:** `same_day_mean` đứng đầu theo MAE tỷ lệ ở cả `whole_feature` và `random_cell`, thắng mọi model 5/5 seed.
   - Hron xếp trên Tsagris theo MAE, nhưng theo MSE thì ngược lại.
   - Ở `whole_feature`, MICoDa A17 không vượt chắc `mean_fill`: `mean_fill` thắng 4/5 seed.
2. **Ablation:**
   - Thời gian là yếu tố chính. Bỏ cửa sổ thời gian làm MAE tăng khoảng 0.10, về ngang `mean_fill`.
   - Quy tắc phần dư đứng thứ hai: `rescale` làm tăng 0.046–0.053, chủ yếu ở các ô có đáp án 0.
   - Châu lục và trọng số N đóng góp nhỏ.
3. **Quy tắc phần dư:**
   - Khi mọi model dùng cùng một quy tắc, khoảng cách `same_day_mean` − Hron gần như hoàn toàn do thuật toán (98–99%).
   - `rescale` kém nhất với mọi model.
   - Lợi thế của A18 so với A17 đến từ quy tắc R2, không phải từ lõi.

**Kiểm tra sau rebase:**

- 435 test pass với `-W error::RuntimeWarning` (`tests/`, `tv2_step1`–`tv2_step4`).
- `tier01.py`, `ablation.py` và `residual.py` sinh lại mọi bảng và manifest từ run có sẵn, giống từng byte.

## Giới hạn

- Seed 0–4 đã được nhìn và dùng để chỉnh, nên mọi kết luận là exploratory. sd không phải khoảng tin cậy. Chưa có bootstrap theo nước.
- Seed 109 (`whole_feature`) xóa hết dòng đầy đủ, làm Hron, Tsagris và MICoDa fail. Seed này chỉ nằm trong bảng trạng thái.
- So sánh MICoDa A17 với `mean_fill` đổi chiều giữa hai lượt:
  - seed 0–4: `mean_fill` thắng;
  - lượt xác nhận seed 200–209 (PR thứ ba): A17 thắng 8/8 seed.
  Không nên dựa vào cặp này.
- Other ngầm chỉ đo được trên 5 nước có dòng đầy đủ.
- Trong các bảng của step 1–8, `micoda` = A17 (`other = false`) và `micoda_other` = A18. Trên main sau PR #5, `micoda` mặc định là A18.

## Chưa làm so với đề xuất

- Chỉ số F1 cho câu hỏi có/vắng mặt.
- Sai số theo thời gian (sai phân, tương quan quỹ đạo).
- Kiểu che `mnar` và `pattern_transplant`.
- JSD composition theo tuần ở Tầng 2.
- Tầng 4.
- TV composition theo dòng và bootstrap theo nước ở Tầng 1.
