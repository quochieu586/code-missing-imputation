# Contract vận hành 3 baseline

## Cập nhật có hiệu lực 2026-11-12

Người dùng đã duyệt **MICoDa `other=true` làm mặc định**, thống nhất D+Other/R2 với Hron/Tsagris. Constructor, cấu hình phát hành và nhãn fallback runner đã đổi. `other=false` giữ lại như legacy sensitivity, không dùng trong benchmark common-budget. Quyết định này **thay thế D03/D04 và mặc định/gate thăng hạng D08 bên dưới**; bảng D01–D08 giữ làm lịch sử, không còn là mặc định vận hành. D07 (interface counts) giữ nguyên.

Contract hiện hành về Other và điều kiện benchmark nằm tại [BASELINE_OTHER_CONTRACT_VI.md](BASELINE_OTHER_CONTRACT_VI.md). Việc chuyển mặc định nhằm cùng bài toán ngân sách, không khẳng định A18 có accuracy cao hơn A17 hay đã chạy benchmark xác nhận main model.

## Phạm vi và trạng thái

Runtime của ba baseline ở [`src/longitudinal_impute/`](../src/longitudinal_impute/). Từ 2026-11-02, workflow này nằm tại gốc repo; code và kết quả KNN + CSDI trước đây được lưu trong [`archive/`](../archive/README.md). Migration giữ nguyên thuật toán và các quyết định MICoDa dưới đây. Luồng chạy tuần tự, không có framework mới.

Nguồn tích hợp lịch sử kế thừa PR #2 tại `4b3e971`. Thông tin PR trong [bản đối chiếu ngày 2026-10-30](CROSSCHECK_RESPONSE_VI.md) mô tả thời điểm audit đó, không phải trạng thái merge hiện tại.

| Lớp | Chủ sở hữu | Trách nhiệm |
|---|---|---|
| Entry point | `run.py:main` | CLI, preflight, config/provenance, native, mask theo seed, gọi từng model, xuất summary/log và exit code |
| Biên dữ liệu | `data.py` | CSV width/schema/count validation; mask; Hron/Tsagris model input và restore; writer không overwrite |
| Hron/Tsagris | `baselines/hron.py`, `baselines/tsagris.py` | fit/predict; kế thừa nguyên bản sửa F1/F2/E1 từ tác giả PR #2 |
| MICoDa (lõi V2 + adapter COVID, một file) | `baselines/micoda.py` | Lõi: ALR, hồi quy, phân bổ, refit — bất khả xâm phạm dataset. Adapter: raw counts + N + mask → D+Other → lõi → R2 mặc định; R1 chỉ khi other=false; status/diagnostics có cấu trúc |
| MICoDa lõi V2 | `baselines/micoda.py` (phần lõi trong cùng file) | ALR, hồi quy intercept-only, phân bổ và refit; kiểm tra state dương |
| Đánh giá | `evaluate.py` | W1 native (không có truth NaN); MSE/MAE ô che; aggregate mean/sample SD trên run ok |

## Luồng plain text

```text
CLI → kiểm tra mask/rate/seed → load CSV/schema → validate model constructors
    → tạo output mới + config.json
    → native: mỗi model dùng instance mới, điền NaN thật
        → Hron/Tsagris: model_input → fit/predict → restore_counts → check
        → MICoDa: impute_counts → prepare → core.fit → restore R2 (legacy R1 opt-in) → check nếu completed
        → completed: _completed.csv + W1
        → partial: _partial.csv + lỗi, không W1
        → failed: không CSV, giữ lỗi gốc + lỗi log
    → với từng seed: tạo một mask chung → mỗi model instance mới
        → chỉ completed + coverage=1 mới tính MSE/MAE trên ô che
        → failed/partial: giữ diagnostics, không metric, tiếp tục model khác
    → summary.csv/json + bảng có space; exit 1 khi có failed/partial, ngược lại 0
```

Rate/seed không hợp lệ gây ValueError trước side effect; argparse vẫn xử lý lỗi cú pháp CLI. Seeds phải không âm và không trùng. Khi mask none, các tùy chọn mask không dùng. Output directory phải chưa tồn tại. Mask rate 0/không che được ô nào không cung cấp metric, không phải bằng chứng mô hình chính xác.

## Contract dữ liệu

- Chỉ NaN thật hoặc ô hidden mới được điền; observed kể cả zero giữ chính xác numeric và giữ nguyên text khi xuất native CSV. Dữ liệu gốc không ghi lại.
- Output luôn là D target, **không xuất cột Other**. `17_targets`/`18_with_other` là nhãn cấu hình COVID; adapter về kỹ thuật nhận D >= 2, nên các nhãn này không phải số chiều tổng quát cho dataset khác.
- Hron/Tsagris dùng D+1 nội bộ: Other chỉ biết trên dòng visible-complete, các dòng partial để Other là NaN. Restore chia residual cho missing targets và Other.
- MICoDa mặc định D+Other/R2; `other=false` là legacy sensitivity. Adapter riêng được dispatch tường minh trong `run_model`; ba model không có interface fit/predict đồng nhất.
- Writer dùng file tạm cùng thư mục và hard link publish không overwrite; thất bại thì dọn file tạm. Cần filesystem hỗ trợ hard link. Đây không phải cam kết durability khi mất điện.

## Lịch sử các quyết định MICoDa trước 2026-11-12 (đọc cùng cập nhật phía trên)

| Mã | Quyết định giữ nguyên | Giới hạn |
|---|---|---|
| D01 | V2 intercept-only, complete-case initialization | Cần >1 dòng complete hợp lệ, không fallback mean/pairwise |
| D02 | δ=min visible positive target proportion/2 mỗi fit; zero replacement + closure complete, headroom partial | Chỉ sửa model-space; không sửa raw observed |
| D03 | D=17 target COVID, không Other | Khác không gian Hron/Tsagris, chưa phải so sánh kiểm soát khoa học |
| D04 | R1: toàn bộ residual gán cho target thiếu | Nếu N gồm Other thì có thiên lệch khối lượng; cần nhóm chốt lại trước benchmark chính |
| D05 | Complete all-zero loại khỏi initial fit và refit, raw output giữ nguyên | Ghi diagnostics; không lấy các dòng này làm donor hồi quy |
| D06 | max_iter=120, tol=1e-8; dùng fitted state cuối; warning nếu không hội tụ | completed chỉ nói đủ output, không chứng minh convergence hay accuracy |
| D07 | **Chốt 2026-10-30:** giữ interface counts `impute_counts` (phương án A); **không** chuyển fit/predict chung | Dispatch `isinstance` trong runner là chủ đích; đổi interface sau này phải kèm version contract mới và duyệt khoa học lại |
| D08 | **Chốt 2026-10-30:** A17/R1 là mặc định và protocol chính thức. Biến thể `other = true` (A18/R2) chỉ là **sensitivity analysis** opt-in trong cùng adapter. Gate thăng hạng "cả MSE count và MSE tỷ lệ cải thiện trên cả hai mask" **chưa đạt** (random-cell MSE count ~×4.5). Prototype `MICoDa18` fit/predict **không nằm trong `src/`** — lưu dạng tài liệu đánh giá | Không dùng A17 lẫn A18 làm ground truth cho NaN tự nhiên; cả hai bị chặn khỏi benchmark xác nhận main model cho đến khi chốt giả định Other/budget và đánh giá lại với thiết kế xác nhận |

Đây là bản đưa vào Git của các lựa chọn đã duyệt trong hội thoại, không phải biên bản duyệt khoa học độc lập. Chưa tự ý đổi R1 sang R2/thêm Other. Khi một target duy nhất thiếu, R1 gán `N − sum visible` bất kể proposal. Ví dụ truth [20,30], N=100, Other=50: che 30 thì điền 80. Sai lệch 50 xuất phát từ ngân sách, không phải lỗi ALR.

**Gate trước main model:** chọn một contract ngân sách dùng chung (ví dụ Other tường minh, hoặc total của riêng target có nguồn xác định không dùng hidden truth). Không tính `total_target` từ truth của ô che để đưa cho model. Nếu vẫn giữ A17/R1, chỉ báo adaptation có giới hạn, không dùng làm baseline xếp hạng xác nhận main model.

## Kiểm chứng

Từ gốc repo, chạy `.venv\Scripts\python.exe scripts/verify_three_baselines.py --output <new-dir> [--data ... --schema ...]` để kiểm thử, lưu log/JUnit/provenance và audit ngân sách tùy chọn. Chạy `scripts/verify_handoff.py --output <new-dir>` bằng cùng Python để kiểm chứng đường dẫn và các lượt CLI trên fixture COVID và microbiome. Xem [README](../README.md) để cài môi trường và chạy trên dữ liệu thật.

Các test trạng thái giả lập chỉ chứng minh orchestration; test insufficient-fit chạy adapter thật. Các số masked không kiểm định độ đúng của NaN tự nhiên. Với run thất bại, summary.json giữ mã lỗi/message/context, row events và unassigned budget.
