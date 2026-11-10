# Contract vận hành 3 baseline

## Phạm vi và trạng thái

Runtime **của ba baseline** ở `nextgen/src/longitudinal_impute/`. Không phải toàn bộ mã nghiên cứu của repo: pipeline diffusion cũ vẫn ở các thư mục khác. Nhánh tích hợp kế thừa PR #2 tại `4b3e971`; không có nghĩa PR đã merge vào main. Chạy tuần tự, không plugin/framework mới.

| Lớp | Chủ sở hữu | Trách nhiệm |
|---|---|---|
| Entry point | `run.py:main` | CLI, preflight, config/provenance, native, mask theo seed, gọi từng model, xuất summary/log và exit code |
| Biên dữ liệu | `data.py` | CSV width/schema/count validation; mask; Hron/Tsagris model input và restore; writer không overwrite |
| Hron/Tsagris | `baselines/hron.py`, `baselines/tsagris.py` | fit/predict; kế thừa nguyên bản sửa F1/F2/E1 từ tác giả PR #2 |
| MICoDa (lõi V2 + adapter COVID, một file) | `baselines/micoda.py` | Lõi: ALR, hồi quy, phân bổ, refit — bất khả xâm phạm dataset. Adapter trong cùng file: raw counts + N + mask → A17 → lõi → R1; status/diagnostics có cấu trúc |
| MICoDa lõi V2 | `baselines/micoda.py` (phần lõi trong cùng file) | ALR, hồi quy intercept-only, phân bổ và refit; kiểm tra state dương |
| Đánh giá | `evaluate.py` | W1 native (không có truth NaN); MSE/MAE ô che; aggregate mean/sample SD trên run ok |

## Luồng plain text

```text
CLI → kiểm tra mask/rate/seed → load CSV/schema → validate model constructors
    → tạo output mới + config.json
    → native: mỗi model dùng instance mới, điền NaN thật
        → Hron/Tsagris: model_input → fit/predict → restore_counts → check
        → MICoDa: impute_counts → prepare → core.fit → restore R1 → check nếu completed
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
- MICoDa chỉ D target. Adapter riêng được dispatch tường minh trong `run_model`; không tuyên bố ba model có interface fit/predict đồng nhất. Một nhánh đơn giản là chủ đích; không cần framework để che khác biệt ngữ nghĩa.
- Writer dùng file tạm cùng thư mục và hard link publish không overwrite; thất bại thì dọn file tạm. Cần filesystem hỗ trợ hard link. Đây không phải cam kết durability khi mất điện.

## Các quyết định MICoDa đã chọn trong phiên làm việc

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

## same_day_mean và ba mức sàn

**`same_day_mean`** (baseline chính thức, bật mặc định trong `models_default.toml`; [`baselines/same_day_mean.py`](../src/longitudinal_impute/baselines/same_day_mean.py)):

- Ô thiếu (i, j) nhận tham chiếu Σc / ΣN (trung bình c/N có trọng số N) của target j trên các dòng **khác** cùng nhóm `group_col` (COVID: châu lục, qua `configs/covid_location_continent.csv`; không khai báo thì cả bảng là một nhóm), trong cửa sổ ±`window_days` ngày quanh ngày của dòng i. Chỉ ô visible được tính; ô che/NaN không bao giờ vào.
- Tham số: `window_days = 28`, `min_obs = 20`, `max_widen = 2`, `exclude_same_location = false`. Cửa sổ có ít hơn `min_obs` ô visible thì gấp đôi, tối đa `max_widen` lần (±28 → ±56 → ±112 ngày); vẫn thiếu thì dùng trung bình toàn cục (mọi nhóm, trọng số N) của target j. `info_['cells_by_level']` ghi số ô ở mỗi mức.
- Time không phải ngày: dạng số xét cùng giá trị rồi ±1, ±2...; dạng nhãn chỉ cùng nhãn; không có cột time thì lấy cả nhóm.
- **Quy tắc phần dư khác Hron/Tsagris:** đề xuất × N cho các ô thiếu; nếu tổng vượt R = N − Σ visible thì **co lại theo tỷ lệ cho bằng R** (Other = 0), nếu không thì Other nhận phần còn lại. Hron/Tsagris dự đoán cả Other rồi chia R theo tỷ lệ, nên luôn dùng hết R cho tỷ lệ đề xuất; hai cách khác nhau khi tổng đề xuất < R.
- **Hạn chế:** trên ô thiếu thật của COVID (lượt native), 25.5% ô (21 946/86 050) không đủ `min_obs` sau 2 lần nới và phải dùng trung bình toàn cục, tức không còn thông tin thời gian/nhóm. Trên ô bị che tỷ lệ này thấp hơn nhiều (nhóm ô `model`: 1.4–9.0% với `whole_feature`, 0.7–1.0% với `random_cell`, seed 0–4; xem `tv2_step1/results_multiseed.md`), nên điểm masked lạc quan hơn chất lượng trên ô thiếu thật.

**Ba mức sàn** (`zero_fill`, `mean_fill`, `marginal_sampler`; `enabled = false`, chạy bằng `--only`): không phải phương pháp nghiêm túc, chỉ để biết điểm của một model "không học gì". `zero_fill` điền 0 (R vào Other); `mean_fill` điền trung bình không trọng số của c/N toàn bảng; `marginal_sampler` rút một c/N từ ô visible của cùng target (seed trong config), khớp phân phối nhưng sai từng ô. Cùng quy tắc "co lại khi vượt R". Một baseline thật phải thắng cả ba trên nhóm ô `model`.

## Kiểm chứng

`python nextgen/scripts/verify_three_baselines.py --output <new-dir> [--data ... --schema ...]` chạy tests, lưu log/JUnit/provenance và audit ngân sách tùy chọn. Các test trạng thái giả lập chỉ chứng minh orchestration; test insufficient-fit chạy adapter thật. Các số masked không kiểm định độ đúng của NaN tự nhiên. Với run thất bại, summary.json giữ mã lỗi/message/context, row events và unassigned budget.
