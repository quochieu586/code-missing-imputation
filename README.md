# Chạy imputation trên bảng count

> Phạm vi bản code này: promote package từ `nextgen` ra root, bổ sung main model
> và CLI `full/init/refine/evaluate`. Không kèm bộ hình/LaTeX, raw data mới hay
> toàn bộ archive/evidence cục bộ. Một số liên kết evidence `.harness` trong tài
> liệu tiến độ là lịch sử local, không được phát hành cùng code.
> [Audit hiện tại](docs/MAIN_MODEL_AUDIT_20261113_VI.md) còn hai lỗi P2; scaffold
> này chưa được coi là sạch bug hoặc sẵn sàng mở rộng benchmark.
> [Kiểm chứng trước PR](docs/MAIN_MODEL_PUBLICATION_CHECKS_VI.md): 257 tests pass
> và subprocess staged CLI parity/no-torch trên fixture.

Workflow hiện hành chạy ba baseline Hron, Tsagris và MICoDa trên toàn bộ bảng count. Source, cấu hình và tests đã chuyển từ `nextgen` ra gốc repo. Thuật toán và giả định ngân sách count giữ nguyên.

## Cài môi trường

Chạy từ gốc repo với Python >= 3.11. Dùng lockfile để cài môi trường tại `.venv`:

```powershell
uv sync --frozen --extra test
.venv\Scripts\python.exe -m pytest -q
```

Nếu không dùng uv, tạo venv rồi chạy `python -m pip install -e ".[test]"` trong môi trường đó. Cách này dùng các khoảng phiên bản trong `pyproject.toml`, không khóa phiên bản như `uv.lock`.

## Chạy

**COVID.** Lệnh dưới chạy lượt điền thật và lượt MSE trên 5 mask `whole_feature`:

```powershell
.venv\Scripts\python.exe -m longitudinal_impute.run --data data/covariants.csv `
  --schema configs/baselines_covariants.toml --models configs/models_default.toml `
  --output outputs/covid_run --mask whole_feature --mask-rate 0.05 --seeds 0 1 2 3 4
```

**Dataset khác** (ví dụ microbiome không có cột total):

```powershell
.venv\Scripts\python.exe -m longitudinal_impute.run --data my_counts.csv `
  --schema configs/schema_microbiome_example.toml --models configs/models_default.toml `
  --output outputs/microbiome_run --mask random_cell
```

Sửa schema ví dụ theo tên cột của dataset. Mọi đường dẫn CLI tương đối được tính từ thư mục đang chạy lệnh. Khi chạy ngoài repo, truyền đường dẫn tuyệt đối cho Python, data, schema, models và output; đặt đường dẫn có khoảng trắng trong dấu nháy.

Tham số:

- **Schema** (`[dataset]`): `entity` và `targets` là bắt buộc; `time` và `total` là tùy chọn.
  - Thiếu `total` thì total của một dòng là tổng các target. Khi đó mọi ô target phải có giá trị.
  - Ô trống, `na`, `nan`, `none` được hiểu là thiếu.
- **`--models`**: mỗi bảng `[hron]`, `[tsagris]`, `[micoda]` chứa tham số constructor; bảng rỗng dùng mặc định. Model không có trong file thì không chạy.
- **`--mask`**:
  - `none` (mặc định): chỉ chạy lượt điền thật.
  - `random_cell`: che từng ô observed với xác suất `--mask-rate`.
  - `whole_feature`: che trọn chuỗi của các cặp (entity, target).
- **`--seeds`** (mặc định `0 1 2 3 4`): mỗi seed tạo một mask, dùng chung cho mọi model; phải không âm và không trùng. Rate/seed sai gây `ValueError` trước mọi output/fit; lỗi cú pháp CLI vẫn do argparse xử lý.

Thư mục `--output` phải chưa tồn tại.

## File output

| File | Nội dung |
|---|---|
| `config.json` | Đường dẫn và sha256 của data, schema, tham số model, mask, seeds, git commit (kèm cờ có thay đổi chưa commit). |
| `<model>_completed.csv` | Bảng gốc đã điền ô thiếu thật. Ô observed và các cột khác giữ nguyên văn bản. Chỉ có cho model chạy thành công. |
| `<model>_partial.csv` | Chỉ có cho MICoDa (adapter counts): còn ô NaN chưa giải được; ô observed giữ nguyên, các ô đã điền vẫn hợp lệ (`allow_partial`). |
| `mask_<kind>_seed<s>.npy` | Mask H (bool, N × D) của từng seed. |
| `summary.csv` | Một dòng cho mỗi model × lượt (`native`/`masked`) × seed. Gồm `status` (`ok`/`partial`/`failed`), `space` (`18_with_other` mặc định cho cả ba; `17_targets` khi MICoDa legacy `other=false`), `seconds`, số ô che, số ô đã giải (`resolved_cells`), số dòng đầy đủ còn lại sau mask, MSE/MAE trên count và tỷ lệ (cột hậu tố `_truth_zero`, `_truth_positive`, `_forced_zero`, `_model`, `_model_truth_zero` là các nhóm con). MSE/MAE chỉ được báo khi coverage đầy đủ; không bỏ ô lỗi để có số đẹp. |
| `summary.json` | Toàn bộ chi tiết: Wasserstein theo target của lượt thật, `info_` của model (với MICoDa gồm δ, reference, hội tụ, warning counts), thống kê mask, sai số theo nhóm, `coverage` của lượt che, `aggregate` (mean ± sd qua seed). |
| `errors.log` | Chỉ xuất hiện khi có lỗi hoặc có run không `completed` (kèm trạng thái `partial`). |

Ô "bị ép về 0" nằm ở dòng mà tổng các ô nhìn thấy đã bằng N. Phần dư của dòng đó là 0, nên model không quyết định được ô này. Định nghĩa metric nằm trong [`evaluate.py`](src/longitudinal_impute/evaluate.py).

## MICoDa và ngân sách Other chung

Từ quyết định người dùng ngày 2026-11-12, `[micoda]` chạy qua `MICoDaCovid` trên **D target + Other**, giống không gian ngân sách của Hron/Tsagris. Constructor và `configs/models_default.toml` đều mặc định `other = true`.

- Input là raw counts + `total_sequence` + mask; chuẩn bị **A (min/2 + closure)** tự nằm trong adapter; giữ chính xác mọi observed kể cả zero.
- Dòng complete all-zero bị loại khỏi initial fit và refit, vẫn giữ nguyên ở output (cờ trong diagnostics).
- Tham số TOML: `max_iter` (mặc định 120), `tol` (1e-8), không có `W`. `reference` không đặt để tự chọn cột ít NaN nhất; đây là lựa chọn adaptation, không phải parity nguyên bản paper/R.
- Trạng thái lượt: `completed` / `partial` / `failed` theo `CovidResult`; NaN còn lại luôn là ô chưa giải được, không có fallback ngầm.
- **Mặc định A18/R2:** Other chỉ tính trực tiếp ở dòng đủ target visible; dòng partial để unknown rồi cùng được dự đoán. Residual được chia cho missing targets và Other theo trọng số proposal. MICoDa giữ hàm restore riêng để bảo toàn partial/error diagnostics, nhưng cùng công thức phân bổ với Hron/Tsagris; có test đối chiếu.
- **Legacy opt-in `other = false` (A17/R1):** toàn residual được gán cho target thiếu; chỉ dùng sensitivity, không trộn vào bảng benchmark ngân sách chung. Trên dòng complete vẫn giữ raw counts. Kết quả cũ trong [MICODA_A18_EVALUATION_VI.md](docs/MICODA_A18_EVALUATION_VI.md) là lịch sử, không phải bằng chứng R2 tốt hơn; quyết định mới nhằm thống nhất budget, không nhằm chọn cấu hình có điểm đẹp.
- Khi adapter `failed`, không xuất CSV; giữ mã lỗi, message, context và unassigned trong `summary.json`, ghi lỗi và exit 1. `partial` chỉ xuất `_partial.csv` ở lượt native, không tính metric masked.
- Chi tiết đặc tả: [Contract 3 baseline](docs/THREE_BASELINE_CONTRACT_VI.md), [đối chiếu cross-check](docs/CROSSCHECK_RESPONSE_VI.md).

Benchmark ba baseline mặc định dùng cùng data/schema, total, D+Other, mask và scored cells. Khởi tạo/zero handling của từng thuật toán vẫn khác và được công bố. Accuracy so sánh trên cùng tập seed cả ba completed/coverage=1, báo mọi partial/failure riêng; không xếp hạng bằng các mean lấy từ tập success khác nhau. Chốt metric/config/seeds trước chạy, không dùng hidden truth dựng Other/total. Xem [contract ngân sách và benchmark](docs/BASELINE_OTHER_CONTRACT_VI.md).

`summary.json.paired` và bảng console paired cho biết seed giữ/loại và aggregate trên common-success seeds; không tạo điểm paired khi trộn R1/R2, không có ô che hoặc không có seed chung hợp lệ. `aggregate` cũ vẫn giữ để tương thích và mô tả success riêng.

## Kiểm chứng chạy lại

```powershell
.venv\Scripts\python.exe scripts/verify_three_baselines.py --output outputs/verification `
  --data data/covariants.csv --schema configs/baselines_covariants.toml
```

Script chạy toàn bộ test với RuntimeWarning là lỗi, lưu log/JUnit/hash nguồn, và tính ngân sách R1 trên các dòng complete bị che (không fit model trong phần audit ngân sách). Không sửa dữ liệu.

Kiểm tra các lượt chạy CLI từ thư mục khác, có khoảng trắng trong đường dẫn:

```powershell
.venv\Scripts\python.exe scripts/verify_handoff.py --output outputs/handoff_check
```

Thêm `--data data/covariants.csv --schema configs/baselines_covariants.toml` để kiểm tra cả bảng COVID thật với hai loại mask. Mỗi lần chạy cần thư mục output mới.

Kết quả migration và phạm vi kiểm chứng nằm trong [báo cáo audit bàn giao](docs/MIGRATION_AUDIT_VI.md).

## Vị trí module

| Đường dẫn | Trách nhiệm |
|---|---|
| `src/longitudinal_impute/data.py` | Đọc CSV, kiểm tra schema/count, mask và khôi phục counts |
| `src/longitudinal_impute/baselines/` | Thuật toán Hron, Tsagris và MICoDa |
| `src/longitudinal_impute/run.py` | CLI và điều phối native/masked |
| `src/longitudinal_impute/evaluate.py` | Wasserstein, MSE, MAE và aggregate |
| `src/longitudinal_impute/main_model.py` | MainModel, hai view an toàn và bounded refinement; torch import trễ |
| `src/longitudinal_impute/network.py` | Network temporal/feature nhỏ, composition head |
| `configs/`, `tests/`, `scripts/` | Cấu hình, kiểm thử và kiểm chứng bàn giao |
| `data/` | Hai CSV nguồn dùng chung, không bị ghi đè khi chạy |
| `outputs/` | Vị trí output theo các lệnh ví dụ, không đưa vào Git |
| `archive/` | Runtime/scripts/tests cũ và lịch sử baseline đã được Git track; không phải toàn bộ archive nghiên cứu local |

## Đọc `errors.log`

Mỗi dòng có dạng `model | lượt | seed | loại lỗi | thông báo [ngữ cảnh]`, ví dụ:

```
tsagris | native | - | BaselineError | row 13: observed parts sum to 0, JSD is undefined [row=13, reason=zero_anchor, entity=Brill, csv_line=15]
```

- `row` là chỉ số dòng 0-based. `entity` và `csv_line` (dòng 1 là header) chỉ ra dòng đó trong CSV.
- `feature` là tên target, hoặc `Other` cho phần không thuộc target nào.
- Lỗi của một model không dừng các model khác. Exit code là 1 nếu có ít nhất một lỗi.

Đặc tả hiện hành nằm trong [contract ba baseline](docs/THREE_BASELINE_CONTRACT_VI.md). Lịch sử KNN + CSDI và hướng dẫn bảo tồn nằm trong [archive](archive/README.md). [Báo cáo thay đổi Hron/Tsagris](archive/legacy_knn_csdi_2026-11-02/docs/tv2_hron_tsagris_changes_vi.md) là tài liệu lịch sử.

## Main model — full và chạy từng stage

Baseline-only vẫn dùng `configs/models_default.toml`, không yêu cầu torch. Để dùng main CPU:

```powershell
uv sync --frozen --extra main --extra test
.venv\Scripts\python.exe -m longitudinal_impute.run --data data/covariants.csv `
  --schema configs/baselines_covariants.toml --models configs/models_all.toml `
  --output outputs/all_models_run --mask whole_feature --mask-rate 0.05 --seeds 0
```

Đổi `--models configs/main.toml` để chỉ chạy main. Mỗi model/lượt vẫn có instance riêng; constructor main không import torch. Đây là ví dụ lệnh trên dữ liệu thật, **chưa được dùng để benchmark accuracy/tốc độ dữ liệu thật**; verification dùng fixture hai dataset. Preset là recipe khởi đầu, không phải cấu hình đã tuning.

Main khởi tạo Tsagris → CLR → train/predict hai view → restore counts, lặp có giới hạn. Thông số nằm trong TOML; giảm `outer_iters`/`epochs_per_iter` trong bản config riêng khi debug. `[main].seed` là model/training seed, khác `--seeds` tạo evaluation masks. Không redraw Q khi Q rỗng/thiếu donor.

Diagnostics trong `summary.json` → native/masked → main → info gồm `trained`, `iterations`, `converged`, `termination_reason`, `history`. Dừng khi max change proportion trên target thiếu **≤ tol**, hoặc hết vòng. `max_iter` với output hợp lệ vẫn completed nhưng `converged=false`; `no_missing` thì `trained=false`, không gọi network. Mỗi vòng ghi loss, change, seconds và converged. Lỗi neural/optional torch (kể cả DLL import trên Windows) được giữ nguyên nguyên nhân trong lỗi domain để baseline tiếp tục; không dùng fallback ngầm.

### Khởi tạo → refinement → chấm lại

```powershell
.venv\Scripts\python.exe -m longitudinal_impute.run --stage init `
  --data data/covariants.csv --schema configs/baselines_covariants.toml `
  --models configs/main.toml --output outputs/main_init `
  --pass both --mask whole_feature --mask-rate 0.05 --seeds 0
.venv\Scripts\python.exe -m longitudinal_impute.run --stage refine `
  --from-run outputs/main_init --output outputs/main_refined
.venv\Scripts\python.exe -m longitudinal_impute.run --stage evaluate `
  --from-run outputs/main_refined --output outputs/main_rescored
```

- Mặc định `--stage full`; `init/refine` chỉ nhận main-only. Init không cần torch, ghi `initialized`, chưa có accuracy của main đã train. Refine dùng snapshot, không chạy lại Tsagris; không resume giữa epoch/iteration.
- `--pass native` chỉ điền thiếu tự nhiên; `masked` chỉ đánh giá mask; `both` chạy cả hai. Hai lựa chọn sau cần `--mask random_cell|whole_feature`. Không truyền `--pass` thì giữ hành vi cũ: native khi mask=none, cả hai khi có mask.
- Evaluate dùng prediction đã lưu, không construct/fit model hay import torch; chấm lại metric, giữ thời gian compute gốc và trạng thái failed/partial. Không chấm tập con partial.
- Mọi output phải là thư mục mới, ngoài run nguồn. Giữ nguyên CSV/schema/models của run nguồn; nếu chuyển chỗ, truyền lại `--data/--schema/--models` có nội dung byte-identical. Đổi data/config thì bắt đầu run mới. Hash/path/mask/state không tương thích bị từ chối trước khi tạo output; checksum không phải chữ ký xác thực.
- `artifacts.json` lập chỉ mục/hash; `predictions/*.npy` lưu prediction theo case/model; `main/<case>/initial.npz` lưu init và `main/<case>/history.csv` lưu history của full/refine. Numeric arrays đọc không dùng pickle.

Chi tiết hiện thực/RED–GREEN và giới hạn kiểm chứng: [MAIN_MODEL_PROGRESS_VI.md](docs/MAIN_MODEL_PROGRESS_VI.md).
