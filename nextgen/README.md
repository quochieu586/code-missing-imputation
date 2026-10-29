# Chạy imputation trên bảng count

Chạy từ gốc repo với Python >= 3.11.

```powershell
python -m pip install -e "./nextgen[test]"
python -m pytest nextgen/tests -q
```

## Chạy

**COVID.** Lệnh dưới chạy lượt điền thật và lượt MSE trên 5 mask `whole_feature`:

```powershell
python -m longitudinal_impute.run --data data/covariants.csv `
  --schema nextgen/configs/baselines_covariants.toml --models nextgen/configs/models_default.toml `
  --output experiments/covid_run --mask whole_feature --mask-rate 0.05 --seeds 0 1 2 3 4
```

**Dataset khác** (ví dụ microbiome không có cột total):

```powershell
python -m longitudinal_impute.run --data my_counts.csv `
  --schema nextgen/configs/schema_microbiome_example.toml --models nextgen/configs/models_default.toml `
  --output experiments/microbiome_run --mask random_cell
```

Tham số:

- **Schema** (`[dataset]`): `entity` và `targets` là bắt buộc; `time` và `total` là tùy chọn.
  - Thiếu `total` thì total của một dòng là tổng các target. Khi đó mọi ô target phải có giá trị.
  - Ô trống, `na`, `nan`, `none` được hiểu là thiếu.
- **`--models`**: mỗi bảng `[hron]`, `[tsagris]`, `[micoda]` chứa tham số constructor; bảng rỗng dùng mặc định. Model không có trong file thì không chạy.
- **`--mask`**:
  - `none` (mặc định): chỉ chạy lượt điền thật.
  - `random_cell`: che từng ô observed với xác suất `--mask-rate`.
  - `whole_feature`: che trọn chuỗi của các cặp (entity, target).
- **`--seeds`** (mặc định `0 1 2 3 4`): mỗi seed tạo một mask, dùng chung cho mọi model.

Thư mục `--output` phải chưa tồn tại.

## File output

| File | Nội dung |
|---|---|
| `config.json` | Đường dẫn và sha256 của data, schema, tham số model, mask, seeds, git commit (kèm cờ có thay đổi chưa commit). |
| `<model>_completed.csv` | Bảng gốc đã điền ô thiếu thật. Ô observed và các cột khác giữ nguyên văn bản. Chỉ có cho model chạy thành công. |
| `<model>_partial.csv` | Chỉ có cho MICoDa (adapter counts): còn ô NaN chưa giải được; ô observed giữ nguyên, các ô đã điền vẫn hợp lệ (`allow_partial`). |
| `mask_<kind>_seed<s>.npy` | Mask H (bool, N × D) của từng seed. |
| `summary.csv` | Một dòng cho mỗi model × lượt (`native`/`masked`) × seed. Gồm `status` (ok/failed), `space` (`18_with_other` cho Hron/Tsagris, `17_targets` cho MICoDa), `seconds`, số ô che, số ô đã giải (`resolved_cells`), số dòng đầy đủ còn lại sau mask, MSE/MAE trên count và tỷ lệ (cột hậu tố `_truth_zero`, `_truth_positive`, `_forced_zero`, `_model`, `_model_truth_zero` là các nhóm con). MSE/MAE chỉ được báo khi coverage đầy đủ; không bỏ ô lỗi để có số đẹp. |
| `summary.json` | Toàn bộ chi tiết: Wasserstein theo target của lượt thật, `info_` của model (với MICoDa gồm δ, reference, hội tụ, warning counts), thống kê mask, sai số theo nhóm, `coverage` của lượt che, `aggregate` (mean ± sd qua seed). |
| `errors.log` | Chỉ xuất hiện khi có lỗi hoặc có run không `completed` (kèm trạng thái `partial`). |

Ô "bị ép về 0" nằm ở dòng mà tổng các ô nhìn thấy đã bằng N. Phần dư của dòng đó là 0, nên model không quyết định được ô này. Định nghĩa metric nằm trong [`evaluate.py`](src/longitudinal_impute/evaluate.py).

## MICoDa (adapter counts, 17 targets)

`[micoda]` chạy qua `MICoDaCovid` trên **17 target, không cột Other**:

- Input là raw counts + `total_sequence` + mask; chuẩn bị **A (min/2 + closure)** tự nằm trong adapter; giữ chính xác mọi observed kể cả zero.
- Hoàn complete all-zero bị loại khỏi initial fit và refit, vẫn giữ nguyên ở output (cờ trong diagnostics).
- Tham số TOML: `max_iter` (mặc định 120), `tol` (1e-8), không có `W`. `reference` không đặt để tự chọn cột ít NaN nhất; ERA này là lựa chọn, không phải parity nguyên bản paper/R.
- Trạng thái lượt: `completed` / `partial` / `failed` theo `CovidResult`; NaN còn lại luôn là ô chưa giải được, không có fallback ngầm.
- `space` trong summary ghi rõ hai không gian khác nhau — **không so metric MICoDa với Hron/Tsagris như cùng một bảng**.
- Chi tiết đặc tả: [reports/pipeline_protocol_3_baselines_2026-10-29.md](../reports/pipeline_protocol_3_baselines_2026-10-29.md).

## Đọc `errors.log`

Mỗi dòng có dạng `model | lượt | seed | loại lỗi | thông báo [ngữ cảnh]`, ví dụ:

```
tsagris | native | - | BaselineError | row 13: observed parts sum to 0, JSD is undefined [row=13, reason=zero_anchor, entity=Brill, csv_line=15]
```

- `row` là chỉ số dòng 0-based. `entity` và `csv_line` (dòng 1 là header) chỉ ra dòng đó trong CSV.
- `feature` là tên target, hoặc `Other` cho phần không thuộc target nào.
- Lỗi của một model không dừng các model khác. Exit code là 1 nếu có ít nhất một lỗi.

Xem thêm: [thay đổi Hron/Tsagris và luồng chạy chung](../docs/tv2_hron_tsagris_changes_vi.md), [BASELINES_VI.md](BASELINES_VI.md).
