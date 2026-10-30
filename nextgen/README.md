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
| `mask_<kind>_seed<s>.npy` | Mask H (bool, N × D) của từng seed. |
| `summary.csv` | Một dòng cho mỗi model × lượt (`native`/`masked`) × seed. Gồm `status` (ok/failed), `seconds`, số ô che, số dòng đầy đủ còn lại sau mask, MSE/MAE trên count và tỷ lệ (cột hậu tố `_truth_zero`, `_truth_positive`, `_forced_zero`, `_model`, `_model_truth_zero` là các nhóm con). |
| `summary.json` | Toàn bộ chi tiết: Wasserstein theo target của lượt thật, `info_` của model, thống kê mask, sai số theo nhóm, `aggregate` (mean ± sd qua seed). |
| `errors.log` | Chỉ xuất hiện khi có lỗi. |

Ô "bị ép về 0" nằm ở dòng mà tổng các ô nhìn thấy đã bằng N. Phần dư của dòng đó là 0, nên model không quyết định được ô này. Định nghĩa metric nằm trong [`evaluate.py`](src/longitudinal_impute/evaluate.py).

## Đọc `errors.log`

Mỗi dòng có dạng `model | lượt | seed | loại lỗi | thông báo [ngữ cảnh]`, ví dụ:

```
tsagris | native | - | BaselineError | row 13: observed parts sum to 0, JSD is undefined [row=13, reason=zero_anchor, entity=Brill, csv_line=15]
```

- `row` là chỉ số dòng 0-based. `entity` và `csv_line` (dòng 1 là header) chỉ ra dòng đó trong CSV.
- `feature` là tên target, hoặc `Other` cho phần không thuộc target nào.
- Lỗi của một model không dừng các model khác. Exit code là 1 nếu có ít nhất một lỗi.

Xem thêm: [thay đổi Hron/Tsagris và luồng chạy chung](../docs/tv2_hron_tsagris_changes_vi.md), [BASELINES_VI.md](BASELINES_VI.md).
