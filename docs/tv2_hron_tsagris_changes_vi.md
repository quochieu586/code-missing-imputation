# Thay đổi Hron / Tsagris (TV2) – ghi chú cho TV1

Branch `feat/tv2-hron-tsagris`. Commit liên quan: `6015a86` (pipeline), `0557a2c` (Hron), `095376c` (Tsagris). Số liệu audit lấy từ `tv2_phase1/REPORT_vi.md`.

## 1. Interface mới và ảnh hưởng tới MICoDa

```python
class HronKNN / TsagrisKNN:
    input_domain = 'positive' | 'nonnegative'
    def __init__(self, ...)            # chỉ lưu và kiểm tra tham số
    def fit(self, values) -> self      # values: N x (17 + Other), tỷ lệ, NaN = thiếu
    def predict(self, values) -> ndarray   # cùng shape, ô observed giữ nguyên tuyệt đối
    info_: dict                        # có sau predict: số ô điền, fallback, ties, tham số đã dùng
```

| Mục | Trước | Sau |
|---|---|---|
| `pipeline.py:13` | `prediction = model.predict_result(values)` rồi `prediction.values * …` | `return restore_counts(raw, model.predict(values) * raw.totals[:, None], hidden)`: một câu lệnh, thay cho 2 dòng cũ |
| Lỗi | 8 lớp error | chỉ dùng `BaselineError(message, context={row, feature, reason})` và `NotFittedError` |
| `fit` | copy-then-swap, `CoreResult`, diagnostics | gán trực tiếp, không có lớp trung gian |
| Import | `common`, `regression`, `registry` | chỉ `numpy` và `..errors` |

**MICoDa:** không sửa `micoda.py`, `common.py`, `regression.py`, `diagnostics.py`, `errors.py`, `data.py`, `evaluation.py` (đã kiểm tra bằng `git diff`). `MICoDa.predict()` vốn trả `predict_result().values`, nên kết quả qua runner không đổi. Test MICoDa giữ nguyên và pass. Trên COVID, MICoDa vẫn lỗi như trước: `MICoDa ALR needs positive parts: declare W > 0 for zeros`. `diagnostics.py` được giữ lại vì `common.CoreResult` (MICoDa đang dùng) còn import nó.

## 2. Quyết định phương pháp và điểm khác paper

| | Quyết định | Lý do / số liệu |
|---|---|---|
| Hron: phạm vi | chỉ kNN chiến lược (2a), Hron et al. 2010 §3.1 | — |
| Hron: khoảng cách, hệ số, gộp | Aitchison (clr + Euclid) trên O_i; Eq.7 median/median (mặc định), Eq.5 là tùy chọn; gộp bằng median Eq.6 | đúng paper |
| Hron: k | 3, không tune | ví dụ expenditures: k=3 → 152.10, k=4 → 144.83 (Bảng 1 báo 152.1 với k=4) |
| Hron: số 0 | giữ cách thay min dương / 2 = 2.30e-06 trong `data.model_input`; `fit` báo lỗi nếu nhận giá trị ≤ 0 | paper giả định dữ liệu dương |
| Hron: ties | `ties='group'` (mặc định): lấy toàn bộ donor hòa với donor thứ k (tolerance 1e-12 × max(1, d_k)); `'index'` giữ đúng hành vi cũ | 57.8 % ô mẫu (audit) và **58 763 / 97 205** ô trên COVID bị hòa ở vị trí k; bản cũ chọn donor theo thứ tự dòng, tức là theo thứ tự alphabet của location |
| Tsagris: thuật toán | 6 bước §2.3.1; Fréchet mean Eq.6 trên toàn bộ vector donor; JSD Eq.5 (×2); phá hòa theo chỉ số | ví dụ paper ra (0.2, 0.27, 0.3, 0.1, 0.13) |
| Tsagris: T = 0 | \|T\| ≤ 1e-12 → ô thiếu = 0; T < −1e-12 → lỗi | 7 595 dòng |
| Tsagris: tổng observed = 0 | `zero_anchor='global'` (mặc định): dùng Fréchet mean của mọi donor | **1 274** dòng (ngoài paper) |
| Tsagris: donor mean = 0 trên các cột thiếu | `zero_missing_mass='uniform'` (mặc định): chia đều T | **77** dòng (ngoài paper) |
| Tsagris: tune | `tune=False` mặc định (k=3, α=1). Khi bật: một cặp (k, α) toàn cục (§2.4.1), lưới k ∈ {1,3,5,10} × α ∈ {.25,.5,.75,1}, `leave_n` cố định = 20 % số donor, 5 lần lặp, che holdout theo pattern lấy ngẫu nhiên từ các dòng thiếu thật, chấm điểm bằng JSD trung bình | quy tắc của paper N = số dòng thiếu = 11 155 ≥ 516 donor nên không chạy được |
| Tsagris: adaptive (§2.5) | bỏ | xem mục 3 |

Hron không đóng (close) output: ô observed giữ nguyên, ô điền nằm trên cùng thang với dòng đó (paper §3). `restore_counts` chỉ dùng các giá trị này làm trọng số. Output của Tsagris thì có tổng mỗi dòng = 1.

## 3. Đã xóa

| Thành phần | Lý do |
|---|---|
| `HronIterative` (LS/LTS) | LS chậm trên toàn bảng; LTS hiện tại là raw, chưa reweight, nên lệch paper (expenditures: 161.49 so với 150.8). Vẫn còn trong git history (`a235e9f`) |
| `TsagrisAdaptive` | quy tắc N của paper không chạy được (11 155 ≥ 516); bản adaptive mặc định lỗi ở các dòng tổng = 0; bản có fallback mất 176 s để fit |
| `baselines/registry.py`, `with_degenerate_policies` | không ai gọi |
| `HronKNN`: `donor_policy`, `aggregation='mean'`, `predict_result`, `impute`, LRU cache | ngoài phạm vi (2a) / Eq.6; cache được thay bằng cách gom theo pattern O_i (clr của donor tính một lần cho mỗi pattern) |
| `tests/test_fit_lifecycle.py`, các test HronIterative / TsagrisAdaptive | chỉ phục vụ những phần đã bỏ |

Test mới nằm ở `nextgen/tests/test_hron.py` và `test_tsagris.py`. Fixture `nextgen/tests/fixtures/expenditures.csv` được copy từ `tests/fixtures/`. `pytest nextgen/tests -q`: **36 passed**. `verify_reference_oracles.py` (cần R) đã sửa import, nhưng chưa chạy được vì máy không có R.

## 4. Kiểm chứng trên COVID

Mask: `--mask-rate 0.05 --seed 7` → 5 518 ô (4 534 ô có giá trị thật = 0, 984 ô dương).

| Model | Thời gian native (runner) | MSE count | MSE tỷ lệ | `info_` (native) |
|---|---|---|---|---|
| Hron cũ (phase 1) | 40.9 s | 23 788.8 | 0.0478 | — |
| Hron `ties='index'` | 7.4 s | 23 788.8 | 0.0478 | output **trùng tuyệt đối** `tv2_phase1/run_native/hron.csv` (0 ô khác) |
| **Hron `ties='group'` (mặc định)** | 8.5 s | **9 555.8** | **0.0302** | 58 763 ô hòa; trung bình 185.8 donor/ô (tối đa 11 643); 4 397 ô output khác bản cũ, nằm trong 392 dòng |
| **Tsagris mặc định** (k=3, α=1) | 0.96 s | 1 163 462.5 | **0.0211** | 1 274 dòng global, 77 dòng uniform, 7 595 dòng T=0, 252 dòng hòa |
| Tsagris `tune=True` | fit 1.0 s | 1 169 696.2 | 0.0210 | native chọn (k=5, α=0.5); khi có mask chọn (k=10, α=0.5); leave_n = 103 |

- MICoDa: vẫn lỗi như trước.
- Khi mask 5 % ô, số dòng đầy đủ (donor của Tsagris) giảm từ 516 xuống **212**, nên kết quả MSE của Tsagris phụ thuộc mạnh vào cách mask.
- Xếp hạng Hron/Tsagris đảo chiều giữa MSE count và MSE tỷ lệ (đã nêu trong phase 1, mục D).

Lệnh chạy lại (từ gốc repo, `PY=tv2_phase1/.venv/Scripts/python.exe`):

```
PY -m pytest nextgen/tests -q
PY -m longitudinal_impute.cli --data data/covariants.csv --schema-config nextgen/configs/baselines_covariants.toml --output tv2_phase2/run_native
PY -m longitudinal_impute.cli --data data/covariants.csv --schema-config nextgen/configs/baselines_covariants.toml --mask-rate 0.05 --seed 7 --output tv2_phase2/run_mse
PY tv2_phase2/verify.py      # so sánh ties, MSE count/tỷ lệ, tune; ghi tv2_phase2/verify.json
```

## 5. Đề xuất cho TV1 (TV2 không làm)

- Hron và Tsagris không còn dùng `common.py`, `regression.py`, `diagnostics.py`. Hiện chỉ còn MICoDa dùng `CoreResult` (kéo theo `diagnostics.py`), `matrix`, `stable_softmax`, `design_matrix` và `ols`. Các hàm `simplex`, `closure`, `clr`, `pivot_basis`, `cov_matrix`, `distance_order` và `regression.lts` không còn được src nào dùng (chỉ còn `test_general_baselines.py` test `pivot_basis`/`lts`). Nếu MICoDa chuyển sang `fit/predict` gọn như trên, có thể gộp phần cần dùng vào `micoda.py` rồi xóa ba file này.
- `errors.py`: Hron/Tsagris chỉ cần `BaselineError` và `NotFittedError`.
- Các quyết định A–D trong `tv2_phase1/REPORT_vi.md` §5 (metric, kiểu mask, schema microbiome) vẫn cần chốt chung.
