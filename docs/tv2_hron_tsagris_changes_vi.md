# Thay đổi Hron / Tsagris (TV2) – ghi chú cho TV1

Branch `feat/tv2-hron-tsagris`. Commit liên quan: `6015a86` (pipeline), `0557a2c` (Hron), `095376c` (Tsagris); luồng chạy chung (mục 6): `a9fff5d` (data), `394d5e7` (evaluate), `aa87901` (run). Số liệu audit lấy từ `tv2_phase1/REPORT_vi.md`.

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

Lệnh chạy lại ở thời điểm phase 2 (từ gốc repo, `PY=tv2_phase1/.venv/Scripts/python.exe`). `cli.py` đã bị thay bằng `run.py` ở `aa87901`; lệnh tương đương ở mục 6.

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

## 6. Luồng chạy chung (`run.py`)

Commit `a9fff5d` (data), `394d5e7` (evaluate), `aa87901` (run). Ba file `pipeline.py`, `cli.py`, `evaluation.py` đã xóa. `micoda.py`, `common.py`, `regression.py`, `diagnostics.py`, `errors.py` và thuật toán trong `hron.py`/`tsagris.py` không đổi. `pytest nextgen/tests -q`: **57 passed**.

### 6.1 Interface

```
python -m longitudinal_impute.run --data <csv> --schema <toml> --models <toml> --output <dir>
       [--mask none|random_cell|whole_feature] [--mask-rate 0.05] [--seeds 0 1 2 3 4]
```

| Phần | Nội dung |
|---|---|
| Schema `[dataset]` | `entity`, `targets` bắt buộc; `time`, `total` tùy chọn; khóa khác bị từ chối. Không có `total` → total = tổng target, ô thiếu là lỗi (`CSV line …, entity …`). Có `time` → (entity, time) phải duy nhất; không còn bắt buộc định dạng ngày ISO (microbiome có thể dùng số lần lấy mẫu). |
| `--models` | Mỗi bảng `[hron]`/`[tsagris]`/`[micoda]` là tham số constructor; model không có trong file thì không chạy. Tham số sai bị báo trước khi chạy. Mặc định: `nextgen/configs/models_default.toml`. |
| `main()` | (1) load + `config.json` → (2) lượt điền thật: `<model>_completed.csv`, Wasserstein, `info_` → (3) mỗi seed một mask dùng chung, mỗi model một instance mới → `masked_errors` → (4) `summary.csv`, `summary.json`, bảng in màn hình. |
| Lỗi | `ValueError` (gồm `BaselineError`) của một model được ghi vào `errors.log` theo dạng `model \| lượt \| seed \| loại \| thông báo [row, entity, csv_line, feature]`, rồi chạy tiếp; exit code 1. Lỗi khác (bug) dừng chương trình. |

### 6.2 Mask (`data.make_mask(data, kind, rate, seed) -> (H, stats)`)

- `random_cell`: giữ nguyên thuật toán và chuỗi số ngẫu nhiên của `evaluation_mask` cũ. Seed 7 cho mask trùng `tv2_phase2/run_mse/evaluation_mask.npy`.
- `whole_feature`: duyệt các cặp (entity, target) có ít nhất một ô observed theo thứ tự ngẫu nhiên. Mỗi cặp được chọn thì che trọn chuỗi observed của nó, và dừng khi số ô che ≥ rate × số ô observed (vượt tối đa một chuỗi). Cặp nào làm một dòng mất ô target nhìn thấy cuối cùng thì bị bỏ qua và đếm vào `pairs_skipped`.
- `stats`: `hidden_cells`, `actual_rate`, `rows_affected`, `complete_rows_before/after` (= số donor của Tsagris), số cặp đã che / bị bỏ.

### 6.3 Metric (`evaluate.py`, TV1 nên xem lại)

- Ô **bị ép về 0**: ô thiếu (hoặc bị che) nằm ở dòng mà tổng các ô nhìn thấy = N đúng tuyệt đối. `restore_counts` gán 0 cho ô này bất kể model đề xuất gì. Mọi ô thiếu khác là ô **do model quyết định**. Ô bị ép luôn có giá trị thật = 0.
- `wasserstein_scores` (lượt thật, không có nhãn): W1 theo từng target, trên count và trên c/N. So observed với (a) cột đã điền đầy đủ, (b) chỉ các ô đã điền, (c) chỉ các ô do model quyết định. Có kèm số ô của mỗi nhóm.
- `masked_errors` (chỉ trên H): MSE và MAE trên count và trên c/N. Tính cho toàn bộ H, và cho các nhóm `truth_zero`, `truth_positive`, `forced_zero`, `model`, `model_truth_zero` (nhóm cuối do TV2 thêm), kèm số ô.
- `summarize`: mean và sd (ddof = 1) qua các seed chạy thành công, cho từng model × lượt × mask.

### 6.4 Bước 0: MSE count của Tsagris

Script: `tv2_phase3/diagnose_tsagris_mse.py`, mask `random_cell` 0.05 seed 7 (5 518 ô). **Không thấy bug.**

| | Hron | Tsagris |
|---|---|---|
| MSE count / MSE tỷ lệ | 9 555.8 / 0.0302 | 1 163 462.5 / 0.0211 |
| Tỷ lệ SSE của top 1 % (55 ô) / top 20 ô | 0.979 / 0.870 | 1.000 / 0.998 |
| Bin N chiếm nhiều SSE nhất | [1 000, 10 000): 71.4 % | [10 000, 100 000): 99.2 % (166 ô) |
| MSE tỷ lệ trong bin [10 000, 100 000) | 2.8e-05 | 0.0063 |
| MSE count khi bỏ ô United Kingdom 2021-09-27 Delta | 9 557.5 | 13 028.3 |

- Một ô duy nhất chiếm **98.9 % SSE** của Tsagris: United Kingdom, 2021-09-27, Delta, N = 79 723, giá trị thật 79 683, giá trị điền 7.98. Ô Kappa cùng dòng cũng bị che. Phần nhìn thấy của dòng chỉ có 21 trình tự. Ba donor gần nhất theo JSD là UK 2020-11-09, Denmark 2021-01-18 và UK 2020-10-12, đều trước thời kỳ Delta. Trung bình của chúng ở cột Delta ≈ 0.0001 và ở Other ≈ 0.18, nên Tsagris đề xuất 0.9996 phần dư cho Other.
- Kiểm tra chống bug:
  - Tổng mỗi dòng Tsagris lệch tối đa 2.2e-16 so với 1.
  - Tổng count đã điền ≤ N.
  - Tổng khối lượng ô che ≤ residual, ở cả hai model.
  - Các dòng fallback `global`/`uniform` chỉ chiếm 0.22 % SSE của Tsagris.
- Kết luận: đây là đặc tính của phương pháp, không phải bug. JSD so subcomposition đã chuẩn hóa, nên bỏ qua việc phần observed rất nhỏ so với N. Thêm vào đó, MSE count tăng theo N², nên vài dòng N lớn chi phối hoàn toàn.

### 6.5 Bước 6: COVID (quan sát, chưa phải kết luận xếp hạng)

Lệnh (`PY=tv2_phase1/.venv/Scripts/python.exe`, `M=nextgen/configs/models_default.toml`, `S=nextgen/configs/baselines_covariants.toml`):

```
PY -m longitudinal_impute.run --data data/covariants.csv --schema S --models M --output tv2_phase3/run_none --mask none
PY -m longitudinal_impute.run ... --output tv2_phase3/run_whole_feature --mask whole_feature --mask-rate 0.05 --seeds 0 1 2 3 4
PY -m longitudinal_impute.run ... --output tv2_phase3/run_random_cell --mask random_cell --mask-rate 0.05 --seeds 0 1 2 3 4
PY tv2_phase3/report_covid.py
```

- Lượt `--mask none`: `hron_completed.csv` và `tsagris_completed.csv` **trùng từng byte** với `tv2_phase2/run_native/hron.csv` và `tsagris.csv`. Thời gian: Hron 9.55 s, Tsagris 1.20 s.
- MICoDa (mặc định `W = 0`) lỗi ở mọi lượt với thông báo `MICoDa ALR needs positive parts: declare W > 0 for zeros`, nên exit code là 1.

Mean ± sd qua 5 seed:

| Mask | Model | MSE tỷ lệ | MSE count | MAE tỷ lệ | MSE tỷ lệ: thật = 0 / > 0 | MSE tỷ lệ: do model / bị ép | Thời gian/seed | Donor Tsagris |
|---|---|---|---|---|---|---|---|---|
| whole_feature | Hron | 0.0299 ± 0.0054 | 2 457 670 ± 5.5e6 | 0.0396 ± 0.0065 | 0.0077 / 0.150 | 0.0660 / 0 | 13.2 s | — |
| whole_feature | Tsagris | 0.0236 ± 0.0053 | 45 015 ± 4.9e4 | 0.0489 ± 0.012 | 0.0119 / 0.0866 | 0.0521 / 0 | 0.86 s | 109, 80, 298, 109, 189 |
| random_cell | Hron | 0.0306 ± 0.0015 | 17 927 ± 8.7e3 | 0.0445 ± 0.0017 | 0.0085 / 0.133 | 0.0613 / 0 | 12.3 s | — |
| random_cell | Tsagris | 0.0241 ± 0.0008 | 24 630 ± 1.1e4 | 0.0498 ± 0.0014 | 0.0155 / 0.0639 | 0.0482 / 0 | 0.93 s | 217, 209, 207, 222, 208 |

Số ô che trung bình mỗi seed:

| Mask | H | thật = 0 / > 0 | bị ép / do model | Rate thực tế |
|---|---|---|---|---|
| whole_feature | 5 661.8 | 4 778.2 / 883.6 | 3 110.6 / 2 551.2 | 0.0501–0.0508; 64–68 cặp, 0 cặp bị bỏ |
| random_cell | 5 587.2 | 4 599.0 / 988.2 | 2 800.2 / 2 787.0 | 0.0493–0.0505 |

Quan sát:

- **Ở seed 0 của `whole_feature`, MSE count của Hron là 12 211 226**, trong khi các seed khác chỉ 9 204–32 145. Chuỗi (United States, Delta) bị che trọn. 5 dòng có N > 100 000 chiếm 97 % SSE, Hron điền ≈ 0.3 thay cho ≈ 105 000–122 000 (`tv2_phase3/hron_whole_feature_seed0.out.txt`). Cơ chế giống Bước 0, chỉ là xảy ra ở model kia.
- **Xếp hạng đổi theo metric.** Tsagris có MSE tỷ lệ thấp hơn, còn Hron có MAE tỷ lệ thấp hơn, ở cả hai kiểu mask. Hron tốt hơn trên ô có giá trị thật = 0; Tsagris tốt hơn trên ô có giá trị thật > 0.
- **Khoảng một nửa H là ô bị ép về 0** (sai số 0 với mọi model), nên metric trên toàn bộ H bị pha loãng. Nên đọc thêm nhóm `_model`.
- **Số donor của Tsagris dao động mạnh với `whole_feature`** (80–298 trên 516), ít hơn hẳn với `random_cell` (207–222).

### 6.6 Điểm TV1 cần xem

- Chốt metric chính: tỷ lệ hay count, MSE hay MAE, trên toàn bộ H hay chỉ nhóm do model quyết định. Định nghĩa nằm hết trong `evaluate.py`.
- Chốt cách chọn rate và tiêu chí dừng của `whole_feature`.
- MICoDa cần một `W > 0` trong `models_default.toml` (bảng `[micoda]` hiện để trống) để chạy được trên COVID. TV2 không chọn giá trị này.
- Dataset microbiome: dùng `configs/schema_microbiome_example.toml` làm mẫu. Nếu không có cột total thì dataset không được có ô thiếu thật, nên chỉ chạy được lượt MSE.
