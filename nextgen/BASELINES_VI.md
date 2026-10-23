# General imputation modules — phiên bản 14/10/2026

Đã hiện thực ba họ thuật toán theo đặc tả confirmed_design_vi.md. Core hoạt
động trên ma trận N×D, độc lập với CSV, số feature17 và các runner legacy.
CountBaselineAdapter là lớp thích nghi CoVariants: counts/N, Other, smoothing,
metadata, observed lock và budget allocation được khai báo riêng.

## Các entry points

| Method ID | Core | Nội dung hiện thực |
|---|---|---|
| hron_knn | HronKNN | Aitchison L2 trên observed subcomposition; complete hoặc available donors; median/sum adjustment; median/mean aggregate |
| hron_ls | HronIterative | Khởi tạo kNN; target-first ILR; OLS; lặp dừng theo covariance |
| hron_lts | HronIterative | Cùng vòng lặp, raw LTS solver: exact h-subsets khi nhỏ, seeded multistart C-steps khi lớn |
| micoda_v1 | MICoDa | Recalibrate partial rows; ALR fit/refit, W và final rescale tùy chọn |
| micoda_v2 | MICoDa | Missing-only paper interpretation, gồm observed-reference case; compatibility mode của code tác giả riêng |
| tsagris_basic | TsagrisKNN | JSD doubled, arithmetic full-donor mean, missing mass allocation |
| tsagris_alpha | TsagrisKNN | Alpha Fréchet mean trên full donor compositions |
| tsagris_global_cv | TsagrisAdaptive | Repeated leave-N-out chọn một cặp alpha/k |
| tsagris_adaptive | TsagrisAdaptive | Chọn cặp alpha/k theo từng missingness pattern |

`fit(values, covariates=None)` không sử dụng query truth. `predict_result`
trả CoreResult(values, diagnostics); `predict` trả ndarray; `impute` cho phép
fit/impute một bảng (transductive paper workflow). Hron/MICoDa iterative
`fit_result_` ghi iteration history và converged; không bảo đảm hội tụ mọi input.

## Môi trường và chạy

Python>=3.11. Baselines chỉ cần NumPy, không cần PyTorch/GPU/SciPy/R.
R chỉ dùng cho kiểm thử đối chiếu ngoài Python. Optional dependencies nằm trong
pyproject.toml; khoảng phiên bản không thay thế lock của environment benchmark.

```powershell
python -m pip install -e './nextgen[baselines,test]'
python -B -m pytest -q -p no:cacheprovider nextgen/tests
```

Lượt này không cài/sửa môi trường người dùng. Kiểm chứng trên CPython3.12.14,
NumPy2.5.3 và pytest9.1.1 đã có trong runtime local. Chưa kiểm wheel build hoặc
mọi phiên bản trong range dependency. Pip install là lệnh cho môi trường được
chọn khi dùng package, không phải thao tác đã được chạy trong audit.

Ví dụ general trên simplex, không có metadata:

```python
import numpy as np
from longitudinal_impute.baselines.hron import HronKNN
from longitudinal_impute.baselines.micoda import MICoDa
from longitudinal_impute.baselines.tsagris import TsagrisKNN

donors = np.random.default_rng(7).dirichlet(np.ones(4)*2, size=30)
query = donors[:2].copy()
query[:, [0, 2]] = np.nan
a = HronKNN(k=3).fit(donors).predict(query)
b = MICoDa(version='v2', reference=3).fit(donors).predict(query)
c = TsagrisKNN(k=3, alpha=.5).fit(donors).predict(query)
```

Hron là reference tỷ lệ tương đối, không tự đóng tổng hay giữ counts. MICoDa
và Tsagris có fixed-total/simplex assumptions; input raw counts phải qua adapter.

Count adapter:

```python
from longitudinal_impute.baselines.adapters import CountBaselineAdapter
from longitudinal_impute.contracts import FitScope

# train/query là ConditioningPanel đã che H và approved metadata; không phải
# RawPanel hoặc ma trận nhãn. IDs chỉ phần được phép fit.
scope = FitScope('inductive', train.row_ids, 'registered_split', train.mask_id)
model = CountBaselineAdapter(HronKNN(3), fallback='error').fit(train, scope)
result = model.predict_result(query)
counts, diagnostics = result.counts, result.diagnostics
```

## CLI CSV hoàn chỉnh

Runner không tự đoán targets hoặc split. Schema mẫu:
`nextgen/configs/baselines_covariants.toml`. Fit IDs JSON là list string IDs;
mỗi ID là JSON encoding `[location,date]` không có spaces, để tránh ambiguity.
Tạo IDs từ `load_csv(...).row_ids` và một split được đăng ký trước.

```powershell
general-impute --data data/original_long_format.csv `
  --schema-config nextgen/configs/baselines_covariants.toml `
  --fit-ids path/to/registered_train_ids.json `
  --method micoda_v2 --parameters '{"W":0.000001,"max_iter":50}' `
  --numeric-metadata median_age --output path/to/NEW_run_directory
```

Hoặc `python -B -m longitudinal_impute.baselines.runner` cùng các flags.
Output: completed.csv, manifest.json, diagnostics.json. Writer dùng chế độ
exclusive, từ chối đường dẫn run đã tồn tại; giữ nguyên text của observed
counts và tất cả metadata/cột không thuộc target. Counts imputed là số thực
kỳ vọng, chưa có integer rounding policy. Output chỉ thuộc các dòng của CSV
đã đưa vào; accuracy native missing không được suy ra từ CSV hoàn chỉnh.

`--fallback train_mean` bật fallback khai báo, không phải mặc định.
`--degenerate-policies` bật global donor mean cho anchor zero và uniform missing
weights khi donor mean thiếu mass. Fit rank/config errors vẫn trả lỗi; fallback
chỉ bắt BaselineError ở query và log lý do từng dòng. Không điền một feature
không có observation nào trong fitting bằng mean tưởng tượng.

Metadata dùng numeric median/scale/missing indicator và categorical one-hot,
train vocabulary, missing/unknown codes. Constant columns bị loại cùng lý do;
unknown category không có training samples không thể có hệ số riêng ước lượng
hợp lệ, nên cột đó có thể bị drop. Core kiểm full rank; không dùng embedding
neural làm covariates cho MICoDa. Metadata unapproved như classification/score/
group không được tự đưa vào. Caller vẫn phải kiểm provenance nguồn được approved.

Kaggle chỉ cần cài package trên và gọi CLI với paths `/kaggle/input/...` và
output mới dưới `/kaggle/working/...`. Core dùng NumPy trên CPU. Không có code
trong notebook cells cần đồng bộ; chưa upload dữ liệu hoặc tạo notebook online.

## Quy tắc counts và thông tin

hide_targets tách labels và conditioning, chỉ nhận approved metadata. Trong
adapter, safe_counts lại dùng C để bỏ giá trị hidden trước mọi phép tính, kể cả
khi caller đưa nhầm giá trị lớn tại ô C=false. FitScope chọn đúng IDs trước
fitting donors/scalers/metadata; mask_id phải khớp. Protocol location disjoint
do split layer/caller sở hữu, không thể suy từ row IDs của riêng fit subset.

Other=(N-sum17)/N chỉ known nếu cả17 parts hiện visible. Khi thiếu/che một raw
part, Other=unknown. Tính phần dư ở count space trước, không lấy 1-sum normalized
values gây phần dư giả ~1e-17. Hron positive replacement chỉ nằm trên encoder
copy, delta fit từ permitted positive proportions. MICoDa W có đơn vị tỷ lệ
trong adapter, không phải counts. Tsagris basic hỗ trợ zero tự nhiên.

Constraint khóa **C khi đánh giá**, **O khi final native imputation**; không
copy H. Có explicit Other proposal thì phân R=N-sum(visible counts) trên phần
unknown+Other theo weights proposal. Không có Other proposal thì giữ missing
proposal nếu feasible, shrink missing nếu vượt R. Đây là adaptation không phải
thuật toán paper. R=0 => missing=0; R<0 => lỗi; không normalize raw observed.
Proposal gốc, repair L1, Other và per-row core/fallback diagnostics được lưu.

Để giữ thông tin scale/coverage khi dùng Hron subcomposition, nên nghiên cứu
thêm c/N/Other và metadata; việc delta/median scale thay đổi mạnh trong bảng
nhiều zeros là nguy cơ thực tế đã đo, không được coi observed-lock là accuracy.

## Chính sách domain và các khác biệt phải biết

1. **Hron:** positive domain; k được cap theo số donors hợp lệ và log effective_k.
   Stable tie-break theo donor order, tolerance1e-12 cho sai số floating point.
   Chỉ tái lập donor1a/2a; không có giao diện cho mọi biến thể1b/2b trong paper.
   Eq8 có D-1 balances nhưng Step8 nói D targets: dùng target-first pivot cho
   mỗi feature. Inverse trả đại diện simplex và có thể đổi observed scale.
   Single-query inductive prediction dùng update-norm stop vì covariance của
   một hàng không xác định; transductive fit vẫn dùng covariance stop.
2. **LTS:** default h=floor((n+p+1)/2), p gồm intercept. Exact enumeration khi
   C(n,h)<=5000; nếu có rank-deficient subsets bỏ qua thì không certify global
   optimum. Các bài lớn dùng seeded64 starts/C-steps và báo approximate. Đây là
   **raw LTS**, chưa tương đương robustbase::ltsReg reweighting/corrections dùng
   trong một số software của paper. Không tuyên bố tái lập toàn bảng LTS.
3. **MICoDa paper:** complete-case initial fit, n_complete>parameter_count và
   full rank; least-missing reference, stable first-column tie. V1 recalibrates,
   V2 xử lý reference observed bằng chỉ phân residual cho missing. W được thêm
   cho encoder data; final rescale nhân C/(C+DW), không subtract W. W>0 có thể
   đổi observed reference output; adapter khóa counts gốc sau đó.
4. **MICoDa author_code:** pairwise initial regression và reference luôn bị
   phân bổ lại. Dùng reference last, W=0, fixed_iterations=True để đối chiếu
   trực tiếp source tác giả. Rank errors/negative residual trong Python trả lỗi
   thay vì âm thầm pseudoinverse/clamp. Khác biệt này nằm ngoài fixture hợp lệ.
5. **Tsagris:** strict full simplex donors, incomplete query observed mass<=1.
   Full donor mean trước subset-normalization; α∈[-1,1], α<0 chỉ miền dương.
   α=0 dùng geometric mean với zeros nếu có ít nhất một part dương ở mọi
   donor; không có common positive support thì closure lỗi.
   α gần0 trên positive domain dùng geometric limit khi |α|<1e-8.
   Default degenerate policy là error, không thêm epsilon tự động. JSD dùng
   natural logs và gấp hai usual JSD theo Eq5.
6. **Tsagris CV:** repeated leave-N-out loại validation rows khỏi donor pool.
   N mặc định bằng số incomplete fit rows ở global mode, và frequency của
   pattern ở adaptive mode. Nếu N>=complete donors thì lỗi, cần chọn leave_n
   explicit và report adaptation. Unseen query pattern có thể tune từ fit
   donors, không từ query values; nếu không có explicit leave_n thì dùng tổng
   training pattern count, có log. Candidate alpha/k failures được ghi, không
   bỏ khỏi score mà vẫn gọi successful. Không dùng test truth chọn alpha/k.

Không có claim “proper multiple imputation” hoặc posterior uncertainty. Một
output dataset là point estimates. Full benchmark, temporal/correlation metrics,
neural architecture và outer loop thuộc các cổng nghiên cứu tiếp theo.

## Source/equation/parity mapping

| Nguồn | Mục | Module/check |
|---|---|---|
| Hron.pdf PDF6–8 §3.1–3.2 Eq5–13 | donor, adjustment, median, balances, LS/LTS, stop | hron.py, regression.py; fixtures base-R, observed ratios/isometry/LTS tests |
| Sasha Micoda.pdf PDF165–170 (chương8) | W, ALR, Lemma1, Algorithms1/2 | micoda.py; reference observed/missing/allmissing, complete/pairwise support tests |
| MICoDa_Aux.R commit786e4902cd0c9cccef9436ccf7fe8f399be737fb | authors' V1/V2 executable behavior | reference_oracles.R gọi source gốc; sai số trên fixture dưới1e-9 |
| tsagris.pdf PDF5,7–10 Eq5–7 §2.3–2.5 | doubled JSD, example, alpha mean, CV/adaptive | tsagris.py; exact toy0.2666667/0.1333333, endpoints, CV domain/reproducibility tests |

Author source: [pinned MICoDa R](https://github.com/sahaAbh/MICoDa/blob/786e4902cd0c9cccef9436ccf7fe8f399be737fb/R_Aux_scripts/MICoDa_Aux.R).
PDF paths/hash và source snapshot nằm trong harness sources manifest.

Expenditures Table1 chưa tái lập toàn bộ: k=4 AitchisonL2 + ratio-of-medians
trên CSV fixture hiện có cho144.82667, khác152.1 của bảng. k=3 cho152.1; legacy
R default dùng CLR L1 thay L2, còn das=TRUE dùng Aitchison formula. LS scaled
cho150.73026 so với150.8 trong bảng. Các khác biệt này chưa được giải quyết
bằng thay thuật toán để khớp số; fixture equation/base-R parity được báo riêng
với table reproduction. Đây là giới hạn scientific parity cần theo dõi.

## Kiểm chứng thực hiện và cách tái chạy

Hồ sơ: `.harness/general-imputation-2026-10-14/`.
Tests kiểm công thức, masks/poisoning ở fit và predict, exact observed zero,
budget edge cases, metadata/rank, donor policy, alpha/CV và CSV text preservation.
Independent R suite đối chiếu9 trường hợp: MICoDaV1/V2 author source, HronkNN/LS
equation oracle, raw LTS toy, Tsagrisα=-.5/0/.5/1. Sai số tối đa~6.2e-16 trên
fixtures đã chạy, không suy rộng sang mọi dataset.

```powershell
python -B nextgen/tests/verify_reference_oracles.py `
  --workdir path/to/NEW_oracle_directory `
  --author-source path/to/pinned_MICoDa_Aux.R `
  --rscript 'C:/Program Files/R/R-4.5.1/bin/Rscript.exe'

python -B -m longitudinal_impute.baselines.smoke `
  --data data/original_long_format.csv `
  --schema-config nextgen/configs/baselines_covariants.toml `
  --output path/to/NEW_smoke_directory
```

Real smoke đã dùng200 fit rows (Canada/Denmark),48 query rows ở location tách
riêng; 24 labelled partial rows+24 native-incomplete rows. Ba primary methods
thỏa finite/nonnegative/observed-lock/budget. Mask chỉ phủ sampled panel, chưa
whole-feature full histories; MAE của smoke không dùng chọn winner. CLI còn
chạy trên CSV248rows, giữ7244 nonempty source cells nguyên text và từ chối
overwrite. Chưa hoàn thành benchmark toàn bộ11.671rows/location folds/seeds.

Không xóa, chuyển hoặc sửa các source/runners legacy. General baseline modules
đã chạy; orchestration neural `pipeline.run()` vẫn có guard chưa triển khai.
