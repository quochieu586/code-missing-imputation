# Audit main model — 13/11/2026

## Kết luận trước

**Chưa đáp ứng trọn bộ tiêu chí “không bug/không lỗi logic”.** Main bám nhánh
direct-composition của proposal và plan MVP đã duyệt, kiến trúc còn gọn, các
metric đã báo được tính đúng. Tuy nhiên có **2 lỗi P2 cần xử lý trước khi mở rộng
benchmark**, thêm **2 lỗi biên số học kế thừa P3** và **1 rủi ro API/P3 có điều kiện**
đã tái hiện. Không có căn
cứ để viết lại hệ thống, thêm framework hoặc đổi sang diffusion.

| Tiêu chí người dùng hỏi | Kết luận audit |
|---|---|
| Bám proposal | **Đạt trong scope MVP đã duyệt**; không phải sao chép nguyên hình mạng hoặc hiện thực probabilistic MI/diffusion. |
| Không phức tạp không cần thiết | **Đạt trong phần đã kiểm.** Hai owner main/network, một block, một loss, reuse data/runner. Hai view là hàng rào leakage cần thiết. |
| Không bug | **Chưa đạt**: RNG coupling và false budget failure; thêm ba lỗi mức thấp được liệt kê riêng. Test xanh không chứng minh không bug. |
| Không lỗi logic | **Chưa đạt toàn bộ**: Q phụ thuộc evaluation RNG; API `initial` có thể nhận sai case. Không thấy lỗi tương ứng trên fresh/full hoặc CLI refine đã validate. |
| Không sai ý nghĩa thống kê | **Đạt ở mức mô tả thận trọng, chưa đủ benchmark kết luận.** Cần công bố lệch rate Q, chưa hội tụ, single-seed, forced-zero và làm rõ weighting train. |

**Audit chỉ báo cáo, chưa sửa runtime/tests/config/data và không chạy lại fit trên
COVID thật.** Kết quả chạy trước được giữ nguyên; đây không phải biên bản sửa lỗi.

## 1. Phạm vi, revision và hai reviewer

- Source: working tree thực tế, HEAD `59a7777f3bb72e1f0439eaf18f192a74232656b2`,
  **DIRTY**. Current runtime phần lớn chưa tracked; không dùng HEAD thay source.
  [Snapshot 48 file](../.harness/main-model/audit-20261113/snapshot.json).
- Đúng **2 native subagent**, model `openai/gpt-6.1-sol#xhigh` theo yêu cầu
  Sol reasoning extra-high. Model catalog công bố variant `xhigh`; không tự thêm
  provider hoặc chọn model khác.
- Reviewer 1 — proposal/thống kê/thuật toán:
  `ses_e2d31b7b7ffeGPs4xpYwvOia2y`, [báo cáo độc lập](../.harness/main-model/audit-20261113/reviewer1/REPORT_VI.md).
- Reviewer 2 — correctness/numerics/lifecycle/test/độ gọn:
  `ses_e2d31b7a6ffeeqsKqN5Nnbzzzb`, [báo cáo độc lập](../.harness/main-model/audit-20261113/reviewer2/REPORT_VI.md).
- Hai initial reports lập độc lập trên cùng snapshot, chưa xem peer findings.
  Parent tự chạy lại cả 5 counterexamples; sau đó mới mở đối chiếu giữa reviewer.
  Không lấy sự đồng thuận làm bằng chứng thay cho reproduction.
- HarnessFlow selection không trả body workflow thực thi được mà báo instructions
  cũ retired. Đây là audit native theo yêu cầu rõ của user về hai reviewer,
  không tuyên bố hook/checker/orchestrator HarnessFlow đã được enforce.

### Độ sâu coverage

| Flow | Đã kiểm | Giới hạn |
|---|---|---|
| Proposal → approved choices → implementation | Reviewer 1 và parent đọc PDF 11 trang, đặc biệt p4–8; current implementation plan/Other contract | Không fact-check toàn bộ paper được proposal trích dẫn. |
| CSV/known total/H/Q → Tsagris/CLR | Trace đầy đủ main/data/Tsagris, H perturbation và RNG/numerical probes | Hron/MICoDa core không được audit sâu toàn bộ như main. |
| Time/pack/padding → attention → loss → restore | Hai reviewer trace; tests gradients/cross-time/cross-feature; D=2/5/17 fixture | Không benchmark GPU/taxa hàng nghìn hoặc mọi dynamic range. |
| Outer loop/fresh state → saved init/refine/evaluate | Suite và fixture round-trip, không torch ở init/evaluate | API `initial` truyền trong RAM có precondition chưa enforce. |
| Scoring/paired → real-data report | Tái tính 8 masked arrays, provenance/hash, chẩn đoán history/Q | Không retrain COVID, không infer uncertainty/CI hoặc NaN-truth accuracy. |

## 2. Findings xác nhận — theo ưu tiên

### A01 — P2 — Trùng RNG H/Q làm sai rate nhãn học, có thể khiến Q luôn rỗng

**Source:** [`main_model.py:136–137`](../src/longitudinal_impute/main_model.py#L136),
`data.py:123–129`, `run.py:481–482`, `main_model.py:376`.
Nguồn reviewer: R1-001.

- H random-cell và Q dùng lại `default_rng(seed)` trên cùng shape. Mặc định
  evaluation seed=0 và model seed=0 nên cùng các uniform draws. Sau khi H bị
  xóa, Q chỉ chọn các draw còn nằm giữa rate H và rate Q.
- Ngoài hiệu ứng giữ anchor, rate Q conditional trên V xấp xỉ
  `(rQ-rH)/(1-rH)`, không phải rQ. Với `rH>=rQ`, Q rỗng tất định;
  anchor repair không khắc phục điều này. Đây không phải “dữ liệu nhỏ ngẫu nhiên
  hết labels” và không phải H/Q overlap hay label leakage.
- Parent fixture 80×4, đủ donor: Hseed=0, rate H=.1 hoặc .2, Qrate=.1 →
  `InsufficientDataError: nonempty training Q required`. Chỉ đổi Hseed=1 ở
  rate .1 → Q=30, prepare thành công.
- **Đã ảnh hưởng recipe chạy thật:** whole-feature Q=10.0916% V,
  random-cell Q=**5.4159% V**, dù cùng `train_mask_rate=.1`. Parent đối chiếu
  số Q với `info.train_cells` của run đã lưu: khớp chính xác, không retrain.
- H errors đã lưu vẫn tái tính đúng; không được suy ra CSV prediction bị sai
  hoặc đã leakage. Nhưng supervision giữa mask kinds/seeds không theo recipe
  dự kiến; chưa nên mở benchmark nhiều rate/seed trên bản này.
- **Sửa nhỏ nhất đề nghị:** namespace/stream Q riêng và reproducible; dùng chung
  recipe đó cho loader kiểm Q tại `initial_from_arrays`. Không redraw/fallback.
  Thêm test trùng seed ở rate H nhỏ/bằng/lớn hơn Q, ghi actual Q support/rate;
  version/ghi provenance của recipe mới, không thay lịch sử run cũ.

### A02 — P2 — Tensor budget check từ chối input float64 hợp lệ vì roundoff

**Source:** [`main_model.py:177–183`](../src/longitudinal_impute/main_model.py#L177),
consumer `:244–248`; core tolerance ở `data.py:204–217`.
Nguồn reviewer: R2-01.

- Dòng complete `[57.4,16.3,8.8,76.4,79.8]`, total=238.7, được loader và
  NumPy initializer chấp nhận. Torch float64 sum=238.70000000000002.
- Residual `-2.842170943040401e-14` bị check `<0` coi là budget vượt thật.
  Training xét cả các dòng không có Q nên một dòng complete như vậy cũng
  làm main lỗi khi có labels/missing ở dòng khác.
- Parent dựng CSV mới và chạy CLI: **main failed, Tsagris ok**, exit=1.
  Không phải donor failure/hội tụ, và không phải đầu vào thực sự vượt total.
- Chưa thấy ảnh hưởng bảng COVID count nguyên hiện tại; domain loader cho phép
  float64 nonnegative nên đây vẫn là bug reachable, không phải input ngoài contract.
- **Sửa nhỏ nhất đề nghị:** tensor allocation dùng cùng row-scaled tolerance
  với core, reject phần vượt tolerance rồi clamp residual âm trong tolerance về
  0; không đổi anchors, không bỏ validator. Test decimal parity và true-overbudget.

### A03 — P3 có điều kiện / API hardening — `initial` không kiểm đúng H/case

**Source:** [`main_model.py:399–401`](../src/longitudinal_impute/main_model.py#L399),
`:304,314–320,418`; countermeasure CLI `run.py:352–394`.
Nguồn reviewer: R1-002; R2-H1 bổ sung tính mutable, không tạo finding thứ sáu.

- Caller `state=model.prepare(data)` native rồi đưa state ấy vào
  `other.impute(data,H,initial=state)` masked. Hai view cũ vẫn coi H là anchor.
- Parent fixture tái hiện hidden truth=30, reused prediction=**30** (MSE H=0),
  fresh prediction=37.9218283. Output cuối qua count validator nhưng validator
  không thể phát hiện prediction đã nhìn truth trong state cũ.
- **Không có bằng chứng current CLI hoặc run COVID bị leakage:** full tự prepare
  đúng H; CLI refine đọc NPZ qua `initial_from_arrays`, có hash/mask/Q/label checks.
- Nếu `initial` chỉ là internal prevalidated state thì trigger này vi phạm
  precondition của caller, không phải lỗi ở đường gọi hợp lệ. Cần chốt/document
  phạm vi API; giữ finding như API footgun, không nâng thành bug CLI.
- Tuple còn bị mutate bởi refinement; repeat tuple không bảo đảm replay cùng
  initialization. Chưa có promise public immutability nên điểm này chỉ là chú ý API.
- **Sửa nhỏ nhất đề nghị:** validate masks/visible counts/Q/labels/domain ngay biên
  `initial`, tái dùng logic thuần dữ liệu không fit initializer lại; hoặc giới hạn
  rõ đây là internal prevalidated/consumed state. Không thêm state manager.

### A04 — P3 — Tsagris xóa residual proportion dương nhỏ khiến main init fail

**Source:** [`baselines/tsagris.py:141–150`](../src/longitudinal_impute/baselines/tsagris.py#L141),
reachable từ `main_model.py:109–113`, fail tại `data.py:193–196`.
Nguồn reviewer: R2-02. **Bug kế thừa, không phải regression do main.**

- Input hợp lệ `[9999999999999,NaN]`, total=1e13, residual counts=**1**,
  có donor dương. Proportion residual khoảng 1e-13 bị threshold 1e-12 đặt bằng 0.
- Parent `initialize` nhận `ValueError: no proposed mass for missing targets and Other`.
- Chỉ extreme-scale fixture, chưa thấy ảnh hưởng COVID hiện tại.
- **Sửa nhỏ nhất đề nghị:** không dùng tolerance budget âm để xóa residual dương;
  giữ mass dương cho R2, test whole route. Không thêm fallback uniform.

### A05 — P3 — Total tự suy bị overflow vẫn được chấp nhận

**Source:** [`data.py:84–91`](../src/longitudinal_impute/data.py#L84),
`:169–171,215–217`, `main_model.py:392–397`.
Nguồn reviewer: R2-03. **Bug data chung, extreme-scale.**

- Schema không total, hai target finite=1e308. Tổng overflow thành Inf;
  loader chỉ kiểm `<=0`, không kiểm finite sau cộng.
- Parent tái hiện main complete fast path báo `no_missing`, raw output finite
  nhưng total Inf, composition chứa NaN. Warnings có thể xuất hiện; probe quan sát
  state với NumPy errstate, không khẳng định mọi warning policy đều trả success.
- Total khai báo từ CSV được parser chặn Inf, chỉ nhánh tự suy total thiếu guard.
- **Sửa nhỏ nhất đề nghị:** finite/positive check sau sum, lỗi rõ CSV line/entity;
  test overflow trước khi model/output. Không cần normalizer mới.

## 3. Proposal và độ tối giản: phần nào thực sự đã đáp ứng

Đọc trực tiếp [proposal](2026_10_09-Revised%20proposal.pdf): p4 nêu inverse LR
không bảo toàn observed ở partial rows; p5 cho phép dự đoán composition trực tiếp;
p7 yêu cầu initialize → LR → fit/predict → so với vòng trước → lặp. Hình p8 có
temporal/feature attention, side information, gate/residual/skip và softmax.

Hiện thực Tsagris → CLR → temporal/feature attention → proposal → khóa anchors,
chia residual D+Other → cập nhật và check stop khớp luồng này. Một block thay vì
chồng nhiều layer, một fixed Q, MSE sau residual allocation, optimizer reset mỗi
round là **lựa chọn MVP được duyệt**, không phải chi tiết đã được PDF đặc tả.
Không có reverse diffusion/noise schedule/MI draws, nên không được gọi model là
diffusion hay multiple imputation có uncertainty. Chỉ output D targets có tổng
≤total; composition đầy đủ phải tính cả Other. Exhaustive targets/unknown-total
missing data vẫn ngoài domain đã chốt, không tự coi fixture đổi D là giải hết domain.

Hai view tách raw Q trước initializer/CLR là cần thiết. Main/network có owner rõ,
không trainer/engine/plugin framework. Differentiable allocation nhân công thức
ngắn của NumPy là cần gradient, không đủ lý do refactor toàn data layer. Runner
native/masked còn code lặp nhưng chưa có burden đo được để đề nghị rewrite.

## 4. Ý nghĩa thống kê: đúng điều gì, chưa được kết luận điều gì

1. **Fresh H isolation có bằng chứng, không chứng minh mọi API an toàn.** Reviewer
   1 đổi H truth 30→70 qua 3 outer iterations: fresh predictions/weights không đổi.
   H/Q không giao nhau. A03 là counterexample riêng khi caller đưa state sai case.
2. **Metric run trước đúng số học.** Reviewer 1 tái tính mọi group của 8 masked
   arrays, khớp summary. Main thấp MAE/MSE proportion trong seed 0 là mô tả đúng;
   Hron thấp hơn count errors ở random-cell cũng được báo. A01 không tự làm metric
   H sai nhưng làm recipe supervision cần chú thích.
3. **Một seed không có bằng chứng ưu thế tổng quát.** `paired` là common-success
   filtering, không phải paired hypothesis test/CI. Chưa có variability model seed,
   entity generalization hay accuracy tại natural NaN. Task là retrospective,
   known-total reconstruction, không forecasting; attention hai chiều là có chủ ý.
4. **Nhiều zero do budget quyết định.** Native 60,595/86,050 NaN bị ép zero
   (~70.4%); masked whole-feature 3,030/5,625, random-cell 2,728/5,579.
   Coverage=100% không đồng nghĩa tất cả ô do mạng suy ra. Model-decided MAE
   proportion main lần lượt 0.0458686 và 0.0271976; nên giữ báo cáo nhóm con.
5. **Chưa đạt fixed point.** Max-change cuối native/whole/random khoảng
   0.4873/0.4687/0.4270 so với tol=1e-4. Code ghi `converged=false/max_iter`
   đúng; không được gọi là hội tụ hoặc tự tăng vòng/tune trên H sau khi nhìn điểm.
6. **Train loss không là validation.** Q cố định dùng fit; nhiều Q cũng forced-zero.
   Giá trị log là mean cell-weighted của online batch losses, không phải đánh giá
   cuối epoch tại một bộ weights cố định.
7. **S01 — weighting cần chốt, chưa xếp bug.** Mỗi minibatch entity dùng mean Q rồi
   một Adam step; các batch ít Q không được scale theo support, trong khi log được
   weighted theo Q count (`main_model.py:235–259`). Vì vậy đừng gọi optimization là
   global cell-weighted MSE chính xác. Entity/batch sampling là một lựa chọn hợp lệ,
   nhưng cần công bố hoặc scale theo support nếu global cell weighting là ý định.
   Plan chưa chốt weighting nên không suy đây là statistical invalidity.
8. **Proposal weights không phải joint distribution/uncertainty.** R2 chuẩn hóa
   lại phần free làm mass proposal trên observed triệt tiêu; chỉ restored counts
   mới là estimator được chấm. Không coi softmax proposal là full composition đã
   được supervised/định danh hay uncertainty đã calibrated.
9. **Wasserstein native chỉ diagnostic**, không chứng minh điền từng dòng đúng;
   observed-vs-completed còn trộn lại các observed bất biến. Protocol trước đã có
   cảnh báo đúng, cần giữ nguyên trong mọi báo cáo tiếp.

## 5. Bằng chứng, kiểm chứng chéo và giới hạn

- Parent fresh suite: **257 passed in 24.36s**, RuntimeWarning-as-error, zero skips:
  [pytest](../.harness/main-model/audit-20261113/regression/pytest.txt),
  [parent checks](../.harness/main-model/audit-20261113/parent-checks.json).
- Reviewer 1: 178 tests chọn lọc pass, H-invariance 3 vòng và recompute 8 arrays.
- Reviewer 2: 257 tests pass; fixtures D=2/5/17 hợp lệ hai vòng; full/refine
  max-diff=0, evaluate rows khớp, init/evaluate chạy khi chặn torch import.
  Các test sets trùng nhau, **không cộng số test thành coverage độc lập**.
- Parent tự tái hiện A01/A03 ở fixture mới:
  [script](../.harness/main-model/audit-20261113/parent_reproduce.py),
  [results](../.harness/main-model/audit-20261113/parent-reproductions.json).
- Parent tự tái hiện A02/A04/A05 ở fixture/CSV/output mới:
  [script](../.harness/main-model/audit-20261113/parent_numerics.py),
  [results](../.harness/main-model/audit-20261113/parent-numerics/results.json).
- Provenance run COVID/source/input hiện tại khớp, 27 artifact file hashes đúng.
  Không sửa artifact run trước, không retrain để có số đẹp hơn.
- Round đối chiếu sau initial reports: [reviewer 1](../.harness/main-model/audit-20261113/reviewer1/RECONCILIATION_VI.md)
  và [reviewer 2](../.harness/main-model/audit-20261113/reviewer2/RECONCILIATION_VI.md).
  Đồng ý hai P2, hai P3 kế thừa; A03 giới hạn API hardening có điều kiện,
  mutability/weighting/single-seed không nâng thành lỗi CLI hoặc reporting bug.
  Mọi retained mechanism đã được parent tái hiện, không phụ thuộc peer agreement.
- [Final integrity](../.harness/main-model/audit-20261113/final-integrity.json)
  kiểm source/input/config của snapshot không đổi và links/evidence tồn tại.

**Không thể chứng nhận “không có bug nào”.** Audit này không phải kiểm mọi platform,
GPU/concurrency/service, extreme-scale bất kỳ, core baseline toàn diện, mọi paper
trích dẫn hoặc toàn bộ archive. Local CLI CPU/single-process là workload được audit.

## 6. Hành động nhỏ nhất đề nghị — chưa thực hiện

1. Sửa A01 và A02 bằng failing-before regressions; không đổi kiến trúc/loss/CLI.
2. Chốt precondition của API `initial` (A03) và weighting training (S01).
3. Numerical hardening kế thừa A04/A05 bằng tests riêng, giữ contract baseline/R2.
4. Sau sửa recipe, chạy lại dưới **run ID mới**, báo cấu hình/actual Q rate, nhiều
   evaluation/model seeds phù hợp protocol chốt trước; giữ nguyên run lịch sử.
5. Chưa quảng cáo superiority, forecasting hoặc natural-missing accuracy.

**Phán quyết bàn giao:** đạt scaffold/MVP có kiểm chứng; **chưa đạt mức có thể
tuyên bố sạch bug hoặc đủ kết luận khoa học xác nhận**. Các repair cần được user
cho phép riêng vì yêu cầu hiện tại chỉ audit.
