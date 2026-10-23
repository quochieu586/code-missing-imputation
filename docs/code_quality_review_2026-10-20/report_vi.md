# Review chất lượng code: Hron, MICoDa, Tsagris và triển khai liên quan

Ngày 20/10/2026. Skill: `thermo-nuclear-code-quality-review`.

**Kết luận: cần chỉnh cấu trúc trước khi mở rộng.** Có 2 finding P1 và 5 finding P2. Những vấn đề chính là trạng thái fit không được cập nhật nguyên khối, persistence không resume được sau gián đoạn, hợp đồng core/adapter chưa đúng với triển khai và diagnostics bị phụ thuộc vào cấu trúc dictionary của từng thuật toán. Các finding dưới đây có vị trí cụ thể và probe nhỏ; không dựa vào phỏng đoán từ kết quả benchmark.

## Phạm vi và cách kiểm tra

Review working tree hiện tại trên `main`: toàn bộ core trong `baselines/`, registry, adapter, regression, CSV runner/I/O, metadata, panel/contracts/constraints; benchmark engine/driver/protocol/masks/metrics/report và ba file pytest. Các scaffold neural/pipeline chỉ được kiểm tra ranh giới tích hợp, không coi chúng là tính năng đã hiện thực.

`nextgen/src/`, `nextgen/tests/` và `nextgen/pyproject.toml` hiện **chưa được Git theo dõi**; `git ls-files` cho các đường dẫn này không trả file. Vì vậy đây là review snapshot working tree, không phải diff của một commit đã push. Source hash vẫn khớp manifest run v2: `d2ed444263648f5016fee6dd0855d875f9edb8eb44484013394b661f05497d15`.

Đã chạy `python -B -m pytest -q -p no:cacheprovider nextgen/tests`: **45 passed in 4.43s**. Runtime review là CPython 3.12.10, NumPy 2.2.5, một BLAS thread; khác runtime đã khóa của benchmark cũ. Không chạy lại benchmark, smoke data thật hoặc oracle R. Các probe mới chỉ dùng ma trận nhỏ và failure injection, không chấm accuracy và không sửa thuật toán.

Evidence nằm tại `.harness/baseline-code-review-2026-10-20/`: `pytest.txt`, `probes.py`, `probe_execution.txt`, `probe_results.json`, `regression_probe.py`, `regression_probe.json`. Thư mục `interrupted_job/` được cố ý giữ để thể hiện trạng thái job dở dang; không xóa file.

## CQ1 — P1: trạng thái fitted được công bố trước khi fit thành công

Vị trí chính: [micoda.py](../../nextgen/src/longitudinal_impute/baselines/micoda.py), dòng 102–123; [adapters.py](../../nextgen/src/longitudinal_impute/baselines/adapters.py), dòng 35–65. HronIterative và Tsagris cũng thay đổi nhiều fitted attributes trực tiếp trong quá trình fit.

MICoDa thay `width_`, `reference_`, `shifted_total_`, `covariate_width_` trước khi regression có thể thất bại, trong khi `coefficients_` từ fit trước vẫn tồn tại. Probe fit thành công với reference 0, sau đó refit thiếu complete cases gây `BaselineError`; reference đã thành 1, coefficients vẫn là bộ cũ. Cùng query vẫn predict thành công nhưng kết quả đổi **0.07140885175** trên thang tỷ lệ. Đây là trạng thái pha trộn hai lần fit, không phải một model hợp lệ.

Adapter cũng gán `core_` trước khi `core_.fit()` thành công, rồi dùng sự tồn tại của `core_` làm fitted guard. Probe first-fit MICoDa(W=0) thất bại vì zero; gọi prediction sau đó vẫn trả counts `[[10,25.5,5]]` qua train-mean fallback, với reason `Fit MICoDa before prediction`. Fallback đang biến lỗi vòng đời model thành một output hợp lệ về mặt count.

**Hướng cấu trúc:** tính toán fitted state trong biến cục bộ/một object riêng, chỉ commit `fitted_` khi toàn bộ fit thành công. Chọn rõ policy failed-refit: giữ model cũ nguyên vẹn hoặc vô hiệu hóa model; không giữ nửa state mới. `NotFittedError` phải đi thẳng ra caller, không được xử lý như một lỗi dòng có thể fallback. Thêm regression checks failed-first-fit và failed-refit trước khi refactor.

## CQ2 — P1: persistence tạo trạng thái không thể resume bằng luồng hiện tại

Vị trí chính: [engine.py](../../nextgen/src/longitudinal_impute/benchmark/engine.py), dòng 145–162. Cùng pattern xuất hiện ở supplementary `execution.txt` dòng 237 và final export dòng 310–318.

`evaluate()` tạo thư mục job trước prediction nhưng khi resume chỉ xét `result.json`; nếu thư mục đã có mà result chưa được ghi, `mkdir(exist_ok=False)` làm lần resume thất bại. Probe inject interruption ngay sau tạo thư mục, chưa gọi model; lần tiếp theo nhận `FileExistsError`. Một interruption bình thường đã đủ khiến một module tự nhận là resumable bị kẹt.

Exclusive writes bảo vệ dữ liệu cũ là đúng, nhưng không thay thế lifecycle cho các attempt. Hơn nữa, fast path chỉ đọc result/marker theo sự tồn tại, không kiểm tra toàn bộ artifacts/hash trước khi chấp nhận cache.

**Hướng cấu trúc:** một lớp persistence nhỏ sở hữu `attempts/<id>/`, manifest artifacts và completion marker được publish nguyên khối sau khi artifacts được ghi/validate. Resume tạo attempt mới, giữ attempt dở làm evidence; không cần xóa để chạy tiếp. Reuse phải kiểm job identity/hash. Engine chỉ điều phối fit/evaluate, không tự lặp các biến thể của cùng quy tắc directory/marker. Test crash trước/sau từng bước ghi và test resume không overwrite.

## CQ3 — P2: adapter đang sở hữu chính sách thuật toán vì hợp đồng core chưa tồn tại đúng nghĩa

Vị trí: [adapters.py](../../nextgen/src/longitudinal_impute/baselines/adapters.py), dòng 48–60 và 77–79; [interfaces.py](../../nextgen/src/longitudinal_impute/baselines/interfaces.py), dòng 6–11; [hron.py](../../nextgen/src/longitudinal_impute/baselines/hron.py), dòng 155–173.

Adapter phân loại core bằng `isinstance`, thay trực tiếp flags Tsagris, tự quyết zero encoding và buộc mọi prediction thành các call một hàng. Protocol `Baseline` lại mô tả `fit(panel,scope)->None` và unconstrained prediction, trong khi core thực tế nhận ndarray/covariates, adapter trả constrained counts và cả hai fit đều trả self. Không có protocol nào mô tả đúng API `predict_result` mà adapter thực sự cần.

Đây còn là coupling hành vi: Hron dùng covariance stop cho batch nhiều hàng, update-norm stop cho một hàng. Probe cùng fitted model/query cho chênh lệch batch so với rowwise **0.01245** ở tol=0.1, **3.339e-5** ở tol=0.001. Cách thích nghi một hàng đã được tài liệu hiện tại công bố; finding này không nói adaptation đó bị giấu. Vấn đề là policy này được kích hoạt ngầm bởi loop fallback trong adapter, nên một tối ưu batching thông thường có thể đổi estimator.

**Hướng cấu trúc:** tách đúng hai contract nhỏ: numerical core và fitted count model. Registry/recipe xác định input-domain/encoding và prediction policy; adapter chỉ chuyển counts/Other, gọi policy đã chọn và áp constraint. Chế độ Hron rowwise/population phải là cấu hình có tên. Đưa validation counts/numerical errors về tầng dùng chung: hiện `constraints.py` và `metadata.py` phải import ngược vào `baselines`, còn adapter import lại chúng; lazy imports chỉ né vòng phụ thuộc, không làm ownership rõ hơn. Không cần một framework plugin phức tạp; chỉ cần boundary phản ánh đúng dữ liệu và trách nhiệm.

## CQ4 — P2: diagnostics theo dictionary lồng nhau làm mất policy của TsagrisAdaptive

Vị trí: [engine.py](../../nextgen/src/longitudinal_impute/benchmark/engine.py), dòng 107–121; [tsagris.py](../../nextgen/src/longitudinal_impute/baselines/tsagris.py), dòng 246–253.

`compact_diagnostics()` giả định `policy` nằm trực tiếp trong mỗi `core.records` entry. TsagrisAdaptive đặt actual row records thêm một tầng trong `prediction_diagnostics`. Probe thực sự dùng `global_donor_mean`, nhưng compact output lại có `core_policies={}`. Số liệu policy được xuất thiếu mà không có lỗi; `train_mean` fallback là cơ chế khác, không thay thế cho số liệu này.

**Hướng cấu trúc:** core trả cùng schema row-events ở boundary, với row ID/index rõ ràng; thông tin tuning theo pattern là summary riêng. Fit summary chỉ giữ một lần, thay vì nhúng lại vào diagnostics của từng row rồi nhân bản khi JSON serialize. Khi đó compactor chỉ đếm một schema, không cần hiểu cây nội bộ của từng thuật toán. Test cả basic, adaptive, zero-anchor và uniform-missing policies. Finding này chứng minh lỗi thống kê diagnostics; không tự suy ra MAE trong accepted run sai.

## CQ5 — P2: taxonomy lỗi dựa trên message và validation xảy ra sai thứ tự

Vị trí: [regression.py](../../nextgen/src/longitudinal_impute/baselines/regression.py), dòng 12–19; [engine.py](../../nextgen/src/longitudinal_impute/benchmark/engine.py), dòng 40–43 và 281–285; [adapters.py](../../nextgen/src/longitudinal_impute/baselines/adapters.py), dòng 82–86.

`BaselineError` đang gộp domain, rank/support, configuration và lifecycle. Engine phân numerical/not-applicable bằng substring tiếng Anh. Thay câu chữ có thể đổi scientific status. OLS còn gọi `matrix_rank()` trước kiểm finite; probe NaN nhận `LinAlgError: SVD did not converge`, vượt ngoài error contract, thay vì lỗi input có cấu trúc. Lỗi được catch ở supplementary chỉ lưu message nên không còn target/iteration/traceback để điều tra LTS.

**Hướng cấu trúc:** validation shape/finite trước linear algebra; mã lỗi/exception rõ cho contract, support/rank, numerical và not-fitted. Fallback chỉ được xử lý các lỗi dòng đã cho phép. Serialize mã lỗi cùng stage/iteration/target/traceback, để message dành cho người đọc. Dùng một nơi sở hữu các error types, không thêm các substring checks mới ở những caller khác.

## CQ6 — P2: API regression khiến cùng phép phân rã bị tính lặp ở tầng gọi và tầng solve

Vị trí: [regression.py](../../nextgen/src/longitudinal_impute/baselines/regression.py), dòng 15–19 và 40–51; [micoda.py](../../nextgen/src/longitudinal_impute/baselines/micoda.py), dòng 57–71.

LTS kiểm rank một subset rồi gọi OLS; OLS lại kiểm rank và gọi `lstsq`. Bản thân `lstsq` đã trả rank/singular values. `ols(x,y)` còn được gọi đầu `lts()` chỉ để validation nhưng tính nghiệm rồi bỏ. Probe nhỏ 30×2, ba random starts và tối đa hai C-steps đã gọi **matrix_rank 21 lần, lstsq 11 lần**. Đây là lặp trách nhiệm kiểm tra/giải, không phải một chi tiết micro-optimization đơn lẻ.

MICoDa `_regress` cũng giải từng response riêng ngay cả khi tất cả dùng cùng eligible rows/design trên miền dương hoàn chỉnh sau update; một phép solve đa response có thể thay nhiều lần phân rã cùng ma trận. Initial pairwise author compatibility có eligibility khác phải được giữ rõ.

**Hướng cấu trúc:** một checked solve trả nghiệm và rank từ cùng phép `lstsq`; caller LTS dùng đúng result/error đó, không preflight bằng solve hoặc SVD khác. Với MICoDa complete-case/common-design, truyền ma trận ALR responses cùng lúc. Đây là tái sử dụng canonical solver và loại logic lặp, chưa phải bằng chứng đạt budget 180 giây; phải đo runtime và parity sau khi triển khai.

## CQ7 — P2: Hron lặp qua exp→CLR dù state nội bộ cần miền log dương nghiêm ngặt

Vị trí: [hron.py](../../nextgen/src/longitudinal_impute/baselines/hron.py), dòng 121–135; [common.py](../../nextgen/src/longitudinal_impute/baselines/common.py), dòng 49–60.

Hron dùng generic softmax trả nonnegative simplex rồi lập tức đưa state vào CLR yêu cầu strictly positive. Probe finite logits `[0,-1000,-10]` tạo một zero chính xác; CLR sau đó báo lỗi. Generic softmax không sai với những caller chấp nhận zero; chính boundary Hron yêu cầu mạnh hơn mà không bảo đảm. Probe này xác minh lỗ hổng hợp đồng, chưa chứng minh target/iteration cụ thể đã làm Hron-LTS thất bại trong run cũ.

**Hướng tái cấu trúc mạnh:** giữ state centered-log/ILR xuyên suốt sweep, đổi pivot bằng linear transforms và chỉ exp/normalize ở output boundary. Cách này loại hẳn round-trip exp→log giữa từng target, thay vì thêm epsilon vào một helper dùng chung. Cần kiểm parity trên miền thông thường và khai báo chính sách output nếu phần nhỏ không biểu diễn dương được trong float64; kiểm finite tại linear-algebra boundaries. Nếu chọn floor/clipping, đó là policy có ảnh hưởng estimator và cần thí nghiệm riêng.

## Đánh giá từng họ và thứ tự sửa

| Phần | Đánh giá chính | Việc cần ưu tiên |
|---|---|---|
| Hron kNN/iterative | Core tách riêng và có equation tests; iterative bị gắn stopping theo shape, exp/log round-trip và solver lặp việc | Prediction contract có tên; log-domain state; checked solve |
| MICoDa V1/V2 | Công thức, semantics và W được khai báo; fitted state có thể pha trộn sau lỗi; nhiều solve cùng design | Atomic fit; multiresponse regression; isolate update policies |
| Tsagris/basic/adaptive | Zero policy và train-only tuning rõ; adaptive diagnostics không cùng schema; adapter mutate flags | Normalize events; cấu hình policy tại factory; giữ tuning state có ownership rõ |
| Adapter/contracts | Counts/Other và hidden-label boundary có tests tốt; protocol/type ownership chưa khớp thực tế | Hai contract đúng nghĩa; fallback có taxonomy; encode policy rõ |
| Benchmark persistence | Hashes và exclusive writes có giá trị; directory/result lifecycle không phục hồi được sau interruption | Attempts + atomic completion + artifact validation |

Sửa CQ1 và CQ2 trước, kèm các failure-injection tests. Tiếp theo CQ3–CQ5 để chuẩn hóa contract/lỗi/diagnostics; sau đó CQ6–CQ7 để tối ưu có parity. Không sửa performance bằng thay batch shape hoặc chèn epsilon âm thầm. Source hiện lớn nhất chỉ 346 dòng (`benchmark/engine.py`), không file nào vượt 1.000 dòng; không có lý do tách file chỉ vì ngưỡng kích thước. Tách theo ownership lifecycle, numerical state và diagnostics sẽ có ích hơn tách cơ học.

Test suite đang bảo vệ nhiều công thức và hidden-label boundaries, nhưng chưa có failed-refit, failed-first-fit prediction, crash/resume, adaptive-policy aggregation hay batch-policy compatibility. 45 tests pass không đáp ứng các điều kiện còn thiếu đó. Review không phê duyệt mở rộng cấu trúc hiện tại, đồng thời không phủ định kết quả v2 đã hoàn tất và kiểm toán trên luồng thành công.

## Giới hạn và trạng thái

Không sửa source, không refactor, không commit/push, không chạy lại jobs benchmark. Chỉ thêm report và module probe `.py`/logs trong task review. Hash source trước và sau bằng hash đã khóa. Các đề xuất tái cấu trúc cần unit/parity checks và experiment mới trước khi đưa ra claim tốc độ/hội tụ/accuracy. Không xác nhận đã giải quyết MICoDa 50 vòng, budget LS/V1 hay root cause LTS cũ trong lượt review này.
