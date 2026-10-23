# Review toàn diện sau sửa P1/P2 — 20/10/2026

**Kết luận: chưa đạt mức phê duyệt để mở rộng pipeline.** Có **2 finding P1 và 5 finding P2** được chứng minh bằng source và probe nhỏ. Các sửa vừa rồi có tác dụng thật: MICoDa/adapter giữ state sau failed refit, evaluation retry dùng attempt mới, lỗi regression được phân loại theo type, row-policy Tsagris không còn bị mất và shared-design MICoDa giảm số lần solve. Tuy nhiên hai vấn đề P1 cũ mới được xử lý ở một số tầng; adapter mới còn làm mất diagnostics ngoài row events.

Đây là **review**, không phải một đợt sửa source tiếp theo. Không thay thuật toán, không chạy lại benchmark offline đã hoàn tất, không sửa/xóa artifact của run được chấp nhận và không commit/push.

## Phạm vi và evidence

- Inventory/AST: **102 file Python** trong `nextgen/src`, `nextgen/tests`, `src`, `scripts`, `tests`; 40 module trong package nextgen. Đối chiếu snapshot source v2: **15 module nextgen khác hoặc mới**. Đọc sâu các core/adapters/constraints/CSV/metadata, benchmark engine/driver/report/metrics/masks/splits; các numerical/model modules legacy, runner/cache/audit và ranh giới scaffold.
- `nextgen/src`, `nextgen/tests`, `nextgen/pyproject.toml` vẫn chưa được Git theo dõi. Kết luận áp dụng cho **working-tree snapshot**, không phải một commit/PR đã push. Không có `AGENTS.md` áp dụng được tìm thấy trong repo và các thư mục cha.
- Những module `pipeline`, neural training/batching và phần lớn Protocol nextgen vẫn là scaffold, được README công bố. Không coi chúng là tính năng đã hiện thực, cũng không kết luận một Protocol tự bảo đảm chống leakage.
- `ref/` là mã bên thứ ba, không phải đối tượng refactor của review này. Báo cáo không xác nhận lại paper tables, hiệu năng full-data, hội tụ sau 50 vòng hay budget 180 giây.
- Evidence chi tiết: [.harness/post-repair-review-2026-10-20](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/.harness/post-repair-review-2026-10-20). Probes là `.py`, dùng dữ liệu tổng hợp và fixture riêng; các file ghi dở/cố ý làm hỏng được giữ để tái hiện, không xóa.

## Kết quả thực thi

| Kiểm tra | Kết quả | Evidence |
|---|---|---|
| Pytest nextgen, Python mặc định | **45 passed**, 1,45 giây | [log](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/.harness/post-repair-review-2026-10-20/pytest_nextgen.txt) |
| Pytest nextgen, native CPython + dependency research | **45 passed**, 1,26 giây; cùng 45 test, không tính thành 90 test khác nhau | [log](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/.harness/post-repair-review-2026-10-20/pytest_nextgen_native.txt) |
| Pytest legacy với native runtime | **122 passed, 1 failed**, 23,28 giây; 2 warning metric trên empty stratum | [log](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/.harness/post-repair-review-2026-10-20/pytest_legacy_native.txt) |
| Oracle base-R và source MICoDa đã pin | **9/9 pass**, max abs error `5.828670879282072e-16`, tolerance `1e-9` | [comparison](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/.harness/post-repair-review-2026-10-20/reference_oracles/comparison.json) |
| 9 probe nextgen | Thực thi đủ; xác nhận cả phần sửa đúng và lỗi còn lại | [kết quả](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/.harness/post-repair-review-2026-10-20/probe_results.json) |
| Cache/input registry legacy | Tái hiện checkpoint dở và NPZ vắng vẫn được nhận; PDF registry rỗng | [kết quả](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/.harness/post-repair-review-2026-10-20/legacy_probe_results.json) |
| Completion marker có đủ status/fold/time/hash nhưng thiếu artifacts | Driver vẫn báo `all_folds_complete`; supplementary vẫn skip | [kết quả](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/.harness/post-repair-review-2026-10-20/marker_complete_results.json) |

Runtime chính của probe/oracle/legacy: native CPython **3.12.14**, NumPy **2.5.3**, Torch **2.14.0+cpu**, dùng dependency sẵn tại `.venv-e3-revised/Lib/site-packages`. Numerical environment đặt một BLAS thread; legacy runner có thiết lập Torch threads riêng. Python mặc định là MinGW, không có Torch; lần chạy legacy ban đầu dừng với 6 lỗi collection do dependency, được giữ trong `pytest_legacy.txt`. Không coi lỗi môi trường này là một lỗi thuật toán. Không cài package mới.

## R1 — P1: completion lifecycle vẫn bị chia ra nhiều tầng, marker cấp trên bỏ qua kiểm tra artifact

**Vị trí:** [driver.py:29](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/nextgen/src/longitudinal_impute/benchmark/driver.py:29), [engine.py:338](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/nextgen/src/longitudinal_impute/benchmark/engine.py:338).

`evaluate()` nay kiểm identity, hash và nội dung prediction tốt hơn. Nhưng driver chỉ kiểm `source_hash/config_hash` của `fold/completion.json` rồi đưa fold vào completed; `run_fold()` lại skip supplementary chỉ vì `finished.json` tồn tại. Hai tầng này không đi qua validation mới của evaluation artifacts. Một artifact mất hoặc hỏng sau khi marker được tạo sẽ bị bỏ qua khi resume bằng public driver.

Probe ghi một completion marker có đúng đủ `status=done`, fold, seconds và hai hash giả lập hợp lệ, nhưng **không có bất kỳ evaluation artifact nào**. Với `load_run` được thay bằng fixture in-memory, driver vẫn trả `all_folds_complete`. Supplementary marker `status=ok`, candidate và seconds cũng được chấp nhận dù không có result/prediction jobs. Đây là kiểm tra orchestration; không có worker hay benchmark thực nào được chạy.

**Sửa cấu trúc:** đưa lifecycle vào một `ArtifactStore/JobStore` nhỏ, sở hữu attempt, job identity, danh sách child/artifact và completion validation. Mọi fast path của driver, selection, supplementary và final export phải dùng chung contract này. Marker cấp fold chỉ hợp lệ khi toàn bộ child bắt buộc hợp lệ; terminal resource/numerical failures vẫn được ghi nhận riêng theo protocol. Thêm kiểm tra resume với marker đúng nhưng child thiếu/hỏng, không chỉ crash trước khi tạo marker.

## R2 — P1: public Hron/Tsagris cores vẫn công bố fitted state trước khi fit thành công

**Vị trí:** [tsagris.py:224](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/nextgen/src/longitudinal_impute/baselines/tsagris.py:224), [hron.py:177](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/nextgen/src/longitudinal_impute/baselines/hron.py:177). TsagrisKNN cũng gán donor/schema trước validation alpha tại dòng 77–83.

Atomic fit đã được sửa ở MICoDa và CountBaselineAdapter, nhưng chưa thành contract chung của numerical cores. TsagrisAdaptive thay `donors_` trước khi kiểm ít nhất hai donor, giữ `selected_` cũ nếu validation thất bại. Sau một successful fit, refit bằng một donor gây `BaselineError`; cùng query sau đó vẫn predict thành công, nhưng kết quả đổi từ `[0.0660271,0.6339729,0.2,0.1]` sang `[0.6,0.1,0.2,0.1]`: **max delta 0,5339729162**. Đây là model pha trộn fitted states.

HronIterative reset `models_` rồi mới regression. Probe refit với design rank-deficient làm fit thất bại; prediction sau đó vượt fitted guard vì `models_` tồn tại và phát `KeyError`. `CountBaselineAdapter` hiện cô lập các lỗi fit này bằng candidate copy, nên không quy lỗi corruption này cho accepted benchmark hoặc wrapper đã sửa; lỗi nằm ở public direct core API.

**Sửa cấu trúc:** mỗi estimator tính một fitted-state object riêng và publish sau thành công; dùng cùng policy failed-refit cho mọi public core. Guard dựa vào state hoàn chỉnh, không dựa vào sự tồn tại của một attribute. Giữ positive probes của MICoDa/adapter, bổ sung failed-first-fit/failed-refit cho HronKNN/Iterative và cả hai Tsagris cores vào test suite thường trực.

## R3 — P2: chuẩn hóa row events làm mất diagnostics hội tụ và CV

**Vị trí:** [adapters.py:102](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/nextgen/src/longitudinal_impute/baselines/adapters.py:102), [adapters.py:126](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/nextgen/src/longitudinal_impute/baselines/adapters.py:126).

Adapter giờ chỉ giữ `result.diagnostics['row_events']`, không giữ phần prediction summary khác. HronIterative trả `iterations/converged/stopping/deltas/output_underflow_parts`, nhưng không có row events: toàn bộ thông tin đó biến mất qua adapter. `fit_diagnostics` chỉ lấy từ `fit_result_`, nên không thay thế được diagnostics hội tụ của prediction.

TsagrisAdaptive trả cả CV tables/pattern CV sizes; core không có `fit_result_`, vì thế adapter ghi `fit_diagnostics=None`, đồng thời bỏ CV từ kết quả prediction. Probe xác nhận `global_donor_mean` đã được giữ đúng, nhưng CV table của cả pattern train và unseen query không xuất hiện trong diagnostics adapter. Flatten policy là sửa đúng; xóa mất diagnostics còn lại là regression mới.

**Sửa cấu trúc:** một schema chung có `fit_summary`, `prediction_summary`, `row_events`, và tham chiếu tuning artifact. Chuẩn hóa một lần tại core boundary, giữ hội tụ theo query row và lưu tuning table một lần cho mỗi fit/pattern. Compact report chỉ aggregate schema chung; không khôi phục việc dò dictionary tree tùy từng thuật toán, cũng không sao chép toàn bộ CV table vào mọi row.

## R4 — P2: policy có nhiều nơi sở hữu, manifest “resolved” không mô tả model thực chạy

**Vị trí:** [registry.py:64](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/nextgen/src/longitudinal_impute/baselines/registry.py:64), [adapters.py:65](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/nextgen/src/longitudinal_impute/baselines/adapters.py:65), [runner.py:58](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/nextgen/src/longitudinal_impute/baselines/runner.py:58).

Hiện có recipe map, class attributes, `make_core(...degenerate_policies=...)`, hai method `with_degenerate_policies()`, và optional capability discovery trong adapter. Benchmark configure policy trong constructor; CLI/smoke configure bản copy trong adapter. Đây là cùng một quyết định bị thể hiện qua nhiều đường.

Probe chạy **CLI execute trên CSV tổng hợp 3 dòng**, bật degenerate policies. Output dùng `global_donor_mean`, nhưng manifest ghi `resolved_core_config.zero_anchor='error'`, bởi runner serialize object `core` trước cấu hình thay vì fitted `adapter.core_`. Người dùng không thể lấy “resolved config” đó để dựng lại cùng estimator. Boolean degenerate flag riêng không làm một trường resolved sai trở thành đúng.

**Sửa cấu trúc:** một method specification/factory sở hữu cấu hình hiệu lực, input domain và prediction policy. Dùng nó cho benchmark, CLI và smoke; adapter chỉ nhận core đã cấu hình. Serialize cấu hình từ đúng fitted core hoặc immutable effective configuration. Loại các đường configure trùng nhau và optional reflective capability khi không còn cần; không chỉ đổi tên thêm một wrapper.

## R5 — P2: count constraint có thể trả inf/NaN dù đầu vào hữu hạn

**Vị trí:** [constraints.py:55](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/nextgen/src/longitudinal_impute/constraints.py:55), [constraints.py:62](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/nextgen/src/longitudinal_impute/constraints.py:62).

Với counts `[10,NaN,NaN]`, total 100 và proposal D+Other `[1,nextafter(0,1),0,0]`, mọi proposal đều finite/nonnegative. `residual/mass` overflow, rồi phép nhân sinh `inf/NaN`. Guard cuối chỉ kiểm âm và vượt budget; so sánh với NaN trả false. Hàm trả **`[10,inf,NaN]` không ném lỗi**. CSV exporter/scorer có thể chặn ở tầng sau, nhưng public CountBudgetConstraint đã vi phạm contract của chính nó; gọi adapter trực tiếp cũng không có output check bổ sung.

Đây là edge case số học, không phải bằng chứng nó đã xuất hiện trong run v2. Hron hiện có thể trả các part cực nhỏ/zero ở bước xuất simplex, nên ranh giới này cần kiểm độc lập với source của proposal.

**Sửa cấu trúc:** normalize vector trọng số bằng scale an toàn trước phân bổ, thay vì tạo một factor có thể overflow; kiểm finite của mọi output/Other ở boundary và trả typed NumericalError nếu không thể bảo đảm. Không thêm epsilon hoặc âm thầm tạo mass. Thêm probes subnormal/very-large proposals và bảo toàn tỷ lệ phần unknown.

## R6 — P2: frozen-input registry legacy phụ thuộc layout cũ và bỏ qua tài liệu bắt buộc một cách im lặng

**Vị trí:** [diagnose_e5_rework.py:25](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/scripts/diagnose_e5_rework.py:25), [test_diagnose_e5_rework.py:80](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/tests/unit/stage_b/test_diagnose_e5_rework.py:80).

`frozen_hashes()` dùng `directive.glob('*.pdf')`. Cấu trúc hiện tại có PDF trong `directive/literature/csdi_2021/` và `directive/proposals/`, nên registry trả **0 PDF**. Test yêu cầu `Hron.pdf/phylo.pdf` fail. Hai PDF này không có trong snapshot `directive` hiện tại; có bản Markdown/assets liên quan. Các tracked deletions gốc đã tồn tại trước review và được giữ nguyên.

Đây không chỉ là test path cũ: so sánh `before == after` có thể báo preservation khi input bắt buộc đã vắng ngay từ đầu và không được đưa vào tập hash. Một phép glob không định nghĩa được contract input khoa học. Đổi sang recursive glob chỉ bổ sung hai PDF hiện hữu, chưa giải quyết nguồn Hron/phylo bắt buộc.

**Sửa cấu trúc:** một source registry tường minh chứa canonical path, expected hash, vai trò và trạng thái required/archived/transcription. Fail trước audit nếu required input vắng; migration đường dẫn phải có mapping tới snapshot gốc. Nếu dùng Markdown thay PDF, ghi rõ phạm vi claim và hash cả assets cần preview. Không xóa assertion để che lỗi và không coi chuyển vị trí file là thay đổi scientific source không cần ghi nhận.

## R7 — P2: runner legacy trộn training/evaluation/cache và công bố checkpoint trước khi ghi xong

**Vị trí:** [run_e5_mask_conditioning.py:394](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/scripts/run_e5_mask_conditioning.py:394), [run_e5_mask_conditioning.py:428](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/scripts/run_e5_mask_conditioning.py:428), [run_e5_mask_conditioning.py:440](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/scripts/run_e5_mask_conditioning.py:440).

File đã có **1.105 dòng**, gồm prepare, mask banks, training, validation fixtures, checkpoint cache, evaluation cache, gates và report; `stage_evaluate` dài 232 dòng, có 31 nhánh điều khiển theo AST. Đây là debt **có trước P1/P2**, không phải một lần vượt ngưỡng 1.000 dòng do các sửa baseline hiện tại. Các runner mới tiếp tục import helpers từ runner cũ, thay vì một experiment library nhỏ có ownership rõ.

Một hệ quả cụ thể: `train_cache_save()` ghi JSON index trước `torch.save()`. Probe tạo checkpoint file dở rồi inject OSError. `train_cache_load()` vẫn trả index hợp lệ vì chỉ kiểm existence/config; `torch.load()` phát `UnpicklingError`. Lần chạy sau tiếp tục chọn cùng cache lỗi. `eval_cache_load()` còn chấp nhận JSON khi file NPZ không tồn tại. Các pattern persistence khác nhau giữa legacy và nextgen đang cùng được gọi là resume-safe.

**Sửa cấu trúc:** extract checkpoint/artifact storage với manifest publish sau artifact, hash và validation; giữ attempt dở và retry ở attempt mới. Tách experiment preparation, training/checkpoint selection và evaluation/report thành module có context/config tường minh; runner chỉ điều phối. Giữ nguyên protocol và test parity, không dùng refactor để thay mask/seed/gate. Không cần chia nhỏ cơ học tất cả helper nếu ownership vẫn rải rác.

## Đối chiếu bảy finding cũ

| Finding cũ | Kết quả sau sửa và kiểm chứng hiện tại |
|---|---|
| CQ1 — fitted-state atomicity | **Sửa đúng ở MICoDa/adapter**, positive probes pass. Còn public Hron/Tsagris: R2. |
| CQ2 — resumable persistence | **Evaluation retry/cache integrity đã cải thiện**, hai attempts được giữ và artifact hỏng bị từ chối. Completion cấp trên vẫn bypass: R1; legacy có R7. |
| CQ3 — core/adapter contract | Hai Protocol mới mô tả đúng API hơn, row/population adaptation tường minh; policy ownership/provenance chưa thống nhất: R4. |
| CQ4 — adaptive policies bị mất | `global_donor_mean` nay đi qua canonical row events đúng. Nhưng prediction/CV summaries bị mất: R3. |
| CQ5 — error taxonomy | NaN regression thành BaselineInputError, rank thành RankDeficientError, injected linalg failure thành NumericalError; status theo type. NotFitted không còn bị fallback ở adapter. Chưa coi mọi numerics trong dự án đã kín: R5. |
| CQ6 — solve lặp | MICoDa shared-design fit với hai fixed iterations dùng **3 lstsq calls** (initial + 2), coefficients khớp per-response solve ở tolerance `1e-12`; LTS không còn rank-preflight kép, oracle toy pass. Không có claim runtime-budget full-data. |
| CQ7 — Hron exp/CLR round-trip | Stress với intercept balance 10.000 chạy qua nhiều sweep, finite output, ghi nhận 2 underflow parts; oracle Hron-LS pass. **Chưa tái hiện/chạy lại ca LTS full-data trước đây**, không tuyên bố đã giải quyết nguyên nhân lịch sử. |

## Đánh giá kiến trúc và thứ tự xử lý

Điểm tốt: numerical cores tách khỏi count/Other constraints; panel boundary che hidden values trước arithmetic; donor/metadata fit theo FitScope; pipeline chưa triển khai có guard; không có module nextgen vượt 1.000 dòng. Core lớn nhất là Tsagris 292 dòng, benchmark engine 468 dòng. Các tests về poisoning, observation lock, budget, metadata/split và 9 oracle nhỏ tiếp tục pass sau sửa.

Chỗ cần thay cấu trúc mạnh nhất là **ownership của fitted state, policy và artifact lifecycle**. Ba khái niệm này đang có nhiều biểu diễn và nhiều đường khởi tạo/validate. Thêm helper ở từng call site sẽ chỉ chuyển complexity đi nơi khác. Cần một state hoàn chỉnh cho estimator, một effective method specification và một artifact contract dùng chung ở mọi tầng completion.

Thứ tự khuyến nghị: xử lý R1/R2 trước; sau đó R3/R4 cùng một lần chỉnh boundary, rồi R5; xử lý R6/R7 trước khi chạy tiếp legacy experiments. Chuyển positive probes phù hợp vào unit suite thường trực trong đợt sửa, giữ fault-injection probes có fixture độc lập. Review này chỉ lưu probe trong harness, chưa tự sửa application source hay bổ sung vào test suite dự án.

Các kết quả MICoDa chưa hội tụ sau 50 vòng, LS/V1 vượt budget và LTS lỗi số học trong **run v2 đã đóng băng** vẫn giữ nguyên ý nghĩa. Source working tree mới không còn là source của accepted v2; các kết quả cũ không tự trở thành evidence cho bản sửa mới. Các hướng dẫn tái chạy phải dùng đúng frozen source hoặc một run mới đã đăng ký, không cập nhật hash cũ để lách kiểm tra.

Trạng thái integrity, checklist và handoff cuối review được ghi tại [checks](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/.harness/post-repair-review-2026-10-20/checks.md) và [verification](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/.harness/post-repair-review-2026-10-20/verification.json). Một review hoàn tất có thể kết luận **không phê duyệt**; test fail và finding còn mở không bị đổi thành pass trong evidence.
