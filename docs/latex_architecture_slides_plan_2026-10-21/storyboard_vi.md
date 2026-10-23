# Storyboard — 24 slide chính và 6 phụ lục

Tên slide là câu kết luận hoặc câu hỏi ngắn. Cột notes là phần giải thích bằng lời, không đưa nguyên văn lên trang. Đường dẫn nguồn bên dưới đều tính từ repo root; bằng chứng đã đọc được ghi thêm trong harness source inventory.

## Bài toán & phạm vi

| # | Tiêu đề / thông điệp | Hình chính và chữ hiện | Notes / căn cứ |
|---|---|---|---|
| 01 | Hiện thực hệ thống điền khuyết compositional | Cover mẫu, một thumbnail ma trận thiếu; phụ đề: Kiến trúc · Ba baseline · Đánh giá | Đặt tên dự án thay QSiMix; không tự gán tên tác giả |
| 02 | Điền phần thiếu, giữ nguyên phần đã biết | TikZ một hàng counts, ô observed/hidden/native-missing; thanh budget và Other | Counts không âm, tổng 17 targets ≤ N; Other không được suy ra từ nhãn đã che. `contracts.py`, `constraints.py`, `baselines/panels.py` |
| 03 | Hai nhánh code, một phạm vi báo cáo | Hai lane: legacy KNN→CLR→CSDI; nextgen core→adapter→benchmark. Neural scaffold nét đứt, nhãn CHƯA HIỆN THỰC | `README.md`, `nextgen/README.md`, `nextgen/src/longitudinal_impute/pipeline.py`; không nối chúng như cùng runtime |

## Kiến trúc codebase

| # | Tiêu đề / thông điệp | Hình chính và chữ hiện | Notes / căn cứ |
|---|---|---|---|
| 04 | Bốn lớp tách thuật toán khỏi dataset | Layer diagram: numerical cores / count adapter / benchmark / artifacts & report; arrows ghi call/data | `baselines/interfaces.py`, `adapters.py`, `benchmark/engine.py`; các box hỗ trợ ở bên thay vì tree dài |
| 05 | Một lượt chạy đi từ schema đến report | Sequence 6 bước: load→split/mask→fit→predict→constrain/score→persist/report | `csv_io.py`, `benchmark/protocol.py`, `masks.py`, `engine.py`, `report.py`; tách tune validation khỏi test |
| 06 | Nhãn đánh giá dừng trước model | TikZ trust boundary: O, H, C=O\H; EvaluationLabels rẽ sang scorer; model chỉ nhận ConditioningPanel | `contracts.py`, `panels.hide_targets/safe_counts/fit_subset`; Other unknown khi thiếu raw part; FitScope không tự chứng minh location-disjoint |
| 07 | Mỗi lớp sở hữu một loại state | Ownership matrix nhỏ: core donors/coefficients; adapter encoder/zero policy; benchmark split/mask/selection; artifact layer identity/hash | `metadata.py`, `adapters.py`, `registry.py`, `benchmark/artifacts.py`, `completion.py`; phân biệt config và fitted state |
| 08 | Core chung interface, khác miền dữ liệu | Interface diagram: fit(values,covariates)→self; predict_result→CoreResult; positive/nonnegative và row/population policy | `baselines/interfaces.py`, `common.py`, `registry.py`; scalar API không phải tensor neural B×T×D |
| 09 | Adapter chuyển proposal thành counts hợp lệ | Counts→c/N+Other→train-fitted transforms→core→proposal→budget/lock; 2 hàng ví dụ trước/sau | `adapters.py`, `panels.extended_proportions`, `constraints.py`; raw observed zeros không bị smoothing; validity không bằng accuracy |

## Hiện thực baseline

| # | Tiêu đề / thông điệp | Hình chính và chữ hiện | Notes / căn cứ |
|---|---|---|---|
| 10 | Hron: chọn donor theo tỷ lệ quan sát | 2D subcomposition sketch, donor ranking→scale adjustment→aggregate; nhãn Aitchison/kNN | `baselines/hron.py`; primary hron_knn. LS/LTS là nhánh phụ lục, không gán score kNN cho iterative |
| 11 | MICoDa: hồi quy ALR rồi cập nhật phần thiếu | Loop diagram ALR→covariates→regression→residual allocation→stop/max_iter | `micoda.py`, `metadata.py`; primary v2, W và max_iter rõ; observed-reference/missing-reference cases trong notes; 5/5 fits chưa hội tụ ở run accepted |
| 12 | Tsagris: donor JSD và alpha mean | Query simplex→JSD neighbors→alpha mean→allocate missing mass; inset alpha/k grid chọn bằng validation | `tsagris.py`, `benchmark/candidates.py`, config; primary tsagris_alpha; doubled JSD convention ghi trong notes |
| 13 | Cùng output, khác thông tin sử dụng | Bảng 3 hàng: domain / donor hay regression / metadata / policy khi không có anchor; mỗi cell vài từ | `registry.py`, `adapters.py`, accepted config; MICoDa có covariates, Hron/Tsagris không dùng regression metadata. So sánh là systems đã cấu hình, không cô lập tác động thuật toán thuần |

## Mở rộng & vận hành

| # | Tiêu đề / thông điệp | Hình chính và chữ hiện | Notes / căn cứ |
|---|---|---|---|
| 14 | Thêm baseline qua bốn điểm nối | TikZ 4 bước: NumericalCore→registry/recipe→candidate config→boundary checks; adapter/benchmark giữ nguyên nếu contract tương thích | `interfaces.py`, `registry.py`, `benchmark/candidates.py`, configs, tests; khác domain/tensor/output cần adapter/protocol mới, không hứa mọi neural model cắm trực tiếp |
| 15 | Fit thành công mới công bố state | State machine unfitted→candidate fit→committed fitted; fail→giữ state cũ; predict trước fit→typed error | Source hiện tại `adapters.py`, core fit và `errors.py`; `tests/test_fit_lifecycle.py`; đây là current behavior, không nói accepted run đã dùng chính bản sửa này |
| 16 | Một run là chuỗi artifact có định danh | prepare→validation/selection→test→final→report; inset fold/method/scenario/seed + result/prediction/attempt/marker | `benchmark/engine.py`, `driver.py`, `artifacts.py`, `completion.py`; marker chỉ có ý nghĩa khi children/identity/hash hợp lệ. Source/config đổi→run mới |
| 17 | Debug đi từ sai số về proposal | Truy vết: metric xấu→location/feature→job diagnostics→raw proposal→constraint repair; lane riêng domain/rank, unavailable/fallback, numerical, resource-limit | `errors.py`, `baselines/diagnostics.py`, engine failure_status/structured_error; proposal_trace của accepted run. Một ví dụ số dùng fixture minh họa, ghi nhãn, không giả là lỗi thật |

## Đánh giá & kết quả

| # | Tiêu đề / thông điệp | Hình chính và chữ hiện | Notes / căn cứ |
|---|---|---|---|
| 18 | Chọn tham số trước khi mở test | Fold bands 90 train/30 validation/30 test locations; 5 folds, 3 mask seeds, 4 mask pictograms | `offline_full_benchmark.json`, `benchmark/protocol.py`, accepted report; mask nominal 30%, whole-feature thực tế 34.911%; offline, không causal |
| 19 | Whole-feature: Tsagris có MAE point thấp nhất | Dot plot + CI95 theo location, pooled ratio MAE | primary_metrics.csv; 117,675 target qua 3 seeds, không phải observations độc lập; chỉ artificial hidden cells |
| 20 | Hiệu quả thay đổi theo kịch bản che | Dot-and-whisker 12 dòng chia 4 nhóm scenario; nhãn method trực tiếp và marker/màu cố định | MAE c/N × 100, CI95 bootstrap theo location; time-block/whole-row có fallback nhiều hơn ở Hron |
| 21 | Chênh lệch ghép cặp theo location | Forest plot 3 paired Δ location-macro MAE × 100 và đường 0 | Cùng location/mask/labels; CI95 mô tả với fixed fits, chưa hiệu chỉnh đa so sánh |
| 22 | Strata điểm và cấu trúc compositional khác nhau | Bảng zero/positive/rare-positive, temporal Δ, coverage/JSD; plot positive có method labels; strip hội tụ/fallback | Support phụ có giới hạn; MICoDA chưa hội tụ; native missing chưa có truth |

## Mở rộng đánh giá

| # | Tiêu đề / thông điệp | Hình chính và chữ hiện | Notes / căn cứ |
|---|---|---|---|
| 23 | Synthetic kiểm tra phần ground truth còn thiếu | Generator→known complete truth→mask mechanism→same benchmark→recovery; outline tiles temporal/zero/depth/correlation | CHỜ THÍ NGHIỆM. Đây là protocol đề xuất trong metrics_protocol_vi.md, không plot giả; giữ section có thể thay khi kết quả mới tới |
| 24 | Kiến trúc đã tách lớp; đánh giá còn các cổng tiếp theo | 3 cards: implemented runtime / observed evidence / pending synthetic & convergence; mini roadmap | Kết luận giới hạn ở accepted real-data. Không có một baseline thắng mọi scenario; source hiện tại và benchmark snapshot cần nối bằng parity hoặc run mới |

## Phụ lục — chỉ mở khi cần

| # | Nội dung | Hình / nguồn |
|---|---|---|
| A1 | Map thư mục→trách nhiệm + sơ đồ dependency chi tiết | `nextgen/src/longitudinal_impute/`; đường import không trộn với runtime data arrows |
| A2 | Hron LS/LTS, MICoDa V1, Tsagris basic/global/adaptive | Variant tree từ registry; supplementary terminal statuses; không ranking các fold thiếu |
| A3 | Metric definitions, support và unit of analysis | Công thức MAE/RMSE, macro weights, location bootstrap, CLR/ILR zero policy; `metrics.py` |
| A4 | Debug cookbook và lệnh CLI/benchmark | CLI runner, inspect-config, engine prepare/driver/final/report; chưa chạy lệnh mới trong lượt này |
| A5 | Full tables, compute budget và snapshot manifest | CSV/config/runtime evidence; source current khác accepted sources; không dùng số test pass cũ như check mới |
| A6 | Nguồn và giới hạn | Source paths, accepted report, local paper registry; chỉ thêm citation nội dung paper sau đọc đúng nguồn đầy đủ |

## Quy tắc nối source vào slide

Mỗi section `.tex` có comment source IDs và notes ngắn: thông điệp, giải thích hình, caveat, transition. Mỗi figure có manifest ghi input path/hash, script, metric key, unit, support và snapshot. Chú giải trên slide dùng tên module ngắn; full paths để phụ lục, không đặt đường dẫn Windows dài ở footer.
