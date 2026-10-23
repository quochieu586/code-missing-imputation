# Ticket 02 — Triển khai frame 11–20 theo plan mới

**Đầu ra:** mười frame đích 11–20 trong deck 30 trang.  
**Deck:** `presentation/implementation_report/main.tex`  
**Plan:** [PLAN_VI.md](PLAN_VI.md)  
**Prompt thực thi:** [PROMPT_TICKET_02_FRAMES_11_20_VI.txt](PROMPT_TICKET_02_FRAMES_11_20_VI.txt)

## 1. Hợp đồng phạm vi

Đây là ticket tiếp theo của plan chỉnh sửa, không phải yêu cầu đánh bóng mười trang hiện đang nằm ở vị trí 11–20. **Thứ tự đích là synthetic 11–13 → Hron 14–16 → MICoDA 17–20.** Tsagris bắt đầu ở trang 21 trong plan cuối và thuộc ticket sau.

Người thực thi sửa deck, compile, render và bàn giao. Người lập ticket không sửa slide ở bước chuẩn bị này.

- Đọc main.tex hiện tại, đếm frame theo thứ tự input; đừng suy số trang từ tên file.
- Chụp trạng thái/hash đầu vào mới nhất. Trang 1–10 có thể đang được sửa sau review Ticket 01: không phục hồi từ snapshot cũ, không sửa body hoặc style của chúng trong Ticket 02.
- Chỉ thay mười slot 11–20 trong main.tex bằng các section phù hợp. Giữ 30 frame active và giữ thứ tự/body của các slot hiện tại 21–30 trong ticket này. Chúng vẫn là phần chờ xử lý, không được gọi là đã khớp toàn bộ plan mới.
- Dùng section mới khi cần để tái dùng nội dung từ source cũ mà không sửa file đang được frame ngoài phạm vi input. Một source đã dùng ở frame 1–10 hoặc 21–30 phải được bảo vệ; có thể trích nội dung sang section mới.
- Giữ theme 16:9, macro và màu hiện có. Không sửa preamble global nếu nó làm đổi các frame ngoài phạm vi. Không xóa nguồn TeX legacy hoặc các plot/data đã có.
- Không sửa model, không benchmark, không generate dữ liệu. Có thể dựng hình từ saved arrays/masks/config và re-export diagram/plot đúng số liệu, có source map.

## 2. Storyboard và yêu cầu từng trang

### Frame 11 — Missingness và ba loại zero

**Thông điệp:** missingness là trạng thái che dữ liệu; zero là trạng thái truth khác.

**Visual:** heatmap mask MCAR/MAR trên cùng entity/time window; một dải ba ô phân biệt true zero (`p=0`), sampling zero (`p>0, X=0`), positive count (`X>0`). Mask phải có ký hiệu/overlay riêng, không dùng cùng màu với zero. Nếu dùng một crop từ data, ghi entity/time/component và chọn theo quy tắc cố định không dựa vào lỗi model.

**Nội dung:** ba declared patterns, mỗi pattern che 4/12 component; case_probability=.3, nominal cell rate=.1. Ghi realized support từ manifest khi trình bày số thực tế; không gọi đây là “30% cells missing”. MAR spec hiện dùng N và entity trait với trọng số khai báo; không vẽ latent p dẫn vào MAR như thể protocol đang chạy dùng hidden truth. Seed generator 2026 khác seed mask spec 2018; ghi seed đúng artifact, không nhập làm một.

**Nguồn:** `nextgen/SCHOUTEN_VI.md`; `nextgen/configs/schouten2018_mcar_patterns.json`, `schouten2018_mar_patterns.json`; manifests/conditioning masks trong `datasets/synthetic/wave_panel_v2/`; synthetic masking source. Bổ sung visual bằng dữ liệu lưu sẵn. Chưa có MNAR không được vẽ như kết quả đã chạy.

### Frame 12 — Protocol synthetic dùng chung cho ba baseline

**Thông điệp:** validation chọn cấu hình, test chỉ dùng báo cáo.

**Visual:** tận dụng split/select/refit từ `17_protocol.tex`, kết hợp public loader/private evaluator boundary đã có. Một sơ đồ chính; tránh ma trận nhiều ô gây che nhãn như Ticket 01.

**Nội dung:** standard entity-disjoint 36 train /12 validation /12 test; chọn candidate bằng validation hidden-cell sample_ratio MAE so X/N, tie theo candidate index. Refit selected model trên train+validation, sau đó test. Public: visible counts, mask, N, row IDs, allowed metadata; private: complete counts, latent p, strata chỉ cho evaluator. Cùng split/mask trong từng profile khi so model. Hash/row alignment phục vụ integrity và truy vết, không phải OS access control.

**Giới hạn quan trọng:** title/caption phải ghi synthetic. COVID engine hiện refit trên train, không dùng cùng nhãn train+validation để mô tả COVID. Không quay lại sửa trang 4 trong ticket này.

**Nguồn:** `scripts/run_synthetic_easy_baselines.py`, run manifests và `nextgen/SYNTHETIC_DATA_VI.md`. Đối chiếu source/run mới nhất trước khẳng định.

### Frame 13 — Metric: count, sample ratio và latent p

**Thông điệp:** cùng một prediction có các mục tiêu đánh giá khác nhau.

**Visual:** prediction tách tới count truth X và latent p; bảng ba hàng nhỏ tái cấu trúc từ `26_appendix_metrics.tex`, không mang số COVID sang synthetic.

**Toán:** định nghĩa rõ tập H là hidden scoring cells; trình bày MAE trên H:
- count: mean `|Xhat−X|`;
- sample ratio: mean `|Xhat/N−X/N|`;
- latent probability: mean `|Xhat/N−p|`.

RMSE có thể ghi là metric bổ sung nhạy với lỗi lớn, công thức vào notes nếu chật. Đơn vị count hoặc ratio/percentage points phải ghi nhất quán. Pooled n hidden không phải n mẫu độc lập; n=0 trả null. Counts MAE giảm theo N không tự chứng minh profile dễ hơn.

**Thống kê:** hiện một seed/profile; không vẽ CI/p-value hoặc nhận định significance khi chưa có artifact. Nhắc ngắn nếu ước lượng uncertainty sau này cần cluster entity và pairing, không bootstrap cells độc lập. Ticket này không tạo phân tích inference mới.

**Nguồn:** `synthetic/scoring.py`, plot README, metric_version/run manifests. Invariant checks finite/nonnegative/observed-preserved/budget là kiểm tính hợp lệ, không gộp thành metric accuracy.

### Frame 14 — Kiến trúc module Hron-kNN

**Thông điệp:** adapter giữ contract counts; Hron core giữ donor state và phép imputation.

**Visual:** tái dùng `09_hron_code.tex`, tách đường fit và predict: fit allowed training rows → fitted donor bank; query → donor eligibility/distance → proposal + diagnostics → adapter constraint → final counts.

**Nội dung:** class/method đúng source, k/variant được cấu hình, fitted state, FitScope và fallback đúng tầng. Không đưa query/test vào donor bank train. Không vẽ core nhận private labels. Không hard-code accepted k=10 cho tất cả run nếu selected_parameters từng run khác nhau.

**Nguồn:** `baselines/hron.py`, `adapters.py`, `interfaces.py`, registry, section cũ. Chỉ sửa slide; không refactor model.

### Frame 15 — Nền tảng Hron: CLR và khoảng cách Aitchison

**Thông điệp:** so donor trên observed subcomposition phù hợp.

**Visual:** giữ scatter/query-neighbors schematic từ `10_hron_mechanism.tex`; highlight tập anchors, không trình bày scatter như embedding dữ liệu thật.

**Toán:** closure/CLR và khoảng cách Euclidean trong CLR trên đúng tập anchors code sử dụng; định nghĩa geometric mean/ký hiệu ngay cạnh. Tối đa hai cụm toán chính. Giải thích yêu cầu positive và zero preprocessing thực tế từ adapter/source, không viết log(0) như hợp lệ.

**Nguồn:** `hron.py`, common CLR helper, adapter zero policy; citation Hron đã có trong source map/paper lưu cục bộ. Claim paper phải có nguồn, claim implementation phải theo code. Không thay bằng Euclidean trên raw counts cho dễ vẽ.

### Frame 16 — Hron: chọn donor, scale và median

**Thông điệp:** chọn donor phù hợp rồi scale target trước khi aggregate.

**Visual:** tách phần còn lại của `10_hron_mechanism.tex`; query anchors → eligible donors → top-k → scale → median proposal → adapter. Có thể dùng một ví dụ schematic nhỏ, ghi rõ không phải kết quả benchmark.

**Toán:** top-k eligibility + scale median observed/query-donor + median aggregation đúng configured variant. Đọc kỹ sequential prediction trong source: chỉ rõ observed/anchor set có thể được cập nhật ở bước nào; không đóng băng giả một tập nếu code đang cập nhật nó. Missing target candidates phải có anchors và target j theo donor policy.

**Giới hạn:** available-donor variant không yêu cầu donor hoàn chỉnh trên tất cả features. Fallback train_mean thuộc adaptation khi core không dự đoán được; không gán cho paper như thuật toán gốc. Output cuối qua constraint, không đồng nhất core proposal với final counts.

### Frame 17 — Kiến trúc module MICoDA-V2

**Thông điệp:** composition và design matrix đi vào coefficient state, prediction trả về composition proposal.

**Visual:** giữ/tinh giản `11_micoda_code.tex`: fit-scope/preprocessing → reference/design → coefficients + fitted state → predict/allocate → adapter.

**Nội dung:** variant V2, W, reference, max_iter/tolerance và β state theo source. Đặt state cạnh nơi sở hữu thay vì thêm một pipeline lặp. Interface hỗ trợ covariates không có nghĩa mọi run đã dùng covariates: synthetic runner hiện khai báo metadata_used=[]; kiểm encoder/design thực tế trước khi ghi nội dung feature. Không tự gán time/trait/N làm covariates cho run chưa dùng chúng.

**Nguồn:** `micoda.py`, metadata encoder, adapters, synthetic run manifest và section cũ. Không mô tả V2 core này thành nghiên cứu uncertainty multiple imputation đầy đủ.

### Frame 18 — Nền tảng MICoDA: ALR và hồi quy

**Thông điệp:** log-ratio biến composition thành bài toán regression theo design matrix.

**Visual:** giữ ý ALR → regression từ `12_micoda_model.tex`, bớt chữ; reference được thể hiện rõ trên composition nhỏ.

**Toán:** ALR với W và reference; dự đoán logit bằng design vector Q_i và β_j. Tránh dùng X vừa là counts vừa là design. Inverse ALR đầy đủ có thể đưa notes nếu không đủ chỗ; phân phối missing residual nằm ở frame 19, không trình bày closure toàn row như thể được phép đổi observed.

**Nguồn:** `_prepare`, `_regress` và helpers thật; reference selection/positive shift/initialization phải khớp source, không suy từ tên method. Citation MICoDA từ tài liệu có sẵn, không fabricate equation number.

### Frame 19 — MICoDA: residual mass và missing allocation

**Thông điệp:** chỉ phân mass còn lại trên các phần chưa biết, giữ phần đã biết.

**Visual:** tách allocation từ `13_micoda_update.tex`: observed cố định | residual; predicted missing logits → normalized weights → missing composition. Một thanh mass và một cụm phương trình đủ.

**Toán:** đúng shifted total/mass khi thêm W, ví dụ T*=T+DW theo định nghĩa chiều của code; residual trừ observed shifted parts; stable softmax trên missing set, reference logit và nhánh reference bị thiếu theo implementation. Không nhầm D named components của dataset với chiều extended core có Other; định nghĩa dimensionality ở notes.

**Giới hạn:** residual bên trong core là compositional mass, không phải private residual truth generator hoặc count budget N−sum counts. Sau core vẫn có adapter/constraint. Không làm mất nhánh reference handling chỉ để có công thức đẹp.

### Frame 20 — MICoDA: fit loop, stopping và convergence

**Thông điệp:** đạt max_iter khác với hội tụ theo tolerance.

**Visual:** tách vòng lặp từ `13_micoda_update.tex`; initialization chỉ một lần; predict/allocate → refit/update → delta → tiếp tục hoặc stop. Vẽ loop đúng thứ tự `_fit_in_place`/`_update` sau khi trace source; không dùng hình tiện tay có feedback về init.

**Nội dung:** iterations, delta, tol, max_iter và converged gắn với diagnostics. Nếu dùng accepted COVID “5/5 folds chạm cap=50”, phải đối chiếu artifact đúng run trước chèn và ghi rõ COVID; không suy sang synthetic. Nếu chưa xác minh artifact thì giữ warning chung và ghi unresolved trong evidence, không bịa số hoặc nói đã hội tụ.

**Nguồn:** `micoda.py`, selected parameters/fit diagnostics trong run accepted; các section cũ. Không thay W/tol/max_iter hoặc chạy lại mô hình để làm báo cáo đẹp hơn.

## 3. Yêu cầu visual và cấu trúc TeX

- Mỗi trang một thông điệp, một hình chính, một takeaway. Tiếng Việt ngắn gọn; identifier/API giữ đúng tiếng Anh. Tối đa hai cụm phương trình chính, giải thích ký hiệu gần công thức.
- Tái dùng style/diagram có sẵn. Chỉ thêm visual missingness; các baseline chủ yếu tách trang và làm rõ. Hình schematic phải có nhãn schematic/minh họa.
- Không lặp lại bug Ticket 01: minimum width/height không ràng buộc kích thước thực của node. Dùng text width, anchors, relative positioning và gap có chủ đích; kiểm inner sep trong bounding width.
- Arrowhead phải có khoảng trống, không bị hộp che hoặc trông đảo chiều. Lane labels/notes ngoài vùng node, không có text vượt panel.
- Không dùng shrink/scale toàn trang, font li ti hoặc offset âm để che lỗi. Không tạo một framework macro tổng quát chỉ để vẽ mười hình.
- Frame tách ra nên là section đơn nhiệm vụ, tên rõ; không thêm conditional mode để dùng chung một file cho nhiều slide khác nội dung. Giữ source legacy để tra cứu.

## 4. Trình tự và bằng chứng bàn giao

1. Đọc skills phù hợp, plan, review Ticket 01 và source. Snapshot main/input map, first10 và suffix21–30 mới nhất; ghi git status để bảo vệ thay đổi người dùng.
2. Viết mapping slot hiện tại → frame đích11–20 và danh sách file được sửa. Dùng inputs mới nếu source cũ được dùng ngoài phạm vi.
3. Triển khai hình và nội dung theo thứ tự11–20. Claim-source map phân biệt method từ paper, variant code và benchmark adaptation.
4. Compile bằng XeLaTeX theo project. Render frame11–20 ≥135dpi, contact sheet đúng10 trang; mở từng ảnh sau final build. Visual pass không được suy từ log no-overfull.
5. Sửa có bằng chứng, tối đa ba lần cho cùng lỗi; lưu log/vấn đề còn lại nếu chưa đạt. Chỉ rerun các kiểm tra phụ thuộc vào thay đổi.
6. So hashes/diff protected sources và main input prefixes/suffixes để chứng minh 1–10/21–30 không bị sửa. Đừng ghi đè snapshot cũ để làm pass.
7. Bàn giao main.tex/PDF30, render11–20, contact sheet, source map, file diff, build log, checks/progress của harness Ticket02. Hash PDF/render final được kiểm.

## 5. Acceptance

- Đúng 30 frame active; mười slot11–20 đúng storyboard nêu trên, không nhầm với mười frame legacy đang ở vị trí đó.
- First10 và body/order của current suffix21–30 được giữ nguyên; không tự nhận suffix đã theo plan cuối.
- Trang11 không lẫn missing/zero hoặc case probability/cell rate; trang12 đúng synthetic protocol; trang13 đúng estimand/support và giới hạn một seed.
- Hron3 trang và MICoDA4 trang có module architecture, kiến thức/công thức và cơ chế cụ thể; không gán adaptation/fallback cho paper gốc.
- Không claim covariate use, selected params hoặc convergence khi chưa có bằng chứng đúng run.
- Compile thành công, mỗi trang11–20 được xem bằng mắt ở kích thước đủ lớn; không chồng/cắt/che chữ hay arrowhead.
- Không sửa model/data/accepted metrics, không chạy benchmark/generate; không thêm p-value/CI/seed results chưa có.
- Trạng thái hoàn thành chỉ áp dụng Ticket02. Báo rõ còn lại: repair Ticket01 nếu chưa xong và Ticket03 cho đích21–30.
