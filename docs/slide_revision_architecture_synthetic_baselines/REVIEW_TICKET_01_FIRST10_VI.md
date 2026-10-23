# Review nghiêm ngặt — Ticket 01, chỉ frame 1–10

**Verdict: chưa đạt nghiệm thu. Cần sửa trước khi đi tiếp.**

Phạm vi: mười frame đầu đang active trong `presentation/implementation_report/main.tex`. Không đánh giá hai mươi frame còn lại. Không sửa slide hay thuật toán trong lần review này.

Áp dụng tiêu chuẩn thermo-nuclear-code-quality-review cho tổ chức TeX, độ rõ của abstraction/boundary và chi phí bảo trì, đồng thời kiểm tính đúng của hình/claim. Không áp máy móc tiêu chí 1.000 dòng: các section đều ngắn, vấn đề nằm ở cấu trúc layout và biểu diễn boundary.

## Bằng chứng đã đọc/xem

- Đọc main.tex và toàn bộ 10 section active; đối chiếu plan và ticket.
- Xem render cả 10 trang. Trang 1–6 được render lại từ PDF hiện tại bằng Poppler ở 135 dpi vào `review_ticket_01/page-01.png` … `page-06.png`; trang 7–10 dùng focus PNG hiện có, đọc đối chiếu TeX.
- Đọc `baselines/interfaces.py`, `registry.py`, `adapters.py`, `panels.py`, `constraints.py`, engine COVID, runner synthetic và generator.
- Đọc source map và visual_qa.md của ticket. Không chạy benchmark hay thay đổi dữ liệu.

## Findings ưu tiên

### R1 · P1 — Layout TikZ có lỗi hệ thống; chưa thể trình chiếu

**Vị trí:** `01_problem.tex:5–9,18–21`; `04_runtime.tex:14–37`; `05_boundary.tex:18–25`; `06_ownership.tex:10–23`; `ticket_01_frame07_artifacts.tex:4–17`; `ticket_01_frame09_generator.tex:8–41`.

Bằng chứng từ render:
- Trang 2: ba hộp O/H/M lấn nhau; phần định nghĩa chạm đáy hộp; nhãn BA CORE đè Hron-kNN.
- Trang 4: các hộp của cột INPUT che DATA/MODEL/EVALUATOR; lane labels bị cắt. Đây là chồng đối tượng trong hình, không phải chỉ font nhỏ.
- Trang 5: hộp công thức che OUTPUT LOCK; câu ràng buộc hàng bị hộp khác phủ; chữ “allocate” xuống dòng thành “al- locate” trong node.
- Trang 6: “selected core” và “composition” đè cả node. Hàng ownership không có khoảng hở, phần chữ đầu bị hộp kế tiếp che.
- Trang 7: các đầu mũi tên bị ép giữa hộp gần chạm nhau; một số đầu tên nhìn chạy sang trái dù source vẽ `.east → .west`. Hình không còn truyền đạt rõ chiều trace.
- Trang 9: chú thích latent dynamics, residual category và counts tràn xuống dưới các panel; phần named-p/r quá sít; title chiếm hai dòng làm giảm diện tích hình.

**Nguyên nhân cấu trúc:** nhiều node dùng minimum width/height và tọa độ tâm hard-code; text width chưa kiểm soát hoặc cộng thêm inner sep làm hộp lớn hơn khoảng giữa các tâm. Compile không báo overfull không phát hiện được các node vẽ đè nhau.

**Yêu cầu sửa:** thiết lập từng hàng bằng anchors/relative positioning, text width xác định, khoảng hở đủ cho arrowhead; tách header/lane label khỏi vùng node. Chọn một quy ước kích thước/gap dùng lại cho các hình thực sự giống nhau. Không tạo framework diagram tổng quát mới, không chữa bằng thêm hàng loạt offset âm hoặc shrink toàn hình.

### R2 · P1 — Trang 8 biểu diễn sai boundary model/evaluator

**Vị trí:** `23_synthetic.tex:16–20,30`.

Hộp “TRUTH / EVALUATOR ONLY” chứa “Count truth X,N”, làm người xem hiểu N cũng riêng tư. Model thực tế cần biết N để tạo composition và phục hồi counts; `load_conditioning` trả totals. Đồng thời takeaway nói “models … score hidden cells”, gán trách nhiệm scorer cho model.

Đường `generator → mask → baselines` cũng thiếu public/private branch: generator box có p/strata, nhưng không thể hiện chúng bị chặn khỏi model.

**Yêu cầu sửa:** vẽ conditioning công khai chứa visible counts, mask, N, metadata → adapter/model → predictions; private complete counts, latent p, strata → evaluator. N được cung cấp công khai cho cả model/evaluator, không đặt dưới nhãn “only”. Hidden truth không có đường vào model. Sửa câu: model dự đoán; evaluator chấm hidden cells. Có thể rút bỏ hai hộp truth hiện tại để thay bằng hai nhánh rõ ràng, không thêm một lớp wrapper.

### R3 · P2 — Trang 4 gọi là runtime tổng quát nhưng refit chỉ đúng COVID

**Vị trí:** `04_runtime.tex:21,28,42`.

“refit winner / train only” phù hợp `benchmark/engine.py` (selected candidate được fit lại trên train). Tuy nhiên `scripts/run_synthetic_easy_baselines.py:86,131–132,197` fit model cuối trên train+validation. Bài trình bày hai nguồn dữ liệu nên không thể dùng một nhãn không giới hạn để mô tả cả hai.

**Yêu cầu sửa:** ghi rõ đây là protocol COVID hoặc giữ một runtime chung với một annotation duy nhất ở bước refit: COVID: train; synthetic: train+validation. Cả hai đều không dùng test để chọn model. Không sửa engine/runner để ép hình thành đúng. Đây là khác biệt thật, không phải thiếu chi tiết vô hại.

### R4 · P2 — Trang 5 không nối constraint vào output và đổi nghĩa ký hiệu M

**Vị trí:** `05_boundary.tex:4–14,18–27`; đối chiếu `01_problem.tex:5–9`.

Pipeline dừng ở “Count proposal N p-hat”; observed lock và budget nằm rời ở dưới. Người xem không thấy proposal phải qua constraint trước khi trở thành final counts. Trang 2 dùng M=thiếu gốc, trang 5 lại dùng hidden M cho toàn phần chưa quan sát; ký hiệu bị đổi nghĩa ngay trong một báo cáo.

Source `adapters.py` gọi `CountBudgetConstraint.apply` sau proposal và trả `repaired.counts`; adapter có D+1 proposal, phân residual chung cho missing targets và Other.

**Yêu cầu sửa:** nối rõ proposal → observed lock + budget allocation → final counts. Dùng một định nghĩa tập hợp xuyên suốt, ví dụ C=conditioning-visible, H=artificial hidden, M=native missing, U=H∪M. Budget R=N−sum_C X. Khẳng định named counts ≤N; không ép named parts bằng N. Giữ đúng Other chỉ derive được nếu named parts đầy đủ.

### R5 · P2 — Trang 3 thiếu architecture contract và trộn ngữ nghĩa mũi tên

**Vị trí:** `03_architecture.tex:4–25`.

Data → Driver → Engine → Scorer dùng dạng flow, Engine → Registry/Adapter lại giống calls/dependencies, nhưng không có legend cho semantics. Data contract chỉ ghi “CSV/schema/panels”, chưa có ConditioningPanel/EvaluationLabels hoặc nhánh synthetic. Vì vậy hình chưa giải thích được boundary cốt lõi đã cam kết trong plan và vẫn một phần lặp runtime trang 4.

**Yêu cầu sửa:** chọn rõ hình trách nhiệm/calls với legend ngắn; data sources COVID/synthetic cung cấp cùng ConditioningPanel; EvaluationLabels chỉ thuộc evaluator; engine gọi registry/adapter/scorer và artifacts nhận outputs. Tránh thêm một hàng pipeline nữa. Tận dụng khối cũ, giảm khái niệm lặp giữa Data contract/Driver/Engine nếu không đủ chỗ.

### R6 · P2 — Trang 10 mới là bảng config, thiếu minh họa profile theo ticket

**Vị trí:** toàn bộ `ticket_01_frame10_profiles.tex`, nhất là dòng 16–42.

Số config khớp source; chưa phát hiện số bịa. Tuy nhiên ticket yêu cầu minh họa từ arrays/config đã lưu, cùng axes/thang và định danh entity/seed. Một bảng chỉ cho biết tham số thay đổi, chưa cho thấy wave/composition/count observations biến đổi thế nào. Lý do “mỗi hàng khác đơn vị” chỉ giải thích việc không vẽ chung trục, không loại trừ small multiples đúng đơn vị.

**Yêu cầu sửa:** giữ bảng như dải config nhỏ hoặc chuyển phần đầy đủ vào notes; thêm ba panel cùng quantity/axes từ bundle v2 có sẵn. Chọn entity/component theo quy tắc cố định trước khi xem lỗi model, ghi IDs/seed; chỉ tải arrays, không sinh data hay fit model. Có thể chọn cùng entity và một vài component để đường không quá dày; phân biệt latent p với X/N. Nếu ưu tiên một plot thay vì hai loại, latent p theo thời gian là đủ, N ghi trong dải config.

### R7 · P2 — Visual QA cũ đã chấp nhận các lỗi còn nhìn thấy

**Vị trí:** `.harness/ticket-01-first-10-frames-2026-10-23/evidence/visual_qa.md`; `checks.md`.

QA ghi trang 4 lanes separated, trang 9 labels không va nhau, và nhìn chung không clipped. Render hiện có mâu thuẫn trực tiếp với các nhận định đó. “No overfull boxes” chỉ là build check, không xác nhận TikZ không chồng đối tượng. Có hash và record consistent không đủ chứng minh slide đạt visual quality.

**Yêu cầu sửa:** khi làm ticket repair, giữ evidence cũ và ghi rõ bị review bác ở R1/R2; render lại đủ frame 1–10 ở ≥135 dpi sau bản build cuối, xem từng trang và cập nhật verdict đúng bằng chứng. Hash PDF/render để tránh QA một phiên bản rồi giao phiên bản khác. Không cần kiểm hay nhận xét trang 11–30.

### R8 · P2 — Trục thời gian ở bìa bị ghi thành location

**Vị trí:** `00_cover.tex:20–24,33–34`.

Cột được gắn t1…t4, hàng j1…j4 nhưng mũi tên ngang ghi “location”. Đây là lỗi ngữ nghĩa trực quan ngay trên bìa.

**Yêu cầu sửa:** đổi nhãn trục ngang thành thời gian/t, hàng là component/j; nếu muốn biểu diễn entity thì dùng dimension riêng có chú thích. Không vẽ một trục có hai nghĩa.

## Nhận xét giới hạn

- Phương trình normalization và multinomial trang 9 khớp generator đã đọc; bảng profile trang 10 đúng PROFILES. Không có căn cứ gọi hai phần đó là hallucination số liệu.
- Trang 9 ghi “sơ đồ sóng khái niệm” nên đường hand-drawn không bị xem là dữ liệu thật; tuy vậy nếu vẽ lại, nên dùng curve đơn giản đúng onset/peak/tail, tránh thêm chi tiết không giúp hiểu generator.
- Không phát hiện file >1.000 dòng hoặc if-spaghetti trong các section. Cơ hội đơn giản hóa là bỏ việc cân chỉnh thủ công lặp và làm rõ một output/boundary trên mỗi hình; không đề nghị refactor code thuật toán.
- Chưa approve mười trang đầu vì R1/R2 là blockers. Các vấn đề khác là yêu cầu sửa có phạm vi rõ, không phải đề nghị thiết kế lại deck.
