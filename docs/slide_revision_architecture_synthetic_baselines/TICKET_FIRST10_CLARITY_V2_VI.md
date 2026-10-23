# Ticket — Cô đọng và giải thích rõ 10 trang đầu, thay debug bằng capability summary

**Đây là cập nhật yêu cầu cho first10, ưu tiên hơn storyboard slot7 cũ.**

Đọc `REVIEW_FIRST10_CLARITY_V2_VI.md`. Giữ chín trang 1–6,8–10 và các visual hữu ích; biên tập chúng để người ngoài đọc hiểu. Loại nội dung “metric bất thường” khỏi slot7, thay bằng summary chức năng đã hiện thực/tùy chỉnh. Vẫn10 trang đầu trong deck30. Không chạm phần11–30 hoặc công việc Ticket02.

## Quy tắc nội dung

1. Mỗi trang trả lời một câu hỏi, title ngắn. Formal academic terms dùng trong hình; ngay dưới có **một câu tiếng Việt giải thích ý nghĩa**, không chỉ định nghĩa tên lớp hoặc legend mũi tên.
2. Tên khối trên hình = formal term + tối đa một dòng vai trò ngắn. Không kèm cả class, method signature, state list và policy trong cùng node.
3. Caption thường15–30 từ, có thể tới35 khi cần đúng nghĩa; tránh đoạn giải thích dài. Gọn bằng bỏ ý lặp, không bằng giảm font. Identifier dài, source paths, run IDs và caveat kỹ thuật chi tiết vào notes/companion; giữ limitation làm thay đổi cách hiểu ngay trên trang.
4. Giữ nguồn, số và plot đang đúng. Ví dụ số tự tạo phải ghi “minh họa”; không trình bày như prediction benchmark.

## Nội dung đích và câu giải thích mẫu

### 1 · Bìa
Giữ hình mask và ba baseline. Subtitle có thể dùng “Điền các số đếm còn thiếu theo thời gian”. Không thêm định nghĩa học thuật lên bìa. Từ compositional có thể giữ trong subtitle phụ nếu cần, không bắt người xem giải mã ngay.

### 2 · Hệ thống nhận gì và trả gì?
Giữ counts, ba trạng thái ô và hai nguồn đánh giá. Thay các định nghĩa lặp bằng một ví dụ: một entity/ngày có A=20, B=?, C=10, tổng N=100. Mục tiêu điền B nhưng giữ20 và10. Không đưa ví dụ này như full dataset hoặc công thức sinh.

Caption mẫu: **“Hệ thống dự đoán các số còn thiếu, giữ nguyên số đã biết và không vượt tổng của mẫu.”**

C/H/M để một legend gọn; số lượng N giải thích là tổng số đếm của một mẫu. COVID là dữ liệu thực, synthetic biết truth để kiểm tra prediction. Không thêm glossary dài.

### 3 · Codebase gồm những phần nào?
Tận dụng hình architecture, làm rõ4 nhóm: Data/Contracts; Benchmark Orchestration; Model Subsystem (Adapter + interchangeable Core); Evaluation/Reporting. File/package nhỏ đặt dưới nhóm tương ứng, không vẽ cây thư mục toàn repo.

Formal labels gợi ý: **Data Contract; Experiment Orchestrator; Model Adapter; Numerical Core; Evaluation Service**. Tên hàm chỉ khi nó giúp tìm source.

Caption mẫu: **“Dữ liệu dùng chung một định dạng; bộ chạy chọn mô hình, còn evaluator chấm dự đoán với đáp án giữ riêng.”**

Sửa C1: core proposal trở lại adapter/constraint trước final counts đưa tới evaluator. Không có đường bypass. Nếu nhiều mũi tên, group adapter+core thành subsystem để chỉ một output. Registry thuộc phần chọn/tạo model, không cần vẽ thành một stage dữ liệu riêng.

### 4 · Một lượt chạy diễn ra thế nào?
Giữ nội dung runtime nhưng rút ma trận12 ô thành4 stage: **Prepare → Fit & Select → Refit → Predict & Evaluate**. Dưới mỗi stage1 câu công việc tiếng Việt ngắn. Input và output dễ thấy, test truth tách tới evaluator.

Caption mẫu: **“Chọn tham số bằng validation, sau đó mới dự đoán test; đáp án test chỉ được dùng khi chấm điểm.”**

Refit annotation nhỏ: COVID=train, synthetic=train+validation. Giữ khác biệt đúng source; không sửa code hoặc mở rộng sửa trang12 ở Ticket02.

### 5 · Vì sao cần adapter?
Giữ pipeline count↔composition và constraint; bỏ các định nghĩa tập hợp lặp ở trang2 nếu không cần. Formal labels **Count Encoding → Numerical Estimation → Constraint Projection → Final Counts**; thuật ngữ Projection chỉ dùng như nhãn ràng buộc, không tuyên bố là phép chiếu tối ưu Euclidean nếu source không làm vậy. Có thể dùng **Budget Enforcement** để sát code hơn.

Ví dụ nối trang2: A20,C10 trong tổng100 → phần chưa phân bổ70, chia cho B và Other; không ép B=70. Nếu muốn cho một số Bhat25/Other45, ghi rõ minh họa, không phải kết quả model.

Caption mẫu: **“Adapter đổi counts thành tỷ lệ cho model, rồi đưa kết quả về counts và giữ nguyên các số đã quan sát.”**

Giải nghĩa Other một lần: phần trong tổng N ngoài các thành phần liệt kê. Other không đồng nghĩa missing rate. Constraint vẫn đảm bảo named sum≤N.

### 6 · Đổi model và thêm model mới như thế nào?
Giữ slide plug-and-play nhưng bỏ hai signatures dài khỏi hình chính. Hai hàng ngắn:
- **Existing model:** chọn model key + params → cùng adapter/runner. Ví dụ hron_knn,k=5 là cấu hình minh họa, không nói selected best.
- **New model:** implement NumericalCore → đăng ký factory/name → dùng adapter/kiểm contract phù hợp.

Một dòng contract đủ: fit/predict_result + khai báo input domain/prediction policy. Ownership chi tiết vào notes, không cần4 hộp cạnh tranh nội dung.

Caption mẫu: **“Đổi tên model trong cấu hình để dùng baseline khác; model mới cần tuân theo cùng giao diện và được đăng ký.”**

Không claim auto plugin discovery hoặc một dòng config chạy được mọi model lạ. Adapter giữ constraints; không chấm core proposal trực tiếp.

### 7 · Đã hiện thực những gì và chỉnh ở đâu?
**Bỏ hoàn toàn nội dung debug/metric bất thường khỏi active slot7.** Tạo bảng booktabs3 cột, tối đa6 hàng; không bọc bảng trong TikZ hay tạo mũi tên không cần thiết.

| Chức năng | Chỉnh gì? | Qua đâu? |
|---|---|---|
| Chọn baseline | Model và k/W/α theo model | Config + registry |
| Sinh dataset | Profile, quy mô, seed | CLI |
| Động lực dữ liệu | Dạng sóng, tương quan, residual | Python API |
| Tạo missingness | MCAR/MAR, pattern, xác suất, seed | JSON spec |
| Adapter policy | Metadata, zero/fallback phù hợp | API / config phần được expose |
| Tái sử dụng dữ liệu | Conditioning, truth, manifest để chấm | Exporter/loader |

Nếu hàng cuối làm bảng quá dày, chuyển nó thành một dòng output bên dưới và giữ5 hàng. Không gọi exporter output là tham số tùy chỉnh. Capability matrix trong review là source cho bảng, không tự bổ sung chức năng chưa có.

Caption mẫu: **“Chọn dữ liệu và model bằng cấu hình; các tham số sinh dữ liệu nâng cao được chỉnh qua API.”**

Ghi nhỏ giới hạn quy mô:30/60/120 entities, luôn120 thời điểm×12 component. Dòng này có thể chuyển sang trang8 hoặc10 nếu chưa đủ chỗ. Fixed-wave nằm companion nếu cần, không giả định đã chạy. Không đưa MNAR chưa hỗ trợ vào danh sách ready.

### 8 · Synthetic dataset gồm gì và dùng để làm gì?
Giữ boundary đúng mới sửa, thêm định nghĩa mẫu:1 hàng=1 entity ở1 thời điểm;12 cột thành phần là các counts cần điền. p là tỷ lệ nền, X là counts đã sinh, N là tổng; trait là covariate theo entity nếu cần đưa notes.

Giữ hai nhánh: bản che cho model; bản đầy đủ cho evaluator. Có thể dùng nhãn **Conditioning Data / Reference Truth / Prediction / Evaluation**, class name vào notes. Một dải size/seed ngắn, không lặp toàn config.

Caption mẫu: **“Giữ một bản đầy đủ để chấm, rồi che một phần dữ liệu trước khi đưa cho model.”**

Một seed/profile là giới hạn đánh giá, không suy rộng ranking. Đừng làm câu caption chính thành danh sách các việc chưa có.

### 9 · Dataset được tạo qua ba bước nào?
Giữ3 panel và phương trình đúng. Formal labels: **Latent Temporal Dynamics → Compositional Normalization → Multinomial Sampling**. Mỗi panel thêm1 dòng tiếng Việt:
1. Tạo mức xuất hiện từng thành phần theo thời gian.
2. Chuyển mức xuất hiện thành tỷ lệ, có phần Other.
3. Lấy mẫu counts từ tỷ lệ với tổng N.

Giải thích gọn w=mức trước chuẩn hóa; p=tỷ lệ nền; r=phần Other; N=tổng counts. Tối đa2 công thức đang có, chi tiết AR(1) vào notes nếu làm panel quá dày. Sóng hand-drawn phải có nhãn minh họa.

Caption mẫu: **“Sinh tỷ lệ nền theo thời gian, dùng tổng N để lấy mẫu counts, rồi mới áp mask cho bài toán điền khuyết.”**

Không ghi Schouten như tác giả wave generator; attribution đặt footer/notes có source, không chiếm câu giải thích chính.

### 10 · Các profile thay đổi dữ liệu ra sao?
Giữ ba plot hiện có từ saved arrays và cùng axes; caption/source ID dài rút xuống notes. Legend tiếng Việt: **p: tỷ lệ nền; X/N: tỷ lệ sau lấy mẫu**. Nếu nét dashed đang được biểu diễn bằng legend line liền tự vẽ, dùng legend line đúng style của plot để không gây hiểu nhầm.

Giữ config thành dải3 cột ngắn hoặc đưa bảng đầy đủ companion, không viết cả5 tham số nối thành paragraph. Có thể nêu cùng scale nhưng N giảm và timing/noise thay đổi theo preset. Không kết luận hard có error luôn cao hơn.

Caption mẫu: **“Các profile đổi độ biến động và tổng N; đường quan sát X/N dao động quanh tỷ lệ nền p.”**

Chỉ là một entity/component minh họa, không chứng minh toàn population hoặc ranking model. Không chọn lại sample chỉ để làm hình “đẹp”.

## Acceptance cho người triển khai

- Source hiện tại được chụp hash trước sửa; không đụng frame11–30 hay dữ liệu/thuật toán/accepted metrics. Giữ30 trang.
- Slot7 không còn bất kỳ nội dung debug/anomaly nào; thay bằng bảng capability bám source hiện thực.
- Mỗi trang2–10 có câu tiếng Việt trả lời câu hỏi của trang; legend kỹ thuật không được tính thay cho giải thích.
- Hình formal nhưng người đọc không phải biết code mới hiểu nhiệm vụ các khối. Không có core→evaluator bypass; plug-and-play chỉ rõ dùng model sẵn và thêm model mới.
- Dataset có entity/time/component, p/X/N/r được giải nghĩa trước khi dùng; bảng phân biệt CLI/API/JSON và giới hạn T/D.
- Compile/render first10≥135dpi và xem từng trang sau final build. Không chồng text/node, không nhét chữ nhỏ hoặc hình scale xuống để chữa lỗi. Hash PDF/render final.
- Kiểm đọc độc lập: ghi trong evidence câu trả lời lấy trực tiếp từ slide cho6 câu: input gì? output gì? adapter làm gì? đổi/thêm model ở đâu? data sinh thế nào? chỉnh những gì qua đâu? Đây là checklist của người thực thi, không claim user study nếu không có người tham gia.
- Bàn giao source map/capability matrix companion, PDF/contact sheet10 trang, findings→changes→evidence; không ghi pass nếu còn câu hỏi không thể trả lời bằng nội dung trên slide.
