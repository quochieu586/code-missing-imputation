# Review lần 2 — 10 trang đầu: dễ hiểu, cô đọng và đúng kiến trúc

## Verdict và phạm vi

**Chưa đạt mục tiêu người ngoài đọc hiểu độc lập.** Bản mới đã sửa nhiều lỗi hình học và boundary của review trước; không tiếp tục dùng các lỗi đã hết để kết luận. Vấn đề chính hiện nay là thiếu ý nghĩa nghiệp vụ: nhiều identifier/API, ít lời giải thích input, output và lợi ích của từng khối.

Theo yêu cầu mới: bỏ nội dung “Lần metric bất thường theo run artifact” ở slot 7; thay bằng bảng chức năng/tùy chỉnh đã có. Giữ chín trang khác, chỉnh lời và hierarchy của hình. Không thêm trang; giữ 10 slot đầu để không làm lệch Ticket 02. Không review phần 11–30.

Review áp dụng thermo-nuclear-code-quality-review theo hướng loại bỏ khái niệm và nhánh giải thích không giúp người xem. Không có căn cứ yêu cầu refactor thuật toán hoặc tạo framework TikZ mới. Người review chỉ tạo tài liệu và render, không sửa TeX/PDF.

## Bằng chứng mới

- Đã đọc main.tex và 10 source active hiện tại, không dùng snapshot TeX cũ.
- Render lại trực tiếp trang 1–10 từ `presentation/implementation_report/main.pdf` tại 135 dpi: `review_first10_clarity_v2/page-01.png` … `page-10.png`; đã xem từng ảnh.
- Đã đối chiếu `contracts.py`, `baselines/interfaces.py`, `registry.py`, `adapters.py`, `panels.py`, `constraints.py`, `metadata.py`, benchmark engine, synthetic generator/CLI/service/masking/capabilities và script trích plot profile.
- CSV profile hiện có: các cực đại p và X/N được kiểm nằm dưới ymax .55 của plot, không thấy bằng chứng clip đỉnh do trục. Không chạy generator/benchmark hoặc fit model.

## Findings

### C1 · P1 — Trang 3 cho kết quả core đi thẳng tới evaluator, bỏ qua bước bảo đảm counts

**Vị trí:** `sections/03_architecture.tex:17–18,31`. Mũi tên `core.south → scorer.north east` biểu diễn core trực tiếp cấp input chấm điểm. Core thực tế trả proposal composition; `adapters.py:117–132` nhân N, gọi `CountBudgetConstraint.apply` rồi mới trả final counts cho engine/evaluator.

Đây là rò rỉ abstraction boundary trong hình: trang 5 nói adapter ràng buộc output, trang 3 lại cho đường đi vòng. Cần để predictions quay về adapter/output boundary trước scorer, hoặc gom adapter+core thành model subsystem rồi chỉ vẽ một output final counts ra ngoài. Cách thứ hai bớt đường nối, không làm tăng số khái niệm.

### C2 · P2 — Trang 2–6 nêu từ chuyên môn trước khi nêu công việc chúng thực hiện

**Vị trí:** `01_problem.tex:7–26`; `03_architecture.tex:5–18,34`; `04_runtime.tex:18–37`; `05_boundary.tex:5–27`; `06_ownership.tex:4–27`.

Những từ như conditioning-visible, FitScope, candidate, registry factory, composition, Other, covariates không giúp người ngoài trả lời “đầu vào là gì, hệ thống làm gì, đầu ra dùng làm gì”. Caption trang 3 chỉ giải thích nét liền/nét đứt, chưa giải thích kiến trúc. Trang 2 lặp C/H/M ở cả hộp lẫn ba dòng định nghĩa, vừa tốn chỗ vừa không có ví dụ cụ thể.

**Sửa cấu trúc thông tin:** hình có thể dùng thuật ngữ formal, nhưng mỗi node chỉ 1 tên + 1 chức năng; ngay dưới hình có một câu tiếng Việt nói hệ thống làm gì. Dùng một ví dụ nhỏ xuyên trang 2/5 thay cho nhiều định nghĩa lặp. Đưa identifier dài vào notes/source map. Không biến caption thành danh sách glossary.

### C3 · P2 — Trang 6 chưa chứng minh được plug-and-play

**Vị trí:** `06_ownership.tex:4–25`. Hai API signatures, ba registry keys và bốn ownership boxes chưa cho biết đổi model sẵn có và thêm model mới khác nhau thế nào.

Code không auto-discover plugin. `registry.py` có METHODS và constructor dispatcher tường minh; `NumericalCore` cần fit/predict_result cùng input_domain/prediction_policy. Model mới phải thực hiện contract và đăng ký; adapter không bảo đảm tương thích mọi model bất kỳ.

**Sửa:** minh họa hai trường hợp: chọn tên + params để đổi baseline đã có; implement interface + register constructor để thêm baseline mới. Chỉ một output invariant: giữ quan sát và đúng ngân sách N nhờ adapter. Không vẽ registry như tự động tải model từ folder, không quảng cáo “thêm file là chạy”.

### C4 · P2 — Trang 7 không phục vụ câu chuyện người dùng đang cần

**Vị trí:** toàn bộ `ticket_01_frame07_artifacts.tex`. Run identity, metric slice, fallback rows và repair L1 là tài liệu vận hành, không làm rõ kiến trúc hoặc cách tạo/tùy chỉnh dataset. Người dùng đã yêu cầu bỏ trang này.

**Sửa:** remove nội dung này khỏi active slot 7, thay bằng bảng 5–6 hàng “Chức năng | Điều chỉnh gì? | Qua đâu?”. Giữ source legacy nếu cần tra cứu; không chuyển nó sang trang khác, không thêm một trang debug tương tự. Bảng chi tiết bên dưới là căn cứ, không phải tất cả đều đưa lên slide.

### C5 · P2 — Trang 8/9 chưa giải thích dataset được sinh như thế nào cho người mới

**Vị trí:** `23_synthetic.tex:4–23`; `ticket_01_frame09_generator.tex:12–39,44`.

Trang 8 đã tách đúng private truth và public N, đây là sửa đúng. Tuy nhiên chỉ nêu p/strata và saved shape mà chưa nói một hàng dữ liệu là entity × time, component là gì. Trang 9 có normalization và multinomial đúng code, nhưng chưa giải nghĩa w, p, r, N. Caption nguồn Schouten quan trọng về provenance nhưng không thay được giải thích cơ chế.

**Sửa:** diễn giải ba bước: tạo mức xuất hiện của từng thành phần theo thời gian → đổi thành tỷ lệ → dùng tổng N để lấy mẫu số đếm. p là tỷ lệ nền biết được khi mô phỏng; X/N là tỷ lệ sau lấy mẫu, có dao động. r là phần chưa liệt kê, không phải tỷ lệ missing. Gắn nhãn “minh họa” cho đường sóng hand-drawn hoặc dùng data đã lưu; không để nó trông như kết quả thực nghiệm. Nguồn Schouten đưa footer ngắn/notes, caption dành cho ý nghĩa.

### C6 · P2 — Chưa cho biết tùy chỉnh ở đâu và tới mức nào

**Vị trí:** trang 6–10. Profile config và registry name xuất hiện rời rạc, không có cái nhìn tổng thể về CLI/API/JSON hoặc giới hạn hiện thực.

**Sửa:** bảng summary slot 7; tài liệu companion phân biệt hỗ trợ bởi code với đã dùng trong run. Không được ghi mọi tham số GeneratorConfig đều là CLI flags. Không ghi D/T chỉnh tự do: validate() cố định120/12. Fixed-wave đã implemented nhưng không thuộc hai masks declared trong suite đang trình bày. MNAR của Table1 reference không phải latent-p MNAR của synthetic count generator.

### C7 · P2 — Độ cô đọng cần đến từ bớt ý lặp; không từ nén chữ

**Vị trí:** `04_runtime.tex:18–37`; `05_boundary.tex:5–27`; `ticket_01_frame10_profiles.tex:3,36–40`.

Trang 4 vẫn có node DATA input cao chạm hàng MODEL; 12 ô chia đều cho data/model/evaluator buộc người xem giải mã một ma trận trước khi hiểu flow. Trang 5 ngắt “optional” trong node và có ký hiệu/luồng chữ dày. Trang10 plot mới tốt hơn bảng cũ nhưng dòng ID/source/config chiếm nhiều sự chú ý hơn ý nghĩa của khác biệt profile.

**Sửa:** trang4 rút thành4 bước xử lý chính, một boundary test, một câu refit khác protocol; trang5 bỏ bớt định nghĩa lặp và dùng ví dụ budget; trang10 giữ plot và rút config thành3 cột nhỏ hoặc notes, caption giải thích điều đọc được. Giữ provenance đầy đủ ở companion. Không giảm font hoặc thu toàn hình để “đạt” layout.

## Những điểm đã cải thiện, không cần làm lại

- Trang1 đã sửa trục thời gian/component.
- Trang8 không còn đặt N trong private truth; model và evaluator đã tách.
- Trang4 đã ghi khác biệt COVID train / synthetic train+validation.
- Trang5 đã có proposal → constraint → final counts và C/H/M/U nhất quán hơn.
- Trang10 có ba panel từ saved arrays, cùng axes, selection rule minh bạch. Giữ dữ liệu này; không chạy lại mô hình.

## Bảng khả năng hiện thực đã đối chiếu source

| Khả năng | Chỉnh qua đâu? | Ý nghĩa / giới hạn đúng code | Bằng chứng |
|---|---|---|---|
| Chọn preset dữ liệu | CLI `generate --profile` | easy/medium/hard/all; preset đổi timing shift, dominance, noise, mean/CV N; tên không chứng minh ranking sai số | `synthetic/cli.py`, `generator.py:12–17` |
| Quy mô và tái tạo | CLI `--scale`, `--seed` |30/60/120 entities; luôn120 timepoints,12 named components; seed hỗ trợ tái tạo, không đồng nghĩa đã có repeated-seed evaluation | `cli.py`, `GeneratorConfig.validate` |
| Dạng sóng và phụ thuộc thời gian | Python API `GeneratorConfig` | wave_width, onset_lead, tail_weight, temporal_correlation; không có flags CLI tương ứng trong CLI hiện hành | `generator.py:20–51` |
| Residual distribution | Python API `residual_mean/residual_sd` | cấu hình Beta của residual probability; count residual có thể0 | `generator.py:30–50,79–90` |
| Entity split | Python API `train_fraction/validation_fraction` | test là phần còn lại; validate phân số và không để split rỗng; standard mặc định36/12/12 | `generator.py:32–33,51–52,91–94` |
| MCAR/MAR declared patterns | JSON MaskSpec + CLI/API | patterns, frequencies, case_probability, seed; MAR relative weights N/trait; nominal rate khác realized rate | `service.py`, `masking.py`, configs |
| Fixed measurement wave | JSON/API, `scenario=fixed_wave` | wave_time, MCAR/MAR; đã có code nhưng suite đang báo cáo dùng hai declared masks | `service.py`, `capabilities.py:13` |
| Đổi model đã có | Config + registry | các primary keys hron_knn, micoda_v2, tsagris_alpha; params như k/W/max_iter/alpha theo constructor | `registry.py`, model constructors |
| Thêm model mới | Sửa source core/registry theo contract | triển khai NumericalCore gồm fit, predict_result, input_domain, prediction_policy; đăng ký factory/name và dùng adapter phù hợp | `interfaces.py:7–15`, `registry.py:11–12,54–68` |
| Metadata và fallback | Python adapter API; benchmark config cho phần được expose | MetadataEncoder, zero_delta, fallback error/train_mean, degenerate policies; fallback có điều kiện input/prediction policy | `metadata.py`, `adapters.py:20–38`, engine make_adapter |
| Xuất dataset/kiểm hợp đồng | Exporter/loader tự thực hiện | public conditioning + private truth + manifest/hashes; không phải “toggle bảo mật” và không phải quyền OS | `generator.py:export_dataset`, `datasets.py`, `publication.py` |
| Chưa hỗ trợ ở synthetic count masking | Không gắn nhãn ready | latent-p MNAR, random temporal blocks, full independent cell-rate grid, exact realized rate; không hứa có vì code reference Table1 có MNAR | `capabilities.py:14–21` |

Phần trình chiếu chỉ lấy các hàng cần thiết; chi tiết này làm companion để kiểm claim, không nhét nguyên bảng lên slide.

## Cách hiểu hệ thống bằng ngôn ngữ đơn giản

Codebase có ba việc chính: chuẩn bị dữ liệu, chạy một phương pháp điền khuyết và chấm kết quả. Các model dùng cùng cấu trúc đầu vào. Adapter chuyển counts sang tỷ lệ cho model và đưa dự đoán trở lại counts, giữ các giá trị đã quan sát và tổng không vượt N. Benchmark quyết định dữ liệu nào để fit, chọn tham số và test; evaluator so prediction với các ô truth được giữ riêng.

Dataset synthetic được tạo từ các thành phần tăng/giảm theo thời gian. Các mức đó được chuẩn hóa thành tỷ lệ; tổng N quyết định số lượt lấy mẫu để sinh counts. Sau đó hệ thống che một phần dữ liệu theo spec, nhưng giữ bản đầy đủ để chấm. Thay profile/scale/seed/mask cho phép tạo các bối cảnh khác nhau trong giới hạn code hiện có.
