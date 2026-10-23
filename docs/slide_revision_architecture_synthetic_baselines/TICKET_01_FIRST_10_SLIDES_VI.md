# Ticket 01 — Chỉnh 10 slide đầu theo storyboard mới

**Loại:** chỉnh sửa deck LaTeX hiện có  
**Ưu tiên:** cao — nền tảng trước khi chỉnh các module baseline  
**Nguồn kế hoạch:** [PLAN_VI.md](PLAN_VI.md)  
**Deck:** `presentation/implementation_report/main.tex`

## Mục tiêu

Chỉnh mười frame đầu để dẫn người xem từ bài toán qua kiến trúc codebase tới synthetic generator/profile. Đây là chặng đầu của plan 30 trang; dùng lại theme và diagram/plot có sẵn, chỉ thêm hình giải thích khi thiếu. Phong cách tối giản, học thuật, ít chữ, ưu tiên TikZ/PGFPlots vector và công thức đặt đúng chỗ.

## Storyboard trong ticket

| Frame | Tiêu đề/mục đích | Tài sản và thao tác |
|---:|---|---|
| 1 | **Kiến trúc và hiện thực ba baseline điền khuyết** — bìa | Giữ template, logo, palette; sửa title/subtitle ngắn. |
| 2 | **Một bài toán, hai nguồn đánh giá, ba module** — input, mục tiêu, lộ trình | Gộp problem + scope; giữ hình observed/hidden/count-budget; thêm dải roadmap. |
| 3 | **Toàn cảnh codebase: trách nhiệm và phụ thuộc** — data/contracts → orchestration → adapter/core → scorer/artifacts | Gộp ý từ architecture + module map; vẽ TikZ khoảng 6–8 khối, mũi tên đúng chiều gọi/phụ thuộc. |
| 4 | **Input → process → output: một lượt chạy thực tế** — load/validate → train → validation selection → refit → test → artifacts | Gộp runtime + data-flow; swimlane data/model/evaluator; labels test chỉ tới scorer. |
| 5 | **Adapter chung: từ counts sang composition rồi trở lại counts** — vai trò adapter và count budget | Tái dùng boundary/budget bar; thể hiện residual Other, observed preservation và proposal từ core. Phân biệt adapter residual với private residual truth. |
| 6 | **Plug-and-play: thay core, giữ hợp đồng** — interface, registry, ownership state | Tận dụng extension/plugin diagram hiện có hoặc appendix; ba core cùng fit/predict interface; ghi giới hạn adaptation. |
| 7 | **Quản lý và debug một run bằng artifact/diagnostic** — lần metric bất thường theo trace nào | Rút gọn lifecycle/artifact/debug; trace config → candidate → selected params → prediction → diagnostics → score. Có thể dùng runtime plot làm inset nhỏ nếu còn dễ đọc. |
| 8 | **Synthetic: ground truth có kiểm soát để đánh giá điều gì?** — counts truth và latent p, zero strata | Thay placeholder “chờ thí nghiệm” bằng trạng thái artifact hiện có; giữ pipeline Sinh → truth → mask → benchmark. |
| 9 | **Generator: latent dynamics → composition → sampled counts** — giải thích ba tầng generator | Thêm một sơ đồ TikZ/vector gọn từ code/docs. Chỉ hai công thức chính: `p_j=(1-r)w_j/Σw_j` và multinomial trên D phần + residual. Ghi rõ wave generator là thiết kế benchmark của dự án. |
| 10 | **Easy / medium / hard: profile thay đổi những gì?** — so sánh cấu hình dữ liệu | Tạo plot minh họa từ arrays/config sẵn; cùng trục/thang, ghi entity/seed. Hiện config `60×120×12`, timing SD `1/4/8`, dominance SD `.10/.35/.70`, noise SD `.05/.15/.35`, E[N] `2000/500/100`. Không dùng error model để giả làm hình dữ liệu. |

## Nguồn chính

- Storyboard đầy đủ: `docs/slide_revision_architecture_synthetic_baselines/PLAN_VI.md`.
- Deck/style/diagram: `presentation/implementation_report/main.tex`, `presentation/implementation_report/sections/` và `presentation/implementation_report/figures/`.
- Contract và generator: `nextgen/SYNTHETIC_DATA_VI.md`, `nextgen/SCHOUTEN_VI.md`, `nextgen/src/longitudinal_impute/synthetic/`.
- Artifact synthetic: `datasets/synthetic/wave_panel_v2/`, `experiments/synthetic_baselines_v2_2026-10-23/plots/README_vi.md` và CSV/PNG cùng thư mục.
- Xem các section hiện có đang được input trong `main.tex`; số trong tên file không phải số trang.

## Phạm vi

- Chỉnh nội dung/diagram của frame 1–10 và thứ tự `main.tex` cần thiết để mười frame đầu theo storyboard.
- Giữ theme 16:9, màu Hron/MICoDA/Tsagris, typography và macro hiện tại; chỉ sửa preamble nếu có lỗi cần thiết và ghi rõ.
- Có thể thêm section TeX mới cho trang synthetic và copy/tái cấu trúc diagram từ nguồn cũ; đừng xóa section cũ vì các ticket sau còn cần tái sử dụng.
- Giữ body của phần frame sau trang 10 nguyên trạng trong ticket này. Không sửa/đánh bóng baseline hoặc kết quả pages 11–30; ticket tiếp theo xử lý chúng theo storyboard.
- Không chạy benchmark/generate dataset. Có thể build plot từ dữ liệu đã lưu nếu không đổi số; ghi lại source CSV/script.
- Không thay số liệu accepted COVID, bảng metrics, cấu hình model hay code thuật toán.

## Điều kiện nghiệm thu

1. Mười frame đầu xuất hiện đúng thứ tự và trả lời đúng mục tiêu storyboard phía trên.
2. `main.tex` vẫn biên dịch được; frame sau trang 10 không bị mất hoặc sửa nội dung trong ticket này.
3. Trang 3 cho thấy module và quan hệ; trang 4 cho thấy luồng input/process/output, hai trang không lặp lại cùng sơ đồ.
4. Trang 5–7 giải thích adapter, cắm model mới, ownership/debug bằng luồng hình dễ theo dõi; chữ nhỏ không bị dùng để nhồi nội dung.
5. Trang 8 phản ánh artifact synthetic hiện có. Trang 9–10 dùng generator/config đúng tài liệu và không gọi dữ liệu minh họa là dữ liệu quan sát thật.
6. Thông tin COVID và synthetic không bị trộn. Không khẳng định synthetic có nhiều seed/CI hoặc MNAR nếu artifact chưa có.
7. Compile XeLaTeX thành công; render trang 1–10 và rà full-slide. Không có overfull/cắt chữ; labels, công thức và nguồn đọc được. Kiểm cả contact sheet và các trang dày 3, 4, 9, 10 ở kích thước lớn.
8. Bàn giao đường dẫn TeX/PDF/render, danh sách file sửa, kết quả build/render và giới hạn còn lại.

## Ngoài phạm vi ticket

Trang 11–30 theo storyboard đầy đủ; tinh chỉnh Hron/MICoDA/Tsagris; cập nhật ba plot COVID; thêm trang metric/CI synthetic. Giữ chúng cho các ticket tiếp theo, không mở rộng phạm vi hiện tại.
