# Main-model presentation — 8 slides

Đây là **project duy nhất được giữ lại** sau khi dọn `presentation/`.

- PDF cuối: `main_model_presentation_8slides.pdf`.
- Nguồn chính: `main.tex`; thông tin tác giả/ngày: `config.tex`.
- Nội dung: `slides/` — tám slide tiếng Anh.
- Hình chỉnh sửa được: `figures/` — năm hình TikZ và style dùng chung.
- Hình đã xuất: `exports/pdf/`, `exports/svg/`, `exports/png/`.
- Theme/template: `theme/proposal_theme.tex`; logo: `assets/bk-logo.png`.
- Lời thuyết trình tiếng Việt: `PRESENTER_NOTES_VI.md`.
- Nguồn khoa học và giới hạn: `SOURCE_MAP_VI.md`.

## Build

Cần Python 3, MiKTeX `pdflatex` và Poppler (`pdftocairo`, `pdftoppm`, `pdfinfo`,
`pdffonts`, `pdftotext`) đã cài và có trên PATH. Các LaTeX packages Beamer, TikZ,
standalone, array và fix-cm phải có sẵn. Không cần PyTorch, dữ liệu hoặc code model.

Từ repository root:

```powershell
python "presentation/main_model_8slides/build.py" all
```

Hoặc gọi đường dẫn tuyệt đối tới `build.py` từ thư mục bất kỳ.

- `all`: build lại hình TikZ, xuất PDF/SVG/PNG, rồi build deck tám trang.
- `deck`: chỉ build deck, dùng các figure PDF đang có trong `exports/pdf/`.
- PDF cuối luôn là `main_model_presentation_8slides.pdf` ngay trong project.
- `build/` được tạo lại khi build, chứa file tạm, logs và ảnh render; có thể xóa
  thư mục này sau khi hoàn tất mà không mất nguồn, hình đã xuất hay PDF cuối.

Script không tự cài packages, không bật shell escape và không chạy model.
Nó kiểm tra số trang, Type3 fonts và cảnh báo tràn khung/font. Sau khi sửa slide,
vẫn nên xem các ảnh `build/pages/slide-*.png` để kiểm tra bố cục.

## Nội dung và provenance

Giữ tên **Huu Phuc Le**, ngày để trống. Theme4:3/navy được tái dựng từ proposal
được cung cấp; toàn bộ nguồn theme và hình cần build đều nằm trong project này.
Proposal gốc trong `docs/` chỉ là tài liệu tham khảo, không phải build dependency.

Slide3 so sánh vanilla RNN, LSTM, CNN/TCN và Transformer bằng đúng ba cột.
Slide8 chứa năm tài liệu IEEE; citation [1]–[5] nhảy tới đúng reference.
Không có measured winner hoặc benchmark mới được công bố trong deck.


Các hình dùng 0.8–0.9 chiều rộng văn bản và ghi nguồn module dự án.
Slide2 dùng ví dụ quốc gia/số ca; slide3 không còn ba dòng tổng kết cũ.
