# Slide báo cáo hiện thực pipeline

Project Beamer 16:9 tiếng Việt, theo theme Hannover và thanh điều hướng mẫu. Gồm 24 slide chính, 6 phụ lục, sơ đồ TikZ và ba biểu đồ PGFPlots vector lấy từ accepted benchmark CSV. Slide synthetic chỉ là đề cương chờ thí nghiệm.

## Mở và biên dịch

Mở thư mục này trong Codex để sửa main.tex cùng các file trong sections/. Cần XeLaTeX và gói Beamer/TikZ/PGFPlots; từ thư mục project chạy:

    xelatex -interaction=nonstopmode -halt-on-error main.tex
    xelatex -interaction=nonstopmode -halt-on-error main.tex

Lần biên dịch thứ hai cập nhật navigation và bookmark.

## Figure và dữ liệu

Chạy python scripts/build_figures.py sau khi thay đổi hai file CSV dưới data/. Script dùng thư viện chuẩn, sinh các fragment tex trong figures/, và ghi data/figure_manifest.json với SHA-256 của đầu vào/đầu ra. Giá trị tỷ lệ được nhân 100 để hiển thị theo điểm phần trăm. CI hiện có là bootstrap theo location với fitted models cố định.

CSV lấy từ docs/benchmark_offline_2026-10-15/; kết quả thuộc run accepted v2 khóa ở 15/10/2026. Source code hiện tại được mô tả trong sơ đồ, còn hiệu năng không được gán ngược cho snapshot source mới hơn. Đánh giá chỉ biết ground truth ở các ô đã che; synthetic chưa chạy.
