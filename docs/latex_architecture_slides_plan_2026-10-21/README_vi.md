# Kế hoạch báo cáo: từ codebase đến kết quả điền khuyết

**Đề xuất: 24 slide chính + 6 slide phụ lục, tiếng Việt, khoảng 25–30 phút.** Khoảng hai phần ba thời gian cho kiến trúc, hiện thực và vận hành; phần còn lại cho protocol, kết quả ba baseline và mở rộng synthetic. Lượt này là kế hoạch, chưa tạo bộ `.tex`/PDF hoàn chỉnh.

## Mạch trình bày

| Section thay vào mẫu | Slide | Câu hỏi cần trả lời |
|---|---|---|
| Bài toán & phạm vi | 1–3 | Hệ thống điền gì, ràng buộc gì, phần nào đã chạy? |
| Kiến trúc codebase | 4–9 | Các lớp liên hệ thế nào, dữ liệu đi đâu, ai sở hữu state? |
| Hiện thực baseline | 10–13 | Hron/MICoDa/Tsagris khác gì dưới cùng interface? |
| Mở rộng & vận hành | 14–17 | Thêm model, chạy/resume và truy lỗi ra sao? |
| Đánh giá & kết quả | 18–22 | So sánh công bằng thế nào, kết luận nào có căn cứ? |
| Mở rộng đánh giá | 23–24 | Synthetic bổ sung gì và bước tiếp theo là gì? |

Chi tiết từng trang trong [storyboard_vi.md](storyboard_vi.md); định nghĩa metric và giới hạn suy luận trong [metrics_protocol_vi.md](metrics_protocol_vi.md).

## Kế thừa mẫu LaTeX

Đã đọc `main.tex`, `sections/08_residual_decoder.tex` và `sections/18_cuprite_synth_paired.tex` ở thư mục mẫu. Giữ Beamer 16:9, theme Hannover, sidebar trái 1.6 cm, BKTemplate xanh `#1E4E79`, BKLight nền nhạt, Ink/Muted, logo BK và cách chia `sections/*.tex`. Đổi toàn bộ title, short title, section navigation, nội dung và figure paths. Giữ logo chỉ khi đúng ngữ cảnh đơn vị của báo cáo; không sao chép tên tác giả hay kết quả QSiMix.

Mẫu hiện dùng inputenc/T1. Bản tiếng Việt chuyển sang XeLaTeX + fontspec và font có đủ glyph Việt đã kiểm tra trên máy; bỏ cặp inputenc/T1 khi dùng XeLaTeX. Giữ hình thức của theme, kiểm lại độ rộng sidebar cho tên section tiếng Việt. Tác giả/đơn vị là thông tin cần điền trước khi xuất bản, chưa cản lập nội dung.

Quy tắc thiết kế: mỗi slide một thông điệp và một hình chiếm khoảng 65–80% vùng nội dung. Chỉ 1–3 ý ngắn, khoảng 20–40 từ giải thích ngoài nhãn sơ đồ; không nhồi source tree/code listing. Chữ nội dung hướng tới ≥18 pt; captions/nguồn nhỏ hơn nhưng phải đọc được. Nếu thiếu chỗ, tách trang thay vì thu nhỏ cả sơ đồ. Cấu trúc chi tiết, công thức dài, source paths và caveats đầy đủ chuyển vào notes/phụ lục. Caveat quyết định kết luận vẫn hiện cạnh plot, không giấu trong notes.

TikZ cho kiến trúc để font, màu và line weight đồng bộ. Mermaid chỉ làm bản nháp kiểm luồng, khi đưa vào slide ưu tiên chuyển sang TikZ hoặc vector SVG/PDF. Plot dùng Python xuất PDF vector; PNG chỉ cho preview. Màu baseline cố định: Hron xanh lam, MICoDa cam, Tsagris xanh lá; thêm marker/line style để phân biệt khi in xám. Không dùng radar chart cho các metric khác đơn vị, biểu đồ 3D hoặc heatmap tự normalize khiến mất ý nghĩa sai số.

## Căn cứ và trạng thái thực thi

Luồng baseline thực: `baselines/csv_io.py → panels.py → CountBaselineAdapter → numerical core → CountBudgetConstraint → benchmark/metrics.py → report.py`. `benchmark/engine.py` điều phối; `driver.py` chạy các fold subprocess; `artifacts.py` và `completion.py` quản lý identity/attempt/artifact validation. `tracking.py` chỉ khai báo manifest, không mô tả nó như logger đã triển khai.

`src/` giữ pipeline KNN/CLR/CSDI legacy. `nextgen/pipeline.py` vẫn ném NotImplementedError, cùng nhiều neural Protocol/scaffold. Không vẽ neural training như tính năng của benchmark ba baseline. Các extension point thực là NumericalCore/FittedCountBaseline, registry, adapter, schema và candidate config; hỗ trợ interface không đồng nghĩa mọi model có thể cắm vào mà không đổi protocol.

Accepted run: `experiments/offline_baselines_2026-10-15_v2/`. Ba cấu hình primary là Hron-kNN, MICoDa-V2, Tsagris-alpha, không phải Hron-LTS/MICoDa-V1/Tsagris-adaptive. Source hiện tại đã có sửa sau benchmark; sơ đồ mô tả hiện tại, bảng performance ghi snapshot accepted. Trước khi làm claim performance cho code hiện tại phải kiểm parity hoặc tạo run mới. Dùng kịch bản `.harness/three-baseline-architecture-talk-2026-10-21/presentation_vi.md` làm notes nền, không copy nguyên đoạn dài lên slide.

## Bộ sản phẩm khi triển khai

```text
presentation/implementation_report/
  main.tex
  sections/       # mỗi slide hoặc cụm rất nhỏ một file
  figures/        # TikZ/PDF vector và previews
  scripts/        # code tạo plot, không chứa logic model
  data/           # CSV dùng cho figure + support manifest
  sources/        # nguồn/mẫu/source snapshot manifest
```

Thứ tự: khóa source/data → dựng theme và 3 slide mẫu (architecture, lifecycle, results) → dựng toàn bộ sơ đồ → tạo plots từ accepted CSV/predictions → ghép sections/notes → compile và render mọi trang → kiểm metric và layout → bàn giao PDF + nguồn. Synthetic là nhánh công việc sau, không phải điều kiện khiến phần kiến trúc phải chờ.

Đủ điều kiện bàn giao slide khi compile thành công, mọi trang đã xem render, không overflow/missing glyph, mọi figure có data/nguồn/support, ba baseline đúng variant, không dùng final-refit làm test, và không có placeholder bị trình bày như kết quả. Việc compile/render và chạy synthetic **chưa được thực hiện trong lượt lập kế hoạch này**.
