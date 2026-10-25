# Review và tối giản nextgen ngày 25/10/2026

Đợt cleanup đã hoàn thành phần hạ tầng. Chưa thể kết luận ba baseline chạy
thành công trên COVID-19. MICoDa và Tsagris còn lỗi cần duyệt phương pháp.

## Phát hiện và xử lý

1. Luồng chạy bị tách qua FitScope, registry, adapter, panel, metadata và một
   benchmark nhiều folds. Các khái niệm đó vượt nhu cầu chạy toàn bộ bảng.
   Đã xóa benchmark, adapter, runner cũ, FitScope và metadata encoder.
2. Phần lớn module ở cấp package là Protocol hoặc scaffold chưa thực thi.
   `pipeline.run()` chỉ ném NotImplementedError. Đã bỏ các module đó và thay
   bằng hàm chạy trực tiếp.
3. Synthetic kéo theo generator, R vendor, mask patterns, CLI, tests và outputs.
   Đã xóa source, dữ liệu trong datasets/synthetic, script chạy synthetic,
   cấu hình Schouten, tests và các thư mục experiment synthetic/Schouten.
   Các tài liệu nghiên cứu lịch sử ngoài nextgen không nằm trong đợt xóa này.
4. CSV, mask và phép chuyển count bị phân tán qua nhiều class dữ liệu.
   Đã gom vào data.py với một record CovidData. Giữ phép chuyển proportion,
   Other, zero replacement Hron và residual count allocation của luồng cũ.
5. Mask ngẫu nhiên ban đầu che hết observed của hai hàng COVID-19. Hron không
   còn anchor để đánh giá MSE. Đã sửa mask để giữ một observed trên mỗi hàng
   có dữ kiện, không thay thuật toán và không chia tập dữ liệu.

Không có file Python hiện tại dài quá 1.000 dòng. Vấn đề chính là số lớp và
số khái niệm phải hiểu, không phải một file quá lớn.

## Cấu trúc còn lại

- cli.py chọn ba constructor hiện có và gọi pipeline.run.
- data.py đọc CSV, tạo mask, chuẩn bị proportion, giữ observed và xuất CSV.
- pipeline.py gọi lần lượt fit và predict_result của từng model.
- evaluation.py tính Wasserstein theo cột. MSE chỉ tính trên mask observed.
- baselines/hron.py, micoda.py và tsagris.py giữ nguyên thuật toán.
- common.py, regression.py, diagnostics.py và errors.py giữ nguyên nội dung.

registry.py chỉ còn một helper mà Tsagris gọi nội bộ. Runner không dùng nó.
Cần duyệt sửa baseline trước khi xóa hai phương thức gọi helper này.

## Kiểm chứng

Mức kiểm chứng gồm đối chiếu số học, kiểm thử hành vi và kiểm tra dữ liệu thật.
Không xây framework kiểm chứng mới.

- Source Python giảm từ 56 file và 4.229 dòng ở HEAD xuống 14 file và 1.263 dòng.
- Bảy file được bảo vệ giữ nguyên byte so với trước cleanup. Xem protected_sources.json.
- Ba model trên fixture dương cho kết quả khớp adapter cũ với sai số 1e-12.
  Xem adapter_reference.json và tests/test_covid_flow.py.
- 57 kiểm thử đều qua. Có kiểm tra mask, giữ observed dưới dạng text, bảo toàn
  count budget, native run tách khỏi MSE run, và tiếp tục sau khi một model lỗi.
- Đã build và cài lại editable package trong nextgen/.venv, không đổi thư viện số.
  `general-impute --help` dùng CLI mới. `uv lock --check --offline` qua.
- COVID-19 có 11.671 hàng, 17 target, 112.357 ô observed và 86.050 ô thiếu thật.
  Hron điền đủ missing, giữ nguyên observed và cột ngoài target, không vượt total.
- CSV Hron trùng byte với lượt native chạy trước khi thay phần xử lý dữ liệu.
- Wasserstein trong summary được tính lại từ CSV và đối chiếu khớp.
- Hron MSE là 23788.833053463888 trên 5.518 ô, mask-rate 0.05, seed 7.
  Đây là kết quả chạy, chưa phải kết luận model đạt chất lượng khoa học.

Bằng chứng tổng hợp nằm ở verification.json. Kết quả hiện tại nằm trong
covid_verified/summary.json và covid_verified/hron.csv.

## Phần cần duyệt trước khi sửa baseline

| Model | Bằng chứng hiện tại | Hướng cần thống nhất |
|---|---|---|
| Hron | HronKNN mặc định chạy được native và MSE. | Nghiên cứu và chốt kNN hay iterative LS/LTS. Cleanup chưa đổi biến thể. |
| MICoDa | W=0 dừng khi gặp zero. | Nghiên cứu xử lý zero và chọn W trên miền proportion. Không tự gán một epsilon. |
| Tsagris | Có 1.301 hàng native có tổng observed bằng zero, model dừng ở hàng đầu tiên. | Chốt giữ báo lỗi hay bật global donor mean và chính sách zero missing mass đã có. Chưa bật fallback. |

Các chính sách này ảnh hưởng phương pháp suy luận, cần người dùng duyệt riêng.
Wasserstein chỉ kiểm tra mức độ khớp phân phối. MSE không được tính trên NaN thật.

VERIFIED đối với cleanup và luồng chạy Hron. NOT VERIFIED đối với tiêu chí
cả ba model đều hoàn thành trên COVID-19. Chưa sửa thuật toán để ép tiêu chí đó qua.

## Chạy lại

```powershell
nextgen/.venv/Scripts/python.exe -m pytest nextgen/tests -q --basetemp nextgen/.test-tmp
nextgen/.venv/Scripts/python.exe -m longitudinal_impute.cli --data data/covariants.csv --schema-config nextgen/configs/baselines_covariants.toml --output experiments/covid_review_rerun --mask-rate 0.05 --seed 7
```

Dùng output chưa tồn tại. Với defaults hiện tại, lệnh thứ hai trả exit code 1
và ghi lỗi MICoDa cùng Tsagris, đồng thời giữ CSV Hron thành công.

## Điểm cần chú ý từ review độc lập

Reviewed by gpt-5.6-sol.

Reviewer kiểm tra code và các bằng chứng, không chạy lại toàn bộ model.
Một phát hiện là hàm xuất CSV thiếu kiểm tra trực tiếp giá trị missing được điền.
Đã thêm kiểm tra finite, nonnegative và row total trước khi tạo file.
Bốn ca kiểm thử từ chối NaN, inf, số âm và giá trị vượt total đều qua.
Đã ghi lại CSV COVID-19 qua hàm xuất mới và xác nhận trùng byte với kết quả trước.
Không còn phát hiện chưa xử lý từ lượt review này.

Trail được đối chiếu với bằng chứng trong workspace và các thao tác của lượt này.
Không có thư mục agent-transcripts được cung cấp để kiểm tra transcript từ file.
Các lỗi model trong bảng chờ duyệt vẫn còn nguyên, không được coi là đã đạt.
