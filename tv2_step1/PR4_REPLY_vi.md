Cảm ơn bạn đã review kỹ, kèm cả repro. Mình đã sửa đủ F1–F5 và thêm guard tối thiểu cho P3. Không sửa `evaluate.py`, không đổi thuật toán Hron/Tsagris/MICoDa, và số của same_day_mean trên COVID giữ nguyên. Chi tiết, số đo và tên test có trong `tv2_step1/FIXES_PR4_vi.md`.

**Findings**
- **F1** (e698e77): mỗi dòng giờ chỉ đọc hai endpoint của cửa sổ; nhánh pair sửa tương tự.
  - Peak `predict()` ở 4.000 dòng / 4.000 ngày: 184,8 → 0,9 MiB (exclude: 369,7 → 2,7 MiB).
  - Test mới: so với oracle duyệt trực tiếp donor, và test bộ nhớ ngưỡng 16 MiB.
  - COVID: CSV điền giống từng byte, metric masked seed 0–4 giống hệt.
- **F2** (9857243, kết quả 2a2c24f): trước khi ghép, script kiểm tra:
  - cùng data hash, schema, kiểu mask, rate, seed và group map;
  - `np.array_equal` từng mask;
  - main có `other = false`, micoda_other có `other = true`.
  
  Sai bất kỳ điểm nào thì báo lỗi và không ghi file. Đã commit `results_manifest.json` (hash data/schema/mask/model/map cho từng run). Có test cho các trường hợp lệch mask, lệch data, `other = false`, lệch rate.
- **F3** (a5807df): `config.json` ghi đường dẫn đã resolve, sha256, số dòng và số nhóm của map thực sự đã parse (`null` nếu không dùng map). `summarize_runs.py` kiểm tra hash map/data trước khi tính lại bảng mức cửa sổ. Có test "chỉ đổi map thì hash đổi".
- **F4** (bf14cc7): sắp xếp bằng kiểm tra `None`/NaN tường minh; 0,0 đứng trước giá trị dương, run lỗi/thiếu đứng cuối. Có test.
- **F5** (d1e391b, tài liệu efa456c): mọi số thắng/thua giờ được sinh từ `results_multiseed.csv`, cho từng metric (MAE/MSE tỷ lệ, MAE/MSE count) ở cả hai mask, rồi tự điền vào REPORT/PR. Đã sửa: Tsagris thắng Hron về MSE tỷ lệ 5/5 ở cả hai mask, về MAE count chỉ 2/5 (whole_feature) và 0/5 (random_cell). Các câu k/5 khác đã được đối chiếu lại với bảng tự sinh.
- **P3** (c3ea968): chọn raise lỗi, không fallback ngầm.
  - `fit` từ chối bảng có tổng N > 2^53 (COVID 1,5e7), nên hai input của bạn (N = 1e308; donor N = 1e20 ngoài cửa sổ) bị từ chối thay vì ra `completed` sai.
  - `predict` báo `NumericalError` khi tổng/mẫu số không hữu hạn hoặc ≤ 0.
  - Miền hỗ trợ được ghi trong docstring và contract.

**Diễn giải đã chỉnh** (d8cdb39)
- Kết luận nay là "same_day_mean tốt hơn các baseline trong lượt che giả (masked pilot) này", không nói tốt nhất trên ô thiếu thật.
- sd là độ lệch chuẩn mẫu qua 5 seed, không phải khoảng tin cậy.
- Lợi thế đến từ bốn yếu tố cùng lúc (thời gian, châu lục, trọng số N, quy tắc co phần dư), chưa tách được; ablation để bước sau.
- 25,5% (mọi ô thiếu thật) và 1,4–9% (ô che thuộc nhóm model) khác mẫu số, không so trực tiếp.
- marginal_sampler bị co theo phần dư nên không đảm bảo Wasserstein ≈ 0.
- Tỷ số MSE count của MICoDa A18 so với D08 được đánh dấu "chưa kiểm chứng độc lập"; số này nay nằm trong CSV.

Cả bốn run đã chạy lại bằng code mới: output giống từng byte bản trước, `summary.json` bằng nhau trừ thời gian chạy. Test: 185 passed với `-W error::RuntimeWarning` (177 trong `nextgen/tests` + 8 cho script tổng hợp).

**Hai câu hỏi cho bạn**
1. Bạn có đồng ý với kiến trúc evaluation 5 tầng không?
2. Bước tiếp theo nên ưu tiên ablation same_day_mean (tách thời gian / châu lục / trọng số N / quy tắc phần dư, đóng băng cấu hình và metric trước khi chạy), hay các metric của Tầng 1?
