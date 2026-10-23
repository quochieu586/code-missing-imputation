# Missingness theo Schouten 2018

Trạng thái ngày 2026-10-23: đã triển khai phạm vi người dùng chốt. Nguồn duy nhất cho cơ chế gây thiếu là Schouten, Lugtig & Vink (2018), DOI 10.1080/00949655.2018.1491577. Các claim, trích dẫn và vị trí nguồn được kiểm trong `docs/synthetic_missingness_claim_evidence_audit_vi.md` và `docs/synthetic_missingness_single_source_scope_vi.md`.

Cập nhật tiếp theo cùng ngày: generator dễ/vừa/khó đã được người dùng cho phép triển khai và bàn giao trong `datasets/synthetic/wave_panel_v1`. Xem `SYNTHETIC_DATA_VI.md`; những mô tả generator chưa triển khai ở bảng lịch sử dưới đây được thay thế bởi tài liệu mới. Các giới hạn missingness NOT-READY không đổi.

## Phạm vi đã thực hiện

| Chức năng | Trạng thái |
|---|---|
| Sinh dữ liệu Gaussian và năm thiết kế của Table 1 | IMPLEMENTED — reference |
| MCAR trên counts với pattern khai báo trước | IMPLEMENTED |
| MAR chỉ dùng N và entity_trait_01 luôn quan sát | IMPLEMENTED |
| Một wave chung cố định, che toàn bộ component của thực thể được chọn | IMPLEMENTED |
| MNAR trực tiếp của Table 1 | REFERENCE-ONLY, chỉ trong lệnh table1 |
| MNAR theo latent p; random temporal blocks; toàn bộ trajectory của component | NOT-READY |
| Lưới beta, lưới tỷ lệ ô độc lập, ép exact realized rate | NOT-READY |
| Tái lập bảng bias/coverage sau PMM; generator waves/counts đầy đủ | Chưa triển khai |

Không fallback sang cơ chế khác. Yêu cầu tính năng NOT-READY qua API phát sinh `NotReadyError`; CLI trả mã 2. Dữ liệu counts đầu vào cần đầy đủ, không âm, nguyên, tổng component không vượt N. Module này nhận dữ liệu đã sinh; chưa sinh toàn bộ benchmark dễ/vừa/khó hoặc chạy ba baseline.

## Implementation và ranh giới phương pháp

Python gọi trực tiếp mã R `ampute` được lưu nguyên bản trong `src/longitudinal_impute/synthetic/_vendor`, pin commit `61083e667fa41cc4bc655492591b100bf8a7e443`. Mỗi lần chạy kiểm SHA256 nguồn và lưu provenance, seed, session R, warnings. Đây là snapshot implementation hiện đại của phương pháp; không tuyên bố là đúng phiên bản phần mềm năm 2018. Xem `_vendor/PROVENANCE.md` và giấy phép đi kèm.

Table 1 giữ n=1000, mean=(5,5,10), variance=1, rho=.1 đến .9, pattern (Y1,Y2,X1)=(0,0,1), case probability=.2 và năm cấu hình MCAR/MAR/weak MAR/MNAR/weak MNAR của nguồn. Phần này dùng chuẩn hóa và calibration liên tục của mã gốc; không có bước sửa mask để đạt quota. Không tái lập PMM hay các kết luận thống kê sau điền khuyết.

Adapter counts đặt weights target bằng 0. MAR có weight âm cho N và dương cho trait, với tỷ trọng tương đối cấu hình được. Trước khi gọi R, chỉ N/trait được chuẩn hóa bằng sample SD; target trong ma trận tính mask là zero placeholders, `std=FALSE`. Điều này tránh nhánh xử lý cột hằng của mã nguồn và bảo đảm việc sinh mask không đọc counts hay ground truth riêng tư. Sum score vẫn được mã nguồn chuẩn hóa và calibration bằng mẫu Normal tham chiếu 10.000 điểm. Đây là cấu hình/preprocessing cho benchmark mới, không phải weights nguyên bản của Table 1 hoặc một công thức xác suất mới.

`n_weight` và `trait_weight` là tỷ trọng tương đối: nhân cả hai cùng một hằng số không tạo điều khiển độ mạnh beta vì nguồn chuẩn hóa score. Không diễn giải chúng như logistic slope. Các covariates quan sát được chuẩn hóa trên panel dùng để sinh mask; đây không phải scaler huấn luyện baseline. Baseline vẫn phải fit preprocessing chỉ trên train.

`case_probability` là xác suất một candidate case nhận pattern, không phải tỷ lệ ô target. Ví dụ ba pattern mỗi pattern che 4/12 component, case probability=.3, cho tỷ lệ ô danh nghĩa .1. Với fixed wave trên T thời điểm và pattern che mọi component tại đúng wave đó, tỷ lệ ô danh nghĩa là case probability/T. Manifest lưu cả tỷ lệ danh nghĩa và đạt được; không cam kết bằng nhau chính xác.

Fixed wave yêu cầu panel cân bằng cùng lưới thời gian. Mỗi thực thể là một case dạng wide; vị trí wave được đăng ký trước. Chỉ N tại wave đó và trait được dùng cho MAR. Đây không phải generator random block có start/length/q.

## Chạy trong PowerShell từ thư mục repo

Yêu cầu Python với dependencies nhóm baselines của nextgen và Rscript có MASS. Không cần cài package mice vì mã nguồn cần thiết đã được pin. Trên môi trường đã kiểm:

```powershell
$env:PYTHONPATH = (Resolve-Path nextgen\src).Path
$env:RSCRIPT = 'C:\Program Files\R\R-4.5.1\bin\Rscript.exe'
& '.\.venv-e3-revised\Scripts\python.exe' -m longitudinal_impute.synthetic capabilities
& '.\.venv-e3-revised\Scripts\python.exe' -m longitudinal_impute.synthetic table1 --output experiments\schouten_table1_new --seed 2018 --replicates 3
& '.\.venv-e3-revised\Scripts\python.exe' -m longitudinal_impute.synthetic mask --data path\complete_counts.csv --schema-config nextgen\configs\schouten2018_schema.toml --spec nextgen\configs\schouten2018_mar_patterns.json --output experiments\schouten_mask_new
```

Output phải là thư mục mới để không ghi đè thí nghiệm. `--replicates 1000` hỗ trợ chạy đủ số lần lặp thiết kế; lần kiểm hiện tại chỉ chạy 3. Spec mẫu MAR là cấu hình benchmark khai báo trước, không phải dữ liệu/pattern Table 1.

Spec fixed wave:

```json
{
  "scenario": "fixed_wave",
  "mechanism": "MAR",
  "case_probability": 0.3,
  "seed": 2018,
  "wave_time": "2020-01-15",
  "n_weight": 1.0,
  "trait_weight": 1.0
}
```

Ngày ISO phải tồn tại trên lưới thời gian của CSV. Với MCAR đổi `mechanism` thành `MCAR`; trait không bắt buộc. Spec declared_patterns dùng 0 cho ô bị che, 1 cho ô giữ lại; mỗi hàng pattern phải che ít nhất một target, frequencies dương và tổng bằng 1.

Output masking gồm `conditioning.csv`, `hidden_mask.csv`, `manifest.json`. File conditioning chỉ có entity, time, N, các target đã che và trait nếu có; p và residual không được xuất. API `MaskResult.condition` dùng cùng ranh giới dữ liệu qua `hide_targets`. Truth giữ ở đầu vào riêng cho evaluator; không truyền file complete truth cho baseline.

## Kết quả kiểm tra đã chạy

- Toàn bộ `nextgen/tests`: **119 passed**, 20.42 giây; riêng module mới 15 test. Có kiểm mask MCAR/MAR trùng lời gọi trực tiếp mã R gốc cùng seed, mask không đổi khi sửa counts/p/residual, fixed wave, cấm tính năng ngoài scope và ranh giới conditioning.
- `experiments/schouten2018_table1_2026-10-23`: **135 design runs = 5 × 9 rho × 3 replicates**. Tỷ lệ case thực tế từ .163 đến .231, mục tiêu .2; X1 không bị che trong cả 135 lượt. Y1/Y2 luôn bị che cùng nhau theo pattern. Đây là kiểm chức năng, chưa phải validation Monte Carlo 1000 lần hoặc chứng minh hiệu quả trên counts.
- Build wheel thành công; kiểm archive có đủ 19 file synthetic, gồm bridge R, nguồn, hash manifest và giấy phép.

Kết quả Gaussian của nguồn chưa chứng minh hiệu quả trên composition/counts mới. Việc đánh giá ba baseline và phân tầng true zero/sampling zero/positive count vẫn cần thí nghiệm riêng khi generator và protocol tương ứng sẵn sàng.


## Contract sau repair 2026-10-23

`MaskSpec` tại `synthetic/service.py` chỉ nhận scenario enum `declared_patterns` hoặc `fixed_wave`, mechanism MCAR hoặc MAR. Capability `table1_design` không phải scenario mask, bị từ chối trước data/output I/O. Fields thừa/thiếu, probability/weights/seed/pattern/frequencies sai bị từ chối. NOT-READY tiếp tục chặn, không fallback.

CLI/API dùng chung service apply-mask/writer; exporter không gọi CLI hay round-trip CSV để lấy hidden mask. Nguồn R, calibration, seed và masks hợp lệ giữ nguyên. CLI giữ format observed cells của CSV đầu vào. Xem `SYNTHETIC_DATA_VI.md` về dual-truth, residual và version; `audit_results_post_fix.json` đối chiếu v2 với v1.
