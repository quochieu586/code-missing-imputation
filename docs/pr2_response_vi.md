# PR #2: phản hồi tái audit Hron/Tsagris

Trả lời `pr_2_hron_tsagris_reaudit.md`. Branch `module-1-general-imputation`.

- Trước sửa: `edfb0a2`. Sau sửa: `3fdc61f`.
- Không sửa `micoda.py`, `common.py`, `regression.py`, `diagnostics.py`, `errors.py`.
- Không thêm lớp trung gian. Mỗi finding nằm trong một commit riêng.

## 1. Tổng hợp

| ID | Commit | Sửa | Test thêm | Trên code cũ |
|---|---|---|---|---|
| **F1** | `6556fb7` | `TsagrisKNN.fit` tính donors, patterns, tuning, (k, alpha) vào biến local và chỉ gán `self` khi mọi bước thành công. `_tune(donors, patterns)` nhận tham số. | `tests/test_fit_lifecycle.py` (7 test, chạy cho cả `TsagrisKNN` và `HronKNN`): first fit lỗi, kể cả tuning lỗi → không có fitted state và `predict` báo `NotFittedError`; refit lỗi → giữ donors/k/alpha/prediction cũ; refit thành công = fit mới. | 3 test Tsagris fail. `HronKNN` vốn đã đạt nên không sửa. |
| **F2** | `c590aac` | `frechet_mean` tính lại trong log-domain theo `alpha_mean` ở `a235e9f`. Hàm không bao giờ trả NaN (chi tiết ở mục 2). | Idempotence với alpha ∈ {-1, -.5, -1e-20, 0, 1e-20, .25, .5, 1}; giới hạn alpha → 0 ≈ trung bình hình học; input 1e-320 với alpha = -1; ví dụ donor `[.1,.3,.6]`, query `[NaN,NaN,.6]`, alpha = ±1e-20 và 0 → `.1, .3`; chốt chặn `NumericalError` ở `predict`. | 8 test fail |
| **F3** | `096c1db` | Loader: dòng có field thừa hoặc thiếu (kể cả cột metadata) → `ValueError` ghi đường dẫn, dòng CSV và tên cột, trước khi parse/fit. Writer: ghi vào file tạm `.<tên>.*.tmp` cùng thư mục, đóng file rồi mới công bố bằng `os.link` (không ghi đè, báo lỗi nếu đích đã có), luôn xóa file tạm. | Field thừa; thiếu target; thiếu metadata; lỗi giữa lúc ghi (monkeypatch `DictWriter.writerow`); đích có sẵn; ở cấp runner: CSV lỗi dừng trước khi tạo output, ghi bị ngắt → `errors.log` có dòng lỗi và không còn `*_completed.csv`. | 6 test fail. Test "đích có sẵn" pass ở cả hai bản, giữ làm regression guard. |
| **F4** | `0762f73` | `run.py`: ngay sau `parse_args`, nếu có mask thì kiểm `--mask-rate` (hữu hạn, thuộc [0, 1), giống `make_mask`) và `--seeds` (khác nhau, không âm). Giá trị sai → `parser.error`, exit 2, trước khi load data, tạo output hay fit. `--mask` vốn đã bị argparse giới hạn. | 6 trường hợp: rate 1.0 / âm / NaN, seed trùng, seed âm, mask không tồn tại → không gọi fit, không tạo output dir. Thêm `--mask none` thì bỏ qua rate. | 5 test fail (mask không tồn tại vốn đã bị chặn) |
| **E1** | `3fdc61f` | Hron rescale thành `(x_lj / scale_l) * scale_i`. Kết quả không hữu hạn → `NumericalError` có row/feature. | `[1e-320,1e-320]` + `[.2,NaN]` → `.2` với cả `median` và `sum`; kết quả không biểu diễn được (1e900) → `NumericalError`. | 2 test fail |

`python -m pytest nextgen/tests -q`: **94 passed**, so với 57 trước sửa. Suite cũng pass khi chạy với `-W error::RuntimeWarning`. Cột "Trên code cũ" có được bằng cách `git stash` riêng file nguồn của finding rồi chạy lại test mới.

## 2. Ghi chú theo finding

**F1.**
- Thứ tự trong `fit`: `_check` → donors → tổng = 1 → patterns → `_tune` → (k, alpha) → một lệnh gán duy nhất.
- `predict` chỉ gán `info_` sau khi có output hợp lệ.

**F2.**
- Ở alpha gần 0, phép lũy thừa trực tiếp mất hết chữ số có nghĩa. Vì vậy code dùng nhánh hình học cho alpha = 0, và cho |alpha| < 1e-8 khi dữ liệu dương.
- alpha = 0 khi có số 0 → dùng phần dương ở mọi dòng, như `a235e9f`. **Bản `edfb0a2` báo lỗi ở trường hợp này.**
- alpha < 0 khi có số 0 → lỗi.
- Mean hoặc prediction không hữu hạn → `NumericalError`. Đây là exception có sẵn trong `errors.py`, là lớp con của `BaselineError`.
- Miền alpha là [-1, 1], giống nhau ở constructor và ở hàm.
- Nhánh alpha = 1 (mặc định) không đổi. Ví dụ paper §2.3.1 vẫn ra (0.2, 0.27, 0.3, 0.1, 0.13).

**F3.**
- `run.py` không cần sửa. Writer vốn được gọi trong khối `except ValueError` của từng model; lỗi ghi nay không để lại artifact nào.
- `os.link` cần filesystem hỗ trợ hard link (NTFS, ext4, APFS đều được). Nếu không hỗ trợ, writer báo `OSError` và vẫn không để lại file.

**F4.** Chọn `parser.error` (exit 2, thông báo một dòng) thay vì exception có traceback, vì đây là lỗi tham số CLI.

**E1.** Đổi thứ tự phép tính làm thay đổi cách làm tròn. Trên COVID, output Hron **không còn trùng từng byte**: 2 429 / 86 050 ô điền khác nhau, nằm trong 534 dòng. Độ lệch tối đa là 2.8e-14 count, tương đối 5.2e-16, tương đương vài ulp. Tôi không thêm nhánh đặc biệt chỉ để giữ nguyên bit cũ.

## 3. Kiểm chứng trên COVID

Cấu hình chỉ gồm Hron và Tsagris, mặc định giống `models_default.toml`. Lệnh chạy ở cả hai phiên bản, output nằm trong `tv2_phase4/` (không commit):

```
run --mask none
run --mask whole_feature --mask-rate 0.05 --seeds 7 42
```

| | Hron | Tsagris |
|---|---|---|
| Native, `--mask none` | 2 429 ô khác do làm tròn (E1), max abs 2.8e-14 | **trùng từng byte** |
| Native, lượt `whole_feature` | như dòng trên | **trùng từng byte** |
| Mask seed 7 / 42 | trùng; 5 644 / 5 658 ô; còn 218 dòng đầy đủ | như cột Hron |
| Masked, seed 7: MSE tỷ lệ / MSE count / MAE tỷ lệ | 0.041893 / 26 728.3 / 0.053328, trùng tuyệt đối trước/sau | 0.029428 / 22 569.2 / 0.054785, trùng tuyệt đối |
| Masked, seed 42 | 0.027451 / 59 179.0 / 0.037216, trùng tuyệt đối | 0.018367 / 79 120.4 / 0.039591, trùng tuyệt đối |

- Mọi lượt đều `ok` và exit 0.
- "Trùng tuyệt đối" nghĩa là mọi metric của mọi nhóm trong `masked_errors` có thay đổi tương đối bằng 0.
- Số ô che khớp mục 2 của bản audit (5 644 / 5 658).

Bằng chứng:
- `tv2_phase4/compare.py` → `compare.json`: so sánh trước/sau.
- `tv2_phase4/e1_check.py` → `e1_check.out.txt`: kiểm tra riêng cho E1.
- `tv2_phase4/{before,after}_{none,whole_feature}/`: output của runner.
- `tv2_phase4/pytest.txt`: kết quả test.
