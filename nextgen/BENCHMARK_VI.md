# Benchmark offline: vận hành và tái chạy

Kết quả đã chấp nhận ở `experiments/offline_baselines_2026-10-15_v2/`.
Báo cáo: [report_vi.md](../docs/benchmark_offline_2026-10-15/report_vi.md).
Protocol: [protocol_vi.md](../docs/benchmark_offline_2026-10-15_protocol_vi.md).
Evidence: `.harness/full-baseline-benchmark-2026-10-15/` từ repo root.

## Module và trách nhiệm

| Module trong `src/longitudinal_impute/` | Trách nhiệm |
|---|---|
| `baselines/` | Các core Hron/MICoDa/Tsagris, registry, encoder, count adapter và CSV I/O |
| `metadata.py`, `contracts.py` | Metadata fit trên train và hợp đồng phạm vi fit |
| `benchmark/protocol.py` | Chia location; khóa train/validation/test |
| `benchmark/masks.py` | Tạo artificial hidden targets từ observed; seed và hash cố định |
| `benchmark/metrics.py` | Score tại targets, strata, temporal/feature diagnostics, location bootstrap |
| `benchmark/engine.py` | Chuẩn bị manifest, tune validation, test, supplementary, final refit |
| `benchmark/driver.py` | Điều phối 1–2 subprocesses, mỗi fold sở hữu thư mục riêng |
| `benchmark/report.py` | Đọc predictions đã lưu, rescore, tổng hợp báo cáo và kiểm tra CSV |

Config benchmark: `configs/offline_full_benchmark.json`; schema CoVariants:
`configs/baselines_covariants.toml`. Khi đổi dataset phải cung cấp schema cột,
entity/time IDs, mẫu số composition và metadata hợp lệ; không suy targets từ
vị trí cột. Dataset khác phải qua audit domain và split/mask phù hợp trước khi chạy.
Các neural Protocol còn lại chưa được hiện thực; không gọi benchmark này là
Transformer hoặc mô hình temporal đã huấn luyện.

## Tái chạy từ repo root

Nguồn tính toán nằm trong `.py`. Notebook local/Kaggle chỉ đặt đường dẫn,
chuẩn bị runtime và gọi các entry points này. Kaggle chưa được thực thi ở lượt này.
Python >=3.11 và NumPy cần có sẵn; runtime đã dùng và phiên bản nằm trong
`runtime_lock.json` của harness. Không cần torch cho benchmark này.

PowerShell, sau khi chọn đúng Python/environment có NumPy:

```powershell
$env:PYTHONPATH = (Resolve-Path nextgen/src).Path
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$benchmarkOutput = 'experiments/offline_baselines_new_run'
python -B -m longitudinal_impute.benchmark.engine prepare --root . --config nextgen/configs/offline_full_benchmark.json --output $benchmarkOutput
python -B -m longitudinal_impute.benchmark.driver --root . --output $benchmarkOutput --workers 2
python -B -m longitudinal_impute.benchmark.engine final --root . --output $benchmarkOutput
python -B -m longitudinal_impute.benchmark.report --root . --output $benchmarkOutput --report docs/offline_baselines_new_report
```

Dùng tên thư mục mới chưa tồn tại. `prepare` khóa dataset/schema/source/config,
splits và mask hashes. Driver bỏ qua fold/job đã hoàn tất hợp lệ, không ghi đè
predictions đã chấm. Khi resume, không chạy hai driver trên cùng output.
Nếu source/config/data thay đổi, tạo experiment mới; giữ nguyên run cũ làm evidence.
Job dở dang cần kiểm tra artifacts và lỗi trước khi xử lý; không tự xóa để resume.

Hai subprocesses là giới hạn đã đo trên máy local; dùng `--workers 1` nếu thiếu RAM.
Supplementary có 180 giây wall budget cho mỗi fold/method, bao gồm fit/predict/
scoring/I/O; không phải microbenchmark tốc độ cô lập. Cấu hình primary giữ 50
iterations của MICoDa và không tự tăng sau khi xem test.

## Kiểm chứng run đã hoàn tất

- `tests_optimized.txt`: 56 tests pass; `reference_prefix_optimized/comparison.json`:
  9 kiểm tra oracle R pass. Các fixtures không chứng minh mọi chi tiết paper parity.
- `real_data_boundary_audit.json`: 60 trường hợp thay hidden labels không làm đổi
  train, conditioning, metadata hoặc Other. Offline cho phép observations tương lai;
  hidden target bị loại trước encoding/Other. Không có triangular causal mask.
- `independent_verification.json`: kiểm toán 225 validation jobs, 180 primary test
  jobs, 45 supplementary test jobs, cấu hình chọn bằng validation và 3 final CSVs.
- `proposal_replay_audit.json` trong experiment: 450 proposal traces và 3 final
  traces tái tạo predictions giống từng bit; raw proposals và constrained outputs
  nằm cạnh từng job. Đây là replay bổ sung, không thay kết quả gốc.
- `rescoring.json` trong report: đọc lại toàn bộ 180 primary predictions để chấm lại.
- `checks.json`, `checks.md`, `evidence_audit.json` trong harness: tiêu chí và hashes.
  Harness verifier kiểm tra record/evidence, không thay thế kiểm toán số học.

Các helper kiểm toán nằm trong harness để phân biệt công cụ evidence của run này
với public pipeline. Ví dụ, sau khi đặt PYTHONPATH đúng:

```powershell
python -B .harness/full-baseline-benchmark-2026-10-15/verify_benchmark.py --run experiments/offline_baselines_2026-10-15_v2 --output .harness/full-baseline-benchmark-2026-10-15/independent_recheck_new.json
```

## Kết quả và giới hạn bắt buộc đọc

Tsagris alpha có MAE tỷ lệ thấp nhất ở whole-feature/random-cell trong lưới đã
đăng ký; MICoDa có point MAE tỷ lệ thấp hơn ở time-block/whole-row. MICoDa-V2
chưa đạt tolerance sau 50 iterations ở cả 5 folds: đây là kết quả budget cố định.
Tsagris basic/global-CV/adaptive hoàn tất. Hron-LS/MICoDa-V1 vượt budget ở cả
5 folds, Hron-LTS lỗi số học ở cả 5 folds; không có ranking accuracy cho các
trường hợp này và không âm thầm thay bằng thuật toán khác.

Ba final CSVs đều có 11671 dòng, điền 86050 ô native missing, giữ nguyên toàn bộ
counts quan sát và metadata, không âm, `sum(17 targets) <= total_sequence`.
Không ép 17 targets có tổng bằng mẫu số; Other nhận phần dư. Counts imputed là
point estimates liên tục, không được tuyên bố đã integer rounding hoặc multiple
imputation. Accuracy chỉ đo tại artificial hidden observations, không biết
accuracy trên native missing. Bootstrap theo location với fits cố định mô tả
biến thiên OOS errors, không phải uncertainty cho từng ô imputed.

Run v1 chậm được giữ nguyên. Tối ưu prefix donor/cache ở v2 có bằng chứng
48 validation jobs chung, 112092 prediction rows bằng từng bit và metrics bằng
nhau (`full_job_optimization_parity.json`). Không xóa file trong quá trình này.
