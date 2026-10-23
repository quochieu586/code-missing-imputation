# Hoàn tất FIX_TICKET_VI — 2026-10-23

Đã sửa F1–F4, làm rõ F5 và bổ sung 16 regression cases. Không thay nguồn R, calibration, seed, sampler hoặc thêm phương pháp missingness. Dataset v1, audit_results.json, findings.json và experiment lịch sử được giữ nguyên.

| Finding | Thay đổi và bằng chứng |
|---|---|
| F1 | Evaluator kiểm hash private trước labels, exact keys/shapes/finite/conservation/strata và consistency CSV/NPZ/public. NPZ v2 có row_ids `(entity,time)`; v1 kiểm grid entity-major/date cố định, từ chối reorder. Test mutation CSV, permutation NPZ, reorder, shape, consistency sau rehash; model loader chạy khi private bị chuyển đi. |
| F2 | Scorer chung tại synthetic/scoring.py tách count, sample_ratio (X/N), latent_p MAE/RMSE; hidden-only, overall/3 strata, nhóm rỗng null. Runner chọn validation sample_ratio_mae, tie theo candidate index, không retune test. Sáu prediction gốc được rescore thêm; report lịch sử không bị thay. |
| F3 | MaskSpec và Scenario enum kiểm trường bắt buộc/thừa, scenario, mechanism, seed, probability, weights, patterns/frequencies/wave; dispatch hai nhánh tường minh. table1_design không thể thành fixed_wave; NOT-READY giữ fail closed. |
| F4 | Service apply_mask/write_mask dùng chung CLI/API; bỏ nested CLI/argv và CSV round-trip lấy mask. Validator kiểm toàn bộ specs trước staging; lỗi giữa chừng dọn staging; rename publish nguyên bundle và suite, retry cùng đường đích thành công. Test cả invalid spec thứ hai và backend failure sau mask thứ nhất. |
| F5 | Contract hiện hành sum(named counts)<=N, sum(named p)<1; count residual có thể 0. Hard 178/7200 trường hợp như vậy là multinomial hợp lệ. Strict count <N mỗi sample chưa đáp ứng; không ép count/loại sample. Các lựa chọn đổi protocol và hệ quả ghi tại SYNTHETIC_DATA_VI.md, chưa tự chọn. |

## Artifact/version

- `datasets/synthetic/wave_panel_v1/`: snapshot đối chiếu, hash và replay giữ nguyên cả ba profiles.
- `datasets/synthetic/wave_panel_v2/`: bản xuất mới standard, seed 2026, easy/medium/hard và hai declared MCAR/MAR masks. `synthetic-panel-v2` thêm Unicode row_ids NPZ; suite index `synthetic-suite-v2`. Các arrays gốc, mask, labels và p đều bằng chính xác v1. CSV formatting/provenance/spec serialization có thể khác, nên không tuyên bố hashes v1/v2 bằng nhau.
- `audit_results_post_fix.json`: audit cập nhật, fault injections độc lập trên bản sao; mọi expected probes pass, kiểm cả v1 và v2. Hash generator/runner lịch sử khác source sau sửa là dự kiến; numerical replay, file integrity và nguồn R vẫn được kiểm riêng.
- `dual_truth_metrics_v2.json`: rescore bổ sung sáu prediction cũ, có hash prediction, overall/strata/checks; không fit lại hoặc đổi model/parameters. Metric version dual-truth-v2; nguồn dataset vẫn v1.
- Runner mặc định output mới `experiments/synthetic_easy_baselines_v2_2026-10-23`; artifact output cũ được bảo vệ bằng refusal to overwrite.

## Kiểm tra đã chạy

PowerShell từ repo root:

```powershell
$env:PYTHONPATH = (Resolve-Path nextgen/src).Path
$env:RSCRIPT = 'C:\Program Files\R\R-4.5.1\bin\Rscript.exe'
& ./.venv-e3-revised/Scripts/python.exe -m pytest nextgen/tests -q -p no:cacheprovider --basetemp tmp/synthetic_fix_tests_final
& ./.venv-e3-revised/Scripts/python.exe docs/synthetic_review_2026-10-23/audit_synthetic.py
```

Kết quả pytest: **144 passed trong 43.12s** (128 cases có sẵn + 16 regression cases). Kiểm cả baseline fit/predict, entity-disjoint splits, observed preservation, prediction budget, Table 1 reference và NOT-READY. Audit exit 0: ba v1 bundles đúng mọi file hash/replay; 135 Table 1 design runs đúng hash và joint patterns; ba v2 bundles đúng hash và bằng chính xác v1 ở arrays/masks/labels/p. Sáu saved predictions finite/nonnegative, observed unchanged, không budget violations; metric count/sample ratio khớp lịch sử.

## Rescore test predictions đã lưu

Chỉ easy/standard/seed 2026 và hai declared masks. Objective chọn model vẫn là validation X/N MAE; latent p dưới đây là báo cáo bổ sung. Không có so sánh đủ để kết luận thứ hạng hiệu quả easy/medium/hard, không chạy lại grid tuning hoặc thêm baseline.

| Mask | Method | Count MAE | Sample ratio MAE | Latent p MAE | Latent p RMSE |
|---|---|---:|---:|---:|---:|
| 01_mcar_declared_patterns | hron_knn | 12.493349 | 0.0062687994 | 0.0060923485 | 0.01504878 |
| 01_mcar_declared_patterns | micoda_v2 | 151.13599 | 0.074209981 | 0.074052022 | 0.16777093 |
| 01_mcar_declared_patterns | tsagris_alpha | 12.169494 | 0.006018512 | 0.0059915929 | 0.014384467 |
| 02_mar_declared_patterns | hron_knn | 11.660371 | 0.0068720794 | 0.006596724 | 0.015732323 |
| 02_mar_declared_patterns | micoda_v2 | 127.15123 | 0.074977589 | 0.074727717 | 0.165765 |
| 02_mar_declared_patterns | tsagris_alpha | 10.695893 | 0.0063181658 | 0.0060302812 | 0.014430324 |

Tài liệu contract hiện hành: `nextgen/SYNTHETIC_DATA_VI.md` và `nextgen/SCHOUTEN_VI.md`. Hash integrity dựa vào manifest đáng tin, không phải chữ ký xác thực chống chỉnh sửa cả manifest và dữ liệu. Atomic publish dùng staging/rename cùng filesystem, không cam kết durability khi mất điện. Strict residual count positivity cần quyết định protocol riêng nếu trở thành yêu cầu bắt buộc.
