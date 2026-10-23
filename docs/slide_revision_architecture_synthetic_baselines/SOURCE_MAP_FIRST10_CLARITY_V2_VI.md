# Source map — 10 trang đầu (clarity V2) · đối chiếu claim với mã nguồn

Mỗi slide giữ provenance ở dòng `\source`; bảng dưới là bản đầy đủ để kiểm từng tuyên bố
đưa lên slide. Trạng thái lấy từ source hiện tại, không gộp "đã hiện thực" với "đã chạy".

## Liên kết frame → file → bằng chứng

| Trang | File | Bằng chứng chính |
|---|---|---|
| 01 | `sections/00_cover.tex` | (bìa, không claim) |
| 02 | `sections/01_problem.tex` | `constraints.py` (lock/tổng N), `benchmark/engine.py`, `synthetic/scoring.py` |
| 03 | `sections/03_architecture.tex` | `contracts.py`, `benchmark/engine.py`, `baselines/registry.py, interfaces.py, adapters.py` |
| 04 | `sections/04_runtime.tex` | `benchmark/engine.py, protocol.py`, `scripts/run_synthetic_easy_baselines.py` (protocol refit train+validation) |
| 05 | `sections/05_boundary.tex` | `baselines/adapters.py` (proposal = output·N; CountBudgetConstraint.apply; ImputationResult), `panels.py` (extended_proportions), `constraints.py` |
| 06 | `sections/06_ownership.tex` | `metadata.py`, `adapters.py:20–40` (fallback error/train_mean; degenerate policies), `metadata.py` (MetadataEncoder), engine make_adapter |
| 07 | `sections/clarity_v2_07_capabilities.tex` | bảng dưới |
| 08 | `sections/23_synthetic.tex` | `datasets/synthetic/wave_panel_v2/{easy,medium,hard}/manifest.json`, `nextgen/SYNTHETIC_DATA_VI.md`, `SCHOUTEN_VI.md` |
| 09 | `sections/ticket_01_frame09_generator.tex` | `generator.py` (generate: weights→p=(1−r)·w/Σw; multinomial(N; p, r)) |
| 10 | `sections/ticket_01_frame10_profiles.tex` | `private/ground_truth.npz` (mảng đã lưu), `scripts/extract_profile_plot_data.py`, `generator.py` (PROFILES) |

## Bảng khả năng (nguồn cho slot 7) — đối chiếu source

| Khả năng | Chỉnh qua đâu? | Giới hạn đúng code | Bằng chứng |
|---|---|---|---|
| Chọn baseline | Config + registry | key (hron_knn, micoda_v2, tsagris_alpha là 3 họ chính; tổng 9 keys); k/W/α theo constructor | `registry.py` METHODS, make_core |
| Sinh dataset | CLI `generate` | profile easy/medium/hard/all; quy mô small/standard/large = 30/60/120 entities; seed | `cli.py:19–20`, `generator.py:17` |
| Động lực dữ liệu | Python API `GeneratorConfig` | sóng (wave_width, onset_lead, tail_weight), temporal_correlation, residual (mean/sd), split fraction — KHÔNG có flags CLI tương ứng | `generator.py:20–33,71–78` |
| Quy mô cố định | (không tùy chỉnh được) | luôn 120 timepoints × 12 components; validate() từ chối giá trị khác | `generator.py:39–40` |
| Tạo missingness | JSON MaskSpec + CLI/`export_dataset` | MCAR/MAR: patterns, frequencies, case_probability, seed; MAR weights theo N/trait (entity_trait_01); nominal ≠ realized rate | `service.py`, `masking.py`, `metadata.py` |
| Fixed measurement wave | JSON/API `scenario=fixed_wave` | ĐÃ CÓ CODE nhưng suite báo cáo đang dùng hai masks khai báo (MCAR/MAR) | `capabilities.py:13` |
| Thêm model mới | Sửa source core/registry theo contract | NumericalCore: fit/predict_result + input_domain + prediction_policy; adapter chỉ bảo đảm contract, không bảo đảm tương thích mọi model | `interfaces.py:7–15`, `adapters.py:33–40` |
| Tái sử dụng dữ liệu | Exporter/loader | conditioning + private truth + manifest/hash; không phải toggle bảo mật/quyền OS | `generator.py:export_dataset`, `datasets.py`, `publication.py` |
| Chưa hỗ trợ (không gắn ready) | — | latent-p MNAR, exact-cell-rate grid, exact realized rate, random temporal blocks, PMM inference | `capabilities.py:15–21` |

## Phân biệt trạng thái (không lẫn implemented với đã chạy)

- **IMPLEMENTED + dùng trong run:** CLI generate (profile/scale/seed/mask-spec), hai masks khai báo MCAR/MAR.
- **IMPLEMENTED + không dùng trong run đang trình bày:** fixed-wave scenario.
- **CHƯA READY (không hứa):** latent-p MNAR, cell-rate grid, exact realized rate, repeated-seed evaluation.

## Dòng "ngôn ngữ đơn giản" đã đưa vào deck (từ review)

- Trang 2/5: ví dụ một mẫu A=20, B=?, C=10 trong tổng N=100 — điền B, giữ 20/10.
- Trang 3 caption: dữ liệu chung một định dạng; bộ chạy chọn mô hình; evaluator chấm với đáp án giữ riêng.
- Trang 9: sinh tỷ lệ nền theo thời gian → lấy mẫu counts theo N → áp mask.
- Trang 10: profile đổi biến động và tổng N; X/N dao động quanh p.
