# Ticket: sửa pipeline synthetic benchmark sau thermo-nuclear review

**Mục tiêu:** sửa F1–F4 của [review](report_vi.md), giải quyết rõ contract F5 trong tài liệu/validation, giữ hành vi MCAR/MAR Schouten và các mask hợp lệ hiện tại. Không mở thêm MNAR latent-p, random temporal blocks hoặc tính năng đang NOT-READY; không đưa phương pháp gây thiếu từ paper khác vào.

## Bối cảnh cần đọc

- Báo cáo: `docs/synthetic_review_2026-10-23/report_vi.md`
- Probe và dữ liệu kiểm: `docs/synthetic_review_2026-10-23/audit_synthetic.py`, `audit_results.json`, `findings.json`
- Code: `nextgen/src/longitudinal_impute/synthetic/{generator,datasets,cli,masking,capabilities}.py`
- Scorer/runner: `scripts/run_synthetic_easy_baselines.py`
- Dataset đã phát hành: `datasets/synthetic/wave_panel_v1/{easy,medium,hard}`; **không ghi đè**.
- Evidence nguồn Schouten: `docs/synthetic_missingness_claim_evidence_audit_vi.md`; giao thức hiện tại: `nextgen/SCHOUTEN_VI.md`, `nextgen/SYNTHETIC_DATA_VI.md`.

Review đã xác nhận mọi hash dataset đúng, replay generator bằng seed cho kết quả bằng chính xác, 135 Table 1 design runs đúng pattern/hash, và 128 test pass. Bốn lỗi bên dưới được tái hiện trên bản sao hoặc tính lại từ prediction lưu sẵn; test xanh hiện tại chưa bắt được chúng.

## Công việc ưu tiên

1. **F1 / P1 — ground-truth integrity và row identity.** `load_evaluation` hiện không kiểm hash private, dùng vị trí CSV private để index NPZ, và NPZ không có row IDs. Probe: sửa hidden CSV count 0→1 được nhận; đảo hàng CSV vẫn căn đúng count labels nhưng latent p đổi, max chênh 0.7113. Bổ sung identity contract rõ ràng cho latent arrays, kiểm hash private/shape/keys/consistency trước khi tạo labels. Model loader phải tiếp tục hoạt động mà không có `private/`.
2. **F2 / P1 — đánh giá hai ground truth.** `score()` hiện tính `ratio_mae` với X/N, chưa tính latent p. Tách rõ `count_*`, `sample_ratio_*`, `latent_p_*`, overall và theo true-zero/sampling-zero/positive-count; chỉ score hidden cells. Tuyên bố rõ objective dùng validation để chọn model; không retune bằng test. Có thể tính metric latent từ sáu prediction đã lưu và phát hành artifact bổ sung, không ghi đè báo cáo lịch sử. Review JSON có các con số đối chiếu.
3. **F3 / P2 — strict spec validation.** Spec `scenario='table1_design'` hiện trả exit 0 và chạy `fixed_wave`. Dùng enum/typed `MaskSpec` riêng cho đúng hai scenario MCAR/MAR được hỗ trợ, validate trước I/O, dispatch tường minh. Capability manifest chỉ báo trạng thái; không dùng như validator của scenario. Giữ NOT-READY fail closed.
4. **F4 / P2 — service boundary và atomic export.** `export_dataset` đang gọi ngược `cli.main(argv)`, lặp validation và round-trip CSV để vận chuyển mask. Spec thứ hai lỗi để lại thư mục final nửa chừng; retry lỗi `FileExistsError`. Đưa apply-mask/write-bundle vào service Python dùng chung cho CLI/API, validate mọi spec trước khi ghi, staging rồi publish nguyên bundle khi hoàn chỉnh. Sau lỗi không để final path nửa chừng; sửa spec và chạy lại được. Giữ R source/calibration/seed và mask output hợp lệ nhất quán.
5. **F5 / P2 — residual contract.** Latent residual >0 nhưng hard có 178/7200 mẫu residual **count** =0 vì multinomial, nên tổng component counts bằng N. Không tự ép count dư ≥1, cắt sample hay đổi sampler đã chốt. Ghi rõ hợp đồng đang thực thi là `sum(named counts) <= N`, còn `sum(named p) < 1`; nếu yêu cầu strict `<N` ở từng sample vẫn cần thì đánh dấu chưa đáp ứng và trình bày phương án có hệ quả để người dùng chọn. Không coi đây là bug của multinomial.

## Hướng kiến trúc kỳ vọng

Tập trung vào ba boundary: một `MaskSpec` validator/dispatcher; một bundle writer có tính giao dịch; một evaluator truth contract có row ID và kiểm toàn vẹn. Xóa vòng gọi CLI lồng nhau, không thêm nhiều wrapper hoặc flag đặc biệt. Giữ baseline `RawPanel` / `ConditioningPanel` / `FitScope` hiện tại để module mới vẫn plug-and-play. Không sửa vendored R source.

## Acceptance checks

- Regression tests cho ba probe F1, invalid scenario F3, export/retry F4, và ví dụ X/N khác latent p cho F2.
- Public loader từ chối checksum/schema lỗi; evaluator từ chối private truth bị đổi, và row reorder không thể âm thầm gắn sai p.
- Valid masks/reproducibility với seed/source snapshot hiện tại được giữ; Table 1 reference và capability NOT-READY vẫn pass.
- Baseline fit/predict và entity-disjoint split còn hoạt động; observed counts không đổi, prediction không vượt N.
- Chạy toàn bộ `nextgen/tests`; chạy script audit hoặc phiên bản cập nhật tương đương; ghi rõ mọi thay đổi artifact/version. Dataset v1 đã phát hành được giữ làm snapshot đối chiếu. Nếu cần format mới, xuất version mới thay vì ghi đè.
- Cập nhật docs/report để phân biệt sampled ratio với latent p, contract residual, trạng thái hiện hành. Không trình bày ba mức easy/medium/hard như thứ hạng hiệu quả đã được chứng minh nếu chưa chạy so sánh đủ.

## Môi trường và lệnh kiểm

Repo root: `C:\Users\admin\Tai_lieu\detaikhoahoc\COVID19\missing_impute\code-missing-imputation`.

```powershell
$env:PYTHONPATH = (Resolve-Path nextgen\src).Path
$env:RSCRIPT = 'C:\Program Files\R\R-4.5.1\bin\Rscript.exe'
& '.\.venv-e3-revised\Scripts\python.exe' -m pytest nextgen\tests -q -p no:cacheprovider --basetemp tmp\synthetic_fix_tests_new
& '.\.venv-e3-revised\Scripts\python.exe' docs\synthetic_review_2026-10-23\audit_synthetic.py
```

Audit script hiện cố ý chứng minh lỗi trên các bản sao tạm; sau sửa, cập nhật expected results của probes và giữ khả năng audit các dataset/prediction cũ. Kết quả baseline easy hiện tại ở `experiments/synthetic_easy_baselines_2026-10-23/`; chỉ có easy, standard scale, một seed và hai declared masks.
