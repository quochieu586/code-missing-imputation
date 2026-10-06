# Audit và sửa E3/E4 theo research plan revised

Ngày: 2026-09-30. Phạm vi: fold 0 Canada, random-cell r=0.30, mask seed 42, một outer pass. Kết quả cũ được bảo toàn; kết quả thay thế nằm trong `artifacts/e3_csdi_revised/` và `artifacts/e4_pilot_revised/`.

## 1. Kết luận

**Nhận xét “E4 PASS, không có blockers” trong review được cung cấp chưa đủ:** review kiểm tra được việc tái sử dụng E3 và phép tính gate E4, nhưng chưa xác nhận E3 thực sự đáp ứng kiến trúc và điều kiện nghiên cứu. E3 cũ có lỗi phương pháp; không thể dùng PASS conditioning cũ làm điều kiện vào E5.

Đã sửa implementation và chạy lại pilot E3 một lần, sau đó chạy E4 từ predictions mới. **49 tests PASS**, đối chiếu DDPM/loss với source CSDI chính thức PASS, restoration/schema/hash/metrics PASS. Tuy nhiên **gate nghiên cứu E3 FAIL, gate efficacy E4 FAIL, E5 BLOCKED, không có finalists**. Sửa đúng code không đồng nghĩa một mô hình đã đạt chất lượng nghiên cứu.

| Hạng mục | Trạng thái sau sửa | Bằng chứng / lý do |
|---|---|---|
| Lõi DDPM/loss, adapter raw-condition → full LR | PASS kiểm thử | Numerical fixture đối chiếu source chính thức; forward/reverse và loss float64 |
| Mask-only > No-init trên validation, mọi transform | **FAIL** | ILR và HKGLR có Mask-only M2 thấp hơn No-init |
| Mask-only > No-init trên test, mọi transform | PASS trong pilot này | Cả 3 cặp đạt bất đẳng thức, nhưng không khắc phục validation FAIL |
| Sampling finite/nonnegative và observed locking | PASS | 18 phương pháp E4 finite/nonnegative, observed giữ nguyên chính xác |
| Scale engineering guard, validation và test | **FAIL / cần xử lý** | Nhiều dòng vượt 1000 × max(visible sum, 1), lưu row IDs; không cắt bỏ hay cap predictions |
| Schema E4, common evaluation rows | PASS | `methods` có 18 entries; cùng 101 eligible rows / 505 scored hidden cells |
| Init+CSDI cải thiện Init-only >10% M2 | **FAIL** | Cả 6 nhánh initialized diffusion không đạt; nonzero CLR MAE cũng xấu hơn |
| E5 | **BLOCKED** | `E3_CONDITIONING_GATE_FAILED`, `E3_SCALE_OR_NUMERICAL_GATE_FAILED`; efficacy E4 cũng FAIL |

`manifest.status = COMPLETE` chỉ nghĩa là chạy xong. Quyết định nghiên cứu phải đọc gate fields và `e5_eligible`/`e5_status`, không dùng COMPLETE thay PASS.

## 2. Tài liệu áp dụng

- `research_plan_revised_2026-09-30.md`: §3 capacity/projection/reference controls; §4 conditioning/restoration; §5 scenarios; §7 E3/E4 gates; §11 thiếu windows.
- `research_plan_knn_jsd_clr_ilr_hkglr_csdi_2026-09-30.md`: §7.1–7.4 raw/LR masks, initialization sau masking và inverse; §8 numerical core; §10 metrics và T12–T20.
- `revision_summary_2026-09-30.md` và review E4 người dùng cung cấp.
- Handoff `pipeline_knn_diffusion_handover.md`: nguyên tắc observed locking và Aitchison geometry. Thiết lập phase mới của người dùng/revised plan được ưu tiên khi khác handoff cũ: single pass, raw initialization trước positivity, nhiều LR, không gán raw mask cho LR mask.

Protocol U giữ nguyên: closure 78.10% thấp hơn 95%; không triển khai JSD initialization, không dùng `total_sequence` làm true mass. Native JSD trên closed vectors chỉ là diagnostic phụ.

## 3. Các lỗi E3 cũ và cách sửa

| Vấn đề | Mức ảnh hưởng | Sửa thực hiện |
|---|---|---|
| Denoiser là row MLP, thiếu temporal/feature attention của CSDI | Chặn diễn giải “CSDI” | `src/stage_b/research_csdi.py`: attention theo time và feature, gated residual/skip; ghi rõ đây là compositional adaptation |
| Training condition dùng median, test condition dùng Hron/Aitchison | Confound train/inference, không kiểm tra được giá trị initializer | 4 training mask banks; chạy initializer thật sau masking, cross-fit loại toàn bộ location của query; lưu cell provenance |
| Seeds khác giữa các ablations; chỉ có Mask-only CLR | Conditioning gate không đối sánh đủ | Seed 42 chung; masks/windows/noising/sampling policy chung; Mask-only riêng cho cả CLR/ILR/HKGLR |
| Schedule 20 bước còn terminal alpha-bar khoảng 0.633 nhưng reverse khởi tạo Gaussian | Prior mismatch lớn | Quadratic beta 1e-4 → 0.5, fixture kiểm tra terminal alpha-bar <0.02, giữ 20 steps |
| Full-LR labels gồm các hàng complete nhưng zero-sum | Nhãn composition không định danh | Chỉ train hàng complete và tổng dương; outer train 388 rows, loại 19 complete zero-sum rows |
| Eligibility phụ thuộc prediction positive sum | Méo so sánh: Init-only 99 rows, diffusion 101 rows | Eligibility dựa riêng ground truth; zero-sum prediction vẫn được chấm ở LR positivity view, không tự loại |
| Thiếu parity test, checkpoint và sampling/failure provenance | Chưa chứng minh numerical core và khó tái lập | Reference snapshot pin commit, parity fixtures, 24 checkpoints, samples, runtime/traces, cache/provenance, config và hashes |
| Chỉ kiểm finite nên count scale rất lớn vẫn “PASS” | Sampling health thiếu chẩn đoán | Guard kỹ thuật đăng ký trong CONFIG; failure logs có CSV row IDs ở validation/test; không clipping ngầm |

Các sửa đổi nằm trong source mới và runner mới; không sửa artifacts E1/E2 hay predictions E3 cũ để tạo kết quả thuận lợi. E1 k=4 tiếp tục dùng như giá trị đã chọn trong inner-CV fold 0. Pilot này không tìm lại k hoặc chọn seed dựa test.

## 4. Numerical reference và giới hạn parity

Snapshot official CSDI: commit `7f24a436f08d98853a6b43d4f7f04e5a65ecdf27`, lưu `ref/csdi_official/` cùng LICENSE và SHA-256. Nguồn: [main_model.py tại commit đã pin](https://github.com/ermongroup/CSDI/blob/7f24a436f08d98853a6b43d4f7f04e5a65ecdf27/main_model.py), [diff_models.py](https://github.com/ermongroup/CSDI/blob/7f24a436f08d98853a6b43d4f7f04e5a65ecdf27/diff_models.py).

Fixture thực thi các method loss/impute nguyên gốc trích AST, cùng oracle denoiser, schedule và random tensors; so forward/loss/reverse với tolerance 1e-12. Reference không bị sửa. Đây là numerical parity của DDPM/loss, **không** phải weight/output parity của toàn denoiser, không tái lập kết quả paper.

Adapter dự án dùng tensor B,L,K; input condition 85 slots (=5×17): raw visible log1p, mask, initializer log1p, fallback, effective_k. No-init và Mask-only zero các slots không được phép dùng; time embedding riêng. Noisy target branch nhận full LR; không cắt loss theo raw observed mask. Chỉ complete positive compositions làm full-LR ground truth.

## 5. Thiết lập đã chạy

| Thiết lập | Giá trị |
|---|---|
| Outer fold / test | 0 / Canada, 109 rows |
| Frozen mask | `fold0_random_r0.30_seed42`, 545 hidden cells |
| Eligible rows / scored hidden | 101 / 505; 8 zero-sum truth rows loại theo eligibility E0 |
| Inner validation | Denmark, chọn alphabetically-first complete outer-train location trước test |
| Complete positive train rows | Inner 285; outer 388 |
| Windows | Inner 38; outer 51; window length ≤8, không padding/overlap/mix location |
| Pilot training | 20 epochs, 20 DDPM steps, 2 MC samples, width 16, 2 residual blocks |
| Training masks | 4 banks cố định, 5 hidden parts/row, cycle qua epochs |
| Initialization | Hron-2a và Aitchison-complete k=4; query location không được làm donor của chính nó |
| LR / positivity | CLR-16, ILR-16, HKGLR-16 actual; float64, natural log; test dùng frozen delta=0.5 |
| Scaler / references | Fit từ partition train; scalar RMS latent scale, đảo scale trước final projection |
| HKGLR H | Omicron, Delta, Alpha, recombinant, Beta; inner/outer train độc lập cho cùng set ở pilot này |
| Projection | Chỉ cuối reverse; CLR sum-zero, HKGLR mean_H=0, ILR none |
| Inference | IID Gaussian khởi tạo, no clipping, mean 2 latent samples rồi inverse; raw observed locking |
| Selection | Checkpoint epoch 20 cố định; validation decision ghi trước outer test; không tuning trên Canada |

Channels được match 16. CLR/HKGLR có 19,713 parameters, ILR 19,697 do 17 vs16 feature embeddings; chênh 16 parameters đã ghi nhận. Outer train mỗi model khoảng 3.75–4.56 s và sample 0.56–0.78 s trên CPU ở lần chạy này; initializer cache preparation tách khỏi model runtime. Runtime lịch sử Init-only không được bịa thêm khi reuse E1.

E3 có 14 entries: 2 Init-only + 6 Init+CSDI + 3 No-init + 3 Mask-only. E4 thêm 4 classical baselines thành 18. Số lượng nhiều hơn pilot cũ vì bổ sung đủ matched Mask-only controls, không phải thêm transform hay initializer ngoài phạm vi.

## 6. Conditioning gate: điểm FAIL cụ thể

Lower M2 is better. Gate yêu cầu Mask-only M2 > No-init M2 trên từng transform.

| Transform | Validation No-init | Validation Mask-only | Gate validation | Test No-init | Test Mask-only | Gate test |
|---|---:|---:|---|---:|---:|---|
| CLR-16 | 173.801968 | 176.040822 | PASS | 149.547630 | 151.910847 | PASS |
| ILR-16 | 138.985062 | 138.194457 | **FAIL** | 154.516581 | 158.367867 | PASS |
| HKGLR-16 actual | 177.279364 | 175.328315 | **FAIL** | 114.894039 | 115.987770 | PASS |

Vì ILR/HKGLR validation FAIL, không kết luận mô hình đã sử dụng conditioning đáng tin cậy. Test PASS không được dùng để đổi validation decision. Một seed và hai samples không đủ chứng minh khác biệt ổn định.

## 7. E4: điểm FAIL efficacy cụ thể

Gate dùng riêng `method_type == init_csdi`, so với cùng initializer Init-only, cùng truth row set; yêu cầu gain M2 **strictly >10%** và nonzero CLR MAE không xấu hơn. No-init/Mask-only không được lẫn vào gate này.

| Initializer | Init-only M2 | CLR Init+CSDI | ILR Init+CSDI | HKGLR Init+CSDI | Gate |
|---|---:|---:|---:|---:|---|
| Hron-2a | 9.714041 | 152.131998 | 149.240465 | 114.250986 | **FAIL cả 3** |
| Aitchison-complete | 9.895114 | 152.215085 | 149.068136 | 114.077675 | **FAIL cả 3** |

Nonzero CLR MAE: Hron Init-only 1.96331; Aitchison Init-only 1.93122. Sáu initialized diffusion branches nằm khoảng 3.62741–4.07550, đều xấu hơn. Đây là pilot negative result, không sửa threshold, lựa seed test khác hoặc nâng finalist để làm gate PASS.

Classical baseline M2: Linear 0.872296; LOCF+NOCB 1.02527; train-mean-scaled 73.2063; row-positive-mean sensitivity 173.270. Linear/LOCF dùng cùng-variant visible neighbors trong scenario random-cell; không suy ra thắng ở whole-variant missingness. Whole-variant là scenario primary theo revised §5.1 và chưa được chạy trong nhiệm vụ audit này.

## 8. Sampling/scale: phân biệt số hữu hạn và output đáng sử dụng

Test diffusion có 7–15 flagged rows/method; maximum predicted-sum / max(visible-sum,1) tới khoảng 2.49×10^9. Validation diffusion có 10–14 flagged rows/method. Finite float64 không đủ đảm bảo thang counts hợp lý.

Guard 1000 là **engineering flag** dưới Protocol U, không phải biological cutoff hay primary accuracy metric. Không dùng true mass, không loại flagged rows khỏi M2, không cap predictions. Unknown-mass restoration dùng median ratio giữa visible positive counts và predicted composition. Các predicted visible probabilities rất nhỏ có thể làm scale rất lớn; đây là cơ chế cần khảo sát trên train/validation.

Initializer cũng được chẩn đoán: test Aitchison Init-only có 1 flagged row, stable CSV ID 1511, ratio 1262.375; Hron test không có flag. Validation hai Init-only mỗi phương pháp có 1 flagged row. Vì vậy cần audit cả restoration/anchor lẫn diffusion, không quy mọi raw-count explosion riêng cho denoiser. 27 failure records cấp phương pháp/partition lưu các stable row IDs; chúng không phải 27 unique rows.

Native JSD của một số Init-only outputs undefined vì zero predicted mass: JSON trả null kèm status explicit, không đổi denominator. Raw-count MAE, prevalence bins và scale diagnostics không dùng chọn method. M2/M3 và CLR MAE vẫn chấm cùng row set đã frozen theo truth eligibility.

## 9. Đối chiếu review người dùng

| Nhận xét | Kết quả xử lý |
|---|---|
| S1 `methods` rỗng | Sửa schema v2 ở runner mới; metrics/predictions nested, count và key sets assert; regression test bắt lại lỗi cũ |
| S2 raw-scale explosion | Ghi rõ raw MAE diagnostic-only; thêm scale logs cho diffusion và initializer; không dùng true total hay clipping ngầm |
| S3 temporal baselines mạnh ở random-cell | Ghi rõ scenario limitation ở E4 report và bản audit này |
| N1 edge extrapolation Linear | Giữ behavior đã công bố; negative extrapolation được clamp, có counts/fraction |
| N2 leading fill của LOCF | Đặt tên `locf_nocb`, báo leading NOCB cell count |
| N3 lọc nhánh gate ngầm | Lọc explicit `method_type == init_csdi`, test loại No-init khỏi gate |
| “Không có blockers” | Không giữ kết luận này: E3 cũ thiếu architecture/protocol prerequisites; E3 revised vẫn FAIL gates |

Prevalence strata còn được sửa thêm: positive fraction chia số **observed nonmissing train cells**, không coi NaN là zero. Feature ties dùng variant name. Donor provenance frozen E1 còn nguyên; training cross-fit donor provenance mới lưu gzip theo mask bank.

## 10. Kiểm chứng và provenance

- `artifacts/e3_csdi_revised/verification_tests.txt`: 49 PASS, gồm legacy tests và revised E3/E4 fixtures. Poisoning hidden inputs/queries không đổi conditioning/initializer; unknown NaN labels bị loại trước arithmetic; window boundaries và raw/LR mask dependency được kiểm tra.
- `artifacts/e3_csdi_revised/verification.json`: official snapshot hashes, frozen inputs, checkpoints/predictions/cache preservation, exact restoration và stable row IDs.
- `artifacts/e4_pilot_revised/verification.json`: 18 methods, frozen signature, independently recomputed M2/M3/CLR MAE tolerance 1e-12, 101 rows/505 hidden, observed zeros chính xác, gates và hashes.
- `file_hashes.json` ở cả hai outputs chứa artifact và current audited code hashes. Manifest E3 tách training code hashes trước khi hoàn tất report. Một correction riêng chỉ sửa stale scale count Init-only trong bảng E3 từ 0→1; `report_correction.json` và E4 `upstream_integrity.json.postrun_report_correction` ghi previous/new hashes. Predictions, metrics và checkpoints không thay trong correction này.
- Data SHA-256 vẫn là `bbb1aacdf46ebef345ca818f086d5586f4aded45475b00b6ee9bd89fc94a90ab`. E0–E4 cũ không đổi; không sửa các PDFs người dùng đã modified.

Code chính: `src/stage_b/research_csdi.py`, `src/evaluation/research_pilot.py`, `scripts/run_e3_revised.py`, `scripts/run_e4_revised.py`. Audit helpers: `finalize_e3_review.py`, `refresh_review_reporting.py`, `verify_e4_review.py`. Tests: `tests/unit/stage_b/test_e3_revised.py`, `test_e4_revised.py`.

Lệnh cho run mới dùng `.venv-e3-revised/Scripts/python.exe -B scripts/run_e3_revised.py`, sau đó finalize verification và E4. Runner từ chối overwrite một run đã hoàn tất. **Không chạy lại E3 hiện tại**: train/predictions đã có. `verify_e4_review.py` dùng để kiểm chứng kết quả hiện tại mà không train.

## 11. Việc còn lại theo revised plan, trước E5

1. Giữ trạng thái E5 BLOCKED, `finalists=[]`. Điều tra vì sao conditioning không giúp trên Denmark, dựa epsilon-loss traces, noise schedule, scalar latent distribution và seed-paired validation. Không dùng Canada để chọn hyperparameters hoặc điều chỉnh gate.
2. Audit unknown-mass scale/anchor trên train/validation, gồm Aitchison row flagged và các rows thiếu positive visible anchor. Nếu thử clipping/scaling thay thế, đăng ký nhánh sensitivity riêng, thresholds train-only, báo clipping fractions; không sửa observed zeros hoặc âm thầm thay outputs hiện có.
3. Revised §11 cảnh báo <200 eligible windows. Pilot hiện chỉ 38 inner/51 outer windows. Spec người dùng cố định channels16 nên run này giữ16; thử mô hình capacity8 hoặc cách tạo windows hợp lệ cần một cấu hình validation mới đã đăng ký, không gọi thử nghiệm hiện tại đã giải quyết thiếu dữ liệu. Chưa tự chạy thêm một vòng refinement.
4. Dimensionality control CLR-17 model và HKGLR random/high-missing diffusion controls chưa chạy trong E3/E4 pilot này. E2 transform roundtrip không thay thế các diffusion controls ấy. Không kết luận Q3/Q4 chỉ từ ba nhánh actual pilot.
5. Chỉ sau conditioning/sampling prerequisites và efficacy >10% đạt trên validation/registered pilot mới chọn ≤3 finalists rồi mở E5. Đánh giá whole-variant cùng random-cell đúng revised matrix; các kết quả Canada hiện tại chỉ mô tả feasibility của pilot.

Không có p-values/CI hoặc claim diffusion kém trên mọi location. Những điểm FAIL đã nêu là kết quả của cấu hình pilot này; đường xử lý tiếp theo phải dùng train/validation và lưu một experiment mới.
