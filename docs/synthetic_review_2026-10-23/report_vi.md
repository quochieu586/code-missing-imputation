# Review synthetic data — logic, code, artifacts và kết quả

Ngày review: 2026-10-23, Asia/Saigon. Kết luận: **REQUEST CHANGES**. Có **2 finding P1 và 3 finding P2** cần xử lý/giải quyết trước khi coi pipeline là benchmark hoàn chỉnh. Không có P0 được xác nhận trong phạm vi kiểm tra.

Điểm quan trọng: **ba dataset đã phát hành đang nhất quán với generator v1 và tái sinh chính xác**. Các lỗi phát hiện chủ yếu ở ranh giới evaluator, phạm vi metric, dispatch cấu hình và xuất artifact; chúng không chứng minh các dataset hiện tại đã bị hỏng.

## 1. Phạm vi và bằng chứng

Áp dụng skill `C:/Users/admin/.codex/skills/thermo-nuclear-code-quality-review/SKILL.md`.

Đã đọc module synthetic, wrapper R và nhánh nguồn ampute liên quan, tests mới, hợp đồng RawPanel/ConditioningPanel/FitScope, CountBaselineAdapter, metadata encoder, tài liệu yêu cầu và artifacts đã bàn giao. Codebase hiện còn có `scripts/run_synthetic_easy_baselines.py` và kết quả easy; review bao gồm cả phần đó. Đây là kiểm định implementation/artifact, không phải một lần systematic review paper mới hay xác nhận lại toàn bộ thuật toán baseline.

Không dùng git diff đơn thuần làm phạm vi vì nhiều file liên quan hiện vẫn untracked. `audit_results.json` lưu hash/line count của snapshot được kiểm tra. Không sửa implementation, dataset hoặc prediction gốc. Fault injection chỉ thực hiện trên bản sao tạm trong thư mục review.

Artifact của review:

- `audit_synthetic.py`: script tái lập phép kiểm và các probe.
- `audit_results.json`: kết quả máy đọc được, gồm hash/replay, thống kê, các lỗi tái hiện và metric tính lại.
- `findings.json`: danh sách finding, mức ưu tiên, vị trí code và hướng khắc phục.

## 2. Findings

### F1 — P1: evaluator chấm được ground truth sai mà không phát hiện

Vị trí: [datasets.py:44](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/nextgen/src/longitudinal_impute/synthetic/datasets.py:44), đặc biệt phép index tại dòng 45–53.

**Lỗi:** `load_conditioning` kiểm một số hash public, nhưng `load_evaluation` đọc `private/complete_counts.csv` và `private/ground_truth.npz` không kiểm hash dù manifest đã ghi chúng. Sau đó nó lấy vị trí hàng trong CSV để index NPZ. NPZ không có row_ids dùng để kiểm tra ánh xạ này.

**Đã tái hiện ba trường hợp:**

1. Đổi một hidden count từ 0 thành 1 trong CSV private. Loader trả label mới 1 mà không báo checksum mismatch.
2. Hoán vị entity-axis của latent p trong NPZ, giữ nguyên shape. Loader vẫn trả latent truth đã đổi.
3. Chỉ đảo thứ tự hàng của CSV private, không sửa giá trị theo entity/date và không sửa NPZ: count labels vẫn căn đúng theo key, nhưng latent p bị căn sai; chênh lệch tuyệt đối lớn nhất **0.7113467420**. Đây là lỗi ghép identity, không chỉ là ví dụ sửa một giá trị tùy ý.

Bằng chứng máy: `probes.private_csv_mutation`, `probes.latent_npz_permutation`, `probes.private_csv_reorder` trong `audit_results.json`.

**Tác động:** metric/ranking có thể được tính với labels hoặc latent p không thuộc sample đang dự đoán. Điều này phá mục đích benchmark biết ground truth. Bản bàn giao hiện tại chưa gặp lỗi này: toàn bộ hash và replay đều đúng; lỗi là loader chưa bảo vệ được ranh giới khi sử dụng/bàn giao về sau.

**Hướng sửa:** evaluator phải kiểm hash private trước khi chấm; truth phải có row identity rõ ràng, độc lập với thứ tự vật lý của một CSV khác. Kiểm format version, shapes, keys và tính nhất quán giữa counts/N/p/strata trước khi tạo labels. Ưu tiên một truth contract có typed fields/row_ids; không tiếp tục dựa vào hai dictionary/array được ghép ngầm bằng vị trí.

**Acceptance:** sửa private file phải bị từ chối; row permutation không bao giờ âm thầm làm đổi latent truth của cùng row_id; kiểm positive/sampling-zero/true-zero bằng labels đã căn identity. Model-side loader vẫn phải chạy khi không có private directory.

### F2 — P1: kết quả easy chưa đánh giá latent p theo yêu cầu hai ground truth

Vị trí: [run_synthetic_easy_baselines.py:65](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/scripts/run_synthetic_easy_baselines.py:65), hàm `score`, các dòng 66–88; selection metric tại dòng 161–164.

**Lỗi phạm vi đánh giá:** `ratio_error = (prediction - sampled_counts) / N`. Vì vậy `ratio_mae` đang là sai số so với **X/N**, không phải so với **latent p**. Đối số `latent` chỉ được dùng để lấy `strata`; `latent['p']` không được dùng trong score. Dữ liệu và loader có cung cấp p, nhưng bộ kết quả hiện tại chưa thực hiện phần đánh giá thứ hai người dùng yêu cầu.

**Phân biệt:** các con số count/ratio đã lưu không tính sai theo định nghĩa mà script đang dùng; thiếu nằm ở việc chưa báo metric khôi phục p. Không được diễn giải ratio metric hiện tại là latent-proportion recovery.

Review đã tính lại từ sáu prediction đã lưu, không fit lại và không retune trên test:

| Mask | Method | MAE so với X/N đã báo | MAE so với p tính thêm trong review |
|---|---|---:|---:|
| MCAR | Hron | .00626880 | .00609235 |
| MCAR | MICoDa | .07420998 | .07405202 |
| MCAR | Tsagris | .00601851 | .00599159 |
| MAR | Hron | .00687208 | .00659672 |
| MAR | MICoDa | .07497759 | .07472772 |
| MAR | Tsagris | .00631817 | .00603028 |

X/N đầy đủ cũng khác p: MAE trên hidden test cells là .00194544 với MCAR và .00228417 với MAR. Đây là khác biệt do sampled truth và latent truth; không thể đổi nhãn metric để thay thế phép tính.

Bằng chứng máy: `results_recomputed[*].metrics` và `latent_strata` trong `audit_results.json`.

**Hướng sửa:** canonical scorer phải tách `count_*`, `sample_ratio_*`, `latent_p_*`, cho overall và cả ba nhóm zero/positive. Khai báo trước objective dùng để chọn hyperparameter; có thể giữ objective counts/X/N hiện tại và báo latent p như một endpoint riêng. Không dùng kết quả latent trên test để chọn lại model.

**Acceptance:** test phải dùng một sample có X/N khác p và chứng minh hai score khác nhau; cả hai đều chỉ chấm hidden cells. Existing predictions đủ để bổ sung latent metrics, không cần sinh lại dữ liệu cho việc này.

### F3 — P2: scenario sai được chấp nhận rồi âm thầm chạy một scenario khác

Vị trí: [cli.py:79](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/nextgen/src/longitudinal_impute/synthetic/cli.py:79) và nhánh `else` tại dòng 96–99; kiểm trùng trong `generator.py:118`.

**Đã tái hiện:** gửi spec có `scenario='table1_design'`, mechanism MCAR và wave_time hợp lệ. CLI trả **exit code 0**, tạo output có diagnostics `scenario='fixed_wave'`.

**Nguyên nhân:** danh sách capability toàn module được dùng như validator của enum scenario. `table1_design` là capability IMPLEMENTED nên `require_capability` cho qua, mặc dù nó không phải scenario masking hợp lệ. Nhánh `else` coi mọi giá trị còn lại là fixed wave. Đây là dispatch sai, không phải thông báo lỗi chưa đẹp.

Bằng chứng: `probes.wrong_scenario` trong `audit_results.json`.

**Tác động:** spec và mask thực tế có thể không mô tả cùng một thí nghiệm. Các tên NOT-READY đang được khai báo vẫn bị chặn trong tests; probe này không chứng minh latent-p MNAR hay random blocks đã được mở. Nó chứng minh lời hứa reject unknown/unsupported scenario chưa được bảo đảm ở boundary.

**Hướng sửa:** `MaskSpec` có enum riêng cho mechanism/scenario, validate một lần trước mọi I/O; chỉ dispatch đúng hai scenario được phép. Capability registry dùng để báo trạng thái tính năng, không làm parser cho mọi loại giá trị. Bỏ nhánh `else -> fixed_wave` ngầm định.

### F4 — P2: export gọi ngược CLI và để lại publication nửa chừng

Vị trí: [generator.py:111](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/nextgen/src/longitudinal_impute/synthetic/generator.py:111), ghi final path tại dòng 124, gọi `mask_cli(argv)` tại dòng 145–148; `cli.py:43` gọi ngược lại export.

**Vấn đề kiến trúc:** luồng hiện tại là CLI.generate → export_dataset → CLI.main → argparse → _mask_csv → đọc lại CSV truth → mask → ghi CSV → export đọc lại mask CSV. Validation scenario/capability còn bị lặp giữa export và CLI. Lớp argparse/print/exit-code đã trở thành business service cho API Python.

**Đã tái hiện:** export với spec thứ nhất hợp lệ, spec thứ hai có case_probability=2. Thư mục đích đã chứa private truth và mask thứ nhất, nhưng không có manifest COMPLETE. Chạy lại cùng output với spec đã sửa thì gặp `FileExistsError`.

Bằng chứng: `probes.non_atomic_export`, gồm danh sách file đã ghi và retry error.

**Phạm vi tác động:** loader hiện từ chối bundle thiếu COMPLETE manifest nên chưa có bằng chứng partial bundle bị chấm như một bộ hoàn chỉnh. Tuy nhiên quá trình tạo/bàn giao không có tính giao dịch, không retry được bằng cùng lệnh và buộc người dùng xử lý artifact nửa chừng.

**Code-judo cần làm:** đưa apply-mask/write-bundle thành service Python có kiểu và exception rõ ràng; cả CLI và export gọi service đó. CLI chỉ parse args, gọi service, định dạng thông báo. Materialize/validate toàn bộ specs một lần, tạo bundle trong staging directory, validate xong mới publish bằng rename trên cùng filesystem. Có thể xóa hẳn vòng gọi CLI lồng nhau và round-trip CSV nội bộ, thay vì chỉ chia hàm hiện tại thành nhiều wrapper.

**Acceptance:** một spec lỗi không được để một final bundle nửa chừng; lệnh đã sửa có thể chạy lại; API Python không tự in CLI status và không tự parse argv; masks với các valid specs/seed hiện tại vẫn giữ nguyên. Không cần parallel hóa ba lần gọi R chỉ để tối ưu nhỏ; ranh giới ownership và publication là vấn đề chính.

### F5 — P2: cần giải quyết nghĩa của residual ở cấp counts trước khi đóng đặc tả

Vị trí: [generator.py:80](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/nextgen/src/longitudinal_impute/synthetic/generator.py:80), residual beta và multinomial tại dòng 80–92.

**Quan sát:** latent residual dương nên tổng 12 latent proportions nhỏ hơn 1. Tuy nhiên bộ hard có **178/7200 samples** với residual count bằng 0; tại đó tổng named-component counts bằng N. Easy/medium có 0 trường hợp như vậy trong seed hiện tại.

Yêu cầu ban đầu nói thành phần dư làm tổng component nhỏ hơn N. Đồng thời sampler multinomial trên đủ 13 phần đã được chốt. Với multinomial, residual probability dương không bảo đảm residual count dương: `P(X_residual=0 | N,r)=(1-r)^N > 0` khi 0<r<1 và N hữu hạn. Đây là suy dẫn từ sampler đã hiện thực, không phải một phát biểu được gán cho Schouten.

**Phân loại:** mâu thuẫn/độ mơ hồ của acceptance contract. Không kết luận implementation multinomial sai. Tài liệu v1 đã công khai residual count có thể bằng 0, nhưng việc đó chưa đồng nghĩa yêu cầu strict `<N` ban đầu đã được đáp ứng.

**Hướng xử lý:** đóng rõ contract: latent residual luôn dương; named count sum ≤N và có thể bằng N. Nếu strict `<N` ở mọi sample là bắt buộc thì phiên bản này chưa đáp ứng và cần một quyết định thiết kế mới. Không tự cắt counts, ép residual ≥1 hoặc đổi sampler trong lượt review; các cách đó thay đổi mô hình sinh đã chốt.

## 3. Đối chiếu logic đã thống nhất

| Yêu cầu | Kết quả kiểm tra | Trạng thái |
|---|---|---|
| Tên artificial, không gắn case study | entity_001, component_01..12, ngày làm chỉ số thời gian | Đạt |
| Standard 60×120×12; small/large 30/120 entity | Config và tests giữ time/component cố định | Đạt trong code; artifacts bàn giao là standard |
| Có latent p và counts sampled từ p | p và residual p riêng; multinomial 13 phần; exact count conservation | Đạt |
| Mẫu hình thay thế nhau, chung schedule, lệch thời điểm/dominance/noise | Cùng peak schedule; shift entity, log dominance và AR noise | Đạt theo thiết kế v1; chưa chứng minh độ khó bằng model |
| p=0 trước onset | Mọi ô trước onset đều p=0 và X=0 | Đạt |
| Tail dương sau khi component xuất hiện | Mọi ô sau onset có p>0 trong artifacts hiện tại | Đạt trong ba realization; tail rule là giả định v1 |
| True zero khác sampling zero | Strata p=0 / p>0,X=0 / X>0 kiểm lại đúng | Đạt |
| N biến đổi theo thực thể và thời gian | Lognormal entity+AR time, CV log không bị áp hai lần | Đạt theo moment mục tiêu; không ép sample moments |
| Residual không cấp trực tiếp cho model | Chỉ ở private/evaluator; complete rows vẫn cho suy ra residual count từ N và tổng counts | Đạt ranh giới đã thiết kế; strict `<N` xem F5 |
| MAR dựa N và trait quan sát | Target placeholders, weights target=0; N âm, trait dương | Đạt với cấu hình hiện tại |
| Cùng rule train/test, split entity-disjoint | 36/12/12 entity, giống nhau ở ba profile | Đạt; tỷ lệ 60/20/20 là lựa chọn triển khai v1 |
| Không dùng hidden p/counts để quyết định MCAR/MAR | Source input cho target là zero placeholders; test đổi truth giữ mask | Đạt ở nhánh hiện tại |
| Ground truth không lọt vào model input | Conditioning chỉ có entity/time/N/targets/trait | Đạt ở các file đã bàn giao |
| Khôi phục cả counts và latent p | Labels/p có sẵn; benchmark mới chấm counts và X/N | Chưa đủ — F2 |
| Loader reliable khi data bị đổi/đổi thứ tự | Public checks có; private/latent identity thiếu | Chưa đạt — F1 |
| Chỉ Schouten cho missingness | Pin original R code; không thêm nguồn missingness thứ hai | Đạt trong phạm vi đọc/test |
| NOT-READY không tự hiện thực | MNAR latent-p, random blocks, whole trajectories, beta/rate grids vẫn chặn | Đạt với tên tính năng hiện tại; invalid-scenario dispatch xem F3 |
| Plug-and-play | Nạp bằng hợp đồng existing baselines, giữ observed cells và budget | Đạt mức interface; chưa đạt mức API/artifact robustness — F1/F3/F4 |

### Phân biệt thông số đã chốt và giả định triển khai

**Đã chốt:** ba profile offset SD 1/4/8, dominance SD .10/.35/.70, noise SD .05/.15/.35, mean N 2000/500/100 và CV .30/.60/1.00; residual mean .10, SD .03; multinomial; quy mô; observed trait cho MAR; ground truth/private boundary; thu hẹp missingness theo Schouten.

**Được chọn trong implementation và đã ghi tài liệu, chưa phải thông số từ paper hay giá trị người dùng từng chốt riêng:** peak linspace(0,119,12), Gaussian width 8, onset lead 16, tail 1e-4, AR correlation .8, initial component onset clamp, beta residual độc lập theo thời gian, chia variance log-N đều entity/time, trait Normal độc lập, split 60/20/20. Chúng là giả định cụ thể hóa v1, không được dùng như bằng chứng paper đã validate generator counts.

`GeneratorConfig` hiện expose vài giả định này nhưng các mean/CV/SD chính nằm trong dictionary PROFILES. Vì vậy “control được” hiện chủ yếu là chọn preset dễ/vừa/khó, chưa phải API đầy đủ để override từng mean/CV/dominance ngoài preset. Đây là giới hạn phạm vi, không phải phép tính sai trên ba cấu hình đã yêu cầu.

## 4. Dữ liệu và missingness thực tế

Cả ba profile: **mọi file hash trong manifest đều khớp**, và **mọi mảng do generate cùng seed tái tạo đều bằng chính xác** với NPZ đã lưu. Kiểm p hữu hạn/không âm, closure, counts integer/nonnegative/conservation, onset zeros, strata và split đều qua.

| Profile | Mean N | CV N | True zero | Sampling zero | Positive count | Residual-count-zero samples |
|---|---:|---:|---:|---:|---:|---:|
| easy | 2059.97 | .3039 | 32883 | 22218 | 31299 | 0 |
| medium | 529.60 | .6190 | 32880 | 25696 | 27824 | 0 |
| hard | 109.60 | 1.0490 | 32964 | 29543 | 23893 | 178 |

Mỗi profile có 86400 named-component cells. Mean/CV hữu hạn không bằng đúng target là hệ quả sampling, đặc biệt có entity-level variation; không coi đó là lỗi quota. Không chỉnh data để ép về moment đích.

MAR có đúng chiều ở artifacts hiện tại, tính theo tỷ lệ sample nhận missing pattern:

| Profile | N thấp: bottom 20% | N cao: top 20% | Trait thấp: bottom 20% | Trait cao: top 20% |
|---|---:|---:|---:|---:|
| easy | .4417 | .1160 | .1438 | .5090 |
| medium | .4125 | .1271 | .1410 | .5167 |
| hard | .3701 | .1440 | .1361 | .5104 |

Đây là diagnostics thực nghiệm, không phải chứng minh nhân quả hay bảo đảm monotonic trong mọi seed. Source cấu hình dùng dấu weights phù hợp; không đọc hidden p để ra missingness.

MCAR target-cell rate .100602 giống nhau ở ba profile. MAR lần lượt .096343/.097269/.095972. Spec case_probability=.3, mỗi pattern che 4/12 target, nominal cell rate .1. Không có exact-rate guarantee; các giá trị thực tế này không chứng minh calibration bị sai.

Mask hiện tại là declared multivariate patterns, không phải ô độc lập Bernoulli và không phải random time blocks. Fixed wave là một wave đăng ký trước. Phần random blocks vẫn NOT-READY; không đánh đồng hai khái niệm để tuyên bố đủ grid ban đầu.

## 5. Kết quả baseline hiện có: dùng được tới đâu

Đã đối chiếu hash bundle và code source được run manifest ghi lại: tất cả khớp snapshot đang review. Tính lại sáu prediction của easy cho thấy:

- Row IDs đúng thứ tự panel; predicted counts hữu hạn/không âm.
- Observed values giữ nguyên; không có vi phạm tổng counts ≤N.
- Count MAE và empirical-ratio MAE khớp số đã báo.
- Diagnostics MICoDa ở cả MCAR/MAR có `converged=true`, iterations=7 tại các bản ghi tương ứng; không có bằng chứng để quy lỗi MICoDa kém ở bộ này cho việc dừng ở max_iter=50.
- Tuning dùng validation, final fit dùng train+validation, test chỉ dùng báo cáo trong luồng code đã đọc. Không thấy fit donor/preprocessor dùng test labels trong đường chạy này.

Kết quả easy cho Tsagris/Hron thấp hơn MICoDa ở các metric hiện tại; review không khẳng định thứ hạng tổng quát. Mới có một seed, scale standard, easy, hai declared masks, một mức nominal cell rate. Chưa có kết quả baseline medium/hard, sensitivity small/large hoặc phép đánh giá uncertainty cho ranking.

Run manifest ghi `metadata_used=[]`, adapter dùng MetadataEncoder mặc định không encode trait. Trait có trong ConditioningPanel, nhưng điều đó không đồng nghĩa các baseline của run hiện tại đã sử dụng nó làm predictor. Đây là cấu hình cần ghi rõ khi diễn giải MAR; không phải bằng chứng trait bị xóa khỏi dataset. Nếu mở covariates bằng cơ chế hiện có thì phải đăng ký một run/protocol mới, không âm thầm thay kết quả hiện tại.

True-zero/sampling-zero/positive-count empirical errors đã tách trong summary. Latent-p errors theo ba nhóm mới được tính bổ sung trong review JSON; chưa được tích hợp thành scorer chính. Có lưu zero-predictor reference trong audit để kiểm tra cách diễn giải, không bổ sung phương pháp imputation vào benchmark.

## 6. Table 1 và ranh giới evidence

135 reference design runs = 5 conditions × 9 rho × 3 replicates: hash mọi file đúng; Y1/Y2 luôn được che cùng nhau, X1 không bị che. Tests cũng kiểm weights và pattern fixtures. Nguồn R được pin commit `61083e667fa41cc4bc655492591b100bf8a7e443` và kiểm hash.

Các phép kiểm trên xác nhận provenance/pattern implementation ở phạm vi fixtures đã chạy. Chúng **không** biến 3 replicates thành 1000, không tái lập PMM/bias/coverage và không chứng minh generator compositional counts có độ khó đúng thứ tự. Snapshot R hiện đại đã được tài liệu phân biệt với phiên bản năm 2018; review không kết luận bitwise parity với software release năm đó.

Nguồn/locator và Claim–Evidence–Analysis của lựa chọn phương pháp nằm ở `docs/synthetic_missingness_claim_evidence_audit_vi.md` và `docs/synthetic_missingness_single_source_scope_vi.md`. Review này không mở rộng inference của các evidence đó.

## 7. Đánh giá kiến trúc theo thermo-nuclear bar

Không có handwritten synthetic module nào vượt 1000 dòng; lớn nhất 186 dòng. Vendored source giữ nguyên để bảo toàn provenance, không đề xuất sửa source R chỉ để làm gọn. Vấn đề chính là ownership/contract, không phải kích thước file.

Ba thay đổi cấu trúc đáng làm nhất:

1. **Bỏ việc API gọi ngược CLI.** Một MaskSpec validator + service apply-mask; CLI và bundle exporter chỉ là hai caller. Xóa duplicate checks, argv construction, parser lồng nhau và vòng đọc/ghi CSV để vận chuyển trạng thái nội bộ.
2. **Một bundle contract làm chủ identity và integrity.** Public loader/evaluator dùng cùng version/schema/row-key validation; evaluator bổ sung kiểm private. Truth có row_ids, không dùng thứ tự CSV khác làm truth index ngầm.
3. **Một scorer làm chủ hai ground truth.** Tách số đo sampled counts, sampled proportions và latent p; runner chỉ chọn protocol và lưu kết quả. Không thêm các `if latent` rải rác vào report/CSV/selection để vá thiếu metric.

Ghi staging rồi publish là phần của bundle writer, không phải một loạt flag/retry đặc thù thêm vào từng lệnh CLI. Không cần tạo nhiều lớp trừu tượng mỏng hoặc một plugin framework mới cho ba kiểu dữ liệu cố định này.

Quan sát nhỏ đã xác nhận: `GeneratorConfig(residual_sd=-.03).validate()` vẫn qua vì validation dùng bình phương. Nên kiểm positivity trực tiếp trong config contract; chưa thấy làm thay đổi dataset mặc định nên không nâng thành một finding độc lập ưu tiên cao.

Tài liệu yêu cầu hiện trộn lịch sử “chưa chọn/chưa triển khai” với cập nhật đã triển khai. Nên có một bảng contract hiện hành với trạng thái agreed/default/NOT-READY và giữ lịch sử ở file riêng; không dùng các câu cũ làm đặc tả mới.

## 8. Thứ tự khắc phục và điều kiện chấp nhận

1. F1: bảo vệ ground truth/identity trước mọi phép chấm. Bổ sung các regression tests từ ba probe đã tái hiện.
2. F2: thêm canonical dual-truth scorer, kiểm bằng ví dụ X/N khác p; bổ sung metric từ prediction gốc vào artifact mới, không ghi đè report lịch sử và không retune bằng test.
3. F3/F4: MaskSpec typed/strict, application service và atomic bundle publication. Giữ source R/calibration/seed và valid-mask behavior nguyên vẹn.
4. F5: ghi một acceptance contract residual nhất quán với multinomial; strict `<N` nếu vẫn bắt buộc thì đánh dấu chưa đáp ứng tới khi có quyết định thiết kế riêng.
5. Chạy lại tests hiện tại cùng các regression probes thành test chính thức, rồi mới mở rộng run medium/hard/small/large. Tất cả missingness NOT-READY tiếp tục chặn; các sửa trên không cần paper/method mới.

Existing test suite: **128 passed trong 49.48 giây** ở lượt review. Test xanh không bao phủ các lỗi đã phát hiện; các probe cố ý kiểm những invariant chưa có test. Snapshot gốc được giữ nguyên để có thể so sánh trước/sau repair.

## 9. Tái lập audit

Từ repo root, dùng Python/R đã có:

```powershell
$env:PYTHONPATH = (Resolve-Path nextgen\src).Path
$env:RSCRIPT = 'C:\Program Files\R\R-4.5.1\bin\Rscript.exe'
& '.\.venv-e3-revised\Scripts\python.exe' docs\synthetic_review_2026-10-23\audit_synthetic.py
```

Script cần artifacts standard/easy benchmark đang có, chỉ đọc original artifacts và tạo bản sao fault-injection tạm. `review_tests` trong JSON ghi lượt pytest riêng đã quan sát, không phải tuyên bố script tự chạy lại pytest.


## 10. Trạng thái sau sửa ticket (2026-10-23)

Finding/con số phía trên là review trước sửa, được giữ đối chiếu. F1–F4 đã có implementation và regression tests; F5 làm rõ theo multinomial, strict count `<N` chưa đáp ứng và không tự đổi sampler. Xem [hoàn tất ticket](FIX_COMPLETION_VI.md), [audit sau sửa](audit_results_post_fix.json), [dual-truth metrics](dual_truth_metrics_v2.json). Audit/findings gốc, dataset v1 và experiment gốc giữ nguyên. V2 thêm row IDs NPZ; scorer tách X, X/N, p. Không suy ra thứ hạng hiệu quả easy/medium/hard từ các sửa này.
