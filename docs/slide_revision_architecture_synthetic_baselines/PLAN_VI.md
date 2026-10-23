# Plan chỉnh sửa deck hiện tại — 30 trang

Đây là plan chỉnh sửa, chưa thay đổi file TeX/PDF. Điểm xuất phát là `presentation/implementation_report/main.tex`: 30 frame active, gồm 24 trang chính và 6 phụ lục. Đề xuất vẫn 30 trang nhưng đưa nội dung hữu ích của phụ lục vào mạch chính, dành chỗ cho synthetic và giải thích baseline. Không tạo một deck/theme mới.

**Flow:** bài toán → kiến trúc codebase → dữ liệu synthetic → Hron → MICoDA → Tsagris → đánh giá → kết luận.

## 1. Nguyên tắc biên tập

- Giữ Hannover 16:9, nhận diện BK, palette ba baseline, macro TikZ và hệ thống PGFPlots hiện có. Chỉnh title/spacing khi cần; không thay toàn bộ template.
- Một trang trả lời một câu hỏi. Một hình chính, một takeaway, khoảng 20–45 từ nội dung ngoài nhãn/công thức/nguồn. Trang công thức tối đa hai cụm phương trình, định nghĩa ký hiệu ngay cạnh.
- Hình codebase và hình luồng dữ liệu là hai hình khác nhau: một hình biểu diễn trách nhiệm/phụ thuộc, hình kia biểu diễn dữ liệu đi đâu và biến đổi thế nào. Không lặp cùng pipeline ở nhiều trang.
- Dùng TikZ cho module, boundary, vòng lặp và simplex; giữ plot khoa học hiện có. Chỉ bổ sung hình còn thiếu để giải thích generator, mask và các bước toán học.
- Phân biệt ba tầng: phương pháp từ paper; core thực sự đã cài đặt; adaptation counts của benchmark. Không gọi run hiện tại là tái lập toàn bộ paper.
- Chi tiết hàm/path/citation dài đưa vào speaker notes hoặc tài liệu nguồn đi kèm. Phần scientific limitation vẫn phải hiện trên trang liên quan.

## 2. Storyboard 30 trang

Số “trang cũ” dưới đây là thứ tự frame active trong `main.tex`, không phải số trong tên file. File TeX cũ không active không được xem là trang đang trình chiếu.

### Mở đầu — trang 1–2

**01 · Kiến trúc và hiện thực ba baseline điền khuyết**
- Vai trò: đặt đúng phạm vi báo cáo implementation và evaluation.
- Giữ bìa cũ, sửa subtitle thành ba điểm ngắn: kiến trúc / synthetic / baseline.
- Hình: giữ nhận diện hiện có; không thêm hình trang trí.
- Nguồn/thao tác: trang cũ 1, `00_cover.tex`; **giữ, chỉnh tiêu đề**.

**02 · Một bài toán, hai nguồn đánh giá, ba module**
- Câu hỏi: hệ thống dự đoán gì và báo cáo sẽ dẫn người xem đi đâu?
- Nội dung: counts quan sát + tổng N + mask → phần thiếu; COVID là case study hiện có, synthetic có truth kiểm soát; ba core chạy qua hợp đồng chung.
- Hình: tận dụng observed/hidden/count-budget của trang cũ 2; thêm dải roadmap kiến trúc → synthetic → ba module → đánh giá. Không vẽ lại hình budget nếu vẫn rõ.
- Nguồn/thao tác: `01_problem.tex` + ý scope từ `02_scope.tex`; **gộp**. Scaffold neural chỉ một chú thích phạm vi, không dành riêng một trang.

### Kiến trúc — trang 3–7

**03 · Toàn cảnh codebase: trách nhiệm và phụ thuộc**
- Phải thấy ngay bốn cụm: contracts/data → orchestration → adapter/core → scoring/artifacts. Synthetic là một nhánh data source đưa vào cùng contract.
- Nội dung trên hình: `ConditioningPanel`, `FitScope`, registry, ba core, scorer; đặt `EvaluationLabels` riêng phía evaluator. Mũi tên biểu diễn đúng quan hệ gọi/phụ thuộc, không nối thành chuỗi chỉ vì tiện bố cục.
- Hình: **gộp và tinh giản** sơ đồ bốn lớp cũ 4 với module map cũ 8. Khoảng 6–8 khối chính; đường nhánh baseline cùng chiều.
- Nguồn: `03_architecture.tex`, `07_codebase_map.tex`; đối chiếu `nextgen/src/longitudinal_impute/`.

**04 · Input → process → output: một lượt chạy thực tế**
- Input: public counts có NaN, mask, N, entity/time, covariates. Process: load/validate → fit train → select validation → refit train+validation → predict test. Output: predictions, diagnostics, metrics/report.
- Hình: dùng swimlane data / model / evaluator của trang cũ 9 và runtime cũ 5; làm rõ mỗi artifact được sinh ở bước nào.
- Không cho hidden labels đi qua fit/predict; không để test ảnh hưởng chọn tham số.
- Nguồn/thao tác: `04_runtime.tex`, `08_data_flow.tex`; **gộp**, giữ boundary logic.

**05 · Adapter chung: từ counts sang composition rồi trở lại counts**
- Giải thích `CountBaselineAdapter`: counts/N, residual Other, core proposal, khôi phục output với observed values cố định và count budget.
- Hình chính: hai miền counts/composition, adapter làm cầu nối; giữ thanh budget cũ, bổ sung một ví dụ số nhỏ minh họa có nhãn “schematic”.
- Công thức: residual budget `B_i=N_i−Σ_observed c_ij`; target prediction không được tiêu quá ngân sách. Phân biệt residual của adapter với private residual truth của generator.
- Nguồn/thao tác: `05_boundary.tex`, `06_ownership.tex`, `08_adapter.tex` (file cũ không active); **trích và ghép**, kiểm `baselines/adapters.py`, `constraints.py` trước triển khai.

**06 · Plug-and-play: thay core, giữ hợp đồng**
- Nội dung: config/name → registry factory → fit/predict interface → diagnostics. Model mới cần đáp ứng đúng input/output, fit scope và invariants; không sửa engine cho mỗi model.
- Hình: tận dụng sơ đồ extension trong phụ lục cũ 25, ba “socket” baseline trên cùng interface. Một snippet tối đa 4 dòng chỉ để minh họa contract thật.
- State sở hữu ở đâu: engine giữ splits/selection; adapter giữ preprocessing/constraints; core giữ donor bank hoặc coefficients; evaluator giữ labels.
- Nguồn/thao tác: `25_appendix_arch.tex` + `06_ownership.tex`; **đưa phụ lục lên chính và gộp**. Không hứa mọi model tùy ý đều dùng được mà không adaptation.

**07 · Quản lý và debug một run bằng artifact/diagnostic**
- Câu hỏi: metric bất thường thì tìm từ đâu? Trace run/config → candidate → selected params → prediction → diagnostic → score; phân biệt resume với refit mới.
- Hình: dùng lifecycle/artifact/debug trace trong phụ lục cũ 27, thêm ba điểm kiểm: fit scope, fallback, count budget/convergence.
- Runtime: tận dụng **panel candidate timing** từ `runtime_by_profile.png` hoặc vẽ lại đúng `candidate_timing.csv` như inset nhỏ; nếu quá chật, giữ trong notes/companion asset. Không nhét full plot hai panel vào cạnh một sơ đồ dày.
- Nguồn/thao tác: `27_appendix_ops.tex`; **rút gọn, chuyển lên chính**. Timing ghi rõ CPU/máy/run, gồm thao tác nào; không so sánh tốc độ tổng quát từ một máy.

### Synthetic — trang 8–13

**08 · Vì sao thêm synthetic và hiện đã có gì?**
- Nội dung: truth biết đầy đủ giúp tách lỗi so counts quan sát mẫu và lỗi so latent p; phân biệt true zero/sampling zero. Snapshot đang có: v2, standard, seed 2026, easy/medium/hard, hai declared masks.
- Hình: sửa pipeline cũ “Sinh → truth → mask → benchmark”, đổi nhãn “CHỜ THÍ NGHIỆM” thành trạng thái đã có artifact và kết quả mô tả.
- Nguồn/thao tác: `23_synthetic.tex`; **di chuyển từ cuối lên đây, cập nhật và mở rộng**. V1 là snapshot đối chiếu; dùng v2 cho contract trình bày.

**09 · Generator: latent dynamics → composition → sampled counts**
- Hình bổ sung: ba tầng căn hàng theo thời gian: wave weights → p gồm residual → multinomial counts. Thể hiện onset/peak/tail bằng một component schematic, không làm người xem tưởng đó là output thật.
- Toán chính: `p_j=(1−r)w_j/Σw`, `(X_1,…,X_D,X_res)~Multinomial(N; p_1,…,p_D,r)`. AR(1), timing shift và dominance chỉ là nhãn ở nhánh tạo w, không dồn toàn bộ công thức vào một trang.
- Nội dung: generator là thiết kế benchmark của dự án; Schouten cung cấp missingness, không phải nguồn của wave generator.
- Nguồn/thao tác: **thêm một hình giải thích mới**, từ `nextgen/SYNTHETIC_DATA_VI.md`, `synthetic/generator.py`; giữ nguyên sampler/config.

**10 · Easy / medium / hard thay đổi những gì?**
- Hình bổ sung từ bundle có sẵn: ba cột cùng trục thể hiện cùng entity/component qua thời gian; một dải nhỏ cho N và cấu hình. Chọn entity bằng quy tắc cố định trước khi xem ranking model, ghi entity/seed.
- Nội dung tối thiểu: standard 60×120×12; timing SD 1/4/8, dominance SD .10/.35/.70, noise SD .05/.15/.35, E[N] 2000/500/100. Các giá trị này là config, không phải moment thực tế của mọi sample.
- Không lấy đường error của model để minh họa dữ liệu; không gọi hard khó hơn trên mọi metric chỉ vì tên profile.
- Nguồn/thao tác: **thêm plot mô tả từ arrays đã có**, không sinh data mới; manifest v2 và generator docs. Trình bày bảng config gọn, công thức chi tiết vào notes.

**11 · Missingness và ba loại zero phải được tách biệt**
- Hình bổ sung: một heatmap mask MCAR/MAR đặt trên cùng sample; ba ô minh họa true zero `p=0`, sampling zero `p>0,X=0`, positive count. Missing là trạng thái mask riêng, không đồng nghĩa zero.
- Đúng thiết kế hiện hành: ba pattern cố định che 4/12, case probability .3, nominal target-cell rate .1; ghi realized support từ manifest thay vì gọi chung “30% missing”. MAR dùng conditioning theo spec, không vẽ mũi tên từ latent truth.
- MCAR/MAR đã có; MNAR và sweep rate/seed rộng là future work, không mô tả như đã chạy.
- Nguồn/thao tác: `nextgen/SCHOUTEN_VI.md`, mask_specs/manifest v2, `synthetic/masking.py`; **thêm hình từ dữ liệu/spec có sẵn**.

**12 · Một protocol cho cả ba baseline**
- Hình: tận dụng split/select/refit/test của cũ 18; bổ sung public loader / private evaluator boundary của cũ 6/9.
- Nội dung: entity-disjoint 36/12/12 ở standard; chọn bằng validation hidden-cell sample_ratio MAE; refit train+validation; test chỉ báo cáo. Cùng mask/split cho so sánh trong mỗi profile.
- Dải artifact: public conditioning + schema/row IDs → model; private counts/p/strata → scorer. Hash/row alignment giúp truy vết, không được mô tả như phân quyền hệ điều hành.
- Nguồn/thao tác: `17_protocol.tex`, `05_boundary.tex`, `08_data_flow.tex` + `nextgen/SYNTHETIC_DATA_VI.md`; **tái dùng và cập nhật**. Ghi protocol COVID riêng khi vào kết quả COVID.

**13 · Đánh giá cái gì: count, sample ratio hay latent p?**
- Ba estimand: `|Xhat−X|`, `|Xhat/N−X/N|`, `|Xhat/N−p|`; MAE là chính, RMSE là độ nhạy với lỗi lớn. Chỉ hidden cells, có support n và strata.
- Hình: sơ đồ cùng một prediction so với hai truth X và p; tái sử dụng bảng metric cũ 26, rút thành ba hàng chính và một dòng invariant checks.
- Đơn vị: counts; ratio hoặc percentage points ghi nhất quán. Không đồng nhất X/N với latent p; count MAE qua profile phụ thuộc N.
- Thống kê: synthetic hiện một seed/profile, số cells không phải số quan sát độc lập. Nếu cần CI, bootstrap ghép cặp theo entity từ saved predictions, xác nhận support; CI đó vẫn conditional trên dataset/fit/seed hiện có. Chưa có thì ghi “chưa ước lượng”, không tạo error bar giả.
- Nguồn/thao tác: `26_appendix_metrics.tex`, `synthetic/scoring.py`, plot README; **tái dùng cấu trúc và thay đúng metric synthetic**.

### Hron-kNN — trang 14–16

**14 · Hron module: input, donor state và output**
- Sơ đồ `adapter → HronKNN.fit → donor bank → predict_result → proposal/diagnostics → adapter`.
- Nội dung: k, fit scope, donor availability, fallback; state chứa gì và prediction đọc gì. Dùng đúng tên hàm trong source, không vẽ train/test chung donor pool.
- Nguồn/thao tác: cũ 10, `09_hron_code.tex`; **giữ, chuyển vị trí**, chỉ chỉnh nhãn và đồng bộ style.

**15 · Kiến thức dùng để hiện thực: CLR và khoảng cách Aitchison**
- Tách từ trang cũ 11: định nghĩa closure/CLR và distance trên tập observed chung; giải thích tại sao khoảng cách làm trên composition thay vì Euclidean counts.
- Hình: giữ scatter/query/neighbors schematic hiện có, làm nổi observed subcomposition. Ghi rõ zero policy đang thực thi; không viết log(0) như thể hợp lệ.
- Hai cụm toán tối đa: CLR và distance; scale/aggregation chuyển sang trang sau.
- Nguồn/thao tác: `10_hron_mechanism.tex` + `baselines/hron.py`; **tách trang dày thành hai bước giải thích**. Paper Hron §3.1 theo citation có sẵn; source xác nhận biến thể available-donor.

**16 · Cơ chế: chọn donor → scale → median → khôi phục counts**
- Hình: giữ/điều chỉnh top-k schematic; thêm một dải donor target được rescale rồi lấy median, không thêm hình trang trí.
- Toán: scale bằng median observed/query-donor và median aggregation đúng code. Giải thích donor có đủ observed anchors và target j; đây là sequential available-donor variant đã cài đặt.
- Chốt: core proposal còn đi qua adapter budget, fallback/degenerate behavior phải được ghi theo code; k chọn từ validation, không gắn một k của COVID cho mọi synthetic run.
- Nguồn/thao tác: nửa sau `10_hron_mechanism.tex`, diagnostics/interface; **tách, bổ sung annotation**.

### MICoDA-V2 — trang 17–20

**17 · MICoDA module: covariates và coefficient state**
- Hình: giữ sơ đồ cũ 12; tách covariates/design matrix khỏi target composition, fit coefficients từ train và dùng state để dự đoán.
- Nội dung: W, max_iter/tolerance, ALR reference, beta state. Ghi variant v2; không mô tả thành multiple-imputation uncertainty đầy đủ.
- Nguồn/thao tác: `11_micoda_code.tex`; **giữ, chuyển**, đối chiếu `baselines/micoda.py`.

**18 · Kiến thức dùng để hiện thực: ALR và hồi quy**
- Hình: giữ mô hình cũ 13; một composition → ALR coordinates → regressions theo covariates.
- Công thức: `z_ij=log(x*_ij/x*_ir)` và `z_ij=q_i^T beta_j`; giải thích reference và pseudocount W đúng implementation. Không dùng cùng chữ X cho counts và design matrix mà không phân biệt.
- Nguồn/thao tác: `12_micoda_model.tex`; **giữ nhưng rút chữ**, nguồn paper/source ở footer.

**19 · Cơ chế cập nhật: chia residual mass cho phần thiếu**
- Hình: tách phần allocate từ cũ 14; observed cố định, missing logits → softmax → chia residual. Một thanh mass trước/sau đủ để thấy bảo toàn.
- Toán: residual sau phần observed và allocation trong missing set; nêu reference logit/positive shift theo code. Phân biệt residual nội bộ core với count budget ở adapter.
- Nguồn/thao tác: `13_micoda_update.tex`; **tách**, giữ phương trình đã có sau khi đối chiếu source.

**20 · Fit loop và cách nhận biết chưa hội tụ**
- Hình: dùng vòng lặp cũ 14, sửa route cho rõ init chạy một lần; update/refit quay về predict, không quay về “khởi tạo complete-case” mỗi vòng.
- Nội dung: initialization → predict/allocate → refit → delta check → stop. Diagnostics iterations/delta/converged đi ra artifact của trang 7.
- Giữ caveat COVID accepted 5/5 folds chạm 50, không gọi là hội tụ. Synthetic phải đọc diagnostics riêng từng run; không suy từ COVID sang synthetic. Không tự tăng max_iter trong nhiệm vụ sửa slide.
- Nguồn/thao tác: `13_micoda_update.tex` và accepted diagnostics; **tách và làm rõ logic**.

### Tsagris-α — trang 21–24

**21 · Tsagris module: donor bank, JSD và α-mean**
- Hình: giữ cũ 15, đánh số ba bước: donor availability → distance/rank → α-mean; adapter xử lý counts ở ngoài.
- Nội dung: k/alpha được chọn validation theo từng run; phân biệt `TsagrisKNN` đang chạy với `TsagrisAdaptive` không phải protocol của run này.
- Nguồn/thao tác: `14_tsagris_code.tex`, `baselines/tsagris.py`; **giữ và chuyển**.

**22 · Kiến thức dùng để hiện thực: JSD trên observed anchors**
- Hình: giữ minh họa JSD cũ 16, query/donor được closure trên anchors trước khi so.
- Hai cụm toán: midpoint và JSD; định nghĩa KL, xử lý zero theo code, quy ước log/scale nếu thể hiện giá trị. Nói rõ distance dùng để chọn donor chứ không là loss huấn luyện neural.
- Nguồn/thao tác: `15_tsagris_jsd.tex`; **giữ, giảm chữ**, kiểm source/citation.

**23 · Kiến thức dùng để hiện thực: α-mean trên simplex**
- Hình: giữ simplex schematic cũ 17; làm rõ points donor và mean. Không dùng sơ đồ 2D như kết quả embedding dữ liệu thật.
- Toán: power transform, average, inverse/closure; α=1 arithmetic và giới hạn α→0 geometric, kèm điều kiện positive/zero policy đang thực hiện.
- Phần nhận xét chỉ số công thức paper và normalization trong code chuyển vào notes có citation; không biến thành kết luận sai paper trên slide chính.
- Nguồn/thao tác: phần mean trong `16_tsagris_mean.tex`; **tách khỏi residual allocation**.

**24 · Cơ chế dự đoán: donor mean → residual allocation → counts**
- Hình: query anchors → top-k JSD → α-mean → phân phần còn thiếu → adapter; highlight bước khác Hron và MICoDA.
- Toán: residual × mean trên missing set rồi quy đổi counts; kiểm đúng nhánh degenerate/fallback. Không tuyên bố observed thay đổi để closure lại toàn row.
- Một dòng so sánh cơ chế: Hron rescaled median / MICoDA regression logits / Tsagris α-mean. Không nhét bảng benchmark ở đây.
- Nguồn/thao tác: phần residual trong `16_tsagris_mean.tex` + adapter; **tách, nối với trang 5**.

### Đánh giá — trang 25–29

**25 · COVID: so sánh chính trên dữ liệu thực hiện có**
- Giữ nguyên dữ liệu/đơn vị của `primary_plot.tex`, kế thừa `18_primary.tex`. Headline chỉ kết luận trong accepted run và đúng scenario/metric.
- Chú thích run `offline_baselines_2026-10-15_v2`, protocol COVID, pooled hay location-macro đúng plot; không dùng bảng synthetic để giải thích CI COVID.
- Nguồn/thao tác: cũ 19; **giữ plot, chuyển và rút caption**.

**26 · COVID: thứ hạng thay đổi theo kịch bản che**
- Giữ `scenario_plot.tex` từ `19_scenarios.tex`; giữ scales và scenario definitions đúng nguồn. Nếu chật, giảm annotation/legend lặp, không gộp metric khác đơn vị cùng trục.
- Takeaway tập trung khác biệt scenario thay vì xếp hạng chung mọi phương pháp.
- Nguồn/thao tác: cũ 20; **giữ**, chỉnh bố cục nhẹ.

**27 · COVID: chênh lệch ghép cặp và phạm vi của CI**
- Giữ `paired_plot.tex` từ `20_paired.tex`. Ghi hướng dấu Δ và mốc 0; bootstrap theo 150 locations, fixed fits; CI95 chưa hiệu chỉnh đa so sánh.
- Không gọi là training uncertainty, không tự thêm p-value/stars. Nếu bổ sung effect size, dùng cùng estimand/location pairing và có script/data đối chiếu.
- Nguồn/thao tác: cũ 21; **giữ**. Bảng đầy đủ cũ 29 đưa tài liệu/notes thay vì thêm trang thứ 31.

**28 · Synthetic: error theo profile và missingness**
- Tái sử dụng `profile_test_errors.png` và `test_metrics.csv`; ưu tiên hai hàng sample-ratio MAE và latent-p MAE, MCAR/MAR theo hai cột. Full PNG sáu panel đang quá cao để thu nhỏ lên 16:9; tách panel hoặc xuất lại vector từ đúng CSV/script, không chạy lại benchmark.
- Count MAE chỉ làm inset/notes để minh họa phụ thuộc N; MAE/RMSE còn lại vào bảng notes. Ghi standard, seed 2026, một run/profile; không có CI giữa seeds.
- Được nêu: các run hiện tại có Hron/Tsagris gần nhau ở easy và khác ở medium/hard; MICoDA có error cao trong cấu hình này. Không suy thành thứ hạng phương pháp tổng quát.
- Nguồn/thao tác: `experiments/synthetic_baselines_v2_2026-10-23/plots/`; **đưa plot đã có vào deck**.

**29 · Synthetic: sai số nằm ở stratum nào?**
- Tái sử dụng `latent_error_by_stratum.png`; giữ true-zero / sampling-zero / positive-count, bổ sung n hidden theo stratum từ bảng nguồn. Không đặt mean error của một nhóm cạnh nhóm rỗng mà quên n=0/null.
- Takeaway: aggregate có thể che khác biệt strata. Tham chiếu `21_strata.tex` như cấu trúc kể chuyện, nhưng không trộn số liệu COVID vào plot synthetic.
- Rà support/scoring version trước chèn; nếu bổ sung CI entity-bootstrap phải tính từ predictions gốc cùng mask, ghi là phân tích mới trên saved outputs; chưa làm trong plan.
- Nguồn/thao tác: plot/CSV đã có; cũ 22 chuyển notes/companion, **tái dùng bố cục, thay đúng nguồn**.

### Kết luận — trang 30

**30 · Đã hiện thực gì, đã kiểm chứng gì, còn thiếu gì?**
- Ba kết luận ngắn: contract/module tách rõ; ba baseline chạy cùng adaptation; synthetic có dual truth nhưng hiện một seed/profile.
- Hình: tái sử dụng ba tầng kết luận cũ 24; mỗi tầng một ý. Next steps: repeats nhiều seed, sensitivity đúng capability, xem convergence/fallback trước tăng độ phức tạp.
- Giữ scope: residual named-counts ≤N; không khẳng định strict <N vì residual count có thể bằng 0. MNAR/sweep rộng là chưa có.
- Sources: footer dẫn repo/source map; giữ nội dung `30_appendix_source.tex` và `28_appendix_variants.tex` làm companion/notes. Nếu bắt buộc trình chiếu bibliography riêng, thay một trang bằng nguồn sau khi thống nhất, không âm thầm tăng số trang.
- Nguồn/thao tác: `24_conclusion.tex`, phụ lục variants/source; **gộp và cập nhật**.

## 3. Bản đồ tái sử dụng

| Tài sản hiện có | Trang đích | Cách dùng |
|---|---:|---|
| `00_cover`, `01_problem`, `02_scope` | 1–2 | Giữ bìa, gộp scope vào bài toán |
| `03_architecture`, `07_codebase_map` | 3 | Gộp hai hình tổng quan, không lặp |
| `04_runtime`, `08_data_flow`, `05_boundary` | 4, 12 | Flow tổng quát ở 4; split/evaluator boundary ở 12 |
| `06_ownership`, `08_adapter` (inactive), `25_appendix_arch` | 5–6 | Adapter và ownership/plugin |
| `27_appendix_ops`, lifecycle/debug files cũ | 7 | Trace ngắn, chi tiết để notes |
| `23_synthetic` | 8 | Thay placeholder, giữ pipeline gốc |
| `17_protocol`, `26_appendix_metrics` | 12–13 | Chuyển từ cuối lên trước baseline, cập nhật protocol |
| `09_hron_code`, `10_hron_mechanism` | 14–16 | Giữ architecture, tách toán/cơ chế |
| `11_micoda_code`, `12_micoda_model`, `13_micoda_update` | 17–20 | Tách residual và fit loop |
| `14_tsagris_code`, `15_tsagris_jsd`, `16_tsagris_mean` | 21–24 | Tách α-mean và residual |
| `figures/primary_plot.tex` | 25 | Giữ accepted COVID data |
| `figures/scenario_plot.tex` | 26 | Giữ accepted COVID data |
| `figures/paired_plot.tex` | 27 | Giữ paired CI và caveat |
| `plots/profile_test_errors.png` | 28 | Tách/re-export panel từ CSV, không đổi số |
| `plots/latent_error_by_stratum.png` | 29 | Giữ strata, thêm support |
| `plots/runtime_by_profile.png` | 7/notes | Chỉ panel phù hợp; giữ full hình ở companion |
| `21_strata`, `29_appendix_data`, `28_appendix_variants`, `30_appendix_source` | notes/companion | Giữ để tra cứu, không làm mất nội dung nguồn |
| `24_conclusion` | 30 | Sửa theo trạng thái hiện tại |

Đường dẫn viết tắt: section là `presentation/implementation_report/sections/`; figures là `presentation/implementation_report/figures/`; plots là `experiments/synthetic_baselines_v2_2026-10-23/plots/`. Chỉ ba trang 9–11 cần hình giải thích dữ liệu mới; những trang khác chủ yếu chuyển, gộp, tách hoặc chú thích tài sản hiện có.

## 4. Quy tắc metric và kết luận thống kê

1. **COVID:** giữ nguyên accepted run trong `data/figure_manifest.json`; ba plot hiển thị ratio/difference ×100 thành percentage points. Bootstrap location, fixed fits; không diễn giải thành độ bất định khi fit lại.
2. **Synthetic:** công bố count MAE/RMSE, sample-ratio MAE/RMSE và latent-p MAE/RMSE ở bảng companion; slide chọn MAE ratio/latent làm chính, RMSE khi nó làm thay đổi nhận định. Không thêm metric chỉ để tăng số lượng.
3. **Strata/support:** chỉ hidden targets; n theo profile/mask/stratum. Macro theo entity có ý nghĩa nếu định nghĩa rõ và tính lại từ saved outputs; không suy macro từ pooled CSV.
4. **Paired inference:** cùng entity/mask/prediction support; resample entity nguyên cụm để giữ phụ thuộc thời gian. Chưa có CI synthetic trong artifact đã kiểm thì không tự vẽ. Muốn phát biểu ổn định qua seed cần thí nghiệm lặp, nằm ngoài phạm vi sửa slide này.
5. **Protocol validity:** selected objective là validation sample_ratio MAE, latent p chỉ chấm bổ sung; không chọn lại tham số dựa vào test hoặc latent p.
6. **Model health:** báo observed-preservation, finite/nonnegative, budget violations, fallback và convergence từ diagnostic đúng run. Metrics dự đoán tốt không tự chứng minh convergence hoặc parity paper.

## 5. Trình tự chỉnh sửa khi triển khai

**Bước A — đóng băng nguồn:** snapshot main/sections/figures/data và synthetic plots/CSV/manifest; map từng claim đến code/artifact. Không sửa CSV accepted. Rà lại nguồn khi bắt đầu, vì các run có thể còn được bổ sung.

**Bước B — đổi thứ tự và tái dùng:** sửa main.tex theo storyboard; giữ file section cũ khi nội dung còn phù hợp, chỉ tạo file mới cho trang tách/bổ sung. Không xóa các phụ lục cũ; lưu làm companion/notes. Đầu ra vẫn 30 frame active.

**Bước C — chỉnh nội dung:** làm kiến trúc 3–7 trước; cập nhật synthetic 8–13; tách các model 14–24; chèn các plot 25–29. Dùng tên biến nhất quán xuyên suốt: counts X/c, composition p/x, total N, observed O, missing M, design matrix Q.

**Bước D — kiểm render:** compile XeLaTeX theo project; kiểm log errors/overfull; render đủ 30 trang; contact sheet và kiểm từng trang ở kích thước đọc thực. Không dùng shrink font toàn trang để che overflow. Kiểm trục, unit, legend, ký hiệu và mũi tên vòng lặp bằng mắt; sửa tối đa ba vòng cho cùng lỗi trước báo vấn đề còn lại.

**Bước E — giao:** TeX/PDF, contact sheet, source/asset map, bảng metrics companion và checks. Không chạy thêm synthetic/benchmark trừ khi được giao riêng; chỉ được re-export plot hoặc tính thống kê minh bạch từ saved outputs nếu cần và ghi provenance.

## 6. Tiêu chí nghiệm thu lần triển khai

- 30 frame active; đúng thứ tự architecture → synthetic → Hron → MICoDA → Tsagris → results.
- Có riêng một trang architecture tổng quan và một trang input/process/output.
- Mỗi baseline có kiến trúc module, kiến thức/công thức và cơ chế cụ thể; nội dung phản ánh variant đang chạy.
- Synthetic placeholder được thay bằng contract/artifact hiện có; không hứa đã có MNAR/repeats.
- Ba plot COVID giữ nguồn accepted; synthetic plot ghi đúng seed/profile/mask/estimand; không đổi nghĩa CI.
- Hình mới chỉ bổ sung chỗ thiếu; diagram/plot cũ được tái dùng như map trên. Theme và màu baseline nhất quán.
- PDF không tràn/cắt chữ, công thức và nhãn đọc được ở full-slide, không dùng một hình sáu panel thu nhỏ đến mất nhãn.
- Mọi kết luận có nguồn; không fabricate p-value, confidence interval, convergence hay paper reproduction.

## Nguồn đã rà cho plan

- `presentation/implementation_report/main.tex`, sections active, ba plot PGFPlots và `data/figure_manifest.json`.
- `nextgen/SYNTHETIC_DATA_VI.md`, `nextgen/SCHOUTEN_VI.md`.
- `docs/synthetic_review_2026-10-23/FIX_COMPLETION_VI.md`.
- `experiments/synthetic_baselines_v2_2026-10-23/medium_hard_results_vi.md`.
- `experiments/synthetic_baselines_v2_2026-10-23/plots/README_vi.md` và các PNG/CSV cùng thư mục.

Plan xác nhận tồn tại và nội dung các nguồn trên; chưa chạy lại benchmark, chưa audit lại toàn bộ numerical results và chưa render bản chỉnh sửa vì chưa sửa deck.
