# Ticket 03 — Mười trang cuối: Tsagris, kết quả và kết luận

**Deck:** `presentation/implementation_report/main.tex`  
**Plan:** [PLAN_VI.md](PLAN_VI.md)  
**Prompt:** [PROMPT_TICKET_03_FRAMES_21_30_VI.txt](PROMPT_TICKET_03_FRAMES_21_30_VI.txt)

## 1. Hợp đồng

Triển khai mười slot đích21–30: **Tsagris4 trang → kết quả5 trang → kết luận1 trang**. Đây là phần cuối của plan chỉnh sửa30 trang, không phải giữ nội dung legacy đang chiếm các slot đó.

Yêu cầu biên tập mới nhất tiếp tục áp dụng: một trang một câu hỏi; hình có thể dùng formal academic terms nhưng ngay dưới có một câu tiếng Việt dễ hiểu; nguồn/identifier dài vào companion/notes. Tái sử dụng sơ đồ/plot có sẵn, không thiết kế lại theme, không thêm trang debug/anomaly.

- Snapshot main.tex, danh sách input active và protected first20 mới nhất trước sửa. Workspace có thể đang có thay đổi từ Ticket01/02; không restore snapshot cũ hoặc sửa body các trang1–20.
- Chỉ thay input/body phục vụ slot21–30; giữ đúng30 frame active. Điều chỉnh section metadata ở phần cuối khi cần, không đổi theme/macros global để chữa một slide.
- Source cũ dùng làm tài sản; tạo section mới khi cần tránh sửa file đang được first20 input. Không xóa appendix/source legacy; chuyển chi tiết không trình chiếu vào companion, không thêm appendix active ngoài30 frame.
- Không sửa thuật toán/model/dataset/accepted CSV. Không benchmark/generate, không retune test. Cho phép sửa plotting code/re-export từ saved CSV để biểu diễn đúng số liệu; ghi rõ provenance và xác minh cận CI.
- Không tự đánh dấu first20 đạt QA; nếu chưa có evidence cho phần đó, chỉ báo hoàn thành Ticket03.

## 2. Storyboard

### Frame21 — Tsagris-α: module dùng dữ liệu và trả kết quả gì?

**Visual:** tái dùng `sections/14_tsagris_code.tex`, rút thành hai đường fit/predict. Fit allowed training rows → complete donor bank; partial query → donor selection → α-mean → ratio proposal → adapter/budget → final counts. Donor bank và query đi vào core như hai input, không vẽ donor bank “tạo ra query”.

**Nội dung:** model đang dùng là `TsagrisKNN` với α-mean; k/α chọn validation theo run. Phân biệt với `TsagrisAdaptive` repeated leave-N-out CV, không claim variant đó là primary run. Core state gồm complete training donors/IDs; không nhận private truth hoặc test donors.

**Caption mẫu:** “Model tìm những mẫu đã biết có tỷ lệ gần với phần quan sát, rồi dùng chúng để ước lượng phần còn thiếu.”

**Nguồn:** `baselines/tsagris.py`, `interfaces.py`, `adapters.py`, registry, section cũ. Không để core proposal đi thẳng tới scorer; final counts phải qua adapter/constraint.

### Frame22 — JSD: chọn donor giống query ở phần đã quan sát

**Visual:** giữ ý query/neighbors của `15_tsagris_jsd.tex`, làm nổi tập anchors và closure trên anchors. Hình simplex2D chỉ minh họa, có nhãn; không giả embedding dữ liệu thật.

**Toán:** query và donor được normalize trên cùng observed anchors; JSD dùng để rank và lấy top-k. Tối đa2 cụm phương trình chính; KL/midpoint có thể gộp một cụm hoặc notes.

**Quy ước phải đúng:** code `jsd` dùng natural log, trả tổng hai KL không có hệ số1/2, tức2×JSD chuẩn. Hoặc ghi D_code=2JSD_standard, hoặc trình bày đúng biểu thức code với ghi chú scale. Không gọi giá trị đó là sqrt-JSD/metric nếu code không sqrt. Hệ số dương không đổi ranking nhưng phải đúng khi nói giá trị.

Zero terms dùng convention phù hợp source. Zero anchor mass có declared global-donor policy khi bật, không mô tả như top-k ngẫu nhiên; không đưa mọi nhánh edge case lên hình chính.

**Caption mẫu:** “Chỉ so tỷ lệ ở các phần query đã biết; các mẫu giống nhất được dùng làm donor.”

### Frame23 — α-mean: gộp thông tin từ các donor

**Visual:** giữ simplex/μ_α từ `16_tsagris_mean.tex`; tách riêng khỏi residual allocation. Chỉ một ý: α điều khiển cách gộp composition của donor.

**Toán:** power transform theo từng donor row → average → inverse transform/closure; định nghĩa closure ngắn. α=1 arithmetic; α→0 geometric trên miền/zero-support phù hợp. Formula và nhánh alpha_mean source phải khớp; không viết log0 vô điều kiện hoặc gọi mọi output là geometric khi có zero.

**Giới hạn:** α là tham số được validation chọn, không giả “α càng nhỏ càng tốt”. Giá trị k/α COVID khác synthetic có thể có; chỉ ghi selected value nếu gắn nguồn/run. Nhận xét typo paper trong source cũ đưa companion với bằng chứng, không đặt thành kết luận chính của slide.

**Caption mẫu:** “Sau khi chọn donor, α quy định cách kết hợp các tỷ lệ của chúng thành một tỷ lệ đại diện.”

### Frame24 — Tsagris điền phần thiếu và trở về counts

**Visual:** tái dùng residual allocation cuối `16_tsagris_mean.tex`, nối rõ query anchors → donor α-mean → missing weights → residual allocation → adapter final counts. Không lặp toàn bộ module map trang21.

**Toán:** core compositional residual =1−sum_visible x; phân trên missing set theo mean weights. Sau đó adapter nhân N và enforce count budget. Định nghĩa chiều core có Other, phân biệt residual core với private residual truth. Quan sát giữ nguyên; named counts không ép bằng N.

**Nội dung:** zero residual → zeros; zero missing mass/global policy là declared behavior, đưa notes hoặc một chú thích, không trình bày như paper-original result. Có thể dùng ví dụ số schematic cùng các trang đầu để nối cách hiểu, không giả benchmark prediction.

**Caption mẫu:** “Phân phần tỷ lệ còn lại cho các ô thiếu, sau đó adapter trả về counts và giữ nguyên các ô đã biết.”

### Frame25 — COVID: so sánh kết quả chính

**Visual:** tái dùng `figures/primary_plot.tex` và `sections/18_primary.tex`; giữ dot/CI plot, màu/marker baseline. Rút caption/header để người đọc thấy ngay MAE thấp hơn là tốt hơn, đúng whole-feature scenario.

**Nguồn số:** `data/primary_metrics_source.csv`, `figure_manifest.json` → accepted run `experiments/offline_baselines_2026-10-15_v2`, accepted report. Đây là pooled hidden-target ratio MAE, không location-macro MAE của trang27. Units×100 là percentage points của tỷ lệ c/N, không phải counts hoặc relative error %.

**Bắt buộc sửa lỗi plotting đã phát hiện:** `scripts/build_figures.py:error_point` tính e1/e2 nhưng chỉ xuất e1, làm primary/scenario whiskers đối xứng sai CSV. Giữ điểm estimate và low/high đúng nguồn, xuất x-minus/x-plus riêng đúng PGFPlots; không dùng một biểu diễn symmetric thay cho cả2 phía. Ví dụ Hron whole-feature: estimate6.9808407446pp, low6.0303359410pp, high7.9879359111pp (sai số làm tròn nhỏ được phép). Viết verification đối chiếu source cho từng điểm; không sửa accepted CSV để fit hình. Nếu sửa builder tái xuất figures, lưu manifest/hash mới và bảo đảm không thay first20.

**Support/caveat:**5folds,3mask seeds,150locations,117675 lượt hidden targets theo source; targets không là số quan sát độc lập. Whiskers là location-bootstrap CI95, fixed fits.

**Caption mẫu:** “Trong kịch bản whole-feature của benchmark này, Tsagris có pooled MAE thấp nhất.” Không nói tốt nhất mọi dữ liệu/phương pháp.

### Frame26 — COVID: kết quả thay đổi theo cách che

**Visual:** tái dùng `scenario_plot.tex`/`19_scenarios.tex`; nếu quá dày chia4 nhóm scenario hoặc small multiples trên cùng slide, không thêm trang. Giữ cùng metric, unit, màu và baseline order; sửa CI bất đối xứng như frame25.

**Nội dung:** whole_feature/random_cell/time_block/whole_row phải giải thích trong một legend ngắn dễ hiểu; không chỉ ghi Feature/Random/Time. Fallback/policies có thể giải thích sự thay đổi nhưng claim phải đối chiếu diagnostics, không suy quan hệ nhân quả từ plot.

Không trộn scenario name COVID với MaskSpec synthetic: whole-row/time-block COVID không có nghĩa synthetic generator đã hỗ trợ random temporal blocks. Giới hạn source/capability cần đúng ngữ cảnh.

**Caption mẫu:** “Khi cách che thay đổi, khoảng cách giữa các baseline cũng thay đổi; không có một thứ hạng chung cho mọi kịch bản.” Chỉ dùng nếu source thực sự hỗ trợ; nếu chưa có reversal, nói “mức sai số thay đổi” thay vì bịa reversal.

### Frame27 — COVID: so chênh lệch ghép cặp theo location

**Visual:** giữ `paired_plot.tex`/`20_paired.tex`; baseline0 rõ; nhãn phép trừ/chiều tốt-xấu rõ. Sửa/đối chiếu asymmetric error endpoints nếu cần, không tự suy CI từ primary.

**Estimand:** location-macro ratio MAE difference,150locations, cùng split/mask/labels; khác pooled ratio MAE frame25. Không lấy hiệu hai điểm pooled của frame25 rồi đặt cạnh CI macro của frame27.

**Uncertainty:** location bootstrap, fixed fits, chưa hiệu chỉnh đa so sánh. Không thêm p-value/significance stars hay nói chứa toàn bộ training uncertainty. Bootstrap CI mô tả uncertainty theo locations trong protocol, không chứng minh tính tổng quát mọi dataset.

**Caption mẫu:** “So hai model trên cùng locations giúp đọc chênh lệch công bằng hơn; chiều dấu cho biết model nào có MAE thấp hơn.”

### Frame28 — Synthetic: so theo profile và MCAR/MAR

**Visual:** tái dùng `experiments/synthetic_baselines_v2_2026-10-23/plots/profile_test_errors.png`, `test_metrics.csv`, `plot_eda.R`. Ưu tiên re-export vector từ CSV hoặc tách panel rõ; không thu nguyên hình6panel cao vào slide ngang.

**Cấu trúc đề xuất:**2cột MCAR/MAR ×2hàng sample-ratio MAE/latent-p MAE, trục và legend đủ đọc. Nếu4panel vẫn chật, một metric chính với metric còn lại ở inset/companion, ghi rõ thay đổi bố cục và không xóa nghĩa khác biệt hai truths.

**Nội dung:** standard,seed2026,3profiles,3baselines; hidden test cells. Source CSV và run manifest đúng version. Hron/Tsagris gần nhau ở easy nhưng khác ở medium/hard là mô tả run nếu số xác nhận; MICoDA error cao ở run này không đủ kết luận lỗi phương pháp hoặc đã hội tụ.

**Count metric:** count MAE có thể giảm khi N giảm; không đọc là hard dễ hơn. Count MAE/RMSE và các metric phụ ở companion để tránh6panel. Không thêm CI khi mới1seed/profile; không đánh dấu significance hoặc nối line như thực nghiệm nhiều repeats.

**Caption mẫu:** “Đánh giá cả tỷ lệ sau lấy mẫu X/N và tỷ lệ nền p để tách hai mục tiêu dự đoán.” Có thể thêm một headline ngắn nêu quan sát đúng số, không gộp thành đoạn văn.

### Frame29 — Synthetic: sai số tập trung ở stratum nào?

**Visual:** tái dùng `latent_error_by_stratum.png` và CSV/script hiện có; subset panel nếu cần, ghi rõ profile/mask được trình bày. Không trộn bộ zero-positive COVID cũ với true-zero/sampling-zero synthetic.

**Nội dung:** true zero (p=0), sampling zero (p>0,X=0), positive count(X>0). Chỉ hidden cells, metric latent-p MAE, kèm n support theo nhóm hoặc companion nếu figure quá nhiều groups. n chưa có trong plot CSV thì lấy từ saved summaries/manifests hoặc evaluator truth+mask cùng split qua bước tổng hợp minh bạch, không fit lại model. Không phỏng đoán n từ mean/total; n=0 trả null, không vẽ thành0error.

**Giới hạn:**1seed/profile. Có thể nói positive-count cells đóng vai trò khác zero strata khi nguồn cho thấy; không gán cơ chế nhân quả hoặc gọi một model luôn tốt trên zero từ một aggregate.

**Caption mẫu:** “Một điểm sai số tổng có thể che đi khác biệt giữa ô thật sự bằng0, ô0 do lấy mẫu và ô có count dương.”

### Frame30 — Kết luận: hệ thống đã làm được gì và còn giới hạn gì?

**Visual:** tái dùng `24_conclusion.tex`, rút thành3 luận điểm: kiến trúc/hợp đồng dùng chung;3baseline variants đã hiện thực; đánh giá thực và synthetic bổ sung nhau. Một dải giới hạn/next step ngắn, không tạo trang cuối đầy bullets.

**Nội dung:** không tuyên bố full-paper reproduction; không khẳng định mọi model hội tụ; chỉ1seed/profile synthetic; native missing không có label; observed preservation/count budget là tính hợp lệ, không thay chứng minh accuracy. Nếu nêu next steps thì ưu tiên repeats/sensitivity/diagnostics đúng phạm vi, không nói MNAR/sweep đã sẵn có.

**Caption mẫu:** “Codebase cho phép đổi baseline trong cùng quy trình; kết quả hiện tại mô tả các run đã kiểm, còn khả năng tổng quát cần thêm thí nghiệm.”

Nguồn/bibliography chi tiết và bảng metric đầy đủ chuyển companion, không bổ sung frame31. Không đưa nội dung debug/anomaly đã bị người dùng yêu cầu bỏ trở lại conclusion/appendix active.

## 3. Tài sản và kiểm chứng cần giữ

| Nhóm | Tài sản dùng lại | Việc được chỉnh |
|---|---|---|
| Tsagris | sections14_tsagris_code,15_tsagris_jsd,16_tsagris_mean | Chia thành4frame, cô đọng, sửa luồng đúng boundary |
| COVID primary/scenarios | figures primary_plot/scenario_plot; accepted CSV | Bố cục/label; sửa xuất CI bất đối xứng đúng low/high |
| COVID paired | paired_plot, paired_location_differences_source.csv | Giữ paired estimand, chiều dấu và asymmetric endpoints |
| Synthetic profiles | profile_test_errors.png, test_metrics.csv, plot_eda.R | Re-export/tách panel; không đổi estimand/version |
| Synthetic strata | latent_error_by_stratum.png, summaries/manifests | Re-export, thêm support n đúng source |
| Kết luận/nguồn |24_conclusion,28_appendix_variants,30_appendix_source | Lọc ý chính; giữ chi tiết companion |

Đường dẫn section/figure tương đối với `presentation/implementation_report/`; synthetic plots tại `experiments/synthetic_baselines_v2_2026-10-23/plots/`. Kiểm file mới nhất trước dùng; không assume file name có nghĩa là số trang.

## 4. Acceptance / bàn giao

1. Đúng30frame active; slot21–30 đúng storyboard, first20source/order được bảo vệ. Không tự xóa legacy sources để đạt số trang.
2. Tsagris4trang có architecture, JSD, α-mean và allocation rõ; các API/zero policies/parameter claims bám code/run.
3. Mỗi trang có một câu tiếng Việt giúp người ngoài hiểu hình; công thức không dùng ký hiệu chưa giải thích. Không paragraph dài hoặc ngôn ngữ hoàn toàn API.
4. Primary/scenario CI endpoints khớp low/high CSV, gồm bất đối xứng; paired estimand không bị tráo với pooled. Ghi unit/support, no invented inference.
5. Synthetic plot dùng đúng CSV/profile/mask/version; chỉ1seed được ghi đúng, không dùng số0 thay null hoặc sốn chưa có nguồn.
6. Compile XeLaTeX, PDF30trang; render21–30≥135dpi sau final build, contact sheet riêng10trang và xem từng ảnh. Kiểm panel, legend, tick, source, arrowhead/công thức, không dựa vào no-overfull để pass.
7. Ghi source map số/công thức/claim, plot generation commands, CI endpoint comparison, protected files hashes/diff, final PDF/render hashes, checks/progress harness riêngTicket03. Chỉ thêm kiểm tra nhỏ phục vụ plotting fix, không chạy suite model.
8. Bàn giao TeX/PDF, render/contact sheet, companion metrics/sources và bảng acceptance pass/fail/unresolved. Báo rõ trạng thái toàn deck nếu chưa review first20, không tự claim đã approve30slide.
