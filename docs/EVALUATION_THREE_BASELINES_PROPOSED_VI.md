# Proposed protocol: evaluate Hron, Tsagris và MICoDa

Ngày: 2026-11-10. Trạng thái: **đề xuất để duyệt**, chưa triển khai/chạy benchmark, không thay thế contract đã duyệt.

## 1. Quyết định đề xuất

Một protocol gọn gồm: **masked reconstruction làm bằng chứng chính; native Wasserstein làm chẩn đoán phụ; kiểm tra reliability và Other/budget trước mọi kết luận xếp hạng**.

- Chạy được ngay để đánh giá **ba pipeline/adaptation hiện tại**, không diễn giải thành so sánh kiểm soát giữa ba thuật toán.
- Benchmark xác nhận chỉ được mở sau quyết định riêng về contract Other/budget chung, không dùng truth bị che để suy ra input ngân sách.
- Không sửa thuật toán, không tự chuyển MICoDa A17/R1 sang A18/R2; không thêm framework, JSD, pre-gap hoặc reference xuyên quốc gia vào core.

## 2. Căn cứ từ paper: Claim → Evidence → Analysis

**P1:** Rania Farjallah và cộng sự, *Evaluation of Missing Data Imputation for Time Series Without Ground Truth*, bản ICC 2025, trang in 6347–6352. DOI: `10.1109/ICC52391.2025.11160761` (trang PDF 1).

Nguồn gốc: [PDF người dùng cung cấp](../../research/Evaluation_of_Missing_Data_Imputation_for_Time_Series_Without_Ground_Truth.pdf). [Metadata/hash](../tmp/pdfs/evaluation_without_ground_truth/source.json). Đã đọc bản trích xuất đủ sáu trang, đối chiếu trang render 3, 5, 6 cho giả định reference và kết quả hình. Không tuyên bố tái lập thực nghiệm của tác giả.

### C1 — WD/JSD trong paper cần reference, không phải phép đo accuracy không có giả định

**Claim — thiết kế của tác giả:** reference là dữ liệu trước gap; tác giả biện minh bằng sự giống phân phối giữa pre-gap và truth.

**E1 — P1, PDF trang 3 / trang in 6349, Table I, hàng “Ground truth requirement”:** “Not required, uses pre-gap data as reference”.

**E2 — cùng trang, §IV-A, đoạn “After filling the gaps…” và Figure 1:** “distribution of pre-gap values closely matches that of the true values”. [Trang render](../tmp/pdfs/evaluation_without_ground_truth/render-3.png).

**Analysis:** được hỗ trợ như mô tả phương pháp/biện minh của tác giả, không phải bảo đảm reference đúng trong mọi dataset. W1 một chiều trên tập giá trị không giữ thứ tự thời gian; hoán vị cùng tập giá trị cho cùng điểm. Đây là tính chất của phép đo đang dùng, không phải kết quả thực nghiệm mà paper chứng minh. Vì vậy điểm phân phối thấp không đủ xác nhận phục hồi đúng chuỗi COVID.

### C2 — Bài học có thể chuyển giao là kiểm chứng metric phụ trên dữ liệu có truth

**Claim — thiết kế thực nghiệm:** tác giả tạo gap có truth rồi đối chiếu metric không-truth với metric reconstruction.

**E3 — P1, PDF trang 3 / 6349, §IV-A:** “calculating and comparing them with traditional metrics, namely the RMSE and MAE”.

**E4 — P1, PDF trang 4 / 6350, §V-B:** “100 distinct artificial gaps with lengths ranging from 2 to 48 hours”; đoạn tiếp theo ghi kết quả được trung bình qua các gap. [Bản trích xuất trang 4](../tmp/pdfs/evaluation_without_ground_truth/page_04.txt).

**Analysis:** áp dụng quy trình kiểm chứng, không sao chép 100 gap hoặc 2–48 giờ sang dữ liệu COVID. Protocol dưới đây dùng masked MAE/MSE hiện có. MSE không phải RMSE của paper: cùng thứ hạng khi căn một MSE trên cùng tập ô, nhưng không được gọi trung bình MSE qua seed là trung bình RMSE.

### C3 — Kết quả paper chưa đủ để thăng WD/JSD thành metric xếp hạng chính ở dự án này

**Claim — phạm vi bằng chứng:** paper báo sự đồng thuận thực nghiệm trên hai dataset giao thông; phần đã đọc không cung cấp một bảo đảm chuyển giao sang dữ liệu COVID.

**E5 — P1, PDF trang 4 / 6350, §IV-B:** Telraam: “missing rate in this dataset is under 1%”; Madrid: “missing data of 0%”.

**E6 — P1, PDF trang 6 / 6352, §VI:** “statistical consistency rather than direct comparison with known values”. Figures 2–5 ở PDF trang 5–6 đối chiếu các đường metric theo gap size. [Trang 5](../tmp/pdfs/evaluation_without_ground_truth/render-5.png), [trang 6](../tmp/pdfs/evaluation_without_ground_truth/render-6.png).

**Analysis:** hình hỗ trợ nhận xét về nhóm phương pháp tốt/kém trong thiết lập của tác giả; không coi đó là chứng minh thứ hạng hoàn toàn đồng nhất. Trong bản sáu trang đã đọc, chưa thấy báo cáo định lượng rank agreement kèm bất định hoặc đặc tả ước lượng phân phối đủ để tái lập chính xác JSD. Kết luận cho dự án: WD giữ nhãn diagnostic; chưa thêm JSD. Đây là quyết định phạm vi của đề xuất, không phải kết luận rằng JSD vô ích.

## 3. Điều kiện dự án và giới hạn phải ghi trong báo cáo

- [Artifact profiling hiện có](../tmp/pdfs/evaluation_without_ground_truth/pre_gap_profile.json) ghi 1.298 chuỗi observed hoàn toàn, 1.252 chuỗi thiếu hoàn toàn, không có chuỗi mixed trên các dòng CSV được ghi; 82,27% ô observed bằng zero. Artifact không chứng minh cơ chế thiếu và không mô tả những dòng thời gian vắng hẳn khỏi CSV. Khi triển khai phải chạy lại [script profiling](../tmp/pdfs/evaluation_without_ground_truth/profile_pre_gap.py), kiểm tra hash dữ liệu/schema trước khi dùng số này.
- Không có pre-gap observed cùng chuỗi cho NaN hiện hữu theo artifact: không áp trực tiếp reference của P1 vào native missing.
- [data.py:117–157](../src/longitudinal_impute/data.py) tạo mask từ observed; `whole_feature` bỏ qua lựa chọn làm một dòng mất mọi target visible. Giống hình dạng thiếu không đồng nghĩa đại diện cơ chế thiếu thật.
- [Contract D03/D04/D08 và gate](THREE_BASELINE_CONTRACT_VI.md): Hron/Tsagris có Other nội bộ, MICoDa A17/R1 phân bổ toàn residual vào target thiếu. Giữ cấu hình hiện tại chỉ cho phép kết luận về adaptation hiện tại; **không dùng để xác nhận main model thắng thuật toán baseline**.

## 4. Protocol chạy cố định cho pilot

### 4.1 Input, mask và ngăn leakage

1. Đóng băng data/schema/model config, commit và hash, phiên bản thư viện, danh sách seed; không tuning bằng kết quả pilot rồi gọi lại chính kết quả đó là xác nhận.
2. Mỗi model fit mới trên cùng input visible và mask. NaN tự nhiên giữ nguyên là NaN trong input; không được dùng làm truth chấm lỗi.
3. **Kịch bản chính:** `whole_feature`, rate yêu cầu 0,05, seeds `[0,1,2,3,4]`.
4. **Kịch bản phụ:** `random_cell`, cùng rate/seeds, trình bày riêng; không trộn hai mask thành điểm tổng hợp. Tổng masked runs: 3 × 2 × 5 = 30. Native: một lần mỗi model/config, tái sử dụng cho hai kịch bản, không chạy lặp vô ích.
5. Rate là tỷ lệ ô observed yêu cầu che, không phải tỷ lệ quốc gia/chuỗi hay ước lượng tỷ lệ missing thật. Báo actual rate, số chuỗi che/bị bỏ qua và số chuỗi khác nhau qua seeds.
6. Truth hidden chỉ được truy cập trong lớp scoring/phân nhóm sau prediction. Preprocessing, zero replacement, fit và mọi reference đều chỉ được tính từ phần còn visible cùng thông tin ngoại sinh được phép. Không dựng total_target hoặc Other từ hidden truth đưa ngược vào model.
7. Đây là retrospective imputation, không phải forecasting; không áp thêm time split chỉ để tạo cảm giác chặt chẽ. Nếu sau này có câu hỏi dự báo tương lai thì thiết kế protocol riêng.

### 4.2 Metric và cách tổng hợp chốt trước

**Đề xuất ưu tiên proportion** để giảm chi phối trực tiếp bởi quy mô `total_sequence`; đây là lựa chọn mục tiêu đánh giá, không phải paper yêu cầu. Nếu mục tiêu khoa học thực sự là count, phải duyệt đổi ưu tiên trước khi chạy, không đổi sau khi thấy model thắng.

- **Metric chính của pilot:** MAE proportion, macro theo chuỗi quốc gia × target bị che trong mỗi seed: tính MAE các ô hidden của từng chuỗi rồi trung bình đều qua chuỗi. Mỗi seed có trọng số bằng nhau khi báo mean/sample SD.
- **Metric phụ cố định:** cell-weighted MAE/MSE count và proportion hiện có; macro MAE count theo chuỗi; bảng lỗi theo target. Không dùng một composite score để trộn count/proportion/WD.
- **Phân nhóm cố định:** all, truth_zero, truth_positive, forced_zero, model-decided. Bổ sung giao `model-decided ∩ truth_positive` từ prediction đã lưu để nhìn phần vừa không bị budget buộc zero vừa có tín hiệu thật. Các nhóm này chồng lấp, không cộng số ô giữa mọi nhóm. Không có ô thì báo NA cùng count=0, không điền metric=0.
- Với phân nhóm macro, chỉ trung bình những chuỗi có ô thuộc nhóm; luôn báo số chuỗi/ô làm mẫu số.
- **Zero predictor:** score zero trên đúng hidden cells làm đối chứng độ khó/zero dominance. Ghi rõ đây là diagnostic không nhất thiết hợp lệ theo budget, không phải baseline vận hành thứ tư.
- Giữ bảng cell-weighted cạnh macro để phát hiện kết luận phụ thuộc cách tổng hợp; mean/SD qua năm seed chỉ mô tả độ nhạy mask, không phải CI hay kiểm định độc lập theo ô.

### 4.3 Reliability và paired comparison

- Báo status completed/partial/failed, coverage hidden cells, warning convergence và lý do lỗi trên **mọi run đã lên lịch**. Completed không có nghĩa đã hội tụ.
- Chỉ chấm reconstruction khi output hợp lệ, completed và coverage=1, giữ contract hiện tại; không tự chấm riêng các ô dễ của partial output.
- Bảng accuracy so ba model dùng **giao seed cả ba thành công**, tách từng mask. Ghi rõ danh sách seed được giữ/loại và số mẫu. Nếu giao rỗng thì không có so sánh accuracy ba bên.
- Báo chênh lệch metric A−B trên cùng seed của tập chung, cùng với từng seed và mean/sample SD; không so hai mean lấy từ hai tập seed khác nhau. Tập riêng thành công mỗi model chỉ để mô tả, không xếp hạng chéo.
- Failure vẫn là kết quả reliability riêng; không quy thành zero error hoặc tự đặt một penalty trộn với accuracy.

### 4.4 Native evaluation không có truth

- Chạy một native completion mỗi model; kiểm tra observed bất biến, output finite/nonnegative, đúng ràng buộc ngân sách của adaptation và diagnostics.
- Giữ W1 count/proportion theo target với tên reference chính xác: `observed_vs_imputed` và `observed_vs_model_imputed`; báo số observed/imputed/forced/model cells. Reference gộp quốc gia/thời gian, **không phải pre-gap của P1**.
- `observed_vs_completed` chỉ giữ ở raw output để tương thích, không đưa vào bảng chất lượng chính vì completed chứa lại observed.
- Không lấy native WD để chọn winner, không cộng WD vào reconstruction score, không gán MAE/MSE cho NaN tự nhiên.

## 5. Phạm vi triển khai tối thiểu sau khi duyệt

**Một script khám phá**, gọi lại loader, mask, runner và metric đang có. Không sửa thuật toán/adapter, không thay CLI mặc định, không dựng hệ thống plugin metric.

Output cần:

1. `manifest.json`: hash, config, revision, environment, mask policy, metric policy và nhãn exploratory adaptation.
2. Masks theo seed; `hidden_predictions.csv`: model, mask, seed, row_id, entity, date, target, total, truth, prediction, forced_zero. Truth là artifact scoring, không dùng làm model input. Partial/failed giữ status và không được biến thành accuracy hợp lệ.
3. `run_status.csv`: toàn bộ run, failure, coverage, convergence warning, thời gian và mask stats, gồm visible-target support trên các dòng bị ảnh hưởng.
4. `reconstruction.csv`: per-seed/per-target/per-series/group và bảng paired trên tập chung; có mẫu số.
5. `native_diagnostics.json` và một báo cáo ngắn: reliability → reconstruction → zero/aggregation sensitivity → native diagnostic → giới hạn budget/missingness.

Runner hiện lưu mask và metric tổng hợp, chưa lưu masked predictions ([run.py:211–257](../src/longitudinal_impute/run.py)); không thể suy lại mọi bảng trên chỉ từ summary cũ.

### Kiểm thử tối thiểu trước chạy dữ liệu thật

- Mask chỉ là subset observed; cả ba model nhận cùng mask; truth hidden không xuất hiện trong input/preprocessing.
- Tái tính metric từ prediction khớp `masked_errors` trên fixture; kiểm tra macro bằng ví dụ hai chuỗi khác độ dài.
- Kiểm tra nhóm rỗng, forced-zero/model-positive, failure/partial và giao seed thành công; không có survivor bias ẩn do aggregate theo tập seed riêng.
- Zero predictor được tính đúng trên cùng ô; observed giữ nguyên và output validation vẫn chạy.
- Cùng seed/config tái lập mask và metric trong tolerance đã ghi; không hứa bitwise equality nếu thư viện/nền tảng không bảo đảm.

## 6. Gate kết thúc và nhánh mở rộng có điều kiện

**Pilot được coi là hoàn thành** khi 30 masked run đều có status (không đòi tất cả thành công), có native status của ba model, provenance/mask/prediction hoặc lỗi tương ứng, kiểm thử fixture đạt và báo cáo không vượt phạm vi chứng cứ.

**Không tự chạy thêm để tìm winner.** Nếu thứ hạng đảo theo seed/aggregation, target coverage không đủ hoặc failure làm mất nhiều đối chiếu, kết luận “chưa ổn định/chưa đủ coverage”, ghi nguyên nhân và đề xuất một lượt bổ sung riêng để duyệt. Không coi hàng nghìn ô cùng chuỗi hoặc các mask chồng lấp là mẫu độc lập.

**Chưa làm ở pilot:** JSD, contiguous-gap/pre-gap, reference quốc gia khác, sweep nhiều rate, tối ưu hyperparameter, bootstrap hoặc significance theo từng ô.

**Chỉ mở kiểm chứng WD khi có quyết định thực tế cần nó:** dùng hidden predictions có truth, reference chỉ từ visible theo target và cùng đơn vị đã công bố; kiểm tra reference-vs-truth trước, sau đó kiểm tra WD có chọn cùng phương pháp tốt hơn theo MAE trong cùng target/seed/đơn vị đánh giá hay không, báo coverage, ties và đảo hạng. Không lấy tương quan pooled khác scale làm bằng chứng. Nếu chọn/tuning reference bằng truth phải có phần giữ riêng trước khi thăng thành metric thường xuyên. Nhánh này chưa được P1 xác nhận cho COVID, không thuộc core pilot.

**Gate benchmark xác nhận:** có quyết định Other/budget chung được duyệt và không rò hidden truth; đóng băng lại cấu hình/protocol; đánh giá trên mask/chuỗi xác nhận giữ riêng với coverage phù hợp. Cùng protocol đo được không đồng nghĩa cùng điều kiện thuật toán. Dù vượt gate, kết quả masking vẫn chưa chứng minh accuracy tại 1.252 chuỗi thiếu thật.

## 7. Chốt phạm vi

Đề xuất triển khai tiếp theo chỉ là script và các kiểm thử/output tại §5, theo protocol §4. Văn bản này không cấp phép đổi budget, thuật toán hay chạy thí nghiệm mới; cần người dùng duyệt triển khai. Thành công của kế hoạch là có đánh giá trung thực và tái lập, **không phải bắt buộc tìm ra một baseline chiến thắng**.
