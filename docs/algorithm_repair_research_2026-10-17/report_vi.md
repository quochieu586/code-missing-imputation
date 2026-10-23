# Nghiên cứu hướng sửa bốn giới hạn baseline

Ngày: 17/10/2026  
Phạm vi: phân tích evidence và đề xuất thí nghiệm sửa; chưa đổi thuật toán hoặc chạy lại job benchmark.

## Kết luận điều hành

Các kết quả v2 vẫn là baseline được chấp nhận và phải giữ nguyên. Evidence hiện đủ để đưa ra hướng sửa, nhưng chưa đủ để tuyên bố đã sửa hoặc xác nhận hiệu năng mới. MICoDa-V2 giảm thay đổi qua vòng lặp nhưng chạm trần 50 trước tolerance; Hron-LS và MICoDa-V1 đều bị giới hạn bởi budget đã đăng ký; Hron-LTS lỗi vì trạng thái đưa vào CLR có ít nhất một phần không dương. Nguyên nhân số học cuối cùng của LTS chưa có traceback nên cần chẩn đoán có ghi nhận ở bước tiếp theo.

Đề nghị ưu tiên theo thứ tự: (1) instrument Hron-LTS và chặn/floor có khai báo các phần suy biến trước CLR; (2) profile rồi vector hóa đường V1 và Hron-LS mà giữ parity; (3) kiểm tra hội tụ MICoDa-V2 trên miền các ô được điền, với trần vòng lặp tăng có giới hạn. Không nới budget sau khi nhìn test rồi gọi điểm số mới là đánh giá xác nhận.

## Evidence đã đọc

- Handoff, progress và checks của benchmark đều ghi run `experiments/offline_baselines_2026-10-15_v2/` đã hoàn tất; 8 check bắt buộc pass. Các artifacts v2 có manifest hash và nguồn được đóng băng. Không job nào được chạy lại trong nghiên cứu này.
- `supplementary_status.csv`: Hron-LS và MICoDa-V1 mỗi phương pháp có 5/5 fold bị timeout ở 180 giây/fold/method. Hron-LTS có 5/5 lỗi với cùng thông báo `CLR requires finite positive parts; use an explicit encoder zero policy`, mất 15.81–45.61 giây. Không có score hợp lệ để xếp hạng ba trường hợp này.
- Primary MICoDa-V2 dùng W=1e-6, max_iter=50; cả 5 fold đều `converged=false`. Relative change vòng cuối theo fold là 3.53e-5, 9.65e-5, 8.20e-5, 1.30e-4 và 6.80e-4. Full-data refit dừng ở 50 với 8.26e-5. Các thay đổi giảm qua phần cuối trace, nhưng chưa đạt tol=1e-8; đây không phải bằng chứng mô hình đã hội tụ.
- `resource_pilot.json` đo MICoDa-V2 fit 7,005 hàng trong 8.93 giây với 50 vòng. Hron-LS pilot trên cùng cỡ train mất 55.93 giây và báo một vòng, chưa hội tụ; con số này chỉ là một probe, không phải benchmark tốc độ chuẩn. Source hiện tại có Hron cập nhật theo target trong `_sweep`, còn MICoDa-V1 chạy vòng target bên trong vòng hàng trong `_update`.
- Module liên quan nằm ở `nextgen/src/longitudinal_impute/baselines/{micoda.py,hron.py,common.py,adapters.py,regression.py}`; các bài test baseline ở `nextgen/tests/test_general_baselines.py`. Report benchmark xác nhận adapter thay zero bằng delta dương trước khi Hron fit. Vì vậy zero thô trong input không giải thích được lỗi CLR; phần không dương nhiều khả năng phát sinh sau biến đổi ngược ở một vòng Hron, nhưng chưa được chứng minh vì log hiện chỉ lưu thông báo lỗi.

## Hướng xử lý theo phương pháp

### MICoDa-V2: đo hội tụ đúng phần đang được điền

Trace hiện giảm đều nhưng chậm và còn xa tolerance. Chỉ tăng `max_iter` có thể kéo dài toàn bộ lưới fit; không nên nâng lên một con số lớn rồi bỏ cờ hội tụ. Thước đo hiện tại `||new-state|| / ||state||` tính cả hàng/phần không đổi. Nên bổ sung diagnostics trên các ô thiếu được update: relative norm, max absolute change, và sai khác ALR dự đoán giữa hai lượt. Giữ cả metric hiện hành để so sánh, tránh đổi nghĩa cờ cũ mà không báo.

Thí nghiệm engineering-only đề nghị thử giới hạn 100, 200 rồi tối đa 500 vòng trên dữ liệu train, ghi trace đầy đủ và thời gian mỗi vòng. Nếu trace tiếp tục giảm, chạy tăng vòng có thể hợp lý; nếu có plateau, dao động hoặc phần imputed vẫn đổi đáng kể, cần nghiên cứu damping/Anderson acceleration có safeguard như một thuật toán mới và so parity trên fixtures. Chọn tolerance/ngưỡng ổn định bằng validation và khai báo trước; kiểm tra cả độ ổn định score, thay đổi trên các giá trị được điền, tính hữu hạn và ràng buộc simplex/count. Không gọi là hội tụ khi chỉ chạm max_iter.

Paper MICoDa đặt phương pháp trong bối cảnh ALR, covariates và multiple imputation; bản triển khai hiện trả point estimates deterministic. Nếu mục tiêu cuối là multiple imputation/proper uncertainty, tăng số vòng lặp point estimator không đáp ứng mục tiêu đó. Giữ rõ khác biệt này trong report.

### Hron-LS: giảm chi phí mỗi sweep, không cắt vòng tùy tiện

Pilot mất 55.93 giây cho một vòng trên khoảng 7,000 hàng; job chính supplementary cho phép tổng cộng 180 giây bao gồm fit, ba mask seeds, dự đoán, scoring và I/O. Do vậy tăng thời gian budget đơn thuần có thể cần tăng nhiều lần và vẫn chưa bảo đảm hội tụ. Trước tiên thêm timing theo sweep/target, rồi profile lại trên train-only probe mới.

Các điểm tối ưu ít đổi thuật toán: tránh tính lại cùng thiết kế/balance matrix khi state không đổi; vector hóa biến đổi theo target ở mức ma trận; hạn chế lặp `matrix_rank`/factorization khi design có cùng hạng; và tái sử dụng phép phân rã LS hợp lệ trong phạm vi một target/sweep. Mọi cache phải bị vô hiệu hóa khi state thay đổi. Giữ `max_iter`, quy tắc covariance stop và kết quả toán học như cũ ở phép parity nhỏ; đo sau đó mới quyết định budget. Nếu vẫn quá chậm, báo rõ đây là giới hạn chi phí của phương pháp trong giao thức, hoặc chỉ tiếp tục nếu có budget/phạm vi nghiên cứu mới được đăng ký trước.

### MICoDa-V1: vector hóa vòng phân bổ theo hàng

V1 hiện có vòng Python qua từng hàng thiếu rồi qua từng target; sau đó adapter cũng gọi `predict_result` theo từng query row để giữ fallback theo dòng. Đây là điểm nghi ngờ về runtime dựa trên cấu trúc source, chưa có cProfile đủ để kết luận nó chiếm bao nhiêu thời gian. Đo riêng fit, `_update`, predict và constraint. Nếu profile xác nhận overhead này, vector hóa các hàng theo từng target, giữ nguyên thứ tự recalibration và logic residual/reference của V1. Không gộp V1 vào công thức V2 vì hai chế độ có semantics khác.

Acceptance cho tối ưu: so output cũ/mới trên oracle R đã pin và fixtures nhỏ, đo cả sai khác coefficient/fit history, rồi kiểm boundary hidden labels, observed lock và tổng count. Nếu thứ tự phép toán làm kết quả khác vài ULP, báo tolerance đã đăng ký thay vì khẳng định bitwise parity. Sau unit/parity mới chạy profiling train-only; không lấy score test hiện có để chọn vectorization.

### Hron-LTS: bắt nguồn zero trước khi sửa

`common.clr` từ chối zero và không-finite. Adapter đã áp zero replacement dương trước khi core Hron chạy; `stable_softmax` trừ max để chống overflow nhưng `exp(logit-max)` vẫn có thể underflow thành đúng 0. Một bước Hron-LTS có thể tạo part bằng 0 rồi vòng sau gặp đúng lỗi CLR. Đây là giả thuyết khớp code path và thông báo, chưa phải root cause đã xác minh. Thông tin hiện hữu không ghi iteration, target hay min part trước lỗi.

Thêm chẩn đoán `.py` ở ranh giới mỗi sweep: iteration/target, min/max logits, số lượng giá trị softmax bằng 0, min positive state, norm hệ số LTS, solver/starts và seed; khi thất bại giữ traceback cùng các con số này. Chạy đúng một probe fold mới trong thư mục nghiên cứu riêng sau khi instrument, không ghi vào run v2. Nếu xác nhận underflow, so sánh hai cách có khai báo: (a) floor theo zero policy đã fit trên train rồi closure lại, (b) giới hạn log-ratio/balance dự đoán theo ngưỡng preregistered. Không thêm epsilon vô danh trong `clr`, vì nó che lỗi cho mọi caller và làm đổi dữ liệu. Kiểm độ nhạy trên validation; giữ nguyên raw LTS hoặc công bố đó là biến thể solver khác.

Để tăng tốc LTS, bản hiện tại dùng seeded random starts cộng C-steps nhưng không dùng FAST-LTS đầy đủ. Có thể đánh giá một solver FAST-LTS đã mô tả trong literature, nhưng phải ghi rõ số starts, C-step, trimming h, intercept correction/reweighting và mức xấp xỉ. So với objective trimmed trên bài nhỏ có nghiệm vét cạn trước khi dùng ở data thật. Đổi solver là thay đổi phương pháp, không phải tối ưu thuần túy.

## Trình tự thí nghiệm được đề nghị

1. **Chỉ sửa diagnostics/tests:** instrument các timing và trạng thái số học trong module `.py`; thêm test softmax extreme-logit, test convergence trace MICoDa, và oracle bảo toàn semantics V1. Chạy unit tests liên quan, không chạy engine benchmark.
2. **Engineering pilots trên train-only:** MICoDa max_iter 100/200/500; Hron-LS profiling một sweep và multi-sweep nhỏ; MICoDa-V1 cProfile và parity; Hron-LTS một fold lỗi với traceback. Ghi mỗi pilot vào output mới, với time cap và không tính score test.
3. **Đóng băng candidate và preregister:** khóa source/config/hash, solver/zero policy/tolerance/iteration/time budget trước khi mở đánh giá. Các test locations/masks cũ đã được dùng để lập report; mọi kết quả sửa trên chúng chỉ là exploratory. Muốn claim xác nhận cần cohort/locations độc lập chưa dùng hoặc dữ liệu mới.
4. **Chỉ đánh giá candidate đạt engineering gates:** kiểm định finiteness/positivity, observed boundary, count budgets, convergence, deterministic replay, runtime và score validation. Giữ mọi failure/resource-limit trong báo cáo, không thay bằng baseline khác.

## Sources

- Hron, Templ & Filzmoser (2010), *Imputation of missing values for compositional data using classical and robust methods*, DOI [10.1016/j.csda.2009.11.023](https://doi.org/10.1016/j.csda.2009.11.023); bản toàn văn tại [Masaryk University repository](https://is.muni.cz/do/rect/habilitace/1431/Hron/habilitace/8_Hron_et_al__2010_.pdf). Phương pháp gốc trình bày kNN và iterative regression trong transformed composition, dùng LS hoặc robust regression.
- Saha et al. (2023), *Multiple Imputation for Compositional Data (MICoDa) Adjusting for Covariates*, [Springer chapter](https://link.springer.com/chapter/10.1007/978-3-031-32729-2_8), DOI [10.1007/978-3-031-32729-2_8](https://doi.org/10.1007/978-3-031-32729-2_8). Chapter mô tả MICoDa là phương pháp multiple imputation có điều chỉnh covariates; implementation local còn có author-code snapshot pin trong BASELINES_VI.md.
- Rousseeuw & Van Driessen (2006), *Computing LTS Regression for Large Data Sets*, [Springer](https://link.springer.com/article/10.1007/s10618-005-0024-4). FAST-LTS dùng chiến lược starts/C-steps và chọn nghiệm trimmed objective cho dataset lớn; áp dụng cần phân biệt rõ khỏi raw multistart hiện tại.
- Filzmoser (2021), *Robust linear regression for high-dimensional data: An overview*, [WIREs Computational Statistics](https://wires.onlinelibrary.wiley.com/doi/10.1002/wics.1524). Tổng quan mô tả LTS/C-steps, ảnh hưởng của khởi tạo và các lựa chọn robust regression.

## Trạng thái

Đây là đề xuất nghiên cứu, chưa có sửa code, chưa có candidate đạt check mới và chưa có bằng chứng hội tụ/speed/fix LTS. Các score và files trong v2 vẫn giữ nguyên. Mọi implementation tiếp theo phải nằm trong module `.py`, có thí nghiệm mới và evidence riêng.
