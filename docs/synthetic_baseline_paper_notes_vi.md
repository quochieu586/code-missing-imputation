# Căn cứ cho thảo luận dữ liệu synthetic

Ngày đọc: 2026-10-23. Đây là ghi chú nghiên cứu, chưa phải thiết kế đã chốt.

## Yêu cầu người dùng đã nêu

- So sánh khả năng điền khuyết của baseline khi biết ground truth.
- Cấu trúc địa điểm × thời gian, các biến thành phần giống COVID.
- Điều khiển mức thiếu, độ khó, MAR và MNAR.
- Tham khảo đặc biệt ba nguồn Hron, Tsagris, MICoDa trong thư mục research.

## Nguồn và bằng chứng

### Hron, Templ, Filzmoser (2010)

Nguồn local: ../research/Hron.pdf tính từ repo root, tức file ở thư mục research bên cạnh repository. Tên: Imputation of missing values for compositional data using classical and robust methods. Section 5; PDF pages 10–12.

- Sinh z ~ Normal đa biến, x = inverse_ilr(z): normal on the simplex.
- Thiết kế đầu: n=100, D=3; mean=(0,2), covariance=[[1.05,.95],[.95,1.05]]. Nhân mỗi hàng với U(0,1).
- Hai loại outlier: đổi tâm trong log-ratio và đổi scale; số mỗi nhóm từ 0 đến 40/100. Che thiếu chỉ trên nhóm không outlier.
- MCAR 20% ở biến 1, 10% ở biến 2, biến 3 không thiếu.
- Thiết kế D=10, n=100: diagonal covariance=1, off-diagonal=.9 hoặc .1 để thay đổi tương quan. Giữ biến cuối không thiếu.
- 1000 lần lặp; metric compositional error variance và sai khác covariance ILR.
- Tác giả nói đã thử MAR và có kết luận tương tự, nhưng Section 5 không cung cấp công thức cơ chế MAR để tái lập; không dùng câu này làm đặc tả MAR/MNAR.
- Không có cấu trúc thời gian/địa điểm trong các thiết kế mô phỏng trên.

Nguồn tác giả: https://file.statistik.tuwien.ac.at/filz/papers/10csda.pdf
DOI: https://doi.org/10.1016/j.csda.2009.11.023

### Tsagris, Stewart, Alenazi (bản local đề ngày 29/05/2026)

Nguồn local: research/tsagris.pdf bên cạnh repository. Section 3–4; PDF pages 11–15.

- Accuracy chủ yếu dựa trên bốn bộ dữ liệu thật, sau đó gây thiếu nhân tạo.
- Wines/Hydrochemical: chọn ngẫu nhiên 10% hàng, che nửa số thành phần trong các hàng đó, lặp 500 lần. Không được diễn giải 10% này thành 10% tổng ô thiếu.
- Adaptive experiment: sắp theo thành phần đầu rồi chia hai nửa, gây hai pattern thiếu khác nhau. Không tự gán nhãn MAR nếu chưa kiểm tra thành phần điều khiển có còn quan sát trong từng pattern hay không.
- Dữ liệu có zero: giữ zero; đánh giá bằng JSD khi Aitchison distance không xác định.
- Computational-cost experiment: Dirichlet với tham số ngẫu nhiên, n=500/1000/2000/5000/10000, D=10/15/20; chọn 10% hàng và che 30% thành phần trên các hàng được chọn; lặp 20 lần. Tác giả nêu rõ không quan trọng việc đặc tả tham số Dirichlet vì mục đích đo chi phí tính toán.
- Conclusion nói chưa đánh giá MNAR.
- Đây không phải bộ sinh longitudinal giống COVID.

Nguồn online đối chiếu: https://arxiv.org/abs/2605.29702

### Saha et al., MICoDa

File research/Sasha Micoda.pdf là sách 280 trang, không phải một bài độc lập; đọc Chapter 8: Multiple Imputation for Compositional Data (MICoDa) Adjusting for Covariates. Section 8.3, printed pages 162–167 (PDF 168–173); Section 8.4.2, printed 175–176 (PDF 181–182).

- V1: sinh 2 covariates liên tục, 2 nhị phân, 1 phân loại 4 mức (3 dummy); 6 thành phần gồm Others.
- Sinh log(y_d/y_other) bằng hồi quy tuyến tính theo covariates cộng nhiễu Normal; inverse ALR cho full data.
- Điều khiển hệ số hồi quy và sigma0=.25/.5/1.
- Xác suất một hàng có thiếu tau=.3/.5/.8; xác suất tau/3 cho mỗi trường hợp thiếu 1, 2 hoặc 3 thành phần. Đây là tỷ lệ hàng có thiếu, không phải tỷ lệ ô.
- n_total=150, train=100, test=50, 1000 lần lặp. Gold standard là mô hình hồi quy fit full data; rMSPE_miss trong mục này tính trên mọi thành phần của các hàng có thiếu, không phải chỉ ô bị che.
- Mô tả gây thiếu của V1 không cung cấp score logistic phụ thuộc covariates hoặc giá trị bị che; không suy ra một benchmark MAR/MNAR đầy đủ chỉ từ abstract nói về covariates.
- V2 microbiome: template HITChip 1006 người, 10 families; sinh Poisson lognormal n=150, thêm covariates và differential abundance; rarefy 100 reads/sample tạo khoảng 27% zeros được xử lý như NA trong ví dụ này. Structural zeros không được impute theo tác giả.
- Không tự đồng nhất zero sinh do lấy mẫu trong ví dụ này với structural zero hoặc NaN của COVID.

Trang nhà xuất bản: https://link.springer.com/book/10.1007/978-3-031-32729-2

### Nguồn bổ sung cho cơ chế MAR/MNAR

Schouten, Lugtig & Vink (2018), Generating missing values for simulation purposes: a multivariate amputation procedure.
https://doi.org/10.1080/00949655.2018.1491577
Full text tác giả: https://rianneschouten.github.io/mice_ampute/manuscript/manuscript.pdf

Paper tách sinh dữ liệu đầy đủ khỏi amputation, cho phép điều khiển patterns, tỷ lệ và cơ chế. MAR có thể dựa vào biến vẫn quan sát; MNAR có thể dựa vào chính giá trị bị che. Khi nhiều biến có thiếu cần kiểm soát joint patterns, tránh mặc nhiên coi các tỷ lệ theo biến hoặc theo hàng là tỷ lệ ô toàn bảng.

## Những điểm cần người dùng thiết kế tiếp

- Ground truth là composition liên tục, counts đầy đủ, hay cả latent composition và sampled counts?
- Tổng biết trước, Others, và khả năng suy ra phần thiếu từ ràng buộc tổng: nếu chỉ một thành phần thiếu trong composition đóng với tổng biết trước, có thể tính trực tiếp bằng phần dư.
- Động học thời gian, khác biệt địa điểm, tương quan, nhiễu, zero và outlier.
- Variables điều khiển MAR phải còn quan sát trong pattern tương ứng; MNAR phải được định nghĩa đối với thông tin quan sát thực tế.
- Mask rate theo ô hay hàng; độ mạnh cơ chế và tỷ lệ thiếu là các tham số riêng.
- Chưa chốt generator, tham số, metric, split hoặc phạm vi code.
