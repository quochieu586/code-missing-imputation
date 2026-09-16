# Phạm vi lượt CLR + diffusion

Yêu cầu trực tiếp của người dùng: tiếp nối Stage A, chạy một lượt và đánh giá. Vì vậy không triển khai/chạy vòng lặp ngoài E2 hoặc T17 nhiều vòng. `max_iter != 1` bị từ chối; kết quả trả `n_iter=1`, `converged=False`.

## Những ràng buộc đã triển khai

- Một composition = một địa điểm/ngày × 17 biến thể. CLR chạy trên trục biến thể, kiểm tra hữu hạn/dương. Z1 và Z2 được lưu riêng.
- Zero là quan sát; missing gốc có mask riêng. Mask benchmark/validation lưu tách biệt; giữ nguyên trong train/sample.
- Target tự giám sát theo ô và chỉ thuộc quan sát train khả dụng; validation/test bị che trước preprocessing/KNN.
- Tách theo địa điểm (đơn vị tương ứng subject); cân bằng fold theo việc địa điểm có hàng đầy đủ, không dùng abundance để chia.
- Pseudo-count, thống kê baseline, KNN và fallback benchmark chỉ dùng tập train. k=8 giữ từ Stage A, chưa chạy chọn k bằng inner CV.
- Cầu nối vào conditioning được tính lại sau khi thay target tự giám sát bằng median train, tránh lộ target qua trung bình log CLR.
- Epsilon Gaussian không phải CLR. Loss dùng đúng epsilon-MSE (§5.1); không chiếu epsilon. Mẫu CLR cuối được trung tâm hóa; metric dùng CLR của composition cuối.
- Khóa counts quan sát bit-identical; báo C1/C2/C3, Shannon, tỷ lệ abundance dưới ngưỡng, M1 theo ba nhóm, M2 và M3.
- Có regression test cho loss bỏ qua NaN/±1e9 ngoài target, oracle epsilon chính xác, phương trình reverse DDPM, train/sample, mask, observed lock, feature order train-only, fallback train-only và giới hạn một vòng.

## Các điểm phải diễn giải rõ

1. Handoff §4.3 (softmax tổng 1) và T12 (counts quan sát bit-identical) không thể cùng đúng trực tiếp cho dữ liệu counts. Code lấy tỷ lệ softmax, neo scale bằng median log-ratio quan sát, rồi khóa lại counts. CLR được tính lại từ output cuối, vì khóa counts có thể đổi tâm log.
2. Ground truth CLR đầy đủ không xác định cho hàng vốn thiếu. Train dùng tâm CLR của X1; chỉ tọa độ quan sát được làm target. Evaluation chỉ chấm các hàng vốn đầy đủ trước khi che. Không lấy KNN làm ground truth.
3. Model pilot tham khảo source CSDI chính thức nhưng dùng kiến trúc nhỏ, thêm conditioning từ khởi tạo và cửa sổ thời gian; không phải bản tái lập nguyên kiến trúc/thí nghiệm Seki.
4. Dữ liệu là biến thể SARS-CoV-2, không có phylum map. Runner dùng standard CSDI thích ứng. Nhánh CNN theo nhóm và helper sắp Spearman có trong code, nhưng không được gọi là kết quả phylogenetic CNN khi chưa có mapping phù hợp.
5. Các hàng đầy đủ chỉ thuộc 5 địa điểm; fold pilot chấm một địa điểm. Kết quả không đại diện cho mọi địa điểm hoặc cơ chế thiếu tự nhiên. Không có bằng chứng MNAR hay downstream.
6. Lượt debug đầu bị loại do sai objective/projection; vẫn giữ dưới `artifacts/stage_b_single_pass` và đánh dấu `REJECTED_IMPLEMENTATION_BUG`. Kết quả dùng báo cáo nằm ở `artifacts/stage_b_single_pass_corrected`. Chạy lại sau sửa bug không phải lặp cập nhật X2 thành X3.

## Các cổng chưa đạt — không tuyên bố hoàn thành toàn bộ handoff

- T18: tái lập một ô DIABIMMUNE trong ±1 SD.
- E1: so diffusion với các khởi tạo mean/LOCF/linear/KNN. Bảng baseline đơn thuần hiện tại không thay thế E1.
- T19: baseline kNN+LTS(ilr).
- 5-fold mean (SD), lưới nhiều tỷ lệ, downstream.
- E2/H1/T17: không thực hiện theo phạm vi một lượt của người dùng; không kết luận vòng lặp có hay không có ích.
