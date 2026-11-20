# Ghi chú thuyết trình — main model hiện tại

Tám slide tiếng Anh gồm bìa và References. Người thực hiện là Huu Phuc Le;
ngày để trống. Các đoạn dưới đây là gợi ý nói theo nội dung hiện tại.

## 1. Estimating Missing Counts in Repeated Measurements (~20 giây)

**Ý chính:** mô hình ước lượng count thiếu qua các lần đo lặp lại và giữ count
quan sát được cùng total đã biết.

**Có thể nói:** “Tôi trình bày cách ước lượng count thiếu trong dữ liệu đo lặp lại.
Thiết kế trong main_model.py và data.py giữ các count đã biết cùng total.”

## 2. Imputation Framework (~1 phút 15 giây)

**Ý chính:** mỗi dòng có counts, total, entity và thời gian. Ví dụ, một quốc gia
báo cáo số ca bệnh. D targets và Other nội bộ tạo K=D+1 parts.

**Có thể nói:** “Tsagris khởi tạo count. CLR tạo features từ proportions sau khi
thay zero để lấy log [5]. Mạng nhận features. Bước khôi phục dùng known counts
và total. Mạng không thực hiện khởi tạo hoặc khôi phục counts.”

**Đọc hình 1:** visible counts đi qua initialization, CLR/network rồi restoration.
Caption dẫn main_model.py, phần initialization và prediction. Điều cần chú ý là
hai bước ngoài mạng. Hình dùng 0.9 chiều rộng văn bản.

**Toán/ký hiệu:** z_ij bằng log proportion đã xử lý zero trừ trung bình log trên
K parts. Các giá trị này là features, không phải counts.

**Chuyển:** “Ta so sánh các thiết kế mô hình chuỗi nào?”

## 3. Comparing Sequence Backbones (~2–3 phút)

**Ý chính:** bảng so sánh thiết kế, không trình bày kết quả thực nghiệm.

**Cách đọc bảng:** ba cột là Main idea, Advantage và Weakness. Tên mô hình nằm
trong cột đầu. Ba dòng tổng kết cũ đã được bỏ. Caption của Table 1 dẫn [1]–[4]
và nhắc mỗi thiết kế có giới hạn bộ nhớ hoặc tính toán khác nhau.

**Vanilla RNN [1]:** mạng đọc từng bước theo thứ tự và nhớ bước trước. Nó dùng
thứ tự chuỗi. Tín hiệu học nhỏ từ các bước xa làm hạn chế bộ nhớ.

**LSTM [2]:** long short-term memory dùng cổng để chọn thông tin giữ, quên và xuất.
Cổng giúp giữ thông tin qua nhiều bước. Các bước chạy lần lượt nên hạn chế
huấn luyện song song.

**CNN/TCN [3]:** mạng tích chập lọc nhóm bước. Các bước được xử lý song song;
đầu vào bộ lọc cách quãng giúp nhìn xa hơn. Bộ lọc có khoảng thời gian hữu hạn.
Thiết kế chỉ dùng quá khứ không dùng bước tương lai.

**Transformer [4]:** attention đặt trọng số lên thông tin từ các bước được phép.
Nó nối bước xa và xử lý song song. Full pairwise attention có chi phí tăng theo
bình phương độ dài chuỗi. Nó cần mã thời gian và quy tắc liên kết.

**Chuyển:** “Training và evaluation dùng những ô nào?”

## 4. Separate Training and Evaluation (~1 phút 25 giây)

**Ý chính:** H đánh dấu ô evaluation. Q đánh dấu ô training. Hai tập không giao nhau.

**Có thể nói:** “Ta loại Q trước khi khởi tạo training. Ta giữ Q cố định và loại
H khỏi cả hai views. Hai nhánh dùng chung theta, không dùng chung completions.”

**Đọc hình 2:** nhánh training loại Q rồi khởi tạo, qua mạng và restore trước loss
trên Q. Nhánh inference khởi tạo riêng và score H sau restore. Hình dùng 0.9
chiều rộng; caption dẫn main_model.py, phần training và prediction.

**Đọc loss:** Q_B chứa các ô training trong entity batch B. Nhãn y_ij=c_ij/T_i;
T_i là total dòng. C-hat^T là count training đã khôi phục. Loss đo sai số bình
phương trên proportions của các ô Q_B.

**Chuyển:** “Mạng thực hiện các phép biến đổi nào?”

## 5. Network Architecture (~1 phút 50 giây)

**Ý chính:** với mỗi entity, attention đặt trọng số theo thời gian rồi theo parts.

**Có thể nói:** “Linear nhận Z và m rồi cộng e_f và e_t. Mô hình học part codes e_f;
sóng sine cố định mã hóa thời gian e_t. Time attention, part attention và gate
tạo một residual block. Mỗi bước normalization đổi thang các giá trị.”

**Đọc hình 3:** hình dùng 0.8 chiều rộng và dẫn network.py:39–60. Mỗi attention
trao đổi thông tin theo một trục. Các update có residual links.

**Phương trình gate:** LayerNorm của h^(2) tạo u,g qua W_g và b_g. h^(3) cộng
tanh(u) nhân từng phần tử với sigmoid(g) vào h^(2).

**Head:** mạng ánh xạ C đầu vào qua C units rồi một output.
Softmax tạo K trọng số dương có tổng bằng một. Z chứa CLR features;
m đánh dấu ô observed. Layer normalization (LN) đổi thang giá trị ẩn. C mặc định
bằng 32. Slide không nêu heads/dropout.

**Chuyển:** “Ta chia phần total còn lại thế nào?”

## 6. Restoring the Count Budget (~1 phút 35 giây)

**Ý chính:** restoration giữ visible counts và chia total còn lại. V_i chứa visible
targets. F_i gồm missing targets M_i và Other o.

**Có thể nói:** “Ta giữ c_ij trên V_i. R_i bằng total trừ tổng visible counts.
Ta chuẩn hóa proposed weights trên F_i rồi nhân với R_i.”

**Đọc hình 4:** hình dùng 0.9 chiều rộng và dẫn data.py:162–217 cùng
main_model.py:166–208. Count đã biết T1=30 được giữ. Với T=100, còn 70. Các weights
.18/.12/.30 chuẩn hóa thành .30/.20/.50, tạo T2=21, T3=14 và Other=35. Targets
xuất ra có tổng 65; Other giữ nội bộ. Đây là ví dụ minh họa, không phải kết quả
thực nghiệm. Caption nhắc quy tắc giữ count đã biết.

**Chuyển:** “Các count này được cập nhật qua round thế nào?”

## 7. Iterative Refinement (~1 phút 20 giây)

**Ý chính:** mỗi round fit training features rồi restore hai views riêng.

**Đọc hình 5:** hình dùng 0.8 chiều rộng và dẫn main_model.py:211–339. Training
counts đi qua fit theta rồi restore training. Inference counts dùng theta đó
để restore riêng. Caption nhắc trọng số dùng chung giữa hai views.

**Đọc tiêu chí dừng:** x_I chứa inference counts; M_I chứa missing targets.
Delta_r là thay đổi count lớn nhất giữa hai rounds chia total dòng. Dừng khi
Delta_r không vượt epsilon hoặc khi round đạt R_max. Mỗi round giữ parameters,
reset Adam và giữ masks cùng labels.

**Stage handoff:** initialization lưu một state. Refinement cập nhật counts.
Prediction tạo outputs. Evaluation kiểm tra outputs mà không training.
Nguồn trên slide là run.py:288–431.

**Kết luận/giới hạn:** tuning và component tests còn phía trước. Chưa bảo đảm hội tụ.
Masks dùng chung random streams. Restoration có thể từ chối nhầm decimal counts.
Slide dẫn audit dự án ngày 2026-11-13.

## 8. References (~15–30 giây hoặc dùng khi hỏi đáp)

Có năm tài liệu [1]–[5]. Citation [1]–[4] ở slide3; [5] ở slide2. Các DOI và
publisher/arXiv links dẫn tới nguồn gốc của các claims.

Mapping: [1] Bengio1994; [2] Hochreiter/Schmidhuber1997; [3] Bai2018 preprint;
[4] Vaswani2017; [5] Martín-Fernández2003.

## Nguồn và phạm vi

SOURCE_MAP_VI.md ghi các module và literature locators. Hình mô tả thiết kế
và số học minh họa, không biểu diễn kết quả thực nghiệm.
