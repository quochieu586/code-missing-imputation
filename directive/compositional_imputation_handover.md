# Hợp đồng bàn giao — Điền khuyết (imputation) cho dữ liệu thành phần (compositional data)

**Nguồn duy nhất:** Hron K., Templ M., Filzmoser P. — *Imputation of missing values for compositional data using classical and robust methods*, preprint gửi Computational Statistics & Data Analysis, 27/11/2009.
**Phạm vi:** chắt lọc 2 hướng xử lý do bài báo đề xuất, kèm đặc tả cài đặt, kết quả kỳ vọng, ràng buộc, và bộ kiểm thử nghiệm thu.
**Không thuộc phạm vi:** mọi phương pháp không có trong bài báo; mọi con số không nằm trong bảng/kết quả của bài báo.

---

## 0. Tóm tắt điều hành

| | Hướng 1 | Hướng 2 |
|---|---|---|
| Tên | kNN theo khoảng cách Aitchison | Điền khuyết lặp dựa trên mô hình (iterative model-based) trong không gian ilr |
| Bản chất | dựa trên khoảng cách (distance-based) | dựa trên hồi quy đa biến (model-based) |
| Có lặp? | Không | Có |
| Khởi tạo | không cần | **bắt buộc dùng kết quả Hướng 1** |
| Biến thể | adjustment theo tổng (5) hoặc theo median (7) | hồi quy LS hoặc LTS |
| Vai trò | ổn định, an toàn, dùng làm khởi tạo | chính xác hơn, khai thác quan hệ đa biến |
| Điểm yếu | không dùng đầy đủ quan hệ đa biến; nhạy mẫu nhỏ | không có chứng minh hội tụ; LS sập khi có outlier |

**Nguyên tắc xuyên suốt:** dữ liệu thành phần chỉ mang thông tin **tương đối** (tỷ lệ giữa các phần). Mọi thao tác — đo khoảng cách, hồi quy, đánh giá sai số — phải thực hiện trong hình học Aitchison, **không** phải hình học Euclid. Bỏ qua điều này là nguồn sai số lớn nhất, lớn hơn cả việc chọn thuật toán.

---

## 1. Nền tảng bắt buộc phải hiểu trước khi code

### 1.1 Định nghĩa và lớp tương đương

Một quan sát `x = (x₁,…,x_D)ᵀ` là một **D-part composition** khi và chỉ khi mọi thành phần **dương thực sự** và toàn bộ thông tin liên quan nằm trong các **tỷ lệ** giữa chúng.

Hệ quả then chốt:

```
x̲ = { c·x , c ∈ ℝ⁺ }
```

`x` và `c·x` là **cùng một thông tin** (gọi là compositionally equivalent). Tổng các phần `κ` có thể khác nhau giữa các quan sát và **không được phép ảnh hưởng tới kết quả điền khuyết**.

*Ý nghĩa thực tế:* nếu mẫu vật chỉ được phân tích một phần, hoặc khối lượng mẫu khác nhau, tổng nồng độ sẽ khác nhau — nhưng thông tin hóa học không đổi.

Ràng buộc tổng hằng kéo theo: D phần chỉ có **D−1 chiều tự do** → ma trận dữ liệu suy biến theo định nghĩa. Đây là lý do PCA/correlation áp thẳng lên dữ liệu thô cho kết quả sai lệch (đã biết từ Pearson 1897).

### 1.2 Khoảng cách Aitchison

```
d_A(x,y) = sqrt( (1/D) · Σ_{i=1}^{D-1} Σ_{j=i+1}^{D} ( ln(x_i/x_j) − ln(y_i/y_j) )² )
```

**Tính chất phải kiểm tra bằng test:** `d_A(c·x, y) = d_A(x, y)` với mọi `c > 0`.

*Ý nghĩa:* khoảng cách này nhận biết được rằng chênh lệch 0.1→0.2 (hệ số 2) là lớn hơn nhiều so với 0.5→0.6 (hệ số 1.2), dù khoảng cách Euclid bằng nhau. Bài báo đưa ví dụ: hai điểm biên có `d_A = 0.57`, hai điểm trung tâm có `d_A = 0.29`.

### 1.3 Phép biến đổi ilr (isometric logratio)

Dùng ilr, **không** dùng alr, **không** dùng clr:

- **clr**: bảo toàn khoảng cách nhưng sinh dữ liệu suy biến → hỏng ước lượng robust.
- **alr**: về ℝ^{D−1} nhưng **không đẳng cự** → không khuyến nghị.
- **ilr**: về ℝ^{D−1}, đẳng cự, cho phép dùng thẳng thống kê Euclid chuẩn.

Tính chất đẳng cự:

```
d_A(x, y) = d_E( ilr(x), ilr(y) )
```

**Cơ sở balance được chọn (sequential binary partition), công thức (8):**

```
z_j = sqrt( (D−j) / (D−j+1) ) · ln( ( Π_{l=j+1}^{D} x_l )^{1/(D−j)} / x_j ),   j = 1,…,D−1
```

**Nghịch đảo, công thức (9)–(11):**

```
x₁ = exp( − sqrt((D−1)/D) · z₁ )
x_j = exp( Σ_{l=1}^{j−1} z_l / sqrt((D−l+1)(D−l))  −  sqrt((D−j)/(D−j+1)) · z_j ),   j = 2,…,D−1
x_D = exp( Σ_{l=1}^{D−1} z_l / sqrt((D−l+1)(D−l)) )
```

Nghịch đảo trả về **một đại diện của lớp tương đương**, không phải vector gốc. Nếu cần tổng hằng thì chuẩn hóa sau.

**Vì sao chọn đúng cơ sở này?** `z₁` gói toàn bộ thông tin tương đối của `x₁` so với tất cả phần còn lại; `z₂,…,z_{D−1}` không chứa `x₁`. Khi ta xếp biến nhiều missing nhất lên vị trí 1, giá trị điền khuyết tồi ở `x₁` chỉ làm bẩn `z₁` — tức là chỉ làm bẩn **vế trái** của phương trình hồi quy, không làm bẩn biến giải thích. Đây là cơ chế chống **lan truyền sai số (error propagation)** và là lý do kỹ thuật quan trọng nhất của Hướng 2.

**Hệ quả cần biết khi debug:** theo công thức nghịch đảo, thay đổi `z_l` không ảnh hưởng `x_1..x_{l−1}`; nó làm đổi `x_l` và đóng góp **một lượng như nhau** vào mọi `x_j` với `j > l`. Do đó tỷ lệ giữa các phần sau vị trí `l` được bảo toàn — đúng như bước 7 của thuật toán mô tả.

---

## 2. HƯỚNG 1 — kNN theo khoảng cách Aitchison

### 2.1 Ký hiệu

- `x_i = (x_{i1},…,x_{iD})ᵀ`, `i = 1..n`
- `M_i ⊂ {1,…,D}`: tập chỉ số ô **missing** của quan sát `i`
- `O_i = {1,…,D} \ M_i`: tập chỉ số ô **quan sát được**

### 2.2 Chọn chiến lược tìm láng giềng

Bài báo liệt kê 4 biến thể:

| Mã | Mô tả | Dùng? |
|---|---|---|
| 1a | điền đồng thời mọi ô, tìm k-NN trong các quan sát **đầy đủ** | không |
| 1b | điền đồng thời, cho phép láng giềng khuyết miễn có đủ thông tin cần | không |
| **2a** | điền **tuần tự từng ô**; láng giềng phải có đủ thông tin tại `O_i` **và** tại biến đang điền | **CHỌN** |
| 2b | như 2a nhưng nới thêm điều kiện | không |

**Quyết định bàn giao: dùng 2a.** Lý do bài báo đưa ra: nói chung sẽ có nhiều láng giềng hợp lệ hơn, và yêu cầu nhiều thông tin hơn trên mỗi quan sát dẫn tới kết quả điền khuyết đáng tin hơn. Khác với nhóm (1), ở nhóm (2) tập k láng giềng **thay đổi** theo từng ô được điền.

### 2.3 Thuật toán

Để điền ô `x_{ij}` với `j ∈ M_i`:

**Bước 1.** Lọc tập ứng viên: mọi quan sát `x_{i_l}` có giá trị **không khuyết** tại vị trí `j` **và** tại toàn bộ `O_i`.

**Bước 2.** Tính `d_A(x_i, x_{i_l})` trên **subcomposition** gồm các phần trong `O_i`. Lấy k quan sát gần nhất.

> Không cần hiệu chỉnh scale ở bước này — `d_A` bất biến với nhân vô hướng dương, đây chính là lý do Aitchison distance phù hợp.

**Bước 3.** Tính hệ số hiệu chỉnh (đây là bước dễ bỏ sót nhất):

Bản gốc, công thức (5):
```
f_{i,i_l} = ( Σ_{o ∈ O_i} x_{io} ) / ( Σ_{o ∈ O_i} x_{i_l o} )
```

Bản robust, công thức (7) — **dùng bản này**:
```
f*_{i,i_l} = ( median_{o ∈ O_i} x_{io} ) / ( median_{o ∈ O_i} x_{i_l o} )
```

*Ý nghĩa:* láng giềng "giống về tỷ lệ" nhưng có thể ở **quy mô tổng thể khác hẳn**. Nếu bê nguyên giá trị ô của láng giềng vào, ta sẽ nhập một giá trị sai quy mô. Hệ số `f` đưa láng giềng về đúng quy mô của quan sát đang xét.

**Bước 4.** Điền bằng median, công thức (6):
```
x*_{ij} = median{ f_{i,i_1}·x_{i_1 j} , … , f_{i,i_k}·x_{i_k j} }
```

Dùng median (không dùng mean) để chống outlier ở phần thứ `j` của các láng giềng.

### 2.4 Chọn k

Không có công thức. Quy trình bài báo chỉ định:

1. Lấy các ô **đang quan sát được**, che ngẫu nhiên thành missing.
2. Điền lại với nhiều giá trị k.
3. Đo sai số so với giá trị gốc (dùng thước đo ở §4).
4. Chọn k cho sai số nhỏ nhất.

Giá trị bài báo dùng: `k = 4` cho ví dụ dữ liệu chi tiêu (D=5, n=20); `k = 8` cho các mô phỏng. Bài báo ghi nhận: với kNN, các lựa chọn k khác có ảnh hưởng khá nhỏ tới kết quả.

### 2.5 Ràng buộc & cảnh báo

- **Ổn định số học:** không lặp, không có rủi ro phân kỳ. Đây là điểm mạnh chính.
- **Không robust về nguyên tắc:** một ô outlier làm lệch khoảng cách. Giảm nhẹ bằng cách tăng k và dùng (7).
- **Mẫu nhỏ là rủi ro thật:** khi tìm láng giềng trên subcomposition, `d_A` có thể chọn phải láng giềng gần nhưng mang thông tin tệ hơn điểm xa. Bài báo lưu ý điều này cũng xảy ra với khoảng cách Euclid trên dữ liệu thường.
- **Không khai thác đầy đủ quan hệ đa biến** — chỉ gián tiếp qua việc chọn láng giềng. Đây chính là động cơ để có Hướng 2.
- Các biến thể kNN khác (đổi 2a sang 1a/1b/2b, đổi median sang mean khi gộp láng giềng) đã được thử và cho kết quả **kém hơn**. Không cần thử lại.

---

## 3. HƯỚNG 2 — Điền khuyết lặp dựa trên mô hình (ilr + LS/LTS)

### 3.1 Ý tưởng

Lặp qua từng biến: coi nó là **biến phụ thuộc**, các biến còn lại là **biến giải thích**, hồi quy để ước lượng lại các ô khuyết. Vì dữ liệu là compositional, hồi quy **phải** thực hiện trong không gian ilr.

Bài toán con: để dựng được balance theo (8) cần ma trận **đầy đủ** — nên phải khởi tạo trước. Đây là chỗ Hướng 1 được nhúng vào.

### 3.2 Thuật toán (9 bước, nguyên văn cấu trúc bài báo)

```
Bước 1: Khởi tạo mọi ô khuyết bằng kNN-Aitchison (Hướng 1).

Bước 2: Sắp xếp các biến theo SỐ LƯỢNG Ô KHUYẾT GIẢM DẦN:
        M(x₁) ≥ M(x₂) ≥ … ≥ M(x_D)

Bước 3: l = 1

Bước 4: Biến đổi ilr toàn bộ ma trận theo công thức (8).

Bước 5: Đặt
          m_l = chỉ số các quan sát VỐN khuyết ở biến x_l  (ghi nhớ từ đầu, KHÔNG cập nhật)
          o_l = phần bù của m_l
          z_l^{o_l}, z_l^{m_l}  : balance thứ l tại phần quan sát / phần khuyết
          Z_{−l}^{o_l}, Z_{−l}^{m_l} : các balance còn lại, có thêm cột 1 làm intercept
        Mô hình (12):   z_l^{o_l} = Z_{−l}^{o_l} · β + ε

Bước 6: Ước lượng β (LS hoặc LTS), rồi (13):
          ẑ_l^{m_l} = Z_{−l}^{m_l} · β̂

Bước 7: Biến đổi ngược về simplex bằng (9)–(11).
        → các ô vốn khuyết ở x_l được cập nhật.
        → các ô KHÔNG khuyết cũng bị đổi giá trị, NHƯNG tỷ lệ giữa chúng không đổi.
           Đây là hành vi ĐÚNG, không phải bug.

Bước 8: Lặp bước 4–7 cho l = 2,…,D.

Bước 9: Lặp lại bước 3–8 cho tới khi
          || S_iter − S_{iter−1} ||  <  ngưỡng
        với S là ma trận hiệp phương sai mẫu của dữ liệu đã ilr theo (8).
```

### 3.3 Chọn LS hay LTS

| | LS | LTS |
|---|---|---|
| Khi dữ liệu sạch | tốt nhất (setup 2) | ngang ngửa |
| Khi có outlier | **hỏng nặng** | ổn định tới ~30–35% outlier |
| Chi phí | rẻ | cao hơn nhưng vẫn nhanh (thuật toán FAST-LTS) |
| Bảo vệ khởi tạo tồi | không | **có** |

Khuyến nghị bàn giao: **mặc định LTS**, bật LS như một tùy chọn khi đã xác nhận dữ liệu sạch.

Lý do bổ sung của bài báo cho LTS: hồi quy robust còn bảo vệ chống **khởi tạo tồi**, bởi biến đổi ilr (8) có thể làm một ô xấu lây nhiễm sang toàn bộ các ô khác trong cùng quan sát.

### 3.4 Hội tụ — cảnh báo phải ghi vào docstring

**Bài báo không có chứng minh hội tụ.** Ghi nhận thực nghiệm: thường hội tụ trong vài vòng lặp, và **sau vòng lặp thứ hai không còn cải thiện đáng kể**.

Hệ quả cho cài đặt:
- Đặt `max_iter` (ví dụ 10) và **luôn** trả về trạng thái hội tụ (`converged: bool`, `n_iter: int`).
- Không được khẳng định "thuật toán hội tụ" trong tài liệu sản phẩm.
- Nếu `n_iter > 3` mà tiêu chí chưa giảm → cờ cảnh báo, nghi ngờ dữ liệu/outlier.

---

## 4. Thước đo đánh giá — **bắt buộc dùng đúng 2 thước đo này**

Gọi `M ⊂ {1,…,n}` là tập quan sát có ít nhất một ô khuyết, `n_M = |M|`.

**(a) Compositional error variance — công thức (14):**
```
(1/n_M) · Σ_{i ∈ M} d_A²( x_i , x̂_i )
```
`x_i` = composition gốc trước khi che; `x̂_i` = composition chỉ với các ô khuyết được điền.
*Đo: độ gần của giá trị điền khuyết trong hình học Aitchison.*

**(b) Difference in covariance structure — công thức (15):**
```
(1/(D−1)) · || S − S̃ ||
```
`S` = hiệp phương sai mẫu của dữ liệu ilr **không outlier** gốc; `S̃` = hiệp phương sai của cùng các quan sát đó sau khi điền tất cả ô khuyết.
*Đo: điền khuyết có làm méo cấu trúc đa biến hay không.*

**CẤM:** dùng RMSE/MAE Euclid trên dữ liệu thô làm tiêu chí chính. Bài báo nói rõ các thước đo dựa trên hình học Euclid là **không phù hợp** trong bối cảnh này. Nếu cần báo cáo RMSE cho người đọc quen thuộc, phải kèm cảnh báo và không dùng nó để ra quyết định chọn mô hình.

---

## 5. Kết quả kỳ vọng (dùng làm mốc nghiệm thu)

### 5.1 Ví dụ dữ liệu thật — chi tiêu hộ gia đình

Dữ liệu: Aitchison (1986), tr. 395 — chi tiêu của 20 người đàn ông độc thân trên 5 nhóm hàng (nhà ở, thực phẩm, rượu–thuốc lá, hàng hóa khác, dịch vụ). Ô bị che: `[1,3]`. **Giá trị thật = 147 HK$.**

Hai kịch bản outlier được tạo trên quan sát thứ 3:
- **outlier 1:** nhân **cột thứ 3** với hệ số 1 / 2 / 10 → outlier trong **cả** hình học Euclid và Aitchison.
- **outlier 2:** nhân **toàn bộ hàng 3** với hệ số 1 / 2 / 10 → tỷ lệ không đổi → outlier **chỉ trong** hình học Euclid.

| Phương pháp | gốc | out1 ×2 | out1 ×10 | out2 ×2 | out2 ×10 |
|---|---|---|---|---|---|
| gmean | 289.6 | 300.3 | 326.9 | 289.6 | 289.6 |
| alr-EM | 157.8 | 155.4 | 150.1 | 157.8 | 157.8 |
| **knn (Aitchison)** | **152.1** | 152.1 | 152.1 | 152.1 | 152.1 |
| **LS (ilr)** | **150.8** | 148.1 | 142.2 | 150.8 | 150.8 |
| **LTS (ilr)** | **150.8** | 150.3 | 150.3 | 150.8 | 150.8 |
| mean | 330.2 | 368.4 | 673.6 | 368.4 | 673.6 |
| EM | 190.2 | 214.9 | 798.4 | 163.5 | 195.6 |
| knn (Euclid) | 155.0 | 155.0 | 155.0 | 155.0 | 155.0 |
| LS (không biến đổi) | 161.0 | 179.2 | 324.5 | 161.3 | 160.3 |
| LTS (không biến đổi) | 161.3 | 158.6 | 158.6 | 161.4 | 153.6 |

**Cách đọc bảng này cho nghiệm thu:**

1. **Cột outlier 2 là bài test bất biến vàng.** Mọi phương pháp làm việc đúng trong hình học Aitchison (knn-Aitch, LS(ilr), LTS(ilr), alr-EM) cho **giá trị y hệt** ở cả 3 hệ số. Nếu cài đặt của bạn không tái tạo được tính bất biến này → **có bug**, dừng lại và sửa trước khi đi tiếp.
2. Điền khuyết đơn biến (mean 330.2, gmean 289.6) là tệ nhất — kể cả gmean, dù gmean đã tính đến hình học simplex. Bài học: đúng hình học **chưa đủ**, còn phải dùng thông tin đa biến.
3. `LS(ilr)` với out1 ×2 cho 148.1 — gần 147 nhất bảng. Bài báo nói thẳng đây là **ăn may**: siêu phẳng hồi quy đã bị outlier làm hỏng, và khi đẩy outlier xa hơn (×10) kết quả tụt xuống 142.2. **Không được lấy con số này làm bằng chứng LS tốt.**
4. EM chuẩn với out1 ×10 cho 798.4 — minh họa outlier phá hỏng hoàn toàn phương pháp không robust, không đúng hình học.

> Cảnh báo phạm vi: bài báo ghi rõ không thể rút ra kết luận tổng quát từ ví dụ đơn lẻ này. Nó chỉ dùng làm **test hồi quy (regression test)** cho cài đặt.

### 5.2 Mô phỏng — thiết kế và kết quả

Phân phối chuẩn trên simplex: `x ~ N^D_S(µ, Σ)` ⟺ `ilr(x)` tuân theo chuẩn đa biến trên ℝ^{D−1}. Tính chuẩn trên simplex độc lập với lựa chọn balance.

**Setup 1 (D=3, n=100):** `µ = (0,2)ᵀ`, `Σ = [[1.05,0.95],[0.95,1.05]]` (điều kiện xấu, gần như toàn bộ biến thiên theo một hướng). Mỗi quan sát nhân với hệ số từ `U(0,1)` — thao tác này **không đổi lớp tương đương**. Outlier group 1: `µ₁=(6,0)ᵀ`, nhân `U(0,10)` → outlier ở cả hai hình học. Outlier group 2: cùng `µ`, nhân `U(0,10)` → **không** là outlier trong hình học Aitchison. Mỗi nhóm từ 0→40%, bước 1%, 1000 bộ dữ liệu mỗi mức. Missing MCAR: 20% ở biến 1, 10% ở biến 2.

Kết quả:
- Bỏ qua bản chất compositional (làm trong không gian Euclid) → **nói chung tệ hơn**. Ngoại lệ: kNN ở thước đo cấu trúc hiệp phương sai cho kết quả tương tự dù dùng Euclid hay Aitchison.
- Làm đúng hình học → điền khuyết lặp dựa trên mô hình **cải thiện đáng kể** so với khởi tạo kNN.
- Không outlier: LS ≈ LTS.
- Có outlier: LTS thắng rõ rệt, **giữ ổn định tới khoảng 35% outlier**.
- kNN tốt nhất tại `k = 8`; các k khác ảnh hưởng nhỏ.

**Setup 2 (D=10, n=100, tương quan cao):** `µ = 0` (cân bằng trên simplex), đường chéo `Σ` = 1, ngoài đường chéo = **0.9**. Missing lần lượt 20%, 10%, 5%, 2%,…,2%, **0%** ở biến cuối. Chỉ outlier loại 1, `µ₁=(6,0,…,0)ᵀ`.
- Không nhiễm bẩn: **LS(ilr) tốt nhất**.
- Tới 30% outlier: **LTS(ilr) tốt nhất**.
- Outlier ảnh hưởng lớn tới mọi phương pháp còn lại, ở **cả hai** thước đo.

**Setup 3 (D=10, tương quan thấp — kịch bản xấu nhất):** ngoài đường chéo = **0.1**, missing 5% đều ở mọi biến trừ biến cuối (biến cuối phải sạch vì alr-EM yêu cầu).
- Mọi phương pháp **hành xử gần như nhau**.
- Có outlier: LTS(ilr) **hơi** tốt hơn.

**Cơ chế missing:** mô phỏng chính chạy MCAR; MAR cũng đã được thử và cho kết luận tương tự, vì kNN và phương pháp model-based xử lý được MAR. Ngược lại, điền khuyết đơn biến (mean/gmean) vốn đã tệ với MCAR thì **không** xử lý được MAR.

### 5.3 Các phương pháp đã bị loại — không cần thử lại

Bài báo đã kiểm tra và ghi nhận **kém hơn**: trung bình cộng, trung bình nhân, EM chuẩn, các thủ tục lặp dựa trên PCA và bản robust của chúng, Bayesian PCA, probabilistic PCA, cùng một số phương pháp điền khuyết có sẵn trong R. Chúng bị lược khỏi hình vẽ để tránh rối.

→ **Không đưa các phương pháp này vào backlog như "ý tưởng mới".**

---

## 6. Ràng buộc cứng (hard constraints)

| # | Ràng buộc | Vi phạm dẫn tới |
|---|---|---|
| R1 | Mọi phần phải **dương thực sự** | ln không xác định; ilr sập |
| R2 | **Zero ≠ missing.** Rounded zeros là bài toán khác | dùng sai thuật toán; kết quả vô nghĩa |
| R3 | Cơ chế khuyết phải là **MCAR hoặc MAR** | suy diễn không hợp lệ (Little & Rubin) |
| R4 | Dùng **ilr**, không dùng alr/clr cho hồi quy robust | clr suy biến → robust estimation hỏng; alr không đẳng cự |
| R5 | Không ép tổng hằng **trước** khi điền | mất thông tin quy mô cần cho hệ số hiệu chỉnh (5)/(7) |
| R6 | Đánh giá bằng (14) và (15), **không** bằng metric Euclid | chọn sai mô hình |
| R7 | Hướng 2 **phải** khởi tạo bằng Hướng 1 | không dựng được balance; lan truyền sai số |
| R8 | `m_l` (chỉ số vốn khuyết) phải cố định từ đầu, không cập nhật theo vòng lặp | thuật toán tự "học" giá trị nó vừa bịa ra |
| R9 | Nếu cần so sánh với alr-EM: **biến cuối cùng phải không có missing** | alr-EM không chạy được |
| R10 | Với LTS: `n` phải đủ lớn so với `D` | ước lượng robust không xác định |

Nếu cần tổng hằng ở đầu ra: chia mọi giá trị (cả quan sát lẫn điền khuyết) cho tổng rồi nhân hằng số mong muốn — làm **sau cùng**. Một lựa chọn thay thế trong tài liệu là chỉ chỉnh các giá trị không khuyết (Martín-Fernández et al. 2003).

---

## 7. Đặc tả API bàn giao

```python
def aitchison_distance(x, y) -> float:
    """Công thức (2). Bất biến với nhân vô hướng dương ở cả hai đối số."""

def ilr_transform(X) -> np.ndarray:
    """Công thức (8). X: (n, D) dương thực sự. Trả về (n, D-1)."""

def ilr_inverse(Z) -> np.ndarray:
    """Công thức (9)-(11). Trả về MỘT ĐẠI DIỆN của lớp tương đương,
       KHÔNG phải vector gốc. Không tự chuẩn hóa tổng."""

def knn_impute_aitchison(X, missing_mask, k, adjust="median") -> np.ndarray:
    """Hướng 1, chiến lược 2a.
       adjust: "median" -> công thức (7) [mặc định]; "sum" -> công thức (5).
       Gộp láng giềng bằng MEDIAN, công thức (6)."""

def iterative_impute_ilr(X, missing_mask, k_init=8, regressor="lts",
                         max_iter=10, tol=1e-6) -> ImputeResult:
    """Hướng 2, 9 bước.
       regressor: "lts" [mặc định] | "ls".
       Trả về: X_imputed, converged, n_iter, cov_distance_history."""
```

**Kiểu trả về bắt buộc có `converged` và `n_iter`** — vì không có bảo đảm hội tụ lý thuyết.

**Cài đặt tham chiếu để đối chiếu:** bài báo công bố cài đặt trong gói R **`robCompositions`** trên CRAN. Dùng nó làm nguồn đối chiếu số học (cross-check), không phải để copy.

---

## 8. Bộ kiểm thử nghiệm thu

| ID | Kiểm thử | Tiêu chí PASS |
|---|---|---|
| T1 | `d_A(c·x, y) == d_A(x, y)` với c ∈ {0.1, 1, 1000} | sai khác < 1e-12 |
| T2 | Đẳng cự: `d_A(x,y) == d_E(ilr(x), ilr(y))` | sai khác < 1e-10 |
| T3 | Khứ hồi: `ilr(ilr_inverse(z)) == z` | sai khác < 1e-10 |
| T4 | `ilr(c·x) == ilr(x)` | sai khác < 1e-12 |
| T5 | **Bất biến outlier-2:** nhân một hàng bất kỳ với 2 và 10 → giá trị điền khuyết **không đổi** (tối đa sai số số học) | tái tạo cột out2 của Bảng §5.1 |
| T6 | Tái tạo Bảng §5.1: knn(Aitch)=152.1, LS(ilr)=150.8, LTS(ilr)=150.8 trên dữ liệu gốc | khớp tới 1 chữ số thập phân |
| T7 | Dữ liệu không có ô khuyết → Hướng 2 không làm đổi **tỷ lệ** giữa các phần | `d_A(x_in, x_out) < 1e-10` mọi hàng |
| T8 | Bước 7: sau khi cập nhật `z_l`, tỷ lệ giữa các phần `j > l` được bảo toàn | sai khác < 1e-10 |
| T9 | Hội tụ: trên dữ liệu mô phỏng setup 2, `n_iter ≤ 5` | PASS hoặc raise cảnh báo |
| T10 | Mô phỏng setup 1: LTS ổn định tới ~35% outlier, LS thì không | tái lập xu hướng định tính |
| T11 | LS(ilr) với outlier-1 ×10 phải **xấu đi** (≈142.2, xa 147 hơn bản gốc) | tái tạo hành vi thất bại — đây là test **kỳ vọng thất bại** |

T5 và T11 là quan trọng nhất. T5 chứng minh hình học đúng. T11 chứng minh bạn không vô tình "sửa" một hành vi vốn phải xấu.

---

## 9. Sai lầm hay gặp (checklist review code)

1. **Quên hệ số hiệu chỉnh `f`** ở Hướng 1 → giá trị điền khuyết sai quy mô một cách hệ thống. Triệu chứng: sai số lớn ở các hàng có tổng khác biệt mạnh.
2. **Dùng mean thay median** ở (6) hoặc (7) → mất tính robust, bài báo đã đo và thấy kém hơn.
3. **Chuẩn hóa tổng hằng trước khi điền** → phá hỏng bước 3 của Hướng 1.
4. **Cập nhật `m_l` theo vòng lặp** → sau vòng 1 không còn ô nào "khuyết", thuật toán đứng yên hoặc tự củng cố giá trị bịa.
5. **Không sắp xếp biến theo số missing giảm dần** (bước 2) → mất toàn bộ lợi ích chống lan truyền sai số của cơ sở balance.
6. **Báo động giả ở bước 7**: thấy ô không-khuyết bị đổi giá trị rồi "sửa" bằng cách ghi đè giá trị cũ → phá vỡ tính nhất quán của lớp tương đương. Hành vi đúng là để nguyên.
7. **Dùng RMSE Euclid để chọn k hoặc chọn LS/LTS** → chọn sai.
8. **Tiêu chí dừng tính trên dữ liệu thô** thay vì trên hiệp phương sai của dữ liệu **đã ilr** → tiêu chí không có ý nghĩa.
9. **Giả định hội tụ** và bỏ `max_iter`.
10. Dùng **clr** vì thấy nó "dễ diễn giải hơn" → ma trận suy biến, LTS sập.

---

## 10. Hạn chế đã biết — phải ghi vào tài liệu sản phẩm

- **Balance (ilr) không diễn giải được** theo các phần gốc. Đây là hệ quả tất yếu của việc dữ liệu chỉ mang thông tin tương đối. Giảm nhẹ bằng sequential binary partition: tách các phần thành nhóm và dựng balance đại diện cho nhóm / quan hệ giữa nhóm.
- **Không có chứng minh hội tụ** cho Hướng 2.
- **Kết quả mô phỏng phụ thuộc cấu hình.** Bài báo trình bày 3 trong số nhiều cấu hình; setup 3 (tương quan thấp) là kịch bản xấu nhất và ở đó phương pháp đề xuất chỉ **ngang bằng**, không vượt trội.
- Phát biểu mạnh nhất mà bằng chứng cho phép: phương pháp lặp đề xuất có hiệu năng **tương đương** các phương pháp khác trong trường hợp xấu nhất, và **vượt trội** trong các tình huống ít cực đoan hơn. Không được phát biểu mạnh hơn mức này.
- Trong mô phỏng, ô khuyết **chỉ được sinh ở nhóm dữ liệu không-outlier**, để so sánh giữa phương pháp cổ điển và robust là công bằng. Nếu bạn đổi thiết kế này, các con số mốc ở §5 **không còn áp dụng**.

---

## Phụ lục A — Liên hệ với bài toán "scale as nuisance"

*(Ghi chú của người bàn giao, không phải nội dung bài báo. Đưa vào vì có khả năng liên quan trực tiếp tới công việc hiện tại của bạn.)*

Cấu trúc `x̲ = {c·x, c ∈ ℝ⁺}` ở §1.1 **chính xác** là cấu trúc của mô hình `y = c · f(a)` với `c ≥ 0` theo từng pixel: độ lớn toàn cục là tham số nhiễu, thông tin nằm ở hướng/tỷ lệ. Hai hệ quả có thể dùng lại:

1. **Bài test bất biến outlier-2** (§5.1) là mẫu kiểm thử trực tiếp áp dụng được: nhân một quan sát với `c > 0` thì đầu ra của mọi thành phần "đã khử scale" phải **không đổi**. Đây là một unit test rẻ và rất sắc để phát hiện rò rỉ scale trong pipeline.
2. **Aitchison geometry là một cách khử scale đã có nền lý thuyết đầy đủ** (không gian Hilbert, đẳng cự, cơ sở trực giao). Nếu công việc của bạn cần một baseline cổ điển cho việc "khử scale rồi suy luận", đây là ứng viên có sẵn, đã được kiểm chứng, và trích dẫn được.

Lưu ý khác biệt: bài báo giả định **dương thực sự** và thao tác trên log-ratio; abundance với ràng buộc `a ≥ 0, Σa = 1` cho phép **bằng 0**, nên ilr không áp dụng trực tiếp lên abundance có zero. Đừng bê nguyên.
