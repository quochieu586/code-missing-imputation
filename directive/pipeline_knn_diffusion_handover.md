# Hợp đồng bàn giao — Pipeline điền khuyết hai tầng
## Tầng A: kNN-Aitchison (khởi tạo) → Tầng B: vòng lặp điền khuyết với lõi diffusion (CSDI)

**Phiên bản:** v1.0 — đặc tả thiết kế, chưa có kết quả thực nghiệm của chúng ta.
**Thay thế / mở rộng:** `compositional_imputation_handover.md` (tài liệu Hron). Tài liệu này là bản **hợp nhất**; khi hai tài liệu mâu thuẫn, lấy tài liệu này.

**Ba nguồn đã xác minh:**

| Ký hiệu | Nguồn | Vai trò trong pipeline |
|---|---|---|
| **[H]** | Hron, Templ, Filzmoser (2009), *Imputation of missing values for compositional data using classical and robust methods* | Hình học Aitchison; Tầng A; khung vòng lặp của Tầng B; thước đo đánh giá |
| **[S]** | Seki, Zhang, Imoto (2025), *Diffusion model for imputing time-series gut microbiome profiles using phylogenetic information and metadata integration*, Bioinformatics Advances 5(1):vbaf181 | Lõi diffusion (CSDI + phylum CNN); tiền xử lý clr; baseline; mốc số liệu |
| **[D]** | Sơ đồ `Missing_imputation_Diffusion_idea_plot.pdf` (nội bộ, 09/09/2026) | Cấu trúc vòng lặp ngoài của Tầng B |

**Mọi con số trong tài liệu này đều trích từ [S]. Không có con số nào là của chúng ta. Không được trích dẫn chúng như kết quả của pipeline này.**

---

# PHẦN 0 — TỔNG QUAN

## 0.1 Bài toán cụ thể của chúng ta

Khác biệt **quyết định** so với cả hai bài báo nguồn:

| | [H] | [S] | **Chúng ta** |
|---|---|---|---|
| Đơn vị khuyết | ô lẻ trong bảng | **toàn bộ time point** (cả hàng `L`) | **ô lẻ, thưa theo điểm** |
| Cấu trúc thời gian | không có | có | có |
| Mask quan sát | ma trận nhị phân | vector theo time point | **ma trận nhị phân đầy đủ `(L,K)`** |

**Hệ quả kỹ thuật phải xử lý ngay từ đầu:**

1. Nội suy tuyến tính / LOCF theo trục thời gian **vẫn dùng được làm baseline**, nhưng chúng chỉ nhìn một chiều (thời gian) trong khi dữ liệu ta khuyết theo cả hai chiều (thời gian × loài). Chúng sẽ yếu — đây là kỳ vọng, không phải thành tích.
2. CSDI [S] về bản chất **đã hỗ trợ mask tùy ý** (nó là mô hình có điều kiện trên tập ô quan sát được). Nhưng chiến lược **chọn target tự giám sát** trong [S] chọn theo *time point*. Phải viết lại thành **chọn theo ô**. Đây là chỗ sửa code bắt buộc, không phải tùy chọn.
3. Kịch bản MNAR trong [S] (chọn time point khuyết theo alpha diversity cao) **không áp dụng trực tiếp**. Cần định nghĩa lại cơ chế MNAR ở mức ô — hoặc thừa nhận ta chỉ đánh giá được MCAR và ghi rõ đó là giới hạn.

## 0.2 Kiến trúc hai tầng

```
                    X₀  (thô, thưa theo ô, có zero, có NaN)
                     │
        ┌────────────▼────────────┐
        │  TIỀN XỬ LÝ  (§2)       │  lọc <0.01% → 0 ; pseudo-count ; mask M₀
        └────────────┬────────────┘
                     │  X₀⁺  (dương thực sự, vẫn còn NaN)
        ┌────────────▼────────────┐
        │  TẦNG A  (§3)           │  kNN-Aitchison, chiến lược 2a
        │  [H] Hướng 1            │  → ma trận ĐẦY ĐỦ sơ cấp
        └────────────┬────────────┘
                     │  X₁  (đầy đủ, dương)
        ┌────────────▼────────────┐
        │  CẦU NỐI  (§4)          │  clr transform
        └────────────┬────────────┘
                     │  Z₁
        ╔════════════▼════════════════════════════════════════╗
        ║  TẦNG B — VÒNG LẶP NGOÀI  (§5)   i = 1 … N          ║
        ║                                                     ║
        ║   Zᵢ ──split theo M₀──► x^co (điều kiện)            ║
        ║                      └─► x^ta (target) ──+noise──┐  ║
        ║                                                  │  ║
        ║   train ε_θ( x^ta_t , t | x^co )  ◄──────────────┘  ║
        ║   [S] CSDI + phylum CNN                             ║
        ║                                                     ║
        ║   sample → điền lại CHỈ các ô M₀=0 → Z_{i+1}        ║
        ║                                                     ║
        ║   kiểm tra hội tụ (§5.4) ──► lặp hoặc dừng          ║
        ╚════════════╤════════════════════════════════════════╝
                     │  Z_N
        ┌────────────▼────────────┐
        │  clr⁻¹ = closure∘exp    │
        └────────────┬────────────┘
                     │
                    X̂  (đầy đủ)
```

## 0.3 Ánh xạ: vòng lặp của [H] → vòng lặp của chúng ta

Đây là bản chất của thiết kế — Tầng B giữ **bộ khung 9 bước của [H] Hướng 2**, thay **bước 6 (hồi quy LS/LTS)** bằng **diffusion model**.

| Bước [H] | Nội dung gốc [H] | Phiên bản của chúng ta | Ghi chú |
|---|---|---|---|
| 1 | khởi tạo bằng kNN-Aitchison | **giữ nguyên** (Tầng A) | |
| 2 | sắp biến theo số missing giảm dần | **BỎ** | không còn vòng lặp theo biến |
| 3–4 | `l=1`; biến đổi ilr | thay bằng clr một lần (§4) | xem §1.3 — quyết định quan trọng |
| 5 | dựng hồi quy `z_l ~ Z_{−l}` | thay bằng split `x^co / x^ta` theo [S] | |
| **6** | **ước lượng β (LS/LTS), điền `ẑ`** | **train ε_θ và sample từ diffusion** | **đây là toàn bộ đóng góp mới** |
| 7 | biến đổi ngược | closure∘exp, làm ở cuối (§4.3) | |
| 8 | lặp `l = 2..D` | **BỎ** — diffusion điền tất cả ô cùng lúc | xem cảnh báo §5.5 |
| 9 | lặp tới khi hiệp phương sai ổn định | **giữ nguyên tinh thần**, đổi tiêu chí (§5.4) | |

> **Cảnh báo thiết kế — đọc kỹ.** Việc bỏ bước 2 và 8 đồng nghĩa ta **vứt bỏ cơ chế chống lan truyền sai số** của [H]. Trong [H], cơ sở balance được chọn có chủ đích để giá trị điền khuyết tồi ở biến `x₁` chỉ làm bẩn vế trái phương trình hồi quy, không làm bẩn biến giải thích. Diffusion điền toàn bộ ô cùng lúc nên không có tính chất đó. Ta **phải** thay thế bằng một cơ chế khác, đó là quy tắc **§5.3 (khóa mask M₀)**. Nếu bỏ quên §5.3, vòng lặp sẽ tự học lại chính giá trị nó bịa ra.

---

# PHẦN 1 — NỀN TẢNG BẮT BUỘC

## 1.1 Dữ liệu thành phần: chỉ có thông tin tương đối

Quan sát `x = (x₁,…,x_K)ᵀ` với mọi thành phần **dương thực sự**. Toàn bộ thông tin nằm trong **tỷ lệ**. Lớp tương đương:

```
x̲ = { c·x ,  c ∈ ℝ⁺ }
```

`x` và `c·x` là cùng một thông tin. Tổng các phần có thể khác nhau giữa các quan sát và **không được ảnh hưởng tới kết quả**. [H] §2

Hệ quả: `K` phần chỉ có `K−1` chiều tự do. Áp thẳng PCA/correlation lên dữ liệu thô cho kết quả sai lệch.

[S] xác nhận điều này áp dụng cho microbiome (viện dẫn Gloor et al. 2017: dữ liệu microbiome là compositional, và đó không phải là tùy chọn) — đây là lý do [S] dùng clr trước khi train.

## 1.2 Khoảng cách Aitchison — và cách tính nhanh

Định nghĩa [H] công thức (2):

```
d_A(x,y) = sqrt( (1/K) · Σ_{i<j} ( ln(x_i/x_j) − ln(y_i/y_j) )² )
```

**Đẳng thức triển khai (phải dùng bản này):**

```
d_A(x, y) = || clr(x) − clr(y) ||₂
```

Đây là cùng một đại lượng, nhưng chi phí `O(K)` thay vì `O(K²)`. Với `K = 833` (WGS trong [S]) thì khác biệt là 833 phép tính thay vì ~347.000 — không phải tối ưu vặt, mà là điều kiện để Tầng A chạy được.

*Kiểm chứng bằng tay, K=2:* `clr(x) = (½ln(x₁/x₂), ½ln(x₂/x₁))`, nên `‖clr x − clr y‖² = ½(ln(x₁/x₂)−ln(y₁/y₂))²`, đúng bằng công thức (2) với `K=2`. ✓ (Đưa vào test T2.)

## 1.3 Quyết định phép biến đổi: **clr, không phải ilr**

[H] khuyến nghị ilr và **phản đối clr** vì clr sinh dữ liệu suy biến, làm hỏng ước lượng robust. [S] dùng clr. Chúng ta **chọn clr**. Lý do:

1. **Lý do quyết định — phylum CNN cần trục đặc trưng khớp với OTU.** [S] thêm các lớp CNN riêng cho từng phylum, rồi ghép các vector đặc trưng lại. Điều này đòi hỏi **một tọa độ ↔ một OTU**. clr cho đúng `K` tọa độ, mỗi tọa độ ứng với một phần. ilr cho `K−1` **balance** — là các tổ hợp log-ratio, **không** ánh xạ 1-1 sang loài. Dùng ilr thì toàn bộ ý tưởng phylogenetic CNN sụp đổ.
2. **Lý do [H] từ chối clr không áp dụng ở đây.** [H] tránh clr vì cần hồi quy robust (LTS) trên ma trận không suy biến. Lõi của ta là mạng nơ-ron, không ước lượng hiệp phương sai robust. Ràng buộc đó biến mất.
3. **Tính đẳng cự vẫn giữ nguyên:** `d_A(x,y) = ‖clr(x)−clr(y)‖ = ‖ilr(x)−ilr(y)‖`. Ta không mất gì về hình học.
4. **Tương thích ngược với [S]:** MAE báo cáo trong [S] tính trên giá trị clr. Muốn so sánh với Bảng 1/Bảng 2 của [S] thì phải cùng không gian.

**Định nghĩa dùng trong dự án:**

```
clr(x)_k = ln( x_k / g(x) )   với  g(x) = (Π_{j=1}^{K} x_j)^{1/K}
clr⁻¹(z) = closure( exp(z) ) = softmax(z)
```

**Ba tính chất phải khắc vào docstring:**

- `clr(c·x) = clr(x)` — bất biến scale. Đây là lý do tồn tại của cả pipeline.
- `Σ_k clr(x)_k = 0` — clr nằm trên siêu phẳng tổng-bằng-0, hạng `K−1`, **suy biến**.
- `softmax(z + c·1) = softmax(z)` — hướng all-ones là **không định danh được (unidentifiable)**.

> **Hệ quả §1.3 phải xử lý trong code (§5.2, quy tắc 4).** Vì hướng all-ones không ảnh hưởng tới composition, mô hình được tự do đi lang thang theo hướng đó. Gradient tiêu tốn vô ích và MAE bị thổi phồng giả tạo. **Bắt buộc trừ trung bình theo hàng của mọi đầu ra clr trước khi tính loss và trước khi tính MAE.** Target clr vốn đã có tổng bằng 0 nên không cần chỉnh.

---

# PHẦN 2 — HỢP ĐỒNG DỮ LIỆU & TIỀN XỬ LÝ

## 2.1 Phân biệt ba loại "không có giá trị" — làm sai ở đây thì mọi thứ sau đều vô nghĩa

| Loại | Ý nghĩa | Xử lý | Có nằm trong mask M₀? |
|---|---|---|---|
| **Missing** | không đo được / mẫu không lấy | **đây là thứ ta điền khuyết** | `M₀ = 0` |
| **Rounded zero** | có mặt nhưng dưới ngưỡng phát hiện | pseudo-count | `M₀ = 1` |
| **Structural zero** | thực sự vắng mặt | pseudo-count (đành chấp nhận) | `M₀ = 1` |

[H] §R2 nói rõ: rounded zeros là một bài toán khác, có phương pháp riêng. Chúng ta **không** giải bài toán đó. Ta dùng pseudo-count theo [S] và **ghi nhận đây là một giới hạn đã biết**, không phải giải pháp.

## 2.2 Quy trình tiền xử lý (bám sát [S] §2.3)

```
Bước P1. Giá trị relative abundance < 0.01%  →  đặt 0.
         (loại false positive; theo [S])
Bước P2. Ghi lại mask quan sát M₀ ∈ {0,1}^(N,L,K).
         M₀ = 0 CHỈ tại ô thật sự missing. Zero KHÔNG phải missing.
Bước P3. pseudo-count = (min abundance khác 0) / 2, cộng vào mọi ô bằng 0.
         (theo [S])
Bước P4. Đóng (closure) về tổng 1 nếu cần — KHÔNG bắt buộc, vì clr bất biến scale.
Bước P5. Kiểm tra: mọi ô quan sát được đều > 0. Assert.
```

**Ràng buộc thứ tự:** P3 **phải** trước Tầng A. kNN-Aitchison không chạy được với giá trị 0 (log không xác định).

**Cảnh báo bệnh lý pseudo-count — bắt buộc báo cáo:** sau clr, các ô pseudo-count có giá trị âm rất lớn và sẽ **chi phối MAE**. [S] tự ghi nhận mô hình gặp khó khi dự đoán zero abundance trên dữ liệu WGS, và quy nguyên nhân cho chiều cao của dữ liệu. Do đó **luôn báo cáo MAE theo ba nhóm**: toàn bộ ô / chỉ ô vốn khác 0 / chỉ ô pseudo-count. Một cải thiện MAE tổng thể mà chỉ đến từ nhóm pseudo-count **không phải là cải thiện sinh học**.

## 2.3 Sắp xếp đặc trưng và chống rò rỉ

[S] sắp các OTU sao cho đặc trưng tương quan nằm cạnh nhau: chia OTU theo phylum, tính Spearman `ρ` giữa mọi cặp OTU trong cùng phylum, rồi tính trung bình nhân, công thức (6):

```
ρ_OTU_j = ( |ρ_{OTU_j1} · ρ_{OTU_j2} · … · ρ_{OTU_jp}| )^(1/p) ,   1 ≤ j ≤ p
```

Sắp giảm dần **trong từng phylum**.

> **[S] nói rõ: việc sắp xếp chỉ được thực hiện trên tập train, để bảo đảm tập test không bị nhìn thấy.** Đây là ràng buộc chống rò rỉ, không phải chi tiết kỹ thuật. Đưa vào test T12.

## 2.4 Chia dữ liệu

- Chia **theo subject**, không theo ô, không theo time point. ([S] dùng 5-fold CV chia ngẫu nhiên subject thành 5 nhóm.)
- Mọi thống kê tiền xử lý (thứ tự OTU, pseudo-count, chuẩn hóa) tính **chỉ trên fold train**.
- Tầng A (kNN) cũng **chỉ được tìm láng giềng trong tập train**. Đây là điểm cực dễ sai: kNN có bản chất transductive, rất dễ vô tình để pixel/subject test làm láng giềng.

---

# PHẦN 3 — TẦNG A: kNN-AITCHISON (khởi tạo)

Nguồn: [H] §3.1. Mục tiêu: từ `X₀⁺` (còn NaN) tạo ma trận **đầy đủ sơ cấp `X₁`**.

## 3.1 Ký hiệu

- `x_i`: một quan sát (ở đây = một cặp *(subject, time point)* đã được làm phẳng, độ dài `K`)
- `M_i ⊂ {1,…,K}`: tập chỉ số ô **missing**
- `O_i = {1,…,K} \ M_i`: tập chỉ số ô **quan sát được**

## 3.2 Chiến lược: 2a (điền tuần tự từng ô)

[H] liệt kê 4 biến thể; chọn **2a**: điền tuần tự từng ô; láng giềng phải có giá trị tại vị trí `j` đang điền **và** tại toàn bộ `O_i`. Lý do của [H]: nhìn chung sẽ có nhiều láng giềng hợp lệ hơn, và yêu cầu nhiều thông tin hơn trên mỗi quan sát cho kết quả đáng tin hơn. Ở nhóm (2), tập `k` láng giềng **thay đổi** theo từng ô.

Các biến thể còn lại (1a, 1b, 2b) và việc gộp láng giềng bằng mean thay vì median đã được [H] thử và cho kết quả **kém hơn**. Không thử lại.

## 3.3 Thuật toán

**A1 — Lọc ứng viên.** Mọi quan sát trong **tập train** có giá trị không khuyết tại `j` **và** tại toàn bộ `O_i`.

**A2 — Tìm `k` láng giềng gần nhất.**

```
d = ‖ clr_sub(x_i | O_i)  −  clr_sub(x_cand | O_i) ‖₂
```

trong đó `clr_sub` là clr tính trên **subcomposition** giới hạn ở `O_i`.

> Hai điểm dễ sai:
> - clr đòi hỏi hàng đầy đủ → **không thể** tính clr trên hàng còn NaN. Bắt buộc giới hạn về subcomposition `O_i` trước. Trung bình nhân `g` phải tính trên `O_i`, không phải trên `K`.
> - **Không** cần hiệu chỉnh scale ở bước này. `d_A` bất biến với nhân vô hướng dương — đó chính là lý do chọn Aitchison.

**A3 — Hệ số hiệu chỉnh.** Bản robust, [H] công thức (7) — **dùng bản này**:

```
f*_{i,i_l} = ( median_{o ∈ O_i} x_{io} ) / ( median_{o ∈ O_i} x_{i_l,o} )
```

Bản gốc dùng tổng, công thức (5), giữ làm tùy chọn.

*Ý nghĩa:* láng giềng giống về **tỷ lệ** nhưng có thể ở **quy mô tổng thể khác hẳn**. Bê nguyên giá trị của láng giềng vào sẽ nhập một giá trị sai quy mô. Hệ số `f` kéo láng giềng về đúng quy mô của quan sát đang xét.

**A4 — Điền bằng median,** [H] công thức (6):

```
x*_{ij} = median{ f_{i,i_1}·x_{i_1,j} , … , f_{i,i_k}·x_{i_k,j} }
```

Median (không phải mean) để chống outlier ở phần thứ `j` của các láng giềng.

## 3.4 Chọn `k`

Không có công thức. Quy trình [H]:

1. Lấy các ô **đang quan sát được** trong tập train, che ngẫu nhiên thành missing.
2. Điền lại với nhiều giá trị `k`.
3. Đo sai số bằng compositional error variance (§6.1).
4. Chọn `k` nhỏ nhất về sai số.

Giá trị tham chiếu trong [H]: `k=4` (bộ chi tiêu, K=5, n=20), `k=8` (mô phỏng). [H] ghi nhận các lựa chọn `k` khác có ảnh hưởng khá nhỏ. **Khởi điểm đề nghị: k=8.** Chọn `k` **chỉ trên tập train**.

## 3.5 Tính chất và giới hạn

- **Ổn định số học** — không lặp, không rủi ro phân kỳ. Đây là lý do nó đóng vai trò khởi tạo.
- **Không robust về nguyên tắc** — một ô outlier làm lệch khoảng cách. Giảm nhẹ bằng `k` lớn hơn và công thức (7).
- **Rủi ro mẫu nhỏ có thật:** khi tìm láng giềng trên subcomposition, `d_A` có thể chọn phải láng giềng gần nhưng mang thông tin tệ hơn điểm xa. [H] lưu ý điều này cũng xảy ra với khoảng cách Euclid trên dữ liệu thường.
- **Không khai thác đầy đủ quan hệ đa biến** — chỉ gián tiếp qua việc chọn láng giềng. Chính đây là động cơ tồn tại của Tầng B.
- **Tầng A không dùng thông tin thời gian.** Nó xử lý mỗi *(subject, time point)* như một quan sát độc lập. Đây là lựa chọn có chủ ý: để Tầng B đảm nhận trục thời gian. Ghi vào tài liệu để không ai "sửa" nó thành kNN có trọng số thời gian mà không đo lại.

---

# PHẦN 4 — CẦU NỐI A → B

```
X₁  (N, L, K)  đầy đủ, dương
 │
 ├─ 4.1  Z = clr(X₁) theo từng hàng (mỗi cặp subject×timepoint là một composition)
 │       → Z có tổng theo trục K bằng 0
 │
 ├─ 4.2  Giữ nguyên M₀. Tầng B KHÔNG được sinh mask mới từ Z.
 │
 └─ 4.3  Back-transform chỉ làm MỘT LẦN ở cuối Tầng B:
         X̂ = softmax(Z_N) theo trục K
         (không cần chiếu về siêu phẳng tổng-0 trước; softmax bất biến với z → z + c·1)
```

**Định nghĩa "một composition" — phải chốt và không đổi:** một hàng clr = một cặp *(subject, time point)*, chuẩn hóa trên `K` loài. Đây là cách [S] làm (relative abundance tại mỗi time point). Không được trộn lẫn với việc chuẩn hóa dọc trục thời gian.

---

# PHẦN 5 — TẦNG B: VÒNG LẶP ĐIỀN KHUYẾT VỚI LÕI DIFFUSION

## 5.1 Lõi: CSDI có điều kiện ([S] §2.1)

Quá trình thuận phá hủy dữ liệu bằng cách thêm nhiễu dần cho tới khi hoàn toàn mất cấu trúc; quá trình ngược tái tạo dần bằng cách khử nhiễu. Quá trình ngược là chuỗi Markov với chuyển tiếp Gauss, khởi từ `p(x_T) = N(x_T; 0, I)`:

```
p_θ(x_{0:T})     := p(x_T) · Π_{t=1}^{T} p_θ(x_{t−1} | x_t)                    (1)
p_θ(x_{t−1}|x_t) := N( x_{t−1} ; µ_θ(x_t,t), σ_θ(x_t,t)·I )                    (2)
```

Bản có điều kiện: dữ liệu tách thành **conditional observations `x^co`** và **imputation targets `x^ta`**:

```
p_θ(x^ta_{0:T} | x^co_0)          := p(x^ta_T) · Π_t p_θ(x^ta_{t−1} | x^ta_t, x^co_0)   (3)
p_θ(x^ta_{t−1} | x^ta_t, x^co_0)  := N( x^ta_{t−1} ; µ_θ(x^ta_t, t | x^co_0),
                                                     σ_θ(x^ta_t, t | x^co_0)·I )        (4)
```

Hàm mất mát:

```
min_θ  L(θ) = min_θ  E_{x₀, ε, t} ‖ ε − ε_θ( x^ta_t , t | x^co_0 ) ‖²₂                   (5)
```

`ε_θ` là hàm khử nhiễu học được, `ε ~ N(0,I)`. [S] bổ sung transformer encoder một lớp cho **mỗi** chiều (thời gian và đặc trưng) để nắm phụ thuộc trong từng chiều.

**Phần phylogenetic ([S] §2.2):** thêm các lớp CNN **riêng cho từng phylum** trong hàm khử nhiễu; vector đặc trưng của các phylum được ghép lại; sau CNN là hàm kích hoạt rectifier.

## 5.2 Năm quy tắc của vòng lặp ngoài — **phần quan trọng nhất tài liệu này**

Sơ đồ [D] mô tả vòng lặp: `X₀ → khởi tạo → X_i → Z_i → (train obs / train target) → noisy target → train diffusion → impute → Z_{i+1} → X_{i+1} → quay lại`.

Vòng lặp này **không có trong [S]**. [S] chạy một lượt. Vòng lặp là đóng góp của chúng ta, và cũng là **rủi ro lớn nhất** của chúng ta. Năm quy tắc sau là điều kiện để nó không tự hủy:

**Quy tắc 1 — `M₀` bị đóng băng vĩnh viễn.**
`M₀` tính một lần từ dữ liệu thô, **không bao giờ** cập nhật theo vòng lặp. Sau vòng 1 mọi ô đều có giá trị; nếu tính lại mask thì sẽ không còn ô nào "missing" và vòng lặp đứng yên — hoặc tệ hơn, tự củng cố giá trị nó bịa ra. Đây là bản sao của ràng buộc [H] R8.

**Quy tắc 2 — Loss CHỈ tính trên ô có `M₀ = 1`.**
Đây là quy tắc sống còn. Ở vòng `i > 1`, `Z_i` chứa cả giá trị quan sát thật lẫn giá trị mô hình tự sinh ở vòng trước. Nếu để loss chạy trên ô tự sinh, mô hình học cách **tái tạo phỏng đoán của chính nó** — độ tự tin tăng, sai số thật không giảm, và mọi metric nội bộ đều trông đẹp. Đây là dạng thất bại thầm lặng nguy hiểm nhất của toàn bộ thiết kế.

```
x^ta  ⊂  { ô có M₀ = 1 }        ← target tự giám sát: chỉ lấy từ ô THẬT
x^co  =  phần còn lại (gồm cả ô đã điền ở vòng trước)   ← dùng làm điều kiện thì được
```

Giá trị tự sinh được phép làm **điều kiện** (đó chính là giá trị của vòng lặp), nhưng **tuyệt đối không** được làm **mục tiêu**.

**Quy tắc 3 — Chọn target theo Ô, không theo time point.**
[S] chọn target theo time point vì bài toán của họ khuyết cả hàng. Ta khuyết theo ô. Tỷ lệ target `r` áp dụng trên tập ô có `M₀=1`, lấy mẫu độc lập theo ô. (Giữ thêm một chế độ chọn theo time point để tái lập kết quả [S] khi cần đối chứng.)

**Quy tắc 4 — Trung tâm hóa clr trước khi tính loss và MAE.**
Xem §1.3. Trừ trung bình theo trục `K` của mọi đầu ra. Bỏ qua bước này thì mô hình lãng phí sức học một hướng không ảnh hưởng tới kết quả, và MAE bị thổi phồng.

**Quy tắc 5 — Không bao giờ ghi đè ô quan sát được.**
Sau khi sample, chỉ thay giá trị tại ô `M₀ = 0`. Ô `M₀ = 1` giữ nguyên giá trị gốc qua **mọi** vòng lặp.

## 5.3 Giả thuyết cần kiểm chứng — không phải giả định

> **H1.** Ở tỷ lệ khuyết cao, tập điều kiện `x^co` quá thưa nên mô hình có ít thông tin để điều kiện lên. Điền dần sẽ làm `x^co` giàu hơn qua các vòng, dẫn tới điền khuyết tốt hơn một-lượt.

**H1 là giả thuyết, chưa có bằng chứng.** Nó phải được kiểm chứng bằng thí nghiệm E2 (§7), và phải có tiêu chí bác bỏ rõ ràng:

- **H1 được ủng hộ** nếu MAE trên ô held-out giảm đơn điệu theo `i` và chạm đáy ở `i > 1`, ở ít nhất 2 tỷ lệ khuyết khác nhau, qua toàn bộ 5 fold.
- **H1 bị bác bỏ** nếu đáy nằm ở `i = 1` (tức là vòng lặp không đem lại gì) hoặc MAE tăng theo `i` (drift).

Nếu H1 bị bác bỏ: **báo cáo đúng như vậy**, giữ pipeline một lượt, và đóng góp còn lại là "kNN-Aitchison làm khởi tạo + CSDI cho dữ liệu thưa theo ô". Đó vẫn là một đóng góp hợp lệ. Không được kéo dài vòng lặp tới khi tìm được một cấu hình cho số đẹp.

## 5.4 Tiêu chí hội tụ

[H] dừng khi khoảng cách Euclid giữa ma trận hiệp phương sai (tính trên dữ liệu đã biến đổi) của vòng hiện tại và vòng trước nhỏ hơn một ngưỡng. Ta giữ tinh thần đó và theo dõi **ba** đại lượng:

```
C1 (theo [H]):  ‖ Cov(Z_i) − Cov(Z_{i−1}) ‖_F / (K−1)      < tol
C2 (ổn định):   mean_{ô M₀=0}  |Z_i − Z_{i−1}|               < tol
C3 (dừng sớm):  MAE trên mask validation giữ riêng           tăng ⇒ dừng
```

**C3 là tiêu chí ra quyết định. C1 và C2 chỉ để chẩn đoán.** Lý do: C1/C2 đo *sự ổn định*, không đo *sự đúng*. Một mô hình drift có thể rất ổn định trong khi đang trôi đều về phía sai.

**Mask validation:** che thêm một tập ô có `M₀=1` (đề nghị 10%), không bao giờ dùng để train, chỉ để chấm điểm giữa các vòng. Đây là thứ duy nhất chống được drift.

> [H] ghi rõ: **không có chứng minh hội tụ**; quan sát thực nghiệm là thuật toán thường hội tụ trong vài vòng và **sau vòng thứ hai không còn cải thiện đáng kể**. Với lõi diffusion, ta thậm chí không có bằng chứng thực nghiệm đó. Bắt buộc: `max_iter` (đề nghị 5) và luôn trả về `converged`, `n_iter`. **Không được viết "thuật toán hội tụ" trong bất kỳ tài liệu nào.**

## 5.5 Metadata (tùy chọn, ưu tiên thấp)

[S] nhúng biến metadata phân loại bằng Feature Tokenizer, công thức (7):

```
Embedding = b^(cat) + e^T · W^(cat)  ∈ ℝ^d ,   d = 8 trong [S]
```

Hai chiến lược ghép, cho `n` biến metadata:
- **spatial**: ghép theo chiều đặc trưng → tensor `(C, L, K + n·d)`
- **channel-wise**: ghép theo chiều kênh → tensor `(C + n·d, L, K)`

Metadata **được dùng khi train, loại bỏ khi đánh giá**.

**Kết quả [S] và khuyến nghị:** [S] báo cáo cải thiện **đo được nhưng khiêm tốn**, rõ hơn khi tỷ lệ khuyết vượt 0.4; không có khác biệt đáng kể giữa hai hướng ghép, dù ghép channel-wise với metadata dị ứng có xu hướng ổn định hơn. [S] quy mức tăng hạn chế cho sức phân biệt vừa phải của metadata và ràng buộc của phương pháp nhúng.

→ **Đưa vào Giai đoạn 3, sau khi H1 đã ngã ngũ.** Thêm metadata trước khi biết vòng lặp có tác dụng hay không là tự tạo thêm một biến gây nhiễu.

---

# PHẦN 6 — GIAO THỨC ĐÁNH GIÁ

## 6.1 Bộ thước đo — dùng cả ba, không thay thế lẫn nhau

**(M1) MAE trong không gian clr** — để so sánh được với [S]. [S] công thức (8):

```
MAE = (1/K) · Σ_{i=1}^{K} | y_i − ŷ_i |
```

Báo cáo **tách ba nhóm** theo §2.2: toàn bộ / ô vốn khác 0 / ô pseudo-count.

**(M2) Compositional error variance** — [H] công thức (14):

```
(1/n_M) · Σ_{i ∈ M} d_A²( x_i , x̂_i )
```

`M` = tập quan sát có ít nhất một ô khuyết; `x_i` = composition gốc trước khi che; `x̂_i` = composition chỉ với ô khuyết được điền.

**(M3) Difference in covariance structure** — [H] công thức (15):

```
(1/(K−1)) · ‖ S − S̃ ‖
```

`S` = hiệp phương sai mẫu của dữ liệu đã biến đổi, gốc; `S̃` = của cùng các quan sát sau khi điền.

> **Vì sao cần cả M2/M3 chứ không chỉ M1?** M1 đo sai số từng ô. M2 đo sai số **trong đúng hình học**. M3 đo việc điền khuyết có **làm méo cấu trúc đa biến** hay không — một phương pháp có thể đạt MAE thấp bằng cách kéo mọi thứ về trung tâm, làm sụp phương sai. M1 không phát hiện được điều đó; M3 phát hiện được. Với vòng lặp tự-huấn-luyện, **co cụm phương sai chính là chế độ hỏng được dự đoán trước**, nên M3 là cảnh báo sớm quan trọng nhất.

## 6.2 Kiểm tra tính hiện thực sinh học ([S] §3.1)

[S] kiểm tra hồ sơ điền khuyết có phản ánh đặc trưng thật hay không, dùng hai chỉ số:

- **Alpha diversity (chỉ số Shannon).** Trong [S], alpha diversity tăng theo thời gian rồi ổn định khi trẻ phát triển và tích lũy đa dạng hệ vi sinh; hồ sơ do phương pháp của họ điền bám theo xu hướng đó, trong khi nội suy tuyến tính cho xu hướng yếu và dải rộng hơn.
- **Tỷ lệ zero abundance.** [S] cũng nắm được xu hướng giảm của tỷ lệ này.

Ta phải chạy **cùng hai kiểm tra đó ở mỗi vòng lặp `i`**, không chỉ ở cuối. Lý do: nếu drift xảy ra, alpha diversity sẽ trôi trước khi MAE trên ô held-out kịp xấu đi. [S] ghi nhận trên WGS phương pháp của họ có xu hướng **dự đoán đa dạng cao hơn** và gặp khó khi dự đoán zero abundance — đó chính xác là chữ ký của drift mà ta cần canh.

## 6.3 Nhiệm vụ downstream

[S] đánh giá bằng mô hình RNN hai chiều dự đoán nhị phân, chấm bằng ROC-AUC và PR-AUC (dùng PR-AUC vì dữ liệu mất cân bằng, nhiều mẫu âm).

**Bài học từ [S] phải nhớ:** trong [S], mô hình **standard CSDI đạt hiệu năng dự đoán cao nhất**, sát với bộ dữ liệu đầy đủ; bản CSDI+phylum CNN thắng nội suy tuyến tính và bộ chưa điền, đặc biệt ở Simulation #1 và #2. Ở Simulation #3 (94% mẫu âm), bản CNN kém, ngang với nội suy tuyến tính và bộ chưa điền. [S] giải thích đây có thể là đánh đổi do lớp CNN: khử nhiễu và ổn định hóa tốt trong điều kiện thông thường, nhưng có thể làm suy giảm các dao động tinh vi quan trọng trong kịch bản mất cân bằng nặng.

→ **MAE thấp hơn không tự động kéo theo downstream tốt hơn.** [S] nói thẳng trong phần Thảo luận: dù phương pháp của họ giảm MAE, lợi ích ở nhiệm vụ dự đoán downstream **bị hạn chế** so với standard CSDI. Nếu pipeline của ta giảm MAE nhưng downstream không cải thiện, đó là **kết quả đã được tiên liệu**, phải báo cáo, không được giấu.

---

# PHẦN 7 — MA TRẬN THÍ NGHIỆM

## 7.1 Hai thí nghiệm quyết định — làm trước, mọi thứ khác là trang trí

| ID | Câu hỏi | Thiết kế | Bác bỏ khi |
|---|---|---|---|
| **E1** | **Tầng A có đáng không?** | Cố định Tầng B một lượt. Đổi khởi tạo: `mean` / `linear` / `LOCF` / **kNN-Aitchison**. | kNN-Aitchison không tốt hơn khởi tạo rẻ nhất → **bỏ Tầng A** |
| **E2** | **Vòng lặp có đáng không? (H1)** | Cố định Tầng A. Chạy `i = 1…5`, ghi MAE held-out từng vòng. | đáy ở `i=1` → **bỏ vòng lặp**, giữ một lượt |

Chạy E1 và E2 **trước** mọi thí nghiệm khác. Nếu cả hai đều bác bỏ, ta không có pipeline — ta chỉ đang chạy lại [S] trên dữ liệu khác, và phải nói đúng như thế.

## 7.2 Bảng so sánh đầy đủ

| # | Phương pháp | Nguồn | Vai trò |
|---|---|---|---|
| 1 | Mean imputation | [S] | sàn |
| 2 | LOCF | [S] | sàn theo thời gian |
| 3 | Linear interpolation | [S] | sàn theo thời gian — **mạnh bất ngờ, xem §8.2** |
| 4 | kNN-Aitchison đơn thuần | [H] | Tầng A một mình |
| 5 | kNN + iterative LTS (ilr) | [H] Hướng 2 | **baseline cổ điển đầy đủ** — cần ilr, xem §1.3 |
| 6 | Standard CSDI | [S] | lõi không có phylo |
| 7 | CSDI + phylum CNN, một lượt | [S] | tái lập [S] |
| 8 | **kNN + CSDI+CNN, một lượt** | ta | = E1 |
| 9 | **kNN + CSDI+CNN, có vòng lặp** | ta | = E2, pipeline đầy đủ |
| 10 | #9 + metadata FT | [S] §2.4 | Giai đoạn 3 |

Mục #5 là baseline cổ điển duy nhất **thực sự cùng cấu trúc** với pipeline của ta (khởi tạo kNN + vòng lặp). Nó là đối chứng công bằng nhất và **phải** có. Nó cần ilr — đó là lý do duy nhất ilr còn tồn tại trong dự án.

## 7.3 Lưới điều kiện

- Tỷ lệ khuyết `r` ∈ {0.1, …, 0.9}, bước 0.1 (theo [S]).
- Cơ chế: **MCAR ở mức ô**. MNAR ở mức ô **cần định nghĩa mới** — nếu chưa định nghĩa được một cách bảo vệ được, chỉ báo cáo MCAR và ghi rõ giới hạn. Không bịa ra một cơ chế MNAR rồi gọi nó là kết quả.
- 5-fold CV chia theo subject; báo cáo **mean (SD)** qua các fold, đúng định dạng [S].
- Với nhiệm vụ downstream, [S] chạy 5× 5-fold CV với các lần chia train/test khác nhau để đánh giá độ biến thiên. Làm theo.

## 7.4 Ngân sách tính toán — phải ước lượng trước khi chạy

Vòng lặp nhân chi phí với `N`. Với `N=5`: **5× chi phí train + 5× chi phí sampling** so với [S]. Trước khi khởi động lưới đầy đủ, chạy một fold duy nhất ở `r=0.5` để đo thời gian thực, rồi mới lên lịch. [S] ghi nhận tài nguyên tính toán do Human Genome Center, Institute of Medical Science và Đại học Tokyo cung cấp — tức đây không phải khối lượng chạy trên laptop.

---

# PHẦN 8 — MỐC SỐ LIỆU THAM CHIẾU (đều là của [S], **không** phải của ta)

## 8.1 16S rRNA — DIABIMMUNE, MAE trung bình 5-fold, mean (SD)

**MCAR**

| r | CSDI+phylum CNN | Standard CSDI | Linear | LOCF | Mean |
|---|---|---|---|---|---|
| 0.1 | **0.225** (0.021) | 0.230 (0.024) | 0.268 (0.018) | 0.291 (0.020) | 0.338 (0.016) |
| 0.3 | **0.220** (0.012) | 0.224 (0.014) | 0.289 (0.019) | 0.319 (0.021) | 0.335 (0.015) |
| 0.5 | **0.230** (0.007) | 0.234 (0.012) | 0.315 (0.014) | 0.339 (0.012) | 0.346 (0.008) |
| 0.7 | 0.237 (0.006) | **0.235** (0.006) | 0.363 (0.021) | 0.373 (0.023) | 0.371 (0.019) |
| 0.9 | **0.240** (0.010) | 0.240 (0.008) | 0.364 (0.010) | 0.367 (0.012) | 0.366 (0.012) |

**MNAR**

| r | CSDI+phylum CNN | Standard CSDI | Linear | LOCF | Mean |
|---|---|---|---|---|---|
| 0.1 | **0.219** (0.019) | 0.220 (0.013) | 0.273 (0.018) | 0.318 (0.026) | 0.326 (0.011) |
| 0.3 | 0.212 (0.006) | **0.208** (0.011) | 0.336 (0.018) | 0.363 (0.018) | 0.362 (0.018) |
| 0.5 | **0.205** (0.003) | 0.206 (0.003) | 0.399 (0.021) | 0.416 (0.018) | 0.407 (0.014) |
| 0.7 | 0.224 (0.010) | **0.222** (0.009) | 0.442 (0.012) | 0.445 (0.012) | 0.439 (0.010) |
| 0.9 | **0.230** (0.005) | 0.234 (0.004) | 0.410 (0.020) | 0.410 (0.020) | 0.409 (0.019) |

**Cách đọc — ba điều quan trọng hơn cả con số:**

1. **Khoảng cách CSDI+CNN vs Standard CSDI rất nhỏ** (thường 0.000–0.006) và **đảo chiều tùy `r`**. SD thường lớn hơn khoảng cách đó. Phylum CNN **không** phải một thắng lợi rõ ràng.
2. **Điều thực sự tách biệt hai nhóm:** cả hai phương pháp CSDI giữ MAE ổn định khi `r` tăng, trong khi các phương pháp khác có xu hướng sai số tăng rõ rệt. Đó mới là phát hiện chắc chắn.
3. [S] tự nêu: do khác biệt về pipeline tiền xử lý (chọn time point, tiêu chí nhận mẫu theo mức đầy đủ metadata, lọc đặc trưng ít phong phú), điểm số ở Bảng 1 **khác** với các nghiên cứu trước. → **Số của ta cũng sẽ khác. Đừng so trực tiếp. Chỉ so xu hướng.**

## 8.2 WGS — BONUS, MAE trung bình 5-fold

| r | CSDI+phylum CNN | Standard CSDI | Linear | LOCF | Mean |
|---|---|---|---|---|---|
| 0.1 | 0.190 (0.008) | 0.194 (0.022) | **0.178** (0.016) | 0.187 (0.016) | 0.202 (0.019) |
| 0.2 | 0.189 (0.012) | 0.184 (0.017) | **0.180** (0.017) | 0.183 (0.018) | 0.202 (0.016) |
| 0.3 | 0.188 (0.010) | **0.170** (0.008) | 0.180 (0.013) | 0.188 (0.016) | 0.203 (0.011) |
| 0.5 | 0.191 (0.009) | 0.201 (0.034) | 0.197 (0.006) | 0.201 (0.010) | 0.211 (0.006) |
| 0.7 | **0.187** (0.008) | 0.190 (0.010) | 0.213 (0.013) | 0.215 (0.014) | 0.220 (0.013) |
| 0.9 | 0.195 (0.024) | **0.190** (0.011) | 0.229 (0.012) | 0.229 (0.012) | 0.230 (0.012) |

> **Kết quả âm phải giữ nguyên, không được làm mờ.** Ở `r = 0.1` và `0.2` trên WGS, **nội suy tuyến tính thắng cả hai phương pháp diffusion**. [S] ghi nhận điều này và đưa một cách giải thích khả dĩ: mô hình phức tạp có thể sinh độ chệch lớn hơn ở tỷ lệ khuyết thấp trên dữ liệu nhiều zero. Dữ liệu của ta thưa theo ô và nhiều zero → **rất có thể ta sẽ gặp lại hiện tượng này**. Nếu gặp, đó là tái lập một quan sát đã công bố, không phải lỗi pipeline.

## 8.3 Thông số dữ liệu trong [S] (để tham chiếu quy mô)

- **DIABIMMUNE (16S):** 116/222 subject, 5 mẫu tại 0, 6, 12, 18, 24 tháng tuổi; 113 OTU ở mức loài.
- **BONUS (WGS):** 207 trẻ mắc xơ nang; 7 time point (3, 4, 5, 6, 8, 10, 12 tháng); lọc còn 157 subject có ≥5 time point; **tỷ lệ khuyết 11%**, **833 đặc trưng** ở mức loài. Metadata **không công khai** → [S] chạy WGS không có metadata.

---

# PHẦN 9 — RÀNG BUỘC CỨNG

| # | Ràng buộc | Nguồn | Vi phạm dẫn tới |
|---|---|---|---|
| R1 | Mọi giá trị quan sát phải **dương thực sự** sau P3 | [H] | log không xác định; clr/kNN sập |
| R2 | **Zero ≠ missing**, ba loại phân biệt rạch ròi | [H]/[S] | dùng sai thuật toán, kết quả vô nghĩa |
| R3 | **`M₀` đóng băng vĩnh viễn** | [H] R8 + §5.2 | vòng lặp tự học giá trị nó bịa |
| R4 | **Loss chỉ trên ô `M₀=1`** | §5.2 quy tắc 2 | **thất bại thầm lặng** — metric đẹp, kết quả sai |
| R5 | Không ghi đè ô quan sát được | §5.2 quy tắc 5 | mất dữ liệu thật |
| R6 | Trung tâm hóa clr trước loss/MAE | §1.3 | lãng phí gradient, MAE thổi phồng |
| R7 | Sắp xếp OTU **chỉ trên tập train** | [S] §2.2 | rò rỉ vào tập test |
| R8 | kNN **chỉ tìm láng giềng trong tập train** | §2.4 | rò rỉ (kNN vốn transductive) |
| R9 | Chia fold **theo subject** | [S] §2.5 | rò rỉ giữa các time point cùng subject |
| R10 | `max_iter` bắt buộc; luôn trả `converged`, `n_iter` | [H] §3.4 | vòng lặp không có bảo đảm dừng |
| R11 | Có **mask validation** riêng cho C3 | §5.4 | không phát hiện được drift |
| R12 | clr cho lõi diffusion; ilr **chỉ** cho baseline #5 | §1.3 | dùng ilr → phylum CNN mất ý nghĩa |
| R13 | Tầng A tính clr trên **subcomposition `O_i`** | §3.3 | NaN lan vào trung bình nhân |
| R14 | Chọn `k` chỉ trên tập train | §3.4 | rò rỉ |

---

# PHẦN 10 — ĐẶC TẢ API

```python
# ---------- Nền tảng ----------
def clr(X, axis=-1):
    """clr(x)_k = ln(x_k / geomean(x)). Yêu cầu X > 0 hoàn toàn.
       Bất biến scale: clr(c·X) == clr(X). Tổng theo axis == 0."""

def clr_inverse(Z, axis=-1):
    """softmax(Z). Bất biến với Z -> Z + c·1, nên không cần chiếu trước."""

def clr_subcomposition(x, idx):
    """clr tính TRÊN subcomposition x[idx]. Dùng cho Tầng A khi hàng còn NaN.
       Trung bình nhân lấy trên idx, KHÔNG trên toàn bộ K."""

def aitchison_distance(x, y, idx=None):
    """= ||clr_sub(x,idx) - clr_sub(y,idx)||_2 . Tương đương công thức (2) của [H]
       nhưng O(K) thay vì O(K^2)."""

# ---------- Tiền xử lý ----------
def preprocess(X_raw, detect_thresh=1e-4):
    """P1-P5 theo §2.2.
       Trả về: X_pos, M0 (mask quan sát), pseudo_count, meta."""

# ---------- Tầng A ----------
def knn_aitchison_impute(X_pos, M0, k=8, adjust="median", train_idx=None):
    """[H] Hướng 1, chiến lược 2a.
       adjust: "median" -> ct (7) [mặc định] | "sum" -> ct (5)
       Gộp láng giềng bằng MEDIAN, ct (6).
       train_idx: BẮT BUỘC ở chế độ CV — chỉ tìm láng giềng trong đây (R8).
       Trả về: X1 đầy đủ."""

# ---------- Tầng B ----------
@dataclass
class LoopResult:
    X_hat: np.ndarray
    Z_history: list          # Z_i mỗi vòng, để chẩn đoán drift
    converged: bool
    n_iter: int
    c1_history: list         # ||ΔCov||
    c2_history: list         # thay đổi trung bình ở ô điền khuyết
    c3_history: list         # MAE trên mask validation  <- TIÊU CHÍ DỪNG
    alpha_div_history: list  # §6.2
    zero_prop_history: list  # §6.2

def iterative_diffusion_impute(
        X1, M0,
        model_cfg,              # CSDI + phylum CNN
        phylum_map,             # OTU -> phylum, cho các nhánh CNN
        target_ratio=0.5,
        target_mode="cell",     # "cell" [mặc định, §5.2 QT3] | "timepoint" (đối chứng [S])
        max_iter=5,
        val_mask_frac=0.10,
        tol=1e-4,
        metadata=None,          # None | FT config, §5.5
        seed=...,
) -> LoopResult:
    """Tầng B. Thực thi R3, R4, R5, R6 ở tầng khung — KHÔNG ủy quyền cho
       mã mô hình. Phải assert các bất biến này ở mỗi vòng lặp."""
```

**Bắt buộc:** `converged` và `n_iter` luôn có trong kiểu trả về — không có bảo đảm hội tụ lý thuyết ([H] §3.4). `*_history` bắt buộc — không có chúng thì không chẩn đoán được drift.

**Cài đặt tham chiếu để đối chiếu:**
- [H] công bố trong gói R **`robCompositions`** trên CRAN.
- [S] công bố mã và dữ liệu tại `https://github.com/misatoseki/metag_time_impute_phylo.git`.

Dùng chúng để **đối chiếu số học**, không phải để sao chép. Ưu tiên cao: tái lập một ô trong Bảng 1 của [S] trước khi viết dòng code nào cho vòng lặp.

---

# PHẦN 11 — BỘ KIỂM THỬ NGHIỆM THU

## 11.1 Tầng nền tảng

| ID | Kiểm thử | PASS |
|---|---|---|
| T1 | `clr(c·x) == clr(x)`, c ∈ {0.1, 1, 1000} | < 1e-12 |
| T2 | `d_A` theo ct (2) của [H] == `‖clr x − clr y‖` | < 1e-10 |
| T3 | Khứ hồi: `clr(softmax(z)) == z − mean(z)` | < 1e-10 |
| T4 | `softmax(z + c·1) == softmax(z)` | < 1e-12 |
| T5 | `Σ_k clr(x)_k == 0` | < 1e-12 |
| T6 | `clr_subcomposition` không chạm tới ô NaN | không có NaN ở đầu ra |

## 11.2 Tầng A

| ID | Kiểm thử | PASS |
|---|---|---|
| T7 | **Bất biến scale đầu-cuối:** nhân một hàng bất kỳ với c>0 → giá trị điền khuyết thuộc cùng lớp tương đương | `d_A < 1e-10` |
| T8 | Tái lập [H]: bộ chi tiêu, ô [1,3], `k=4` → **152.1** | khớp 1 chữ số thập phân |
| T9 | Có `train_idx` → không subject test nào lọt vào tập láng giềng | assert cứng |
| T10 | Bỏ hệ số hiệu chỉnh `f` (test âm) → sai số **phải xấu đi** rõ rệt | test kỳ vọng thất bại |

> T8 dùng bảng tham chiếu của [H]: giá trị thật 147; knn(Aitchison) 152.1; LS(ilr) 150.8; LTS(ilr) 150.8. Cột outlier-2 (nhân cả hàng với 2 và 10) phải cho **giá trị y hệt** — đây là bài test bất biến vàng.

## 11.3 Tầng B — **quan trọng nhất**

| ID | Kiểm thử | PASS |
|---|---|---|
| **T11** | **Loss không bao giờ chạm ô `M₀=0`.** Đặt ô `M₀=0` thành NaN/±1e9 trước khi tính loss → loss **không đổi** | bit-identical |
| **T12** | **Ô quan sát được không bao giờ bị ghi đè.** So `X̂[M₀=1]` với `X₁[M₀=1]` sau mỗi vòng | bit-identical |
| T13 | `M₀` không đổi qua các vòng | bit-identical |
| T14 | Đầu ra clr có tổng theo `K` bằng 0 sau khi trung tâm hóa | < 1e-8 |
| T15 | Thứ tự OTU tính trên fold train không đổi khi hoán vị dữ liệu test | bit-identical |
| T16 | `max_iter` được tôn trọng; `converged=False` được báo đúng | assert |
| T17 | **Test drift:** chạy `max_iter=20` trên dữ liệu tổng hợp có ground truth; ghi C3 | C3 **không** tăng đơn điệu; nếu tăng → sự cố, dừng dự án cho tới khi giải thích được |

**T11 là bài test đắt giá nhất trong toàn bộ dự án.** Nó là cách duy nhất phát hiện dạng thất bại thầm lặng ở R4. Viết nó **trước** khi viết vòng lặp.

## 11.4 Đối chiếu

| ID | Kiểm thử | PASS |
|---|---|---|
| T18 | Tái lập một ô Bảng 1 của [S] trên dữ liệu công khai DIABIMMUNE | trong khoảng ±1 SD đã công bố |
| T19 | Baseline #5 (kNN + LTS ilr) chạy được và cho kết quả hợp lý | không lỗi, MAE hữu hạn |

---

# PHẦN 12 — CHẾ ĐỘ HỎNG & GIẢM THIỂU

| Chế độ hỏng | Triệu chứng | Phát hiện bằng | Giảm thiểu |
|---|---|---|---|
| **Tự-huấn-luyện lặp** | loss giảm đều, MAE held-out đứng yên | T11 + C3 | R4 (§5.2 QT2) |
| **Co cụm phương sai** | M1 tốt lên, M3 xấu đi, alpha diversity co lại | M3 + §6.2 mỗi vòng | dừng sớm theo C3 |
| **Drift** | mọi metric nội bộ đẹp, C3 tăng chậm | mask validation | R11, `max_iter` |
| **Pseudo-count chi phối** | MAE tổng cải thiện, MAE nhóm khác-0 không đổi | tách nhóm §2.2 | báo cáo tách nhóm |
| **Rò rỉ qua kNN** | kết quả tốt bất thường ở `r` cao | T9 | R8 |
| **Rò rỉ qua thứ tự OTU** | tương tự | T15 | R7 |
| **Lặp lại nghịch lý WGS `r` thấp** | linear thắng diffusion ở `r`∈{0.1,0.2} | so với §8.2 | **không phải bug** — [S] đã báo cáo; ghi nhận và giải thích |
| **MAE tốt, downstream không** | AUC ngang hoặc kém baseline | §6.3 | **không phải bug** — [S] đã báo cáo; ghi nhận |
| Thất bại Tầng A do mẫu nhỏ | láng giềng vô nghĩa ở subject hiếm | phân tích theo subject | tăng `k`; báo cáo theo subject |

**Hai dòng in đậm cần nhấn mạnh:** cả hai đều đã được [S] công bố. Nếu ta gặp lại, đó là **tái lập**, không phải thất bại. Nhưng nếu ta **không** báo cáo chúng thì mới là gian lận.

---

# PHẦN 13 — THỨ TỰ TRIỂN KHAI

| GĐ | Nội dung | Điều kiện hoàn thành | Rủi ro nếu bỏ qua |
|---|---|---|---|
| **0** | Nền tảng: clr, `d_A`, subcomposition | T1–T6 xanh | mọi thứ sau sai |
| **1** | Tiền xử lý + hợp đồng mask | T13, T15 xanh; ba loại zero/missing tách bạch | R2 hỏng toàn bộ |
| **2** | Tầng A | T7–T10 xanh; T8 tái lập [H] | khởi tạo tồi kéo cả Tầng B |
| **3** | Tái lập [S], một lượt | T18 trong ±1 SD | không có mốc → không biết mình sai ở đâu |
| **4** | **E1** — Tầng A có đáng không? | quyết định GIỮ / BỎ Tầng A | xây vòng lặp lên nền vô ích |
| **5** | Khung vòng lặp + **T11 viết TRƯỚC** | T11, T12, T16, T17 xanh | thất bại thầm lặng |
| **6** | **E2** — vòng lặp có đáng không? (H1) | quyết định GIỮ / BỎ vòng lặp | kéo dài một hướng đã chết |
| **7** | Lưới đầy đủ, 5-fold, bảng so sánh | bảng đầy đủ mean (SD) | |
| **8** | Downstream (§6.3) | ROC-AUC + PR-AUC, 5× 5-fold | |
| **9** | Metadata FT (§5.5) | chỉ khi GĐ 6 = GIỮ | thêm biến gây nhiễu |

**Hai cổng quyết định là GĐ 4 và GĐ 6.** Cả hai đều có kết quả "BỎ" hoàn toàn hợp lệ. Thiết kế tài liệu này cho phép đóng góp co lại — điều không được phép là đi tiếp mà giả vờ cổng đã mở.

---

# PHẦN 14 — CÂU HỎI MỞ (phải chốt trước GĐ 5)

1. **MNAR ở mức ô nghĩa là gì trong dữ liệu của ta?** Cơ chế MNAR của [S] chọn theo time point dựa trên alpha diversity cao. Ở mức ô, ứng viên hợp lý: xác suất khuyết tăng theo độ phong phú thấp (dưới ngưỡng phát hiện) — nhưng như thế MNAR và rounded-zero **trộn lẫn**, và ta đang lấy pseudo-count đè lên chính hiện tượng đó. **Chưa có lời giải. Nếu không giải được, chỉ báo cáo MCAR.**

2. **Một "composition" là gì?** Đã chốt: một cặp *(subject, time point)* chuẩn hóa trên `K` loài (§4). Ghi lại để không ai đổi âm thầm.

3. **Tầng A có nên dùng thông tin thời gian không?** Hiện tại: không (§3.5). Nếu E1 cho thấy Tầng A yếu, đây là biến thể đầu tiên đáng thử — nhưng phải thử **sau** E1, như một thí nghiệm riêng.

4. **`target_ratio` cho vòng lặp nên là bao nhiêu và có nên đổi theo `i`?** [S] dùng tỷ lệ cố định cho một lượt. Với vòng lặp, một lịch trình (ví dụ giảm dần) là hợp lý nhưng **thuần suy đoán**. Mặc định: cố định. Đổi thì phải đo.

5. **Train lại từ đầu hay train tiếp ở mỗi vòng?** Sơ đồ [D] không nói rõ. Train lại từ đầu: sạch hơn, đắt gấp `N` lần. Train tiếp: rẻ, nhưng trạng thái optimizer mang theo thiên lệch từ các vòng trước và làm mờ diễn giải. **Mặc định: train lại từ đầu.** Nếu đổi vì lý do chi phí, phải ghi rõ trong kết quả.

6. **Nếu H1 bị bác bỏ thì đóng góp còn lại là gì?** Trả lời trung thực: "kNN-Aitchison làm khởi tạo đúng hình học + CSDI thích ứng cho dữ liệu compositional thưa theo ô, kèm bằng chứng vòng lặp không giúp ích". Đó là một đóng góp hợp lệ và báo cáo được. Chốt điều này **bây giờ**, trước khi có số liệu, để sau này không bị cám dỗ.
