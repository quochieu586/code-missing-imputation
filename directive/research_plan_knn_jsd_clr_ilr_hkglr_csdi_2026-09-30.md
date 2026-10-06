# Kế hoạch nghiên cứu: khởi tạo KNN/JSD, CLR–ILR–HKGLR và diffusion có điều kiện

**Ngày lập:** 30/09/2026. **Trạng thái:** tổng hợp lý thuyết và đặc tả nghiên cứu; các thí nghiệm đề xuất dưới đây chưa được chạy.

**Đầu ra của task hiện tại:** bản kế hoạch Markdown này. Việc implement thuật toán mới, tái lập paper và chạy ma trận thí nghiệm là các bước tiếp theo được đặc tả trong tài liệu.

## 1. Mục tiêu và các quyết định chính

Nghiên cứu trả lời ba câu hỏi: (i) khởi tạo bằng KNN-Aitchison của Hron hay JSD của Tsagris phù hợp hơn với dữ liệu variants; (ii) lựa chọn CLR, ILR hoặc HKGLR ảnh hưởng thế nào đến mô hình điền khuyết; (iii) diffusion có cải thiện khởi tạo và các baseline đơn giản khi đánh giá đúng cơ chế thiếu của dữ liệu hay không.

Các quyết định triển khai:

1. Tái lập thuật toán khởi tạo độc lập trước khi ghép diffusion. JSD của Tsagris gồm cả tìm láng giềng, phép lấy trung bình và phân bổ khối lượng thiếu; thay riêng distance trong KNN hiện có chưa phải tái lập thuật toán đó.
2. HKGLR dùng **5 variants có tỷ lệ missing thấp nhất**, theo yêu cầu người dùng. Trong CV, tính thứ hạng chỉ trên dữ liệu train của từng fold, đóng băng bộ tham chiếu trước khi chấm validation/test.
3. Thêm ILR và HKGLR thành nhánh nghiên cứu chính. Điều này mở rộng quy định cũ chỉ dùng ILR cho baseline LTS. Giữ quy tắc chống rò rỉ, khóa quan sát và giới hạn vòng lặp của handoff cũ.
4. Giữ **một vòng diffusion bên ngoài** trong ma trận nghiên cứu chính: `max_iter=1`, `n_iter=1`, `converged=False`. Epoch huấn luyện, bước reverse DDPM và nhiều mẫu Monte Carlo nằm trong vòng này.
5. Mask thiếu được định nghĩa trong không gian variants gốc. Khi đổi LR phải thiết kế lại cách conditioning và supervision; không sao chép mask 17 chiều sang ILR 16 chiều.
6. So sánh tất cả phương pháp bằng metric trên cùng composition sau inverse và cùng mask gốc. MAE trong tọa độ ILR/HKGLR chỉ dùng chẩn đoán nội bộ.
7. Xem thiếu cả chuỗi của một variant tại một địa điểm là kịch bản chính; MCAR theo ô là đối chứng. Audit hiện tại cho thấy mỗi địa điểm có đúng một pattern quan sát xuyên suốt thời gian.
8. Dùng standard CSDI làm lõi đối chứng khi so LR. Chỉ thử CNN theo nhóm khi có mapping sinh học được xác nhận; không tự suy ra phylum từ tên các biến thể virus.

## 2. Nguồn đã đọc và cách sử dụng

Số trang dưới đây là số trang trong PDF, không nhất thiết là số trang tạp chí. Bản Hron địa phương là preprint ngày 27/11/2009; bài xuất bản được trích dẫn là **Hron et al., 2010**. Hai mốc này giải thích sự khác nhau về năm trong handoff cũ.

| Mã | Tài liệu và vị trí | Nội dung dùng trong kế hoạch |
|---|---|---|
| H | [Hron, Templ & Filzmoser: Imputation of missing values for compositional data using classical and robust methods](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/research/Hron.pdf), 17 trang; bản xuất bản 2010, DOI `10.1016/j.csda.2009.11.023` | §3.1, PDF tr. 5–7: KNN 2a, scale adjustment và median; §3.2: ILR và hồi quy lặp; §4, Table 1: fixture 152.1; metric compositional |
| T | [Tsagris, Stewart & Alenazi, 2026: A Jensen-Shannon divergence based k–NN algorithm for missing value imputation in compositional data](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/research/tsagris.pdf), 18 trang, arXiv v1 `2605.29702` | §2.2–2.5, PDF tr. 4–10: JSD, sáu bước KNN, α-Fréchet mean, tuning và adaptive patterns; §3–4: giới hạn và đối chứng |
| C | [Huang et al., 2025: Compositional data modeling of high-dimensional single cell RNA-seq (CoDA-hd): its advantages over commonly used normalization approaches](<C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/directive/2025_Compositional data modeling of high-dimensional single cell RNA-seq (CoDA-hd) -- its advantages over commonly used normalization approaches.pdf>), 20 trang | PDF tr. 5–7: zero handling, CLR/ILR/HKGLR; tr. 12–13: HKGLR có kết quả kém trong các thí nghiệm của tác giả. Không suy diễn ưu thế sang SARS-CoV-2 |
| D | [Tashiro et al., 2021: CSDI](https://arxiv.org/pdf/2107.03502), NeurIPS 2021 | §4–5; Appendix B, D, E: conditional diffusion, target strategies, mask, xử lý missing khi train và kiến trúc |
| S23 | [Seki et al., 2023: Imputing time-series microbiome abundance profiles with diffusion model](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/directive/csdi_2023.pdf), 6 trang | Tiền thân áp dụng CSDI cho microbiome, đặc biệt bài toán thiếu time point |
| S25 | [Seki et al., 2025: Diffusion model for imputing time-series gut microbiome profiles using phylogenetic information and metadata integration](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/research/csdi.pdf), 9 trang, DOI `10.1093/bioadv/vbaf181` | §2: CLR, preprocessing, conditional diffusion, phylum CNN, chia subject; §3: đánh giá và các trường hợp baseline tốt hơn |

Nguồn triển khai để đối chiếu:

- Hron: package R `robCompositions`, fixture địa phương trong `tests/fixtures`; xác minh phiên bản package và chế độ thuật toán trước khi dùng làm oracle.
- Tsagris: package tác giả [CompositionalNAimp](https://cran.mirror.garr.it/CRAN/web/packages/CompositionalNAimp/index.html), bản 1.1 được liệt kê ngày 29/05/2026; API `knnimp`, `alfa.knnimp`, `cv.knnimp`. [Tài liệu `frechet` của Compositional](https://search.r-project.org/CRAN/refmans/Compositional/html/frechet.html) dùng đối chiếu α-mean. Không nhầm với `comp.knn`, vốn là thuật toán phân loại.
- CoDA-hd: [repository của tác giả](https://github.com/GO3295/CoDAhd).
- CSDI gốc: [ermongroup/CSDI](https://github.com/ermongroup/CSDI), đặc biệt [main_model.py](https://raw.githubusercontent.com/ermongroup/CSDI/main/main_model.py).
- Seki 2025: [repository chính thức metag_time_impute_phylo](https://github.com/misatoseki/metag_time_impute_phylo). Lúc kiểm tra đường dẫn cuối cùng, thư mục `code-missing-imputation/metag_time_impute_phylo` được nêu trong yêu cầu cũ không còn hiện diện ở workspace; cần lấy lại snapshot xác định trước khi làm source-level parity. Paper địa phương vẫn có và đã được đọc.

**Quy ước nguồn:** các công thức ghi H/T/C/S là từ tài liệu; các thiết kế thực nghiệm, xử lý biên và kiến trúc thích ứng ghi dưới đây là đề xuất của dự án. Chưa thực hiện numerical parity với các package R trong task lập kế hoạch này. Bản T địa phương có lỗi ký hiệu và chỗ tham chiếu chưa hoàn thiện; các quyết định diễn giải được ghi ở §5.

## 3. Hiện trạng dữ liệu và code đã kiểm tra

Audit trực tiếp [covariants.csv](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/data/covariants.csv) cho kết quả:

| Đại lượng | Kết quả |
|---|---:|
| Hàng địa điểm–ngày | 11.671 |
| Địa điểm | 150 |
| Variants | 17 |
| Ô thiếu thật | 86.050 / 198.407 |
| Ô zero quan sát được, trước xử lý ngưỡng | 92.440 |
| Hàng đầy đủ 17 variants | 516 |
| Hàng đầy đủ nhưng tổng 17 counts bằng 0 | 27 |
| Hàng đầy đủ có tổng dương, dùng được cho closure gốc | 489 |
| Địa điểm có hàng đầy đủ | Canada, Denmark, Netherlands, United Kingdom, United States |
| Số pattern mask khác nhau trong mỗi địa điểm | 1, tại cả 150 địa điểm |
| Hàng có đủ 5 variants tham chiếu toàn cục | 7.369, tương đương 63,14% |

**Diễn giải:** dữ liệu thiếu có cấu trúc theo địa điểm–variant. Điều này không chứng minh MCAR, MAR hoặc MNAR; nó chỉ mô tả pattern quan sát. Linear/LOCF có thể mạnh trong bài test che ô ngẫu nhiên nhưng không có điểm cùng variant trong địa điểm để nội suy khi thiếu cả chuỗi. Phải báo riêng hai tình huống.

Trong 516 hàng đầy đủ, chỉ 255 hàng có tổng 17 counts đúng bằng `total_sequence`; tỷ số tổng counts/`total_sequence` có min 0 và max 1. Chưa được coi `total_sequence` là tổng của đúng 17 thành phần. Tổng các counts không vượt tổng sequence là điều kiện cần, chưa chứng minh các nhóm không chồng lắp và bao phủ toàn bộ mẫu.

### 3.1 Năm variants tham chiếu HKGLR

| Thứ hạng toàn cục, chỉ để mô tả | Variant | Số ô thiếu | Missing rate |
|---|---|---:|---:|
| 1 | Omicron | 13 | 0,1114% |
| 2 | Delta | 15 | 0,1285% |
| 3 | Alpha | 346 | 2,9646% |
| 4 | recombinant | 1.958 | 16,7766% |
| 5 | Beta | 3.066 | 26,2702% |

Tỷ lệ zero trong các ô quan sát của năm variants lần lượt khoảng 41,82%; 74,54%; 83,28%; 74,65%; 88,94%. Vì vậy “ít missing” không đồng nghĩa “phong phú ổn định” hoặc “ít cần pseudo-count”. Đây là một proxy thống kê theo yêu cầu người dùng, không được gọi là nhóm housekeeping có cơ sở sinh học.

Trong mỗi outer/inner fold: tính missing rate theo số hàng train, tie-break bằng tên variant theo thứ tự cố định, lấy đúng 5 variants và lưu danh sách. Dùng mask missing gốc của phần train trước khi tạo mask đánh giá nhân tạo. Nếu nghiên cứu một protocol trong đó mask nhân tạo chính là toàn bộ dữ liệu khả dụng, phải fit lại selector theo protocol đó và ghi rõ; không dùng trạng thái test để chọn tham chiếu. Phân tích độ nhạy có thể tính rate trung bình đều theo địa điểm, nhưng quy tắc chính vẫn là rate theo ô.

### 3.2 Kết quả đang tồn tại trong workspace

Đọc lại [results.json](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/artifacts/stage_b_single_pass_final/results.json) và [báo cáo hiện tại](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/reports/stage_b_single_pass_report.md), thay vì dùng số từ các lượt chat cũ:

| Phương pháp | MAE CLR toàn bộ | MAE khác 0 | MAE pseudo-count | M2 |
|---|---:|---:|---:|---:|
| Linear | 0,316564 | 0,580492 | 0,227913 | 4,316993 |
| KNN-Aitchison | 1,096386 | 2,820314 | 0,517331 | 44,089025 |
| KNN + CSDI một lượt | 1,054418 | 3,268653 | 0,310673 | 46,549894 |

Đây là pilot fold 0, 80 hàng test tại United States, 696 ô che; config thực trong JSON: 20 epochs, 20 DDPM steps, 2 samples, channels 16, batch 8, window 8. MAE tổng giảm khoảng 3,83% so với KNN nhưng lỗi nhóm khác 0 và M2 tăng. Chưa có cơ sở kết luận cải thiện có ý nghĩa sinh học hoặc tổng quát hóa sang địa điểm khác. Trong 80 hàng này có cả các hàng tổng counts bằng 0 đã được pseudo-count hóa; protocol mới phải tách chúng khỏi benchmark native simplex.

Code đã có CLR/ILR, KNN 2a, CSDI thích ứng và các kiểm thử mask. Tuy nhiên:

- `ilr_inverse` hiện trả một đại diện chưa closure; cần adapter trả composition tổng 1 để đánh giá nhất quán.
- Tie-break KNN hiện dựa trên chỉ số pool; cần ID hàng ổn định khi kiểm tra bất biến dưới đổi thứ tự donor.
- Sampling hiện dùng median thay missing trong conditioning và clipping khoảng CLR theo train. Đây là các lựa chọn mô hình cần ablation.
- KNN có thể tác động vào tâm CLR của nhãn và khoảng clipping, dù giá trị KNN bị bỏ khỏi conditioning tại missing. Vì vậy tên “KNN + CSDI” chưa tự chứng minh mô hình thực sự tận dụng KNN như thông tin điều kiện.
- Test loại bỏ nhãn ngoài mask ở loss reducer chưa chứng minh toàn bộ preprocessing/conditioning không rò rỉ target qua tâm log.

## 4. Hợp đồng dữ liệu: chốt trước khi implement JSD

### 4.1 Ba biểu diễn cần lưu

1. `X_raw`, `M_raw`: counts/abundance gốc, zero quan sát vẫn là zero, NaN là missing.
2. `X_nonnegative`: dữ liệu không âm sau ngưỡng phát hiện đã chốt; dùng cho nhánh native JSD. Lưu `zero_raw` và `zero_after_threshold` riêng.
3. `X_positive`: phiên bản dương dùng LR/Aitchison. Pseudo-count và mọi thống kê fit chỉ trên phần train khả dụng. Lưu rõ ô nào được sửa và lượng sửa.

Không dùng một mảng đã pseudo-count hóa để tuyên bố kiểm tra khả năng giữ zero của Tsagris. Khi đánh giá LR, áp dụng cùng quy tắc làm dương cho truth và prediction. Khi xuất counts cuối, khôi phục chính xác quan sát gốc, gồm observed zeros; phiên bản dương phục vụ LR được lưu riêng. Đây là mở rộng cần thiết so với output cũ khóa giá trị pseudo-count.

### 4.2 Hai chế độ về tổng khối lượng

**K: biết tổng của đúng các thành phần.** Cho tổng hợp lệ `S_i`, đặt `p_ij=x_ij/S_i`; tổng phần thiếu `T_i=1−Σ_{j∈O_i}p_ij` là xác định. Có thể chạy Tsagris nguyên bản và khóa các phần quan sát. Benchmark tổng hợp có thể closure hàng đầy đủ trước khi che, nhưng phải ghi đây là protocol có biết tổng, vì thông tin tổng giúp suy ra phần thiếu.

**U: chưa biết tổng của đúng 17 variants.** Không đặt `S_i=Σ observed`, vì sẽ ép `T_i=0`. Không tự đặt `S_i=total_sequence` khi chưa xác minh độ bao phủ và phân loại. Trong chế độ này, đánh giá Hron và LR vẫn thực hiện được; Tsagris nguyên bản chỉ chạy trên benchmark K. Một thuật toán JSD dùng scale adjustment theo donor phải mang tên biến thể mới, không gắn nhãn tái lập Tsagris.

Audit cần kiểm tra mã tạo bảng, định nghĩa `recombinant`/`S:677`, tính loại trừ lẫn nhau của nhóm, và ý nghĩa `total_sequence`. Nếu có “Other” được đo độc lập, có thể mở composition thành 18 phần và chạy thí nghiệm riêng. Không lấy `total_sequence−Σ observed` làm observed Other khi trong tổng đó còn các variants đang missing. Nếu chỉ có tổng sequence nhưng Other cũng chưa biết, có thể coi Other là một thành phần thiếu nữa; khả năng định danh và donor coverage phải được báo riêng.

Hàng tổng bằng 0 không có closure xác định: gắn trạng thái `all_zero_composition`, loại khỏi benchmark simplex chính và báo số lượng. Việc biến toàn bộ hàng đó thành uniform bằng pseudo-count là giả định mạnh, chỉ khảo sát trong phân tích độ nhạy.

## 5. Đặc tả khởi tạo imputation

### 5.1 Hron: KNN-Aitchison, chiến lược 2a

Với hàng i, `O_i` là tập quan sát gốc và `M_i` là tập thiếu. Với mỗi `j∈M_i`:

1. Donor pool gồm **các hàng train** có quan sát tại tất cả vị trí `O_i∪{j}`; loại chính hàng i. Giá trị đã điền không trở thành quan sát để mở rộng donor pool.
2. Tính khoảng cách trên cùng subcomposition:

   `d(i,r)=||clr(x_i[O_i])−clr(x_r[O_i])||₂`.

3. Chọn k donor gần nhất; dùng stable row ID để giải quyết ties.
4. Hiệu chỉnh robust: `f_ir=median(x_i[O_i])/median(x_r[O_i])`.
5. Điền `x̂_ij=median_r(f_ir·x_rj)`, giữ nguyên mọi quan sát.

Đây là công thức (6), (7), §3.1 của H. Bản dùng tổng ở công thức (5) là đối chứng scale-adjustment, không thay mặc định robust. `O_i` đóng băng; không thêm ô mới điền vào điều kiện tìm donor.

**Biên bắt buộc:** ít donor hơn k thì dùng `min(k,n_valid)` và báo `effective_k`; không donor thì trả trạng thái fallback. Với `|O_i|=1`, mọi distance Aitchison trên subcomposition đều bằng 0 nên phải báo mất khả năng phân biệt láng giềng. Với `|O_i|=0`, không có scale từ quan sát; fallback theo prior train phải được gắn nhãn. Median cột không giữ được bất biến scale của toàn bộ pipeline khi thiếu neo scale; không tuyên bố T7 chung cho các trường hợp fallback đó.

**Tuning:** k trong `{2,…,10}`, có thể thêm k=1 làm kiểm tra biên; inner CV theo địa điểm, tạo mask tương ứng kịch bản đánh giá. Giữ k=8 như baseline kế thừa, không gọi là k tối ưu. Ghi donor IDs, khoảng cách, hệ số f, effective k và tỷ lệ fallback.

### 5.2 Tsagris: JSD KNN nguyên bản

T dùng phiên bản JSD bằng hai lần định nghĩa thường gặp:

`J_T(p,q)=Σ_j[p_j ln(2p_j/(p_j+q_j))+q_j ln(2q_j/(p_j+q_j))]`.

Quy ước đóng góp tại zero bằng 0; cả hai cùng zero cũng đóng góp 0. Miền giá trị `[0,2 ln 2]`; `sqrt(J_T)` là metric. Thứ hạng donor không đổi khi dùng J, sqrt(J) hoặc nhân hằng số, nhưng giá trị metric báo cáo phải ghi đúng quy ước. `scipy.spatial.distance.jensenshannon` có quy ước khác về căn bậc hai và hệ số; cần test đối chiếu trước khi dùng.

Thuật toán §2.3, PDF tr. 7–9:

1. Lấy **donor train đầy đủ toàn bộ D phần**. Đây là khác biệt với Hron 2a.
2. Với hàng thiếu i, closure các phần quan sát của i và các phần tương ứng của mỗi donor: `p_i^O=C(p_i[O_i])`, `p_r^O=C(p_r[O_i])`.
3. Chọn k donor có `J_T(p_i^O,p_r^O)` nhỏ nhất.
4. Lấy trung bình số học của **toàn bộ composition donor**: `μ=(1/k)Σ_r p_r`.
5. Tính `T_i=1−Σ_{j∈O_i}p_ij` từ tổng hợp lệ ở chế độ K.
6. Điền đồng thời `p̂_ij=T_i μ_j / Σ_{m∈M_i}μ_m`, với `j∈M_i`; quan sát giữ nguyên.

Fixture của paper: `[0.2,NA,0.3,0.1,NA]`, hai donor chọn được có mean `[0.15,0.30,0.30,0.10,0.15]`; missing mass 0.4; giá trị đúng chưa làm tròn là `4/15` và `2/15`, tương ứng khoảng 0.27 và 0.13 trong paper.

**Biên:** `Σ observed=0` làm subcomposition quan sát không xác định; `T_i<0` là lỗi hợp đồng tổng; `T_i=0` cho các phần thiếu bằng 0; `Σ μ_M=0` trong khi `T_i>0` là thiếu hỗ trợ donor. Phải trả status rõ ràng. Không âm thầm cộng epsilon, chia đều hay chuyển donor pool rồi vẫn gọi là thuật toán gốc. Báo coverage của phương pháp cùng metric trên tập chung hợp lệ.

**Lỗi ký hiệu của bản T:** tr. 8 gọi `T_i=1−Σ observed` là tổng phần không thiếu và có dòng chuẩn hóa dùng ký hiệu mâu thuẫn. Công thức phân bổ và ví dụ 0.4 cho thấy T là **khối lượng phần thiếu**; chuẩn hóa quan sát phải chia cho `Σ observed`. Chốt diễn giải bằng fixture và package tác giả trước triển khai chính thức.

### 5.3 JSD α–KNN và adaptive α–KNN

Chỉ thay bước lấy mean, giữ donor ranking theo JSD trên subcomposition như nhánh cơ bản. Với k donor `p_r`:

`u_rj=p_rj^α / Σ_l p_rl^α`,

`ū_j=(1/k)Σ_r u_rj`,

`μ_α=C((ū_j^(1/α))_j)`.

Phải chuẩn hóa power **trong từng donor trước khi lấy mean**; không thay bằng componentwise power mean thông thường. Khi α=1 thu được arithmetic mean; với dữ liệu dương, giới hạn α→0 là closed geometric mean. Công thức (6) trong PDF có chỉ số mẫu/thành phần không nhất quán; biểu thức triển khai trên được kiểm tra bằng các tính chất này và `Compositional::frechet`.

Khi có zero: grid chính `α∈{0.1,0.2,…,1.0}`; không dùng α≤0. Với bộ dữ liệu dương riêng: có thể khảo sát `{-1,-0.9,…,1}`, xử lý α=0 bằng giới hạn giải tích. Không dựa vào câu giải thích tính lồi của `x^α` trong bản preprint để chứng minh duy nhất: với `0<α<1`, hàm này là lõm. Nghiên cứu cần numerical parity và tính chất biến đổi đã xác minh, không cần lặp lại lập luận đó.

Tuning đồng thời `(k,α)` trong inner train. T đề xuất repeated leave-N-out trên complete donor và tái tạo pattern thiếu; ở dữ liệu thời gian này, sửa protocol thành chia theo **địa điểm**, ghi rõ đây là thích ứng chống rò rỉ. Dùng JSD khi có zero và Aitchison khi dữ liệu dương theo paper; báo thêm metric chung để tránh chỉ tối ưu lợi thế cho một phương pháp.

Adaptive: chọn `(k,α)` theo pattern chỉ khi có đủ địa điểm/hàng train để kiểm chứng. Mặc định dùng một cặp global. Pattern không đủ hỗ trợ hoặc chưa thấy dùng cặp global, không học từ test. Chỉ mở rộng adaptive nếu nested validation chứng minh lợi ích; chính T báo lợi ích adaptive không nhất quán.

### 5.4 So sánh công bằng và deliverables của khởi tạo

Chạy hai nhóm so sánh: (a) thuật toán nguyên bản Hron 2a và Tsagris complete-donor, có báo khác biệt donor coverage; (b) đối chứng dùng cùng complete-donor pool để tách ảnh hưởng của distance/aggregation khỏi khả năng dùng donor chưa đầy đủ. Nhóm (b) là thí nghiệm kiểm soát được đặt tên riêng, không thay kết quả thuật toán gốc.

Đầu ra yêu cầu: `X_init`, mask gốc, trạng thái từng ô/hàng, donor IDs, các tham số chọn trong inner CV, tổng thời gian tuning/imputation và số trường hợp không định danh. Số liệu benchmark từ paper chỉ dùng tái lập, không nhập vào bảng kết quả của dự án.

## 6. Đặc tả CLR, ILR và HKGLR

Các công thức dưới đây áp dụng sau cùng một positivity policy cho một hàng `x∈R_+^D`; `C` là closure. Sử dụng natural log và float64 cho kiểm thử số học.

### 6.1 CLR

`z_j=ln x_j−(1/D)Σ_l ln x_l`; inverse là `softmax(z)`.

Có D tọa độ, rank D−1, tổng tọa độ bằng 0; bất biến với nhân x bởi một số dương. Chỉ chiếu đầu ra CLR cuối về tổng 0. Nhiễu epsilon Gaussian của DDPM không phải CLR, nên không tự ý trừ mean của epsilon trong loss.

### 6.2 ILR

Lưu một ma trận cơ sở cố định `V∈R^(D×(D−1))` với `VᵀV=I`, `Vᵀ1=0`:

`u=Vᵀ ln x=Vᵀ clr(x)`, `x=C(exp(Vu))`.

Với D=17, u có 16 tọa độ. Chọn một cơ sở pivot/Helmert được ghi rõ thứ tự và dấu. Công thức pivot của CoDA-hd dùng `ln(x_i/g(x_{i+1:D}))`; code Hron hiện tại dùng tỷ số đảo. Hai cách lệch dấu nhưng đều hợp lệ nếu inverse khớp. Fixture không được nhầm khác dấu với lỗi hình học.

**Không trừ trung bình của vector ILR 16 chiều:** việc đó loại mất một chiều hợp lệ. Không gắn balance với một variant đơn lẻ hoặc áp phylum CNN trên balance. Thử một cơ sở trực chuẩn khác trong sensitivity; distance bất biến với đổi cơ sở, nhưng mạng phi tuyến có thể không bất biến.

### 6.3 HKGLR với năm variants ít thiếu nhất

Cho tập tham chiếu H có đúng 5 phần, đóng băng trên train:

`h_j=ln x_j−(1/5)Σ_{r∈H}ln x_r`, `x=C(exp(h))`.

Giữ đủ 17 tọa độ, kể cả năm tọa độ tham chiếu. Có `Σ_{r∈H}h_r=0`, rank D−1; tổng trên **toàn bộ** 17 tọa độ thường khác 0. Khi cần đưa output về biểu diễn HKGLR chuẩn, trừ `mean(h[H])` khỏi tất cả các tọa độ.

Đẳng thức cần kiểm thử và dùng để diễn giải:

`h−mean_all(h)·1=clr(x)`.

Do đó, nếu pipeline tự động center HKGLR theo cả hàng trước khi đưa vào model, nhánh HKGLR đã biến thành CLR. Đồng thời, HKGLR đủ D chiều vẫn chứa cùng thông tin composition như CLR; một cải thiện thực nghiệm đến từ cách biểu diễn, conditioning, prior nhiễu hoặc kiến trúc, không phải từ thêm thông tin sinh học.

Khi một reference missing, denominator thật chưa biết. Phải lưu `reference_complete` và số reference phải impute. Không thay H theo từng hàng vì sẽ tạo hệ tọa độ khác nhau giữa các mẫu. Khi reference bằng zero quan sát, áp cùng positivity policy; phân tích sai số theo số reference được pseudo-count hóa.

CoDA-hd dùng năm genes do kiến thức chuyên ngành gợi ý; kết quả của paper không bảo đảm HKGLR tốt hơn CLR. Quy tắc năm variants ít missing là **biến thể HKGLR của dự án** và cần kiểm chứng độc lập. [C, PDF tr. 7, 12–13]

### 6.4 Preprocessing và hình học nhiễu là hai yếu tố gây nhiễu thí nghiệm

Giữ cùng positivity policy khi so ba LR. Chỉ sau đó mới khảo sát pseudo-count thay zero hiện tại so với một count-addition scheme của C. Count addition vào mọi ô làm đổi cả quan sát dương, còn zero replacement chỉ sửa zero: phải ghi rõ, không gộp hai phép đó thành một lựa chọn “pseudo-count”. SGM cần tổng counts thực và thống kê geometric mean của tổng chỉ trên train; chưa dùng trên hàng thiếu với tổng chưa định danh.

IID Gaussian trong các tensor CLR, ILR và HKGLR không có cùng covariance tọa độ hay cùng phần dư ngoài subspace hợp lệ. Tuy nhiên, Gaussian D chiều sau projection CLR và Gaussian D−1 chiều qua basis ILR trực chuẩn tạo cùng phân phối nhiễu trong simplex ở cùng variance; không mặc định mọi khác biệt tọa độ đều làm đổi prior composition. Đối chứng ghép cùng realization: lấy `η∼N(0,I_(D−1))`, dựng `δz=Vη`, rồi `δh=δz−mean_H(δz)·1`. Ba perturbations này biểu diễn cùng thay đổi log-ratio. Ghi rõ covariance, projection và phần dư của từng nhánh; native-coordinate DDPM và model xử lý redundant directions vẫn có thể khác nhau dù nhiễu sau closure tương đương.

Chuẩn hóa từng feature bằng mean/std riêng cũng thay hình học và có thể phá ràng buộc CLR/HKGLR. Phải đảo chuẩn hóa trước projection, inverse và metric; mọi scaler chỉ fit train. Không dùng quy tắc center CLR cho tất cả LR.

## 7. Thiết kế mask, conditioning và inverse: điều kiện để so sánh có ý nghĩa

### 7.1 Ba mask khác nhau, không được nhập làm một

- `observed_raw`: giá trị counts thật có quan sát trong dữ liệu gốc; zero quan sát vẫn là observed.
- `hidden_eval`: tập ô có ground truth nhưng bị che nhân tạo để validation/test. Không cho initializer, scaler, donor selection hoặc conditioning đọc giá trị này.
- `hidden_train`: tập ô observed bị che trong một lượt self-supervision. Phải che **trước** mọi phép tính có thể truyền thông tin sang các ô khác.

Đặt `visible = observed_raw & ~hidden_eval & ~hidden_train`. Giá trị do initializer sinh ra phải có cờ `is_imputed`; không nâng thành ground truth chỉ vì đã điền đủ ma trận. Natural missing không tham gia loss hoặc metric cần ground truth.

| Biểu diễn | Khi nào một tọa độ thật được định danh từ counts nhìn thấy? | Hệ quả |
|---|---|---|
| CLR | Cần geometric mean của toàn bộ D phần, trừ khi có đủ ràng buộc bổ sung để suy ra phần thiếu | Một count observed không làm CLR tương ứng trở thành observed |
| ILR | Cần mọi phần trong support của balance tương ứng, hoặc đủ thông tin suy ra ratio | Mask D counts không phải mask D−1 balances; không cắt bỏ một cột mask để khớp shape |
| HKGLR | Cần numerator và geometric mean của 5 references; từng tọa độ có thể có support khác do numerator cũng thuộc H | References bị che có thể làm nhiều tọa độ mất định danh; không dùng denominator chứa hidden truth |

Các log-ratio giữa hai phần cùng observed có thể vẫn biết chính xác, nhưng điều đó không đồng nghĩa biết tọa độ CLR đầy đủ. Đây là vấn đề định danh, không chỉ là lỗi shape của tensor.

### 7.2 Phương án chính cho nghiên cứu LR + diffusion

Tách hai nhánh đầu vào:

1. **Condition encoder trong không gian raw:** nhận các counts/proportions nhìn thấy, visible mask, metadata thời gian và các ước lượng initialization có cờ provenance. Giá trị hidden phải bị thay bằng sentinel an toàn trước khi chạy initializer. Mọi tỷ lệ sử dụng tổng counts phải theo mass protocol ở mục 4.
2. **Noisy target branch trong không gian LR:** nhận toàn bộ latent LR có nhiễu. Ground truth composition đầy đủ chỉ dùng để tạo `z0`, forward noising và loss; không dùng để tạo condition branch.

Dùng cùng kiểu condition encoder, cùng information budget và cùng cơ chế attention khi so CLR/ILR/HKGLR. Khác biệt D so với D−1 phải được ghi trong số tham số; giữ ngân sách mô hình gần tương đương. Đây là **adaptation của CSDI cho composition**, không gọi là bản sao nguyên trạng CSDI.

Nhánh đánh giá chính chỉ học full-LR denoising labels từ các hàng có composition thật đầy đủ, tổng dương. Nếu dùng temporal windows, cần các đoạn có đầy đủ labels theo quy tắc đã định; không âm thầm dùng hàng partially observed làm full-LR ground truth. Dataset hiện chỉ có 489 hàng đầy đủ, tổng dương thuộc 5 locations, nên nhánh này có rủi ro thiếu dữ liệu rõ rệt. Báo riêng số windows thực tế sau split, padding và eligibility, rồi mới quyết định độ lớn mô hình.

Huấn luyện bằng full-LR của dữ liệu đã impute là một **nhánh pseudo-label**, có thể nghiên cứu thứ cấp nhưng phải đặt tên riêng, báo tỷ lệ nhãn ước lượng và so với nhánh complete-label. Loss chỉ trên observed indices của CLR không tự loại bỏ ảnh hưởng của nhãn imputed trong geometric mean.

Đối chứng HKGLR hạn chế: giữ cả 5 references thật sự visible, chỉ che 12 variants còn lại, và dùng cùng target set cho mọi phương pháp. Nhánh này giúp kiểm tra khi denominator có thật, nhưng không thay thế benchmark chính trên cả 17 variants. Không bảo vệ references khỏi bị che trong benchmark chính để tạo lợi thế cho HKGLR.

### 7.3 Initializer phải được chạy lại sau masking

Với mỗi training mask hoặc mask đã cache:

1. Che target trên bản dữ liệu đầu vào.
2. Tìm donors từ nguồn train hợp lệ. Với một training row làm query, loại chính row đó và ngăn hidden values của nó quay lại qua donor/cache; có thể dùng cross-fitting theo location.
3. Chạy Hron/JSD trên dữ liệu đã che; lưu donor IDs, overlap, fallback và tham số.
4. Tạo condition từ visible data và initializer output, không từ bản dữ liệu gốc trước che.

Không chỉ ghi đè giá trị target sau khi đã impute: hidden truth có thể đã ảnh hưởng tới donor ranking hoặc các ô imputed khác. Tách cache theo split, mask hash, initializer config và positivity policy.

Thêm nhánh `no_init`: chỉ raw visible + mask + time metadata. Nếu nhánh có initialization không cải thiện, không kết luận diffusion đã tận dụng KNN chỉ vì KNN xuất hiện trong tên pipeline.

### 7.4 Inverse và bảo toàn observed counts

- **Mass của D phần biết hợp lệ:** inverse LR tạo composition; giữ nguyên raw observed, chỉ phân phối residual mass cho missing parts theo tỷ lệ dự đoán. Reject/flag residual âm; không sửa observed để ép hợp lệ. Không dùng `total_sequence` làm mass này trước khi giải quyết ý nghĩa của 17 variants.
- **Mass chưa biết:** composition chỉ định danh ratios. Muốn báo counts phải thêm giả định scale, ví dụ robust anchor từ các observed dương; ghi rõ estimator và fallback khi không có anchor. Sau scaling, khôi phục raw observed chính xác, kể cả raw zero.
- Closure sau observed restoration chỉ là một view để đánh giá composition; nó không đồng thời bảo toàn mọi observed count trên thang đã đóng. Lưu riêng output counts và output composition.
- Kiểm tra finite, nonnegative, residual mass, observed preservation và scale ở từng output. Một guard như `max_scale_ratio=1000` chỉ là ngưỡng kỹ thuật; vượt/không vượt ngưỡng không chứng minh tính hợp lý sinh học.

## 8. CSDI: lý thuyết cần giữ đúng và phần cần sửa cho bài toán này

### 8.1 Forward/reverse diffusion

Đặt `a_t=1−β_t`, `ᾱ_t=∏_(s=1)^t a_s`. Với target latent `z0`, forward diffusion là:

```text
ε ~ N(0, I)
z_t = sqrt(ᾱ_t) z0 + sqrt(1−ᾱ_t) ε
L = E[ || ε − εθ(z_t, t, condition) ||² ]
μθ = [z_t − β_t / sqrt(1−ᾱ_t) · εθ] / sqrt(a_t)
β̃_t = β_t (1−ᾱ_(t−1)) / (1−ᾱ_t)
z_(t−1) = μθ + sqrt(β̃_t) ξ,  ξ ~ N(0,I)
```

Đây là dạng DDPM epsilon prediction với posterior variance; ở bước cuối không thêm noise. Nếu code dùng phương sai hoặc tham số hóa khác, phải ghi đúng lựa chọn và kiểm chứng đối chiếu thay vì trộn công thức. Chỉ tiêu chuẩn hóa noise theo coordinate system đã đăng ký; không tự center Gaussian noise rồi vẫn diễn giải như IID Gaussian. [D, §4 và mã nguồn chính thức]

Với original CSDI trên dữ liệu real-valued, các observed entries có thể được tách thành condition và self-supervised targets; loss chỉ tính ở targets hợp lệ. Natural missing được xử lý để không trở thành nhãn huấn luyện. Quy tắc dummy/noisy targets trong implementation và Appendix D cần được kiểm tra cùng nhau. Điều này không giải quyết tự động dependency do log-ratio gây ra. [D, §4, Appendix D]

### 8.2 Ba mức đối chiếu, không trộn nguồn

| Mức | Mục tiêu đối chiếu | Không được suy ra |
|---|---|---|
| Original CSDI 2021 | Schedule, forward/reverse equations, epsilon loss, condition/target mask và sampling | Stock mask của CSDI dùng được nguyên xi cho LR |
| Seki 2023/2025 và source `metag_time_impute_phylo` | Cách áp diffusion cho metagenomic time series, preprocessing, temporal/feature modeling và đánh giá | Mọi thiết lập của hai phiên bản paper giống nhau, hoặc COVID variants có cùng cấu trúc với taxa |
| Adaptation của dự án | Raw conditioning + LR latent, initializer channel, inverse có observed locking, single outer pass | Là replication nguyên trạng của hai nguồn trên |

Trước khi thêm LR, chạy một fixture real-valued/synthetic nhỏ để đối chiếu phép noising, loss và một reverse step với original CSDI ở cùng schedule và cùng random tensors. Mục tiêu là numerical parity của lõi, không tuyên bố tái lập toàn bộ kết quả paper. Ghi commit/source snapshot của reference code; không sửa trực tiếp thư mục source chính thức để làm implementation chính.

### 8.3 Một vòng nghĩa là gì?

Pipeline nghiên cứu:

```text
visible raw data
  → initialize một lần bằng Hron hoặc JSD
  → tạo LR/context đúng mask
  → một mô hình conditional diffusion
  → reverse diffusion T bước cho mỗi sample
  → inverse + observed locking
  → đánh giá
```

Không đưa output diffusion quay lại KNN, không refit lại bằng predictions và không lặp outer refinement. `T` diffusion steps, training epochs và nhiều posterior samples là các thành phần bên trong một lần áp dụng diffusion, không phải nhiều vòng refinement. Cross-validation là các lần đánh giá độc lập, không phải lặp cải thiện trên cùng test output.

### 8.4 Những điểm cần audit trong implementation hiện tại

- Dù mask loss chỉ chọn observed bị che, center của labels/context có thể phụ thuộc initial imputation. Phải kiểm tra toàn bộ đường dữ liệu, không chỉ dòng tính loss.
- Inference hiện thay missing context bằng train median; vì vậy cần xác định KNN đang ảnh hưởng qua đâu: labels, initialization statistics, clipping hay condition thực tế. So sánh `no_init` sẽ làm rõ đóng góp.
- Bounds dùng để clip `x0` phải fit trên train, lưu trong artifact; báo fraction bị clip theo step, variant và fold. So sánh có/không clipping trên validation. Không chọn bounds sau khi nhìn test.
- Kiểm tra temporal positions, padding, feature embeddings và mask ở attention. Không để padding trở thành zero quan sát thật. Nếu windows overlap, mỗi evaluation cell chỉ được chấm một lần sau quy tắc aggregation đã định.
- Giữ variance/schedule và seed của train/sampling trong config. Seed đơn không chứng minh độ ổn định; samples=2 của pilot không đủ đánh giá uncertainty.
- Theo dõi latent norm, predicted-noise norm, tỷ lệ nonfinite, count scale và observed preservation. Không coi clipping exponent ở ±700 là cách chữa mô hình phân kỳ.
- Với CLR có thể project latent về sum-zero trước inverse; HKGLR project bằng trừ mean trên H; ILR không cần projection. Projection sau từng reverse step là thay đổi kernel, chỉ dùng ở nhánh đã đăng ký và không gọi là stock DDPM.

## 9. Câu hỏi nghiên cứu và ma trận thí nghiệm

### 9.1 Giả thuyết cần kiểm chứng

| ID | Câu hỏi | Dấu hiệu ủng hộ | Điều kiện bác bỏ hoặc giới hạn |
|---|---|---|---|
| Q1 | JSD KNN phù hợp hơn Hron khi dữ liệu nhiều raw zero? | Giảm JSD error và không đánh đổi lớn lỗi nonzero trên cùng masks | Chỉ tốt khi thay donor pool/zero policy hoặc chỉ giảm lỗi nhóm zero |
| Q2 | Alpha-Fréchet cải thiện basic JSD KNN? | Thắng ngoài fold tune alpha, ổn định trên locations | Gain biến mất khi alpha chọn train-only; numerical optimizer không ổn định |
| Q3 | ILR giúp diffusion học hình học phù hợp hơn CLR? | Gain vượt kiểm soát roundtrip và giữ được khi kiểm soát noise/preprocessing | Gain chỉ do khác scaler, clipping, capacity hoặc target eligibility |
| Q4 | HKGLR với 5 variants ít missing có ích? | Cải thiện cả khi references có missing/zero và trên all-variant benchmark | Chỉ tốt khi references được bảo vệ; lỗi tăng ở reference-incomplete strata |
| Q5 | Initialization giúp conditional diffusion? | Init branches thắng `no_init` cùng condition encoder | Diffusion không dùng init hoặc gain chỉ đến từ pseudo-label leakage |
| Q6 | Diffusion một vòng cải thiện initialization? | Giảm lỗi tổng và lỗi nonzero, không phá counts/observed; ổn định qua folds | Aggregate gain che khuất deterioration ở nonzero, như pilot hiện tại |

Đây là giả thuyết, không phải kết luận đã có. Không mặc định mô hình phức tạp hơn phải thắng.

### 9.2 Thí nghiệm theo thứ tự phụ thuộc

| Mã | Nội dung | Output cần có trước khi đi tiếp |
|---|---|---|
| E0 | Audit missingness, raw zeros, total/mass semantics và eligibility | Data manifest, split manifest, missingness report, mass protocol |
| E1 | Hron method 2a; basic JSD KNN; alpha-Fréchet JSD; adaptive-k thứ cấp | Unit/numerical checks, donor diagnostics, initializer benchmark độc lập |
| E2 | CLR/ILR/HKGLR forward–inverse trên cùng initialized data | Roundtrip đúng; không có “gain” do transform rồi inverse thuần túy |
| E3 | Lõi CSDI đối chiếu reference trên fixture nhỏ | Parity report, mask/poison tests đạt, sampling finite |
| E4 | Hai initializers chính × ba LR: 6 nhánh diffusion một vòng | Predictions, metrics và config cho cùng fold/mask |
| E4-control | Ba LR với `no_init`; initializer-only; mean, LOCF, linear | Định lượng đóng góp của initialization và diffusion |
| E5 | Ablations có chọn lọc: alpha/adaptive, geometry, positivity, H coverage, clipping, pseudo-label | Chỉ thay một yếu tố mỗi lần; không mở toàn bộ tích Descartes |

Hai initializers chính trong E4 là Hron 2a và **basic JSD KNN**. Alpha-Fréchet/adaptive là nhánh riêng ở E1/E5; nếu validation chọn làm finalist thì giữ thêm basic JSD làm đối chứng, ghi trước quy tắc lựa chọn. Không thay định nghĩa “JSD” giữa các bảng mà không đổi method ID.

**Ràng buộc mass cho ma trận:** sáu nhánh Hron/JSD × LR chỉ so trực tiếp trong benchmark **K**, nơi tổng của D thành phần được cung cấp công khai, như nhau cho mọi phương pháp. Model/condition encoder phải được nhận cùng thông tin tổng đó. Với deployment protocol **U**, bắt đầu bằng Hron × ba LR và controls; JSD nguyên bản không áp dụng nếu chưa định danh missing mass. Nếu thêm JSD có ước lượng scale, đặt method ID khác và báo như extension. Không gộp kết quả K và U trong cùng bảng xếp hạng; không lấy tổng ground truth làm thông tin phụ chỉ cho JSD.

KNN+LTS theo Hron có giá trị làm classical secondary baseline nếu đủ thời gian và implementation kiểm chứng được; không đánh đồng KNN-only với toàn bộ phương pháp tốt nhất của Hron. Không để baseline này trì hoãn việc hoàn tất E0–E4.

### 9.3 Quy tắc lựa chọn và kết luận

Chọn hyperparameters bằng inner validation theo **M2: mean squared Aitchison distance trên eligible affected rows** làm mục tiêu chính của LR pipeline; báo thêm nonzero CLR MAE và native JSD để kiểm tra đánh đổi. Với benchmark initializer native-zero, dùng native JSD làm mục tiêu chính riêng. Không dùng test để chọn `k`, alpha, references, clipping, checkpoint hoặc noise schedule.

Gọi một nhánh “cải thiện nhất quán” khi hướng thay đổi thuận lợi ở đa số outer folds và không che giấu deterioration lớn ở nonzero/rare variants. Luôn báo effect size từng fold; nếu các metric mâu thuẫn, kết luận là trade-off. Với chỉ 5 locations có full ground truth, tránh tuyên bố ưu thế tổng quát hoặc significance mạnh. Một kết quả âm sau kiểm chứng leakage cũng là kết quả nghiên cứu hợp lệ.

## 10. Protocol đánh giá và thống kê

### 10.1 Split theo location, không split ngẫu nhiên từng dòng

Đóng băng outer 5-fold sao cho mỗi fold có một trong 5 complete-data locations làm test; phân bổ 145 locations còn lại cân bằng theo số hàng/missingness bằng quy tắc xác định trước. Train/test locations rời nhau. Inner validation cũng tách theo location; với chỉ 4 complete locations ở outer train, ưu tiên vòng validation nhỏ hoặc leave-one-complete-location-out và báo rõ độ bất định khi tune.

Donor pool, H references, scaler, positivity statistics, empirical training masks và model fitting chỉ dùng outer train; khi tune phải fit lại trên inner train. Có thể dùng các observed values của query tại inference vì đó là thông tin điều kiện hợp lệ, nhưng không nhập test rows vào training donor pool. Không dùng train/test boundary để tạo temporal windows xuyên locations hoặc xuyên partitions.

Giữ thêm pilot fold hiện hành làm regression record. Primary benchmark mới loại 27 zero-total rows khỏi metrics composition không định nghĩa được; vì vậy US có 75 eligible rows thay vì 80 trong pilot cũ. Không so trực tiếp số lỗi mới với pilot rồi nhận là model gain khi evaluation set đã đổi.

### 10.2 Mask scenarios

1. **Random-cell control:** r = 0.10, 0.30, 0.50 trên observed ground truth. Giữ tối thiểu số visible parts theo contract, lưu actual rate sau constraints. Dùng cùng mask files cho mọi phương pháp.
2. **Whole-variant trajectory:** che một số variants trên toàn trajectory/window của location, số lượng và pattern lấy từ train missingness. Đây là scenario chính phù hợp cấu trúc dữ liệu hiện có; có thể báo severity theo realized fraction thay vì gán mọi pattern vào đúng một rate giả tạo.
3. **Contiguous time gap:** nhánh phụ để đo khả năng phục hồi diễn biến thời gian, với gap lengths đăng ký trước và không chọn theo test performance.

Observed mask trong dataset là hằng theo thời gian trong mỗi location. Điều này không đủ để kết luận MCAR, MAR hay MNAR. Random masking là một benchmark có ground truth, không chứng minh mô hình xử lý được cơ chế missing thật.

All-missing query hoặc zero-overlap nên nằm trong stress tests riêng với fallback/abstention được định nghĩa rõ. Không đưa chúng vào main benchmark rồi âm thầm loại khỏi metric khi thuật toán thất bại. Natural missing không có true values: báo diagnostics ở đó, không báo accuracy giả.

### 10.3 Metrics và nhóm phân tích

| Nhóm | Metric/report | Quy tắc |
|---|---|---|
| Log-ratio cell error | CLR MAE trên hidden cells; riêng true nonzero và true raw zero | Mọi phương pháp đánh giá bằng cùng CLR view/policy, không so trực tiếp MAE của ILR với MAE của CLR |
| Compositional row error | M2 = `mean_(i∈affected) d_A²(x_i,x̂_i)` theo handoff | Chấm trên hàng có full truth hợp lệ; observed parts được restore theo cùng protocol; mean distance nếu báo thêm phải mang tên khác |
| Covariance structure | M3 = `‖Cov(CLR(X))−Cov(CLR(X̂))‖_F/(D−1)` theo implementation hiện có | Sample covariance `ddof=1`, cùng full-truth row set, D=17; không thay covariance estimator giữa methods |
| Native zero-compatible | JSD theo convention của Tsagris, ghi có/không square root và log base | Dùng nonnegative closed vectors, tổng dương; không ép zero thành pseudo-count trước metric này |
| Counts/proportions | MAE/RMSE theo variant và pooled; mass residual và scale ratio | Counts chỉ diễn giải khi scale protocol hợp lệ; tránh MAPE tại zero |
| Temporal fidelity | Sai số first differences, trajectory plots; correlation khi đủ variance | Chỉ trên ground truth hợp lệ, báo số điểm; không gọi correlation cao là calibrated uncertainty |
| Reference sensitivity | Errors theo reference-complete/incomplete, số zero references và độ ổn định H qua folds | Cùng strata cho CLR/ILR để so sánh công bằng |
| Rare/nonzero recovery | Errors theo prevalence/abundance bins xác định trên train | Không dùng zero-heavy aggregate để đại diện khả năng phục hồi tín hiệu hiếm |
| Numerical validity | Nonfinite, negatives, fallback rate, clipped fraction, observed error, runtime/memory | Báo trên tất cả queries, gồm cả failures |

Chỉ số diversity nếu bổ sung là mô tả composition, không tự chuyển thành diễn giải dịch tễ/sinh học. Định nghĩa M1/M2/M3 cần version hóa cùng code vì tên metric một mình không đủ tái lập.

### 10.4 Point estimate và uncertainty

Đăng ký point estimate chính là **mean các latent log-ratio samples**, sau đó inverse và observed restoration. Vì CLR, ILR và HKGLR liên hệ tuyến tính trên latent hợp lệ, lấy mean tránh thêm khác biệt do componentwise median không bất biến dưới đổi basis. Median latent là sensitivity analysis riêng, đặc biệt nếu muốn tiếp nối pilot hiện hành; không đổi aggregation theo metric nào đang có lợi.

Đối với uncertainty, inverse từng sample rồi áp cùng mass/observed restoration; tính CRPS, coverage và interval width cho hidden raw/proportion targets theo protocol đã định. Không tính uncertainty bằng cách inverse hai đầu interval LR mà bỏ qua mixing giữa tọa độ. Dùng 5 samples cho smoke/pilot, tối thiểu 50 samples cho các finalists; tăng nếu Monte Carlo error còn đáng kể. Samples=2 hiện có chỉ đủ kiểm tra luồng chạy.

Khoảng dự đoán sau projection/locking là kết quả của pipeline đã chỉnh; phải kiểm tra calibration thực nghiệm. Không lấy việc sample được nhiều lần làm bằng chứng uncertainty đã calibrated.

### 10.5 Đơn vị thống kê và báo cáo

- Báo macro-average theo location song song pooled-cell metrics; chênh lệch cùng method pair phải dùng cùng rows và masks.
- Dùng 3 model seeds cho finalists và 3 fixed mask seeds khi ngân sách cho phép; nhiều masks/dates của một country không trở thành nhiều subjects độc lập.
- Báo mean/SD, từng outer fold và paired differences. Chỉ có 5 complete locations: bootstrap theo location hoặc CI nếu trình bày phải ghi là exploratory và không ổn định; không bootstrap cells để tạo CI hẹp giả.
- Lưu tất cả failed runs và lý do. Loại row/method phải theo quy tắc đã đăng ký, không theo magnitude của lỗi.
- Predictions ở natural missing được cung cấp như estimates có uncertainty/flags, không cộng vào mẫu đánh giá có ground truth.

## 11. Bộ kiểm chứng cần đạt trước benchmark lớn

Các checks dưới đây tập trung vào định nghĩa toán học, leakage và invariants; không viết tests chỉ để lặp lại từng dòng implementation.

| ID | Check | Tiêu chí chấp nhận |
|---|---|---|
| T01 | Input contracts | Phân biệt NaN với raw zero; reject negative/infinite counts; row IDs và feature order ổn định |
| T02 | Hron golden fixture | Tái hiện ví dụ paper với tolerance theo precision số liệu in; donor distances và scale có thể truy vết |
| T03 | Hron scale/subcomposition | Query/donor rescaling và overlapping observed parts cho kết quả theo đúng công thức; fallback được báo |
| T04 | JSD identities | Symmetry, J(p,p)=0, hữu hạn tại zeros, bounds đúng với log base/convention; disjoint support fixture |
| T05 | Tsagris completion | Complete-donor fixture, residual mass và observed locking đúng; không dùng unseen true total ngầm |
| T06 | Alpha-Fréchet | α=1 khớp arithmetic/closure theo định nghĩa; limit α→0 trên strictly positive data khớp geometric center; zeros xử lý đúng miền |
| T07 | Official implementation parity | So các fixtures với CompositionalNAimp cho đúng method/API khi source/runtime sẵn có; ghi version/tolerance; nếu chưa làm ghi chưa verified |
| T08 | Stable KNN ties | Permutation input rows không đổi output khi stable row IDs không đổi; ties, k lớn hơn donors và zero-overlap có policy |
| T09 | CLR roundtrip | Sum-zero, inverse phục hồi composition; thêm constant vào log coordinates không đổi closure |
| T10 | ILR algebra | VᵀV=I, Vᵀ1=0; inverse đúng; distance ILR = Aitchison; kiểm tra sign convention của pivot basis |
| T11 | HKGLR algebra | Mean_H(h)=0; h−mean_all(h)=CLR(x); inverse đúng; H fixed theo fold |
| T12 | Poison hidden targets | Thay hidden values bằng số cực lớn/NaN trên bản ground truth không làm đổi conditioning, initializer hoặc fitted statistics; labels được phép đổi; metadata mass ngoại sinh đã công bố phải giữ cố định trong fixture |
| T13 | Train/test separation | Test-value perturbation không đổi H, scalers, donors, clip bounds, hyperparameters; query visible values chỉ tác động prediction tương ứng |
| T14 | Raw/LR mask dependency | Fixture D=3 hoặc D=4 chứng minh không coi raw observed mask là true CLR/ILR mask; reference bị che không lộ qua HK denominator |
| T15 | CSDI numerical core | Cùng random tensors cho forward/reverse step khớp reference/tính tay; t=1 không thêm noise; loss không dùng natural-missing truth |
| T16 | Sampling health | Finite latent/counts, không có clipping/fallback im lặng; failure trả explicit status và được tổng hợp |
| T17 | Output invariants | Raw observed giữ nguyên kể cả zero; mass residual đúng khi mass biết; positivity view tách raw output |
| T18 | Windows/metrics | Không trộn location/folds, padding bị loại, overlap score once; hand-calculated metrics và raw-zero strata khớp |
| T19 | Roundtrip-only control | CLR/ILR/HKGLR transform→inverse không learning cho cùng composition/error trong tolerance |
| T20 | Experiment provenance | Config, split/mask hashes, data hash, feature/H order, code version, seed, failures và predictions đủ để tái chạy |

Tolerance mặc định cho algebra float64 có thể bắt đầu `atol=1e−10, rtol=1e−8`, điều chỉnh có lý do cho near-zero/extreme ratios. Với training float32 dùng tolerance riêng; không làm lỏng tolerance chỉ để che lỗi. Numerical checks dùng dữ liệu nhỏ xác định trước; smoke training kiểm tra luồng chạy, không thay benchmark accuracy.

## 12. Kế hoạch implementation trong repository

Các tên module dưới đây là **đề xuất**, chưa phải các file đã được tạo trong task lập kế hoạch này. Trước khi code phải đối chiếu cấu trúc hiện tại để mở rộng/reuse thay vì tạo hai implementation song song.

```text
src/core/                 LR transforms, inverse, basis, positivity/mass contracts
src/initialization/       Hron2a, JSD basic, alpha-Fréchet, donor policies
src/stage_b/              condition builder, LR target adapter, CSDI integration
src/evaluation/           masks, folds, metrics, uncertainty, provenance
tests/                    fixtures toán học, leakage, integration
configs/research_lr/      frozen method definitions và pilot/final configs
artifacts/research_lr/    run manifests, predictions, diagnostics, reports
```

API khái niệm:

```python
initializer.fit(train_data, train_mask, row_ids, config)
initialized, diagnostics = initializer.transform(query, visible_mask, metadata)
transform.fit(train_positive_data, feature_order, references, config)
latent = transform.forward(positive_composition)
composition = transform.inverse(latent)
condition = build_condition(raw_visible, visible_mask, initialized, provenance)
samples = model.sample(condition, transform_spec, n_samples, seed)
output = restore_observed_and_mass(samples, raw_visible, visible_mask, mass_spec)
```

`transform.fit` chỉ chọn H, basis/scaler nếu cần; không được tự học từ test. `transform_spec` lưu feature order, reference set, basis/sign, policy zeros và scaler. `diagnostics` gồm donors, overlap, fallback, known/unknown mass và failure status. Không truyền full hidden truth vào object initializer/condition builder để rồi trông chờ hàm con không dùng nó.

Thứ tự thay đổi nhỏ có thể review:

1. Data contracts, fold/mask manifests và metric fixtures.
2. Audit/sửa Hron theo paper; thêm basic JSD và parity fixtures.
3. ILR/HKGLR + roundtrip/algebra tests, giữ API CLR hiện có khi có thể.
4. Condition builder và leakage tests; tách raw mask khỏi LR representation.
5. CSDI core parity, rồi adapter CLR; chỉ sau khi đạt gate mới mở ILR/HKGLR.
6. Pilot 6 branches và baselines, diagnostics/report tự động.
7. Alpha/adaptive và ablations có chọn lọc, rồi final benchmark.

Không ghi đè artifact pilot hiện có. Dùng run ID mới, config tự mô tả và bảng so sánh chỉ ghép các runs có cùng evaluation signature. Official source của `metag_time_impute_phylo` sẽ được chốt snapshot để đối chiếu, giữ provenance riêng với code của dự án; tình trạng thiếu thư mục địa phương đã ghi ở mục 2.

## 13. Lộ trình, ngân sách và các điểm dừng kiểm tra

| Giai đoạn | Việc chính | Ước lượng công nghiên cứu/lập trình | Gate |
|---|---|---|---|
| P0 | Chốt data/mass contract, split/mask và audit eligibility | 1–2 ngày | Không còn total/missing/zero semantics bị dùng ngầm |
| P1 | Hron/JSD basic, alpha-Fréchet và kiểm chứng initializer | 3–4 ngày | T01–T08; initializer report có failures/donors |
| P2 | ILR/HKGLR, positivity và mask architecture | 2–3 ngày | T09–T14, T19; no leakage ở fixtures |
| P3 | CSDI reference core và adapter | 3–4 ngày | T15–T18, T20; finite smoke run |
| P4 | Pilot một fold, một severity, một seed; 6 branches + controls | 2–3 ngày | Có metrics/diagnostics, không có hidden test tuning |
| P5 | Outer CV, seeds và ablations của finalists | 5–8 ngày | Frozen configs, cùng masks, đủ negative/failed results |
| P6 | Tổng hợp, limitations và cập nhật handoff | 2 ngày | Báo cáo tái lập được, kết luận đúng phạm vi dữ liệu |

Tổng khoảng **18–26 ngày công**, chưa phải cam kết thời gian chạy máy. Các bước có thể rút ngắn khi reuse được implementation đã verified. Cần đo pilot trên phần cứng thật trước khi ước lượng GPU/CPU hours; runtime 157 giây của pilot nhỏ hiện có không đủ ngoại suy tuyến tính sang mọi cấu hình.

Ví dụ chi phí grid: `6 branches × 5 folds × 3 severities × 3 model seeds = 270 fits` cho một mask family, nếu mỗi severity huấn luyện riêng. Hai families thành 540 fits, chưa gồm controls/tuning. Nếu model huấn luyện với hỗn hợp severities rồi chỉ thay test masks thì số fits giảm; phải ghi rõ protocol này. Không tự chạy toàn grid ngay. Gate P4 chọn kiến trúc khả thi và finalists bằng validation trước khi mở rộng.

Giữ số epoch/steps/channels nhỏ cho smoke; checkpoint selection và early stopping chỉ theo validation. Thất bại ở leakage/algebra gate thì sửa implementation trước, không tăng model capacity để che lỗi.

## 14. Quan hệ với handoff và việc cần làm tiếp

Handoff `pipeline_knn_diffusion_handover.md` vẫn là nền cho pipeline hiện có: phân biệt missing với zero, tránh leakage, bảo toàn observed và đánh giá có ground truth. Yêu cầu mới của người dùng mở rộng nghiên cứu sang JSD, ILR và HKGLR; bản kế hoạch này ghi rõ các nhánh mở rộng thay vì giả định handoff cũ đã mô tả chúng. Ràng buộc **một outer pass, không lặp refinement** được giữ xuyên suốt.

Các quyết định chưa thể kết luận chỉ từ papers:

- 17 variants có loại trừ nhau và bao phủ toàn bộ `total_sequence` hay không; cần metadata/định nghĩa nguồn dữ liệu. Trong lúc chưa xác minh, dùng unknown-mass protocol, không lấy total làm residual budget chính.
- Dataset có đủ complete-label temporal windows để học LR diffusion hay không; xác định sau split và báo số lượng. Nếu quá ít, giảm mô hình hoặc báo giới hạn thay vì biến imputed values thành ground truth ngầm.
- Hron/JSD và HK references có đủ donor/reference coverage ở từng fold hay không; đo trực tiếp và báo failure/fallback.
- Ưu thế của HKGLR ở scRNA-seq có chuyển sang variants hay không; chỉ thí nghiệm mới trả lời được. Bộ 5 variants theo missing rate là quy tắc kỹ thuật của dự án, không phải kết luận sinh học.

**Công việc kế tiếp ưu tiên:** đóng băng data/fold/mask manifest → kiểm chứng Hron và basic JSD → roundtrip ILR/HKGLR → sửa conditioning/loss contract → parity CSDI → pilot 6 branches một vòng → đánh giá ngay cùng baselines. Không chạy lại bước đã có chỉ để lấy lại cùng artifact; chỉ rerun khi implementation, protocol hoặc evaluation signature thực sự thay đổi và nêu rõ lý do.

**Phạm vi hoàn thành của tài liệu này:** đã đọc và tổng hợp lý thuyết, đối chiếu code/artifacts hiện có, audit dữ liệu và lập protocol. Các thuật toán mới, parity R, ma trận CV và kết quả ILR/HKGLR/JSD nêu ở trên là việc sẽ thực hiện; chưa được coi là đã implement hoặc đã chứng minh tốt hơn.
