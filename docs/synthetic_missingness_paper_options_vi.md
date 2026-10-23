# Các lựa chọn gây thiếu có căn cứ paper và mã tác giả

Ngày nghiên cứu: 2026-10-23.
Trạng thái: tài liệu lựa chọn, chưa chốt phương án và chưa triển khai generator.
Kiểm toán bổ sung theo skill Claim → Evidences → Analysis: xem synthetic_missingness_claim_evidence_audit_vi.md và docs/evidence/synthetic_missingness_sources/. Audit này là nguồn định vị/phiên bản ưu tiên; các công thức suy dẫn và ánh xạ không phải phương trình nguyên bản của paper.
Thay thế vai trò đề xuất công thức của synthetic_missingness_spec_vi.md; tài liệu cũ không được dùng làm đặc tả đã duyệt.

## 1. Phạm vi và mức bằng chứng

Thư mục nguồn local: C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/research.
Đã trích text các PDF trong thư mục để tìm vị trí liên quan; đọc kỹ các phần nêu dưới và kiểm tra hình ảnh trang quan trọng. File Missing_imputation_Diffusion_idea_plot.pdf chỉ có một trang gần như không trích được text, không dùng làm bằng chứng paper. Các file lâm sàng, conformal và temperature scaling không phải nguồn công thức masking được dùng trong tài liệu này.

Phân biệt ba mức:
- **Nguồn quy định:** nội dung kiểm tra được trong paper/mã tác giả.
- **Diễn giải toán học:** viết lại chính quy trình nguồn bằng ký hiệu thống nhất, không gán số phương trình giả cho paper.
- **Ánh xạ vào benchmark:** thay biến nguồn bằng N, trait hoặc p và quy định pattern của dữ liệu đang thiết kế. Đây là phần tùy biến cần người dùng chọn; không gọi là tái lập nguyên xi.

Không nguồn local nào đã kiểm chứng cung cấp nguyên bộ: MAR theo N+trait, MNAR theo latent p có structural zeros, counts lấy mẫu, ba dạng mask với độ dài/số component/rate độc lập. Vì vậy phải chọn giữa tái lập một phần nguồn local và bổ sung phương pháp amputation ngoài folder. Các nguồn bổ sung được ghi rõ bên dưới.

## 2. Những gì paper local thực sự quy định

| Nguồn | Vị trí đã đọc | Quy trình gây thiếu thực tế | Không được suy ra thêm |
|---|---|---|---|
| Hron.pdf | Section 5, PDF p10–12 | MCAR theo biến; ví dụ D=3: 20% biến 1, 10% biến 2, 0% biến 3; chỉ gây thiếu ở nhóm không outlier. D=10 có các cấu hình tương quan và rates theo biến khác. | Câu đã thử MAR không kèm công thức MAR để tái lập; không có đặc tả MNAR/khối thời gian ở thí nghiệm này. |
| tsagris.pdf | Section 2.1 p4; Sections 3.1–3.4 p11–14; conclusion p15 | Định nghĩa MCAR/MAR; accuracy: chọn 10% hàng rồi che nửa component, lặp 500 lần. Adaptive: chia theo thành phần đầu và gán hai patterns khác nhau. Dirichlet dùng đo tốc độ. | Định nghĩa MAR không phải thuật toán sinh MAR. Adaptive chưa đủ thông tin để tự gán nhãn MAR nếu không kiểm tra biến điều khiển còn observed; conclusion nói chưa đánh giá MNAR. |
| Sasha Micoda.pdf | Chapter 8, Section 8.3, printed p162–167 / PDF p168–173; Section 8.4.2 PDF p181–182 | V1: xác suất tau/3 cho thiếu 1,2,3 hoạt động; tau=.3/.5/.8 là xác suất hàng thiếu, không phải cell rate. V2: PLN và rarefaction 100 reads, khoảng 27% sampling zeros được xử lý như NA trong ví dụ. | Abstract nói missingness associated with covariates không đủ để bịa một logistic MAR recipe cho Section 8.3. Rarefaction không phải cơ chế mask MAR/MNAR theo p như yêu cầu. |
| csdi_2023.pdf | PDF p2 / printed p915; p3 | Số timepoints thiếu theo mỗi subject được lấy từ Binomial; chọn timepoints ngẫu nhiên; đánh giá r=.1.. .9. | Không có MAR/MNAR logistic, không có block lengths độc lập trong phần mô tả này. |
| csdi.pdf (Seki et al. 2025) | Section 3.1, PDF p4 | MCAR: bỏ timepoints ngẫu nhiên. MNAR: chọn timepoints có alpha diversity cao. | Main text không cho hàm xác suất, ngưỡng, weighted sampling hay cách tie-breaking đầy đủ. Không được biến thành -log(p+epsilon), cũng không suy ra chắc chắn là top-K entropy. |
| 2021_CSDI -- Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation.pdf | Section 4.3 p5–6; Appendix B.2 Algorithms 3–4 p14 | Random target choice; historical mask transfer; mix 1:1. Random r Uniform(0,100) dùng trong self-supervised TRAINING. | Đây không phải đặc tả test MAR/MNAR đầy đủ. Không đem training target strategy thay cho benchmark missingness mechanism. |
| CoDA-hd 2025 PDF | Methods, PDF p7; reference 30–31 p20 | Dẫn Splatter/SplatPop cho dữ liệu mô phỏng. | Không có trực tiếp công thức MAR/MNAR cho benchmark hiện tại. Dropout tạo zero khác với che counts bằng NA. |

### 2.1 Định nghĩa, không phải generator

Tsagris Eq. (2): P(R=r | X=x)=P(R=r) cho MCAR.
Tsagris Eq. (3): P(R=r | X=x)=P(R=r | X_obs=x_obs) cho MAR.
R=1 là observed trong paper; trong tài liệu này H=1 là hidden. Phải đổi quy ước rõ khi hiện thực.

### 2.2 Viết lại đúng các recipe MCAR local

**MICoDa V1:** K thuộc {0,1,2,3}, với P(K=0)=1-tau và P(K=k)=tau/3 cho k=1,2,3; chọn K component theo quy trình ngẫu nhiên của nguồn. E[K]=2*tau. Nếu K được chọn đồng đều trên D target components thì cell rate kỳ vọng =2*tau/D. D=6 của thí nghiệm nguồn khác với D=12 đang thiết kế; tau=.8 không phải 80% cell missing.

**Tsagris accuracy:** chọn rho=.1 số hàng và q=D/2 component trên mỗi hàng được chọn. Cell rate xấp xỉ rho*q/D=.05, không phải .10. Chuyển rho hoặc q là thay tham số thí nghiệm, không giữ nguyên recipe số học của paper.

**Seki 2023:** diễn giải quy trình: K_i~Binomial(T_i,r), S_i chọn đều trong các tập timepoints có |S_i|=K_i; H_itj=1[t thuộc S_i] cho mọi j. Điều này tương đương Bernoulli MCAR r trên timepoints; không bảo đảm K_i=r*T_i mỗi subject và không tạo các đoạn liên tiếp theo L.

**CSDI historical:** H_new = O_new AND H_reference, reference được lấy từ training missing-pattern dataset, theo Algorithm 4. Pattern transfer không tự định nghĩa được MAR hay MNAR theo giá trị mới.

### 2.3 MNAR alpha-diversity local

Shannon entropy thường viết D_it=-sum_j q_itj*log(q_itj), với 0*log0=0. Seki nói alpha diversity assessed using Shannon index, nhưng main text không cho chi tiết chọn timepoints có diversity cao. Không gọi H=topK(D) là code gốc khi chưa xác minh mask-generation code/supplement.

Tình trạng truy cập: repo tác giả đã xác định là https://github.com/misatoseki/metag_time_impute_phylo ; README có mask inputs. Chưa xác minh được script sinh MNAR trong lượt nghiên cứu này. Đường link nguồn không phải bằng chứng rằng đã đọc được thuật toán sinh mask.

## 3. Option A: tái lập recipe local làm reference

**Hiện thực được ngay về mặt đặc tả:** Hron/Tsagris/MICoDa MCAR và Seki2023 whole-timepoint MCAR bằng các quy trình 2.2.
**p=0:** không tác động lựa chọn ngẫu nhiên; không thêm epsilon hay biến zero thành missing tự động.
**Hiệu chỉnh:** dùng đúng mẫu số của nguồn (theo hàng/timepoint/component) rồi báo cell rate quy đổi và thực đạt.
**Giới hạn:** không đáp ứng đủ MAR+rare-p MNAR hoặc temporal blocks yêu cầu. Seki2025 MNAR chỉ có mô tả đủ xác định hướng chọn, chưa đủ để cam kết tái lập công thức.

## 4. Option B: multivariate amputation liên tục, Schouten et al. (2018)

Đây là nguồn bổ sung ngoài folder, không phải công thức của Hron/MICoDa/Tsagris.

Paper: Section 3.1–3.2, Figure 3 và Table 1.
https://doi.org/10.1080/00949655.2018.1491577
Full text tác giả: https://rianneschouten.github.io/mice_ampute/manuscript/manuscript.pdf
Implementation/docs: https://amices.org/mice/reference/ampute.html
Continuous source: https://github.com/amices/mice/blob/HEAD/R/ampute.continuous.R
Readable source mirror: https://rdrr.io/cran/mice/src/R/ampute.continuous.R
Score source: https://rdrr.io/cran/mice/src/R/ampute.R (sumscores).

### 4.1 Công thức nguồn và sampling

Mỗi case i được gán NGẪU NHIÊN một candidate pattern k theo freq_k. Pattern xác định một tập biến bị che đồng thời nếu case được chọn. Không phải tự sinh Bernoulli độc lập từng ô rồi mở rộng mask.

    wss_ik = sum_v w_kv*y_iv

Đây là công thức explicit trong Section 3.1. Implementation chuẩn hóa score theo candidate pattern khi có biến thiên. Gọi score sau chuẩn hóa là s và mean(s)=m. Dùng sigma(x)=1/(1+exp(-x)). Các hàm trong mã tác giả:

    RIGHT: pi(s)=sigma(s-m+b)
    LEFT:  pi(s)=sigma(m-s+b)
    MID:   pi(s)=sigma(-abs(s-m)+.75+b)
    TAIL:  pi(s)=sigma(abs(s-m)-.75+b)

Số .75 lấy từ code, không phải tham số tự đặt cho dataset này. Sau đó draw Bernoulli trên candidate case và che các biến có pattern=0.

### 4.2 Ánh xạ theo yêu cầu người dùng (tùy biến, chưa chốt)

- Sinh mask trên một bảng dùng proportions p làm target-values cho mask generator, cộng các covariates N,a luôn observed. Cuối cùng chỉ lấy mask áp lên counts X; không đưa bảng p cho baseline.
- MAR: weights của p targets=0; chỉ N và a có weights khác 0. Ví dụ weights N âm, a dương với RIGHT để N thấp/trait cao có missingness cao. Dùng raw N hay một transform phải được chọn rõ; paper không quy định log N cho dataset này.
- Rare-p MNAR: LEFT với weight dương trên p của component sẽ bị che; weights N,a=0 nếu muốn cơ chế chỉ dựa trên p. Với một component, s xuất phát trực tiếp từ p; với nhiều component đồng thời, các weights trên từng p là thiết kế pattern score, không mặc nhiên dùng mean log p.
- p=0 hợp lệ trực tiếp: linear score hữu hạn và sigmoid hữu hạn; không cần epsilon. Nhiều p=0 giống nhau nhận cùng score nếu các biến điều khiển khác giống nhau; không phá ties để tạo tín hiệu giả.
- MCAR: chọn nhánh MCAR của ampute, cùng pattern bank.

### 4.3 Rate và độ mạnh: khác biệt quan trọng với đề xuất cũ

Implementation continuous đã đọc tạo 10000 draws Normal(0,1), chuẩn hóa chúng, rồi binary search b để trung bình pi trên bộ THAM CHIẾU này đạt prop. Không phải empirical root trên chính tất cả p của evaluation panel. Với distribution nhiều mass tại zero, rate thực tế có thể khác mục tiêu; phải báo nó.

Mặc định prop của ampute theo CASES; bycases=FALSE đổi sang cells của toàn bộ input matrix. Nếu input có helper N,a, tỷ lệ target-only cần quy đổi mẫu số, không đặt prop=r một cách máy móc.

sumscores chuẩn hóa điểm kể cả khi std=FALSE (std điều khiển chuẩn hóa input). Vì vậy nhân tất cả weights với cùng hệ số dương .5/2 KHÔNG tạo độ mạnh mới. Bản gốc không có đối số beta với nghĩa như đề xuất cũ. Nếu muốn beta-slope grid phải gọi là extension được duyệt riêng.

Paper Table 1 có weak MAR bằng cách chia thành phần MCAR và MAR; đó là một căn cứ để lựa chọn mức pha trộn thay cho việc giả nhận beta là tham số của ampute. Định lượng mixture levels mới ngoài thí nghiệm của nguồn vẫn phải ghi là cấu hình benchmark.

**Hiện thực:** dùng ampute gốc có version lock để tạo masks, hoặc port từng công đoạn và đối chiếu output/probabilities với R. Không gọi baseline mice để điền khuyết; chỉ dùng công cụ sinh mask. Chưa kiểm runtime R ở lượt nghiên cứu này.

## 5. Option C: amputation phân nhóm theo score, Brand/mice::ampute.discrete

Nguồn bổ sung ngoài folder; được ampute phân biệt với nhánh continuous. Nền tảng dẫn Brand (1999), được triển khai trong mice; không nói công thức này nằm trong ba paper baseline gốc.
Docs: https://amices.org/mice/reference/ampute.discrete.html
Source: https://rdrr.io/cran/mice/src/R/ampute.discrete.R

Giữ cơ chế candidate patterns và weighted sum scores như B. Chia score thành G nhóm phân vị. Với trọng số xác suất tương đối d_g, công thức đúng từ code:

    pi_g = G * prop * d_g / sum_h d_h

Ví dụ d=(1,2,3,4) là mặc định package, là relative probabilities, KHÔNG phải logistic odds ratio. Đảo thứ tự weights nếu score nhỏ phải dễ thiếu hơn. MAR score chỉ từ N,a; MNAR score từ p kín; MCAR dùng equal probabilities hoặc nhánh MCAR riêng.

**Độ mạnh:** thay tỷ số d_g giữa nhóm. Đây là control có thật trong API, khác với nhân toàn bộ weights score cùng hằng số.

**p=0 và ties:** không có log nên zero hợp lệ về miền giá trị, nhưng nhiều zeros tạo quantile cutpoints trùng nhau. Code dừng khi không tách được quantile groups. Không tự jitter p hay phá ties để tiếp tục. Phải chọn ít nhóm hơn hoặc chọn option khác và ghi trước; xử lý tie tùy biến là extension.

**Rate:** công thức nhắm prop khi các nhóm có kích thước bằng nhau. Nếu pi_g>=1, code cảnh báo; saturation hoặc nhóm không cân có thể làm rate khác mục tiêu. Không tự clip rồi tuyên bố vẫn giữ đúng rate. Muốn calibration theo group masses thực hoặc capped normalization là thay đổi thuật toán nguồn.

**Hiện thực:** dùng cont=FALSE và odds matrix đã đăng ký; cùng ánh xạ pattern/target denominator như B. Không chạy được mọi tổ hợp odds/rate nếu vi phạm miền khả thi.

## 6. Option D: quantile-censoring MNAR, Muzellec et al. (ICML 2020)

Nguồn bổ sung ngoài folder.
Paper: Section 4, PDF p7, đoạn Missing value generation mechanisms.
https://proceedings.mlr.press/v119/muzellec20a.html
https://proceedings.mlr.press/v119/muzellec20a/muzellec20a.pdf
Code: https://github.com/BorisMuzellec/MissingDataOT/blob/master/utils.py
Raw: https://raw.githubusercontent.com/BorisMuzellec/MissingDataOT/master/utils.py
Hàm MNAR_mask_quantiles; bản audit mới đã pin commit 6417eacfde1c1052a63568350dfec2d0373ac056, dòng 314–381. Số dòng 282–344 trước đây thuộc bản web đã đọc, không dùng thay locator của snapshot mới.

### 6.1 Công thức viết lại từ code

Code hỗ trợ cut=lower/upper/both; paper mô tả thí nghiệm hai đuôi. Một đuôi dưới là nhánh trong code, không phải nguyên cấu hình hai đuôi được paper mô tả.

    A_ij = 1[x_ij <= Q_j(q)]             (cut='lower')
    H^Q_ij = A_ij * Bernoulli(rho)

Áp vào latent p thay x là tùy biến theo yêu cầu người dùng; Q_j tính trên p của component j. Nếu không thêm cơ chế nền, chỉ low-tail có thể thiếu: không đáp ứng điều kiện mọi ô có xác suất thiếu dương.

Code có MCAR=True, tạo thêm mask và lấy OR:

    H_ij = H^Q_ij OR Bernoulli(rho)

Hai random draws độc lập. Từ đúng phép OR này suy ra (không phải phương trình đánh số trong paper):

    P(H=1 | A=0)=rho
    P(H=1 | A=1)=2*rho-rho^2

Vì vậy option D có thể cho mọi ô một xác suất nền, đồng thời ưu tiên phần đuôi hiếm. Chọn p_params=1 nếu muốn mọi component được xét nhánh quantile.

### 6.2 Rate và p=0

Gọi f là tỷ lệ ô thực sự nằm trong low-tail, bao gồm ties tại threshold. Không được tự coi f=q khi có nhiều zeros.

- Không MCAR: expected cell rate =rho*f; rate mục tiêu r chỉ khả thi nếu r<=f, đặt rho=r/f là quy đổi đại số từ cơ chế, không phải code calibration gốc.
- Có MCAR=True: expected rate =rho+(1-rho)*rho*f. Tìm rho trong [0,1] từ phương trình này là một bước calibration theo yêu cầu benchmark, phải ghi rõ là bổ sung, không nói tác giả tự đặt rho như vậy.
- p=0: threshold comparison hợp lệ, không epsilon. Các zeros bằng nhau cùng thuộc nhóm low-tail; f có thể lớn hơn q. Không random tie-break để ép f=q trừ khi chọn extension riêng.

**Giới hạn:** không có logistic slope beta; q và MCAR nền thay đổi dạng/độ tập trung missingness. MAR vẫn phải lấy từ một phương án khác. Code nguyên bản không sinh temporal blocks L,q; mở rộng lên khối cần một thiết kế có căn cứ riêng, không dùng phép mean rarity cũ rồi gọi là paper gốc.

## 7. Ghép missingness với temporal patterns: giới hạn phải giữ rõ

B/C có pattern matrix, nên có thể biểu diễn time-component grid của một entity thành vector và dùng pattern chứa một hoặc nhiều khối. Chọn pattern bank, số blocks và vị trí cụ thể là cấu hình benchmark của người dùng trong framework paper, không có bộ mặc định địa điểm×120×12 trong paper.

Một pattern chỉ che 1 khối q components x L steps trên một entity sẽ che q*L/(12*120) phần bảng của entity nếu được kích hoạt. Vì activation probability<=1, không thể hiệu chỉnh rate vượt mức này. Ví dụ q=3,L=12 => tối đa 2.5% toàn bộ target cells trên entity cho một block duy nhất.

Muốn 10/30/50% cần patterns chứa nhiều khối hoặc đổi định nghĩa rate. Một lựa chọn không được tự chốt: dùng pattern bank nhiều khối với tổng missing-cell count cố định, kích hoạt theo cơ chế nguồn; một lựa chọn khác: đổi grid để giữ một block/entity. Khả năng ghép/đụng khối và q là per-block hay per-time maximum phải được người dùng chọn, không tự hợp thức hóa bằng partition proposal cũ.

Seki2023 giúp làm reference whole-timepoint MCAR. CSDI2021 historical giúp lấy patterns từ train nhưng không sinh ra rare-p MNAR cho panel mới. Không có source đã kiểm chứng trong folder đảm bảo cả exact run length lẫn exact rate cho mọi grid combination đã bàn.

## 8. Kết luận để người dùng lựa chọn

- A: sát recipe local nhất, thích hợp reference MCAR; không đủ bài toán MAR/MNAR hiện tại.
- B: framework paper rõ cho MAR/MNAR và multivariate patterns; xử lý p=0 trực tiếp; phải bỏ hoặc thay diễn giải beta grid cũ và chấp nhận rate calibration nguyên bản/extension ghi rõ.
- C: cùng framework nhưng xác suất rời rạc, có độ mạnh bằng relative group probabilities; khó với nhiều ties/structural zeros.
- D: MNAR đuôi thấp rõ ràng, không epsilon, có thể có MCAR nền để mọi ô có cơ hội thiếu; cần ghép MAR và thiết kế temporal riêng.

Không phương án nào đã được người dùng chọn ở lượt nghiên cứu này. Các thông số tạo dữ liệu/quy mô đã chốt không bị tự ý thay đổi. Những phần masking chưa có nguồn hoặc chưa tương thích lưới vẫn để mở.

## 9. Ghi chú kiểm toán code ngoài phạm vi lựa chọn

MissingDataOT có thêm MNAR_self_mask_logistic, nhưng đoạn fit_intercepts(self_mask=True) trong snapshot đã đọc sử dụng mean trên cả X thay vì riêng cột j. Không dùng nó làm bằng chứng đã bảo đảm đúng marginal rate từng feature, và không tự sửa rồi gọi là tái lập code tác giả. Option D ở trên dựa vào MNAR_mask_quantiles, không dựa vào helper này.
