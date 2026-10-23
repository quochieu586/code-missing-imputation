# Kiểm chứng missingness: Claim → Evidences → Analysis

Ngày: 2026-10-23. Phạm vi: kiểm tra các phát biểu/công thức đã dùng khi đề xuất masking cho dữ liệu synthetic 60 thực thể × 120 thời điểm × 12 component, có latent p, counts X, N và trait quan sát. Chỉ đọc các phần liên quan, không tuyên bố đã đọc toàn văn mọi tài liệu. Đã xem ảnh các trang có công thức/đoạn then chốt. Chưa chạy generator, chưa chọn option và chưa kiểm thử thực nghiệm các phương án.

## Nguồn và phiên bản

| Mã | Nguồn đã đọc | Phiên bản và phạm vi |
|---|---|---|
| H | [Hron, Templ, Filzmoser](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/research/Hron.pdf) — Imputation of missing values for compositional data using classical and robust methods | File local 17 trang; footer trang 1 ghi preprint 27/11/2009. Không gọi nó là đúng bản dàn trang xuất bản 2010. Section 5, PDF/in trang 10–12. |
| T | [Tsagris, Stewart, Alenazi](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/research/tsagris.pdf) — A Jensen-Shannon divergence based k–NN algorithm for missing value imputation in compositional data | Preprint local 18 trang; trang 1 đề ngày 29/05/2026, arXiv:2605.29702v1. Sections 2.1, 3.1.1, 4. |
| M | [Saha et al., Chapter 8](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/research/Sasha%20Micoda.pdf) — Multiple Imputation for Compositional Data (MICoDa) Adjusting for Covariates | Sách local 280 trang; chương bắt đầu PDF 163 / in 157; footer xác nhận 2023, DOI 10.1007/978-3-031-32729-2_8. Section 8.3, PDF 171 / in 165. |
| S | [Seki, Zhang, Imoto](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/research/csdi.pdf) — Diffusion model for imputing time-series gut microbiome profiles using phylogenetic information and metadata integration | Bản local 9 trang, Bioinformatics Advances, 2025, vbaf181, DOI 10.1093/bioadv/vbaf181; Section 3.1, PDF/in 4. |
| A | [Schouten, Lugtig, Vink (2018)](https://rianneschouten.github.io/mice_ampute/manuscript/manuscript.pdf) — Generating missing values for simulation purposes: a multivariate amputation procedure | Bản tác giả 23 trang có trang bìa; DOI 10.1080/00949655.2018.1491577. Section 3.1: PDF 6 / in 5, PDF 8 / in 7, PDF 9 / in 8. Nguồn ngoài folder. |
| O | [Muzellec, Josse, Boyer, Cuturi (2020)](https://proceedings.mlr.press/v119/muzellec20a/muzellec20a.pdf) — Missing Data Imputation using Optimal Transport | Bản proceedings ICML/PMLR 119, PDF 11 trang. Section 4, PDF 7, đoạn Missing value generation mechanisms. Nguồn ngoài folder. |
| AC | [Mã mice được pin](https://github.com/amices/mice/tree/61083e667fa41cc4bc655492591b100bf8a7e443/R) | Commit 61083e667fa41cc4bc655492591b100bf8a7e443; ampute.R, ampute.continuous.R, ampute.discrete.R. Đây là evidence code, không phải câu trích trong paper A. |
| OC | [Mã MissingDataOT được pin](https://github.com/BorisMuzellec/MissingDataOT/blob/6417eacfde1c1052a63568350dfec2d0373ac056/utils.py) | Commit 6417eacfde1c1052a63568350dfec2d0373ac056; utils.py. Đây là evidence code, không phải phương trình đánh số trong paper O. |

Snapshot mã, bản trích xuất từng trang, SHA256 và ledger ở [thư mục evidence](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/docs/evidence/synthetic_missingness_sources). Trang PDF dùng chỉ số 1-based. Các số dòng bên dưới là dòng **snapshot mã nguồn**, không phải dòng paper. Bản trích xuất PDF chỉ hỗ trợ tìm kiếm; công thức và đoạn quan trọng đã đối chiếu ảnh trang.

## C1 — Hron có mô tả MCAR, nhưng đoạn nhắc MAR chưa đủ để tái lập MAR

**Claim — tác giả phát biểu / mô tả thí nghiệm:** Section 5 quy định MCAR theo biến và nói đã thử MAR; trong phần đã đọc không có một công thức xác suất MAR đủ để lấy làm đặc tả.

**Evidences:**
- **E01 — H; PDF/in 10; Section 5; đoạn bắt đầu “The amount of missing values”:** “set to be missing completely at random.” Ngữ cảnh: 20% biến thứ nhất, 10% biến thứ hai; chỉ gây thiếu trên nhóm không outlier. Hỗ trợ mô tả MCAR, không phải logistic MAR.
- **E02 — H; PDF/in 12; Section 5; đoạn cuối bắt đầu “Finally, in our simulation”:** “We also tested the MAR situation, leading to analogous conclusions.” Hỗ trợ việc tác giả nói có thử MAR, không cung cấp riêng recipe sinh MAR.

**Analysis:** Được hỗ trợ trong phạm vi E01–E02. Cần thu hẹp phát biểu âm: chưa tìm được recipe MAR trong đoạn/section đã kiểm, không kết luận rằng không có trong mọi code/supplement liên quan. Áp dụng: có thể giữ thí nghiệm MCAR làm reference; không gán các công thức -log(p+epsilon), N+trait hoặc beta của agent cho Hron. Chi tiết random selection/rounding vẫn phải lấy từ code nếu yêu cầu paper parity.

## C2 — Hai phương trình Tsagris định nghĩa MCAR/MAR, không phải thuật toán tạo mask

**Claim — định nghĩa của tác giả:** T, Section 2.1 đưa điều kiện MCAR/MAR theo phân phối missingness.

**Evidences:**
- **E03 — T; PDF/in 4; Definition 2.2, Eq. (3); neo “The missing data are MAR if:”:** “The missing data are MAR if:” Công thức gốc, đã kiểm trên ảnh trang:

  Eq. (2): P(R=r | X=x)=P(R=r).

  Eq. (3): P(R=r | X=x)=P(R=r | X^o=x^o).

  Ngay trên đó R_j=1 nghĩa observed, R_j=0 nghĩa missing; X^o được định nghĩa là observed components. Hỗ trợ định nghĩa/phạm vi, không hỗ trợ một hàm logistic cụ thể.

**Analysis:** Được hỗ trợ bởi E03. Nếu dùng H=1 nghĩa missing trong code thì cần đổi quy ước. Một công thức thỏa định nghĩa không mặc nhiên là công thức tác giả đã dùng. Các ràng buộc composition và thông tin baseline quan sát phải được xét khi gắn nhãn cơ chế; latent-p masking của mình chưa được chứng minh chỉ bằng Eq. (3).

## C3 — Tsagris che hàng/component trong thí nghiệm chính và không đánh giá MNAR ở bản này

**Claim — mô tả thí nghiệm và giới hạn tác giả:** Section 3.1.1 chọn 10% vectors rồi che nửa components; conclusion giới hạn việc đánh giá MNAR.

**Evidences:**
- **E04 — T; PDF/in 11; Section 3.1.1; đoạn đầu:** “10% of the compositional vectors were randomly selected”. Phần tiếp theo của cùng câu nói chọn nửa components của mỗi vector và thay bằng NA. Hỗ trợ recipe chọn hàng rồi thành phần; không phải 10% tổng ô.
- **E05 — T; PDF/in 15; Section 4; đoạn cuối trước Acknowledgments:** “we have not assessed our algorithm under these circumstances.” Điều kiện trong cùng câu là các observations MNAR ở official statistics; giữ ngữ cảnh này khi diễn giải.

**Analysis:** Được hỗ trợ bởi E04–E05. Nếu nửa components được chọn đúng bằng D/2 thì suy dẫn agent là cell rate .1*.5=.05; với D lẻ cần quy tắc rounding, không giả định 5% chính xác. Không dùng phần định nghĩa/diagnostics để gọi đây là recipe MNAR hiếm theo p.

## C4 — MICoDa dùng tau cho tỷ lệ cá thể có thiếu, không phải tỷ lệ ô

**Claim — mô tả thí nghiệm:** Section 8.3 nói mỗi subject có xác suất tau/3 cho từng trường hợp thiếu 1,2,3 activities, với tổng xác suất tau.

**Evidences:**
- **E06 — M; PDF 171 / in 165; Section 8.3; mục “Generation of Missing Data”:** “one activity or two activities or three activities”. Đọc cả đoạn: câu trước định nghĩa tau là proportion of individuals; cùng câu quy định xác suất tau/3; đoạn tiếp theo cho tau=.3,.5,.8. Hỗ trợ diễn giải dưới đây, không phải một công thức được đánh số trong chương.

**Analysis:** Viết lại quy trình nguồn, không gán Eq. giả: P(K=0)=1-tau và P(K=k)=tau/3, k=1,2,3. Suy dẫn agent: E[K]=2*tau; chia cho D cho tỷ lệ ô trong một bảng có D thành phần và K chính là số ô bị che trong D đó. Không cần giả định uniform component selection để suy ra tổng ô, nhưng nguồn chưa đủ chi tiết để tuyên bố mọi subset được chọn đồng xác suất. Được hỗ trợ ở mức recipe và phép suy dẫn, không hỗ trợ MAR logistic N+trait.

## C5 — MNAR trong Seki 2025 dựa trên diversity của timepoint, không phải độ hiếm của từng component

**Claim — tác giả phát biểu:** S mô tả chọn timepoints cho MNAR dựa trên alpha diversity cao.

**Evidences:**
- **E07 — S; PDF/in 4; Section 3.1; đoạn đầu, câu bắt đầu “For MCAR”:** “based on high alpha diversity.” Chủ ngữ là missing time points; phần trước đối chiếu với việc bỏ timepoints ngẫu nhiên ở MCAR. Hỗ trợ hướng gây thiếu ở mức mẫu/timepoint.

**Analysis:** Được hỗ trợ bởi E07. Không đủ chứng cứ để nói tác giả dùng top-K, weighted sampling hay một sigmoid nào. Cũng không được thay alpha diversity bằng -log(p_j+epsilon) rồi gọi là Seki. Chưa xác minh script/supplement tạo mask nên chưa cam kết tái lập MNAR của S. Áp dụng vào component hiếm sẽ là thay đổi đối tượng score, không phải tái lập.

## C6 — Schouten thực sự có công thức weighted score và framework candidate patterns

**Claim — mô tả phương pháp của tác giả:** A cho phép nhà nghiên cứu đặt weights, phân case vào candidate patterns rồi dùng score để xác định xác suất gây thiếu.

**Evidences:**
- **E08 — A; PDF 8 / in 7; Section 3.1; đoạn sau Figure 3, công thức không đánh số:** “the coefficients are determined by the researcher.” Công thức gốc đã xem trên ảnh:

  wss_i = w_1*y_1i + w_2*y_2i + ... + w_m*y_mi.

  y_vi là giá trị biến v của case i; w_v là weights nhà nghiên cứu quy định. Không gán thêm Eq. number.
- **E09 — A; PDF 9 / in 8; Section 3.1 tiếp nối:** “all variables that will be amputed.” Ngữ cảnh: MAR đặt zero weights cho những biến sẽ bị che; MNAR cho phép weights khác zero ở biến bị che.
- **E10 — A; PDF 6 / in 5; Section 3.1; đoạn phân nhóm candidate:** “candidate for one pattern only”. Hỗ trợ thiết kế phân case vào pattern trước khi quyết định có gây thiếu không, khác với tự mở rộng mask ô sau draw.

**Analysis:** Công thức/khung được hỗ trợ bởi E08–E10; đây là nguồn NGOÀI folder. Dùng N và trait làm y observed là cấu hình weights trong framework. Dùng latent p thay y, rồi áp mask lên counts X là ánh xạ benchmark của agent; paper A không nghiên cứu trực tiếp cặp latent-proportions/sampled-counts này. Nhiều component hoặc nhiều timepoints cần khai báo cả pattern và weight vector, không mặc nhiên mean-log-rarity. Framework không bảo đảm hiệu quả baseline trên synthetic data đang thiết kế.

## C7 — Công thức LEFT/RIGHT/MID/TAIL cụ thể được kiểm từ code, không phải Eq. của paper

**Claim — mã tác giả:** AC, ampute.continuous.R triển khai bốn nhánh probability và reference calibration.

**Evidences:**
- **E11 — AC; [snapshot ampute.continuous.R dòng 47](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/docs/evidence/synthetic_missingness_sources/ampute.continuous.R:47); function ampute.continuous, LEFT branch:** `logit(mean(x) - x + b)`. Hỗ trợ LEFT. Các nhánh RIGHT/MID/TAIL nằm ở dòng 48–50; logit ở dòng 37 được code định nghĩa bằng exp(x)/(1+exp(x)), tức sigmoid dù tên hàm là logit.
- **E12 — AC; [cùng snapshot dòng 36](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/docs/evidence/synthetic_missingness_sources/ampute.continuous.R:36); calibration dòng 52–57:** `rnorm(n = 10000, mean = 0, sd = 1)`. Hỗ trợ Normal reference sample; binary search dùng formula trên testset đó để nhắm prop.

**Analysis:** Viết lại code bằng toán (không phải Eq. đánh số trong A), đặt m=mean(s), sigma là sigmoid:

LEFT: sigma(m-s+b); RIGHT: sigma(s-m+b).

MID: sigma(-|s-m|+.75+b); TAIL: sigma(|s-m|-.75+b).

Được hỗ trợ bởi E11–E12 và trực tiếp đọc nhánh code cùng ngữ cảnh. .75 là hằng trong code, không phải số agent tự chọn. Không nói calibration này bảo đảm empirical cell rate đúng trên panel zero-inflated: calibration dùng 10000 draws chuẩn, sau đó sampling còn gây dao động. Suy dẫn agent: thay bằng root trên panel thật là một extension/thuật toán khác, không phải giữ nguyên implementation AC. Nhánh score không biến thiên có xử lý xác suất prop tại dòng 77–87; chưa chạy kiểm thử R trong audit này.

## C8 — Nhân mọi weight cùng hệ số dương không tạo strength grid trong sumscores này

**Claim — suy dẫn từ code tác giả:** Với score không hằng và cùng candidates, đổi weights w thành c*w, c>0, không đổi score chuẩn hóa trong AC.

**Evidences:**
- **E13 — AC; [ampute.R dòng 550–552](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/docs/evidence/synthetic_missingness_sources/ampute.R:550); sumscores:** `scores <- scale(scores)`. Dòng 550 tạo weighted sum; điều kiện dòng 551 yêu cầu đủ candidates và score có biến thiên. std điều khiển chuẩn hóa input ở dòng 539–547, không tắt score scaling ở dòng 552.

**Analysis:** Suy dẫn agent, không phải câu tác giả phát biểu: Z(c*s)=(c*s-c*mean(s))/(c*SD(s))=Z(s) khi c>0. Do đó beta=.5/1/2 bằng cách chỉ nhân toàn bộ weights không hoạt động như đề xuất cũ. Phát biểu này được hỗ trợ bởi E13 trong miền điều kiện nêu trên. Muốn đổi slope sau scaling phải tạo extension và kiểm chứng; muốn thay phương pháp strength theo nguồn khác phải chọn lại, không lặng lẽ gọi là tham số gốc ampute.

## C9 — Formula xác suất rời rạc có trong code mice, chưa đối chiếu nguyên bản luận án Brand

**Claim — mã tác giả:** Nhánh discrete tính xác suất từ số nhóm, prop và relative probability weights.

**Evidences:**
- **E14 — AC; [ampute.discrete.R dòng 67](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/docs/evidence/synthetic_missingness_sources/ampute.discrete.R:67):** `(ng * prop * odds[i, l])`. Phần tiếp theo của cùng dòng chia cho tổng odds của pattern i. Dòng 14–20 giải thích odds là relative probabilities; dòng 27–29 dẫn Brand 1999.
- **E15 — AC; [cùng file dòng 40–49](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/docs/evidence/synthetic_missingness_sources/ampute.discrete.R:40):** `anyDuplicated(quantiles)`. Điều kiện trùng cutpoints hoặc NA dẫn tới stop; dòng 68–75 cảnh báo khi xác suất >=1.

**Analysis:** Viết lại code: pi_g=G*prop*d_g/sum_h(d_h), cho group g và pattern đang xét. Được hỗ trợ bởi E14–E15; không gọi là công thức đã kiểm ở một trang luận án Brand vì chưa đọc luận án đó. Zero không làm lỗi log nhưng có thể tạo ties; không bảo đảm option này chạy được trên mọi p=0 panel. Nhóm không đều và saturation làm rate cần được kiểm tra; tự sửa normalization là extension.

## C10 — Muzellec paper có MAR logistic và MNAR hai đuôi; lower-tail riêng được xác minh trong code

**Claim — tác giả mô tả thí nghiệm:** O nêu MAR dùng observed inputs của logistic model, và self-masked MNAR dùng đuôi dưới lẫn đuôi trên.

**Evidences:**
- **E16 — O; PDF 7; Section 4; heading “Missing value generation mechanisms”, cột trái:** “random weights”. Ngữ cảnh trước/sau: chọn subset biến luôn observed; các biến còn lại có missingness theo logistic lấy những observed variables làm input; fit bias bằng line search. Hỗ trợ đúng phạm vi MAR của nguồn, không phải weights dấu N/trait đã được paper dùng.
- **E17 — O; PDF 7; cùng đoạn tiếp sang cột phải:** “lower and upper p-th percentiles”. Ngữ cảnh: che theo Bernoulli ở hai đuôi, vùng giữa không thiếu. Hỗ trợ MNAR hai đuôi mô tả trong thí nghiệm, không tự chứng minh mọi ô đều có missing probability dương.
- **E18 — OC; [snapshot utils.py dòng 365](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/docs/evidence/synthetic_missingness_sources/MissingDataOT_utils.py:365); MNAR_mask_quantiles:** `cut == 'lower'`. Dòng 366–367 tính quantile và điều kiện <=. Dòng 368–371 là nhánh both.
- **E19 — OC; [cùng file dòng 375](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/docs/evidence/synthetic_missingness_sources/MissingDataOT_utils.py:375):** `(ber < p) & m`. Hỗ trợ Bernoulli selection giao với tail-membership.

**Analysis:** Được hỗ trợ ở mức paper E16–E17 và code E18–E19; lower-tail không phải cấu hình hai đuôi nguyên bản của đoạn paper. Viết lại code: A_ij=1[x_ij<=Q_j(q)]; H^Q_ij=A_ij*B_ij, B~Bernoulli(rho). Đổi x sang latent p của mình là extension ánh xạ biến; rho là tham số Bernoulli trong code, không đồng nghĩa global missing-cell rate r. p=0 của mình chỉ cần so sánh threshold, không epsilon; mass zero có thể khiến tỷ lệ ô thỏa <=quantile lớn hơn q. Đây là suy dẫn miền giá trị, chưa phải kết quả chạy trên synthetic dataset.

## C11 — OR với MCAR là code; công thức xác suất và calibration sau OR là suy dẫn của agent

**Claim — mã tác giả và suy dẫn tách biệt:** OC có tùy chọn thêm MCAR nền; những xác suất đã viết trước là đại số suy ra từ phép OR, không phải Eq. trong paper O.

**Evidences:**
- **E20 — OC; [utils.py dòng 377–379](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/docs/evidence/synthetic_missingness_sources/MissingDataOT_utils.py:377); if MCAR:** `mask = mask |`. Phần phải của dòng 379 lấy torch.rand mới và so với cùng p; mặc định MCAR=False ở signature dòng 314. Cùng đọc E19 để xác định random draw đuôi là draw riêng.

**Analysis:** Dưới independent draws và dùng mọi feature (p_params=1), suy dẫn agent:

P(H=1|A=0)=rho; P(H=1|A=1)=1-(1-rho)^2=2*rho-rho^2.

Nếu f là tỷ lệ ô trong tail thực tế, E[r_hat]=rho+(1-rho)*rho*f. Không nói đây là phương trình paper đã đánh số hay code tự giải nó. Giải rho từ target r là calibration bổ sung; ép exact realized rate còn là lựa chọn khác. Với p_params<1 cần tính thêm phần feature-selection khi quy đổi global rate. Chỉ hỗ trợ có điều kiện bởi E19–E20, chưa được kiểm chứng thực nghiệm.

## C12 — Các công thức rare-p, epsilon và ghép khối cũ không có provenance để gán cho paper

**Claim — kết quả kiểm toán phát biểu của agent:** Không evidence nào ở C1–C11 hỗ trợ việc gán nguyên cụm -log(p+1e-6), mean rarity theo khối, global cell-weighted root và partition L*q của đề xuất cũ cho H/M/T/S/A/O như một phương pháp nguyên bản.

**Evidences:** E01–E20 chỉ hỗ trợ đúng các recipe, định nghĩa, score, code branches và suy dẫn đã nêu. Đặc biệt E12 cho một calibration khác đề xuất cũ; E13 phản chứng cách hiểu strength; E17 khác hướng lower-tail-only; E07 khác đối tượng score.

**Analysis:** Kết luận là thiếu provenance TRONG PHẠM VI NGUỒN ĐÃ KIỂM, không phải chứng minh các công thức đó không tồn tại ở mọi tài liệu khác. Loại chúng khỏi thiết kế đã duyệt. Việc triển khai cần chọn phương pháp nguồn, sau đó liệt kê riêng từng adaptation và phép kiểm tra. p kín gây thiếu counts X là latent-driven mechanism, không tự gọi là bản sao của self-masking x->x trong nguồn.

## C13 — Cận khả thi về block rate là suy dẫn từ kích thước dataset, không phải claim của paper

**Claim — suy dẫn agent từ yêu cầu người dùng:** Với một entity có 120*12=1440 target cells, một khối q*L ô che được tối đa q*L/1440 tổng số ô nếu mỗi entity chỉ có một khối.

**Evidences:** Yêu cầu đã xác nhận trong hội thoại và [.vibe-wise/project-map.md](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/.vibe-wise/project-map.md): T=120, D=12, các khối có q component và L thời điểm. Không có câu trích paper cho phép đếm này vì đây là số của benchmark đang thiết kế.

**Analysis:** q=3,L=12 => tối đa 36/1440=2.5%. Nhiều khối, pattern unions và các điều kiện tiếp giáp là phần thiết kế tiếp, chưa tự chọn. Cận này được hỗ trợ bởi phép đếm, không chứng minh có thể ghép mọi grid combination bằng một thuật toán từ paper. Giữ desired grid và quay lại tính feasibility sau khi người dùng chọn nguồn.

## Phụ lục — Kiểm tra thêm hai nguồn CSDI đã được đề cập trước

### C14 — Seki 2023 mô tả Binomial timepoint masking

**Claim — tác giả phát biểu:** Số timepoints thiếu theo subject được sinh bằng phân phối Binomial trong phần mô tả imputation targets.

**Evidence E21:** [csdi_2023.pdf](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/research/csdi_2023.pdf), PDF 2 / in 915, Section II.A, cột phải, đoạn bắt đầu “The microbiome relative abundance data”: “binomial distribution.” Câu trước nói targets được chọn ngẫu nhiên; không có một Eq. masking riêng tại đây.

**Analysis:** Viết K_i~Binomial(T_i,r) rồi chọn subset timepoints đều là cách diễn giải cần giả định chọn subset đồng đều; main text không viết đầy đủ tham số/mã lựa chọn subset. Do đó không gọi mọi chi tiết của cách hiện thực này là đã xác minh nguyên xi. Nguồn không đủ cho rare-p MNAR hay independent L control trong đoạn đã đọc.

### C15 — CSDI 2021 Algorithm 3 là target choice trong self-supervised training

**Claim — thuật toán nguồn:** Algorithm 3 lấy target ratio ngẫu nhiên rồi chọn phần observations làm training targets.

**Evidence E22:** [CSDI 2021 PDF](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/research/2021_CSDI%20--%20Conditional%20Score-based%20Diffusion%20Models%20for%20Probabilistic%20Time%20Series%20Imputation.pdf), PDF/in 14, Appendix B.2, Algorithm 3, step 3: “Draw target ratio”. Công thức trên ảnh ghi r~Uniform(0,100), step 4 chọn r% observed values; Section 4.3 đặt ngữ cảnh self-supervised learning.

**Analysis:** Được hỗ trợ cho training-target selection; chưa hỗ trợ một recipe test MAR/MNAR như user muốn. Algorithm 4 chuyển missing patterns từ dataset tham chiếu, không tự biến chúng thành dependence theo latent p trên panel mới.

## Kết quả kiểm tra và thay đổi đối với đề xuất trước

- 22/22 đoạn trích trong [evidence_ledger.json](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/docs/evidence/synthetic_missingness_sources/evidence_ledger.json) khớp nguồn snapshot sau Unicode NFKC + nối whitespace. Kiểm tra này không tự chứng minh diễn giải đúng; diễn giải được giới hạn trong từng Analysis.
- Các code sources được pin commit và lưu SHA256 tại manifest.json; local PDF hashes ở local_pdf_manifest.json. Chưa chạy code R/MissingDataOT để kiểm distribution/rate.
- Được hỗ trợ: các recipe/định nghĩa local ở C1–C5; framework weighted score/patterns C6; implementation C7–C11 dưới đúng điều kiện.
- Phải sửa cách trình bày: code không phải paper equation; alpha-diversity MNAR không phải rare-component MNAR; beta-scaling không phải mặc định của ampute; calibration bổ sung không phải code gốc.
- Chưa được chốt: chọn option, mapping latent p -> count mask, strength protocol thay thế, pattern bank nhiều khối và rate calibration. Không chuyển sang code chỉ vì hoàn tất audit.
