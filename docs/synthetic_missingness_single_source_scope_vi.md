# Đề xuất phạm vi một nguồn cho missingness

Ngày: 2026-10-23. Người dùng đã chốt Schouten 2018 và cho phép triển khai Table 1, MCAR/MAR, fixed patterns. Phạm vi này đã triển khai và kiểm tra; xem `nextgen/SCHOUTEN_VI.md`. Generator dữ liệu đầy đủ và các mục NOT-READY chưa triển khai. Phần đề xuất bên dưới giữ lại để truy vết căn cứ khoa học.

## Đề xuất chính

Chọn duy nhất Schouten, Lugtig & Vink (2018), Generating missing values for simulation purposes: a multivariate amputation procedure, làm nguồn phương pháp gây thiếu. Dùng implementation ampute thuộc chính phương pháp đó làm oracle, không đưa thêm quantile-censoring của Muzellec hay alpha-diversity selection của Seki vào cùng protocol.

Đây là khuyến nghị của agent. Paper nguồn không phải một nghiên cứu riêng trên composition/counts; cần giữ giới hạn này rõ nếu người dùng ưu tiên đúng domain hơn tính thống nhất phương pháp. Không có căn cứ để nói dùng một paper tự chứng minh hiệu quả trên dữ liệu người dùng.

## C16 — Một framework nguồn cho MCAR/MAR/MNAR, thay vì ghép nhiều phương pháp

**Claim — phương pháp tác giả:** Paper A cung cấp một framework weighted sum score và candidate patterns cho nhiều cơ chế.

**Evidences:**
- **E08 đã kiểm:** A, PDF 8 / in 7, Section 3.1, công thức không đánh số; “the coefficients are determined by the researcher.” Công thức wss_i=sum_v w_v*y_vi và Figure 3.
- **E09 đã kiểm:** A, PDF 9 / in 8, Section 3.1 tiếp nối; “all variables that will be amputed.” Ngữ cảnh quy định MAR weights bằng zero ở biến bị che, MNAR cho phép weights khác zero tại đó.
- **E10 đã kiểm:** A, PDF 6 / in 5, Section 3.1; “candidate for one pattern only”.

**Analysis:** Có thể lấy cùng framework làm nguồn cho MCAR/MAR và MNAR reference. Chọn covariates và weights là cấu hình được phương pháp cho phép, không phải ghép một phương pháp từ paper thứ hai. Tuy vậy, weights/dataset cụ thể vẫn phải được gọi là cấu hình benchmark, không phải đúng thí nghiệm paper. Được hỗ trợ trong phạm vi E08–E10; không chứng minh toàn bộ lưới hiện tại đã paper-validated.

## C17 — Paper nhắc missingness ở một wave longitudinal, không đặc tả đầy đủ random temporal blocks của benchmark

**Claim — ví dụ tác giả và giới hạn nguồn:** Có ví dụ pattern của một lần đo longitudinal; chưa đủ căn cứ để gọi bộ sinh khối L,q,rate độc lập là implementation nguyên bản.

**Evidence E23:** A, PDF 6 / in 5, Section 3.1, đoạn bắt đầu “The amputation procedure starts”; “second round of a longitudinal study”. Ngữ cảnh ví dụ người tham gia không xuất hiện ở lần đo thứ hai, nên các biến của wave đó bị thiếu.

**Analysis:** Pattern cố định bỏ một wave là biểu diễn trực tiếp trong framework. Không suy ra tác giả đã đưa thuật toán chọn ngẫu nhiên start/length, chống khối chồng/kề hay giữ exact missing-cell rate trên panel 120×12. Do vậy fixed-wave pattern có căn cứ để đưa vào phạm vi kế tiếp; random temporal-block grid cần NOT-READY ở phiên bản này. Đây là giới hạn của phạm vi triển khai đã có evidence, không phải khẳng định framework không thể mở rộng.

## C18 — Validation của paper không trực tiếp chứng minh hiệu quả trên latent proportions và sampled counts

**Claim — thiết kế thí nghiệm tác giả:** Section 4.1/Table 1 dùng mô phỏng Gaussian và các biến Y1,Y2,X1, không phải mô hình latent p -> multinomial counts của benchmark đang thiết kế.

**Evidence E24:** A, PDF 9 / in 8, Section 4.1, đoạn bắt đầu “We perform a model-based simulation”; neo `mvrnorm`. Đoạn nêu sinh N=1000 cases từ multivariate normal; công thức mu và Sigma nằm ngay dưới.

**Evidence E25:** A, PDF 10 / in 9, Section 4.1, Table 1 và đoạn sau bảng; “20% of the cases”. Table 1: pattern Y1,Y2,X1=(0,0,1); weights MCAR=(0,0,0), MAR=(0,0,1), MNAR=(1,0,0). Tác giả đánh giá mean Y1 bằng bias, coverage và interval width, không phải ranking Hron/MICoDa/Tsagris trên sampled counts.

**Analysis:** Phải tách kiểm chứng implementation đúng nguồn khỏi đo hiệu quả baseline trên dataset mới. Cả hai cần thực hiện; một paper duy nhất chỉ cải thiện truy vết phương pháp. MNAR theo latent p nhưng che counts X không phải y tự chi phối việc che y như thí nghiệm nguồn; không tự đổi sang counts-driven MNAR để lấp ô yêu cầu. Phần latent-driven này giữ NOT-READY.

## Quy ước trạng thái

- **IMPLEMENT-NEXT:** có evidence để đưa vào scope triển khai sau khi chọn nguồn; không có nghĩa code đã hoàn thành hoặc đã chạy.
- **NOT-READY:** chưa triển khai trong phiên bản này; không tự thay bằng phương án khác, không sinh kết quả giả cho cấu hình đó.
- **REFERENCE-ONLY:** tái lập thiết kế paper để kiểm chứng công cụ; không tính là đã đáp ứng yêu cầu synthetic tương ứng.

## Ma trận phạm vi đề xuất

| Chức năng | Trạng thái đề xuất | Căn cứ và giới hạn |
|---|---|---|
| Reproduce Table 1: MCAR/MAR/MNAR source fixtures | IMPLEMENT-NEXT / REFERENCE-ONLY | Paper Table 1 và implementation ampute được pin. Dùng để xác minh patterns, score, probabilities và sampling, không ép benchmark counts mới thành Gaussian. |
| MCAR trên các target counts với pattern đã khai báo | IMPLEMENT-NEXT | Cùng framework; probabilities không phụ thuộc giá trị. Cấu hình pattern phải được đăng ký trước. Không hứa mọi missing rate đều khả thi. |
| MAR theo N và entity_trait_01 luôn quan sát | IMPLEMENT-NEXT | Weights chỉ trên covariates observed; target weights zero. Chọn dấu/tỷ trọng là cấu hình benchmark theo framework, không phải weights Table 1 nguyên bản. |
| Pattern cố định làm mất một lần đo trên panel | IMPLEMENT-NEXT | Ví dụ longitudinal wave trong Section 3.1; ghi rõ vị trí wave, không gọi là random block generator. |
| MNAR theo biến trực tiếp bị che, như source Table 1 | REFERENCE-ONLY | Đủ để kiểm phương pháp, nhưng không thay yêu cầu rare latent-p MNAR của user. |
| MNAR: p tiềm ẩn thấp -> counts X dễ thiếu | NOT-READY | Mapping latent cause/measurement khác variable tự bị che trong thí nghiệm nguồn; cần căn cứ/validation riêng, không ghép thêm paper trong phiên bản này. |
| Random temporal blocks L=6/12/24 với q=1/3/6/12 và rate độc lập | NOT-READY | Chưa có algorithm nguồn đã xác minh cho toàn bộ chức năng; không dùng tile/merge proposal cũ. |
| Whole-component-within-entity scenario tự sinh theo cơ chế | NOT-READY | Chưa chọn pattern bank/activation protocol có căn cứ đủ sát để gọi là triển khai xong. |
| beta=.5/1/2 như logistic slope control | NOT-READY | ampute chuẩn hóa score; không giữ strength interpretation cũ. |
| Lưới target-cell rates .10/.30/.50 đầy đủ cho mọi pattern | NOT-READY ở cấp toàn lưới | Phải kiểm mẫu số target-only, calibration gốc và khả thi từng pattern. Chỉ những cấu hình khả thi, được đăng ký và kiểm tra mới được mở riêng. |
| Bảo đảm exact realized rate | NOT-READY | Không sửa Bernoulli draws hoặc cắt khối để ép số ô. |
| Nguồn -log(p+epsilon), quantile-censoring hoặc diversity-based MNAR từ paper khác | Không đưa vào scope | Giữ một nguồn thống nhất như ưu tiên của người dùng. |

## Scope triển khai sau khi nguồn được lựa chọn

1. Source-reproduction layer: Table 1 fixtures, pin nguồn/code, masks có seed và source/config provenance. Kiểm functional parity với implementation gốc; equality tiêu thụ RNG khác ngôn ngữ không được tự đòi/khẳng định bitwise nếu không đồng bộ draws.
2. Count-data conditioning adapter cho MCAR/MAR từ pattern matrix đã đăng ký; chỉ dùng N, trait để tạo MAR. Không đưa p hoặc residual vào baseline inputs.
3. Capability manifest chặn các cấu hình NOT-READY bằng thông báo rõ ràng. Không fallback từ MNAR sang MCAR hoặc từ partial blocks sang whole-row masks.
4. Lưu target rate, achieved rate, stratum support và pattern feasibility. Kiểm hidden counts không lọt vào fit/encoding; giữ cùng mask cho các baseline.

Scope này đã được người dùng chốt và hiện thực; bảng IMPLEMENT-NEXT phía trên ghi lại trạng thái đề xuất tại thời điểm lựa chọn nguồn. Trạng thái hiện tại và kết quả chạy nằm ở `nextgen/SCHOUTEN_VI.md` và capability manifest của module synthetic. Công thức ampute, calibration và provenance chi tiết đã có trong synthetic_missingness_claim_evidence_audit_vi.md.

## Liên kết nguồn

- [Paper A: bản tác giả 23 trang](https://rianneschouten.github.io/mice_ampute/manuscript/manuscript.pdf).
- [DOI](https://doi.org/10.1080/00949655.2018.1491577).
- [Snapshot ampute đã pin](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/docs/evidence/synthetic_missingness_sources/manifest.json).
- [Audit Claim/Evidence/Analysis trước](C:/Users/admin/Tai_lieu/detaikhoahoc/COVID19/missing_impute/code-missing-imputation/docs/synthetic_missingness_claim_evidence_audit_vi.md).
