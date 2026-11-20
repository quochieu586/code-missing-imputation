# Metadata cho MainModel: phương án nhỏ nhất, giải thích được

**Chỉ thiết kế, chưa sửa source/chạy train.** Theo `plan.md`; thay thế khuyến nghị phức tạp hơn trong `DESIGN_VI.md`, không sửa tài liệu đó. Dẫn dòng theo working tree 2026-11-20, HEAD `59a7777f3bb72e1f0439eaf18f192a74232656b2` (working tree có nhiều thay đổi có sẵn).

**Chốt:** bốn categorical lookup + một linear cho tuổi → cộng thành một vector C chiều/location → cộng vào hidden `h`. Giữ location đúng yêu cầu 5 trường, nhưng ablation bỏ location là bắt buộc. Mặc định tắt, không thêm attention hay pipeline mới.

## 1. Đọc bài: Claim → Evidence → Ý nghĩa cho COVID

Trang dưới đây là **trang PDF**, không phải số trang do công cụ đọc tính cả đoạn rỗng cuối file. Nguồn nằm ở `../research/`; trích ngắn giữ nguyên tiếng Anh, nối dòng khi cần; công thức lỗi font được ghi rõ là diễn giải.

### Seki, Zhang, Imoto 2025 — Bioinformatics Advances, vbaf181

- **Claim:** `csdi.pdf` và `phylo.pdf` là cùng bài, nhưng **không cùng bytes**. **Evidence:** p1 cùng tên *Diffusion model for imputing time-series gut microbiome profiles using phylogenetic information and metadata integration*, DOI `10.1093/bioadv/vbaf181`, ba tác giả Misato Seki, Yao-zhong Zhang, Seiya Imoto. SHA256 lần lượt `d09dbd5e857eeed214dea3feb1b3dd2d2d28b1681d4f42419a07cf840f45e231` và `944516c4aab9803283ddd09e56b2df76cccf7d4b6bb1664cd1d0c033e5b67417`; text `pdftotext -layout` bằng nhau hoàn toàn. **Ý nghĩa:** dùng một nguồn khoa học, không đếm thành hai nghiên cứu độc lập.
- **Claim:** metadata thực nghiệm là country và tình trạng dị ứng milk/egg/peanut, không phải ID subject. **Evidence:** p6 §3.3: “conducted training using metadata for country and allergy presence”; p6 cũng nêu ba loại dị ứng. Age được nhắc như nguồn biến thiên, không đủ để khẳng định được đưa vào embedding trong thí nghiệm này. BONUS không có metadata công khai nên không tích hợp metadata (p6 §3.4). **Ý nghĩa:** continent/classification/group tương tự thuộc tính nhóm; location của ta lại là ID entity, khác country của cohort gồm nhiều trẻ trong cùng quốc gia.
- **Claim:** categorical dùng Feature Tokenizer (FT), d=8. **Evidence:** p3 §2.4: “were embedded using a FT (Gorishniy et al. 2021)”. Eq.7 **diễn giải do ký tự toán lỗi font**: `E_f = b_f + e_fᵀW_f ∈ R^d`, `d=8`, e là one-hot; phần chữ sạch: “is a one-hot vector representing the categorical values”. **Ý nghĩa:** mỗi nhãn chọn một hàng tham số rồi cộng bias; không có thứ tự số học giữa các nhãn. Đây là tokenizer, không bắt buộc dùng toàn bộ FT-Transformer.
- **Claim:** thử hai hướng nối. **Evidence:** p3 §2.4, **diễn giải shape từ công thức lỗi font**: spatial `(C,L,K+n·d)`, channel-wise `(C+n·d,L,K)`; chữ sạch: “appended along the feature dimension” và “concatenated along the channel dimension”. **Ý nghĩa:** spatial thêm vị trí feature giả; channel-wise giữ nguyên các feature sinh học, thêm mô tả tại từng vị trí. Không được hiểu n·d là số biến thể mới.
- **Claim:** không có bên thắng rõ; channel-wise allergy ổn định hơn. **Evidence:** p6 §3.3: “No substantial difference was observed between the two embedding concatenation directions”; “more stable results”; lợi ích “modest”, rõ hơn khi missing ratio vượt 0.4. Fig.5a p7 biểu diễn MAE; §3.3 không nêu một ΔMAE hay % cải thiện tổng hợp. **Ý nghĩa:** hiệu ứng được báo là nhỏ, không bịa phần trăm hoặc lấy mức cải thiện của CNN so với CSDI làm hiệu ứng metadata. Chưa số hoá Fig.5, nên chưa có effect size số chính xác; không hứa metadata chắc chắn tốt hơn cho COVID.
- **Claim:** bài nói train-only nhưng cách loại trong evaluation chưa rõ hoàn toàn. **Evidence:** p3 §2.4: “Metadata variables were included during model training and excluded during evaluation.” p8 lại mô tả metadata là “conditional inputs during denoising”. **Ý nghĩa:** chắc chắn không chấm lỗi tái tạo metadata như abundance; nhưng câu đầu có thể còn chỉ việc bỏ metadata khỏi input lúc test. Văn bản chưa mô tả zero/mask/remove tensor thế nào, nên **không kết luận chắc rằng metadata vẫn được cung cấp lúc suy luận**, cũng không tự dựng một cơ chế train-only. Cần kiểm code tác giả/xác nhận trước khi tuyên bố tái lập Seki. Thiết kế COVID bên dưới dùng metadata đã biết ở cả train và inference, chỉ chấm target: đây là lựa chọn rõ ràng của ta, không gọi là tái lập nguyên xi train-only của bài.
- **Claim:** lợi ích có giới hạn. **Evidence:** p6 §3.3 nói “moderate discriminative power” và “constraints of the embedding method”; p8 nói ảnh hưởng hạn chế của background metadata lên chuỗi compositional. **Ý nghĩa:** thuộc tính tĩnh không giải thích được mọi đợt dịch; tương quan không phải quan hệ nhân quả, và nguồn/năm metadata COVID cần xác minh.

### Seki, Zhang, Imoto 2023 — IEEE BIBM

- **Claim:** bản 2023 dùng CSDI với sắp thứ tự feature theo tương quan trong nhóm phylum, **không dùng phylum CNN**, chưa mô tả FT metadata. **Evidence:** `csdi_2023.pdf` p1 xác nhận đúng ba tác giả, tên *Imputing time-series microbiome abundance profiles with diffusion model*, DOI `10.1109/BIBM58861.2023.10385703`; p2: “features were ordered based on their correlation”, “One-layer Transformer encoders are included for each of temporal and feature dimension”. Phylum được dùng để nhóm/sắp feature; CNN chỉ được nhắc ở p1 khi dẫn TaxoNN. P3 có “metadata information of allergies” trong lựa chọn cohort, không phải công thức embedding. **Ý nghĩa:** phân biệt sắp feature + Transformer năm 2023 với phylum CNN và FT metadata năm 2025; không đưa CNN/phylogeny vào biến thể COVID chỉ vì dùng cùng dòng bài.
- **Phạm vi tìm thêm:** truy vấn công khai `"Seki" "Zhang" "Imoto" microbiome imputation diffusion publications` trả hai bài trên qua IEEE, OUP và PubMed. Chưa xác định bài thứ ba cùng dòng với đúng nhóm tác giả; không khẳng định tìm kiếm đầy đủ. Chỉ đọc tiêu đề/tóm tắt/snippet web, không dùng chúng làm bằng chứng phương pháp ngoài full text local. Liên kết: <https://ieeexplore.ieee.org/document/10385703>, <https://pubmed.ncbi.nlm.nih.gov/40861397/>. Các bài diffusion microbiome khác trả về không mặc nhiên là bài của Seki.

### CSDI 2021 và proposal

- **Claim:** CSDI đã có time và feature side information. **Evidence:** `2021_CSDI -- Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation.pdf`, Tashiro et al., p7 §5: “128-dimensions temporal embedding”; categorical embedding K features có dimension 16; p16: “we expand each side information and concatenate all the side information”. **Ý nghĩa:** `CompositionNet` đã có feature embedding (`network.py:17`) và sin/cos thời gian (`18–19,43–46`); cùng ý tưởng, khác kích thước và phép **cộng**, không phải bản CSDI diffusion đầy đủ. Metadata là bổ sung thứ ba, không thay hai embedding cũ.
- **Claim:** proposal ưu tiên trực tiếp dự đoán composition, không có đặc tả metadata trong text. **Evidence:** `2026_10_09-Revised proposal.pdf` p2 “Compositional”, “Longitudinal”, “Sparsity”; p7 “Fit model f using Zt to predict Xt+1”; p3 và p10–11 dẫn Seki 2023/2025. P8 là hình “Proposed Network”, không có text chi tiết trích xuất được. **Ý nghĩa:** giữ vòng lặp MainModel, thêm side input nhỏ; metadata là mở rộng có căn cứ, không khẳng định proposal đã yêu cầu nó. Chưa xác minh nội dung hình p8 bằng thị giác.

## 2. Ánh xạ dữ liệu và tensor

Audit lại bằng `.venv\Scripts\python.exe -I -`, chỉ stdlib csv/collections/statistics/hashlib/pathlib:
- `original_long_format.csv`: **11671 dòng, 150 location**; cả 5 trường không thiếu và không đổi trong từng location. Cardinality theo thứ tự continent/location/median_age/classification/group = **6/150/114/4/4**.
- `covariants.csv`: cùng 11671 dòng và cùng thứ tự khoá `(location,date)`. Schema `configs/baselines_covariants.toml:6` liệt kê **17 target biến thể**. Không cần đổi file target.
- Tuổi theo một dòng/location: min–max 16.4–54.4, mean 32.468, population sd 9.122838154872637. Không cân theo số ngày quan sát.
- B là số location trong batch (toàn bộ dataset có 150); L là số thời điểm đã padding; K=17 ở bài toán. **Trong network hiện tại có K′=18** vì MainModel thêm residual/other (`main_model.py:412`), nên `h=(B,L,K′,C)`. Không loại cột residual để thêm metadata.

## 3. Một bảng quyết định duy nhất

Ký hiệu C=`channels` (mặc định 32); mỗi trường phân loại dùng `Embedding(q,C)`. **Không thêm bias categorical riêng:** tổng các bias theo trường của FT gộp được vào bias sẵn có của `Linear(1,C)` cho tuổi, tránh tham số dư không định danh.

| Trường/quyết định | Encoding → dim | Lý do chọn, ý nghĩa; vì sao không chọn cách khác |
|---|---|---|
| continent | Lookup 6 nhãn → C | Vector học chung giữa các nước cùng châu lục. Lookup chính là one-hot × W nhưng không phải tạo one-hot thật; không dùng mã 0..5 như số vì tạo thứ tự giả. |
| classification | Lookup 4 nhãn → C | Nhóm thu nhập là category; không giả định khoảng cách thu nhập bằng nhau. Chuẩn hoá bằng mapping tường minh `LOWER-MIDDLE INCOME` → `LOWER-MIDDLE-INCOME`, strip khoảng trắng, giữ ba nhãn còn lại; không split/regex suy diễn nhãn. |
| group | Lookup 4 nhãn → C | Học liên hệ nhóm chế độ với dữ liệu, không áp thứ bậc số hay diễn giải nhân quả. One-hot cố định vẫn cần linear để vào h; lookup gọn hơn và tương đương. |
| location | Lookup 150 ID → C; giữ trong main_meta | Là bias riêng cho entity, có thể nhớ đặc điểm từng nước; **bắt buộc main_meta_noloc**. Giữ để đáp ứng 5 trường và đo giá trị ID, không khẳng định nó là covariate tổng quát; bỏ hẳn sẽ không trả lời được câu hỏi đó. Không hỗ trợ suy rộng sang nước mới trong bản tối thiểu. |
| median_age | z=(age−μ)/σ; `Linear(1,C)` → C | Tuổi là số liên tục, linear học một hướng thay đổi theo tuổi; chuẩn hoá giúp thang đo ổn định. Không binning (mất thông tin, phải chọn ngưỡng), không lookup 114 tuổi (học thuộc), không MLP (thêm phi tuyến chưa có bằng chứng cần). |
| d | Chọn C, không d=8 + projection | Seki d=8 là lựa chọn thực nghiệm của họ, không định luật. C khớp h nên không thêm lớp chiếu/hyperparameter; nhiều tham số ID hơn d=8 nhưng chỉ một bảng nhỏ. Không tăng channels hay dò d. |
| Ghép vào mạng | `m=Σ lookup_f + Linear(z)`; `h += m[:,None,None,:]` sau network.py:46 | Metadata là ngữ cảnh tĩnh, broadcast theo thời gian và feature. Gần **hướng kênh** của Seki hơn spatial vì không tạo feature giả; không phải literal concatenation. Cộng là fusion tuyến tính bị ràng buộc, không tương đương toàn bộ kiến trúc concat của Seki. Spatial phải thay K′/mask/head; channel concat phải chiếu lại C hoặc sửa attention. |

Vocabulary sort theo nhãn sau chuẩn hoá; thống kê tuổi tính trên tập location của dữ liệu đang fit, lưu cùng mapping để dùng lại khi dự đoán. Với bài toán che ô trên bảng cố định, dùng covariate của toàn bảng là transductive hợp lệ, không đọc đáp án target; không coi đây là đánh giá nước chưa thấy. Chọn **fail-fast** khi thiếu/nonfinite/nhãn lạ/metadata không tĩnh hoặc **sd tuổi bằng 0** (không chia cho 0, không tự thay sd): dữ liệu hiện tại đầy đủ, chưa cần unknown slot hay missing indicator. Nếu sau này có split theo nước, phải fit preprocessing chỉ trên train và thiết kế unknown riêng.

## 4. Không làm

- Không metadata theo thời gian, cases/vaccination/total_sequence: ngoài đúng 5 trường, dễ lẫn mẫu số hoặc tương lai.
- Không FT-Transformer, cross-attention, gating mới, graph/phylogeny: đổi bài toán, khó quy công lợi ích cho metadata.
- Không sinh CSV trung gian/script ETL mới: đọc đúng 5 cột từ nguồn hiện có, tránh hai bản metadata lệch nhau.
- Không config danh sách encoder/dimension theo từng field: danh sách 5 trường cố định đủ dùng.
- Không unknown/missing pipeline, dropout đặc biệt, ID regularizer hay tuning: chưa cần cho dữ liệu đầy đủ; ablation đơn giản hơn để thấy vai trò ID.
- Không bỏ metadata tại inference một cách tuỳ tiện để bắt chước một câu chưa rõ của bài: train/test lệch điều kiện; muốn nghiên cứu train-only phải là thí nghiệm khác.
- Không đổi loss, phép đóng composition, mask, baseline anh em, evaluation code; không train/cài đặt trong ticket này.

## 5. Thay đổi dự kiến (chưa thực hiện)

1. **`data.py:16,21–41,44–61,64`**: thêm một schema key tuỳ chọn `metadata_path` (relative với schema, như group_map). **Có key thì `load_csv` luôn đọc/kiểm metadata, bất kể cờ model**; runner tải data ở `run.py:505` trước models ở `507`, nên loader không dựa vào cờ model. Đọc whitelist đúng 5 trường, group theo location, kiểm tính tĩnh và đủ entity; chỉ giữ bảng entity nhỏ, không target từ nguồn metadata. Thêm `Dataset.metadata=None` và provenance path/SHA256. Không có key thì không mở thêm file. `_visible_data` dùng dataclass replace (`main_model.py:125`) nên giữ metadata.
2. **`main_model.py:344–366`**: cờ `metadata=False`, `metadata_location=True`; chỉ dựng vocabulary/μ/σ khi bật. `pack_sequences:52–75` sort entity tại dòng 64: dùng đúng thứ tự này để tạo tensor. `_network_inputs:284–294`, hai call network `242,275`: truyền và slice metadata theo **chính view và [start:end]** đang dùng, cả training lẫn inference. `impute:410–412`: truyền spec metadata khi tạo mạng; thiếu metadata báo lỗi trước train. Ghi field/mapping/μ/σ vào info chỉ khi bật.
3. **`network.py:8–26,28,46`**: optional spec/input; tạo lookup và Linear tuổi (có bias) **sau mọi module cũ**, chỉ khi bật; cộng m sau dòng 46. Bật nhưng thiếu tensor/shape sai → lỗi rõ. Tắt thì không tạo module mới, không tiêu thụ RNG mới, giữ đường gọi 4 tensor cũ. Padding vẫn xử lý ở dòng 60.
4. **`run.py:138–140,153–180`** đã hỗ trợ alias `model="main"`, không cần registry mới. `563` chỉ thêm provenance metadata khi có; `315–317` đã hash source liên quan. `508` chỉ cho init/refine khi tên là main: dùng **stage full** cho ba model. `resolve_models:184–195` có thể ghi thêm default flags; không hứa config.json byte-identical vì code hash cũng đổi.
5. **Configs mới dự kiến:** `configs/baselines_covariants_meta.toml` sao schema cũ, thêm `metadata_path="../data/original_long_format.csv"`; giữ input chính `data/covariants.csv`. `configs/models_main_meta.toml`: main giữ tham số `models_all.toml: [main]`; main_meta và main_meta_noloc là alias model main, bật metadata, noloc đặt metadata_location=false. Không sửa config cũ. Model off **bỏ qua metadata đã tải**: parity là predictions/history/metrics trên đầu vào hợp lệ, không phải I/O hay provenance/config.json. Schema mới vẫn cần file metadata hợp lệ dù mọi model off; schema cũ không có phụ thuộc đó.

### Tối đa 6 test cần viết khi triển khai

1. **Off parity với bản cũ:** cùng seed/fixture, state_dict cũ và output impute bằng nhau chính xác, không metadata info; thử schema cũ và schema mới. Không lấy hai lời gọi cùng code mới làm bằng chứng parity.
2. **Init parity:** cùng seed, các tham số cũ của mạng bật/tắt bằng nhau; module mới tạo sau module cũ.
3. **Data validation/normalisation:** whitelist 5 cột; alias classification đúng; thiếu entity/tuổi lỗi/nonstatic/sd tuổi bằng 0 → fail-fast; μ/σ tính mỗi entity một lần.
4. **Alignment:** đảo thứ tự CSV, minibatch cuối ngắn, hai view khác nhau; metadata khớp entity/features trong cả train và predict.
5. **Forward contract:** shape/broadcast/padding đúng, output finite tổng bằng 1 trên bước thật; gradient tới encoder; đổi metadata có tác động trên fixture phân biệt được, off không tác động.
6. **Runner/ablation:** alias ba model qua stage full; noloc không tạo lookup location; bật mà thiếu metadata lỗi trước train; loader chỉ mở metadata khi schema có metadata_path, model off bỏ qua metadata đã tải. Chạy thêm bộ test sở hữu network/main/data/run hiện có sau triển khai.

### Đánh giá được đăng ký trước, không thực hiện ở đây

- Tuning chỉ seed 0–4. Trước lượt xác nhận kiểm `used_seeds.json` và manifest: **300–309 chưa dùng** mới được dùng; nếu đã dùng thì dừng xin bộ seed mới, không ghi đè. Tạo/freeze config `configs/evaluation/eval_v1_main_meta.toml` trước chạy theo `evaluation/README.md`.
- Cùng runner run, cùng mask từng seed, cùng siêu tham số/training seed policy, so **main / main_meta / main_meta_noloc**, `--stage full --pass both --save-predictions`; mask chính **whole_feature**, phụ **random_cell**, seed 300–309. Lưu mask hash và các lỗi run; không sinh lại mask tuỳ kết quả.
- Dùng `evaluation.tier1` qua pipeline evaluation hiện có: MAE chính, MSE phụ trên đúng ô target che có đáp án, ghép cặp seed chung, báo ΔMAE, số seed thắng, sign test, số cặp hợp lệ, theo continent/N. Chốt trước giả thuyết main_meta và noloc giảm MAE so main; không thay tiêu chí sau khi nhìn kết quả. Không coi 10 mask seed là 10 dataset độc lập, không suy ra độ chính xác ô thiếu thật.
- **Diễn giải:** cả meta và noloc tốt hơn nhất quán → bằng chứng thuộc tính nhóm/tuổi có ích trên bảng này; chỉ meta tốt → lợi ích phụ thuộc ID, phù hợp ghi nhớ/đặc thù entity, chưa chứng minh cơ chế overfit; noloc tốt ngang hoặc hơn → ưu tiên bỏ location khi triển khai. Không model nào tốt → chấp nhận metadata không giúp. Muốn chứng minh tổng quát sang nước mới cần split theo entity riêng, không kết luận từ ablation che ô.

## 6. Năm ý đưa lên slide

- Mỗi nước có một “thẻ mô tả” gồm châu lục, tên nước, tuổi, nhóm thu nhập và nhóm chế độ.
- Mô hình học cách dùng thẻ này cùng đường biến động theo thời gian, không biến nó thành biến thể virus.
- Chỉ thêm một vector vào mạng hiện tại; tắt công tắc thì kết quả cũ phải giữ nguyên.
- Tên nước có thể giúp học thuộc, nên luôn so thêm phiên bản không có tên nước.
- Chỉ nói metadata có ích nếu thắng trên cùng các ô bị che; bài Seki cũng chỉ báo cải thiện nhỏ.

## 7. Bằng chứng thực thi và giới hạn bàn giao

- Đã chạy `pdftotext -layout <pdf> %TEMP%\seki_simple_producer\{csdi,phylo,c23,c21,proposal}.txt`; đọc các phần phương pháp/kết quả liên quan. csdi có cảnh báo annotation/bounding box/flate stream; text đoạn dẫn vẫn đọc được, công thức đã đánh dấu diễn giải. Các file có 9/9/6/30/11 trang nội dung tương ứng.
- Đã chạy audit stdlib qua PowerShell here-string → `.venv\Scripts\python.exe -I -` (exit 0): số liệu §2, SHA256 và text equality §1. Đọc trực tiếp code và dùng `Select-String` đối chiếu dòng ở §5; `git rev-parse HEAD` cho revision ở đầu. `git status --short` cho thấy nhiều thay đổi có sẵn, không sửa/khôi phục chúng.
- Artifact duy nhất được viết trong repo: **`docs/metadata_embedding_plan/DESIGN_SIMPLE_VI.md`**. Temp sở hữu: `%TEMP%\seki_simple_producer\` với 5 file text; giữ lại để QA đọc, không daemon/process nền, không resource khác cần dọn.
- Chưa train, chưa chạy test source/runtime, chưa independent QA. Còn bất định: cơ chế “excluded during evaluation” của Seki, ΔMAE số từ Fig.5, hình proposal p8 và provenance/năm các metadata COVID. Đây không cản thiết kế tối thiểu nhưng phải ghi rõ khi trình bày khoa học.
