# Kế hoạch triển khai framework điền khuyết dữ liệu compositional longitudinal

## 1. Mục tiêu và phạm vi

Xây dựng một pipeline tái lập được với luồng xử lý:

```text
data/covariants.csv
        |
        v
Tạo M_observed (raw != NaN) và M_target (raw == NaN); đóng băng vĩnh viễn
        |
        +-----------------------------+
        |                             |
        v                             v
3 Tsagris baselines              Fit ZINB chỉ trên observed
        |                             |
        v                             v
dataset_00_tsagris               dataset_01_zinb + zero-state posterior
        |                             |
        +-------------+---------------+
                      v
       Hợp nhất theo observed/zero/non-zero masks
                      |
                      v
dataset_0_fused: observed + target-zero khóa cứng + target-nonzero warm-start Tsagris
                      |
                      v
GAN chỉ dự đoán magnitude trên target-nonzero
                      |
                      v
dataset_i -> dataset_(i+1), lặp đến điều kiện dừng
```

Mục tiêu của framework được chốt theo thứ tự:

1. Tsagris tạo một tensor full sơ bộ để không còn `NaN` và cung cấp warm-start cho magnitude.
2. ZINB mô hình hóa riêng quá trình sinh zero/count và quyết định target cell thuộc zero hay non-zero.
3. Mask hợp nhất hai nhánh để observed values và target-zero không bao giờ bị GAN sửa.
4. GAN chỉ reconstruct magnitude của `M_target_nonzero`; không học lại zero state.
5. Đánh giá tập trung vào reconstruction MSE và độ khớp phân phối trước/sau imputation; chưa làm downstream task trong v1.

Sản phẩm cuối phải bao gồm mã nguồn, cấu hình, kiểm thử, báo cáo đánh giá, manifest tái lập và các bộ dữ liệu bàn giao trên GitHub repository:

`https://github.com/quochieu586/code-missing-imputation.git`

### Ngoài phạm vi của phiên bản đầu

- Không triển khai downstream classification, regression, survival hoặc causal task.
- Không để GAN tự quyết định zero/non-zero; quyết định này thuộc ZINB zero-state branch.
- Không coi positive magnitude do ZINB sample ra là giá trị imputation cuối.
- Không tuyên bố structural/sampling state là nhãn sinh học chắc chắn khi không có ground truth tương ứng.
- Không coi giá trị 0 quan sát được là missing.
- Không thay đổi các giá trị gốc đã quan sát.
- Không tự suy diễn taxonomy/phylogeny giữa các biến thể COVID-19 nếu chưa có mapping được duyệt.
- Không dùng notebook làm implementation chính; notebook chỉ dùng để minh họa hoặc kiểm tra kết quả.

## 2. Dữ liệu đầu vào và các ràng buộc đã xác nhận

File đầu vào: `data/covariants.csv`.

| Thuộc tính | Giá trị đã kiểm tra |
|---|---:|
| Số dòng | 11,671 |
| Số cột | 20 |
| Số location | 150 |
| Số mốc ngày khác nhau | 109 |
| Khoảng ngày | 2020-04-27 đến 2024-07-29 |
| Số thành phần variant | 17 |
| Số ô missing trong 17 thành phần | 86,050 / 198,407 |
| Tỷ lệ missing theo cell | 43.37% |
| Dòng có ít nhất một missing | 11,155 (95.58%) |
| Dòng đầy đủ cả 17 thành phần | 516 |
| Khoảng cách thời gian trung vị | 14 ngày |
| Khoảng cách thời gian lớn nhất | 784 ngày |

Các cột định danh và quy mô:

- `location`: định danh chuỗi thời gian.
- `date`: mốc thời gian.
- `total_sequence`: tổng số sequence của location tại thời điểm đó.

Các thành phần compositional:

`recombinant`, `20A`, `20B`, `20C`, `20E`, `Beta`, `Alpha`, `Gamma`, `Delta`, `Kappa`, `Epsilon`, `Eta`, `Iota`, `Lambda`, `Mu`, `Omicron`, `S:677`.

Thêm thành phần dẫn xuất:

```text
other = total_sequence - sum(17 variant parts)
```

Với một dòng còn missing, `other` cũng được coi là chưa xác định. Tổng khối lượng còn lại cần phân bổ là:

```text
remaining = total_sequence - sum(observed variant counts)
```

### Mask bất biến từ raw data

Với mỗi cell `(location, date, variant)`:

```text
M_observed = 1 nếu raw value không phải NaN; ngược lại bằng 0
M_target   = 1 - M_observed
```

`M_observed` chứa cả observed positive count và observed zero. Cả hai đều là dữ liệu thật được ghi nhận, phải được giữ nguyên tuyệt đối. `M_target` chỉ chứa các cell ban đầu là `NaN`; đây là miền duy nhất được phép impute.

Hai mask này phải được tạo trực tiếp từ raw CSV trước mọi phép điền, lưu thành artifact có checksum và không bao giờ suy lại từ một dataset full.

### Invariant bắt buộc cho mọi dataset full

1. Không còn `NaN` trong 17 cột variant.
2. Mọi count là số nguyên không âm.
3. Mọi giá trị quan sát ban đầu được giữ nguyên tuyệt đối.
4. `sum(17 variants) + other == total_sequence` trên từng dòng.
5. Khóa `(location, date)` là duy nhất và thứ tự dòng đầu ra ổn định.
6. `M_observed` và `M_target` được lưu riêng, bù nhau hoàn toàn và không suy lại từ `dataset_i`.
7. Mỗi artifact có checksum, config, seed, Git commit và source dataset checksum.
8. Mọi `M_observed=1`, kể cả observed zero, phải bằng raw data ở từng bit/count.
9. Chỉ `M_target_nonzero` được phép nhận output magnitude từ GAN.
10. Mọi target cell được ZINB gate là zero phải luôn bằng 0 trong run tương ứng.

## 3. Quyết định chọn bài báo cho nhánh `dataset_00_tsagris`

### Ma trận phù hợp

| Bài báo | Compositional | Missing | Zero trực tiếp | Longitudinal | Kết luận |
|---|---|---|---|---|---|
| Hron et al. | Có, Aitchison/ILR | MCAR/MAR | Không; log-ratio yêu cầu số dương | Không | Không chọn vì dữ liệu có rất nhiều zero |
| Saha et al. - MICoDa | Có, ALR và covariate | Có | Dùng hằng số `W`; structural zero phải biết trước | Không | Không chọn vì zero treatment phụ thuộc pseudo-count và dữ liệu hiện không có bộ covariate đầy đủ như thiết kế gốc |
| Tsagris et al. | Có, JSD trên simplex | MAR | Có; `0 log 0 = 0`, không cần log-ratio | Không | Chọn cho `dataset_00_tsagris` |

### Quyết định

Chọn bài **Tsagris et al.** và chạy lại đủ ba biến thể được mô tả trong bài:

1. `JSD-kNN`: JSD distance, arithmetic mean của hàng xóm.
2. `JSD-alpha-kNN`: JSD distance, Fréchet mean với `alpha` được tune.
3. `Adaptive JSD-alpha-kNN`: tune `(alpha, k)` riêng theo missingness pattern khi đủ dữ liệu.

Lý do chính:

- Là phương án duy nhất trong ba bài xử lý zero trực tiếp mà không cần pseudo-count ở bước baseline.
- Phân bổ đúng phần khối lượng còn thiếu trên simplex.
- Có ba biến thể cụ thể, phù hợp yêu cầu chạy lại ba thuật toán rồi chọn kết quả tốt nhất.
- Có chiến lược cross-validation để tune `k` và `alpha`.
- Kết quả được dùng làm magnitude warm-start cho target-nonzero, không làm nhãn đúng cho ZINB hoặc GAN.

### Giới hạn phải ghi rõ trong báo cáo

Tsagris là phương pháp **zero-tolerant**, không phải mô hình xác suất zero-inflated. Phương pháp giả định MAR, không phân biệt structural zero với sampling zero và không mô hình hóa quan hệ thời gian. Do đó:

- Zero quan sát được phải được khóa, không được sửa.
- Đánh giá baseline phải có thêm time-block validation.
- Temporal learning được giao cho DeepMicroGen ở giai đoạn sau.
- Kết quả không được tuyên bố là xử lý MNAR nếu chưa có sensitivity analysis riêng.

## 4. Thiết kế framework

### 4.1 Cấu trúc repository đích

```text
code-missing-imputation/
├── IMPLEMENTATION_PLAN.md
├── README.md
├── pyproject.toml
├── Makefile
├── configs/
│   ├── data.yaml
│   ├── baseline/
│   │   ├── jsd_knn.yaml
│   │   ├── jsd_alpha_knn.yaml
│   │   └── adaptive_jsd_alpha_knn.yaml
│   ├── zero_state/
│   │   └── zinb.yaml
│   └── gan/
│       ├── reproduction.yaml
│       └── covariants.yaml
├── data/
│   └── covariants.csv
├── src/missing_imputation/
│   ├── cli.py
│   ├── data/
│   │   ├── schema.py
│   │   ├── loader.py
│   │   ├── masks.py
│   │   ├── panel.py
│   │   └── closure.py
│   ├── baselines/
│   │   ├── jsd.py
│   │   ├── frechet.py
│   │   ├── jsd_knn.py
│   │   ├── jsd_alpha_knn.py
│   │   └── adaptive_jsd_alpha_knn.py
│   ├── zero_state/
│   │   ├── zinb.py
│   │   ├── posterior.py
│   │   ├── sampler.py
│   │   └── calibration.py
│   ├── deepmicrogen/
│   │   ├── preprocessing.py
│   │   ├── feature_extractor.py
│   │   ├── generator.py
│   │   ├── discriminator.py
│   │   ├── losses.py
│   │   ├── trainer.py
│   │   └── postprocessing.py
│   ├── pipeline/
│   │   ├── run_baselines.py
│   │   ├── select_baseline.py
│   │   ├── run_zinb.py
│   │   ├── fuse_initializations.py
│   │   └── refine.py
│   ├── evaluation/
│   │   ├── splits.py
│   │   ├── metrics.py
│   │   └── report.py
│   └── tracking/
│       ├── manifest.py
│       └── convergence.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── parity/
│   └── fixtures/
├── reports/
│   ├── baseline_selection.md
│   ├── zero_state_evaluation.md
│   ├── fusion_audit.md
│   ├── deepmicrogen_parity.md
│   └── final_evaluation.md
└── artifacts/
    └── manifests/
```

Code hiện có trong `src/variables` và `src/engines/FCS` được giữ nguyên trong giai đoạn đầu, nhưng không được dùng ngầm trong pipeline mới khi chưa có test. Việc xóa hoặc di chuyển code cũ phải là một PR riêng.

### 4.2 Data contract nội bộ

Hai biểu diễn được dùng song song:

- Count table: dữ liệu bàn giao, giữ `total_sequence`, count nguyên và `other`.
- Proportion tensor: `count / total_sequence`, dùng cho JSD và CLR/GAN.

Tensor longitudinal chuẩn:

```text
X_raw             : [location, time, feature], còn NaN
M_observed        : 1 nếu X_raw được quan sát, gồm cả zero và positive
M_target          : 1 nếu X_raw là NaN; M_target = 1 - M_observed
M_artificial_zero : observed zero/positive cells che giả lập để calibrate ZINB
M_artificial_mag  : observed positive cells che giả lập để train/evaluate GAN magnitude
M_target_zero     : target cells được ZINB sample/gate là zero
M_target_nonzero  : target cells được ZINB sample/gate là non-zero
M_fixed           : M_observed OR M_target_zero
M_gan             : M_target_nonzero
M_row             : [location, time], 1 nếu dòng có trong CSV gốc
M_padding         : [location, time], 1 nếu là padding nội bộ
delta_f, delta_b  : time gap thuận/nghịch
```

Các đẳng thức bắt buộc:

```text
M_observed AND M_target = 0
M_observed OR  M_target = 1
M_target_zero AND M_target_nonzero = 0
M_target_zero OR  M_target_nonzero = M_target
M_fixed AND M_gan = 0
```

Hai artificial masks chỉ được lấy từ `M_observed`, tạm che trong một batch rồi khôi phục raw value sau khi tính loss. `M_artificial_mag` còn phải thỏa `X_raw > 0`, vì GAN không chịu trách nhiệm học zero state. Chúng không làm thay đổi `M_observed`, `M_target` hoặc output dataset.

Toàn bộ location được reindex nội bộ lên lưới 14 ngày. Dòng bổ sung chỉ phục vụ mô hình; output mặc định chỉ trả lại các khóa có trong CSV gốc. Tùy chọn `--emit-grid` mới được phép xuất thêm các dòng thời gian chưa có trong dữ liệu gốc.

### 4.3 Nhánh A - Tsagris tạo `dataset_00_tsagris`

1. Validate schema, kiểu dữ liệu, uniqueness và closure lower bound.
2. Chuyển observed counts thành proportions bằng `total_sequence`.
3. Với 516 dòng đầy đủ, tính `other` chính xác và dùng làm neighbor pool.
4. Với dòng thiếu, coi tất cả component missing và `other` là phần cần phân bổ trong `remaining`.
5. Chạy ba biến thể Tsagris độc lập với cùng split và seed list.
6. Tune hyperparameter bằng repeated masked cross-validation mô phỏng missingness pattern thực tế.
7. Chọn thuật toán theo rule ở Mục 6.
8. Chuyển proportion về count bằng largest-remainder allocation chỉ trên các ô missing và `other`.
9. Giữ nguyên observed counts và kiểm tra invariant.
10. Xuất `dataset_00_tsagris.csv`, `M_observed.npz`, `M_target.npz`, metrics và manifest.

`dataset_00_tsagris` phải full nhưng **không phải ground truth**. Giá trị Tsagris tại `M_target` chỉ là candidate magnitude/warm-start; không được dùng để fit ZINB, làm reconstruction target hay tính supervised loss.

Fallback của adaptive algorithm:

- Chỉ tune riêng một missingness pattern khi pattern có đủ số hàng complete/masked theo `min_pattern_support` trong config.
- Nếu không đủ support, dùng `(alpha, k)` global của `JSD-alpha-kNN` và ghi rõ fallback trong manifest.
- Không âm thầm bỏ qua pattern hiếm.

### 4.4 Nhánh B - ZINB tạo `dataset_01_zinb` và zero-state masks

#### 4.4.1 Mục tiêu thống kê

Fit một ZINB riêng cho mỗi variant trên **các cell thuộc `M_observed` duy nhất**. Với count `Y`, ZINB có hai quá trình:

```text
pi(x)  = P(structural zero | context)
Y | non-structural ~ NegativeBinomial(mu(x), theta)
```

`log(total_sequence)` phải được dùng làm offset/exposure hoặc một predictor được kiểm soát tương đương, vì xác suất sampling zero phụ thuộc sequencing depth. Context tối thiểu gồm variant, date/time basis, location representation và temporal neighbor features không gây leakage.

Với một target cell chưa quan sát, predictive state probabilities là:

```text
P(structural) = pi
P(sampling)   = (1 - pi) * NB(Y = 0 | mu, theta)
P(non-zero)   = (1 - pi) * (1 - NB(Y = 0 | mu, theta))
```

Ba xác suất phải hữu hạn, không âm và tổng bằng 1. Đây là posterior/predictive state, không phải nhãn sinh học chắc chắn.

#### 4.4.2 Fit, calibration và sampling

1. Fit ZINB chỉ bằng raw observed cells; không đọc giá trị Tsagris tại `M_target`.
2. Observed positive count chắc chắn thuộc count component. Observed zero đóng góp qua mixture likelihood, không được gán cứng structural hay sampling.
3. Dùng `M_artificial_zero` để che observed zero/positive cells, dự đoán lại zero/non-zero và đo NLL, Brier score, calibration curve, precision/recall/F1.
4. Tune model/threshold trên validation masks; test masks phải được giữ kín.
5. Với mỗi `M_target=1`, lưu `P(structural)`, `P(sampling)`, `P(non-zero)`.
6. Sample state bằng seed cố định:
   - structural hoặc NB draw bằng 0 -> `M_target_zero=1`, output bằng 0;
   - NB draw dương -> `M_target_nonzero=1`.
7. Project sampled states về miền khả thi theo từng dòng trước khi tạo mask chính thức:
   - đặt `remaining = total_sequence - sum(observed counts)`;
   - mỗi target-nonzero count nguyên cần tối thiểu 1;
   - nếu số sampled non-zero lớn hơn `remaining`, giữ các cell có `P(non-zero)` cao nhất trong budget và override phần còn lại về target-zero;
   - lưu cả `sampled_state_raw` và `sampled_state_feasible`; không override âm thầm.
8. Tạo `dataset_01_zinb` full/closed: khóa observed, khóa target-zero bằng 0, cấp tối thiểu 1 cho feasible target-nonzero, phân bổ phần dư theo positive ZINB draws và để `other` hấp thụ residual.
9. Xuất `dataset_01_zinb.csv`, zero-state posterior, hai state masks, seed, feasibility overrides và model manifest.

`dataset_01_zinb` là một full stochastic draw phục vụ gating/audit và phải thỏa closure sau feasibility projection. Positive count do ZINB draw ra **không được dùng làm magnitude cuối**. Để tránh quyết định phụ thuộc một draw may rủi, evaluation phải chạy nhiều seed hoặc so sánh thêm mode MAP/threshold; mỗi draw tạo một imputation lineage riêng.

#### 4.4.3 Structural và sampling zero

- Với observed zero, ZINB có thể tính posterior structural-vs-sampling sau khi đã thấy `Y=0`.
- Với target chưa quan sát, framework chỉ có predictive probabilities của ba outcome và một sampled state.
- Cả structural zero và sampling zero đều có output count bằng 0, nên chúng được gộp vào `M_target_zero` để khóa magnitude. Phân biệt hai loại vẫn được lưu trong posterior để nghiên cứu và audit.
- Không báo cáo accuracy structural-vs-sampling nếu không có nhãn ngoại sinh; chỉ báo calibration zero-vs-non-zero và phân bố posterior.

### 4.5 Hợp nhất `dataset_00` và `dataset_01` thành `dataset_0_fused`

Đây là contract quyết định duy nhất cho fusion:

| Vùng | Nguồn giá trị trong `dataset_0_fused` | Trạng thái | GAN được sửa? |
|---|---|---|---:|
| `M_observed=1`, raw positive | Raw data | Fixed observed | Không |
| `M_observed=1`, raw zero | Raw data bằng 0 | Fixed observed zero | Không |
| `M_target=1` và `M_target_zero=1` | Ép bằng 0 theo ZINB sampled gate | Fixed inferred zero trong lineage | Không |
| `M_target=1` và `M_target_nonzero=1` | Giá trị từ `dataset_00_tsagris` | Warm-start magnitude | Có |

Công thức cell-wise:

```text
X_fused = where(M_observed, X_raw,
           where(M_target_zero, 0,
             X_tsagris))
```

Sau fusion phải thực hiện constrained closure theo từng dòng:

1. Khóa raw observed counts và target-zero ở 0.
2. Tính `remaining = total_sequence - sum(fixed observed variant counts)`.
3. Cấp tối thiểu 1 cho mỗi feasible `M_target_nonzero`, rồi rescale/integerize phần magnitude Tsagris còn lại cùng thành phần `other` trong budget `remaining`.
4. Nếu không có target-nonzero, toàn bộ phần dư đi vào `other`.
5. Không được thay observed count để cứu closure; conflict phải fail fast và ghi audit report.

Artifacts bắt buộc:

- `dataset_0_fused.csv`.
- `M_fixed.npz`, `M_gan.npz`, `M_target_zero.npz`, `M_target_nonzero.npz`.
- `cell_provenance.parquet` với source enum: `RAW_OBSERVED`, `ZINB_ZERO`, `TSAGRIS_WARM_START`, `GAN_ROUND_i`.
- `fusion_audit.md` kiểm tra truth-table, closure và checksum của hai parent datasets.

### 4.6 Port DeepMicroGen

Nguồn chuẩn phải được pin theo repository chính thức và commit:

```text
https://github.com/joungmin-choi/DeepMicroGen.git
commit: da2093d3c054dddb29415da8838c351ad7349f8d
```

Implementation mới dùng PyTorch hiện hành nhưng phải có bảng ánh xạ 1:1 với mã TensorFlow gốc:

- CNN feature extractor.
- Forward và backward RNN generator.
- Trainable time-decay fusion.
- Masked blending giữa observed và generated values.
- Adversarial loss.
- Reconstruction loss.
- Forward/backward consistency loss.
- Discriminator real/fake head.
- Discriminator timepoint head.
- CLR preprocessing và inverse/postprocessing.

Hai mode bắt buộc:

- `reproduction`: giữ shape và logic gần mã gốc nhất để chạy parity trên dữ liệu mẫu DeepMicroGen.
- `covariants`: hỗ trợ variable-length panel, cell-level mask và lưới thời gian COVID-19.

Các thay đổi so với paper phải nằm trong `reports/deepmicrogen_parity.md`, không được trộn với phần tái lập mà không ghi chú.

### 4.7 Adapter DeepMicroGen cho Covariants

Các khác biệt bắt buộc phải xử lý:

1. DeepMicroGen gốc chủ yếu mask cả sample/timepoint; Covariants missing theo từng cell. Mọi loss và blending phải hỗ trợ `M_fixed`, `M_gan` và feature-level artificial magnitude mask.
2. DeepMicroGen giả định grid thời gian tương đối đều; Covariants là panel bất cân bằng. Adapter phải reindex lưới 14 ngày, dùng padding mask và time gap thực.
3. DeepMicroGen gốc dùng taxonomy/phylum để nhóm OTU. Covariants không có taxonomy tương đương. Framework phải hỗ trợ:
   - `feature_groups.yaml` nếu nhóm variant đã được chuyên gia duyệt.
   - Một nhóm duy nhất với Spearman ordering làm fallback.
   - Ablation `identity/no-CNN` để xác định CNN có thực sự mang lại lợi ích.
4. CLR không nhận zero. Pseudocount chỉ được dùng bên trong bước GAN, phải được ghi trong config và không làm thay đổi observed zero hoặc ZINB-gated zero ở output.
5. Postprocessing phải project về simplex, khóa toàn bộ `M_fixed` và dùng largest-remainder chỉ trên `M_gan` cùng `other` để trả về count nguyên.

### 4.8 GAN magnitude refinement: `dataset_i -> dataset_(i+1)`

`M_observed`, `M_target_zero`, `M_target_nonzero`, `M_fixed` và `M_gan` được đóng băng trong một ZINB lineage. GAN không được tự đổi gate giữa các vòng. Muốn đổi sampled gate phải tạo lineage mới với seed/model manifest khác.

Ở vòng `i`:

1. Vòng đầu đọc `dataset_0_fused`; vòng sau đọc `dataset_i` cùng toàn bộ parent checksums.
2. Chuyển count sang proportion, zero replacement có kiểm soát và CLR; zero replacement chỉ là biểu diễn nội bộ.
3. Dùng magnitude hiện tại tại `M_gan` làm warm-start/context với stop-gradient.
4. Sinh `M_artificial_mag` chỉ từ observed positive cells (`M_observed=1 AND X_raw>0`) theo random-cell, empirical-pattern và time-block recipes.
5. Train/continue DeepMicroGen; reconstruction MSE chỉ tính tại `M_artificial_mag` với raw observed positive làm ground truth. Không lấy Tsagris, ZINB draw hoặc output GAN vòng trước làm target.
6. Sinh positive magnitude candidate chỉ tại `M_gan`; dùng positive activation/floor để không tự tạo zero.
7. Khóa `M_fixed`, inverse CLR và constrained closure trong remaining budget; không threshold output GAN thành zero vì zero state đã do ZINB quyết định.
8. Ghi `dataset_(i+1)`, provenance `GAN_ROUND_(i+1)`, checkpoint, reconstruction/distribution metrics và manifest.

Điều kiện dừng mặc định:

- validation MSE và JSD không cải thiện quá `min_delta` trong `patience` vòng; hoặc
- thay đổi magnitude trung bình trên `M_gan` giữa hai vòng nhỏ hơn `epsilon`; hoặc
- đạt `max_rounds`.

Checkpoint được chọn trước hết bằng validation magnitude-MSE, sau đó dùng JSD/Wasserstein làm tie-breaker. Không chọn chỉ dựa trên adversarial loss hoặc training loss.

## 5. Phân công cho hai người

Vai trò tạm dùng trong plan:

- **Người 1 - Data, Tsagris & ZINB Zero-state (P1)**.
- **Người 2 - Fusion, DeepMicroGen & Longitudinal GAN (P2)**.

Tên thật có thể thay vào GitHub issue assignee mà không làm đổi dependency hay output.

### Bảng task chính

| ID | Owner | Phụ thuộc | Công việc | Output bàn giao | Tiêu chí nghiệm thu | Ước lượng |
|---|---|---|---|---|---|---:|
| T00 | P1 + P2 | - | Chốt data contract, naming, seed policy, artifact policy và Definition of Done | `docs/adr/0001-data-contract.md`, config schema | Hai người approve ADR; CI đọc được config | 1 ngày |
| T01 | P1 | T00 | Audit `covariants.csv`, tạo schema validator, checksum và báo cáo missing/zero/time gap | `src/.../data/*`, `reports/data_audit.md`, tests | Phát hiện sai schema, duplicate key, count âm, observed sum > total | 1.5 ngày |
| T02 | P1 | T01 | Xây count/proportion, `other`, `M_observed`, `M_target`, artificial masks và closure primitives | `closure.py`, `masks.py`, unit tests | Masks bù nhau; observed zero/positive đều immutable; closure property tests pass | 1.5 ngày |
| T03 | P1 | T02 | Implement `JSD-kNN` theo 6 bước của Tsagris | `jsd.py`, `jsd_knn.py`, tests fixture từ paper | Reproduce ví dụ paper; hỗ trợ zero; không có NaN output | 2 ngày |
| T04 | P1 | T03 | Implement Fréchet mean và `JSD-alpha-kNN` | `frechet.py`, `jsd_alpha_knn.py`, tests | `alpha=1` khớp arithmetic mean; CV chọn được `(alpha,k)` | 2 ngày |
| T05 | P1 | T04 | Implement adaptive algorithm và fallback cho sparse pattern | `adaptive_jsd_alpha_knn.py`, config, tests | Pattern đủ support tune riêng; pattern thiếu support ghi fallback | 2 ngày |
| T06 | P1 | T05 | Repeated masking CV, chọn Tsagris champion và sinh `dataset_00_tsagris` | 3 baseline outputs, `dataset_00_tsagris.csv`, report, manifest | Full, giữ observed, không bị dùng làm ground truth | 2.5 ngày |
| T07 | P1 | T02 | Implement per-variant ZINB với exposure/context và posterior ba trạng thái | `zero_state/zinb.py`, `posterior.py`, tests | Fit chỉ trên observed; probabilities tổng bằng 1; không đọc Tsagris target | 3 ngày |
| T08 | P1 | T07 | Artificial-mask calibration, seeded sampling và sinh `dataset_01_zinb` | sampler, calibration report, posterior/masks, manifest | Zero/non-zero calibrated; structural/sampling không claim accuracy thiếu nhãn | 2 ngày |
| T09 | P2 | T00 | Lập source manifest và đặc tả parity cho DeepMicroGen commit đã pin | `source_manifest.yaml`, `deepmicrogen_mapping.md` | Mọi module/loss gốc có mapping và test plan | 1 ngày |
| T10 | P2 | T09 | Port preprocessing, CNN, biRNN generator, time decay, discriminator và losses sang PyTorch | `src/.../deepmicrogen/*`, unit tests | Shape/loss test pass; không còn phụ thuộc TensorFlow 1.x | 4 ngày |
| T11 | P2 | T10 | Chạy reproduction/parity trên dữ liệu mẫu chính thức | parity tests, `deepmicrogen_parity.md`, checkpoint test | Pipeline gốc và bản port khớp contract; sai khác trong tolerance | 2 ngày |
| T12 | P2 | T01, T10 | Xây panel adapter: lưới 14 ngày, padding, actual delta và cell-level masks | `panel.py`, adapter tests | Hỗ trợ 150 chuỗi; không leakage qua padding | 2.5 ngày |
| T13 | P2 | T02, T06, T08 | Implement fusion truth-table, provenance và sinh `dataset_0_fused` | `fuse_initializations.py`, masks, `fusion_audit.md` | Mỗi cell đúng source; fixed cells bất biến; closure chính xác | 2 ngày |
| T14 | P2 | T11, T12, T13 | Xây CLR/inverse CLR và constrained postprocessing trên `M_gan` | preprocessing/postprocessing, configs, tests | Chỉ `M_gan` thay đổi; không GAN-threshold về zero | 2 ngày |
| T15 | P2 | T14 | Xây gated refinement `dataset_i -> dataset_(i+1)`, checkpoint/resume | `refine.py`, CLI, integration tests | GAN chỉ chạy `M_gan`; resume deterministic; lineage đầy đủ | 3 ngày |
| T16 | P1 | T06, T08, T15 | Evaluation suite: reconstruction và distribution trên bốn split | `evaluation/*`, metrics tables | MSE/JSD/Wasserstein không leakage; cùng split cho mọi method | 2 ngày |
| T17 | P2 | T15, T16 | Tuning/ablation và chạy nhiều GAN rounds/seeds | checkpoints, per-round metrics, candidates | Có MSE/distribution curves, seed variance, stopping reason | 2.5 ngày |
| T18 | P1 | T16, T17 | Đánh giá cuối và chọn artifact | `final_evaluation.md`, bảng so sánh | Không downstream claim; ghi trung thực nếu GAN không hơn fused init | 1.5 ngày |
| T19 | P1 + P2 | T18 | Hardening, README, CI, release notes và GitHub release | release `v1.0.0`, artifacts, checksums | Fresh clone chạy smoke test bằng một command | 2 ngày |

### Cân bằng khối lượng

| Người | Phạm vi chính | Ước lượng riêng |
|---|---|---:|
| P1 | Data/masks, 3 Tsagris algorithms, ZINB zero-state, evaluation | khoảng 20 ngày công, chưa tính task chung |
| P2 | Fusion, DeepMicroGen port, longitudinal adapter, gated GAN | khoảng 19 ngày công, chưa tính task chung |

T00 và T19 là task chung. Mỗi PR của P1 do P2 review và ngược lại; người viết code không tự merge PR của mình.

## 6. Giao thức đánh giá và chọn mô hình

### 6.1 Split bắt buộc

1. `random-cell`: sanity check, không dùng làm kết luận chính.
2. `empirical-pattern`: che giả lập theo phân phối missingness pattern của dữ liệu thật.
3. `time-block`: che một đoạn liên tiếp trong chuỗi của từng location.
4. `country-holdout`: giữ riêng một nhóm location để kiểm tra khả năng tổng quát hóa.

Tất cả method phải dùng chung file split IDs đã version hóa.

### 6.2 Metrics

Không có downstream metric trong v1. Metrics reconstruction chỉ được tính trên observed ground truth bị artificial-mask, không bao giờ tính against Tsagris/ZINB/GAN-imputed target.

- Primary: MSE trên proportions tại `M_artificial_mag`.
- Secondary: RMSE/MAE trên proportions và count; count metrics phân tầng theo `total_sequence`.
- Distribution fidelity: JSD, Wasserstein theo variant, sai khác mean/variance/quantiles/zero prevalence và correlation matrix.
- Temporal fidelity: temporal roughness, lag-1 change và JSD theo time window.
- Zero-state: held-out zero/non-zero NLL, Brier score, calibration error, precision/recall/F1 và predicted zero prevalence.
- Structural-vs-sampling: chỉ báo posterior distribution/sensitivity; không báo accuracy nếu không có nhãn ngoại sinh.

Metrics ràng buộc:

- Tỷ lệ dòng vi phạm closure.
- Tỷ lệ observed cell bị thay đổi.
- Tỷ lệ output âm, không nguyên hoặc NaN.

Mọi report phải tách kết quả của `dataset_00_tsagris`, `dataset_01_zinb`, `dataset_0_fused`, ungated GAN ablation và gated GAN. Không gộp các artifact sơ bộ thành một tên `dataset_0` mơ hồ.

### 6.3 Rule chọn thuật toán tạo `dataset_00_tsagris`

1. Loại ngay method vi phạm invariant.
2. Xếp hạng trước bằng MSE; dùng mean JSD của `empirical-pattern` và `time-block` làm tie-breaker.
3. Nếu chênh lệch MSE/JSD trong khoảng bất định của repeated runs, ưu tiên method đơn giản hơn và nhanh hơn.
4. Không chọn adaptive method chỉ vì tốt trên random-cell nếu kém trên time-block.
5. Ghi cả mean, standard deviation, seed list và runtime.

### 6.4 Rule chấp nhận GAN

Gated GAN chỉ được coi là cải thiện nếu:

- Không vi phạm invariant.
- Validation magnitude-MSE thấp hơn `dataset_0_fused` và ungated GAN ablation, hoặc tương đương trong tolerance nhưng tốt hơn rõ về JSD/Wasserstein.
- Mean JSD trên time-block không tệ hơn `dataset_0_fused` quá tolerance đã chốt trong ADR.
- Zero prevalence của output phù hợp với ZINB gate; GAN không được tự tạo/xóa zero.
- Kết quả ổn định qua tối thiểu 5 seed hoặc có giải thích rõ độ biến thiên.

Nếu GAN không vượt baseline, `dataset_0_fused` vẫn là output hợp lệ; báo cáo không được ép chọn checkpoint GAN kém hơn và không dùng downstream score để đảo quyết định.

## 7. Kiểm thử

### Unit tests

- JSD đối xứng, hữu hạn và bằng 0 khi hai composition giống nhau.
- JSD hoạt động với component bằng 0.
- Fréchet mean tại `alpha=1` bằng arithmetic mean sau closure.
- Ví dụ 5 thành phần trong paper Tsagris cho kết quả gần `(0.20, 0.27, 0.30, 0.10, 0.13)`.
- Largest-remainder giữ count quan sát và tổng chính xác.
- `M_observed`/`M_target` bù nhau; observed zero và positive đều immutable.
- ZINB fit data không chứa bất kỳ Tsagris-imputed target value nào.
- Ba zero-state probabilities hữu hạn, không âm và tổng bằng 1.
- Seeded ZINB sampling tái lập được và sinh hai mask target zero/non-zero bù nhau.
- Fusion truth-table chọn đúng source cho từng cell và ghi đúng provenance.
- CLR/inverse CLR round-trip trong tolerance.
- GAN mask chỉ bằng 1 tại `M_target_nonzero`; fixed cells không nhận gradient/update.
- Time decay và padding không tạo NaN.

### Property tests

- Với nhiều `total_sequence` và missing patterns ngẫu nhiên, output luôn thỏa closure.
- Thay scale count nhưng giữ proportions không làm đổi neighbor ranking ngoài sai số số học.
- Thay positive magnitude trong `dataset_01_zinb` không làm đổi `dataset_0_fused` khi sampled state giữ nguyên.
- Thay Tsagris target values không làm đổi ZINB fit/loss/posterior.
- Checkpoint resume và run liên tục cho cùng output khi dùng deterministic mode.

### Integration tests

- Chạy ba baseline từ raw đến `dataset_00_tsagris`.
- Chạy ZINB từ raw observed đến `dataset_01_zinb` và posterior artifacts.
- Fuse end-to-end thành `dataset_0_fused`; kiểm tra từng source enum.
- Chạy một vòng gated GAN nhỏ `dataset_0_fused -> dataset_1` trên CPU.
- Chứng minh GAN không sửa `M_observed` hoặc `M_target_zero`.
- Chạy full CLI smoke test bằng config tối giản.
- Kiểm tra manifest chain, checksum và provenance.

### Parity tests

- Dùng input mẫu của DeepMicroGen chính thức.
- Đối chiếu preprocessing, tensor shape, mask behavior, từng loss component và output contract.
- Tolerance số học phải được ghi trước khi xem kết quả cuối.

## 8. Output bàn giao

### Output của P1

- Ba implementation Tsagris và test tương ứng.
- Bộ split đánh giá cố định.
- Metrics raw và báo cáo chọn baseline.
- `dataset_00_tsagris.csv` và manifest.
- ZINB model, `dataset_01_zinb.csv`, posterior/calibration report và seeded state masks.
- Reconstruction/distribution evaluation suite và final evaluation report.

### Output của P2

- Source manifest của DeepMicroGen.
- Bản port PyTorch và parity report.
- Covariants panel adapter.
- Fusion/provenance module và `dataset_0_fused.csv`.
- Gated GAN refinement, checkpoint/resume và per-round artifacts.
- Candidate `dataset_1 ... dataset_n` và training metrics.

### Output chung cuối cùng

```text
release/v1.0.0/
├── dataset_00_tsagris.csv
├── dataset_01_zinb.csv
├── dataset_0_fused.csv
├── dataset_final.csv
├── M_observed.npz
├── M_target.npz
├── M_target_zero.npz
├── M_target_nonzero.npz
├── zero_state_posterior.npz
├── cell_provenance.parquet
├── baseline_metrics.csv
├── zero_state_metrics.csv
├── gan_round_metrics.csv
├── reconstruction_distribution_metrics.csv
├── final_evaluation.md
├── run_manifest.json
├── checksums.sha256
└── model_checkpoint.pt
```

Git repository lưu code, config, tests, report và manifest. Checkpoint lớn và các CSV trung gian được đưa vào GitHub Release assets; không commit toàn bộ checkpoint/history vào Git. Ba initialization artifacts, `dataset_final`, masks, posterior và checksums phải cùng xuất hiện trong release để tái lập lineage.

## 9. CLI dự kiến

```bash
# 1. Kiểm tra dữ liệu
python -m missing_imputation audit-data --config configs/data.yaml

# 2. Chạy đủ ba Tsagris baselines
python -m missing_imputation run-baselines --config-dir configs/baseline

# 3. Chọn champion và sinh dataset_00_tsagris + raw masks
python -m missing_imputation select-baseline --metrics artifacts/baselines/metrics.csv

# 4. Fit/calibrate ZINB và sample dataset_01_zinb
python -m missing_imputation run-zinb \
  --raw data/covariants.csv \
  --observed-mask artifacts/M_observed.npz \
  --config configs/zero_state/zinb.yaml \
  --seed 42

# 5. Fuse theo truth-table
python -m missing_imputation fuse-initializations \
  --raw data/covariants.csv \
  --tsagris artifacts/dataset_00_tsagris.csv \
  --zinb artifacts/dataset_01_zinb.csv \
  --posterior artifacts/zero_state_posterior.npz

# 6. Parity DeepMicroGen
python -m missing_imputation deepmicrogen-parity --config configs/gan/reproduction.yaml

# 7. Gated magnitude refinement
python -m missing_imputation refine \
  --input artifacts/dataset_0_fused.csv \
  --fixed-mask artifacts/M_fixed.npz \
  --gan-mask artifacts/M_target_nonzero.npz \
  --config configs/gan/covariants.yaml \
  --max-rounds 10

# 8. Đánh giá reconstruction/distribution và đóng gói release
python -m missing_imputation evaluate --run-dir artifacts/runs/<run_id>
python -m missing_imputation package-release --run-dir artifacts/runs/<run_id>
```

## 10. Quy trình GitHub

### Branch và issue

- Mỗi task `Txx` có một GitHub issue, owner, dependency và checklist output.
- Branch: `feat/Txx-short-name`, `test/Txx-short-name`, `docs/Txx-short-name`.
- Không push trực tiếp vào `main`.
- Mỗi PR chỉ giải quyết một task hoặc một output có thể review độc lập.

### Pull request gate

PR chỉ được merge khi:

1. CI xanh: lint, type check, unit test và smoke test.
2. Có test mới cho behavior mới.
3. Có reviewer là người còn lại.
4. Không commit dataset/checkpoint trung gian ngoài policy.
5. Config và seed được ghi vào manifest.
6. Nếu thay đổi logic paper, PR phải cập nhật parity/deviation report.

### Milestone và release

| Milestone | Nội dung | Tag/Release |
|---|---|---|
| M0 | Data contract và repo scaffold | Không tag |
| M1 | Ba Tsagris baselines + `dataset_00_tsagris` | `v0.1.0-baseline` |
| M2 | DeepMicroGen reproduction/parity | `v0.2.0-deepmicrogen` |
| M3 | ZINB `dataset_01_zinb` + fusion `dataset_0_fused` | `v0.3.0-fusion` |
| M4 | Gated magnitude refinement | `v0.4.0-integration` |
| M5 | Reconstruction/distribution evaluation và handoff | `v1.0.0` |

## 11. Timeline đề xuất

Hai người làm song song, tổng thời gian dự kiến 5 tuần làm việc:

| Tuần | P1 | P2 | Gate cuối tuần |
|---|---|---|---|
| 1 | T00-T03 | T00, T09-T10 | Data/mask contract và core tests |
| 2 | T04-T06, T07 | T10-T12 | Có `dataset_00_tsagris`; DeepMicroGen port gần hoàn tất |
| 3 | T07-T08 | T11-T13 | Có calibrated ZINB, `dataset_01_zinb` và `dataset_0_fused` |
| 4 | T16 | T14-T17 | Chạy được gated `dataset_0_fused -> dataset_1` và ablations |
| 5 | T18-T19 | T17, T19 | Final reconstruction/distribution report và release |

Nếu chỉ có CPU, T17 có thể kéo dài thêm. Smoke/parity phải chạy được trên CPU; full tuning có thể chạy GPU nhưng phải cung cấp config và log đầy đủ.

## 12. Risk register và phương án xử lý

| Risk | Mức | Ảnh hưởng | Biện pháp |
|---|---|---|---|
| Chỉ có 516 dòng complete cho neighbor pool | Cao | Neighbor yếu, adaptive pattern thiếu support | Repeated CV, fallback global, báo cáo coverage theo pattern |
| 95.58% dòng có missing | Cao | Self-training dễ củng cố sai số baseline | Không coi initialization là truth; giữ raw masks bất biến |
| ZINB độc lập theo variant vi phạm closure | Cao | `dataset_01_zinb` có tổng count sai | Chỉ dùng ZINB cho state gate; magnitude ZINB bị bỏ; fusion closure riêng |
| Một ZINB draw làm gate không ổn định | Cao | Lineage đổi mạnh theo seed | Nhiều seed/MAP ablation, lưu posterior và seed, báo uncertainty |
| Structural và sampling zero không identifiable hoàn toàn | Cao | Dễ overclaim biological absence | Chỉ gọi posterior latent; không báo accuracy thiếu nhãn ngoại sinh |
| ZINB convergence/separation ở variant quá thưa | Cao | Fit lỗi hoặc all-zero gate | Regularization/fallback pooled model, diagnostics, fail-fast theo variant |
| ZINB gate nhầm non-zero thành zero | Cao | GAN mất cơ hội reconstruct magnitude | Tune threshold/cost, report recall non-zero, sensitivity analysis |
| Sampled non-zero states vượt remaining count | Cao | Không thể vừa positive integer vừa closure | Feasibility projection theo posterior rank; audit mọi override |
| Fusion rescale làm méo Tsagris warm-start | Trung bình | Initialization distribution đổi | Audit trước/sau closure và giữ `other` làm residual absorber |
| Target-nonzero không có nhiều positive observed analogues | Cao | GAN magnitude học yếu | Per-variant support report, pooling/embedding và uncertainty flags |
| Tsagris không có temporal model | Cao | `dataset_00_tsagris` thiếu mượt theo thời gian | Time-block evaluation; DeepMicroGen refinement |
| DeepMicroGen gốc mask theo sample, dữ liệu mask theo cell | Cao | Port nguyên trạng sẽ sai contract | Feature-level mask tests và parity/deviation report |
| Khoảng thời gian không đều | Cao | Timepoint head và RNN lệch | Reindex lưới 14 ngày, padding mask, actual delta |
| Không có taxonomy cho 17 variant | Trung bình | CNN grouping có thể không có ý nghĩa | Config mapping được duyệt, one-group fallback, no-CNN ablation |
| CLR cần pseudo-count | Cao | Có thể làm sai zero structure | Pseudocount chỉ nội bộ, observed zero khóa cứng, zero metrics |
| GAN không ổn định theo seed | Cao | Khó chọn `dataset_final` | Tối thiểu 5 seed, early stopping, report variance |
| Output integer làm tăng sai số sau inverse CLR | Trung bình | Metrics proportion/count khác nhau | Largest-remainder và báo cáo cả trước/sau integerization |
| MNAR chưa được mô hình hóa | Cao | Không thể khẳng định unbiased | Ghi giới hạn, sensitivity analysis là milestone sau v1 |

## 13. Definition of Done

Dự án chỉ được coi là hoàn tất khi:

- [ ] Có implementation và test cho đủ ba Tsagris algorithms.
- [ ] Có báo cáo định lượng giải thích vì sao chọn thuật toán tạo `dataset_00_tsagris`.
- [ ] `M_observed`/`M_target` tạo từ raw, có checksum và observed zero/positive đều immutable.
- [ ] ZINB fit chỉ trên observed raw data; có calibration và posterior ba trạng thái.
- [ ] `dataset_01_zinb` tái lập theo seed; positive ZINB draw không được dùng làm magnitude cuối.
- [ ] Fusion truth-table được test cho mọi cell và `dataset_0_fused` thỏa closure.
- [ ] Có `cell_provenance` truy được nguồn của từng cell.
- [ ] DeepMicroGen source commit được pin và có parity/deviation report.
- [ ] Chạy được ít nhất một vòng gated `dataset_0_fused -> dataset_1` bằng CLI.
- [ ] GAN chỉ thay `M_target_nonzero`; không sửa observed hoặc target-zero.
- [ ] Vòng lặp có checkpoint, resume và stopping rule.
- [ ] Đánh giá có random-cell, empirical-pattern, time-block và country-holdout.
- [ ] Observed values và observed zeros không bị thay đổi.
- [ ] Có final report MSE/RMSE/MAE và distribution fidelity cho `dataset_00`, `dataset_01`, fused và GAN rounds.
- [ ] Không có downstream metric/claim trong nghiệm thu v1.
- [ ] CI xanh trên `main`.
- [ ] Release GitHub có artifact, checksum, manifest và hướng dẫn tái lập.
- [ ] Một người không viết code chính chạy lại smoke test từ fresh clone thành công.

## 14. Điểm cần xác nhận trước khi bắt đầu code

Các điểm sau là gate của T00, không cản trở việc mở issue và scaffold repo:

1. Tên thật/GitHub username tương ứng với P1 và P2.
2. Có mapping chuyên môn cho nhóm 17 variant hay dùng one-group + ablation.
3. GPU mục tiêu và giới hạn thời gian cho full tuning.
4. Có cần xuất thêm các country-date không tồn tại trong CSV gốc hay chỉ impute cell trong 11,671 dòng hiện có.
5. ZINB gate mặc định dùng stochastic draw, MAP hay calibrated threshold; plan yêu cầu chạy sensitivity cho cả ba nhưng release cần một default.
6. Chi phí ưu tiên của gate: tránh false zero (giữ recall non-zero) hay tránh false non-zero.
7. Số ZINB seeds/multiple-imputation lineages cần bàn giao.
8. Tolerance cụ thể để chấp nhận GAN theo validation MSE, JSD và Wasserstein.

## 15. Nguồn kỹ thuật được dùng để lập plan

- `missing_impute/research/Hron.pdf`: Hron, Templ và Filzmoser, imputation trong Aitchison/ILR space bằng kNN, iterative LS và iterative LTS.
- `missing_impute/research/Sasha Micoda.pdf`: Saha et al., MICoDa V.1/V.2 trong ALR space, điều chỉnh theo covariate.
- `missing_impute/research/tsagris.pdf`: Tsagris, Stewart và Alenazi, ba biến thể JSD-kNN được chọn cho `dataset_00_tsagris`.
- `missing_impute/diffusion based/DeepMicroGen.pdf`: kiến trúc CNN, bidirectional RNN GAN, time-decay, losses và giới hạn irregular sampling.
- `https://github.com/joungmin-choi/DeepMicroGen.git`: source chính thức, pin tại commit `da2093d3c054dddb29415da8838c351ad7349f8d`.
