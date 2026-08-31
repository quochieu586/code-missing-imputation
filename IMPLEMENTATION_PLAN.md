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
3 Tsagris baselines              Reduced/pooled ZINB chỉ trên observed
        |                             |
        v                             v
dataset_00_tsagris               posterior + quality/abstain masks
                                  (+ optional dataset_01_zinb diagnostic)
        |                             |
        +-------------+---------------+
                      v
       Hợp nhất theo observed/confident-zero/non-zero/uncertain masks
                      |
                      v
dataset_0_fused: observed + confident-zero khóa cứng + phần còn lại warm-start Tsagris
                      |
                      v
GAN dự đoán magnitude trên target-nonzero và target-uncertain/abstain
                      |
                      v
dataset_i -> dataset_(i+1), lặp đến điều kiện dừng
```

Mục tiêu của framework được chốt theo thứ tự:

1. Tsagris tạo một tensor full sơ bộ để không còn `NaN` và cung cấp warm-start cho magnitude.
2. Zero-state branch ưu tiên reduced per-variant ZINB; variant thưa dùng pooled ZINB để mượn thông tin.
3. Chỉ model vượt quality gate mới được khóa confident-zero; model yếu phải abstain thay vì ép zero.
4. Mask hợp nhất hai nhánh để observed values và confident-zero không bao giờ bị GAN sửa.
5. GAN reconstruct magnitude của `M_target_nonzero OR M_target_uncertain`, giữ vai trò imputation chính.
6. Đánh giá tập trung vào reconstruction MSE và độ khớp phân phối trước/sau imputation; chưa làm downstream task trong v1.

Sản phẩm cuối phải bao gồm mã nguồn, cấu hình, kiểm thử, báo cáo đánh giá, manifest tái lập và các bộ dữ liệu bàn giao trên GitHub repository:

`https://github.com/quochieu586/code-missing-imputation.git`

### Ngoài phạm vi của phiên bản đầu

- Không triển khai downstream classification, regression, survival hoặc causal task.
- Không dùng Hurdle model.
- Không dùng ZIP/Poisson fallback để tạo hard zero gate; chúng chỉ được phép làm diagnostic baseline.
- Không bắt zero-state branch phải quyết định mọi target; `abstain` là output hợp lệ và được chuyển sang GAN.
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
M_target   = 1 nếu raw value là NaN trên một hàng raw tồn tại; ngược lại bằng 0
```

`M_observed` chứa cả observed positive count và observed zero. Cả hai đều là dữ liệu thật được ghi nhận, phải được giữ nguyên tuyệt đối. `M_target` chỉ chứa các cell ban đầu là `NaN`; đây là miền duy nhất được phép impute. Trên miền raw rows, `M_target = 1 - M_observed`; trên padding, cả hai bằng 0.

Hai mask này phải được tạo trực tiếp từ raw CSV trước mọi phép điền, lưu thành artifact có checksum và không bao giờ suy lại từ một dataset full.

### Invariant bắt buộc cho mọi dataset full

1. Không còn `NaN` trong 17 cột variant.
2. Mọi count là số nguyên không âm.
3. Mọi giá trị quan sát ban đầu được giữ nguyên tuyệt đối.
4. `sum(17 variants) + other == total_sequence` trên từng dòng.
5. Khóa `(location, date)` là duy nhất và thứ tự dòng đầu ra ổn định.
6. `M_observed` và `M_target` được lưu riêng, bù nhau hoàn toàn trên raw rows, cùng bằng 0 trên padding và không suy lại từ `dataset_i`.
7. Mỗi artifact có checksum, config, seed, Git commit và source dataset checksum.
8. Mọi `M_observed=1`, kể cả observed zero, phải bằng raw data ở từng bit/count.
9. Chỉ `M_target_nonzero OR M_target_uncertain` được phép nhận output magnitude từ GAN.
10. Mọi target cell được quality-gated là confident-zero phải luôn bằng 0 trong lineage tương ứng.
11. Model không đạt quality gate không được tạo confident-zero; các cell đó phải thuộc `M_target_uncertain`.

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
│   │   ├── design.py
│   │   ├── zinb.py
│   │   ├── pooled_zinb.py
│   │   ├── routing.py
│   │   ├── posterior.py
│   │   ├── gating.py
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
│   │   ├── run_zero_state.py
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
M_target          : 1 nếu X_raw là NaN trên raw row; padding luôn bằng 0
M_artificial_zero : observed zero/positive cells che giả lập để calibrate ZINB
M_artificial_mag  : observed positive cells che giả lập để train/evaluate GAN magnitude
M_target_zero     : target cells được model đạt quality gate khóa confident-zero
M_target_nonzero  : target cells có bằng chứng non-zero đủ mạnh
M_target_uncertain: target cells model abstain/không đủ tin cậy
M_fixed           : M_observed OR M_target_zero
M_gan             : M_target_nonzero OR M_target_uncertain
M_row             : [location, time], 1 nếu dòng có trong CSV gốc
M_padding         : [location, time], 1 nếu là padding nội bộ
delta_f, delta_b  : time gap thuận/nghịch
```

Các đẳng thức bắt buộc:

```text
M_observed AND M_target = 0
M_observed OR M_target = broadcast(M_row) AND NOT broadcast(M_padding)
pairwise_disjoint(M_target_zero, M_target_nonzero, M_target_uncertain)
M_target_zero OR M_target_nonzero OR M_target_uncertain = M_target
M_fixed AND M_gan = 0
```

Hai artificial masks chỉ được lấy từ `M_observed`, tạm che trong một batch rồi khôi phục raw value sau khi tính loss. `M_artificial_mag` còn phải thỏa `X_raw > 0`, vì GAN không chịu trách nhiệm học zero state. Chúng không làm thay đổi `M_observed`, `M_target` hoặc output dataset.

`M_target` phải được tạo từ các `NaN` thật trên những hàng tồn tại trong raw CSV:

```text
M_target = (X_raw is NaN) AND M_row AND NOT M_padding
```

Padding trên lưới 14 ngày là context nội bộ, không phải missing target. Số cell của `M_target` phải bằng đúng số `NaN` trong 17 cột variant của raw CSV (hiện tại 86,050), không phải toàn bộ ô 0 trong tensor reindex. Output mặc định chỉ trả lại các khóa có trong CSV gốc; tùy chọn `--emit-grid` là một bài toán forecasting/interpolation riêng và nằm ngoài v1.

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

### 4.4 Nhánh B - Reduced/Pooled ZINB với abstain-to-GAN

#### 4.4.1 Vai trò giới hạn của zero-state branch

Zero-state branch chỉ trả lời:

> Có đủ bằng chứng xác suất và calibration để khóa target cell này bằng 0 hay không?

Nó không phải mô hình imputation magnitude chính. ZINB không cung cấp positive magnitude cho fusion; Tsagris cung cấp warm-start và GAN học magnitude cuối. Với count `Y`:

```text
pi(x) = P(structural zero | observed context)
Y | non-structural ~ NegativeBinomial(mu(x), theta)

P(structural) = pi
P(sampling)   = (1 - pi) * NB(Y = 0 | mu, theta)
P(non-zero)   = (1 - pi) * (1 - NB(Y = 0 | mu, theta))
```

Ba xác suất phải hữu hạn, không âm và tổng bằng 1. Structural/sampling là latent posterior, không phải ground-truth biological labels.

#### 4.4.2 Reduced per-variant ZINB

Không dùng 149 location fixed effects trong per-variant model. Mỗi variant chỉ dùng khoảng 8-12 predictors để tránh separation và non-identifiability.

Zero-inflation component:

```text
logit(pi_itj) = gamma_0j
              + natural_spline(global_day_index, df=3..5)
              + gamma_depth * log(total_sequence_it)
              + gamma_lag  * I(observed y_i,t-1,j > 0)
              + gamma_lead * I(observed y_i,t+1,j > 0)
              + gamma_avail_lag  * I(lag is observed)
              + gamma_avail_lead * I(lead is observed)
              + optional time-gap terms
```

Count component:

```text
log(mu_itj) = offset(log(total_sequence_it))
            + beta_0j
            + natural_spline(global_day_index, df=3..5)
            + beta_lag  * log1p(observed lag count)
            + beta_lead * log1p(observed lead count)
            + beta_availability/time-gap terms
```

Quy tắc predictor:

- `global_day_index` tính từ ngày đầu toàn dataset, không reset theo location.
- Lag/lead chỉ được dùng nếu cell lân cận thuộc `M_observed`; Tsagris/ZINB/GAN values không bao giờ làm predictor khi fit/calibrate.
- Khi lag/lead không observed, magnitude predictor được đặt giá trị trung tính và availability indicator bắt buộc bằng 0.
- `log(total_sequence)` là exposure offset trong count component; trong zero component nó có thể là predictor để học sampling-depth effect.
- Không dùng toàn bộ 16 variant còn lại làm predictors trong MVP.
- L1 regularization dùng native `statsmodels.fit_regularized`; penalty tune trong training/validation folds, không tune trên test.

#### 4.4.3 Phân tầng support và pooled ZINB

Trước fit, tạo `variant_support_report.csv` gồm: observed count, observed zero, observed positive, positive rate, số location có positive, events-per-parameter và missing rate.

Routing mặc định:

| Support | Model thử đầu tiên | Ghi chú |
|---|---|---|
| `n_positive > 500` và đủ location support | Reduced per-variant ZINB | Model riêng từng variant |
| `200 <= n_positive <= 500` | Reduced per-variant ZINB | Chỉ dùng nếu vượt quality gate |
| `n_positive < 200` hoặc events-per-parameter thấp | Pooled sparse-variant ZINB | Không cố fit model riêng lớn |

Các ngưỡng là default khởi đầu, phải được version hóa trong config và kiểm tra sensitivity. Pooled ZINB stack các sparse variants thành long table, thêm categorical variant intercept trong cả zero và count components, nhưng chia sẻ spline/exposure/temporal coefficients. Đây vẫn là ZINB, không phải Hurdle.

Nếu reduced per-variant ZINB không vượt quality gate, thử pooled ZINB. Nếu pooled ZINB vẫn không vượt quality gate, model **abstain** cho variant/cell liên quan; không fallback sang ZIP/Poisson để tạo hard gate.

#### 4.4.4 Out-of-fold calibration và quality gate

Ba recipe bắt buộc: random-cell, empirical-pattern và time-block. Với mỗi fold:

1. Tạo `M_artificial_zero` từ observed zero và positive cells.
2. Refit model từ đầu chỉ trên train fold.
3. Recompute lag/lead features sau khi áp fold mask; held-out counts không được xuất hiện trong neighbor predictors.
4. Tune L1/predictor tier chỉ bằng train/validation.
5. Predict held-out fold chưa từng được model thấy.
6. Gom out-of-fold predictions để tune threshold.
7. Đóng băng model specification và threshold trước final test.
8. Refit final model trên toàn observed data chỉ sau khi evaluation đã hoàn thành.

Metrics bắt buộc: NLL, Brier score, calibration curve/ECE, precision, recall và F1 cho zero-vs-non-zero. Threshold không tối ưu F1 đơn thuần; ưu tiên non-zero recall cao vì false zero sẽ chặn GAN. Default acceptance target là non-zero recall >= 0.95, sau đó chọn threshold có precision/calibration tốt nhất trong miền đạt recall; ngưỡng phải cấu hình được.

Một model chỉ `PASS` khi:

- optimizer hội tụ và tham số hữu hạn;
- design matrix không rank-deficient; dispersion hợp lệ;
- out-of-fold NLL/Brier tốt hơn prevalence-only baseline;
- non-zero recall đạt ngưỡng;
- prediction không collapse thành all-zero/all-non-zero;
- prediction schema đúng bằng fit schema.

Cờ `converged=True` một mình không đủ để chấp nhận model.

#### 4.4.5 Gate có khả năng abstain

Release mặc định dùng calibrated deterministic threshold theo variant/model tier:

```text
if model_quality != PASS:
    state = UNCERTAIN
elif P(non-zero) >= threshold_nonzero:
    state = NONZERO
elif P(zero) >= threshold_zero:
    state = CONFIDENT_ZERO
else:
    state = UNCERTAIN
```

Trong đó `P(zero) = P(structural) + P(sampling)`. MAP và stochastic draws chỉ dùng sensitivity analysis, không phải release default.

- `CONFIDENT_ZERO` -> `M_target_zero=1`, khóa 0.
- `NONZERO` -> `M_target_nonzero=1`, Tsagris warm-start rồi GAN.
- `UNCERTAIN/ABSTAIN` -> `M_target_uncertain=1`, Tsagris warm-start rồi GAN, đồng thời giữ uncertainty flag.

Feasibility projection chỉ được phép giảm số `NONZERO` vượt count budget. Cell bị override không được chuyển thành confident zero; nó phải chuyển thành `UNCERTAIN` để GAN/fusion xử lý và audit.

#### 4.4.6 Artifacts và model schema

Artifacts bắt buộc:

- `zero_state_posterior.npz` với ba probabilities và model-quality flag.
- `M_target_zero.npz`, `M_target_nonzero.npz`, `M_target_uncertain.npz`.
- `variant_support_report.csv`, `model_routing.csv`, `oof_predictions.parquet`.
- `zero_state_metrics.csv`, calibration curves, thresholds theo variant/tier.
- `design_schema.json` cho từng fitted model: predictor names/order, spline knots/basis metadata, centering/scaling, offset policy, variant levels và model tier.
- `model_manifest.json` với optimizer status, QC, fallback/routing reason, checksums và Git commit.

Fit, posterior, calibration và inference phải gọi cùng một `DesignEncoder.fit/transform`; không được tự build lại matrix bằng hard-coded spline df hoặc full predictor set. `dataset_01_zinb.csv` có thể xuất như diagnostic artifact để tương thích lineage, nhưng positive ZINB draws bị loại khỏi critical path và không phải input magnitude cho fusion.

#### 4.4.7 Acceptance gate từ log thực tế

Run hiện tại cho thấy cả 17 variant rơi xuống Poisson sau khi regularized ZINB với khoảng 153 parameters thất bại QC; vì vậy run này không được coi là zero-state ZINB hợp lệ. Trước khi cho phép fusion, pipeline phải kiểm tra:

- `M_target.sum()` bằng đúng số raw NaN variant cells và loại `M_padding`;
- không có variant nào dùng ZIP/Poisson để tạo `M_target_zero`;
- calibration là out-of-fold refit thật;
- threshold theo variant được truyền vào gate, không hard-code `0.5`;
- transform matrix có đúng số/tên cột của fitted model;
- mọi zero-state artifact tồn tại và checksum hợp lệ; nếu stage thất bại, fusion phải fail fast với lỗi parent-stage, không chạy tiếp rồi báo thiếu file.

Support quan sát giải thích vì sao cần routing:

| Variant ví dụ | Observed positive | Hướng mặc định |
|---|---:|---|
| `S:677` | 95 | Pooled sparse-variant ZINB |
| `20C` | 112 | Pooled sparse-variant ZINB |
| `Kappa` | 180 | Pooled sparse-variant ZINB |
| `Iota` | 199 | Pooled sparse-variant ZINB |
| `Epsilon` | 249 | Thử reduced per-variant, fail thì pooled |
| `Alpha` | 1,894 | Reduced per-variant ZINB |
| `Delta` | 2,968 | Reduced per-variant ZINB |
| `Omicron` | 6,783 | Reduced per-variant ZINB |

Log cũng cho thấy lỗi prediction `size 5 is different from 154`: fallback Poisson được fit với design rút gọn nhưng sampler dựng lại full design. Đây là lý do `DesignEncoder` và schema assertions là release blocker, không phải cải tiến tùy chọn.

#### 4.4.8 Debug protocol sau run `zinb_1788164041`

Run hoàn thành nhưng cả 17 variants `FAIL` quality gate và toàn bộ target được gán `UNCERTAIN`. Đây là **fail-safe đúng** của abstain contract, không phải lý do để nới quality gate. Tuy nhiên, chưa được kết luận ZINB không phù hợp vì run còn lỗi target indexing, rank deficiency và pooled OOF.

Thứ tự debug bắt buộc:

**D0 - Sửa miền target trước mọi model run**

1. Không dùng `target_mask = M_target.any(axis=2)` rồi broadcast cho mọi variant.
2. Posterior/gating của variant `j` phải nhận đúng `M_target[:, :, j]`.
3. `M_target.sum()` phải bằng 86,050 raw NaN variant cells; observed cells trên một incomplete row không được biến thành target.
4. Padding phải bằng 0 trong cả `M_observed` và `M_target`.
5. Raw `covariants.csv` hiện không có dòng `total_sequence=0`; diagnostic/report code phải phân biệt “all 17 variant counts bằng 0” với `total_sequence=0`.

Con số 276,828 trong run bằng `16,284 incomplete location-times x 17 variants`, còn 8,772 bằng `516 complete rows x 17`. Đây là bằng chứng target mask đã được broadcast theo row và là release blocker độc lập với ZINB.

**D1 - Chốt spline/intercept convention full-rank**

DesignEncoder dùng một convention duy nhất:

```text
Patsy formula: cr(global_day_index, df=4, constraints="center")
Không thêm intercept thủ công; dùng intercept do Patsy tạo.
```

Centered constraint loại constant direction khỏi spline. Trên fixture chuẩn, `[Intercept + centered natural spline]` phải full column rank. `df=3/4/5` được kiểm tra trong validation, nhưng mọi candidate phải qua rank assertion trước khi fit.

Sau khi thêm lag/lead features:

- Drop zero-variance columns theo **training fold** (trừ intercept/offset policy).
- Dùng pivoted QR hoặc SVD để phát hiện near-collinearity; lưu `active_columns`, dropped columns, rank, singular values và condition number vào schema/fold diagnostics.
- `DesignEncoder.transform` chỉ xuất đúng active columns đã học từ train; không tự quyết định lại trên validation/test.

**D2 - Chốt pooled variant coding và rank check đúng matrix**

Pooled ZINB dùng `K-1` effect/treatment-coded variant columns với một reference level đã serialize; không append đủ `K` one-hot columns cùng intercept.

OOF pooled rank check phải chạy trên **actual stacked pooled training matrix** chứa tất cả pooled variants của fold. Không được dựng matrix chỉ từ một variant với một dummy toàn 1, vì matrix đó collinear với intercept và không phải matrix đã dùng để fit.

**D3 - OOF refit và baseline công bằng**

Mỗi fold phải:

1. Refit reduced model hoặc rebuild/refit toàn bộ pooled long-table model từ scratch.
2. Recompute lag/lead sau khi áp fold mask.
3. Fit `DesignEncoder` chỉ trên fold train và transform validation/test bằng schema đó.
4. Tính prevalence baseline từ fold train rồi dự đoán fold test; không dùng prevalence từ toàn observed dataset để so với OOF model.
5. Lưu cell-level `y_true`, raw/calibrated `p_nonzero`, baseline probability, recipe, fold và indices vào `oof_predictions.parquet`.

Fold fit thất bại không được âm thầm bỏ khỏi aggregate. Thêm `fold_coverage_ok`: default yêu cầu ít nhất 80% folds thành công và mỗi recipe có số fold hợp lệ tối thiểu cấu hình được. Ví dụ Alpha chỉ thành công 11/15 folds phải bị flag coverage thay vì tính metric như thể đủ 15 folds.

**D4 - Threshold và metric semantics**

- Threshold được chọn từ OOF predictions theo recall constraint phải được serialize và dùng thật tại inference.
- Các cột report phải đặt tên rõ: `frac_nonzero_at_selected_threshold` và `frac_nonzero_at_0_5`; không dùng chung một tên cho hai threshold.
- Quality gate so sánh OOF model NLL/Brier với OOF prevalence baseline trên đúng cùng test cells.
- Không nới `recall >= 0.95`, NLL/Brier hoặc rank gate trước khi D0-D3 pass.

**D5 - Rerun theo phạm vi tăng dần**

1. Unit test encoder rank với df 3/4/5, constant lag/lead và unseen transform.
2. Smoke một high-support reduced variant (`Omicron` hoặc `Delta`).
3. Smoke một pooled sparse group (`S:677`, `20C`, `Kappa`, `Iota`).
4. Chạy một recipe/2 folds để xác minh OOF refit và artifacts.
5. Chạy đủ 3 recipes x 5 folds x 17 variants chỉ sau khi các smoke gates pass.

Sau khi D0-D5 hoàn tất:

- Nếu rank full và NLL/Brier cải thiện: cho model tiếp tục qua recall/quality gate.
- Nếu rank full nhưng calibration kém: kiểm tra predictor/spline ablation và một probability calibrator được fit trong nested validation; không calibrate trên final test.
- Nếu discrimination/calibration vẫn không hơn prevalence baseline: giữ toàn bộ cell liên quan ở `UNCERTAIN` và chuyển GAN. Đây là kết quả khoa học hợp lệ, không phải pipeline failure.

**D6 - Logging và artifact correctness**

- `n_iterations` không truy xuất được từ statsmodels phải ghi `null/unknown`, không ghi sai là `0`; lưu optimizer metadata khả dụng (`mle_retvals` hoặc equivalent) nếu có.
- Zero-state stage phải phát hành manifest `SUCCESS/FAILED`. Fusion chỉ chạy khi parent manifest `SUCCESS` và mask/posterior checksums hợp lệ.
- Khi tất cả models abstain, integration test phải xác minh `M_target_uncertain == M_target`, `M_target_zero == 0`, `M_target_nonzero == 0`; fused target values lấy Tsagris và observed values giữ nguyên.

### 4.5 Hợp nhất Tsagris và zero-state posterior thành `dataset_0_fused`

Đây là contract quyết định duy nhất cho fusion:

| Vùng | Nguồn giá trị trong `dataset_0_fused` | Trạng thái | GAN được sửa? |
|---|---|---|---:|
| `M_observed=1`, raw positive | Raw data | Fixed observed | Không |
| `M_observed=1`, raw zero | Raw data bằng 0 | Fixed observed zero | Không |
| `M_target=1` và `M_target_zero=1` | Ép bằng 0 theo confident calibrated gate | Fixed inferred zero trong lineage | Không |
| `M_target=1` và `M_target_nonzero=1` | Giá trị từ `dataset_00_tsagris` | Warm-start magnitude | Có |
| `M_target=1` và `M_target_uncertain=1` | Giá trị từ `dataset_00_tsagris` | Abstain warm-start + uncertainty flag | Có |

Công thức cell-wise:

```text
X_fused = where(M_observed, X_raw,
           where(M_target_zero, 0,
             X_tsagris))
```

Sau fusion phải thực hiện constrained closure theo từng dòng:

1. Khóa raw observed counts và target-zero ở 0.
2. Tính `remaining = total_sequence - sum(fixed observed variant counts)`.
3. Cấp tối thiểu 1 cho feasible `M_target_nonzero`; `M_target_uncertain` không bị ép positive trước GAN.
4. Rescale/integerize magnitude Tsagris trên `M_gan` cùng `other` trong budget `remaining`; lưu uncertainty flags độc lập với count warm-start.
5. Nếu không có `M_gan`, toàn bộ phần dư đi vào `other`.
6. Không được thay observed count để cứu closure; conflict phải fail fast và ghi audit report.

Artifacts bắt buộc:

- `dataset_0_fused.csv`.
- `M_fixed.npz`, `M_gan.npz`, `M_target_zero.npz`, `M_target_nonzero.npz`, `M_target_uncertain.npz`.
- `cell_provenance.parquet` với source enum: `RAW_OBSERVED`, `ZINB_CONFIDENT_ZERO`, `TSAGRIS_NONZERO_WARM_START`, `TSAGRIS_ABSTAIN_WARM_START`, `GAN_ROUND_i`.
- `fusion_audit.md` kiểm tra truth-table, closure, abstain rate, quality flags và checksum của parent artifacts.

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
4. CLR không nhận zero. Pseudocount chỉ được dùng bên trong bước GAN, phải được ghi trong config và không làm thay đổi observed zero hoặc confident-zero ở output.
5. Postprocessing phải project về simplex, khóa toàn bộ `M_fixed` và dùng largest-remainder chỉ trên `M_gan` cùng `other` để trả về count nguyên.

### 4.8 GAN magnitude refinement: `dataset_i -> dataset_(i+1)`

`M_observed`, `M_target_zero`, `M_target_nonzero`, `M_target_uncertain`, `M_fixed` và `M_gan` được đóng băng trong một zero-state lineage. GAN không được tự đổi confident-zero gate giữa các vòng. Muốn đổi model/threshold phải tạo lineage mới với manifest khác.

Ở vòng `i`:

1. Vòng đầu đọc `dataset_0_fused`; vòng sau đọc `dataset_i` cùng toàn bộ parent checksums.
2. Chuyển count sang proportion, zero replacement có kiểm soát và CLR; zero replacement chỉ là biểu diễn nội bộ.
3. Dùng magnitude hiện tại tại `M_gan` làm warm-start/context với stop-gradient.
4. Sinh `M_artificial_mag` chỉ từ observed positive cells (`M_observed=1 AND X_raw>0`) theo random-cell, empirical-pattern và time-block recipes.
5. Train/continue DeepMicroGen; reconstruction MSE chỉ tính tại `M_artificial_mag` với raw observed positive làm ground truth. Không lấy Tsagris, ZINB draw hoặc output GAN vòng trước làm target.
6. Sinh magnitude candidate tại `M_gan`; giữ cờ `NONZERO` và `UNCERTAIN` riêng để phân tích sensitivity. Positive floor chỉ bắt buộc cho `M_target_nonzero`; uncertain cells không được diễn giải là proven-positive.
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
| T07 | P1 | T02 | Implement per-variant target indexing, centered-spline `DesignEncoder`, reduced ZINB, support routing và pooled ZINB | encoder/model/routing modules, support report, rank/schema tests | Target=86,050; full-rank centered spline; K-1 pooled coding; fit/predict schema giống hệt | 5 ngày |
| T08 | P1 | T07 | Implement true reduced/pooled OOF refit, fold baseline, quality/coverage gates, thresholds và abstain masks | OOF predictions, diagnostics, posterior, three masks, manifest | Pooled refit mỗi fold; baseline cùng test cells; failed folds counted; model fail -> uncertain | 4 ngày |
| T09 | P2 | T00 | Lập source manifest và đặc tả parity cho DeepMicroGen commit đã pin | `source_manifest.yaml`, `deepmicrogen_mapping.md` | Mọi module/loss gốc có mapping và test plan | 1 ngày |
| T10 | P2 | T09 | Port preprocessing, CNN, biRNN generator, time decay, discriminator và losses sang PyTorch | `src/.../deepmicrogen/*`, unit tests | Shape/loss test pass; không còn phụ thuộc TensorFlow 1.x | 4 ngày |
| T11 | P2 | T10 | Chạy reproduction/parity trên dữ liệu mẫu chính thức | parity tests, `deepmicrogen_parity.md`, checkpoint test | Pipeline gốc và bản port khớp contract; sai khác trong tolerance | 2 ngày |
| T12 | P2 | T01, T10 | Xây panel adapter: lưới 14 ngày, padding, actual delta và cell-level masks | `panel.py`, adapter tests | Hỗ trợ 150 chuỗi; không leakage qua padding | 2.5 ngày |
| T13 | P2 | T02, T06, T08 | Implement parent-manifest gate, four-state fusion, provenance và sinh `dataset_0_fused` | fusion module, masks, `fusion_audit.md` | Fail fast nếu zero-state failed; four states đúng source; closure chính xác | 2.5 ngày |
| T14 | P2 | T11, T12, T13 | Xây CLR/inverse CLR và constrained postprocessing trên `M_gan` | preprocessing/postprocessing, configs, tests | Chỉ `M_gan` thay đổi; không GAN-threshold về zero | 2 ngày |
| T15 | P2 | T14 | Xây gated refinement `dataset_i -> dataset_(i+1)`, checkpoint/resume | `refine.py`, CLI, integration tests | GAN chỉ chạy `M_gan`; resume deterministic; lineage đầy đủ | 3 ngày |
| T16 | P1 | T06, T08, T15 | Evaluation suite: OOF zero-state, abstain coverage, reconstruction và distribution trên bốn split | `evaluation/*`, metrics tables | Không leakage; quality vs coverage; MSE/JSD/Wasserstein dùng cùng split | 2.5 ngày |
| T17 | P2 | T15, T16 | Tuning/ablation và chạy nhiều GAN rounds/seeds | checkpoints, per-round metrics, candidates | Có MSE/distribution curves, seed variance, stopping reason | 2.5 ngày |
| T18 | P1 | T16, T17 | Đánh giá cuối và chọn artifact | `final_evaluation.md`, bảng so sánh | Không downstream claim; ghi trung thực nếu GAN không hơn fused init | 1.5 ngày |
| T19 | P1 + P2 | T18 | Hardening, README, CI, release notes và GitHub release | release `v1.0.0`, artifacts, checksums | Fresh clone chạy smoke test bằng một command | 2 ngày |

### Cân bằng khối lượng

| Người | Phạm vi chính | Ước lượng riêng |
|---|---|---:|
| P1 | Data/masks, Tsagris, reduced/pooled ZINB, abstain calibration, evaluation | khoảng 24.5 ngày công, chưa tính task chung |
| P2 | Four-state fusion, DeepMicroGen port, longitudinal adapter, gated GAN | khoảng 19.5 ngày công, chưa tính task chung |

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
- Zero-state: out-of-fold zero/non-zero NLL, Brier score, calibration error, precision/recall/F1, non-zero recall, confident-zero coverage và abstain rate.
- Structural-vs-sampling: chỉ báo posterior distribution/sensitivity; không báo accuracy nếu không có nhãn ngoại sinh.

Metrics ràng buộc:

- Tỷ lệ dòng vi phạm closure.
- Tỷ lệ observed cell bị thay đổi.
- Tỷ lệ output âm, không nguyên hoặc NaN.

Mọi report phải tách kết quả của `dataset_00_tsagris`, zero-state posterior/routing, optional `dataset_01_zinb` diagnostic, `dataset_0_fused`, ungated GAN ablation và gated GAN. Không gộp các artifact sơ bộ thành một tên `dataset_0` mơ hồ.

### 6.3 Rule chọn thuật toán tạo `dataset_00_tsagris`

1. Loại ngay method vi phạm invariant.
2. Xếp hạng trước bằng MSE; dùng mean JSD của `empirical-pattern` và `time-block` làm tie-breaker.
3. Nếu chênh lệch MSE/JSD trong khoảng bất định của repeated runs, ưu tiên method đơn giản hơn và nhanh hơn.
4. Không chọn adaptive method chỉ vì tốt trên random-cell nếu kém trên time-block.
5. Ghi cả mean, standard deviation, seed list và runtime.

### 6.4 Rule chấp nhận zero-state model

Mỗi reduced/pooled ZINB được so sánh với prevalence-only baseline trên cùng out-of-fold predictions. Chỉ model `PASS` mới được tạo confident-zero.

Thứ tự routing:

1. Reduced per-variant ZINB nếu support đủ.
2. Pooled sparse-variant ZINB nếu support thấp hoặc model riêng fail.
3. Abstain-to-GAN nếu pooled model không đạt quality gate.

Không dùng ZIP, Poisson, all-zero rule hoặc giá trị Tsagris để thay thế một zero-state model không đạt. Report phải biểu diễn quality-coverage curve: zero gate càng bảo thủ thì abstain/GAN coverage càng tăng.

### 6.5 Rule chấp nhận GAN

Gated GAN chỉ được coi là cải thiện nếu:

- Không vi phạm invariant.
- Validation magnitude-MSE thấp hơn `dataset_0_fused` và ungated GAN ablation, hoặc tương đương trong tolerance nhưng tốt hơn rõ về JSD/Wasserstein.
- Mean JSD trên time-block không tệ hơn `dataset_0_fused` quá tolerance đã chốt trong ADR.
- Zero prevalence của output phù hợp với confident-zero gate; uncertain cells được báo cáo riêng, không được diễn giải như zero-state ground truth.
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
- `M_target.sum()` bằng raw variant NaN count; padding không xuất hiện trong target.
- Target coordinates của variant `j` bằng đúng `np.where(M_target[:, :, j])`; không broadcast row-level `any(axis=2)`.
- ZINB fit data không chứa bất kỳ Tsagris-imputed target value nào.
- Ba zero-state probabilities hữu hạn, không âm và tổng bằng 1.
- Reduced model không tạo 149 location dummies; pooled model giữ đúng variant levels.
- Centered natural spline + Patsy intercept full rank cho df 3/4/5; không có explicit duplicate intercept.
- Pooled design dùng K-1 variant columns và full rank trên actual stacked long-table matrix.
- Zero-variance/collinear feature pruning học trên train và được giữ nguyên khi transform test.
- `DesignEncoder.transform` giữ đúng tên/thứ tự/số cột đã fit cho mọi model tier.
- Ba masks confident-zero/nonzero/uncertain pairwise-disjoint và hợp đúng thành `M_target`.
- Model quality fail luôn tạo uncertain, không tạo hard zero.
- ZIP/Poisson diagnostic không được phép ghi vào `M_target_zero`.
- Fusion truth-table chọn đúng source cho từng cell và ghi đúng provenance.
- CLR/inverse CLR round-trip trong tolerance.
- GAN mask bằng `M_target_nonzero OR M_target_uncertain`; fixed cells không nhận gradient/update.
- Time decay và padding không tạo NaN.

### Property tests

- Với nhiều `total_sequence` và missing patterns ngẫu nhiên, output luôn thỏa closure.
- Thay scale count nhưng giữ proportions không làm đổi neighbor ranking ngoài sai số số học.
- Thay positive magnitude trong optional `dataset_01_zinb` không làm đổi masks hoặc `dataset_0_fused`.
- Thay Tsagris target values không làm đổi ZINB fit/loss/posterior.
- Thay padding extent không làm đổi raw target count hoặc zero-state evaluation cells.
- Mọi artificial calibration fold refit model mà không chứa test indices.
- Pooled calibration refit toàn bộ stacked pooled model trong từng fold, không tái sử dụng full-data pooled model.
- OOF prevalence baseline học từ train fold và được đánh giá trên đúng test indices của model.
- Failed folds làm giảm fold coverage; không bị loại âm thầm khỏi aggregate.
- Checkpoint resume và run liên tục cho cùng output khi dùng deterministic mode.

### Integration tests

- Chạy ba baseline từ raw đến `dataset_00_tsagris`.
- Chạy support routing -> reduced/pooled ZINB -> OOF quality gate -> posterior/abstain artifacts.
- Chạy encoder-rank smoke, high-support reduced smoke và sparse pooled smoke trước full 17-variant run.
- Test sparse variant đi qua pooled model; test cả reduced và pooled fail thì chuyển toàn bộ cell liên quan sang uncertain.
- Fuse end-to-end thành `dataset_0_fused`; kiểm tra bốn source states.
- Chạy một vòng gated GAN nhỏ `dataset_0_fused -> dataset_1` trên CPU.
- Chứng minh GAN không sửa `M_observed` hoặc `M_target_zero`, nhưng nhận cả nonzero và uncertain cells.
- Test all-abstain path tạo fused target giống Tsagris và không làm thay observed cells.
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
- Reduced/pooled ZINB models, support/routing report, OOF posterior/calibration, quality flags và three-state operational masks.
- Optional `dataset_01_zinb.csv` diagnostic artifact; positive draws không nằm trên critical path.
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
├── dataset_01_zinb.csv                 # optional diagnostic
├── dataset_0_fused.csv
├── dataset_final.csv
├── M_observed.npz
├── M_target.npz
├── M_target_zero.npz
├── M_target_nonzero.npz
├── M_target_uncertain.npz
├── zero_state_posterior.npz
├── variant_support_report.csv
├── model_routing.csv
├── oof_predictions.parquet
├── design_schemas/
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

Git repository lưu code, config, tests, report và manifest. Checkpoint lớn và các CSV trung gian được đưa vào GitHub Release assets; không commit toàn bộ checkpoint/history vào Git. Tsagris/fused datasets, `dataset_final`, three-state masks, posterior, routing/design schemas và checksums phải cùng xuất hiện trong release để tái lập lineage. `dataset_01_zinb.csv` là optional diagnostic, không phải parent magnitude bắt buộc của fusion.

## 9. CLI dự kiến

```bash
# 1. Kiểm tra dữ liệu
python -m missing_imputation audit-data --config configs/data.yaml

# 2. Chạy đủ ba Tsagris baselines
python -m missing_imputation run-baselines --config-dir configs/baseline

# 3. Chọn champion và sinh dataset_00_tsagris + raw masks
python -m missing_imputation select-baseline --metrics artifacts/baselines/metrics.csv

# 4. Support routing + reduced/pooled ZINB + OOF quality gate
python -m missing_imputation run-zero-state \
  --raw data/covariants.csv \
  --observed-mask artifacts/M_observed.npz \
  --config configs/zero_state/zinb.yaml \
  --gate-mode calibrated_threshold

# 5. Fuse theo truth-table
python -m missing_imputation fuse-initializations \
  --raw data/covariants.csv \
  --tsagris artifacts/dataset_00_tsagris.csv \
  --posterior artifacts/zero_state/zero_state_posterior.npz \
  --zero-mask artifacts/zero_state/M_target_zero.npz \
  --nonzero-mask artifacts/zero_state/M_target_nonzero.npz \
  --uncertain-mask artifacts/zero_state/M_target_uncertain.npz

# 6. Parity DeepMicroGen
python -m missing_imputation deepmicrogen-parity --config configs/gan/reproduction.yaml

# 7. Gated magnitude refinement
python -m missing_imputation refine \
  --input artifacts/dataset_0_fused.csv \
  --fixed-mask artifacts/M_fixed.npz \
  --gan-mask artifacts/M_gan.npz \
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
| M3 | Reduced/pooled ZINB + abstain masks + fusion `dataset_0_fused` | `v0.3.0-fusion` |
| M4 | Gated magnitude refinement | `v0.4.0-integration` |
| M5 | Reconstruction/distribution evaluation và handoff | `v1.0.0` |

## 11. Timeline đề xuất

Hai người làm song song, tổng thời gian dự kiến 5 tuần làm việc:

| Tuần | P1 | P2 | Gate cuối tuần |
|---|---|---|---|
| 1 | T00-T03 | T00, T09-T10 | Data/mask contract và core tests |
| 2 | T04-T06, T07 | T10-T12 | Có `dataset_00_tsagris`; DeepMicroGen port gần hoàn tất |
| 3 | T07-T08 | T11-T13 | Có OOF-calibrated routing/abstain masks và `dataset_0_fused` |
| 4 | T16 | T14-T17 | Chạy được gated `dataset_0_fused -> dataset_1` và ablations |
| 5 | T18-T19 | T17, T19 | Final reconstruction/distribution report và release |

Nếu chỉ có CPU, T17 có thể kéo dài thêm. Smoke/parity phải chạy được trên CPU; full tuning có thể chạy GPU nhưng phải cung cấp config và log đầy đủ.

## 12. Risk register và phương án xử lý

| Risk | Mức | Ảnh hưởng | Biện pháp |
|---|---|---|---|
| Chỉ có 516 dòng complete cho neighbor pool | Cao | Neighbor yếu, adaptive pattern thiếu support | Repeated CV, fallback global, báo cáo coverage theo pattern |
| 95.58% dòng có missing | Cao | Self-training dễ củng cố sai số baseline | Không coi initialization là truth; giữ raw masks bất biến |
| ZINB độc lập theo variant vi phạm closure | Cao | Positive draws có tổng count sai | Bỏ positive draws khỏi critical path; fusion closure dùng Tsagris/GAN magnitude |
| Binary gate ép model yếu phải quyết định | Cao | Sparse variant bị khóa zero sai | Thêm uncertain/abstain state; uncertain luôn chuyển GAN |
| Structural và sampling zero không identifiable hoàn toàn | Cao | Dễ overclaim biological absence | Chỉ gọi posterior latent; không báo accuracy thiếu nhãn ngoại sinh |
| 149 location FE gây separation/rank deficiency | Cao | Cả 17 model rơi xuống Poisson | Reduced design 8-12 predictors; location history qua observed lag/lead features |
| ZINB convergence ở variant quá thưa | Cao | Fit lỗi hoặc all-zero gate | Support routing -> pooled ZINB -> abstain; không ZIP/Poisson hard gate |
| ZINB gate nhầm non-zero thành zero | Cao | GAN mất cơ hội reconstruct magnitude | Tune threshold theo non-zero recall >= target; uncertain band; sensitivity analysis |
| Pooled model che lấp đặc trưng variant hiếm | Trung bình | Posterior bị kéo về nhóm | Variant intercepts, per-variant OOF metrics và abstain nếu calibration kém |
| Calibration leakage do không refit fold | Cao | NLL/Brier/F1 lạc quan | Bắt buộc refit mỗi fold; lưu train/test indices và OOF predictions |
| Fit/predict design schema lệch | Cao | Runtime shape error hoặc prediction sai | Shared DesignEncoder + serialized schema; dimension/name assertions |
| Padding bị coi là real target | Cao | Impute thêm 87k cell ngoài raw NaN | `M_target = raw_nan AND M_row AND NOT M_padding`; exact-count invariant |
| Threshold tune nhưng inference hard-code | Cao | Gate không dùng calibration đã báo cáo | Serialize/apply threshold theo variant/tier; integration assertion |
| Zero-state stage fail nhưng fusion vẫn chạy | Trung bình | Lỗi dây chuyền thiếu artifact khó chẩn đoán | Stage status/manifest gate; fail fast trước fusion |
| Natural spline basis chứa constant cùng explicit intercept | Cao | Rank deficiency ở mọi fold, tham số cực trị | Centered spline constraint + một Patsy intercept; rank/condition tests trước fit |
| Pooled K one-hot columns cộng intercept | Cao | Pooled design rank-deficient | K-1 effect/treatment coding, frozen reference level |
| Pooled OOF dùng full-data model | Cao | Leakage và metric không đại diện generalization | Rebuild stacked data và refit pooled model trong từng fold |
| Rank check trên single-variant dummy matrix | Cao | False rank failure dù pooled fit matrix khác | Check actual stacked training matrix returned by encoder |
| Baseline prevalence tính trên toàn observed | Trung bình | So sánh NLL/Brier không cùng information set | Fit baseline theo train fold, score cùng test cells |
| Failed OOF folds bị bỏ khỏi aggregate | Cao | Metric lạc quan cho variant khó fit | Fold coverage gate và per-recipe minimum; failed fold được lưu rõ |
| Diagnostic nhầm all-variant-zero với `total_sequence=0` | Thấp | Root-cause report sai hướng | Raw-data invariant và tên metric tách biệt; hiện raw có 0 dòng total_sequence=0 |
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
- [ ] `M_observed`/`M_target` tạo từ raw, loại padding, target count khớp raw NaN và observed zero/positive immutable.
- [ ] Posterior/masks dùng per-variant target coordinates; không broadcast incomplete row sang cả 17 variants.
- [ ] Có support report và routing reduced per-variant/pooled/abstain cho đủ 17 variants.
- [ ] ZINB fit chỉ trên observed raw data, không dùng 149 location FE hoặc Tsagris predictors.
- [ ] Centered natural spline/intercept convention và pooled K-1 coding pass rank/condition tests trên mọi successful fold.
- [ ] Calibration refit theo fold, có OOF predictions và quality gate so với prevalence baseline.
- [ ] Pooled model được rebuild/refit trên stacked train data trong từng fold; rank check dùng đúng fitted matrix.
- [ ] Prevalence baseline học theo train fold, score trên cùng held-out cells; fold coverage được quality-gate.
- [ ] Threshold theo variant/tier đạt target non-zero recall và được dùng thật tại inference.
- [ ] Model không đạt quality gate sinh `M_target_uncertain`, không sinh hard zero.
- [ ] ZIP/Poisson/Hurdle không được dùng để tạo `M_target_zero`.
- [ ] Fit/predict/sampling dùng cùng serialized DesignEncoder schema; không có dimension mismatch.
- [ ] `n_iterations` không khả dụng được ghi `null`, không ghi sai `0`; metric fractions ghi rõ threshold tương ứng.
- [ ] Fusion four-state truth-table được test cho mọi cell và `dataset_0_fused` thỏa closure.
- [ ] Có `cell_provenance` truy được nguồn của từng cell.
- [ ] DeepMicroGen source commit được pin và có parity/deviation report.
- [ ] Chạy được ít nhất một vòng gated `dataset_0_fused -> dataset_1` bằng CLI.
- [ ] GAN thay `M_target_nonzero OR M_target_uncertain`; không sửa observed hoặc confident-zero.
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
5. Target non-zero recall tối thiểu của zero gate; default đề xuất 0.95, cần chốt theo validation coverage.
6. Ngưỡng support/events-per-parameter chính thức để route reduced hay pooled ZINB; defaults hiện là 200/500 positives.
7. Pooled sparse groups dùng một nhóm chung hay chia theo temporal era/variant family nếu có mapping chuyên môn.
8. Tolerance cụ thể để chấp nhận GAN theo validation MSE, JSD và Wasserstein.

## 15. Nguồn kỹ thuật được dùng để lập plan

- `missing_impute/research/Hron.pdf`: Hron, Templ và Filzmoser, imputation trong Aitchison/ILR space bằng kNN, iterative LS và iterative LTS.
- `missing_impute/research/Sasha Micoda.pdf`: Saha et al., MICoDa V.1/V.2 trong ALR space, điều chỉnh theo covariate.
- `missing_impute/research/tsagris.pdf`: Tsagris, Stewart và Alenazi, ba biến thể JSD-kNN được chọn cho `dataset_00_tsagris`.
- `missing_impute/diffusion based/DeepMicroGen.pdf`: kiến trúc CNN, bidirectional RNN GAN, time-decay, losses và giới hạn irregular sampling.
- `https://github.com/joungmin-choi/DeepMicroGen.git`: source chính thức, pin tại commit `da2093d3c054dddb29415da8838c351ad7349f8d`.
- `diagnostic_report.md`: bằng chứng thực nghiệm của run `zinb_1788164041`; dùng để định nghĩa D0-D6 debug gates. Các claim trong report phải được kiểm tra lại với raw invariants trước khi đưa thành kết luận.
