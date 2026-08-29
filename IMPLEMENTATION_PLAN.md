# Kế hoạch triển khai framework điền khuyết dữ liệu compositional longitudinal

## 1. Mục tiêu và phạm vi

Xây dựng một pipeline tái lập được với luồng xử lý:

```text
data/covariants.csv
        |
        v
Chạy 3 biến thể JSD-kNN của Tsagris
        |
        v
Đánh giá và chọn biến thể tốt nhất
        |
        v
dataset_0 (đầy đủ, không còn NaN)
        |
        v
CLR -> DeepMicroGen GAN -> inverse CLR -> projection/rounding
        |
        v
dataset_i -> dataset_(i+1), lặp đến khi đạt điều kiện dừng
```

Sản phẩm cuối phải bao gồm mã nguồn, cấu hình, kiểm thử, báo cáo đánh giá, manifest tái lập và các bộ dữ liệu bàn giao trên GitHub repository:

`https://github.com/quochieu586/code-missing-imputation.git`

### Ngoài phạm vi của phiên bản đầu

- Không phát triển một phân phối zero-inflated mới cho GAN.
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

### Invariant bắt buộc cho mọi `dataset_i`

1. Không còn `NaN` trong 17 cột variant.
2. Mọi count là số nguyên không âm.
3. Mọi giá trị quan sát ban đầu được giữ nguyên tuyệt đối.
4. `sum(17 variants) + other == total_sequence` trên từng dòng.
5. Khóa `(location, date)` là duy nhất và thứ tự dòng đầu ra ổn định.
6. Mặt nạ missing gốc được lưu riêng và không suy lại từ `dataset_i`.
7. Mỗi artifact có checksum, config, seed, Git commit và source dataset checksum.

## 3. Quyết định chọn bài báo cho giai đoạn `dataset_0`

### Ma trận phù hợp

| Bài báo | Compositional | Missing | Zero trực tiếp | Longitudinal | Kết luận |
|---|---|---|---|---|---|
| Hron et al. | Có, Aitchison/ILR | MCAR/MAR | Không; log-ratio yêu cầu số dương | Không | Không chọn vì dữ liệu có rất nhiều zero |
| Saha et al. - MICoDa | Có, ALR và covariate | Có | Dùng hằng số `W`; structural zero phải biết trước | Không | Không chọn vì zero treatment phụ thuộc pseudo-count và dữ liệu hiện không có bộ covariate đầy đủ như thiết kế gốc |
| Tsagris et al. | Có, JSD trên simplex | MAR | Có; `0 log 0 = 0`, không cần log-ratio | Không | Chọn cho `dataset_0` |

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
X          : [location, time, feature]
M_observed : [location, time, feature], 1 = quan sát gốc, 0 = missing gốc
M_row      : [location, time], 1 = dòng có trong CSV gốc
delta_f    : [location, time, feature]
delta_b    : [location, time, feature]
```

Toàn bộ location được reindex nội bộ lên lưới 14 ngày. Dòng bổ sung chỉ phục vụ mô hình; output mặc định chỉ trả lại các khóa có trong CSV gốc. Tùy chọn `--emit-grid` mới được phép xuất thêm các dòng thời gian chưa có trong dữ liệu gốc.

### 4.3 Tạo `dataset_0`

1. Validate schema, kiểu dữ liệu, uniqueness và closure lower bound.
2. Chuyển observed counts thành proportions bằng `total_sequence`.
3. Với 516 dòng đầy đủ, tính `other` chính xác và dùng làm neighbor pool.
4. Với dòng thiếu, coi tất cả component missing và `other` là phần cần phân bổ trong `remaining`.
5. Chạy ba biến thể Tsagris độc lập với cùng split và seed list.
6. Tune hyperparameter bằng repeated masked cross-validation mô phỏng missingness pattern thực tế.
7. Chọn thuật toán theo rule ở Mục 6.
8. Chuyển proportion về count bằng largest-remainder allocation chỉ trên các ô missing và `other`.
9. Giữ nguyên observed counts và kiểm tra invariant.
10. Xuất `dataset_0.csv`, `original_mask.npz`, metrics và manifest.

Fallback của adaptive algorithm:

- Chỉ tune riêng một missingness pattern khi pattern có đủ số hàng complete/masked theo `min_pattern_support` trong config.
- Nếu không đủ support, dùng `(alpha, k)` global của `JSD-alpha-kNN` và ghi rõ fallback trong manifest.
- Không âm thầm bỏ qua pattern hiếm.

### 4.4 Port DeepMicroGen

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

### 4.5 Adapter DeepMicroGen cho Covariants

Các khác biệt bắt buộc phải xử lý:

1. DeepMicroGen gốc chủ yếu mask cả sample/timepoint; Covariants missing theo từng cell. Mọi loss và blending phải hỗ trợ feature-level mask.
2. DeepMicroGen giả định grid thời gian tương đối đều; Covariants là panel bất cân bằng. Adapter phải reindex lưới 14 ngày, dùng padding mask và time gap thực.
3. DeepMicroGen gốc dùng taxonomy/phylum để nhóm OTU. Covariants không có taxonomy tương đương. Framework phải hỗ trợ:
   - `feature_groups.yaml` nếu nhóm variant đã được chuyên gia duyệt.
   - Một nhóm duy nhất với Spearman ordering làm fallback.
   - Ablation `identity/no-CNN` để xác định CNN có thực sự mang lại lợi ích.
4. CLR không nhận zero. Pseudocount chỉ được dùng bên trong bước GAN, phải được ghi trong config và không làm thay đổi zero quan sát ở output.
5. Postprocessing phải project về simplex, khóa observed cells và dùng largest-remainder để trả về count nguyên.

### 4.6 Vòng lặp `dataset_i -> dataset_(i+1)`

Mặt nạ gốc `M_observed` luôn được giữ cố định.

Ở vòng `i`:

1. Đọc `dataset_i` và manifest của vòng trước.
2. Chuyển count sang proportion, zero replacement có kiểm soát và CLR.
3. Dùng `dataset_i` làm warm-start/context cho các ô missing gốc.
4. Train hoặc tiếp tục train DeepMicroGen với artificial masks chỉ lấy từ ô quan sát gốc.
5. Reconstruction loss chỉ tính trên ground-truth quan sát hoặc ô quan sát được che giả lập; không coi giá trị imputed từ vòng trước là ground truth.
6. Sinh candidate cho các ô missing gốc.
7. Khóa observed cells, inverse CLR, threshold zero theo config, project lên simplex và integerize.
8. Ghi `dataset_(i+1)`, checkpoint, metrics và manifest.

Điều kiện dừng mặc định:

- `validation_jsd` không cải thiện quá `min_delta` trong `patience` vòng; hoặc
- thay đổi trung bình trên các ô missing giữa hai vòng nhỏ hơn `epsilon`; hoặc
- đạt `max_rounds`.

Không chọn checkpoint chỉ dựa trên training loss.

## 5. Phân công cho hai người

Vai trò tạm dùng trong plan:

- **Người 1 - Data & Compositional Baseline (P1)**.
- **Người 2 - GAN & Longitudinal Pipeline (P2)**.

Tên thật có thể thay vào GitHub issue assignee mà không làm đổi dependency hay output.

### Bảng task chính

| ID | Owner | Phụ thuộc | Công việc | Output bàn giao | Tiêu chí nghiệm thu | Ước lượng |
|---|---|---|---|---|---|---:|
| T00 | P1 + P2 | - | Chốt data contract, naming, seed policy, artifact policy và Definition of Done | `docs/adr/0001-data-contract.md`, config schema | Hai người approve ADR; CI đọc được config | 1 ngày |
| T01 | P1 | T00 | Audit `covariants.csv`, tạo schema validator, checksum và báo cáo missing/zero/time gap | `src/.../data/*`, `reports/data_audit.md`, tests | Phát hiện sai schema, duplicate key, count âm, observed sum > total | 1.5 ngày |
| T02 | P1 | T01 | Xây dựng conversion count/proportion, `other`, mask và largest-remainder closure | `closure.py`, `masks.py`, unit tests | Property tests giữ observed cells và tổng đúng `total_sequence` | 1.5 ngày |
| T03 | P1 | T02 | Implement `JSD-kNN` theo 6 bước của Tsagris | `jsd.py`, `jsd_knn.py`, tests fixture từ paper | Reproduce ví dụ paper; hỗ trợ zero; không có NaN output | 2 ngày |
| T04 | P1 | T03 | Implement Fréchet mean và `JSD-alpha-kNN` | `frechet.py`, `jsd_alpha_knn.py`, tests | `alpha=1` khớp arithmetic mean; CV chọn được `(alpha,k)` | 2 ngày |
| T05 | P1 | T04 | Implement adaptive algorithm và fallback cho sparse pattern | `adaptive_jsd_alpha_knn.py`, config, tests | Pattern đủ support tune riêng; pattern thiếu support ghi fallback | 2 ngày |
| T06 | P1 | T05 | Repeated masking CV, chạy ba baseline, chọn champion và sinh `dataset_0` | 3 output baseline, `dataset_0.csv`, `baseline_selection.md`, manifest | Đạt toàn bộ invariant; selection rule tái lập với fixed seeds | 2.5 ngày |
| T07 | P2 | T00 | Lập source manifest và đặc tả parity cho DeepMicroGen commit đã pin | `source_manifest.yaml`, `deepmicrogen_mapping.md` | Mọi module/loss gốc có mapping và test plan | 1 ngày |
| T08 | P2 | T07 | Port preprocessing, CNN, biRNN generator, time decay, discriminator và losses sang PyTorch | `src/.../deepmicrogen/*`, unit tests | Shape/loss test pass; không còn phụ thuộc TensorFlow 1.x | 4 ngày |
| T09 | P2 | T08 | Chạy reproduction/parity trên dữ liệu mẫu chính thức | parity tests, `deepmicrogen_parity.md`, checkpoint test | Pipeline gốc và bản port khớp contract; sai khác số học nằm trong tolerance đã ghi | 2 ngày |
| T10 | P2 | T01, T08 | Xây panel adapter: lưới 14 ngày, padding, actual delta, cell-level mask | `panel.py`, adapter tests | Hỗ trợ 150 chuỗi dài khác nhau; không leakage qua padding | 2.5 ngày |
| T11 | P2 | T02, T10 | Xây CLR/inverse CLR, zero policy, feature grouping và postprocessing count | preprocessing/postprocessing, configs, ablation switch | Observed zero giữ nguyên; closure và count integer pass | 2 ngày |
| T12 | P2 | T06, T09, T11 | Xây refinement orchestrator `dataset_i -> dataset_(i+1)` và checkpoint/resume | `pipeline/refine.py`, CLI, integration tests | Resume tạo cùng kết quả; manifest chain không bị đứt | 2.5 ngày |
| T13 | P1 | T06, T12 | Xây evaluation suite: random-cell, empirical-pattern, time-block và country holdout | `evaluation/*`, metrics tables | Split không leakage; cùng split cho mọi method | 2 ngày |
| T14 | P2 | T12, T13 | Chạy tuning/ablation DeepMicroGen và vòng lặp đến điều kiện dừng | checkpoints, per-round metrics, final candidate | Có learning curves, seed variance, stopping reason | 2.5 ngày |
| T15 | P1 | T13, T14 | Đánh giá cuối, kiểm tra statistical/practical significance và chọn final artifact | `final_evaluation.md`, bảng so sánh | GAN phải tốt hơn hoặc ghi trung thực trường hợp không tốt hơn `dataset_0` | 1.5 ngày |
| T16 | P1 + P2 | T15 | Hardening, README, CLI examples, CI, release notes và GitHub release | release `v1.0.0`, artifacts, checksums | Người khác clone repo và chạy smoke test bằng một command | 2 ngày |

### Cân bằng khối lượng

| Người | Phạm vi chính | Ước lượng riêng |
|---|---|---:|
| P1 | Data contract, 3 Tsagris algorithms, baseline selection, evaluation | 15 ngày công, chưa tính task chung |
| P2 | DeepMicroGen port, longitudinal adapter, iterative pipeline, training | 16.5 ngày công, chưa tính task chung |

T00 và T16 là task chung. Mỗi PR của P1 do P2 review và ngược lại; người viết code không tự merge PR của mình.

## 6. Giao thức đánh giá và chọn mô hình

### 6.1 Split bắt buộc

1. `random-cell`: sanity check, không dùng làm kết luận chính.
2. `empirical-pattern`: che giả lập theo phân phối missingness pattern của dữ liệu thật.
3. `time-block`: che một đoạn liên tiếp trong chuỗi của từng location.
4. `country-holdout`: giữ riêng một nhóm location để kiểm tra khả năng tổng quát hóa.

Tất cả method phải dùng chung file split IDs đã version hóa.

### 6.2 Metrics

Metrics chính:

- JSD trên proportions, vì chấp nhận zero.
- MAE và RMSE trên proportions ở các ô che giả lập.
- MAE trên count, có báo cáo theo các bin của `total_sequence`.

Metrics ràng buộc:

- Tỷ lệ dòng vi phạm closure.
- Tỷ lệ observed cell bị thay đổi.
- Tỷ lệ output âm, không nguyên hoặc NaN.

Metrics zero và temporal:

- Precision/recall/F1 cho zero trên các ô che giả lập.
- Sai khác zero prevalence theo từng variant.
- Temporal roughness và lag-1 change trên mỗi location.
- JSD giữa phân phối observed và imputed theo time window.

### 6.3 Rule chọn thuật toán tạo `dataset_0`

1. Loại ngay method vi phạm invariant.
2. Xếp hạng theo mean JSD của `empirical-pattern` và `time-block`.
3. Nếu chênh lệch JSD trong khoảng bất định của repeated runs, ưu tiên method đơn giản hơn và nhanh hơn.
4. Không chọn adaptive method chỉ vì tốt trên random-cell nếu kém trên time-block.
5. Ghi cả mean, standard deviation, seed list và runtime.

### 6.4 Rule chấp nhận GAN

GAN chỉ được coi là cải thiện nếu:

- Không vi phạm invariant.
- Mean JSD trên time-block không tệ hơn `dataset_0` quá tolerance đã chốt trong ADR.
- Có cải thiện ở ít nhất một trong hai: empirical-pattern JSD hoặc temporal metric.
- Kết quả ổn định qua tối thiểu 5 seed hoặc có giải thích rõ độ biến thiên.

Nếu GAN không vượt baseline, `dataset_0` vẫn là output khoa học hợp lệ; báo cáo không được ép chọn checkpoint GAN kém hơn.

## 7. Kiểm thử

### Unit tests

- JSD đối xứng, hữu hạn và bằng 0 khi hai composition giống nhau.
- JSD hoạt động với component bằng 0.
- Fréchet mean tại `alpha=1` bằng arithmetic mean sau closure.
- Ví dụ 5 thành phần trong paper Tsagris cho kết quả gần `(0.20, 0.27, 0.30, 0.10, 0.13)`.
- Largest-remainder giữ count quan sát và tổng chính xác.
- CLR/inverse CLR round-trip trong tolerance.
- Mask blending không sửa observed cell.
- Time decay và padding không tạo NaN.

### Property tests

- Với nhiều `total_sequence` và missing patterns ngẫu nhiên, output luôn thỏa closure.
- Thay scale count nhưng giữ proportions không làm đổi neighbor ranking ngoài sai số số học.
- Checkpoint resume và run liên tục cho cùng output khi dùng deterministic mode.

### Integration tests

- Chạy ba baseline trên fixture nhỏ từ đầu đến `dataset_0`.
- Chạy một vòng GAN nhỏ `dataset_0 -> dataset_1` trên CPU.
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
- `dataset_0.csv` và manifest.
- Evaluation suite và final evaluation report.

### Output của P2

- Source manifest của DeepMicroGen.
- Bản port PyTorch và parity report.
- Covariants panel adapter.
- CLI refinement, checkpoint/resume và per-round artifacts.
- Candidate `dataset_1 ... dataset_n` và training metrics.

### Output chung cuối cùng

```text
release/v1.0.0/
├── dataset_0.csv
├── dataset_final.csv
├── original_mask.npz
├── baseline_metrics.csv
├── gan_round_metrics.csv
├── final_evaluation.md
├── run_manifest.json
├── checksums.sha256
└── model_checkpoint.pt
```

Git repository lưu code, config, tests, report và manifest. Checkpoint lớn và các CSV trung gian được đưa vào GitHub Release assets; không commit toàn bộ checkpoint/history vào Git. Chỉ `dataset_0`, `dataset_final` và checksum cuối được đưa vào release chính thức.

## 9. CLI dự kiến

```bash
# 1. Kiểm tra dữ liệu
python -m missing_imputation audit-data --config configs/data.yaml

# 2. Chạy đủ ba Tsagris baselines
python -m missing_imputation run-baselines --config-dir configs/baseline

# 3. Chọn champion và sinh dataset_0
python -m missing_imputation select-baseline --metrics artifacts/baselines/metrics.csv

# 4. Parity DeepMicroGen
python -m missing_imputation deepmicrogen-parity --config configs/gan/reproduction.yaml

# 5. Refinement
python -m missing_imputation refine \
  --input artifacts/dataset_0.csv \
  --mask artifacts/original_mask.npz \
  --config configs/gan/covariants.yaml \
  --max-rounds 10

# 6. Đánh giá và đóng gói release
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
| M1 | Ba Tsagris baselines + `dataset_0` | `v0.1.0-baseline` |
| M2 | DeepMicroGen reproduction/parity | `v0.2.0-deepmicrogen` |
| M3 | Covariants adapter + iterative refinement | `v0.3.0-integration` |
| M4 | Final evaluation và handoff | `v1.0.0` |

## 11. Timeline đề xuất

Hai người làm song song, tổng thời gian dự kiến 4 tuần làm việc:

| Tuần | P1 | P2 | Gate cuối tuần |
|---|---|---|---|
| 1 | T00-T03 | T00, T07-T08 | Data contract và core algorithm tests |
| 2 | T04-T06 | T08-T09 | Có `dataset_0`; DeepMicroGen parity pass |
| 3 | T13 phần split/metrics | T10-T12 | Chạy được `dataset_0 -> dataset_1` |
| 4 | T13, T15, T16 | T14, T16 | Final evaluation và release `v1.0.0` |

Nếu chỉ có CPU, task T14 có thể kéo dài thêm. Smoke/parity phải chạy được trên CPU; full tuning có thể chạy GPU nhưng phải cung cấp config và log đầy đủ.

## 12. Risk register và phương án xử lý

| Risk | Mức | Ảnh hưởng | Biện pháp |
|---|---|---|---|
| Chỉ có 516 dòng complete cho neighbor pool | Cao | Neighbor yếu, adaptive pattern thiếu support | Repeated CV, fallback global, báo cáo coverage theo pattern |
| 95.58% dòng có missing | Cao | Self-training dễ củng cố sai số baseline | Loss không coi imputed value là truth; giữ original mask bất biến |
| Tsagris không có temporal model | Cao | `dataset_0` thiếu mượt theo thời gian | Time-block evaluation; DeepMicroGen refinement |
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
- [ ] Có báo cáo định lượng giải thích vì sao chọn thuật toán tạo `dataset_0`.
- [ ] `dataset_0` thỏa toàn bộ invariant.
- [ ] DeepMicroGen source commit được pin và có parity/deviation report.
- [ ] Chạy được ít nhất một vòng `dataset_0 -> dataset_1` bằng CLI.
- [ ] Vòng lặp có checkpoint, resume và stopping rule.
- [ ] Đánh giá có random-cell, empirical-pattern, time-block và country-holdout.
- [ ] Observed values và observed zeros không bị thay đổi.
- [ ] Có final report so sánh ba baseline, `dataset_0` và GAN rounds.
- [ ] CI xanh trên `main`.
- [ ] Release GitHub có artifact, checksum, manifest và hướng dẫn tái lập.
- [ ] Một người không viết code chính chạy lại smoke test từ fresh clone thành công.

## 14. Điểm cần xác nhận trước khi bắt đầu code

Các điểm sau là gate của T00, không cản trở việc mở issue và scaffold repo:

1. Tên thật/GitHub username tương ứng với P1 và P2.
2. Có mapping chuyên môn cho nhóm 17 variant hay dùng one-group + ablation.
3. GPU mục tiêu và giới hạn thời gian cho full tuning.
4. Có cần xuất thêm các country-date không tồn tại trong CSV gốc hay chỉ impute cell trong 11,671 dòng hiện có.
5. Tolerance cụ thể để chấp nhận GAN không tệ hơn baseline trên time-block JSD.

## 15. Nguồn kỹ thuật được dùng để lập plan

- `missing_impute/research/Hron.pdf`: Hron, Templ và Filzmoser, imputation trong Aitchison/ILR space bằng kNN, iterative LS và iterative LTS.
- `missing_impute/research/Sasha Micoda.pdf`: Saha et al., MICoDa V.1/V.2 trong ALR space, điều chỉnh theo covariate.
- `missing_impute/research/tsagris.pdf`: Tsagris, Stewart và Alenazi, ba biến thể JSD-kNN được chọn cho `dataset_0`.
- `missing_impute/diffusion based/DeepMicroGen.pdf`: kiến trúc CNN, bidirectional RNN GAN, time-decay, losses và giới hạn irregular sampling.
- `https://github.com/joungmin-choi/DeepMicroGen.git`: source chính thức, pin tại commit `da2093d3c054dddb29415da8838c351ad7349f8d`.
