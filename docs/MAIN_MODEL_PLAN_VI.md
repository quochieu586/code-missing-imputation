# Plan sườn main model: một luồng chạy, ít thành phần

Ngày: 2026-11-12. **Đề xuất thiết kế, chưa triển khai main model.**

> Bản phác thảo lịch sử. Plan hiện hành là [MAIN_MODEL_IMPLEMENTATION_PLAN_VI.md](MAIN_MODEL_IMPLEMENTATION_PLAN_VI.md), đã cập nhật Other mặc định của ba baseline và cắt bớt CLI/checkpoint không cần thiết. Những mặc định MICoDa/gate cũ bên dưới không thay thế contract Other đã duyệt.

## 1. Chốt hướng đề xuất

**Giữ `longitudinal_impute.run` làm cửa vào duy nhất. Thêm main model vào bên cạnh ba baseline, không dựng pipeline mới.**

Main model gồm hai file có trách nhiệm tách biệt:

- `main_model.py`: khởi tạo bằng baseline, chuẩn bị chuỗi, train, cập nhật và kiểm tra dừng.
- `network.py`: mạng PyTorch và `forward`, không đọc CSV, không điều phối thí nghiệm.

Giữ `data.py` là nơi sở hữu quy tắc dữ liệu/ngân sách; `evaluate.py` sở hữu metric; `run.py` sở hữu CLI, vòng native/masked và output. Không thêm framework, plugin, base trainer hay hệ thống adapter chung.

Đổi dataset nghĩa là đổi **CSV + schema**, không viết model mới. Model học lại cho dataset mới; không có cam kết dùng lại weights khi tên/số lượng target thay đổi.

## 2. Căn cứ đã đọc

| Nguồn | Điều cần giữ trong thiết kế |
|---|---|
| [Proposal](2026_10_09-Revised%20proposal.pdf), trang 2, 4 | Composition, longitudinal, nhiều zero; phải bảo toàn observed ở dòng thiếu một phần. |
| Proposal, trang 7 | Baseline tạo `X0`; LR transform; fit `f`; dự đoán composition; so với vòng trước và lặp. |
| Proposal, hình trang 8 | Temporal transformer, feature transformer, side information, gated residual/skip và đầu ra softmax trực tiếp. |
| [`run.py:28–47`](../src/longitudinal_impute/run.py) | Dictionary model và dispatch tường minh đã đủ; interface ba baseline hiện không đồng nhất. |
| [`run.py:176–248`](../src/longitudinal_impute/run.py) | Native riêng, masked riêng; mỗi lượt instance mới, mask chung giữa các model. |
| [`data.py:160–215`](../src/longitudinal_impute/data.py) | Đã có input visible-only, Other, chia residual, giữ observed và kiểm tra output. |
| [Contract baseline](THREE_BASELINE_CONTRACT_VI.md), D03–D08 | Không tự đổi MICoDa A17/R1 thành A18/R2; khác budget chưa cho phép kết luận so sánh thuật toán được kiểm soát. |
| [`PHAN_CONG.md`](../PHAN_CONG.md) | Kế thừa code, không thêm lớp trung gian; retrospective imputation trên bảng hiện có. |
| [Archive](../archive/README.md) | KNN+CSDI cũ là lịch sử, không phải route được hỗ trợ hiện tại. |

**Phân biệt với diễn giải quá mức:** proposal chưa đặc tả loss, cách tạo training labels, zero replacement, warm-start hoặc tiêu chí dừng cụ thể. Hình trang 8 có nét kiến trúc CSDI, nhưng không đặc tả noise schedule, epsilon objective hay reverse diffusion. Vì vậy MVP dưới đây là **iterative direct-composition model**, không tự gọi là diffusion hoặc multiple imputation có phân phối bất định.

Kiểm tra mới trong lượt lập plan:

- `115 passed in 8.65s`, với RuntimeWarning-as-error. Lượt chạy đầu lỗi quyền thư mục tạm; chạy lại bằng thư mục mới trong `C:\Users\admin\AppData\Local\Temp\opencode` và tắt pytest cache thì đạt.
- COVID: 11.671 dòng, 17 target, 150 entity; mỗi entity 13–109 mốc; 86.050 NaN, 516 dòng complete.
- Python hiện tại 3.14.0; chưa cài PyTorch. `pyproject.toml` hiện chỉ yêu cầu NumPy cho runtime.
- Chưa có dataset microbiome thật trong `data/`; microbiome hiện được kiểm thử bằng fixture.

Các kiểm tra này xác nhận nền baseline, **không** xác nhận main model hoặc tốc độ/chất lượng của thiết kế chưa triển khai.

## 3. Luồng duy nhất sau khi triển khai

```text
CLI hiện tại
  → load CSV/schema, kiểm tra cấu hình
  → chọn native và/hoặc masked; tạo mask H một lần mỗi seed
  → gọi lần lượt các model được chọn
      ├─ Hron / Tsagris / MICoDa: giữ route hiện tại
      └─ main:
          visible input → Tsagris khởi tạo X0
            → [CLR → fit mạng → dự đoán composition
                → giữ observed + chia residual → kiểm tra dừng] × tối đa n vòng
  → check_completed
  → lưu prediction
  → evaluate.py
  → summary/log/output như hiện tại
```

Main không gọi ngược `run.main()` hay `run_model()` để khởi tạo. Nó gọi trực tiếp `TsagrisKNN` cùng các hàm dữ liệu đã có; tránh vòng phụ thuộc giữa runner và model.

### Hai mức “chạy cùng nhau”

1. **Tích hợp vận hành:** baseline và main dùng chung CLI, data, mask, validator, metric và báo cáo. Có thể hoàn thành ngay trong sườn.
2. **So sánh khoa học được kiểm soát:** phải chốt budget/Other của comparator. Cùng một lệnh chạy không tự làm MICoDa R1 và main R2 trở thành cùng phương pháp phân bổ.

Không chặn dựng sườn vì gate thứ hai; cũng không tuyên bố thắng baseline khi gate đó chưa đạt.

## 4. Data contract nhỏ, nhưng phải rõ

### Giữ nguyên

- Counts đầu vào: `N_rows × D`, NaN là thiếu; zero là observed hợp lệ.
- `total` là thông tin ngân sách được phép biết, không phải giá trị suy lại từ truth bị che.
- Output CSV vẫn chỉ D target; observed giữ nguyên numeric và text như writer hiện tại.
- Mỗi lượt masked che H **trước** initializer, transform và training. Không tái sử dụng initializer/weights từ lượt native sang lượt masked.

### Phần longitudinal cần bổ sung

`Dataset` chỉ cần thêm trường thời gian gốc tùy chọn; loader lưu giá trị, chưa ép baseline phải hiểu kiểu ngày. Main mới parse thành số từ thời gian ISO hoặc visit dạng số.

- Main yêu cầu `entity` và `time`; thiếu time thì báo lỗi rõ, không ngầm lấy thứ tự CSV làm thời gian.
- Group theo entity, sort theo thời gian thật; lưu row indices để trả output về đúng thứ tự CSV.
- Dùng ngày lịch chung giữa các entity khi có ngày, visit số khi dữ liệu dùng visit. Không biến mọi chuỗi thành `0,1,2,...` và mất khoảng cách thời gian thực.
- Batch theo entity; pad độ dài và có `valid_steps` cho attention/loss. Không trộn hai entity vào một chuỗi, không sinh thêm dòng thời gian không có trong CSV.
- `D` lấy từ schema; không hard-code 17/18. Checkpoint kiểm tra cả tên và thứ tự target.

### Other: giữ tương thích, mở rộng đúng một thông tin schema nếu cần

Mặc định dùng route hiện tại: D target + Other; Other chưa biết ở dòng partial. Main khởi tạo và cập nhật theo `restore_counts`.

Nếu microbiome có **D target là toàn bộ composition**, cần khai báo điều này, không để model tùy ý dành mass cho Other. Đề xuất bổ sung duy nhất:

```toml
[dataset]
# entity, time, total, targets như hiện tại
has_other = false  # chỉ dùng khi targets thực sự exhaustive
```

- Không khai báo: giữ hành vi hiện tại (`has_other = true`), không đổi kết quả cũ.
- `false`: Other là structural zero; chia residual chỉ cho target thiếu; dòng complete phải khớp total trong tolerance đã thống nhất.
- Quy tắc này nằm tại `data.py`, không nhân bản theo dataset/model. Hron/Tsagris/main dùng cùng quy tắc; MICoDa vẫn giữ interface và adaptation được khai báo riêng.
- Tổ hợp trái nhau như dataset `has_other=false` nhưng MICoDa `other=true` phải bị từ chối ở preflight, không âm thầm sửa cấu hình.
- Đây là **đề xuất mở rộng contract cần duyệt**, không tự sửa mặc định baseline. Không tự suy `has_other` từ tên dataset hoặc từ một vài dòng có tổng khớp.

Nếu chưa duyệt mở rộng này, scaffold đầu tiên chỉ cam kết domain D+Other hiện hành; chưa gọi đó là hỗ trợ đầy đủ exhaustive microbiome.

**Giới hạn không được giấu:** loader hiện chỉ suy total từ tổng target khi CSV không có NaN. Dataset có NaN thật mà không biết total cần chuẩn bị total từ nguồn hợp lệ hoặc xác định lại bài toán; không thể giải quyết bằng đổi vài hyperparameter. Với total suy từ bảng complete rồi giữ lại khi masking, phải ghi rõ đây là bài toán **known-total reconstruction**.

## 5. Main model MVP

### 5.1 Một initializer

Chọn **Tsagris hiện tại** làm initializer đầu tiên: xử lý zero trực tiếp, đã có fit/predict và route residual dùng lại được. Dùng cấu hình mặc định hiện hành, chỉ mở `init_k` nếu cần.

Không đồng thời phát triển ba initializer và cơ chế tự chọn. Không đủ donor thì main báo `failed` cho lượt đó; không fallback ngầm sang mean hoặc một baseline khác. Các model khác vẫn tiếp tục.

### 5.2 Một transformation

Chọn CLR cho MVP: giữ K=D+1 chiều nội bộ, không cần chọn reference hoặc thêm cơ sở ILR.

```text
counts hoàn chỉnh → chia total → thay zero chỉ trên bản dùng làm feature
                 → log → trừ mean log theo hàng → Z
```

Đề xuất delta bằng một nửa positive proportion nhỏ nhất trong phần visible được phép biết, cố định trong từng lượt và lưu vào state. Với training view, tính chỉ từ V\Q; với inference view, chỉ từ V. Không lấy min từ truth H hoặc thay đổi raw observed zero. Không cần inverse CLR: mạng dự đoán composition trực tiếp.

### 5.3 Một mạng nhỏ, bám hình trang 8

```text
Z + observed mask
  → pointwise projection sang C channels
  → 2 residual blocks giống nhau:
      temporal attention → feature attention
      + side information (time embedding + feature embedding)
      → gated activation → residual + skip
  → cộng skip → pointwise head → softmax theo component
```

- Shape công khai: input `[B,L,K]`, hidden `[B,L,K,C]`, output `[B,L,K]`.
- Điểm xuất phát: `channels=32`, `heads=4`, hai block, dropout 0. Các giá trị này là cấu hình pilot, không phải thông số đã tối ưu.
- Temporal attention chạy riêng từng entity×feature; feature attention chạy trong từng timepoint. Attention được dùng hai chiều vì đây là retrospective imputation, không phải forecasting.
- Observed mask phân biệt “zero đã quan sát” với “giá trị đang điền”; padding mask là mask khác, không được gộp hai khái niệm.
- Đầu ra là composition đầy đủ; khi `has_other=false`, Other bị khóa zero.
- Không thêm phylogeny/phylum CNN, metadata fusion, country embedding, diffusion step hay noise scheduler trong MVP.

Hai block không có nghĩa hai hệ thống model lồng nhau: cùng một block nhỏ được lặp bằng `ModuleList`, mọi forward đều xem được trong `network.py`.

### 5.4 Training tối thiểu có đáp án thật, không rò label

Ký hiệu: O là observed gốc; H là evaluation mask; V=O\H; Q là self-mask để train, Q⊂V. Native thì H rỗng.

**Không train bằng cách coi X0 ở NaN thật là ground truth. Không đưa nguyên Xt chứa Q vào CLR rồi chỉ mask loss**: thông tin Q có thể đã đi qua initializer và mean log.

Đề xuất một training view cố định mỗi lượt để dễ debug:

1. Tạo Q dạng random-cell từ V, seed riêng của model; giữ anchor và kiểm tra donor support. Đây là train mask, không phải validation/test mask.
2. Khởi tạo **hai view tách biệt**: inference view từ V; training view từ V\Q. Q phải bị xóa trước initializer của training view, kể cả khi dựng Other.
3. Ở mỗi outer iteration, fit mạng trên CLR của training view; chỉ Q có supervised loss. Labels là raw observed proportions trên Q; NaN tự nhiên và H không có label.
4. Loss duy nhất: MSE proportion trên Q **sau phép phân bổ residual khả vi**, với visible anchors là V\Q. Vì Q không còn là anchor nên không bị overwrite thành truth trước loss.
5. Cập nhật training view bằng prediction, vẫn khóa V\Q; cập nhật inference view riêng, khóa V. Không chép inference view sang training view vì nó đã nhìn thấy Q.

Phép phân bổ dùng trong loss chỉ là tensor implementation ngắn của cùng công thức `restore_counts`, kiểm thử đối chiếu với NumPy; không dựng thêm một budget engine. Q được giữ cố định cho MVP để không phải tái khởi tạo kNN mỗi minibatch. Training loss chỉ dùng debug, không báo nó như độ chính xác ngoài mẫu.

Khi xây nghiên cứu tiếp theo mới cân nhắc nhiều self-mask/whole-feature training. Đây không phải điều kiện để có sườn chạy đúng đầu tiên.

### 5.5 Update và stopping

Với dòng i, giữ toàn bộ visible count. Phần dư là:

```text
r_i = total_i − sum(visible counts_i)
```

Phân bổ r_i theo trọng số mạng trên **missing targets + Other**, hoặc chỉ missing targets khi không có Other. Không softmax toàn hàng rồi overwrite observed mà bỏ qua việc sửa budget.

- Khởi tạo weights một lần mỗi lượt; warm-start qua các outer iteration, fit thêm một số epoch cố định mỗi vòng. Optimizer có thể khởi tạo lại mỗi vòng để resume ở ranh giới vòng đơn giản.
- Không backprop xuyên các outer iteration; detach state sau mỗi cập nhật.
- Sau mỗi vòng: gọi `check_completed`, ghi loss và `max_abs_change_proportion` trên target cần điền.
- Dừng khi thay đổi nhỏ hơn `tol`, hoặc chạm `outer_iters`. Đạt giới hạn vòng với output hợp lệ: `completed` nhưng `converged=false`; không giả vờ đã hội tụ.
- Không có ô cần điền: trả nguyên dữ liệu và ghi `no_missing`, không train vô ích.
- NaN/Inf hoặc vi phạm invariant: lỗi rõ, không clip tùy tiện để biến output hỏng thành thành công.

## 6. Sườn file và điểm nối

```text
src/longitudinal_impute/
  data.py            # giữ data/budget/check/writer; thêm metadata thời gian
  evaluate.py        # giữ metric hiện tại
  run.py             # thêm dispatch main và lựa chọn chạy
  baselines/         # không đổi interface/thuật toán mặc định
  main_model.py      # MỚI: MainModel + các hàm prepare/init/refine rõ tên
  network.py         # MỚI: CompositionNet + residual block

configs/models_all.toml       # MỚI: ba baseline giữ params cũ + [main]
tests/test_main_model.py      # MỚI: thuật toán/invariant
tests/test_network.py         # MỚI: shape/mask/gradient
```

`run_model` chỉ thêm một nhánh tường minh cho `MainModel`, trả cùng `(values, status, info)` mà runner đang dùng. Không bắt MICoDa đổi sang fit/predict chung, không tạo `BaseImputer` để làm đẹp hình thức.

Tách các hàm trong `main_model.py` để gọi trực tiếp khi debug; không biến mỗi hàm thành một module/class. Không chia `trainer/`, `engine/`, `bridge/`, `loop/`, `state_manager/`.

## 7. CLI: full và từng phần, cùng một entry point

**Các option/lệnh mới bên dưới là thiết kế tương lai, hiện chưa chạy được.** Lệnh baseline hiện hành phải tiếp tục chạy không sửa.

### Lựa chọn nhỏ cần thêm

| Option | Ý nghĩa |
|---|---|
| `--only hron tsagris micoda main` | Lọc trong model đã khai báo ở TOML; không phải registry mới. Bỏ qua option thì chạy toàn bộ model trong file như hiện nay. |
| `--pass native\|masked\|both` | Native riêng, masked riêng hoặc cả hai. Khi không truyền, giữ hành vi cũ theo `--mask`. |
| `--stage full\|init\|refine\|evaluate` | `full` mặc định; `init/refine` chỉ dành cho main; `evaluate` chấm prediction đã lưu, không fit lại. |
| `--from-run <dir>` | Đọc artifact của lượt trước; luôn xuất sang output mới, không sửa run cũ. |

Không tách `train` và `impute` thành hai pipeline trái proposal: một vòng refinement vốn gồm fit → predict → update. Đơn vị resume tối thiểu là **ranh giới outer iteration**, chưa hỗ trợ khôi phục giữa minibatch.

```powershell
# Full: baseline + main; native và masked dùng chung cơ chế hiện tại.
.venv\Scripts\python.exe -m longitudinal_impute.run `
  --data data/covariants.csv --schema configs/baselines_covariants.toml `
  --models configs/models_all.toml --output outputs/all_run `
  --mask whole_feature --mask-rate 0.05 --seeds 0 1 2 3 4

# Chỉ main, chỉ native, dừng sau khởi tạo; chưa cần PyTorch.
.venv\Scripts\python.exe -m longitudinal_impute.run `
  --data data/covariants.csv --schema configs/baselines_covariants.toml `
  --models configs/models_all.toml --only main --pass native `
  --stage init --output outputs/main_init

# Tiếp từ khởi tạo, không gọi lại initializer.
.venv\Scripts\python.exe -m longitudinal_impute.run `
  --data data/covariants.csv --schema configs/baselines_covariants.toml `
  --models configs/models_all.toml --only main --pass native `
  --stage refine --from-run outputs/main_init --output outputs/main_refined

# Chấm lại toàn bộ prediction của run đã lưu; không train/fit lại.
.venv\Scripts\python.exe -m longitudinal_impute.run `
  --stage evaluate --from-run outputs/all_run --output outputs/all_rescored
```

Ở `evaluate`, parser lấy data/schema/mask/model metadata từ manifest của run nguồn, kiểm tra file và hash; các đối số data/schema/models chỉ bắt buộc cho stage có tính toán model. Thiếu prediction thì báo không thể chấm, không âm thầm refit. Các tổ hợp vô nghĩa, như `--stage init --only hron`, phải bị từ chối trước khi tạo output.

Đổi sang microbiome vẫn là cùng lệnh full, đổi `--data` và `--schema`. Các hạn chế known-total/Other ở §4 vẫn áp dụng; không có nhánh `if dataset == covid/microbiome` trong model.

### Cấu hình main ban đầu

`models_all.toml` sao chép nguyên ba bảng baseline hiện tại, thêm:

```toml
[main]
init_k = 3
channels = 32
heads = 4
blocks = 2
batch_size = 8
epochs_per_iter = 10
outer_iters = 5
tol = 1e-4
learning_rate = 1e-3
train_mask_rate = 0.1
seed = 0
device = "cpu"
```

Đây là preset khởi đầu cần smoke-test, không phải cấu hình chứng minh hiệu quả. `--seeds` tiếp tục là seed evaluation mask; `[main].seed` là seed weights/training Q, ghi riêng, không tráo nghĩa.

## 8. Artifact đủ dùng, không có hệ thống quản lý thí nghiệm mới

Giữ `config.json`, completed CSV native, masks, `summary.csv/json`, `errors.log`. Bổ sung:

- `predictions/`: mảng counts theo model × pass × mask seed, gồm cả masked predictions của baseline. Cần phần này để evaluate riêng mà không fit lại; runner hiện chưa lưu nó.
- `main/<case>/initial.npz`: inference/training initialization, Q và metadata cần cho refine.
- `main/<case>/state.pt`: weights và state ở ranh giới outer iteration, gồm hai view, số vòng và RNG state cần thiết. Chỉ tạo khi refine; metadata thuần cấu hình để ở JSON.
- `main/<case>/history.csv`: iteration, training loss, thay đổi proportion, thời gian và convergence. Không dump tensor lớn sau từng batch mặc định.

`<case>` phân biệt native với từng mask-kind/seed. Không dùng chung checkpoint giữa các case. Resume kiểm tra data/schema hash, thứ tự target, model config, budget mode và **mask thực tế**, không chỉ seed label.

Stage `init` ghi trạng thái initialized, chưa được gắn nhãn completed của main; chất lượng initializer không được báo như chất lượng mạng đã train. Stage `evaluate` không tạo artifact weights mới. Run cũ thiếu prediction arrays không tự nhiên có khả năng rescore.

PyTorch là optional extra `main`, import khi refine; baseline-only và stage init vẫn chạy trong môi trường NumPy hiện tại. Không cài torch trong lượt lập plan. Khi triển khai phải kiểm tra wheel phù hợp Python 3.14/Windows trước; nếu không phù hợp mới đề xuất môi trường Python được hỗ trợ, không âm thầm thay venv đang dùng.

Test baseline chạy được không có extra; test main phải thực sự chạy trong môi trường có torch CPU. Skip test main vì thiếu dependency không được báo thành main đã được kiểm chứng.

## 9. Thứ tự triển khai và gate nhỏ

| Bước | Việc làm | Được coi là xong khi |
|---|---|---|
| 1. Data + init | Metadata thời gian, group/sort/unpack, initializer, chốt semantics Other | Output init hợp lệ; observed bất biến; trả đúng thứ tự CSV; cả hai fixture chạy được. |
| 2. Một vòng mạng | Optional torch, network nhỏ, training Q không leakage, `outer_iters=1` | Forward/backward finite; loss có gradient trên fixture có bậc tự do cần học; một vòng trả counts hợp lệ, không bắt buộc thắng initializer. |
| 3. Loop + runner | Lặp có giới hạn, dispatch main, lựa chọn model/pass, giữ schema output cũ | Một lệnh chạy cả bốn model; lỗi main không làm mất output baseline; test cũ vẫn đạt. |
| 4. Từng phần | Lưu prediction/state, init→refine, evaluate riêng, README ngắn | Full và chạy tách phần khớp trong tolerance; evaluate không gọi fit; đổi fixture chỉ đổi data/schema. |

Mỗi bước chạy test hồi quy; không đợi viết hết main mới kiểm tra baseline. Smoke-test dữ liệu thật phải có lệnh/rate/seed và giới hạn thời gian rõ khi được duyệt; lượt lập plan không chạy training/benchmark.

### Kiểm thử bắt buộc, tập trung đúng đường chạy

1. Baseline-only không cần torch; hành vi và tham số mặc định cũ không đổi.
2. Temporal sort/unpack, độ dài khác nhau, row order bị xáo, visit số và ngày ISO.
3. Padding không tác động prediction của phần thật; không có attention xuyên entity.
4. Thay truth H trong dữ liệu trong khi giữ V/totals/mask cố định không đổi init, preprocessing, weights hoặc prediction. Test thêm ở tầng Q trước training; mọi kênh CLR/Other đều được kiểm tra.
5. Q không lấy từ H hay NaN thật; loss không đọc nhãn tại hai tập đó.
6. Observed và observed zero giữ nguyên sau mọi vòng; output finite/nonnegative, đúng budget; residual=0 cho forced zero.
7. Tensor residual allocation khớp `restore_counts`; mode exhaustive kiểm tra sum target=total, mode Other kiểm tra sum target≤total.
8. Thiếu donor, thiếu time, cấu hình sai, numerical failure và không hội tụ đều có trạng thái/lý do rõ. Không fallback ngầm.
9. Fresh instance/state theo pass/seed; full khớp init→refine, resume boundary khớp chạy liên tục trong tolerance đã ghi.
10. Evaluate từ prediction khớp metric full; chấm lại không fit; từ chối state khác data/schema/target/mask.
11. Tất cả model nhận cùng H; masked truth chỉ dùng scoring. Không xuất masked CSV qua native writer rồi vô tình giữ lại đáp án.
12. Cùng core chạy fixture COVID và microbiome với D khác nhau; việc này chưa thay thế kiểm chứng trên microbiome thật.

## 10. Những gì không làm trong MVP

- Không hồi sinh toàn bộ code/archive KNN+CSDI, không có bridge CLR↔diffusion↔projection nhiều tầng.
- Không thêm nhiều transform/initializer/loss để thử cùng lúc, không sweep tìm winner.
- Không tự đổi MICoDa default, thêm metric framework, downstream analysis hay thực thi đề xuất evaluation chưa duyệt.
- Không hứa exact zero tại missing bằng softmax: observed zero và forced zero được bảo toàn; zero còn lại là giới hạn của point-estimator MVP. Chưa có hurdle/zero-inflated head.
- Không tuyên bố uncertainty/MI chỉ vì lặp nhiều lần; chưa có independent imputation draws.
- Không hứa tốc độ cho microbiome hàng nghìn taxa: feature attention có chi phí tăng theo K². Trước hết đo smoke trên D thực tế; chỉ đổi kiến trúc khi có bottleneck được chứng minh.

## 11. Các quyết định đề xuất để duyệt trước khi code

1. **Main v1 là direct-composition iterative network** ở trang 7–8; không tự thêm diffusion.
2. **Tsagris + CLR + một MSE proportion**, mạng hai block nhỏ; training view Q tách khỏi inference view để tránh leakage.
3. **Giữ contract baseline mặc định**; main dùng residual/Other hiện tại. Chỉ bổ sung `has_other=false` khi xác nhận dataset exhaustive; thay đổi đó và common-budget benchmark cần duyệt rõ.
4. **Một CLI, hai file mới**, chạy từng phần ở init/refine/evaluate; không tạo hệ sinh thái module mới.

**Định nghĩa hoàn thành sườn:** gọi cùng entry point để chạy baseline hoặc main, full hoặc từng phần; đổi dataset hợp lệ qua schema; invariant/test đạt và lỗi nhìn được. Việc main có tốt hơn baseline là câu hỏi thực nghiệm tiếp theo, không phải điều kiện để che lỗi hoặc làm phức tạp sườn.
