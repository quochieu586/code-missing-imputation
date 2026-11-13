# Plan hiện thực main model — bản tinh giản để duyệt

**Trạng thái: hoàn thành bốn mốc MVP.** Mốc 4 đã qua suite, CLI fixture và review độc lập sau sửa lỗi preflight. Chưa benchmark dữ liệu thật hoặc kết luận main hơn baseline. Xem [tiến độ và bằng chứng](MAIN_MODEL_PROGRESS_VI.md). Đây là plan hiện hành, thay các bản phác thảo trước khi ba baseline thống nhất Other.

## 1. Quyết định giữ sườn nhỏ

| Phần | Chọn cho bản đầu | Không làm |
|---|---|---|
| Route | `longitudinal_impute.run` hiện tại | Pipeline/CLI mới |
| Runtime mới | `main_model.py` và `network.py` | Tách thêm trainer, engine, bridge, plugin, state manager |
| Initializer | Tsagris hiện tại, một cấu hình cố định ban đầu | Tự chọn hoặc fallback giữa nhiều initializer |
| Transform | CLR, zero replacement chỉ trong bản tạo feature | CLR/ILR/ALR chạy nhiều nhánh |
| Network | Một block nhỏ: temporal → feature attention → gate/residual → head softmax | Full encoder–decoder, diffusion, phylogeny CNN, metadata fusion |
| Training | Một self-mask Q cố định mỗi lượt, một MSE proportion | Nhiều loss, pseudo-label NaN thật, hyperparameter sweep |
| Budget | D+Other/R2 theo contract đã chốt | Thêm budget mode hoặc đổi baseline lần nữa |
| Dependency | PyTorch optional cho main | Lightning, Transformers/Hugging Face, orchestration framework |

Một block là điểm bắt đầu để kiểm tra đúng trước, không phải số tầng đã được tối ưu. Các quyết định initializer/CLR/loss là lựa chọn hiện thực đề xuất, không phải chi tiết mà proposal đã đặc tả đầy đủ. Main dự đoán composition trực tiếp; không gọi đây là diffusion hay multiple imputation có uncertainty.

## 2. Một đường đi của dữ liệu

```text
run.py: đọc data/schema → native hoặc masked → gọi lần lượt các model
  ├─ ba baseline: giữ nguyên route hiện tại
  └─ main_model.py:
       xóa evaluation truth khỏi input
       → Tsagris khởi tạo
       → [CLR → train → predict → restore_counts → kiểm tra dừng]
       → counts + info
run.py: kiểm output → evaluate.py → lưu prediction / summary / log
```

Main không gọi ngược runner để chạy initializer. Nó gọi trực tiếp Tsagris và các hàm dữ liệu đã có. `run_model` chỉ thêm một nhánh tường minh cho main, vẫn trả `(values, status, info)`; không đổi interface ba baseline.

**Một nguyên tắc đọc code:** mở `MainModel.impute(...)` là thấy thứ tự xử lý từ đầu đến cuối. Các bước là hàm có tên rõ trong cùng file, không phải mỗi bước một class/module.

## 3. Mỗi file một trách nhiệm

| File | Trách nhiệm và thay đổi dự kiến |
|---|---|
| `src/longitudinal_impute/main_model.py` — mới | Orchestrator ngắn; group/pack chuỗi, khởi tạo, CLR, train một vòng, predict và lặp cập nhật. Không tự xuất báo cáo. |
| `src/longitudinal_impute/network.py` — mới | `CompositionNet`, forward và một block attention. Không đọc CSV, không tạo mask thí nghiệm, không chứa loop train. |
| `src/longitudinal_impute/data.py` | Giữ owner input/budget/validator/writer. Chỉ thêm trường thời gian gốc tùy chọn vào Dataset và nạp từ schema. |
| `src/longitudinal_impute/run.py` | Thêm dispatch main; chọn lượt/stage; đọc/ghi artifact nhỏ và báo lỗi theo model/case. |
| `src/longitudinal_impute/evaluate.py` | Dùng lại metric và paired summary, không thêm metric mới cho task main. |
| `configs/`, `tests/` | Thêm config main và test tập trung; không sửa mặc định ba baseline. |

Các hàm chính dự kiến trong `main_model.py`: `initialize`, `clr_features`, `pack_sequences`, `train_one_round`, `predict_counts`, `refine`. Chỉ tách helper khi nó có một việc rõ và kiểm thử được; không dựng generic abstraction để tránh vài dòng gọi trực tiếp.

## 4. Data và network phải nhìn thấy được shape

| Biến | Shape / ý nghĩa |
|---|---|
| Raw counts | `[N,D]`; giữ float64 ở biên dữ liệu, NaN là thiếu, zero là observed |
| Composition nội bộ | `[N,K]`, `K=D+1`, gồm Other |
| Batch của network | `[B,L,K]`, mỗi phần tử batch là một entity, pad theo chiều thời gian |
| Hidden features | `[B,L,K,C]`; preset ban đầu `C=32`, 4 attention heads |
| Network output | `[B,L,K]`, composition proposals; chưa phải counts cuối cùng |

- Main bắt buộc có `entity` và `time`. Loader chỉ lưu time gốc; main parse ISO date hoặc numeric visit, sort, rồi trả output về row order ban đầu.
- Giữ khoảng cách thời gian thật, dùng cùng gốc ngày lịch giữa các entity; không coi mọi chuỗi đều có cadence bằng nhau.
- Hai mask khác nhau: `observed_mask` chỉ giá trị đã biết; `valid_steps` loại padding. Không trộn chúng thành một mask.
- Temporal attention trong một entity×component; feature attention trong một entity×timepoint. Không attention trực tiếp xuyên entity. Trọng số network được học chung qua các entity.
- Network dùng các primitive PyTorch trực tiếp; không chồng full Transformer encoder–decoder lên một custom block đã làm cùng nhiệm vụ.
- Không hard-code 17/18. Dataset mới đổi CSV/schema và fit mới, không tự tái sử dụng weights khi target khác.

## 5. Vòng lặp cần kiểm soát

### Hai bản dữ liệu, nhưng chỉ một network

Gọi H là evaluation mask, V là observed còn visible sau H. Q là một subset của V, dùng làm training labels; Q không bao gồm H hoặc NaN thật.

1. Tạo **inference view từ V** và **training view từ V\Q**. Xóa Q trước initializer và CLR của training view, kể cả khi dựng Other.
2. Chạy Tsagris một lần cho mỗi view. Không đủ donor thì báo lỗi; không đổi seed hoặc fallback để làm run thành công.
3. Đặt tên tường minh `inference_counts`, `training_counts`, `train_mask`, `observed_mask`; không truyền một dictionary lẫn truth và features khắp model.
4. Không sao chép inference view sang training view: inference view đã nhìn Q. Hai view chỉ dùng chung network weights.

Đây là phần cần thiết để tránh leakage, không phải thêm một tầng model. Q cố định trong từng lượt giúp không phải chạy lại kNN mỗi minibatch. Training loss chỉ dùng debug, không phải validation accuracy.

### Thứ tự đọc được trong một vòng

```text
network = tạo một lần với model seed
for iteration in range(outer_iters):
    train_one_round(network, training_counts, labels_Q)
    training_counts  = predict_counts(network, training_view)
    inference_counts = predict_counts(network, inference_view)
    check_completed(...)
    ghi loss, mức thay đổi, thời gian
    nếu mức thay đổi nhỏ hơn tol: dừng
```

Trong train/predict, luôn tạo CLR từ view hiện tại và giữ đúng mask anchors của view đó. Labels Q chỉ vào loss, không vào feature. Không backprop xuyên outer iterations; network weights warm-start, dữ liệu cập nhật được detach. Optimizer khởi tạo lại mỗi round là quy tắc cố định cho bản đầu.

**Loss duy nhất:** MSE proportion trên Q sau chia residual khả vi, anchors là V\Q. Dùng một helper tensor ngắn cho phép phân bổ này và đối chiếu với `data.restore_counts`; không nhân bản toàn bộ data layer. Zero replacement cho CLR lấy min-positive/2 từ visible hợp lệ của từng view, cố định trong lượt, không lấy từ prediction hoặc hidden truth.

**Dừng:** max absolute change proportion trên target cần điền không vượt `tol` (≤), hoặc hết `outer_iters`. Output hợp lệ nhưng chưa hội tụ vẫn ghi `converged=false`. Không có ô cần điền thì trả nguyên bảng, không train vô ích.

## 6. Cơ chế Other không thiết kế lại

Theo [contract đã thống nhất](BASELINE_OTHER_CONTRACT_VI.md):

- Other biết trực tiếp chỉ khi mọi target của dòng còn visible; dòng partial phải dự đoán Other cùng target thiếu.
- Raw observed kể cả zero luôn khóa. Residual chia theo proposal weights của missing targets + Other.
- Reuse `restore_counts` và `check_completed` cho counts cuối. Softmax không tự bảo toàn observed, nên không bỏ bước restore.
- Cùng `space` với ba baseline mặc định và cùng evaluation masks; dùng paired summary hiện có. Không trộn MICoDa legacy R1 vào common-budget comparison.
- Không thêm `has_other=false` schema trong MVP này. Exhaustive-target datasets cần contract riêng được duyệt; dataset thiếu thật mà không biết total vẫn ngoài domain hiện tại.

## 7. Chạy full / từng phần, nhưng không làm hệ thống resume

Giữ entry point cũ và cách chọn model bằng file TOML. **Không cần thêm `--only`** vì model không có trong TOML vốn không chạy.

| Nhu cầu | Giao diện dự kiến |
|---|---|
| Baseline-only | `--models configs/models_default.toml` — giữ như hiện tại |
| Main-only | `--models configs/main.toml` — file mới chứa `[main]` |
| Cả bốn | `--models configs/models_all.toml` — params baseline giữ nguyên |
| Native / masked / cả hai | Thêm `--pass native\|masked\|both`, không truyền thì giữ hành vi cũ |
| Khởi tạo riêng main | `--stage init`, ghi trạng thái initialized, không báo là main đã train |
| Main refinement | `--stage refine --from-run <init-run>`, không chạy lại initializer |
| Chấm prediction đã lưu | `--stage evaluate --from-run <completed-run>`, không fit lại |

`--stage full` là mặc định. `init/refine` chỉ nhận main-only; `evaluate` đọc manifest của run nguồn. Các tổ hợp sai bị từ chối trước output. Khi có compute, data/schema/models vẫn là tham số cũ; stage evaluate lấy chúng từ manifest và kiểm tra hash, không bắt nhập lặp lại.

**Cắt bỏ so với phác thảo trước:** không resume giữa minibatch, epoch hay outer iteration; chỉ nối hai stage `init → refine` và chấm lại prediction. Nếu refine bị ngắt thì chạy lại refine từ init. Điều này tránh phải quản lý optimizer/RNG checkpoints phức tạp ngay trong MVP.

Artifact chỉ gồm: config/masks/CSV/summary/log hiện có; predictions theo model/pass/seed để rescore; main init arrays (hai view, Q, nhãn Q được phép biết, delta) trong NPZ; `history.csv` của refine. Không dump tensor từng batch mặc định. Stage sau ghi output directory mới, kiểm tra data/schema, ordered targets, mask thực tế và model config trước khi dùng artifact cũ.

PyTorch import trễ khi cần refine; constructor/config validation và init không kéo torch vào baseline-only. Thêm extra `main`, kiểm tra khả năng cài trên môi trường Windows/Python hiện tại trước, không âm thầm thay venv. Tests bị skip vì thiếu torch không được gọi là main đã đạt.

## 8. Bốn mốc hiện thực — xong mốc nào kiểm mốc đó

| Mốc | Việc làm | Bằng chứng cần có trước khi đi tiếp |
|---|---|---|
| 1. Data + init | Time metadata, sort/pack/unpack, Tsagris, Q và hai view | Đúng row order/shape; observed bất biến; thay H không đổi input; init chạy không cần torch. |
| 2. Một vòng hoàn chỉnh | Network nhỏ, loss, train/predict/restore với `outer_iters=1` | Forward/backward finite; loss có gradient trên fixture có bậc tự do cần học; padding không ảnh hưởng phần thật; budget hợp lệ. |
| 3. Loop + route chung | Dừng có giới hạn, dispatch main, native/masked, model config và optional dependency | Lệnh chung chạy cả bốn; fresh state từng case; lỗi main không chặn baseline; paired report nhận đúng model space. |
| 4. Chạy tách + bàn giao | Init→refine, evaluate riêng, prediction/history, hướng dẫn lệnh ngắn | Full khớp chạy tách trong tolerance; evaluate không fit; COVID/microbiome fixture đổi D chỉ qua data/schema; baseline regression không hỏng. |

Mỗi mốc sửa ít file rồi chạy test liên quan và suite baseline. Không viết cả hệ thống xong mới debug. Mốc 2 phải chạy thành công **trước** khi làm nhiều outer iteration hoặc tiện ích stage.

## 9. Debug và kiểm thử tối thiểu

- Tại các biên init/CLR/network/restore: kiểm shape, finite, mask; lỗi ghi stage, iteration, entity/row và target nếu xác định được. Không log hoặc nuốt exception rồi tiếp tục bằng dữ liệu cũ.
- Một dòng mỗi iteration: train loss, max change proportion, seconds và converged. Chạy một vòng bằng config `outer_iters=1`, không cần debug engine riêng.
- Regression bắt buộc: H/Q không đi vào preprocessing sai view; residual=0; observed zero; thiếu donor/time; row permutation; padding; D thay đổi; full/tách stage; no-torch baseline; failure không mất khỏi summary.
- Giữ checks counts float64 của core; ghi rõ tolerance khi qua network float32. Không hứa bitwise equality giữa CPU/GPU.
- Verification baseline gần nhất là 121 tests và bốn CLI fixture cases của task thống nhất Other. Đây là bằng chứng nền đã ghi, không phải kiểm chứng main chưa được viết.

**Ngoài phạm vi:** benchmark để chọn winner, tối ưu nhiều hyperparameter, uncertainty intervals/MI draws, exact-zero head cho missing chưa bị budget buộc zero, và mở rộng full attention cho hàng nghìn taxa. Feature attention có chi phí tăng theo K²; không hứa hiệu năng trên microbiome thật khi chưa biết D và chưa đo.

**Điều kiện hoàn thành MVP:** một core dễ đọc, chạy chung được với baseline, chạy full/tách stage trên domain dữ liệu hiện hành, invariant/test đạt và lỗi truy vết được. Không lấy số module ít làm lý do bỏ kiểm tra leakage/budget; không lấy độ khó nghiên cứu làm lý do thêm framework.
