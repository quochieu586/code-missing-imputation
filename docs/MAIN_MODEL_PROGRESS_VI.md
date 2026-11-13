# Tiến độ hiện thực main model

## Trạng thái theo mốc

| Mốc | Trạng thái | Nội dung |
|---|---|---|
| 1. Data + init | **Hoàn thành** | Time metadata, sort/pack/unpack, Tsagris, training/inference views và CLR không cần torch. |
| 2. Một vòng neural | **Hoàn thành** | Network temporal/feature nhỏ, loss residual khả vi, train/predict một vòng trên CPU. |
| 3. Loop + runner | **Hoàn thành** | MainModel orchestrator, bounded loop, cấu hình main/all, native/masked qua runner hiện tại. |
| 4. Chạy từng phần | **Hoàn thành** | CLI init/refine/evaluate, prediction/history, kiểm artifact, CLI handoff và review độc lập sau sửa đã đạt. |

Hiện main chạy full hoặc `init → refine`, rồi `evaluate` prediction đã lưu qua **CLI chung**. Chỉ nối stage, không resume giữa minibatch/epoch/outer iteration. Mốc 4 không thay baseline defaults hoặc dữ liệu nguồn. PyTorch CPU là dependency optional `main`, đã được người dùng cho phép cài ở mốc 2.

## Dùng các hàm mốc 1

```python
from longitudinal_impute.data import load_csv
from longitudinal_impute.main_model import (
    initialize, clr_features, pack_sequences, unpack_sequences,
)

data = load_csv("data/covariants.csv", "configs/baselines_covariants.toml")
view = initialize(data)  # Tsagris; chỉ dùng phần visible nếu truyền hidden=H
features = clr_features(view)
packed, times, valid_steps, row_indices = pack_sequences(view.data, features)
restored_order = unpack_sequences(packed, row_indices)
```

Đây là ví dụ API, **không phải bằng chứng đã chạy full dữ liệu COVID**. Test mốc 1 dùng hai fixture và các trường hợp nhỏ xác định được kết quả. `prepare_views(data, hidden=H, seed=...)` trả inference view, training view, Q và labels proportion (NaN ngoài Q). Q không chọn từ H hoặc natural NaN; không redraw khi thiếu donor để né lỗi. Mask ngẫu nhiên có thể rỗng trên dữ liệu nhỏ; mốc neural phải xử lý thiếu labels rõ ràng, không coi loss rỗng là bằng chứng đã học.

## Bằng chứng TDD

- Trước code: **17 test fail**, do thiếu time metadata và API mới: [red.txt](../.harness/main-model/milestone-1/red.txt).
- Lượt sau code đầu tiên: 16 pass, một test gặp zero-mass donor thật. Fixture kiểm tra isolation được bổ sung positive support; tình huống zero-mass được giữ thành regression riêng, không thay fallback của baseline: [green.txt (lượt trung gian)](../.harness/main-model/milestone-1/green.txt).
- Sau sửa và bổ sung edge cases: **146 tests pass** (25 mới + 121 baseline), RuntimeWarning-as-error: [pytest.txt](../.harness/main-model/milestone-1/regression/pytest.txt).
- Bốn lượt CLI fixture, COVID/microbiome × random_cell/whole_feature, mỗi lượt có native và masked; kiểm tra installed entrypoints, cwd có khoảng trắng, giữ metadata/observed và không overwrite: [verification.json](../.harness/main-model/milestone-1/handoff/verification.json).

HarnessFlow TDD activation trước đó bị chặn `IDENTITY_MISMATCH`; mốc này dùng TDD native, không thực thi/simulate workflow bị chặn và không có independent reviewer/subagent. Main chưa có metric accuracy/performance mới.

## Mốc 2: một vòng neural

Lượt này nạp được `hf-tdd`; root `plan.md` đã qua checker trước RED. Không diễn giải các import/dependency errors thành RED. Từng callable scaffold báo NotImplementedError ở hành vi cần có, rồi được thay bằng code chạy thật:

| Vòng TDD | RED | GREEN |
|---|---|---|
| Network forward, simplex, gradient, padding/entity isolation | [5 fail](../.harness/main-model/milestone-2/network-red.txt) | [5 pass](../.harness/main-model/milestone-2/network-green.txt) |
| Tensor residual/loss và invalid training inputs | [7 fail](../.harness/main-model/milestone-2/allocation-red.txt) | [7 pass](../.harness/main-model/milestone-2/allocation-green.txt) |
| Train/predict, H-truth isolation, empty Q, hai fixture | [5 fail / 7 pass](../.harness/main-model/milestone-2/round-red.txt) | [12 pass](../.harness/main-model/milestone-2/round-green.txt) |

`network.py` chỉ chứa mạng; `main_model.py` thêm `allocate_residual`, `masked_composition_loss`, `train_one_round`, `predict_counts` và một helper pack tensor. Import torch nằm trong các hàm neural, không có import torch ở đầu main_model.py.

Cài và kiểm thử neural trên Windows CPU:

```powershell
uv sync --frozen --extra main --extra test
.venv\Scripts\python.exe -m pytest tests/test_network.py tests/test_neural_round.py -q -p no:cacheprovider --basetemp C:/Users/admin/AppData/Local/Temp/opencode/neural-tests-new
```

Chọn basetemp mới khi có các lượt chạy đang tồn tại. Thiếu extra thì tests neural có thể skip; skip **không** chứng minh mốc 2 đạt. Baseline-only vẫn có thể cài riêng extra `test`.

API cho một vòng (xem tests/test_neural_round.py để có dữ liệu và Q xác định):

```python
import torch
from longitudinal_impute.main_model import prepare_views, train_one_round, predict_counts
from longitudinal_impute.network import CompositionNet

# H và Q là mask bool; Q thuộc observed còn visible, không thuộc H.
inference, training, q, labels = prepare_views(data, hidden=h, train_mask=q)
torch.manual_seed(0)
network = CompositionNet(len(data.targets) + 1)
info = train_one_round(network, training, q, labels, epochs=1)
training.counts = predict_counts(network, training)
inference.counts = predict_counts(network, inference)
```

Prediction luôn qua NumPy `restore_counts` và validator. Tensor allocation chỉ phục vụ loss có gradient và được test đối chiếu công thức. Training không tự cập nhật temporary counts; caller cập nhật tường minh như trên. Loss ghi lại là training loss, không phải validation accuracy.

Môi trường được ghi trong [environment.json](../.harness/main-model/milestone-2/environment.json). Người dùng đã cho phép cài CPU; lần sync đầu timeout khi cài, chạy lại cùng frozen lock hoàn tất, không đổi Python. Chưa train/full benchmark dữ liệu COVID hoặc microbiome thật, chưa chứng minh main tốt hơn baseline.

### Verification cuối mốc 2

- **165 tests pass**, RuntimeWarning-as-error: [regression-final/pytest.txt](../.harness/main-model/milestone-2/regression-final/pytest.txt). Riêng mốc 1 + network/neural có **44 tests pass, không skip**: [targeted-final.txt](../.harness/main-model/milestone-2/targeted-final.txt).
- Bốn CLI baseline fixture cases pass: [handoff/verification.json](../.harness/main-model/milestone-2/handoff/verification.json). Baseline/init chạy được trong process mới chặn mọi torch import: [no-torch.txt](../.harness/main-model/milestone-2/no-torch.txt). `uv lock --check` cũng đạt.
- Independent verifier session `ses_e2f8705cbffeEu6mnFHgs7Zb5f` không tìm thấy correctness blocker; phát hiện 5 test network ban đầu chưa phát hiện khi loại bỏ attention. Đã bổ sung kiểm tra gradient truyền qua thời gian và qua feature (dùng log-ratio output để loại ảnh hưởng riêng của softmax).
- [Network thật: 7 pass](../.harness/main-model/milestone-2/attention-sensitivity-green.txt); [tắt attention trong subprocess: 2 test mới fail, 5 test cũ vẫn pass](../.harness/main-model/milestone-2/attention-disabled.txt). Đây là kiểm tra độ nhạy bổ sung sau GREEN, không giả nhận là RED trước production code của hai test mới. Không sửa baseline hay bỏ gradient/attention để làm test xanh.
- Chi tiết review và giới hạn: [review.md](../.harness/main-model/milestone-2/review.md).

## Mốc 3: loop + runner chung

- `MainModel.impute(data, hidden)` xác thực config/input, tạo hai view một lần, tạo network mới theo model seed mỗi lần gọi rồi warm-start qua các vòng. Không dùng state native cho masked hoặc state fit trước cho fit sau. Construction RNG được cô lập.
- `refine(...)` train → cập nhật training view → cập nhật inference view → kiểm output, đo max change proportion trên target thiếu và ghi history. Dừng khi change ≤ tol hoặc hết vòng; không dùng truth H để chọn dừng. Counts/state hai view không trộn nhau.
- Input không thiếu thì trả nguyên counts, không gọi torch/network, `trained=false`, iterations=0. Q rỗng/thiếu donor báo lỗi, không thử seed khác.
- `configs/main.toml` chạy main-only; `configs/models_all.toml` giữ nguyên mọi parameter baseline trong `models_default.toml` và thêm cùng main recipe. `run.py` thêm một dispatch rõ ràng, không đổi interface baseline hay thêm pipeline.
- History/console từng vòng có loss, change proportion, converged, seconds; info cuối có iterations và termination_reason. Output hợp lệ nhưng hết vòng chưa hội tụ không bị đổi thành accuracy success giả.

### TDD và verification

| Vòng | RED | GREEN |
|---|---|---|
| Loop update/stop/validation | [3 fail](../.harness/main-model/milestone-3/loop-red.txt) | [3 pass](../.harness/main-model/milestone-3/loop-green.txt) |
| Lifecycle/config/seed/no-missing | [18 fail](../.harness/main-model/milestone-3/lifecycle-red.txt) | [21 pass gồm loop](../.harness/main-model/milestone-3/lifecycle-green.txt) |
| CLI registry + integration | [10 fail](../.harness/main-model/milestone-3/cli-red.txt) | [10 pass](../.harness/main-model/milestone-3/cli-green.txt) |
| DLL-import isolation sau review | [2 fail](../.harness/main-model/milestone-3/dll-red.txt) | [2 pass](../.harness/main-model/milestone-3/dll-green.txt) |
| Converged trong iteration diagnostics | [2 fail / 1 pass](../.harness/main-model/milestone-3/diagnostics-red.txt) | [3 pass](../.harness/main-model/milestone-3/diagnostics-green.txt) |

Independent verifier `ses_e2f761ecdffe5UR0pg6j04iDSe` chứng minh lỗi import torch dạng Windows DLL `OSError` thoát runner và chặn mọi baseline. Đã sửa **chỉ trong optional import block**, giữ nguyên cause; không bắt rộng OSError filesystem. Verifier chạy lại 34 scoped tests pass, xác nhận DLL case direct/CLI đã sửa và lỗi filesystem không bị nuốt. Các thử DLL là fault injection, không phải môi trường này thật sự hỏng DLL.

Verification cuối: **199 tests pass**, RuntimeWarning-as-error, [regression-final/pytest.txt](../.harness/main-model/milestone-3/regression-final/pytest.txt). [Năm lệnh CLI thật với preset phát hành](../.harness/main-model/milestone-3/preset-cli-final/verification.json) pass trên fixture: all-model × hai dataset × hai mask, và main-only có cùng native prediction với combined. [Bốn CLI baseline handoff](../.harness/main-model/milestone-3/handoff-final/verification.json) vẫn pass. [Fresh process không torch](../.harness/main-model/milestone-3/no-torch.txt) chấp nhận all-model config, bỏ neural khi không missing, báo lỗi main và vẫn hoàn thành baseline khi cần neural.

Tại thời điểm chốt mốc 3, stage/rescore vẫn chưa triển khai; phần dưới ghi kết quả mốc 4. Chưa benchmark dữ liệu thật, chưa có kết luận main hơn baseline. Review mốc 3: [review.md](../.harness/main-model/milestone-3/review.md).

## Mốc 4: stage và bàn giao

Giữ nguyên một entry point `python -m longitudinal_impute.run` và hai runtime owner main_model/network. Không thêm dependency, checkpoint engine hay CLI riêng.

- `--stage full` mặc định; `--pass native|masked|both` chọn lượt, không truyền thì giữ hành vi cũ.
- `init` chỉ nhận main-only; `MainModel.prepare` tạo hai view/Q/labels/delta không import torch, ghi numeric NPZ và trạng thái `initialized`, không xuất completed CSV hoặc accuracy main đã train.
- `refine --from-run <init>` đọc snapshot đã validate, không fit lại initializer; khởi tạo network mới và chạy refinement như full. Đây không phải tiếp tục optimizer/RNG giữa các vòng.
- `evaluate --from-run <full-or-refine>` đọc prediction, tính lại metric và paired summary, không construct/fit model hay import torch. Thời gian compute gốc được giữ; failed/partial vẫn nằm trong coverage, partial không được chấm trên tập con tiện lợi.
- `artifacts.json` ghi checksum/path của config, masks, predictions, initial arrays, history và output. Chặn traversal/symlink ra ngoài source; đọc arrays với `allow_pickle=False`. Mask/Q/labels/delta/observed anchors được đối chiếu recipe và input; checksum chỉ phục vụ integrity, không xác thực người tạo.
- Stage sau lấy recipe từ source config, yêu cầu CSV/schema/models còn nguyên bytes (có thể truyền đường dẫn mới nếu byte-identical); override mask/seed/pass khác bị từ chối. Refine còn đối chiếu fingerprint preprocessing/main/network/initializer. Output phải mới và ngoài source run.

### TDD và verification

Root `plan.md` đã qua plan-check trước RED của mốc 4. Ba vòng chính:

| Vòng | RED | GREEN |
|---|---|---|
| Pass selector + prediction/history | [3 fail](../.harness/main-model/milestone-4/persistence-red.txt) | [16 pass gồm runner regression](../.harness/main-model/milestone-4/persistence-green.txt) |
| Init/refine + compatibility | [10 fail](../.harness/main-model/milestone-4/stages-red.txt) | [32 pass gồm lifecycle](../.harness/main-model/milestone-4/stages-green.txt) |
| No-fit evaluate + metric/failure semantics | [4 fail, 1 pass traversal](../.harness/main-model/milestone-4/evaluate-red.txt) | [18 stage tests pass](../.harness/main-model/milestone-4/evaluate-green.txt) |
| Metadata preflight sau review | [16 fail](../.harness/main-model/milestone-4/metadata-red.txt) | [42 stage tests pass](../.harness/main-model/milestone-4/metadata-green.txt) |
| Diagnostic/history preflight sau review tiếp | [16 fail](../.harness/main-model/milestone-4/diagnostics-red.txt) | [58 stage tests pass](../.harness/main-model/milestone-4/diagnostics-green.txt) |

Coverage bổ sung sau GREEN (không giả nhận là RED trước code): rehashed-invalid NPZ/Q/labels/delta/counts/mask, object arrays, source immutability và main rescore không neural dependency; tổng [26 stage tests pass](../.harness/main-model/milestone-4/validation.txt).

- **257 tests pass**, RuntimeWarning-as-error, không skip: [pytest.txt](../.harness/main-model/milestone-4/regression-final-v2/pytest.txt); [source/test hashes](../.harness/main-model/milestone-4/regression-final-v2/verification.json).
- [Bốn baseline CLI handoff](../.harness/main-model/milestone-4/handoff-final-v2/verification.json) pass; observed/metadata/budget/mask và chống overwrite không hỏng.
- [16 subprocess CLI thật](../.harness/main-model/milestone-4/staged-cli-final-v2/verification.json): COVID/microbiome × random_cell/whole_feature × full/init/refine/evaluate, cwd có khoảng trắng, init/evaluate chặn torch imports. Main full/staged khớp trong `1e-10`; rows/paired khi rescore khớp; file nguồn và run nguồn giữ nguyên.
- Independent verifier `ses_e2f5e0702ffeQVlplx90wlVqbx` tìm được metadata thiếu `seconds`, `pass=null`, rồi diagnostic lỗi thiếu `code` bị phát hiện sau output creation. Đã sửa preflight và thêm tổng 32 regression cases cho evaluate/refine. RED của các nhóm mở rộng còn bao gồm reject sai kiểu/lỗi không chuẩn hóa; không phải cả 32 đều chứng minh output đã được tạo. Verifier chạy lại ba probe với output mới: đều từ chối trước output và giữ nguyên bytes nguồn; chạy riêng **58 stage tests pass**, không còn blocker được chứng minh trong phạm vi review sửa lỗi. Xem [review.md](../.harness/main-model/milestone-4/review.md) và [checks.json](../.harness/main-model/milestone-4/checks.json).

Lệnh ngắn ở [README](../README.md#khởi-tạo--refinement--chấm-lại). Đổi D/domain chỉ qua CSV/schema đã được thử trên hai fixture, **chưa chứng minh accuracy tốt hơn baseline hay hiệu năng dữ liệu thật**. Feature attention vẫn có chi phí K². Không có kết luận benchmark từ thời gian fixture.
