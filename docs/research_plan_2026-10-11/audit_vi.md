# Audit setup và codebase — 11/10/2026

## Phạm vi và mức độ kết luận

Audit phục vụ thiết kế nghiên cứu mới: đọc source/config/tests, kiểm tra runtime
và chạy tập kiểm thử nhỏ không ghi cache. Không phải full benchmark, không chứng
minh toàn bộ pipeline cũ có hoặc không có leakage. Không chạy training, không
cài dependency, không xóa hay di chuyển file. Skeleton mới nằm trong `nextgen/`.

Ưu tiên P0: sai mục tiêu hoặc sai biên thông tin; P1: tái lập và khả năng mở
rộng; P2: hiệu năng/tổ chức. Đây là ưu tiên cho nhánh mới, không mặc định kết
luận các thí nghiệm legacy đã sai theo protocol riêng của chúng.

## Kết quả chính có chứng cứ

| ID | Mức | Quan sát trực tiếp | Ý nghĩa / việc cần làm |
|---|---|---|---|
| A01 | P0 | `src/preprocessing/prepare.py`, `preprocess`: mặc định loại đúng location/date/total_sequence rồi coi các cột còn lại là variants | Với original_long_format có metadata, inference này nhận sai target và có thể lỗi ép chuỗi thành float. Bắt buộc DatasetSchema khai báo targets/metadata riêng. |
| A02 | P0 | Cùng hàm: threshold relative abundance, thay zero bằng min positive/2; output X_pos | Đây là biểu diễn dương legacy, không phải bản sao counts gốc. Nhánh mới cần raw store bất biến và bản encode riêng; final output phải lấy observed counts từ raw store. |
| A03 | P0 | `src/stage_b/research_csdi.py`, condition_features dùng np.where(mask,x,0); epsilon_loss index eligible trước arithmetic | Có biện pháp tránh hidden raw trong conditioner và unknown labels trong loss. Không suy ra toàn pipeline an toàn: initializer, LR, metadata, cache và window vẫn cần kiểm thử đầu-cuối. |
| A04 | P0 | Mask legacy có cả True=VISIBLE (`masks`) và True=HIDDEN (`make_random_cell_mask`, evaluator runners) | Dễ đảo nghĩa khi tái sử dụng. Đặt tên nguyên nghĩa conditioning_mask/target_mask/padding_mask; assert quan hệ tập, không truyền biến tên chung mask giữa các lớp. |
| A05 | P1 | `scripts/run_e3_revised.py:39`: load_transform_module nạp `artifacts/e2_transforms/transforms.py` | Artifact đang là dependency executable. Clean checkout không bảo đảm có mã này vì artifacts bị ignore. Module mới phải nằm trong package; artifact chỉ lưu state và provenance. Giữ snapshot cũ cho tái lập lịch sử. |
| A06 | P1 | `scripts/run_e5_rework.py:42–46` import helper từ run_e3_revised, run_e4_revised và audit_e0_extended | Runner kiêm library tạo coupling theo lịch sử experiment. Chuyển dần helper vào library có hợp đồng; CLI chỉ điều phối. Không chuyển ở lượt skeleton này. |
| A07 | P1 | `scripts/rework_runtime.py`: chèn root và `.venv-e3-revised/Lib/site-packages` vào sys.path | Chạy được trong phiên audit, nhưng phụ thuộc layout máy và tính tương thích binary. Thay bằng package install và lock theo runtime sau khi kiểm chứng; không dùng workaround này làm hướng dẫn Kaggle. |
| A08 | P1 | requirements-stage-b.txt dùng khoảng version; `.venv` ghi Python MinGW 3.12.10, `.venv-e3-revised` ghi CPython 3.12.6 | Chưa có lock được dùng như nguồn chuẩn cho nhánh mới. File pyvenv.cfg không chứng minh executable hiện còn chạy được. Manifest phải ghi interpreter thực tế và resolved packages. |
| A09 | P1 | `run_e5_rework.py:50,56,115` cố định covariants.csv, transform variants, n_cols=17 | Phù hợp frozen run nhưng khó đổi dataset. Adapter/schema giữ D; representation sở hữu latent dimension; model lấy dimensions từ spec. |
| A10 | P1 | `src/evaluation/metrics.py` yêu cầu complete ground truth, CLR positive, đánh giá covariance cả ma trận | Không đủ cho labels thưa, nhiều zero, native missing. Viết evaluator có support mask và domain hợp lệ; chỉ so relationship khi đủ paired truth. Giữ metrics legacy riêng. |
| A11 | P1 | `src/core/transforms.py`: ilr_inverse trả representative chưa closure; ilr không kiểm finite giống clr | Cần phân biệt inverse relative / restore scale / constraints. Bổ sung kiểm thử NaN/Inf, basis dimensions và zero policy khi migrate. Không đổi semantics âm thầm. |
| A12 | P1 | `src/baselines/__init__.py`, `src/pipeline/__init__.py`, `src/stage_b/loop/__init__.py` rỗng trong snapshot | Có thư mục chưa có implementation; không coi ba paper baseline hoặc outer loop đã hoàn thành. |
| A13 | P1 | `.gitignore` ignore reports/*, artifacts/* | Báo cáo nghiên cứu trước tồn tại local nhưng không tự vào Git. Kế hoạch mới đặt docs/; cần chọn lọc version hóa báo cáo nguồn hoặc bản xuất, không add toàn bộ artifacts/data. |
| A14 | P2 | `research_csdi_rework.py` yêu cầu float64; CSDI có time/feature attention và residual blocks | Đã có nền tảng để parity-test. Không đổi dtype trước khi có kiểm tra ổn định; float32/AMP là thí nghiệm hiệu năng, không mặc định tương đương. |
| A15 | P1 | `src/stage_b/csdi/model.py` forward chưa nhận padding mask; legacy windows có thể không padding | Nhánh batch variable-length mới phải xử lý padding trong attention và loss. Chưa thể gọi đây là lỗi ở runner hiện tại nếu runner không padding. |

## Runtime thực tế được kiểm tra

- Interpreter audit: bundled CPython **3.12.14**, dùng `-B`; đã đọc packages
  bằng cách thêm `.venv-e3-revised/Lib/site-packages` theo workaround hiện có.
- Resolved versions qua importlib.metadata: numpy **2.5.3**, pandas **2.3.3**,
  scipy **1.18.1**, torch **2.14.0**, pytest **9.1.1**.
- Torch import thành công, `torch.version.cuda=None`, `cuda.is_available=False`.
  Đây là bằng chứng runtime audit là CPU, không phải kết luận máy không có GPU.
  `nvidia-smi` không nằm trong PATH; chưa đo RAM vật lý hoặc GPU bằng cách khác.
- `.venv-e3-revised/Scripts/python.exe` tồn tại; audit chưa chạy trực tiếp nó.
  `.venv/Scripts/python.exe` không được tìm thấy ở vị trí chuẩn đã kiểm tra.
- Không áp dụng `KMP_DUPLICATE_LIB_OK`; không cài/sửa môi trường hiện có.
- Không xác minh phiên bản mới nhất trên internet; các số trên là môi trường
  local đã đọc, không phải khuyến nghị dependency cho mọi máy.

## Kiểm chứng thực hiện

Hai test files đã đọc trước khi chạy để tránh test ghi/xóa thư mục tạm:
`tests/unit/core/test_transforms.py` và
`tests/unit/stage_a/test_scale_invariance.py`.

Kết quả: **11 passed, 0.52 s**. Kiểm tra CLR/ILR roundtrip, scale invariance,
Aitchison equivalence và donor tie-break. Chạy pytest với plugin autoload tắt,
`-p no:cacheprovider`, Python `-B`; không chạy full suite/training runners.
Kết quả này không xác nhận baseline paper parity, metadata encoding,
count-budget feasibility hoặc leakage toàn pipeline mới.

Skeleton được kiểm tra riêng qua AST/import/TOML/CLI; log thực tế lưu tại
`verification.json`. Không có thuật toán mới để benchmark trong lượt này.

## Trạng thái Git và giới hạn bảo toàn

Ngay trước thay đổi đã có các tracked deletions ở directive/: bốn PDF legacy
(2023 Imputing…, Hron, Missing_imputation_Diffusion_idea_plot, phylo) và bốn
file sec3/4/4_5/5 text. Đây là trạng thái có sẵn, không do lượt audit/skeleton.
Không restore, stage, commit, xóa hay diễn giải chúng thành chủ trương dọn dẹp.

Các thay đổi của lượt này ban đầu bổ sung `docs/research_plan_2026-10-11/` và
`nextgen/`. Những đề xuất archive/xóa sau này cần danh sách file cụ thể và
được người dùng cho phép trước; chưa có đề xuất xóa nào được thực hiện.
