# Kiểm chứng code trước khi tạo PR

Ngày: 2026-11-13. Phạm vi: package được promote từ `nextgen` lên root, main model,
baseline D+Other/R2, tests/configs và CLI staged; không kèm bộ hình/LaTeX.

## Before / After

- Base `a5a918a93d85124b6317d583c8a809d8b86184e7`: CLI baseline dưới `nextgen/`
  chưa có `--stage`, `--from-run` và `--pass`.
- Candidate: `python -m longitudinal_impute.run` hỗ trợ `full/init/refine/evaluate`;
  baseline-only vẫn không yêu cầu PyTorch.

## Checks thực chạy

1. Chạy từ worktree PR riêng; xác nhận module được import từ chính `src/` của
   worktree, không từ editable install trong workspace gốc:

   ```text
   python -m pytest -q -W error::RuntimeWarning -p no:cacheprovider
   257 passed in 20.91s
   ```

   Không skip. Python3.14.0, NumPy2.5.4, pytest9.1.1, PyTorch2.15.0+cpu.

2. Bốn subprocess CLI thật từ cwd khác, đường dẫn có khoảng trắng, trên fixture
   `tests/fixtures/covid_small.csv`: `full`, `init`, `refine`, `evaluate` đều exit0.
   Recipe kiểm tra: channels8, heads2, một outer round/epoch, batch2, model seed0;
   whole-feature mask rate0.1, evaluation seed7; native và masked cùng được kiểm.

   - Max absolute prediction difference full vs init→refine: **0.0** ở cả hai case.
   - Các metric rows evaluate bằng chính xác rows đã lưu ở refine.
   - `init` và `evaluate` chạy thành công với hook process mới từ chối mọi import
     `torch`; không cần fit hoặc construct neural network cho hai stage này.
   - Input fixture/schema/models giữ nguyên hash; cwd ngoài repo không bị ghi output.

3. `uv lock --check --offline` với interpreter đã kiểm và project-environment
   riêng: lock hợp lệ. Virtualenv Linux cũ bị track trong base không còn được
   version-control trong candidate; môi trường workspace gốc không bị xóa.

Các test staged parity, no-refit/no-torch và corrupt-artifact preflight có trong
`tests/test_stages.py`; không cần dữ liệu COVID thật để tái chạy regression suite.
Đây là operational checks, không phải benchmark accuracy hay chứng nhận sạch bug.

## Giới hạn còn mở

Xem [audit main-model](MAIN_MODEL_AUDIT_20261113_VI.md): hai P2 về coupling RNG H/Q
và tensor-budget roundoff vẫn chưa sửa, cùng các caveat P3/API đã công bố.
PR được mở dạng draft; không có claim main tốt hơn baseline, hội tụ bảo đảm,
natural-missing accuracy hoặc mọi kiểu input/platform đều đúng.

Các receipts `.harness/`/`outputs/` được nhắc trong tài liệu tiến độ là evidence
local lịch sử, không được publish trong code PR này. Không có benchmark/retraining
mới trên dataset thật khi chuẩn bị PR.
