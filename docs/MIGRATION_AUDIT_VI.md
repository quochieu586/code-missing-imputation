# Bàn giao workflow ba baseline tại gốc repo

> Báo cáo dưới đây mô tả migration và kho bằng chứng của workspace local vào
> 2026-11-02. PR code chỉ promote package/config/tests/scripts/docs cần thiết và
> giữ snapshot runtime cũ đã được Git track; không publish toàn bộ archive local,
> môi trường, dataset mới hay các receipts trong `outputs/` và `.harness/`.

Ngày kiểm chứng: 2026-11-02. Kết quả: migration hoàn tất; audit độc lập theo `thermo-nuclear-code-quality-review` không còn finding bắt buộc.

## Cấu trúc đã chuyển

- `src/longitudinal_impute/`, `tests/`, `configs/`, `scripts/`, `pyproject.toml`, `uv.lock` và tài liệu hiện hành được đưa từ `nextgen` ra gốc. Không có wrapper hoặc bản runtime thứ hai.
- Code KNN + CSDI, runner E0–E5, tài liệu, kết quả, checkpoint, presentation và output cũ nằm trong `archive/legacy_knn_csdi_2026-11-02/`.
- Project map, tiến độ và bằng chứng cũ trong `.archify`, `.harness`, `.skill-build`, `.vibe-wise` cũng được archive. Review baseline cũ nằm trong `archive/nextgen_history_2026-11-02/`.
- Hai CSV nguồn vẫn ở `data/`; output mới theo các ví dụ nằm trong `outputs/`. Cấu hình công cụ `.kilo`, `.opencode` và `opencode.json` được giữ tại gốc.

## Bằng chứng kiểm chứng

| Yêu cầu | Kết quả | Bằng chứng |
|---|---|---|
| Giữ nguyên thuật toán | Cả 12 file Python runtime trùng SHA256 với bản trước khi chuyển | [Inventory gốc](../archive/migration_2026-11-02/inventory.json), [tổng hợp kiểm chứng](migration_verification.json) |
| Bảo tồn working tree cũ | Kiểm tra lại 13.716 mục của manifest chính và 8.599 file lịch sử bổ sung; cả hai lượt kết thúc exit 0 | [Script manifest chính](../archive/migration_2026-11-02/promote_nextgen.py), [manifest bổ sung](../archive/migration_2026-11-02/workspace_history.json) |
| Dữ liệu không đổi | SHA256 của cả hai CSV nguồn khớp trước migration | [Tổng hợp kiểm chứng](migration_verification.json) |
| Import và đóng gói đúng root | Editable install nạp root `src`; hai console entrypoint chạy được | [Log và report handoff](../outputs/handoff%20audit/verification.json) |
| Test hiện hành | 115 passed; RuntimeWarning là lỗi; 24 file source/test vẫn khớp hash của lượt test | [JUnit](../outputs/migration-tests/tests.xml), [log](../outputs/migration-tests/pytest.txt) |
| Không thu thập test legacy | `pytest --collect-only` mặc định chỉ thu 115 test hiện hành | `testpaths` và `norecursedirs` trong [pyproject.toml](../pyproject.toml) |
| Dữ liệu thật và schema tổng quát | Sáu case đều đạt: fixture COVID và microbiome, bảng COVID thật, mỗi dataset với hai mask | [Validator cuối](../outputs/handoff%20audit/final-validation.json) |
| Đường dẫn có khoảng trắng và cwd khác | Module CLI chạy với đường dẫn tuyệt đối; verifier unit được gọi từ ngoài repo | [Report handoff](../outputs/handoff%20audit/verification.json), [report unit](../outputs/migration-tests/verification.json) |
| Output đúng hợp đồng | Giữ observed numeric/text và metadata; hữu hạn, không âm, không vượt total; masked coverage 100%; CSV/JSON nhất quán; từ chối overwrite và giữ nguyên output cũ | [Validator cuối](../outputs/handoff%20audit/final-validation.json) |

Bảng COVID thật có 11.671 dòng, 17 target và 86.050 NaN tự nhiên. Kiểm chứng dùng seed 7, rate 5%: che 5.518 ô với `random_cell` và 5.644 ô với `whole_feature`. Cả ba model đều hoàn thành native và masked. Toàn bộ sáu case gồm 36 lượt model native/masked.

Validator được bổ sung kiểm tra CSV/JSON trong khi lượt COVID đang chạy. Sau khi process kết thúc, phiên bản validator cuối đã kiểm tra lại cả sáu output; `final-validation.json` ghi hash script tương ứng. Không cần fit lại vì không có thay đổi thuật toán.

## Chạy lại

Chạy từ gốc repo, dùng thư mục output mới mỗi lần:

```powershell
uv sync --frozen --extra test
.venv\Scripts\python.exe scripts/verify_three_baselines.py --output outputs/verification_new `
  --data data/covariants.csv --schema configs/baselines_covariants.toml
.venv\Scripts\python.exe scripts/verify_handoff.py --output "outputs/handoff new" `
  --data data/covariants.csv --schema configs/baselines_covariants.toml
.venv\Scripts\python.exe archive/migration_2026-11-02/promote_nextgen.py
.venv\Scripts\python.exe archive/migration_2026-11-02/archive_workspace_history.py
```

Môi trường đã kiểm chứng là Windows CPython 3.14.0, NumPy 2.5.4 và pytest 9.1.1, cài từ lockfile. Virtualenv cũ được bảo tồn cục bộ trong `archive/environments/`, không có cam kết chạy sau khi di chuyển. Một cache pytest do môi trường quản lý có thể còn ở gốc; cache không chứa runtime hay project map và không được dùng trong lượt kiểm chứng.

## Phạm vi kết luận

Kiểm chứng xác nhận khả năng vận hành và bảo toàn hành vi sau migration. MICoDa vẫn dùng A17/R1 mặc định; giới hạn Other/budget trong [contract](THREE_BASELINE_CONTRACT_VI.md) không đổi. Chưa dùng các kết quả này để xếp hạng khoa học giữa baseline.

Microbiome được kiểm chứng bằng fixture đi cùng tests; chưa có dataset microbiome thực tế để chạy. Không chạy lại R oracle, huấn luyện diffusion hoặc kiểm định độ chính xác của NaN tự nhiên. Archive giữ snapshot lịch sử, không cam kết phục hồi pipeline cũ thành workflow độc lập.
