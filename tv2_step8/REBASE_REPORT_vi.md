# Báo cáo rebase step 1–8 lên main sau PR #5

2026-11-16. Phạm vi: 77 commit của `step1-decoys-sameday` → `step8-eval-v1-confirm` (từ `a5a918a`).

**Kết quả:**

- Toàn bộ 77 commit đã nằm trên `origin/main` mới (`53dbf7b`), thứ tự 1:1, chia thành 3 PR xếp chồng.
- Không đổi thuật toán, metric, ngưỡng, giả thuyết hay mask.
- Chạy lại model, sinh lại bảng step 1–7 và chạy lại `evaluate_v1.py` đều cho output giống từng byte.
- Ngoại lệ duy nhất: 7 trường `models_file` trong `tv2_step5/tier2_manifest.json` (mục 6).
- Chưa push.

## 1. Sao lưu và cách làm

**Sao lưu.**

- Tag cục bộ, không push: `backup/step1-orig` (`14818f3`), `backup/step4-orig` (`c1f0625`), `backup/step8-orig` (`1fbf2f3`).
- Branch cũ không bị xóa hay ghi đè.

**Rebase.**

- Từng commit được cherry-pick theo đúng thứ tự lên `origin/main`, với `merge.directoryRenames=true`.
- Name, email và ngày committer của commit gốc được giữ. Kết quả tương đương `git rebase`, và giữ được cả ngày committer.
- Branch `eval/rebased` chứa 77 commit đó, chưa có commit sửa nào.
- File mới trong `nextgen/src`, `nextgen/tests`, `nextgen/configs`, `nextgen/docs` được git đặt sang `src/`, `tests/`, `configs/`, `docs/`.
- Với mọi file không xung đột (`models_default.toml`, `micoda.py`, `evaluate.py`, tests, docs...), đã kiểm rằng kết quả bằng đúng merge 3 chiều sạch giữa bản cũ, `a5a918a` và main.

**Message và email (Bước 2).** Dùng `git filter-branch`, chỉ trên dải `origin/main..eval/rebased`:

- xóa 51 dòng `Co-Authored-By`;
- đổi author và committer `ngotuananh@MacBook-Pro-cua-Ngo.local` → `ngotuananh <tuananhngo2909@gmail.com>`;
- ngày giữ nguyên.

Kiểm tra sau khi chạy:

- grep `co-authored|claude` trên các message: 0;
- email: chỉ còn `tuananhngo2909@gmail.com`;
- message, ngày và tiêu đề khớp bản gốc, trừ dòng trailer.

## 2. Xung đột và cách giải

### `data.py` (a6c6bba, a5807df)

Main thêm `Dataset.times`, chỉ main model đọc; PR này thêm `days`, `week`, `time_kind`, `groups`, `group_map`.

- Code và test của main luôn truyền `times` bằng keyword.
- Test của PR này truyền `days ... groups` theo vị trí.

Vì vậy `times` được đặt **sau cùng** và `load_csv` truyền nó bằng keyword. Không mất chức năng nào của bên nào. Lần gộp đầu đặt `times` ở giữa và làm hỏng 15 test; lần làm lại sửa đúng hai commit này.

### `run.py` (8 commit: 9902bfc, a5807df, 7ef9146, c982e25, 39d78a0, 5ca70d8, 896673d, 3bc59a9)

Main xóa `nextgen/.../run.py` và viết lại `src/.../run.py`. Thay đổi của từng commit được chuyển vào file mới (3-way, sửa tay phần xung đột). File cũ không được bê lại. `MainModel`, các stage `full/init/refine/evaluate`, `artifacts.json` và `paired_summary` giữ nguyên.

| Commit | Thay đổi được chuyển | Điều chỉnh để chạy với run.py mới |
|---|---|---|
| 9902bfc | 4 model mới, `enabled = false`, `--only`, context (N, ngày, nhóm) cho `same_day_mean` | Stage `evaluate` (replay) chọn cùng tập bảng mà không dựng model: `read_models(..., construct=False)`. `--only` lấy lại từ run nguồn và từ chối nếu khác. |
| a5807df | `group_map` vào `config.json` | — |
| 7ef9146 | `--save-predictions`, `error_code`, `cell_counts`, `native_cells` | `predictions_<mask>.csv.gz` được ghi **trước** khi `artifacts.json` băm mọi file. |
| c982e25 | Dòng ghi chú "không dùng để xếp hạng" | Main in thêm bảng ghép cặp trên seed chung. Ghi chú chuyển ra ngay sau bảng tổng hợp, không nằm trong `print_table`, để không gắn nhầm vào bảng ghép cặp. |
| 39d78a0 | Khóa `model = "..."`, `build`, `resolved_models` | `validate_run_metadata` và `space()` đọc model qua `model_kind`, để replay chấp nhận bảng có tên riêng. |
| 5ca70d8 | `effective_params` | — |
| 896673d | `residual_rule` ở tầng restore | `execute()` truyền rule tới `run_model`; nhánh `MainModel` của main giữ ở đầu `run_model`. |
| 3bc59a9 | `whole_feature_matched`, `--mask-target` | `scheduled_cases` (replay) truyền `config['mask_target']` vào `make_mask`. Stage refine/evaluate lấy lại target từ run nguồn và kiểm sha256. |

### `nextgen/README.md` (7 commit)

Main xóa file này. Phần mỗi commit thêm được chuyển vào `README.md` ở gốc, đúng mục tương ứng. Link `../tv2_stepN/...` thành `tv2_stepN/...`.

## 3. Commit sửa thêm vào mỗi PR

Mỗi PR có một commit `chore: paths after repository restructure (nextgen → root)`. PR 2 có thêm commit sửa golden test. Mỗi PR có một commit docs.

### Đường dẫn

- `PYTHONPATH=nextgen/src` → `src`.
- `nextgen/{src,configs,tests,docs}` → `{src,configs,tests,docs}` trong `tv2_step*/`: script, `run_all.sh`, test, `eval_v1.toml` và báo cáo, ngoài các khối `auto:*` do script sinh.
- `config.json` của các run cũ ghi đường dẫn tuyệt đối dưới `nextgen/`. `summarize_runs.current_path()` ánh xạ chúng sang gốc repo, và chỉ dùng khi **đọc** file. Manifest vẫn chép nguyên giá trị đã ghi.
- `tests/test_time_and_groups.py` tính sai gốc repo (`parents[1].parent`), nên test trên bảng COVID đầy đủ bị skip âm thầm. Đã sửa; test này nay chạy.

### MICoDa A17 / A18

Trên main sau PR #5, `[micoda]` mặc định là A18. Vì vậy:

- **`configs/models_default_a17.toml`:** `models_default.toml` của step 1–8 (không đổi suốt 8 step), với `[micoda] other = false` ghi tường minh. `run_all.sh` của step 2/7, `tier2.py`, `tier3.py` và `eval_v1.toml` dùng file này. `configs/models_default.toml` của main giữ A18.
- **`summarize_runs.micoda_other(config)`:** khóa ghi tường minh được ưu tiên. Không có khóa thì dùng mặc định của runner đã ghi `config.json`: A17 nếu không có khóa `stage` (trước PR #5), A18 nếu có. `validate_pair` dùng hàm này.
- **`tier2.recorded_table()`:** bảng `[micoda]` của run cũ (không có `other`) được điền `other = false` trước khi so với file A17.
- **`run.check_rule`:** bảng `[micoda]` không có `other` lấy mặc định của `MICoDaCovid` (A18). Chỉ `other = false` bị từ chối residual rule.
- **Test:**
  - các test từng giả định mặc định A17 nay ghi A17 tường minh (`test_residual_rule`, `test_tier01_eval`, `test_sameday_ablation`, `test_tier01`, `test_residual`);
  - model `main` không còn nằm trong danh sách baseline của các test firewall và golden.
- **Ghi chú:** đầu `REPORT_vi.md` của step 1–8 và trong `eval_v1.toml` có câu: micoda trong các step này = A17 (other=false); micoda_other = A18. Trên main sau PR #5, micoda mặc định = A18. Tên model trong bảng kết quả không đổi.
- **`configs/models_all.toml`:** preset "mọi model" của main thêm `[same_day_mean]` (baseline mặc định từ step 1), kèm test của main.

### `eval_v1.toml` và kiểm tra đóng băng

`evaluate_v1.py` từ chối chạy nếu sha256 của `eval_v1.toml` khác `masks_manifest.json`, tức khác file đã đóng băng trước khi sinh mask. Đổi đường dẫn thì phải sửa file, nhưng manifest đóng băng thì không được sửa. Cách giải:

- Đường dẫn mới và ghi chú nằm trong một khối `# --- amendment (rebase on PR #5) ---`.
- `v1.frozen_bytes()` bỏ khối đó và đổi đúng các đường dẫn đã khai báo (`v1.PATH_MOVES`) về như cũ. Kết quả **giống từng byte** file đóng băng gốc, và sha256 bằng `config_sha256` trong `masks_manifest.json`.
- Kiểm tra đóng băng và trường `config_sha256` của `v1_manifest.json` dùng `v1.frozen_sha256()`.
- Bất kỳ thay đổi nào khác vẫn làm kiểm tra fail; có test cho việc này.

### Golden test (Bước 4)

`test_without_a_rule_every_model_is_unchanged_byte_for_byte`:

- Hron, Tsagris, `same_day_mean` và các mức sàn vẫn so hash từng byte.
- `micoda` và `micoda_other` so với mảng golden (`tests/fixtures/residual_rule_micoda_golden.json`) bằng `assert_allclose(rtol=1e-9, atol=1e-9)`; status phải khớp chính xác.
- Các mảng được sinh từ code trước step 4. Trên máy này chúng tái lập đúng 6 hash GOLDEN cũ; script sinh chỉ ghi file nếu khớp.
- Không nới test nào khác.

## 4. Kiểm tra (Bước 6)

Môi trường: macOS, Python 3.13.5, numpy 2.1.3, `-W error::RuntimeWarning`.

| Kiểm tra | Kết quả |
|---|---|
| Test ở đầu `eval/pr4-step1` (`tests/`, `tv2_step1`) | 329 passed, 0 skipped |
| Test ở đầu `eval/pr-tier01-ablation-residual` (+ `tv2_step2`–`4`) | 435 passed, 0 skipped |
| Test ở đầu `eval/pr-tier23-v1` (+ `tv2_step5`–`8`) | 492 passed, 0 skipped |
| Chạy lại native: 11 model (7 model chính với A17, micoda_other, hron_cap, tsagris_cap, micoda_other_cap) | 11/11 CSV đã điền giống từng byte, khớp `csv_sha256` trong `tier2_manifest.json` và file trong thư mục run cũ |
| Chạy lại masked `whole_feature` seed 0: hron, tsagris, same_day_mean, micoda (A17), micoda_other | Mask giống hệt; mọi dòng long file (5 625 ô mỗi model) giống hệt run step 2 |
| `evaluate_v1.py check` | ok, 29 file của step 1–7, không seed xác nhận nào |
| `evaluate_v1.py evaluate` trên run có sẵn (seed 200–209, không chạy model) | `v1_results.md`, `v1_pairs.csv`, `v1_cases.json`, `v1_manifest.json` và khối auto của `REPORT_vi.md` giống từng byte bản đã commit. Lượt chạy lại Tầng 2 bên trong cũng giống từng byte output step 5. |
| Sinh lại bảng step 1–7 bằng script của từng step (`summarize_runs`, `tier01`, `ablation`, `residual`, `tier2 --rerun`, `tier3`, `matched`) | Mọi bảng, CSV, manifest và khối auto của báo cáo giống từng byte |
| Message / email trên `origin/main..eval/pr-tier23-v1` | 0 dòng `Co-Authored-By`/`Claude`; email duy nhất `tuananhngo2909@gmail.com` |
| File không được commit | Không có `runs/`, `runs_attempt1/`, `tv2_m0/`, `tv2_m1/`, `.DS_Store`, `__pycache__`, `*.log` |

## 5. Lần chạy lại dùng run.py mới của main

- CSV đã điền và long file giống từng byte, nên thuật toán và đường đi dữ liệu không đổi.
- `summary.json` và `config.json` của lượt chạy lại **không** được so, vì vốn khác: thời gian chạy, commit, đường dẫn, và các khóa mới của main (`stage`, `pass`, `artifacts.json`, `predictions/*.npy`, `paired`).
- Các script step 1–8 đọc được cả run cũ lẫn run của runner mới. Test end-to-end trên fixture chạy bằng runner mới.

## 6. Chỗ không giống từng byte

1. **`tv2_step5/tier2_manifest.json`:** 7 giá trị `models_file` của lượt main đổi từ `models_default.toml` sang `models_default_a17.toml`, vì kiểm tra tái dùng nay đọc file A17. File được sinh lại bằng `tier2.py` từ run có sẵn và commit trong chore của PR 3. Mọi byte khác và mọi bảng step 5 không đổi. Nếu không sinh lại, cờ "Tầng 2 giống step 5" của step 8 sẽ thành False.
2. **`tv2_step8/eval_v1.toml`:** khác byte vì có khối amendment. Bản đóng băng suy ra từ nó giống từng byte bản gốc (mục 3).
3. **Hash commit:** đổi do rebase và filter-branch. Xem [HASH_MAP.md](HASH_MAP.md). Báo cáo step 6–8 ghi hash mới, hash cũ trong ngoặc. Khối auto và thư mục run vẫn ghi hash cũ, vì đó là commit lúc chạy.

## 7. `nextgen` còn lại (có chủ ý)

- Dòng ghi chú ở đầu mỗi `REPORT_vi.md`.
- Số test lúc chạy, dạng "nextgen 199 (…)".
- Câu lịch sử "`nextgen/` giống hệt giữa hai commit" ở step 2.
- Các `*_manifest.json` đã sinh: path group map do run ghi.
- Traceback của lượt đầu trong `attempt1_evaluate.txt`.
- Code, docstring và test của `current_path` / `PATH_MOVES`.
- Khối amendment trong `eval_v1.toml`.
- `HASH_MAP.md`, `PR4_UPDATE_vi.md`.
- Hai dòng mở đầu `README.md` của main.

Ngoài phạm vi: 25 file docs lịch sử của main dưới `docs/` (audit, review cũ) không bị đụng tới.

## 8. Chưa xác minh

- Golden test chỉ chạy trên macOS/numpy 2.1.3. Chưa chạy trên Linux/numpy 2.5.5 để xác nhận sai lệch của micoda_other:wf nằm trong `rtol=1e-9`.
- `tv2_m0/`, `tv2_m1/` và branch `m0-framework-skeleton`, `m1-single-pass` không thuộc nhiệm vụ, không bị đụng tới.
