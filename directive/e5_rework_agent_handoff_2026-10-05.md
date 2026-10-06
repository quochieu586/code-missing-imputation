# Prompt bàn giao cho agent tiếp theo — sửa E3/E4 và quyết định E5

Bạn là kỹ sư nghiên cứu tiếp tục công việc trong repository này. Hãy **thực hiện công việc**, không chỉ đề xuất kế hoạch. Mục tiêu là xác định nguyên nhân và chạy một đợt sửa đã đăng ký; kết quả âm được chấp nhận. Đừng tạo PASS bằng thay ngưỡng, chọn test seed hay loại rows. Người dùng chỉ yêu cầu chuẩn bị bàn giao ở lượt trước; agent trước chưa chạy training mới.

## 1. Workspace và môi trường chạy đã xác nhận

```powershell
Set-Location -LiteralPath 'C:\Users\admin\Tai_lieu\detaikhoahoc\COVID19\missing_impute\code-missing-imputation'
$ResearchPython = 'C:\Users\admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
```

Bundled CPython **3.12.14**, PyTorch **2.14.0+cpu**, pytest **9.1.1**. `.venv-e3-revised/Scripts/python.exe` không launch được base interpreter. Đã xác nhận bundled Python có thể import các research wheels từ `.venv-e3-revised/Lib/site-packages`.

`scripts/rework_runtime.py` thêm repository root và research site-packages vào sys.path, disable bytecode. Đặt `import rework_runtime` ở đầu script mới trong `scripts/`; hoặc dùng command dưới đây. Không cần cài torch mới để tiếp tục.

```powershell
@'
import sys
from pathlib import Path
sys.path.insert(0, str(Path('scripts').resolve()))
import rework_runtime
import torch, pytest
print(sys.version)
print(torch.__version__, pytest.__version__)
'@ | & $ResearchPython -B -
```

Git HEAD lúc audit: `9ba4e53fcf715f2659d547f28418163092727f9a`. Working tree có nhiều untracked source/artifacts và các PDF/.gitignore đã modified từ trước. **Không reset/stash/clean** các thay đổi đó. Git commit một mình không định danh code hiện tại; lưu code SHA-256.

## 2. Tài liệu phải đọc

1. `directive/e3_e4_review_revised_2026-09-30.md`.
2. `directive/implementation_checklist_2026-09-30.md`, đặc biệt Execution status.
3. `directive/research_plan_revised_2026-09-30.md` §3/4/5/7/11, original LR/CSDI plan §7–8/10, `directive/revision_summary_2026-09-30.md`.
4. `src/stage_b/research_csdi.py`, `src/evaluation/research_pilot.py`, `src/stage_a/initializers_e1.py`, `scripts/run_e3_revised.py`, `scripts/run_e4_revised.py`, tests revised E3/E4.
5. **Pha A mới:** `artifacts/e3e4_rework_2026-10-05/diagnosis.md`, `phase_a.json`, `verification.json`, các CSV và `scripts/diagnose_e5_rework.py`.

Prompt gốc đầy đủ của người dùng còn ở:
`C:/Users/admin/.codex/attachments/98435e14-6973-413e-8532-7824a1895022/Pasted text.txt`.
Nếu attachment không còn truy cập được thì prompt này giữ các yêu cầu quan trọng; đừng tự suy ra yêu cầu đã được nới lỏng.

## 3. Trạng thái đã chạy và output thực tế

### E0.1/E0.2 hoàn thành

- CSV có 11,671 rows, 150 locations, 17 variants thực tế: recombinant,20A,20B,20C,20E,Beta,Alpha,Gamma,Delta,Kappa,Epsilon,Eta,Iota,Lambda,Mu,Omicron,S:677.
- Closure: 403/516 complete = **78.10%**; positive-sum sensitivity 403/489 = **82.41%**; 27 zero-sum complete rows.
- Revised rule >=95% **all complete rows** chưa đạt → **Protocol U**, không dùng original JSD-complete hoặc `total_sequence` như true mass.
- Missingness: 150/150 locations có đúng 1 pattern; whole-variant **primary**, random-cell secondary.
- Output: `artifacts/e0_checklist_2026-10-05/`; scripts `audit_composition_closure.py`, `audit_missingness_patterns.py`.
- Data SHA-256: `bbb1aacdf46ebef345ca818f086d5586f4aded45475b00b6ee9bd89fc94a90ab`.

### E3/E4 revised cũ: research gates FAIL

- Inner Denmark: 285 full-positive train rows /38 windows. Outer: 388/51; window<=8, no overlap/padding. Complete-label data chỉ thuộc 3 inner /4 outer training countries, dù donor pool dùng nhiều partial locations.
- 20 epochs, 20 steps, 2 MC samples, channels16, 2 layers, ~19.7k params, seed42.
- Conditioning validation: CLR No-init M2 173.801968 vs Mask-only176.040822 PASS; ILR138.985062 vs138.194457 FAIL; HKGLR177.279364 vs175.328315 FAIL.
- Canada Init-only M2 9.714041/9.895114 vs Init+CSDI114–152; nonzero CLR MAE initializer~1.93–1.96 vs diffusion~3.63–4.08. Historical test chỉ mô tả, **không dùng chọn config**.
- Scale flags cả diffusion và initializer. E4 efficacy FAIL. E5 BLOCKED, finalists=[].

### Pha A mới đã chạy; chưa Pha B/C/D

Command đã chạy thành công:

```powershell
& $ResearchPython -B scripts/diagnose_e5_rework.py
```

**Không chạy lại command này vào cùng folder**: đã có `phase_a.json`, runner từ chối overwrite. Muốn bổ sung diagnostics hãy dùng script/folder mới, đọc output cũ.

Output chính:

```json
{
  "status": "COMPLETE",
  "metrics_verified": 28,
  "checkpoint_count": 24,
  "failure_records": 27,
  "terminal_alpha_bar": 0.014780417246189144,
  "zero_epsilon_reverse_amplification": 8.225392918039187,
  "frozen_files_unchanged": true
}
```

- `training_traces.csv`: 24 stochastic train traces, mỗi trace20 epochs.
- `loss_summary.csv`: 20/24 slopes cuối âm; **4 inner ILR slopes dương**, không nói mọi model đang tiếp tục giảm loss. Artifact cũ **không có validation-loss history**. Final checkpoint row-length1 diagnostic không phải comparable train/val loss gap.
- `phase_a.json`: train/sampling latent distributions, oracle, poisoning, environment/runtime/commit.
- No-init validation standardized train absmax~4.47–4.87 vs sampled~15.86–18.91; sample-to-sample latent RMS~5.93–7.67. Chỉ 2 samples, chưa định lượng MC error.
- `row_diagnostics.csv`: stable row IDs, location/date, eligible, flag, per-row M2, visible positives, visible probability/scale; `m2_contributions.csv`: flagged validation diffusion rows đóng góp~17.6–37.8% M2; nonflagged M2~98–157. Trimmed/median chỉ diagnostic.
- **Sửa hiểu nhầm H6:** M2 được tính từ squared CLR/Aitchison distance, không phải raw-count squared error. Sau locking/zero replacement, scale có thể tác động ratios.
- Oracle **train/Denmark only**: anchored M2~1.44e-30 Denmark; overall5.0924; no-positive-anchor9 rows, trong đó3 eligible. Oracle vẫn2 scale flags, max3177. Train oracle M2~17.48, anchored~1.23e-30, 8 flags. Positivity oracle hidden raw zeros=.5 theo view, không tuyên bố raw roundtrip exact.
- Poison hidden Denmark NaN/±1e200: actual initializers và all4 conditions identical; donor IDs/crossfit training banks hợp lệ. Không tìm thấy leak trong paths đã kiểm.
- 49 existing tests PASS. Unit tests riêng cho **diagnostic script mới chưa viết**; cần bổ sung trước Phase C.

Audit mới `verification.json`: legacy E3 validation/test + E4 **46 entries**, independently recomputed M2/M3/3 CLR MAEs abs tol1e-12, common truth eligibility, finite/nonnegative, exact observed locking; frozen files unchanged; 49 regression tests PASS.

## 4. Command kiểm tra và audit an toàn

```powershell
Get-Content -LiteralPath 'artifacts/e3e4_rework_2026-10-05/diagnosis.md' -Raw
Get-Content -LiteralPath 'artifacts/e3e4_rework_2026-10-05/verification.json' -Raw
Get-Content -LiteralPath 'artifacts/e3e4_rework_2026-10-05/verification_tests.txt' -Raw
Get-Content -LiteralPath 'artifacts/e3e4_rework_2026-10-05/loss_summary.csv'
```

Chạy tests tiếp theo bằng runtime có torch, không cache:

```powershell
@'
import sys
from pathlib import Path
sys.path.insert(0, str(Path('scripts').resolve()))
import rework_runtime
import pytest
raise SystemExit(pytest.main(['tests/unit', '-q', '-p', 'no:cacheprovider']))
'@ | & $ResearchPython -B -
```

Kiểm frozen files trước/sau bất kỳ công việc tiếp theo:

```powershell
@'
import sys,json,hashlib
from pathlib import Path
root=Path.cwd()
baseline=json.loads((root/'artifacts/e3e4_rework_2026-10-05/frozen_hashes_before.json').read_text())
changed=[rel for rel,want in baseline.items() if not (root/rel).exists() or hashlib.sha256((root/rel).read_bytes()).hexdigest()!=want]
print({'checked':len(baseline),'changed':changed})
raise SystemExit(bool(changed))
'@ | & $ResearchPython -B -
```

`scripts/audit_e5_handoff.py` là verifier đã dùng để tạo audit mới. **Không chạy lại vào folder cũ** khi `verification.json` tồn tại. Đặc biệt **không chạy `scripts/verify_e4_review.py`** để “chỉ đọc”: main của nó ghi lại verification/hashes trong **E4 cũ**, trái yêu cầu bảo toàn. Dùng phiên bản verifier mới ghi ở experiment mới.

## 5. Công việc tiếp theo: hoàn thành A rồi B→C→D

### A bổ sung và chốt trước B

Đọc `diagnosis.md`; kết luận H1/H2/H3 được ủng hộ có giới hạn, H4 chưa rõ, H5 gross mismatch đã loại nhưng MC chưa rõ, H6 hiểu sai raw metric đã bác, H7 không tìm thấy leak đã kiểm.
Bổ sung tests diagnostic functions và xác nhận poisoning/statistics/reference selection độc lập cho code mới. A1 không có lịch sử val loss thì ghi thiếu, không fabricate. Nếu cần thêm oracle/covariance decomposition, chỉ inner train/validation. Giữ mọi observations về Canada là descriptive.

### B: preregistration bắt buộc trước training

Viết `preregistration_<date>.md` và machine-readable config ở **folder mới**, gợi ý `artifacts/e3e4_rework_2026-10-05_b/` (hoặc actual client date nếu khác). Folder A ở trên là completed handoff, đừng sửa lịch sử.
Chỉ chọn thay đổi dựa evidence train/validation; khả năng hợp lý để kiểm tra là epochs/validation checkpointing, capacity8/shorter windows và MC>=8. Không khẳng định các thay đổi này chữa được lỗi trước khi chạy.
Đăng ký grid hữu hạn, seed ít nhất3 (ví dụ42/43/44), validation locations, masks, early stopping rule/max epochs, scope controls, runtime và quy tắc stopping. Chọn config **chỉ validation**, không Canada.
Đăng ký restoration sensitivity riêng nếu thử; train-only estimator/threshold, log clipping fraction; giữ legacy restoration làm control. Không che việc zero-positive-visible scale không định danh dưới U; cần fallback/abstention minh bạch, không âm thầm bỏ khỏi metrics.
Nếu A không đủ rõ để chọn một thay đổi lớn, trình bày options cụ thể rồi hỏi người dùng ở gate; không tự chạy grid tùy hứng.

### C: implementation, tests và controlled experiment

- Thêm experimental model/runner mới; giữ source/reference/artifacts được hashes cũ bảo toàn. Không monkeypatch CONFIG/OUT của old runner rồi gọi main để tạo run mới.
- **CSDICore hiện hardcode channels16**; TransformerEncoder nhead4 đòi channels chia hết4; **channels17 không chạy được với cấu trúc này**. Khi implement CLR17, đăng ký và báo policy heads/attention projection; giữ architecture comparison công bằng, tránh head-count confound. Không chỉ đổi tên clr17 nhưng train channels16.
- ≥3 model seeds, seed-paired No-init/Mask-only/Init+CSDI, same masks/rows/sampling policy.
- Bắt buộc controls CLR16 vs **actual channels17**, HKGLRactual vs random vs high-missing, random reference3 fixed seeds; refs fit inner train riêng từng split, tie-break name.
- Chạy **whole-variant primary** và random-cell secondary. Các whole masks train-only rare/middle/common, n-hidden1/2/3, frozen before scores; ghi realized coverage. Baselines Linear/LOCF dùng train fallback khi toàn trajectory variant thiếu; báo theo scenario riêng.
- Full-LR labels chỉ complete-positive ground truth; không nâng partial locations thành labels qua imputation. Overlap windows không mix locations/partitions, không padding thật giả; eval mỗi row/cell một lần bằng aggregation đã đăng ký.
- Log validation epsilon loss bằng fixed fixture có same windows/steps/noise qua epochs; save selected checkpoint/epoch theo rule validation. Không so noisy train mean với length1 fixture để claim gap.
- Save config/commit/code+data hashes, row IDs, checkpoints, predictions, latent samples, condition and initializer provenance, runtime, all failures. Runner từ chối overwrite completed runs và ghi failed run honest.
- Mọi code change có meaningful unit tests; 49 old tests phải PASS; thêm tests channels17/8, deterministic windows/aggregation, leakage poisoning, sensitivity restoration, gate boundaries. Không làm yếu old tests.
- Verification **independent** M2/M3/CLR MAE tolerance1e-12, common truth eligibility, stable IDs, finite/nonnegative, observed zeros exact, hashes old artifacts unchanged.
- Freeze selected config **before Canada**; đánh giá Canada đúng một lần cho frozen selected set. Nếu validation không có qualified config, dừng negative và không mở Canada mới để tune.

### Gates giữ nguyên và D

Conditioning: Mask-only M2 > No-init trên validation cho **mọi registered transform/control**, báo seed-wise và mean±sd theo rule preregistered. Efficacy: same initializer Init+CSDI improves M2 **strictly >10%**, nonzero CLR MAE không xấu hơn. Scale guard1000, finite/nonnegative, observed locking exact. Sensitivity xử lý scale phải được đăng ký và giải thích; **không tự coi flag “đã giải thích” là PASS hoặc nới ngưỡng**.
Nếu tất cả required gates PASS, chọn≤3 finalists bằng validation, ghi E5 UNBLOCKED nhưng **chưa chạy E5 khi chưa được user duyệt**. Nếu không PASS sau finite registered grid, ghi negative result, E5 BLOCKED/finalists=[]; không thêm tuning rounds ngoài đăng ký.

## 6. Bảo toàn và output cuối

Cấm tune/chọn seed/checkpoint theo Canada, dùng true mass, chỉnh metric/gate/eligibility, silent clipping, overwrite E0–E4/revised/PDF, retrain current E3 in place, hoặc lấy COMPLETE=PASS.
Tạo mới `diagnosis.md`, preregistration, config, results từng pha, verification/tests/hash snapshots, `report.md`. Báo H1–H7 trước/sau, bảng gates tách validation/test, exact changes và rationale, limitations/seed/sample sizes, E5 status/finalists/bước tiếp. Không p-values/claims generalization từ một outer fold.

Tiêu chí bàn giao: có thí nghiệm được đăng ký + kết quả + audit, hoặc có gate chặn được ghi rõ và câu hỏi cụ thể; không kết thúc bằng một kế hoạch chung hoặc bảo người dùng tự làm phần đã được giao.
