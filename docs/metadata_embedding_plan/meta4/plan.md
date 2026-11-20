# MainModel metadata = 4 fields (no location) + full 10-seed evaluation

## Intent
- loại việc: Small code/config change (default metadata_location off) then full real-data evaluation and slide-ready report.
- consumer: User preparing slide report comparing MainModel (with metadata) against MainModel without metadata and baselines.
- contract: User 2026-11-20: "bỏ loc chính là meta luôn, lấy 4 trường thôi" — main uses continent, median_age, classification, group; location embedding off by default. Then run full seeds 300–309, masks whole_feature and random_cell, evaluate with evaluation.tier1.
- parity: main_nometa (metadata=False) stays numerically identical to pre-metadata code; new main must equal previous main_meta_noloc smoke (seed 300 MAE prop 0.025663725159758847). Baselines unaffected.
- anh em: Hron/Tsagris/MICoDA/same_day_mean source, evaluation module, data files, existing outputs (eval_all_*, meta_smoke*) untouched.
- non-goals: Confirmatory/blind claims (seeds 300–309 already seen), accuracy on natural NaN, tuning, commit/push, installs.

## Acceptance criteria
| ID | Behavior | Check |
|---|---|---|
| AC1 | MainModel default metadata=True, metadata_location=False; flag kept; configs/models_all.toml has [main] (4 fields) and [main_nometa]; [main_meta_noloc] removed; README/tests updated | Updated tests RED then GREEN; full suite passes |
| AC2 | New main equals previous main_meta_noloc; main_nometa equals old main | Seed-300 real-data numbers match previous smoke/eval exactly |
| AC3 | Full run: hron tsagris micoda same_day_mean main main_nometa, seeds 300–309, two masks, predictions saved, all statuses recorded | summary.csv status per model/seed, logs |
| AC4 | Slide-ready report: tier1 tables, paired main vs each (incl. main_nometa) with sign test, runtime, limits | REPORT_VI.md + CSVs + SHA256 manifest; independent QA recompute |

## Context
| Nguồn sự thật | Đã đọc | AC |
|---|---|---|
| src/longitudinal_impute/main_model.py | main_model.py:377 `metadata=True, metadata_location=True`; :409 location embedding gated by flag | AC1 AC2 |
| configs/models_all.toml | models_all.toml:35 [main_nometa] metadata=false; models_all.toml:47 [main_meta_noloc] metadata_location=false | AC1 AC3 |
| tests/test_metadata_embedding.py, tests/test_main_run.py | test_metadata_embedding.py:132-151 asserts location default True and three aliases; test_main_run.py:127 alias set | AC1 |
| Smoke outputs/meta_smoke2_20261120_152903/summary.csv | `Import-Csv outputs/meta_smoke2_20261120_152903/summary.csv` → main_meta_noloc mae_proportion 0.025663725159758847; main_nometa 0.02860001186177639 = old main | AC2 |
| outputs/eval_all_20261120_seed300_309 | `Get-ChildItem outputs/eval_all_20261120_seed300_309/report -Name` → (tier1_tables.csv, paired_main_vs_baselines.csv, runtime.csv, REPORT_VI.md, make_report.py) | AC3 AC4 |

## Resolved decisions
- Keep `metadata_location` flag in code (default False) so the location variant stays reproducible; no alias in models_all.toml.
- Rerun all six models fresh in one run directory per mask (shared masks), not reuse old runs.
- Output root: outputs/eval_meta4_20261120_seed300_309/.
- Worker/QA model: openai/gpt-6.1-sol#low (user choice for this line of work).

## Steps
1. Plan-check.
2. hf-worker: TDD change (AC1), full suite, seed-300 parity smoke (AC2).
3. hf-qa: independent check of frozen code revision.
4. Coordinator launches full run (AC3) in background; waits for completion.
5. hf-worker builds report (AC4); hf-qa recomputes; coordinator reproduces headline numbers.

## Status
WAITING 2026-11-20 ~20:57. AC1: code edited by interrupted worker (RED 2 failed / GREEN 19 passed logs in %LOCALAPPDATA%\Temp\opencode\meta4-*.log); coordinator full suite 502 passed 2 skipped; default metadata=True metadata_location=False. AC2 PASS: outputs/meta4_smoke_20261120_204919 main 0.025663725159758847, main_nometa 0.02860001186177639 (exact). Frozen revision: outputs/eval_meta4_20261120_seed300_309/code_revision_sha256.csv. AC3 running detached (Start-Process), pids in outputs/eval_meta4_20261120_seed300_309/pids.txt; check summary.csv in whole_feature/ and random_cell/ on resume, do NOT relaunch while pids alive. Pending: independent QA of AC1, AC4 report.

QA ses_e06c125e8ffeHs740mbnneOK2c: AC1+AC2 PASS, no defects; independent suite 502 passed 2 skipped; test hashes test_metadata_embedding.py E6E1FFB6..., test_main_run.py 4E9A1F17..., README 8A84E37A.... AC3 still running (both child python processes alive, ~885 s CPU each at check). Next: AC4 report after both summary.csv exist.
