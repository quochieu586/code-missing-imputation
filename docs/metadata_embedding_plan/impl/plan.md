# MainModel metadata embedding — implementation (TDD)

## Intent
- loại việc: Feature implementation following docs/metadata_embedding_plan/DESIGN_SIMPLE_VI.md (QA PASS, SHA256 63AADD19...9B13).
- consumer: User comparing main vs main_meta vs main_meta_noloc.
- contract: Embed exactly 5 static fields (continent, location, median_age, classification, group) via lookup Embedding(q,C) + Linear(1,C) for z-scored age, summed and added to h after network.py:46; flag MainModel(metadata=True) DEFAULT ON (user 2026-11-20: current main is a skeleton, metadata needed to beat same_day_mean); metadata=False keeps old skeleton as ablation; schema key metadata_path; fail-fast validation.
- parity: metadata=False ⇒ predictions/history/metrics identical to pre-change code; existing tests pass (tests that build MainModel on fixtures without metadata pass metadata=False explicitly).
- anh em: Hron/Tsagris/MICoDA, evaluation module and data files untouched; configs/baselines_covariants.toml gains metadata_path (baselines ignore it), models_all.toml [main] uses default (on) and gains [main_nometa] metadata=false.
- non-goals: Running the seed 300-309 comparison, tuning, unknown-label handling, time-varying metadata, commit/push.

## Acceptance criteria
| ID | Behavior | Check |
|---|---|---|
| AC1 | ≤7 new tests (6 + baseline-parity test added with default-on change) written first and failing before implementation (design §5 tests) | RED run output recorded |
| AC2 | Implementation makes new tests pass; default on; metadata=False is numerically identical to pre-change code | Off-parity test against pre-change snapshot of outputs |
| AC3 | Full test suite passes | pytest tests -q with basetemp in %LOCALAPPDATA%\Temp\opencode |
| AC4 | New configs (baselines_covariants_meta.toml, models_main_meta.toml) and a small real-data smoke run of main_meta on a fixture or 1 seed | Command + summary status |
| AC5 | Independent QA of final diff | Separate hf-qa session |

## Context
| Nguồn sự thật | Đã đọc | AC |
|---|---|---|
| Design | docs/metadata_embedding_plan/DESIGN_SIMPLE_VI.md:39-87 decision table, change list, tests | AC1 AC2 AC4 |
| Network | src/longitudinal_impute/network.py:7-60 CompositionNet, injection after :46 | AC2 |
| MainModel | src/longitudinal_impute/main_model.py:52-75,242,275,284-294,342-412 | AC2 |
| Data/runner | src/longitudinal_impute/data.py:16-64; run.py:505-508 dataset loaded before models | AC2 AC4 |
| Baseline suite | pytest tests → 495 passed, 2 skipped (prior session) | AC3 AC5 |

## Steps
1. Worker (openai/gpt-6.1-sol#low): snapshot off-parity reference, write tests (RED), implement (GREEN), full suite, smoke run.
2. Coordinator reproduces decisive checks.
3. hf-qa reviews diff and reruns tests.
4. Repair via same worker if needed (≤3).

## Status
DONE 2026-11-20. Worker 1 ses_e0802751effe4H9cnqw2EEwxug (default off: RED 6 failed, GREEN, 501 passed). Worker 2 ses_e07f0c0b9ffe6RA5jjzMxTlYiX (default ON, configs, README; full 502 passed, 2 skipped). QA ses_e07ebbeaaffe1c5aBcm1gtxMTj: no logic defects; 2 test findings (nonportable oracle, history length) repaired by worker 2, rechecked by coordinator: 502 passed 2 skipped; oracle absent -> 6 passed 1 skipped. Real-data parity: main_nometa seed 300 MAE prop 0.02860001186177639 == old main in outputs/eval_all_20261120_seed300_309. Smoke outputs/meta_smoke2_20261120_152903 all ok. Repair after QA not re-reviewed by QA (test-only change, coordinator reran). Revision hashes: %LOCALAPPDATA%\Temp\opencode\meta2-revision.json.
