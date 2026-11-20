# Evaluation Tier 0–3 (main, main_nometa, baselines) + slides

## Intent
- loại việc: Score finished runs with the merged evaluation module (Tier 0–2 via existing script, Tier 3 U1–U4 via a driver reusing tier3 functions with main/main_nometa added), then add per-tier slides to the existing 8-slide Beamer deck.
- consumer: User presenting MainModel (4-field metadata) vs main_nometa vs Hron/Tsagris/MICoDA A18/same_day_mean on slides.
- contract: Runs in outputs/eval_meta4_20261120_seed300_309 (seeds 300–309; masks whole_feature, random_cell, whole_feature_matched; native pass). No refit, no source change in src/. Tier 3 reuses tier3.one_case/u1–u4 tables and frozen thresholds unchanged; only the model list and run loading are adapted (protocol gates for seeds 0–4 / a17 config / clean tree do not apply and are reported as a deviation). Exploratory label everywhere.
- parity: Tier 0–2 numbers come from evaluate_5tier.py as written; Tier 3 thresholds/functions from src/longitudinal_impute/evaluation/tier3.py unchanged.
- anh em: Existing slides 01–08, figures, theme, outputs of earlier runs and src/ stay untouched except main.tex/build.py inclusion of new slides/figures.
- non-goals: Tier 4 (no code exists, task not chosen), new training, confirmatory claims, installing packages (no matplotlib; use pgfplots), commit/push.

## Acceptance criteria
| ID | Behavior | Check |
|---|---|---|
| AC1 | Tier 0–2 report produced after whole_feature_matched finishes | REPORT_5TIER_VI.md + CSVs exist in outputs/eval_meta4_20261120_seed300_309/report |
| AC2 | Tier 3 U1–U4 on seeds 300–309 for all three masks, with main and main_nometa in the model set | tier3 report + CSV exist; thresholds equal tier3.py constants |
| AC3 | Deck extended: one summary-table slide per tier (0,1,2,3) then at most 2–3 selected classic plots (bar/line/scatter, pgfplots); builds with build.py, page count updated, no overfull/Type3 issues | build.py all exit 0; inspect rendered pages |
| AC4 | Independent QA recomputes headline numbers and checks slides match CSVs | hf-qa session report |
| AC5 | Network figure (figures/04_network.tex) redrawn in the research-paper pipeline style (boxes plus small illustrative glyphs, like the user's screenshot: grid icons for Z,m, attention glyphs, gate curves, bar output p) and it shows the metadata branch: continent, classification, group embeddings plus Linear(median age), summed into one vector m and added to h(0) for every time and part (network.py:53-67); minimal text on the figure | Render exports/png/04_network.png and inspect |
| AC6 | One new slide explains the metadata embedding: short intro sentence, compact table or figure (field, encoding, size), ASD-STE100 sentences, no trivial notes | Render and inspect the slide |
| AC7 | Whole deck visual review: no overlapping or clipped text, no overflow, figures clear and Q1-style; defects fixed; build.py page count and inputs correct | build.py all exit 0 with 0 diagnostics; inspect every build/pages png |

## Context
| Nguồn sự thật | Đã đọc | AC |
|---|---|---|
| Run state | `Get-ChildItem outputs/eval_meta4_20261120_seed300_309/*/predictions` → whole_feature 60, random_cell 60, whole_feature_matched 53 (running, pid 6344) | AC1 AC2 |
| Tier 0–2 scorer | outputs/eval_meta4_20261120_seed300_309/evaluate_5tier.py:1 docstring "Score the eval_meta4 runs with the evaluation module (tier 0, 1, 2)" | AC1 |
| Tier 3 loader gates | src/longitudinal_impute/evaluation/tier3.py:201-249 fixed SEEDS 0–4, RUNS a17 config, clean-tree check, MODELS = REAL + DECOYS | AC2 |
| Tier 3 scoring entry | src/longitudinal_impute/evaluation/pipeline.py:215-227 tier3_mask → one_case, u1_table, u2_table, u3_table, sensitivity_lines, u4_table | AC2 |
| Deck build | presentation/main_model_8slides/build.py:18-19 FIGURES tuple, EXPECTED_PAGES = 8; main.tex inputs slides/01..08 | AC3 AC4 |
| Plot tooling | `kpsewhich pgfplots.sty` → found; `python -c "import matplotlib"` → ModuleNotFoundError | AC3 |
| Metadata injection | src/longitudinal_impute/network.py:29-32 nn.Embedding per field + nn.Linear(1,C) for median_age; :60-67 m = age term + sum of field embeddings; h = h + m[:, None, None, :] after h = Linear(Z,m) + e_f + e_t (:51-52) | AC5 AC6 |
| Deck state 2026-11-21 00:32 | `Get-Content main.tex; Select-String build.py EXPECTED_PAGES` → main.tex and build.py rewritten 00:19 by the user's parallel edits: eval slides 09-14 no longer included, EXPECTED_PAGES = 8; figures/04_network.tex is a plain-box version; backup %LOCALAPPDATA%\Temp\opencode\deck_backup_20261121_0032 | AC5 AC7 |
| QA ses_e0607ba15ffePWtW9OsaaTg6AW | `Select-String slides/13_eval_tier3.tex "D3 reliably"` → line 14 overstated; QA numbers verified; defects: slides not included, D3 conclusion wording overstates, U1-U3 named before definition on 13_eval_tier3.tex, "A18" undefined | AC3 AC4 |

## Style rule for new slides (user, 2026-11-20)
- Write like ASD-STE100: complete sentences with subject, verb and object; no abbreviations and no clipped phrases.
- Each slide starts with a short intro sentence so a reader outside the project can follow.
- Remove unneeded words, but keep the information a newcomer needs (define every metric and model name on first use).
- Avoid dense multi-line tables that are hard to read; keep tables short and labelled clearly.
- The user is editing existing slides in parallel: add NEW slide/figure files only; touch main.tex/build.py minimally; never rewrite 01–08.


## Resolved decisions
- Tier 3 seeds = 300–309 (only seeds with saved predictions); deviation from frozen protocol stated on slide.
- Plots in pgfplots standalone figures under figures/, exported like the existing ones.
- New slides inserted before 08_references; references slide stays last.

## Steps
1. Plan-check.
2. Wait (native background wait) for whole_feature_matched; run evaluate_5tier.py (AC1).
3. hf-worker: Tier 3 driver + report (AC2), then slides/figures + build (AC3).
4. hf-qa: recompute and inspect (AC4); repairs via same worker.
5. Report to user.

## Status
DONE with open defects 2026-11-21. Frozen deck SHA256 3684F12521B0D25AD73447DDF1478F06F8019B4B6C62289910E2D82432E9979F (15 pages). Producer ses_e05f064f2ffeFPe0eo8wJuc5h0; QA ses_e05d6e629ffeVWeGF6guaCoDeA (claude-agent/sonnet#medium): PASS with minor concerns. Confirmed defects (not fixed, user asked one QA then handoff): wrong source line ranges in captions - 04_network.tex:5 network.py:51--86 (file has 81 lines); 04b_metadata.tex:21 omits main_model.py:403--421 (_fit_metadata, fields line 408); 05_r2.tex:10 data.py:162--217 (restore_counts is at data.py:337).
