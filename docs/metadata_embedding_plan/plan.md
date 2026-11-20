# MainModel metadata embedding — v2: Seki-paper analysis + simplest explainable plan

## Intent
- loại việc: Research + design plan only (no source edits in this ticket). Supersedes DESIGN_VI.md scope; that file is kept as history.
- consumer: User implementing and defending (slides/report) metadata embedding in MainModel.
- contract: Analyse Misato Seki's CSDI-based papers (local csdi.pdf = phylo.pdf, Bioinformatics Advances 2025 §2.4 metadata FT embedding; csdi_2023.pdf BIBM 2023) plus other Seki/Zhang/Imoto papers on this model line found publicly, CSDI 2021 and the revised proposal; then give the simplest plan to embed exactly 5 static fields (continent, location, median_age, classification, group) of data/original_long_format.csv, with explicit "why this choice / what it means" for every decision.
- parity: MainModel default (metadata off) output unchanged; same masks/seeds 300-309 and evaluation.tier1 for later comparison.
- anh em: Hron/Tsagris/MICoDA, evaluation module, data files untouched.
- non-goals: Other covariates, time-varying metadata, attention/fusion modules, hyperparameter search, training, commit/push.

## Acceptance criteria
| ID | Behavior | Check |
|---|---|---|
| AC1 | Paper analysis: how Seki et al. embed metadata (FT embedding eq. 7, d=8, spatial vs channel-wise concat, train-only metadata, reported effect), how CSDI 2021 side info works, what the proposal says; claim → evidence (file + page) → meaning for us | QA re-reads cited pages via pdftotext |
| AC2 | Simplest design per field with rationale and meaning (why embedding vs one-hot vs scalar, why this dim, why this injection, why location is/isn't included) and what is deliberately NOT done | QA checks simplicity, consistency with code and data |
| AC3 | Concrete change list (files/functions with current line refs), default-off flag, tests, evaluation protocol (main vs main_meta vs main_meta_noloc) | QA checks line refs in src/longitudinal_impute |
| AC4 | Independent QA of frozen DESIGN_SIMPLE_VI.md | Separate hf-qa session, hash |

## Context
| Nguồn sự thật | Đã đọc | AC |
|---|---|---|
| Seki 2025 | pdftotext -l 1 ../research/csdi.pdf → "Bioinformatics Advances, 2025 ... Misato Seki, Yao-zhong Zhang, Seiya Imoto"; phylo.pdf same header | AC1 |
| Seki 2025 §2.4 | pdftotext -f 3 -l 3 ../research/csdi.pdf → line 59 "embedded using a FT (Gorishniy et al. 2021)", line 76 "channel-wise concatenation approach" | AC1 AC2 |
| CSDI 2023 / 2021 / proposal | pdftotext -l 1 → csdi_2023.pdf BIBM 2023 DOI 10.1109/BIBM58861.2023.10385703; 2021_CSDI Tashiro et al.; proposal T. Q. Hieu Oct 2026 | AC1 |
| Data | docs/metadata_embedding_plan/DESIGN_VI.md:42 section 2 data audit, recomputed by QA: static per location, no missing, cardinalities 6/150/114/4/4 | AC2 AC4 |
| Network code | src/longitudinal_impute/network.py:7-60 CompositionNet init/forward, h injection point after line 46 verified by QA | AC3 |

## Resolved decisions
- Output: docs/metadata_embedding_plan/DESIGN_SIMPLE_VI.md (Vietnamese). Prior DESIGN_VI.md kept untouched.
- Worker and QA model: openai/gpt-6.1-sol#low (user re-selected 2026-11-20 after OpenCode credit failure).
- Web search allowed only for public paper metadata (titles, authors); no private data in queries.

## Steps
1. Plan-check.
2. hf-worker: read papers, find related Seki papers, write DESIGN_SIMPLE_VI.md.
3. Coordinator inspects; hf-qa verifies frozen hash.
4. Repair via same worker if needed; report to user.

## Status
DONE (design) 2026-11-20. Worker ses_e08210630ffeoxLF0gemFIwsKN (openai/gpt-6.1-sol#low) wrote DESIGN_SIMPLE_VI.md. QA ses_e081d6aaaffeU8tGgt1Lh70893 found 2 defects (2023 CNN misattribution, contradictory loading contract) + 2 concerns (extra bias, age sd=0); same worker repaired once; QA rechecked affected parts: PASS on SHA256 63AADD193D2D7254C65CDD35AB5A2B8A5EBC5C2FB7B259F7A1DA6B89EA779B13. No source code changed. Temp: %TEMP%\seki_simple_producer, %TEMP%\qa_simple.
