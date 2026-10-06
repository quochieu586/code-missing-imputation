# Revised Research Plan: Compositional Imputation for SARS-CoV-2 Variants

**Date:** 2026-09-30  
**Status:** Methodological revision — addresses identifiability, confounds, and unfair baselines

**Implementation update — 2026-10-06:** E0 has established **Protocol U**; E1/E2 and the revised E3/E4 experiment have been implemented and audited. **E3/E4 mask-conditioning diagnostic (§11) completed 2026-10-06**: R1–R5 DONE, gates FAIL (conditioning/efficacy), numerical PASS. E5 remains **BLOCKED**, `finalists=[]`, `canada_evaluated=false`. The matched-mask/budget pilot (2 policies × 4 modes × 3 seeds × 2 budgets = 48 entries) confirms: (a) random-cell training does not transfer to whole-variant missingness; (b) 1120 updates improve conditioning for intervention policy but diffusion remains harmful vs init-only. Negative result recorded; Canada stays sealed. This document tracks implementation and reporting; no new planning Markdown required.

## Executive Summary of Changes

The original plan contains several threats to validity:

1. **Identifiability crisis:** JSD method requires known composition totals (Protocol K), but data audit shows uncertainty about whether 17 variants sum to `total_sequence`. Running JSD without resolving this makes results uninterpretable.

2. **Confounded comparison:** Hron uses flexible donors (partial overlap), JSD uses complete donors. Donor coverage differences confound distance metric comparison.

3. **Statistical power failure:** Only 5 complete-data locations = n=1 per test fold. Cannot support hypothesis testing or significance claims.

4. **Missing controls:** No dimensionality control (D vs D-1), no mask-only diffusion, no random reference selection for HKGLR.

5. **Evaluation mismatch:** Data has structured missingness (whole variants per location), but random-cell masking tests MCAR assumption. These measure different phenomena.

This revision reorganizes experiments to isolate causal factors, adds essential controls, and acknowledges statistical limitations.

## 1. Resolve Data Semantics First (Blocking Issue)

**Action E0-extended:** Before any method comparison, determine:

### 1.1 Composition Closure

```
For each row i with all 17 variants observed:
  - Does sum(17 counts) = total_sequence?
  - If not, what is the distribution of ratios?
  - Are there other variant categories not in the 17 columns?
```

**Decision tree:**
- If **YES** (≥95% of complete rows match within tolerance): Use Protocol K everywhere. Known total enables JSD and validates mass-based inverse.
- If **NO**: Use Protocol U. JSD cannot run in original form. Must either (a) drop JSD, (b) extend JSD with scale estimation (new method, separate evaluation), or (c) restrict to synthetic benchmarks with known closure.

**Do not proceed to method comparison until this is resolved.** Current plan defers this, allowing experiments that may be mathematically invalid.

### 1.2 Missingness Mechanism Documentation

Audit already shows: each location has exactly one missingness pattern across all time points.

```
For each location:
  - Which variants are never observed?
  - Is this pattern related to sequencing protocol, date range, or variant prevalence?
  - Document external reasons if known (e.g., assay design, reporting requirements)
```

**Impact:** If missing-completely-at-location (MCAL) rather than MCAR, then:
- Random-cell masking is an unrealistic benchmark
- Whole-variant masking is the primary scenario
- Linear/LOCF baselines will appear artificially weak in random-cell tests

## 2. Fair Initializer Comparison

Original plan conflates two factors: distance metric and donor eligibility.

### 2.1 Factorial Design

| Method ID | Distance | Donor Pool | Notes |
|---|---|---|---|
| **Hron-2a** | Aitchison on subcomposition | Flexible: must have query observed ∪ {target} | Original Hron |
| **JSD-complete** | JSD on subcomposition | Complete: all D parts observed | Original Tsagris (Protocol K only) |
| **Aitchison-complete** | Aitchison on subcomposition | Complete: all D parts observed | **Control**: isolates distance from donor pool |
| **JSD-flexible** | JSD on subcomposition | Flexible: must have query observed ∪ {target} | **Control**: isolates donor pool from distance (requires scale adjustment, mark as extension) |

**Primary comparison:** Hron-2a vs Aitchison-complete isolates donor flexibility effect. JSD-complete vs Aitchison-complete isolates distance metric effect. Only run JSD methods if Protocol K is confirmed.

### 2.2 Report Donor Coverage

For each method × fold × query pattern:
```
- Number of eligible donors
- Distribution of overlap sizes
- Fallback rate (fewer than k donors available)
- Effective k (actual neighbors used)
```

**Hypothesis revision:** 
- Original Q1 asks "Is JSD better for data with many zeros?"
- **Revised Q1:** "Does JSD distance better rank donors than Aitchison distance when both use the same donor pool?"

### 2.3 Alpha-Fréchet: Secondary Analysis Only

Do not include α-Fréchet in primary comparison. Reasons:
1. Adds a tuned hyperparameter (α) that Hron doesn't have
2. Original paper (Tsagris 2026) reports inconsistent gains for adaptive-α
3. Increases computational cost 10× (grid over k × α)

**Gate:** Run α-Fréchet only if basic JSD shows consistent improvement over Aitchison-complete and budget allows. Pre-register α grid and selection criterion.

## 3. Fair Log-Ratio Comparison

Original plan acknowledges dimensionality difference (D vs D-1) but doesn't control for it.

### 3.1 Capacity-Matched CSDI Variants

| Transform | Latent Dim | Parameterization | Model Capacity Control |
|---|---|---|---|
| CLR | 17 (rank 16) | Redundant, sum-to-zero | Channels = 16 to match ILR |
| ILR | 16 (full rank) | Minimal | Channels = 16 (baseline) |
| HKGLR | 17 (rank 16) | Redundant, 5-ref sum-to-zero | Channels = 16 to match ILR |

**Critical fix:** Original plan uses channels=16 for all but doesn't acknowledge CLR/HKGLR redundancy may interact with architecture. 

**Additional control:** Train one CLR model with channels=17 (unconstrained) vs channels=16 (matched). If results differ, dimensionality is confounded.

### 3.2 Projection Policy (Affects Hypothesis Tests)

Original §6.4 correctly notes: IID Gaussian noise in CLR vs ILR coordinates creates different covariance in composition space only if one is projected and the other isn't.

**Decision:**
- **No projection during reverse diffusion** (default DDPM, maintains Gaussian properties)
- **Project only at final step** before inverse: CLR → sum-zero, HKGLR → mean_H = 0, ILR → none needed
- Document this choice; projection-per-step is a different model

### 3.3 HKGLR Reference Selection Bias

Current rule: "5 variants with lowest missing rate globally."

**Problem:** Low missing rate may correlate with:
- High abundance (easier to detect)
- Temporal stability (less susceptible to measurement noise)
- Sequencing bias

These are biological/technical confounds, not "housekeeping" properties.

**Control experiment (E5-ref):**
```
For each fold:
  - Actual: 5 lowest-missing variants from train
  - Random: 5 random variants from train (3 seeds)
  - High-missing: 5 highest-missing variants from train
  
Compare HKGLR performance across reference sets.
If random ≈ actual, the selection rule doesn't matter (good).
If random < actual, low-missing has information content (interpret carefully).
If high-missing ≈ actual, the rule is reversed (big problem).
```

**Hypothesis revision:**
- Original Q4: "Is HKGLR with 5 low-missing variants helpful?"
- **Revised Q4:** "Does HKGLR improve over CLR when reference selection is not biased by the test query?"

## 4. Diffusion Controls (Tests What Diffusion Actually Learns)

Original plan has KNN+CSDI but unclear whether CSDI uses KNN initialization.

### 4.1 Mandatory Ablations

| Method | Initialization | Diffusion | Conditioning | Tests |
|---|---|---|---|---|
| **Init-only** | Hron or JSD | None | N/A | Initializer ceiling |
| **No-init** | None | CSDI | Raw visible + mask + time | Whether init helps |
| **Mask-only** | None | CSDI | Mask + time only (no values) | Whether model memorizes dataset stats |
| **Init+CSDI** | Hron or JSD | CSDI | Raw visible + init + mask + time | Full pipeline |

**Critical test:** If Mask-only ≈ Init+CSDI, then the model has learned dataset-level patterns and diffusion is not using the conditioning information. This would invalidate claims about imputation quality.

### 4.2 Conditioning Architecture Must Be Transparent

Original §7.2 proposes:
```
Condition encoder: raw visible + mask + metadata + init estimates
Target branch: full LR with noise
```

**Problem:** If condition encoder sees init estimates, and init estimates were computed from visible values, there's information redundancy. Model may ignore init and use only raw visible.

**Revised architecture:**
```
Branch 1: Raw-only condition encoder
  Input: visible counts/proportions, mask, time
  
Branch 2: Init-augmented condition encoder  
  Input: visible counts/proportions, mask, time, init estimates, init provenance flags
  
Both feed same target branch: noisy full LR
```

**Test:** Compare Branch 1 (No-init) vs Branch 2 (Init+CSDI). Improvement >5% on validation supports Q5; smaller improvement means useful initialization has not been demonstrated, rather than proving the network never uses init values. Supplement with fixed-checkpoint value ablations (remove raw values and init values separately, preserve masks/provenance/time); these are diagnostics and do not replace the performance gates.

### 4.3 Observed-Value Restoration Test

After diffusion inverse, observed counts must be restored exactly (not approximately).

**Test T17-extended:**
```python
for each row i:
    for each observed index j:
        assert output[i,j] == input_raw[i,j]  # exact equality
```

If this fails, the inverse/restoration logic is broken. Original plan mentions this but doesn't make it a hard gate.

## 5. Evaluation Scenarios (Matched to Data Structure)

### 5.1 Primary Scenario: Whole-Variant Missingness

**Justification:** Data audit shows each location has one fixed pattern = whole variants missing.

**Protocol:**
```
For each test location:
  1. Identify which variants are naturally missing there
  2. Hold out 1–3 additional complete variants as hidden eval set
  3. Train on remaining observed variants from train locations
  4. Predict the held-out variants across entire trajectory
  5. Evaluate MAE, M2, M3 on held-out variants
```

**Stratify by:**
- Number of variants hidden (1, 2, 3)
- Abundance of hidden variants (rare vs common, defined on train)
- Number of observed variants remaining (affects initialization feasibility)

**Strength:** Matches actual data structure. Tests whether methods can interpolate missing variant trajectories from other variants.

**Limitation:** Can only hold out variants that are actually observed in that location. Cannot test arbitrary variant combinations.

### 5.2 Secondary Scenario: Random-Cell Masking

**Justification:** Provides controlled comparison with known ground truth.

**Protocol:**
```
Use only the 5 complete-data locations (489 eligible rows).
For each row:
  1. Mask random cells at rates r ∈ {0.1, 0.3, 0.5}
  2. Ensure ≥3 observed parts remain (identifiability minimum)
  3. Use same mask across all methods
```

**Stratify by:**
- True zero vs true nonzero (original plan does this correctly)
- Rare vs common variants

**Strength:** Clean ground truth, all methods feasible.

**Limitation:** Unrealistic scenario. Good for method comparison, not deployment.

### 5.3 Removed: Contiguous Time Gap

Original plan includes this as tertiary scenario. 

**Recommendation: Drop it.** Reasons:
1. Data has whole-variant missingness, not temporal gaps within a variant
2. Only 489 complete rows across 5 locations = insufficient temporal sequences
3. LOCF/Linear baselines are strong for time interpolation; diffusion unlikely to help

If temporal imputation is important, collect more data or use a different dataset.

## 6. Statistical Inference (Acknowledging Limitations)

### 6.1 The n=1 Problem

With 5 complete-data locations and 5-fold CV, each test fold has **exactly 1 location**.

**What this means:**
- Cannot compute valid standard errors across locations
- Cannot do t-tests or ANOVA for location-level effects
- Cannot claim "statistically significant" based on 5 numbers

**Original plan §10.5 says:** "Only 5 complete locations: bootstrap or CI must be labeled exploratory and unstable; don't bootstrap cells."

**Revised position:** Do not report p-values or confidence intervals at all. Report:
```
- Point estimates per location (5 numbers)
- Mean and range across locations  
- Paired differences (Method A - Method B) per location
- Sign test (how many of 5 favor Method A?) — valid, non-parametric
```

### 6.2 Within-Location Inference

For random-cell masking at a given location:
- Multiple rows and multiple masks → can estimate variance
- Valid for "Does Method A beat Method B at this location?"
- **Not** valid for "Does Method A beat Method B in general?"

**Report separately:**
- Location-level summaries (n=5, descriptive only)
- Within-location tests (e.g., United States: 75 eligible rows, 1000 masks → paired t-test is valid for US only)

### 6.3 Hypothesis Decisions

Original plan has 6 hypotheses (Q1–Q6) with acceptance criteria.

**Revised acceptance rule:**
- **Consistent improvement:** Method A beats Method B in ≥4 of 5 locations (sign test p<0.19, one-sided)
- **Practical significance:** Mean improvement >10% on M2 OR >15% on nonzero MAE
- **No severe deterioration:** No location shows >20% harm on nonzero MAE

**Do not claim:** "statistically significant at p<0.05" based on 5 locations.

## 7. Revised Experimental Matrix

### Phase E0: Data Audit & Manifests (BLOCKING)

- Resolve 17-variant closure vs `total_sequence` 
- Document missingness mechanism per location
- Create frozen split manifests (location assignments, row IDs, feature order)
- Create mask manifests for both scenarios (whole-variant and random-cell)
- Define eligibility rules (exclude zero-total rows, minimum observed parts)

**Gate:** Numeric manifest files with hashes. Protocol K or U decision documented.

### Phase E1: Initializer Core

**Methods:**
- Hron-2a (flexible donors)
- Aitchison-complete (complete donors) — control
- JSD-complete (if Protocol K confirmed)

**Deliverables:**
- Implementations with unit tests T01–T08
- Donor coverage diagnostics
- Standalone benchmark (initialization quality without diffusion)

**Gate:** Numerical parity with Hron fixture. Fallback rates <5% on complete-data locations.

### Phase E2: Log-Ratio Transforms

**Methods:**
- CLR (channels=16 and channels=17)
- ILR (channels=16, Helmert basis)
- HKGLR (channels=16, actual 5-lowest, random 5, high-missing 5)

**Deliverables:**
- Forward/inverse with tests T09–T11
- Roundtrip error <1e-8 on float64
- Positivity policy documented and frozen

**Gate:** No transform-only "gain" (test T19). Reference selection control results available.

### Phase E3: CSDI Core + Controls

**Methods:**
- CSDI numerical parity with reference implementation (small fixture)
- No-init (raw conditioning only)
- Mask-only (tests memorization)

**Deliverables:**
- Tests T12–T17
- Leakage fixtures pass
- Sampling produces finite outputs

**Gate:** Mask-only performs worse than No-init (confirms conditioning is used). Observed restoration exact.

### Phase E4: Pilot Comparison (1 fold, 1 severity, 1 seed)

**Grid (random-cell scenario, complete-data fold):**
```
Initializers: {Hron-2a, Aitchison-complete, [JSD-complete if K], None}
Transforms: {CLR-16, ILR-16, HKGLR-16-actual}
Baselines: {Init-only, Linear, LOCF, Mean}

Core: 3 init × 3 transform = 9 combinations
Controls: 3 no-init diffusion, 4 baselines
Total: ~16 methods
```

**Metrics:**
- M2 (primary for method selection)
- Nonzero CLR MAE (primary for variant fidelity)
- JSD native (if Protocol K)
- M3, rare-variant MAE, donor coverage, runtime

**Gate:** 
- At least one diffusion method beats Init-only by >10% on M2
- No-init vs Init+diffusion isolates initialization value
- Choose ≤3 finalists for Phase E5

### Phase E5: Full Cross-Validation (Finalists Only)

**Scenarios:**
1. Random-cell masking: 5 folds (1 location each), 3 severities {0.1, 0.3, 0.5}, 3 seeds
2. Whole-variant masking: 5 folds, stratified by variant abundance, 3 seeds

**Grid size estimate:**
```
3 finalists × 5 folds × 3 severities × 3 seeds = 135 runs (random-cell)
3 finalists × 5 folds × 3 variant-sets × 3 seeds = 135 runs (whole-variant)
Total: 270 runs + baselines
```

**Deliverables:**
- Predictions, diagnostics, and metrics for all runs
- Per-location and pooled summaries
- Sign tests for consistent improvement
- Failure/fallback logs

**Gate:** Results stable across seeds (mean CV <20%). No silent failures.

### Phase E6: Sensitivity Analyses (If Budget Allows)

- Alpha-Fréchet (if JSD was promising)
- Pseudo-count policies (if zero-handling was critical)
- Projection-per-step (if geometry matters)
- Iterative refinement (challenges "single-pass" constraint)

**Do not run these unless primary questions (Q1-Q6) are answered.**

## 8. Revised Research Questions

| ID | Question | Fair Test | Acceptance |
|---|---|---|---|
| **Q1-revised** | Does JSD distance rank donors better than Aitchison when both use complete donors? | JSD-complete vs Aitchison-complete on Protocol K data | Consistent improvement (≥4/5 locations), M2 gain >10%, no severe harm |
| **Q2-revised** | Do flexible donors improve imputation over complete donors? | Hron-2a vs Aitchison-complete (both use Aitchison distance) | Measure donor coverage gain and error tradeoff; report conditional on overlap |
| **Q3-revised** | Does ILR geometry help diffusion over CLR, controlling for capacity? | ILR-16 vs CLR-16 vs CLR-17, all with same init and conditioning | Consistent improvement after checking dimensionality control |
| **Q4-revised** | Does HKGLR improve over CLR when references are not selected on test properties? | HKGLR-actual vs HKGLR-random vs CLR, all with same init | Actual beats random, and actual beats CLR consistently |
| **Q5-revised** | Does initialization improve conditional diffusion? | Init+CSDI vs No-init, both with same transform and architecture | Consistent improvement >5% on M2; check that No-init beats Mask-only |
| **Q6-revised** | Does diffusion improve initialization? | Init+CSDI vs Init-only, for each (init, transform) pair | Consistent improvement >10% on M2, no deterioration on nonzero MAE |

## 9. What This Revision Changes

### Removed Threats to Validity

1. **Identifiability:** Forced resolution of Protocol K vs U before running JSD
2. **Confounded comparison:** Added factorial design to separate distance from donor pool
3. **Unfair baselines:** Added capacity-matched CLR variants and reference selection controls
4. **Missing controls:** Added No-init, Mask-only, random-reference, dimensionality checks
5. **Evaluation mismatch:** Separated whole-variant (realistic) from random-cell (controlled) scenarios

### Acknowledged Limitations

1. **Statistical power:** Removed false precision (p-values, CIs) and use sign tests only
2. **Data scarcity:** 489 complete rows limits diffusion training; report sample-size sensitivity
3. **Temporal structure:** Removed contiguous-gap scenario (not matched to data)
4. **Biological interpretation:** Clarified that 5-lowest-missing references are a heuristic, not validated housekeeping genes

### Maintained Good Practices from Original

1. Location-based splits (prevent leakage)
2. Frozen masks (same evaluation for all methods)
3. Observed restoration (compositional constraints)
4. Comprehensive diagnostics (donors, fallback, failures)
5. Separate train/validation/test (no peeking)
6. Single outer pass (test iterative refinement only in sensitivity)

## 10. Implementation Priority

```
MUST HAVE (blocking):
  - E0: Data semantics resolution → Protocol K or U decision
  - E1: Hron + Aitchison-complete → fair donor comparison
  - E2: CLR-16 vs CLR-17 → dimensionality control
  - E3: No-init vs Mask-only → conditioning validation
  - E4: Pilot → method feasibility before full grid

SHOULD HAVE (if pilot shows promise):
  - E5: Full CV for finalists
  - Reference selection control for HKGLR
  - Whole-variant scenario (primary use case)

NICE TO HAVE (sensitivity only):
  - JSD-flexible (if Protocol K confirmed)
  - Alpha-Fréchet (if basic JSD wins)
  - Iterative refinement (challenges design constraint)
  - Pseudo-count policies
```

## 11. Blocker Resolution Path

**If Protocol K is NOT confirmed:**
- Drop JSD-complete (requires known total)
- Compare only Hron-2a vs Aitchison-complete
- Acknowledge this as a significant limitation
- Propose data collection or synthetic benchmark for future JSD testing

**If donor coverage is too low (<k neighbors for >20% of queries):**
- Reduce k (but report sensitivity)
- Add nearest-donor extrapolation fallback
- Report conditional metrics (stratified by neighbor availability)
- Acknowledge that methods differ in applicability, not just accuracy

**If complete-data sample is too small (<200 eligible windows after splits):**
- Reduce model capacity (channels=8, shorter windows)
- Use pretrained embeddings if available
- Focus on initialization quality, treat diffusion as exploratory
- Acknowledge sample size as primary limitation

The 200-window threshold is a planning heuristic, not a statistical minimum. The completed capacity8 branch changed width, window length and update count together and did not rescue efficacy; it does not establish capacity as the cause.

### 11.1 Current Evidence and Tracking — 2026-10-06

| Item | Verified state | Meaning for the next experiment |
|---|---|---|
| Finalized numerical audit | 500 method–scenario entries reproduced; 108 tests pass; no leakage found in tested current paths | Negative result is not evidence of a logic bug; preserve completed artifacts |
| E5 prerequisites | Conditioning fails 23/24 cells; efficacy passes 0/16 pairs; no finalists | Continue E3/E4 only; Canada stays sealed |
| Labels and capacity | 285 full-positive rows, 3 training countries, 74.96% raw zeros; main CLR has 19,713 parameters | Keep Protocol U/full-LR eligibility; partial donor rows do not become diffusion labels |
| Training budget | Main: 38 windows, 7 updates/epoch, 280 updates; 34/96 checkpoints select epoch40 | Undertraining is plausible for some models, not proven as the sole cause |
| Actual training masks | Among 140 full-length8 window-bank combinations: 1 hides one full variant, **0 hides ≥2** | Train scarcely encounters the primary whole-variant task |
| Conditioning diagnostic | 54 checkpoints evaluated; modes have similar epsilon losses; main value ablation changes epsilon RMS by 0.474–5.522% | Weak useful conditioning; no strong overfit evidence from this diagnostic |

Evidence: `artifacts/e3e4_rework_2026-10-05_b/`, `reports/e5_codebase_audit_2026-10-06.md`, `reports/e5_model_capacity_review_2026-10-06.md`, and `reports/e5_undertraining_review_2026-10-06.md`. These are historical evidence, not results of the proposed experiment below.

### 11.2 First Implementation: Matched Training Masks × Optimization Budget

**Scope:** new, preregistered E3/E4 experiment; same location split (train excludes Denmark/Canada, validation Denmark), CLR rank16 / **17 coordinates**, channels16, heads1, layers2, window8, Adam LR0.001, frozen transforms/positivity/restoration, MC8. Keep batch weighting identical across arms and log both batch-mean and cell-weighted loss. No architecture, scheduler or capacity change in this experiment.

| Factor | Control | Intervention |
|---|---|---|
| Training mask policy | 18 fixed random-cell banks, 5 hidden parts/row | 18 fixed banks: 9 identical paired random-cell banks + 9 whole-variant banks |
| Optimization budget | 280 updates (40 epochs with existing batching) | 1,120 updates (160 epochs), a diagnostic budget rather than an asserted optimum |

The 9 whole-variant banks cover **n_hidden={1,2,3} × abundance={rare,middle,common}**, selected using training data only. In each bank, each training location keeps the same hidden variant set across its entire trajectory and every window. Mask arrays/IDs, seed rule and abundance order are frozen before scoring. Control also uses18 banks to avoid confounding mask type with bank count; the earlier4-bank run is a historical reference, not the matched control.

For every policy, train **Mask-only, No-init, Hron+CSDI, Aitchison+CSDI**, seeds42/43/44. Recompute initialization **after each mask**, excluding the whole query location from donors. Keep full-positive LR targets separate from masked conditions; retain zero-valued observations and all truth-eligible rows, including rows without positive visible anchors.

This is **24 training trajectories** (2 policies ×4 modes ×3 seeds), each with selected checkpoints within the two nested budgets: **48 method–budget entries**. Save the best state within updates1–280 and within1–1,120 from the same trajectory; paired budgets are not independent replicates. Keep the random-cell branch secondary and report both mask families separately.

**Checkpoint rule:** each epoch evaluate a fixed, cell-weighted epsilon-loss fixture over **all103 eligible Denmark rows**, all windows and20 steps, with saved noise/row IDs. Use the three frozen primary whole-variant n2 masks; select the earliest argmin of their equal-weight mean. Log each scenario separately; random-cell loss does not select the checkpoint. Use train-fitted mean/scale/refs. This is a new selection rule applied equally to every arm, not a retrospective rewrite of existing checkpoints.

**Pilot decision:** assess each arm with the existing primary M2, nonzero CLR MAE, exact observed restoration and numerical requirements. Conditioning requires No-init M2 < Mask-only M2 for every seed and primary scenario; efficacy requires Init+CSDI >10% M2 improvement over the matching Init-only and no worse nonzero CLR MAE, for every seed/scenario. Q5's >5% gain over No-init is reported separately. Do not change thresholds, remove difficult rows, or substitute epsilon loss/value sensitivity for these gates.

Only qualifying arms advance. If several qualify, choose one shared mask/budget policy by equal-weight primary-scenario mean M2 of qualifying Init+CSDI pairs across seeds; break ties by shorter budget, then fixed policy ID. Lock this rule and IDs before the run. If none qualifies, record a completed negative pilot; E5 remains blocked.

### 11.3 Conditioning Architecture and Confirmation — Conditional Next Steps

If the matched-mask/budget pilot still fails conditioning, a **separately registered E3/E4 architecture ablation** may test raw-feature condition tokens: encode each of17 parts with visible value, mask, init value and provenance; let each noisy-LR target token attend to all raw-condition tokens. This preserves cross-part dependencies of CLR/ILR; do not copy a raw mask onto latent coordinates or treat hidden target values as input. Keep raw-only and init-augmented paths transparent as §4.2 requires.

Compare against the existing broadcast85→C encoder with the **same locked masks, updates, windows, optimizer and seeds**. Declare the policy independently of favorable test outcomes; when no pilot arm qualifies, use the preregistered mixed-mask/1,120-update diagnostic policy. Register parameter counts before training; match capacity within a declared tolerance (proposed±5%) or explicitly report an architecture-plus-capacity comparison. Include the four condition modes, Init-only/baselines, separate raw/init value ablations, step-band losses and gradient norms. Do not call a larger model's improvement proof that token conditioning alone helped.

If a pilot configuration qualifies, **confirm it before E5** across the original transform/control grid: CLR channels16/17, ILR16, HKGLR actual/random/high references, both initializers, Mask-only/No-init and three seeds. Keep all other settings shared and references train-only. Apply the full registered E3/E4 gates; a CLR-only pilot pass cannot waive geometry/reference controls or open full CV. Canada is evaluated only after qualification/selection is frozen, using the transferred training rule; no tuning on Canada.

### 11.4 Implementation Checklist and Fields for the Later Report

All steps below are **PLANNED — not implemented or run as of2026-10-06**. Update this table in place with `DONE / FAIL / BLOCKED`, artifact paths and concise results; keep failures visible.

| ID | Implementation / acceptance evidence | Status / result |
|---|---|---|
| R1 | New experiment config and output directory; freeze mask/budget/selection rules and source/input hashes; preserve old runs | **DONE** `artifacts/e3e4_mask_conditioning_2026-10-06/` config.json, source_hashes.json, masks_*, fixture_noise_metadata.json |
| R2 | `scripts/run_e5_mask_conditioning.py`: explicit mask-policy/bank inputs; location-wide masks; training-only strata; masked cross-fit initialization | **DONE** CLI with `--stage prepare|train|evaluate --out-dir --no-test`; 18 banks/policy, paired banks, 9 whole-variant strata |
| R3 | Runner: optimizer-update counter, two nested budget checkpoints; full-validation fixed fixture; fitted statistics/condition/checkpoint provenance | **DONE** 7 updates/epoch, budget280(epoch≤40)/budget1120(epoch≤160) from same trajectory; fixture: 103 DK rows × all windows × 20 steps, 3 primary n2 masks, seed 919, earliest argmin; mean/scale/refs logged |
| R4 | Tests: mask constant per location across windows, all 9 strata covered, no partition mixing/leak, budget/checkpoint replay, weighted aggregation, observed locking | **DONE** `tests/unit/stage_b/test_e5_mask_conditioning.py` 15 tests pass; full suite 123 tests pass |
| R5 | Run 24 paired trajectories; score 48 budget entries on frozen primary/secondary masks; recompute controls/gates and verify input preservation | **DONE** 24 traj × 160 epochs = 3840 epochs; 48 scored entries; gates recomputed per policy/budget; artifacts preserved |
| R6 | Architecture ablation in `src/stage_b/research_csdi_rework.py` only if needed; separate config/source/parameter counts and the same controls | **DEFERRED** pending R5 qualification (no arm qualified) |
| R7 | Full transform/reference confirmation and finalist freeze; open E5 only if all required gates pass | **BLOCKED** pending qualification |

**Gate verdicts (both budgets):**
- **Conditioning**: FAIL — Control policy fails (mask_only M2 !> no_init M2 for rare scenario); Intervention passes 2/3 at budget1120 but control failure blocks overall
- **Efficacy**: FAIL — Init+CSDI worse than init-only for all (init, transform) pairs; diffusion adds noise (M2 gain negative -11 to -33%)
- **Numerical/Restoration**: PASS — 128 entries checked, 0 violations (finite, non-negative, observed-locked)

**Selected epochs:** Budget280 checkpoints at epoch 32-40; Budget1120 at epoch 147-160 (earliest argmin over fixture). No checkpoint selected beyond epoch 40 for budget280.

**Mask coverage verified:** 9 intervention whole-variant banks cover all 3×3 strata; paired banks 0-8 identical to control; location-wide constant masking confirmed.

**Conclusion:** Negative result confirmed — random-cell training does not transfer to whole-variant; 1120 updates improve conditioning for intervention but efficacy remains negative (diffusion harms). E5 remains BLOCKED, finalists=[], Canada sealed. R6 architecture token-conditioning remains conditional; record proposal in plan if pursued separately.

For reporting, fill here: **experiment ID; config/source hash; actual rows/windows/updates/parameters; mask coverage; selected epochs and checkpoint rule; scenario-wise M2/nonzero MAE mean±seed SD; paired policy/budget differences; conditioning/efficacy/numerical verdicts; high-noise loss and value-ablation diagnostics; runtime; failures; finalists; Canada evaluated yes/no; conclusion and limits.** Store detailed machine results in the new experiment directory as JSON/CSV/PT/NPZ; this document remains the plan/status/report outline. Three seeds and one validation location support descriptive diagnostics, not independent location replication or significance/generalization claims.

### 11.5 Copyable Implementation Prompt for the Next Agent

Copy the following prompt into the implementation agent. Paths below are relative to the explicit workspace root; commands use that root. This is an implementation/training handoff, not a request to repeat the earlier audit or assume that BLOCKED means a software defect.

```text
Bạn làm việc tại:
C:\Users\admin\Tai_lieu\detaikhoahoc\COVID19\missing_impute\code-missing-imputation

Mục tiêu: hiện thực và chạy R1–R5 trong §11 của
directive/research_plan_revised_2026-09-30.md để kiểm tra hai giả thuyết:
(a) random-cell training không phù hợp whole-variant missingness;
(b) 280 optimizer updates chưa đủ. Giữ nguyên các controls/gates đã đăng ký.
Không hứa chữa được negative result; PASS hay FAIL đều phải audit và báo cáo.

ĐỌC CONTEXT theo thứ tự, không chỉ đọc kết luận:
1. directive/research_plan_revised_2026-09-30.md — §3–5, §7–8, §11.1–11.4.
2. reports/e5_model_capacity_review_2026-10-06.md
   reports/e5_undertraining_review_2026-10-06.md
   reports/e5_codebase_audit_2026-10-06.md
3. reports/e5_model_training_2026-10-06/assessment.json,
   scope.json, mask_and_schedule_checks.json, supporting_checks.json,
   learning_curves_summary.csv, checkpoint_full_loss.csv, noise_step_bands.csv.
4. artifacts/e3e4_rework_2026-10-05_b/config.json,
   report.md, addendum_2026-10-06.md, verification.json,
   validation_gates.json, validation_finalists.json, e5_status.json.
5. directive/e5_rework_agent_handoff_2026-10-05.md và
   directive/e3_e4_review_revised_2026-09-30.md — historical constraints.
6. scripts/run_e5_rework.py, scripts/rework_runtime.py,
   src/stage_b/research_csdi_rework.py, src/stage_b/research_csdi.py,
   scripts/run_e3_revised.py, src/stage_a/initializers_e1.py,
   artifacts/e2_transforms/transforms.py,
   src/evaluation/rework_gates.py, src/evaluation/research_pilot.py.
7. scripts/audit_rework_codebase_2026_10_06.py,
   scripts/audit_model_training_2026_10_06.py,
   scripts/summarize_model_training_2026_10_06.py,
   scripts/audit_e5_rework.py và tests/unit/.

FACTS PHẢI GIỮ:
- Protocol U; total_sequence không phải true mass; không bật JSD-complete.
- Inner train285 positive complete rows/3 countries; Denmark validation103
  eligible rows trong109 rows; Canada sealed. Không tự biến partial donors
  thành full-LR labels hoặc bỏ rows không có positive visible anchor.
- Current finalized result: E5 BLOCKED, finalists=[]; metrics500 entries khớp,
  108 tests PASS. Chưa tìm thấy logic/leak gây failure ở finalized paths.
- Main CLR16 là K17/rank16, C16, 19,713 parameters, L8, heads1, layers2;
 40 epochs=280 updates. Không cắt CLR thành16 coordinates.
- Dữ liệu thưa và conditioning yếu là hạn chế/giả thuyết, không phải một
  software bug đã được xác nhận. Superseded results không dùng làm evidence.

HIỆN THỰC:
1. Tạo experiment directory mới, đề xuất
   artifacts/e3e4_mask_conditioning_2026-10-06/; không overwrite nếu đã tồn tại.
   Lưu config/masks/noise/source/input hashes trước scores. Không tạo thêm MD:
   cập nhật status/results ngắn ngay §11.4 của research plan hiện tại;
   chi tiết machine-readable dùng JSON/CSV/PT/NPZ và log text.
2. Thêm runner riêng scripts/run_e5_mask_conditioning.py, tái sử dụng helpers
   đã audit. Đừng đổi defaults của runner cũ hoặc sửa frozen numerical core,
   E2 transforms/old metrics chỉ để làm kết quả đẹp hơn.
   Hiện thực CLI mới --stage prepare|train|evaluate, --out-dir, --no-test.
   Canada phải bị vô hiệu hóa cho toàn bộ R1–R5, kể cả khi pilot PASS.
3. Làm đúng factorial §11.2: control18 random-cell banks; intervention9 paired
   random +9 whole-variant banks; cùng bank count và frozen scheduling.
   Whole masks che cùng variant set suốt location/trajectory, đủ9 strata
   rare/middle/common ×1/2/3, train-only ordering. Các modes dùng cùng masks.
   Initializers tính sau masking, exclude query location, lưu provenance.
4. CLR16/C16/L8/LR0.001/heads1/layers2/MC8 giữ nguyên cho pilot.
   2 policies ×4 modes ×3 seeds=24 trajectories, 160 epochs=1,120 updates.
   Save best state trong budget280 và1,120 updates:48 scored entries;
   budget280 checkpoint không được chọn từ epochs sau40.
   Log optimizer updates, batch-mean/cell-weighted loss, fixed full-validation
   fixture từng epoch, selected epoch, mean/scale/refs, per-step losses và
   gradient norms. Reconstruct/replay được từ metadata đã lưu.
5. Selection fixture: all103 eligible Denmark rows/windows,20 fixed noises
   per step, primary whole-variant n2 rare/middle/common, equal scenario weight,
   earliest argmin. Không regenerate khác scoring masks vì filter rows.
   Evaluate primary masks và secondary random-cell riêng; full-LR truth chỉ
   là label, tuyệt đối không vào masked condition. Không copy raw mask sang
   latent loss. Fit normalization/references chỉ trên train.
6. Tests trước train: location-wide masks,9-stratum coverage, paired banks,
   split/cross-fit/hidden-value poisoning, window coverage, weighted fixture,
   deterministic checkpoint replay/budget boundaries, exact observed values.
   Giữ tests cũ và thêm test meaningful cho code mới; run toàn tests/unit.
7. Audit mới phải đọc --out-dir mới và chỉ ghi tại đó; không chạy audit cũ
   có B hard-coded để ghi lại experiment đã hoàn thành. Tính độc lập metrics
   từ predictions, re-derive gates, verify artifacts/old-source hashes,
   assert không có Canada predictions/cache_test/lock. Resume cache chỉ hợp
   lệ khi config/source/row/mask/fitted metadata signatures khớp.
8. Áp gates/selection §11.2, không đổi thresholds, không clip hoặc bỏ outliers.
   Lưu verdict và failure details. Nếu không qualify: cập nhật negative result
   và E5 BLOCKED. Không tự chạy thêm grid ngoài đăng ký để săn PASS.
   R6 architecture token-conditioning và R7 confirmation vẫn conditional;
   ghi đề xuất/status trong plan, không mở full CV/Canada từ CLR pilot PASS.

KẾT THÚC PHẢI BÀN GIAO:
- File/code đã sửa, commands thực chạy, test pass/fail, logs/artifact paths.
- Mask coverage và actual updates/parameter counts/checkpoint epochs.
- Bảng mỗi arm/scenario: M2, nonzero CLR MAE mean±seed SD, paired effects,
  conditioning/efficacy/numerical verdict; raw/init ablations và step-band loss.
- Frozen preservation/Canada seal verdict; finalists và E5 status trung thực.
- Điền R1–R5 ở §11.4: DONE/FAIL/BLOCKED và path bằng chứng. Chỉ ghi DONE khi
  work thực sự hoàn thành; phân biệt software audit PASS với research gate PASS.
```

Working runtime and initial test command (PowerShell):

```powershell
$researchRoot = 'C:\Users\admin\Tai_lieu\detaikhoahoc\COVID19\missing_impute\code-missing-imputation'
$researchPython = 'C:\Users\admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
Set-Location -LiteralPath $researchRoot
& $researchPython -B -c "import sys; sys.path.insert(0, 'scripts'); import rework_runtime; import torch, pytest; print(sys.version); print(torch.__version__)"
& $researchPython -B -c "import sys; sys.path.insert(0, 'scripts'); import rework_runtime; import pytest; raise SystemExit(pytest.main(['tests/unit', '-q', '-p', 'no:cacheprovider', '--basetemp=tmp/pytest_mask_conditioning']))"
```

The following is the **required new CLI contract**, not an existing runnable script as of this handoff. The agent must implement and verify `--help` before using these commands; all stages write only to the new output directory:

```powershell
$researchOut = 'artifacts/e3e4_mask_conditioning_2026-10-06'
& $researchPython -B scripts/run_e5_mask_conditioning.py --help
& $researchPython -B scripts/run_e5_mask_conditioning.py --stage prepare --out-dir $researchOut --no-test
& $researchPython -B scripts/run_e5_mask_conditioning.py --stage train --out-dir $researchOut --no-test
& $researchPython -B scripts/run_e5_mask_conditioning.py --stage evaluate --out-dir $researchOut --no-test
```

## 12. Conclusion

The original plan is thorough but contains methodological issues that would compromise interpretation:

1. **Identifiability** is deferred when it should block experiments
2. **Confounds** between distance and donor pool make Q1 unanswerable
3. **Statistical power** is overstated (n=5 locations cannot support hypothesis testing)
4. **Missing controls** leave alternative explanations unexamined

This revision reorganizes the experimental matrix to:
- Isolate causal factors (factorial design, matched controls)
- Acknowledge statistical limits (descriptive inference for n=5)
- Match evaluation to data structure (whole-variant primary, random-cell secondary)
- Force blocker resolution (Protocol K vs U) before method comparison

**Estimated effort with revisions:** 22–32 research days, comparable to original but with higher validity.

**Next immediate action (updated 2026-10-06):** E3/E4 mask-conditioning diagnostic (§11) completed. R1–R5 DONE; gates FAIL (conditioning: control policy fails; efficacy: diffusion harmful; numerical: PASS). Negative result recorded in `artifacts/e3e4_mask_conditioning_2026-10-06/` with `e5_mask_conditioning_status.json` showing BLOCKED, `finalists=[]`. R6 architecture token-conditioning ablation remains conditional — if pursued, register separately with parameter-count matching. R7 confirmation and E5 remain BLOCKED pending qualification. Do not rerun E0 or launch E5 as though decisions were pending.
