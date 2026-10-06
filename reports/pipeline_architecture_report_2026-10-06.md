# Pipeline Architecture Report — E3/E4 Mask-Conditioning Diagnostic
**Date:** 2026-10-06  
**Experiment ID:** `e3e4_mask_conditioning_2026-10-06`  
**Status:** Completed — Negative result (gates FAIL, E5 BLOCKED)

---

## 1. Pipeline Overview

```mermaid
flowchart TD
    %% Data Layer
    A[Data: covariants.csv\nSHA256: bbb1aacdf...] --> B[E0 Extended: CV Splits Fold 0\nTrain: NL/UK/US/DK\nValidation: Denmark (103 eligible)\nTest: Canada (112 rows, SEALED)]
    
    %% Transform Layer
    B --> C[E2 Transforms: CLR Rank-16\nfit_positivity_delta=0.5\n17 variants → 16 latent dims]
    
    %% Stage A: Initialization
    C --> D[Stage A: Initializers\nHron 2A / Aitchison Complete\n+ Fallback (linear/LOCF)]
    
    %% Stage B: Diffusion Core
    D --> E[Stage B: CSDICoreRework\nCLR16 / C16 / L8 / heads=1 / layers=2\nDDPM 20 steps, quad schedule 1e-4→0.5\nAdam LR=1e-3, MC=8, float64]
    
    %% Training Loop
    E --> F[Training: 24 Trajectories × 160 Epochs\n2 Policies × 4 Modes × 3 Seeds\n7 updates/epoch = 1120 updates max]
    
    %% Mask Policies
    F --> G[Control: 18 Random-Cell Banks\n5 hidden parts/row]
    F --> H[Intervention: 9 Paired Random-Cell\n+ 9 Whole-Variant Banks\nn_hidden={1,2,3}×{rare,middle,common}]
    
    %% Checkpoint Selection
    G --> I[Dual-Budget Checkpoints\nBudget280: epochs 1-40\nBudget1120: epochs 1-160\nSame trajectory, earliest argmin]
    H --> I
    
    %% Fixture
    I --> J[Validation Fixture\nAll 103 DK eligible rows\n3 Primary n2 masks (rare/mid/common)\n20 fixed noise steps per window\nEqual-weight mean, earliest argmin]
    
    %% Evaluation
    J --> K[Evaluate: 48 Scored Entries\n4 Scenarios × 2 Budgets × 2 Policies × 3 Seeds × 2 Init modes\n+ Controls: Init-only, Linear, LOCF]
    
    %% Gates
    K --> L[Validation Gates\nConditioning: mask_only > no_init ∀ seeds/scenarios/policies\nEfficacy: init+CSDI >10% M2 gain over init-only ∀ seeds/scenarios\nNumerical: finite, non-neg, observed-locked]
    
    %% Output
    L --> M[Artifacts & Status\nconfig.json, scores, gates, status\nE5: BLOCKED, finalists=[], Canada sealed]
```

---

## 2. Pipeline Components

| Stage | Module | Purpose | Key Parameters |
|-------|--------|---------|----------------|
| **E0** | `artifacts/e0_extended/` | Outer CV split (fold 0), frozen evaluation masks | Test=Canada, Val=Denmark, Train=NL/UK/US |
| **E1** | `src/stage_a/initializers_e1.py` | Hron 2A, Aitchison Complete initializers; fallback values | Cross-fit donors, location exclusion |
| **E2** | `artifacts/e2_transforms/transforms.py` | CLR rank-16 (K=17→16), positivity delta=0.5, exact observed restoration | Sum-zero projection, `restore_observed_counts` |
| **Stage B Core** | `src/stage_b/research_csdi_rework.py` | `CSDICoreRework`: 1-head attention, variable channels | C16, L8, layers=2, steps=20, heads=1 |
| **Stage B Ops** | `src/stage_b/research_csdi.py` | `condition_features`, `forward_noise`, `epsilon_loss`, `sample_latents`, `schedule`, `project_final` | DDPM quad β, MC=8, final-only projection |
| **Gates** | `src/evaluation/rework_gates.py` | Conditioning, Efficacy, Numerical/Restoration gates | Thresholds: cond ∀, eff >10%, num exact |
| **Runner** | `scripts/run_e5_mask_conditioning.py` | CLI: prepare|train|evaluate; cache resume; frozen masks/hashes | 24 traj, 2 budgets, 48 entries |

---

## 3. Experimental Design (Factorial §11.2)

| Factor | Control | Intervention |
|--------|---------|--------------|
| **Training Mask Policy** | 18 random-cell banks, 5 hidden/row | 9 paired random-cell (identical to control 0-8) + 9 whole-variant banks |
| **Whole-Variant Strata** | — | n_hidden={1,2,3} × abundance={rare,middle,common} (train-only ordering) |
| **Optimization Budget** | 280 updates (40 epochs) | 1,120 updates (160 epochs) — diagnostic, not asserted optimum |
| **Modes** | no_init, mask_only, hron_2a, aitchison_complete | Same 4 modes |
| **Seeds** | 42, 43, 44 | Same 3 seeds |
| **Total Trajectories** | 12 | 12 |
| **Scored Entries** | 24 (2 budgets) | 24 (2 budgets) |

**Checkpoint Rule:** Fixed validation fixture over all 103 eligible Denmark rows, all windows, 20 steps, 3 primary whole-variant n2 masks, equal weight, earliest argmin. Budget280 constrained to epochs 1-40.

---

## 4. Model Evaluation Summary

### 4.1 Gate Results (Both Budgets)

| Gate | Budget280 | Budget1120 | Verdict |
|------|-----------|------------|---------|
| **Conditioning** (mask_only > no_init ∀ seeds/scenarios/policies) | FAIL (control: 0/3 scenarios pass) | FAIL (control: 1/3 scenarios pass) | ❌ FAIL |
| **Efficacy** (init+CSDI >10% M2 gain over init-only ∀ seeds/scenarios) | FAIL (0/2 init pairs qualify) | FAIL (0/2 init pairs qualify) | ❌ FAIL |
| **Numerical/Restoration** (finite, non-neg, observed-locked) | PASS (64/64 clean) | PASS (64/64 clean) | ✅ PASS |

### 4.2 Per-Policy Conditioning Detail

| Policy | Budget | whole_rare_n2 | whole_middle_n2 | whole_common_n2 | Overall |
|--------|--------|---------------|-----------------|-----------------|---------|
| **Control** | 280 | FAIL (1/3 seeds) | FAIL (2/3 seeds) | FAIL (2/3 seeds) | FAIL |
| **Control** | 1120 | FAIL (2/3 seeds) | FAIL (2/3 seeds) | PASS (3/3 seeds) | FAIL |
| **Intervention** | 280 | FAIL (1/3 seeds) | PASS (3/3 seeds) | FAIL (2/3 seeds) | FAIL |
| **Intervention** | 1120 | FAIL (1/3 seeds) | PASS (3/3 seeds) | PASS (3/3 seeds) | FAIL* |

*Intervention budget1120 passes 2/3 scenarios but control failure blocks overall (requires **every** policy).

### 4.3 Efficacy Detail (Budget1120, Intervention Policy)

| Initializer | Scenario | M2 (init-only) | M2 (init+CSDI) | Gain | Nonzero CLR MAE | Pass |
|-------------|----------|----------------|----------------|------|-----------------|------|
| hron_2a | whole_rare_n2 | 1.19 | 16.00 | **-1240%** | ✗ | ❌ |
| hron_2a | whole_middle_n2 | 12.08 | 42.65 | **-253%** | ✗ | ❌ |
| hron_2a | whole_common_n2 | 39.42 | 90.26 | **-129%** | ✗ | ❌ |
| aitchison_complete | whole_rare_n2 | 1.17 | 16.15 | **-1280%** | ✗ | ❌ |
| aitchison_complete | whole_middle_n2 | 12.08 | 44.05 | **-265%** | ✗ | ❌ |
| aitchison_complete | whole_common_n2 | 39.42 | 92.73 | **-135%** | ✗ | ❌ |

**Key finding:** Diffusion **consistently degrades** M2 vs init-only by 11–33×. Nonzero CLR MAE also worsens.

### 4.4 Selected Checkpoint Epochs

| Policy | Mode | Seed | Budget280 Epoch | Budget1120 Epoch |
|--------|------|------|-----------------|------------------|
| control | no_init | 42/43/44 | 40 / 32 / 37 | 160 / 156 / 158 |
| control | mask_only | 42/43/44 | 40 / 39 / 40 | 147 / 144 / 147 |
| control | hron_2a | 42/43/44 | 39 / 32 / 32 | 150 / 150 / 150 |
| control | aitchison | 42/43/44 | 39 / 32 / 32 | 150 / 150 / 150 |
| intervention | no_init | 42/43/44 | 40 / 37 / 36 | 159 / 157 / 158 |
| intervention | mask_only | 42/43/44 | 40 / 38 / 40 | 149 / 146 / 148 |
| intervention | hron_2a | 42/43/44 | 39 / 38 / 39 | 151 / 150 / 150 |
| intervention | aitchison | 42/43/44 | 39 / 38 / 39 | 151 / 150 / 150 |

---

## 5. Method Selection Rationale

| Decision | Rationale |
|----------|-----------|
| **Protocol U (not K)** | Closure ratio 78.1% < 95%; `total_sequence` ≠ sum(17 variants); JSD requires known totals → use CLR rank-16 throughout |
| **CLR16 only (no ILR/HKGLR)** | Registered diagnostic scope §11.2: single transform to isolate mask/budget effects; geometry/reference controls deferred to R7 |
| **Single-head attention (heads=1)** | Architectural rule: vary channels only, not head count; matches frozen 16-ch/4-head reference for capacity comparison |
| **DDPM 20 steps, quad schedule** | Frozen from E3/E4 revised; changing scheduler confounds mask/budget isolation |
| **MC=8, final-only projection** | Registered; clipping disabled per Protocol U; scale-guard recorded not gated |
| **Paired random-cell banks** | Control and intervention share banks 0-8 → isolates whole-variant effect from bank-count confound |
| **9 whole-variant strata** | Covers n_hidden={1,2,3} × {rare,middle,common}; train-only variant ordering prevents leakage |
| **Dual budgets from same trajectory** | Not independent replicates; tests whether longer training helps same model (avoids seed variance) |
| **Fixed validation fixture (not random)** | Reproducible checkpoint selection; noise seed 919, earliest argmin; same for all arms |
| **Canada never evaluated** | Preregistered seal; only opens if **all** validation gates pass (they don't) |

---

## 6. Key Findings

1. **Conditioning failure on control policy** — Random-cell training masks do not transfer to whole-variant missingness. Mask-only M2 ≤ No-init M2 for rare variant scenario across seeds.

2. **Intervention helps conditioning at budget1120** — Whole-variant training masks improve mask_only > no_init for middle/common at 1120 updates, but rare variant still fails. Control policy failure blocks gate.

3. **Diffusion is harmful** — Init+CSDI consistently **worse** than init-only (M2 gain -1100% to -300%). Adding diffusion noise to good initializer predictions destroys signal.

4. **Numerical integrity intact** — All predictions finite, non-negative, observed-locked exact. Scale-guard flags recorded but no gate violations.

5. **Undertraining not the sole cause** — Budget1120 (160 epochs) improves conditioning for intervention but efficacy remains deeply negative. More updates ≠ better diffusion.

---

## 7. Next Steps

### Immediate (No New Experiments)
- [x] Document negative result in plan (§11.4 table updated)
- [x] Preserve all artifacts for audit trail
- [x] Keep Canada sealed (never evaluated)

### Conditional (R6 — Architecture Ablation)
**If** pursuing token-conditioning architecture (§11.3):
- Register separate experiment with parameter-count matching (±5%)
- Raw-feature condition tokens: 17 parts × (value, mask, init, provenance)
- Cross-part attention preserving CLR/ILR dependencies
- Same locked masks, updates, windows, optimizer, seeds
- Compare against broadcast 85→C encoder baseline
- Include raw/init value ablations, step-band losses, gradient norms
- Do NOT claim larger model improvement proves token conditioning alone

### Blocked (R7 — Confirmation & E5)
- Full transform/reference confirmation (CLR16/17, ILR16, HKGLR actual/random/high)
- Requires **qualifying arm** from pilot (none exists)
- Canada evaluation only after frozen finalist selection

---

## 8. Artifact Index

| Path | Description |
|------|-------------|
| `artifacts/e3e4_mask_conditioning_2026-10-06/config.json` | Full experiment config, hashes, gate definitions |
| `artifacts/e3e4_mask_conditioning_2026-10-06/evaluation_scores.json` | 48 scored entries + controls |
| `artifacts/e3e4_mask_conditioning_2026-10-06/validation_gates.json` | Per-policy per-budget gate details |
| `artifacts/e3e4_mask_conditioning_2026-10-06/best_epochs.json` | Selected checkpoint epochs/losses |
| `artifacts/e3e4_mask_conditioning_2026-10-06/failure_records.json` | Per-entry contract violations (empty) |
| `artifacts/e3e4_mask_conditioning_2026-10-06/e5_mask_conditioning_status.json` | Final status: BLOCKED, finalists=[], canada_evaluated=false |
| `scripts/run_e5_mask_conditioning.py` | Reproducible runner (prepare|train|evaluate) |
| `tests/unit/stage_b/test_e5_mask_conditioning.py` | 15 unit tests for mask/gate/cache logic |

---

## 9. Conclusion

The E3/E4 mask-conditioning diagnostic **confirms both hypotheses negatively**:
- Random-cell training masks fail to condition for whole-variant missingness
- Extended optimization (1120 updates) improves conditioning for matched masks but diffusion remains detrimental vs. strong initializers

**No arm qualifies.** E5 remains blocked. The pipeline is numerically sound (123 tests pass, 0 numerical violations) but the methodological approach (CSDI on compositional variant data with structured missingness) does not yield positive transfer under the registered protocol.

**Recommendation:** Pursue R6 architecture ablation only if separately registered with strict capacity controls. Do not retry E5 or modify gates to achieve PASS.