# Research Plan Revision Summary

**Date:** 2026-09-30  
**Original:** `research_plan_knn_jsd_clr_ilr_hkglr_csdi_2026-09-30.md`  
**Revised:** `research_plan_revised_2026-09-30.md`

## Critical Issues Identified

### 1. Identifiability Crisis (Blocking)

**Issue:** JSD method requires knowing that observed variants sum to a valid total (Protocol K), but data audit shows uncertainty about whether 17 variants actually sum to `total_sequence`.

**Original approach:** Defer resolution, define two protocols (K and U), proceed with experiments.

**Problem:** Running JSD experiments before confirming Protocol K wastes resources on potentially invalid mathematics. If the 17 variants don't form a closed composition, JSD's mass allocation step is undefined.

**Revision:** Make E0 (data audit) blocking. Resolve composition closure before implementing any methods. If Protocol K fails, drop original JSD or redesign as a new method with scale estimation.

---

### 2. Confounded Baseline

**Issue:** Hron uses flexible donors (must have query's observed parts + target), while JSD uses complete donors (all D parts observed). Comparison conflates distance metric with donor eligibility.

**Original approach:** Acknowledge difference in §5.4, propose secondary comparison with same donor pool, but don't prioritize it.

**Problem:** Cannot answer "Is JSD distance better?" when donor pools differ. JSD may win because it selects from higher-quality donors, not because Jensen-Shannon divergence outperforms Aitchison distance.

**Revision:** 
- **Primary comparison:** Hron-2a vs Aitchison-complete (both use Aitchison, different donor pools) → tests donor flexibility
- **Primary comparison:** JSD-complete vs Aitchison-complete (both use complete donors, different distances) → tests distance metric
- Original Hron vs JSD comparison becomes secondary, interpreted as joint effect

---

### 3. Statistical Power Failure

**Issue:** Only 5 locations have complete data. 5-fold CV means 1 location per test fold (n=1 for location-level inference).

**Original approach (§10.5):** Acknowledge "bootstrap must be labeled exploratory," but still plan to report means, standard deviations, and comparisons.

**Problem:** 
- Cannot compute valid standard errors across 5 numbers
- Cannot do t-tests or claim "statistically significant"
- Presenting SD without acknowledging n=1 misleads readers about precision

**Revision:**
- Report point estimates per location (show all 5)
- Report mean and range (descriptive only)
- Use sign test for consistency (non-parametric, valid for n=5)
- Remove all p-values and confidence intervals for cross-location inference
- Within-location inference (e.g., 75 rows × 10 masks in US test fold) remains valid but generalizes only to that location

---

### 4. Missing Controls

**4a. Dimensionality confound**

**Issue:** CLR has D=17 coordinates (rank 16), ILR has D-1=16 (full rank), HKGLR has D=17 (rank 16). Original plan uses channels=16 for all, but doesn't test whether redundant coordinates affect learning.

**Revision:** Add CLR with channels=17 (unconstrained capacity) as control. If CLR-17 ≠ CLR-16, dimensionality is confounded with geometry.

**4b. Memorization control**

**Issue:** Diffusion model might learn dataset-level statistics (e.g., "Omicron is usually 40%") rather than using conditioning information.

**Revision:** Add "Mask-only" baseline: CSDI conditioned only on mask pattern and time, no observed values. If Mask-only ≈ Full-conditioning, model has memorized dataset and isn't truly imputing from observations.

**4c. Reference selection bias**

**Issue:** HKGLR uses "5 variants with lowest missing rate." Low-missing may correlate with high abundance, temporal stability, or sequencing bias—biological/technical confounds, not housekeeping properties.

**Revision:** Compare actual 5-lowest vs random 5 vs 5-highest missing rates. If random ≈ actual, selection rule doesn't matter (good). If random < actual, low-missing carries information content that must be interpreted carefully, not claimed as a generic property.

---

### 5. Evaluation Mismatch

**Issue:** Data structure shows whole-variant missingness per location (each location has one fixed pattern across time). But primary evaluation is random-cell masking, which tests MCAR assumption.

**Original approach:** Define three scenarios including whole-variant, but structure suggests random-cell is primary.

**Problem:** 
- Random-cell masking is unrealistic for this data (variants don't go randomly missing within a location)
- Linear/LOCF baselines are strong for cell-level interpolation but cannot handle whole-variant missingness (no same-variant points within location)
- Averaging random-cell and whole-variant results mixes two different phenomena

**Revision:**
- **Primary scenario:** Whole-variant missingness (matched to data structure)
- **Secondary scenario:** Random-cell masking (controlled comparison with clean ground truth)
- Report separately, do not average
- Drop contiguous time-gap scenario (data has variant-level missingness, not temporal gaps)

---

### 6. Unjustified Design Constraint

**Issue:** Original §1.4 mandates "one outer diffusion loop" (no iterative refinement) as a design decision.

**Original justification:** Contrasts with iterative imputation methods, keeps computational cost bounded.

**Problem:** This is a constraint, not an empirical finding. Iterative refinement might improve results but is excluded a priori without testing.

**Revision:** Keep single-pass as default (matches computational constraints), but move iterative refinement to Phase E6 sensitivity analysis. If it shows large gains, the constraint should be revisited. Report as a limitation if not tested.

---

### 7. Pseudo-count Confound

**Issue:** JSD claims native zero-handling advantage (doesn't require pseudo-counts). But all methods apply upstream positivity policy before log-ratio transforms.

**Original approach:** Acknowledge in §4.1 that nonnegative data is kept separate from positive data, but still compare "native zero" JSD with pseudo-counted LR methods.

**Problem:** If pseudo-counting happens before initialization, JSD sees the same modified data as Hron. The "native zero" advantage is only realized if:
1. JSD runs on truly nonnegative data (Protocol K, no pseudo-counts)
2. LR methods require pseudo-counts on same data
3. Comparison uses both nonnegative (for JSD) and positive-forced (for LR) views

If all methods see pseudo-counted data, the zero-handling claim is confounded with implementation choices.

**Revision:** 
- Define two data pipelines: 
  - Pipeline A (nonnegative): JSD + native-zero metrics
  - Pipeline B (positive): Hron/LR + pseudo-count metrics
- Report where pseudo-counts are applied and to how many cells
- Ablate pseudo-count policy in Phase E6 (e.g., count addition vs zero replacement)
- Do not claim "JSD preserves zeros" if comparison uses same pseudo-counted inputs

---

## Summary Table: Original vs Revised

| Issue | Original Plan | Revised Plan | Impact |
|---|---|---|---|
| **Protocol K/U** | Defer decision, run both | Blocking audit before methods | Prevents invalid JSD experiments |
| **Donor pool** | Acknowledge difference | Factorial design isolates factors | Unconfounds distance from eligibility |
| **n=5 locations** | Report SD, label bootstrap "exploratory" | Remove inferential stats, use sign tests | Honest about statistical power |
| **Dimensionality** | Channels=16 for all | Add CLR-17 control | Tests capacity confound |
| **Memorization** | Not tested | Add Mask-only baseline | Validates conditioning |
| **HKGLR references** | 5-lowest missing rule | Add random/high-missing controls | Tests selection bias |
| **Scenarios** | Three scenarios, implied equal weight | Primary=whole-variant, secondary=random-cell | Matches data structure |
| **Single-pass** | Mandated design | Sensitivity analysis for iterative | Tests constraint empirically |
| **Zero-handling** | JSD native advantage assumed | Separate pipelines, ablate policy | Unconfounds implementation |

---

## What the Revision Preserves

The original plan has many strengths that are retained:

1. **Location-based splits** prevent temporal/spatial leakage
2. **Frozen evaluation masks** ensure fair comparison
3. **Observed value restoration** maintains compositional constraints
4. **Comprehensive diagnostics** (donor coverage, fallback rates, failures)
5. **Separate positivity policies** for different data views
6. **Mathematical rigor** in LR definitions and inverse operations
7. **Leakage tests** (T12-T14) ensure hidden targets don't leak into conditioning

The revision reorganizes experiments to isolate causal factors, not to discard careful design.

---

## Practical Implications

### Immediate Actions (Blocking)

1. **Run E0-extended data audit:**
   - Check if `sum(17 counts) ≈ total_sequence` for complete rows
   - If YES (>95% match): Protocol K confirmed, proceed with JSD
   - If NO: Protocol U, drop original JSD or redesign with scale estimation

2. **Do not implement initializers** until Protocol K/U is resolved

3. **Do not implement diffusion** until initializer controls (Aitchison-complete) are ready

### Experimental Sequence

```
E0 (BLOCKING) → Audit data semantics
    ↓
E1 → Hron + Aitchison-complete + [JSD if Protocol K]
    ↓
E2 → CLR-16, CLR-17, ILR-16, HKGLR-16 (actual/random/high)
    ↓
E3 → CSDI core + No-init + Mask-only controls
    ↓
E4 → Pilot (1 fold, finalists only)
    ↓ (gates: feasibility, No-init beats Mask-only, diffusion beats Init-only)
E5 → Full CV (3 finalists × 5 folds × 2 scenarios × 3 seeds)
    ↓
E6 → Sensitivity (alpha-Fréchet, pseudo-count, iterative refinement)
```

Each phase has a **gate** (pass/fail criteria). Do not proceed to next phase if gate fails.

### Resource Implications

**Original estimate:** 18-26 research days

**Revised estimate:** 22-32 research days
- E0-extended adds 1-2 days (but prevents wasted E1 work if Protocol K fails)
- Factorial design adds 2-3 days (controls must be implemented)
- Reference selection control adds 1-2 days (3× HKGLR variants)
- Whole-variant scenario adds 2-3 days (different masking logic)
- Statistical revision saves time (no bootstrap CI, simpler reporting)

**Net:** Comparable effort, much higher validity.

---

## Recommended Next Steps

1. **Present this revision to collaborators** and confirm:
   - Is Protocol K/U resolution feasible? (Requires data provenance/metadata)
   - Is n=5 limitation acceptable? (May need more locations or acknowledge as pilot)
   - Are factorial controls worth the added cost? (Yes, if causal claims matter)

2. **Run E0-extended immediately:**
   ```python
   # Check composition closure
   complete_rows = data[data.notna().all(axis=1)]
   ratio = complete_rows[variant_cols].sum(axis=1) / complete_rows['total_sequence']
   print(f"Mean ratio: {ratio.mean():.4f}, SD: {ratio.std():.4f}")
   print(f"Fraction in [0.95, 1.05]: {((ratio > 0.95) & (ratio < 1.05)).mean():.2%}")
   
   # If >95% match → Protocol K confirmed
   # If <80% match → Protocol U, JSD requires redesign
   ```

3. **Freeze the decision** and document in `data_manifest.json`:
   ```json
   {
     "protocol": "K" or "U",
     "composition_closure_check": {
       "mean_ratio": 0.98,
       "in_tolerance": "96.3%",
       "decision_rule": "≥95% match",
       "jsd_applicable": true
     }
   }
   ```

4. **Update implementation plan** based on E0 result:
   - If Protocol K: Implement all methods as revised
   - If Protocol U: Drop JSD-complete, focus on Hron/Aitchison/LR comparisons

---

## Conclusion

The original plan demonstrates strong compositional data expertise and careful experimental design. However, it contains methodological issues that would compromise interpretation:

- **Identifiability** deferred when it should block
- **Confounds** between multiple factors make causal claims ambiguous  
- **Statistical power** overstated for n=5
- **Missing controls** leave alternative explanations unexamined

The revised plan reorganizes experiments to isolate factors, adds essential controls, and acknowledges limitations honestly. Estimated effort is comparable, but validity is substantially higher.

**Core message:** Run the data audit first. Everything else depends on whether the 17 variants form a closed composition.
