# Implementation Checklist: Revised Research Plan

**Purpose:** Actionable steps to execute the revised research plan with proper gates and decision points.

## Execution status — 2026-10-05

Scope requested: first two tasks, **E0.1 and E0.2**. Read together with `e3_e4_review_revised_2026-09-30.md` and the revised plan; task completion does not imply a research gate passed.

- [x] **E0.1 executed and verified:** 516 complete rows, 489 positive-sum rows, 27 zero-sum rows. 403/516 = **78.10%** of all complete rows within [0.95, 1.05]; positive-only sensitivity 403/489 = **82.41%**. Closure gate FAIL; **Protocol U**, original JSD-complete disabled. Revised §1.1 requires >=95% of all complete rows; the checklist's positive-only K-weak/CAUTION does not establish known total mass.
- [x] **E0.2 executed and verified:** **150/150 locations (100%)** have exactly one missingness pattern; 86,050 missing cells. Gate PASS: **whole-variant primary**, random-cell secondary. External reasons for missingness remain unresolved.
- [ ] E0.3 and subsequent tasks: not executed in this request; existing artifacts remain preserved.

Evidence: `artifacts/e0_checklist_2026-10-05/` contains `data_manifest.json`, `closure_rows.csv`, `missingness_patterns.csv`, `missingness_summary.json`, `verification.json`, and `report.md`. Runners: `scripts/audit_composition_closure.py`, `scripts/audit_missingness_patterns.py`.

Input SHA-256 matches the frozen audit: `bbb1aacdf46ebef345ca818f086d5586f4aded45475b00b6ee9bd89fc94a90ab`. **E5 remains BLOCKED; finalists=[]** as recorded in the E3/E4 revised review. No E3 retraining or E4/E5 run performed.

Execution used the bundled Python runtime because `.venv-e3-revised` could not launch its configured base interpreter.

## Phase E0: Data Audit (BLOCKING — DO THIS FIRST)

### Task E0.1: Composition Closure Check ⚠️ BLOCKS ALL METHOD WORK

**Question:** Do the 17 variants form a closed composition?

**Implementation:**
```python
# File: scripts/audit_composition_closure.py

import pandas as pd
import numpy as np

data = pd.read_csv('data/covariants.csv')

# Get columns for 17 variants (verify these names)
variant_cols = [
    'Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta', 
    'Theta', 'Iota', 'Kappa', 'Lambda', 'Mu', 'Omicron', 
    'recombinant', 'S:677H', 'S:677P', 'other'  # CHECK EXACT NAMES
]

# Filter to rows with all 17 variants observed
complete_mask = data[variant_cols].notna().all(axis=1)
complete_rows = data[complete_mask].copy()

# Remove rows where sum is zero (composition undefined)
complete_rows['sum_17'] = complete_rows[variant_cols].sum(axis=1)
positive_complete = complete_rows[complete_rows['sum_17'] > 0].copy()

# Check closure
positive_complete['ratio'] = positive_complete['sum_17'] / positive_complete['total_sequence']

# Report
print(f"Complete rows (all 17 observed): {len(complete_rows)}")
print(f"Complete with positive sum: {len(positive_complete)}")
print(f"\nRatio sum(17 variants) / total_sequence:")
print(f"  Mean: {positive_complete['ratio'].mean():.6f}")
print(f"  Std:  {positive_complete['ratio'].std():.6f}")
print(f"  Min:  {positive_complete['ratio'].min():.6f}")
print(f"  Max:  {positive_complete['ratio'].max():.6f}")

# Decision criterion
in_tolerance = ((positive_complete['ratio'] >= 0.95) & 
                (positive_complete['ratio'] <= 1.05))
pct_in_tolerance = in_tolerance.mean() * 100

print(f"\nFraction in [0.95, 1.05]: {pct_in_tolerance:.1f}%")

# Decision
if pct_in_tolerance >= 95:
    decision = "Protocol K: Closed composition CONFIRMED"
    jsd_applicable = True
elif pct_in_tolerance >= 80:
    decision = "Protocol K-weak: Mostly closed (investigate discrepancies)"
    jsd_applicable = "with_caution"
else:
    decision = "Protocol U: Composition NOT closed"
    jsd_applicable = False

print(f"\n{'='*60}")
print(f"DECISION: {decision}")
print(f"JSD method applicable: {jsd_applicable}")
print(f"{'='*60}")

# Save manifest
manifest = {
    'date': '2026-09-30',
    'protocol': 'K' if jsd_applicable else 'U',
    'n_complete_rows': int(len(complete_rows)),
    'n_positive_complete': int(len(positive_complete)),
    'ratio_mean': float(positive_complete['ratio'].mean()),
    'ratio_std': float(positive_complete['ratio'].std()),
    'ratio_min': float(positive_complete['ratio'].min()),
    'ratio_max': float(positive_complete['ratio'].max()),
    'pct_in_tolerance': float(pct_in_tolerance),
    'decision': decision,
    'jsd_applicable': jsd_applicable
}

import json
with open('artifacts/data_manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)

print(f"\nManifest saved to artifacts/data_manifest.json")
```

**GATE:** 
- ✅ PASS if ≥95% in tolerance → Proceed with JSD methods
- ⚠️ CAUTION if 80-95% → Investigate discrepancies, may proceed with documentation
- ❌ FAIL if <80% → Drop JSD-complete, use Protocol U only

**Estimated time:** 2-4 hours (including data exploration)

---

### Task E0.2: Missingness Pattern Analysis

**Question:** Is missingness structured by location-variant or random?

**Implementation:**
```python
# File: scripts/audit_missingness_patterns.py

# For each location, compute:
# - Which variants are NEVER observed
# - Which variants are ALWAYS observed  
# - Whether pattern changes over time

location_patterns = []

for location in data['location'].unique():
    loc_data = data[data['location'] == location]
    
    # Missingness by variant
    missing_rate = loc_data[variant_cols].isna().mean()
    always_missing = missing_rate[missing_rate == 1.0].index.tolist()
    always_observed = missing_rate[missing_rate == 0.0].index.tolist()
    
    # Check temporal stability
    loc_data['pattern_hash'] = (
        loc_data[variant_cols].isna().astype(int)
        .apply(lambda x: ''.join(x.astype(str)), axis=1)
    )
    n_unique_patterns = loc_data['pattern_hash'].nunique()
    
    location_patterns.append({
        'location': location,
        'n_rows': len(loc_data),
        'always_missing': always_missing,
        'always_observed': always_observed,
        'n_unique_patterns': n_unique_patterns,
        'is_constant_pattern': n_unique_patterns == 1
    })

patterns_df = pd.DataFrame(location_patterns)

print("Missingness pattern stability:")
print(f"Locations with constant pattern: {patterns_df['is_constant_pattern'].sum()} / {len(patterns_df)}")
print(f"Mean unique patterns per location: {patterns_df['n_unique_patterns'].mean():.2f}")

# Save
patterns_df.to_csv('artifacts/missingness_patterns.csv', index=False)

# Decision
if patterns_df['is_constant_pattern'].mean() > 0.9:
    print("\n✅ CONFIRMED: Missingness is structured (location-variant level)")
    print("   → Whole-variant masking should be PRIMARY scenario")
    print("   → Random-cell masking is SECONDARY (controlled comparison)")
else:
    print("\n⚠️ Mixed patterns detected")
    print("   → Investigate further before choosing primary scenario")
```

**GATE:**
- ✅ PASS if >90% locations have constant pattern → Whole-variant primary
- ⚠️ INVESTIGATE if 50-90% constant
- ❌ RETHINK if <50% constant → May need different experimental design

**Estimated time:** 2-3 hours

---

### Task E0.3: Freeze Splits and Masks

**Generate and save:**
```python
# File: scripts/generate_splits_and_masks.py

import hashlib

# 5-fold splits by location (locations with complete data = test)
complete_locations = ['Canada', 'Denmark', 'Netherlands', 'United Kingdom', 'United States']

folds = []
for i, test_loc in enumerate(complete_locations):
    train_locs = [loc for loc in complete_locations if loc != test_loc]
    
    folds.append({
        'fold_id': i,
        'test_location': test_loc,
        'train_locations': train_locs,
        'test_rows': data[data['location'] == test_loc].index.tolist(),
        'train_rows': data[data['location'].isin(train_locs)].index.tolist()
    })

# Save splits
with open('artifacts/cv_splits.json', 'w') as f:
    json.dump(folds, f, indent=2)

# Generate evaluation masks (random-cell scenario)
# For each fold's test set, create masks at 10%, 30%, 50%
# Save with hash for reproducibility

masks = {}
for fold in folds:
    test_data = data.loc[fold['test_rows'], variant_cols]
    
    for severity in [0.1, 0.3, 0.5]:
        for seed in [42, 43, 44]:  # 3 mask replicates
            np.random.seed(seed)
            
            mask = test_data.notna().copy()  # Start with observed cells
            n_to_hide = int(mask.sum().sum() * severity)
            
            # Randomly select cells to hide (constrain: ≥3 observed remain per row)
            # ... masking logic ...
            
            mask_id = f"fold{fold['fold_id']}_r{severity}_seed{seed}"
            masks[mask_id] = {
                'fold': fold['fold_id'],
                'severity': severity,
                'seed': seed,
                'mask': mask.values.tolist(),  # or save as separate .npy
                'n_hidden': int((~mask).sum().sum()),
                'hash': hashlib.md5(mask.values.tobytes()).hexdigest()
            }

# Save masks
with open('artifacts/eval_masks_random_cell.json', 'w') as f:
    json.dump(masks, f, indent=2)

print(f"Generated {len(folds)} CV folds")
print(f"Generated {len(masks)} evaluation masks")
```

**GATE:**
- ✅ Splits and masks saved with hashes
- ✅ Each test fold has exactly 1 complete location
- ✅ Masks ensure ≥3 observed parts per row

**Estimated time:** 3-4 hours (including validation)

---

## Phase E1: Initializer Methods (After E0 passes)

### Task E1.1: Implement Hron Method 2a

**Status check:** Original plan says this already exists. Verify:

```bash
# Check if implementation exists
grep -r "def.*hron.*2a\|class.*Hron" src/
```

**If exists:** Audit against paper specification
- Distance on subcomposition CLR
- Flexible donors (must have observed ∪ {target})
- Robust scale adjustment (median ratio)
- Stable tie-breaking

**If missing:** Implement following original plan §5.1

**Tests needed (T02, T03 from original plan):**
```python
# tests/test_hron.py

def test_hron_golden_fixture():
    """Reproduce Hron paper example."""
    # From Hron Table 1, fixture 152.1
    # Expected: specific imputed values (within tolerance)
    pass

def test_hron_subcomposition_distance():
    """Distance computed only on overlapping observed parts."""
    pass

def test_hron_scale_adjustment():
    """Robust median scale factor applied correctly."""
    pass

def test_hron_donor_pool():
    """Only donors with observed ∪ {target} included."""
    pass

def test_hron_fallback():
    """Fewer than k donors handled gracefully."""
    pass
```

**GATE:**
- ✅ Golden fixture matches paper (tolerance 1e-4)
- ✅ Tests T02-T03 pass
- ✅ Fallback rate <5% on complete-location folds

**Estimated time:** 3-5 days if implementing from scratch, 1-2 days if auditing existing

---

### Task E1.2: Implement Aitchison-Complete Control

**Purpose:** Same distance as Hron, but restricted to complete donors

```python
# src/initialization/aitchison_complete.py

class AitchisonCompleteKNN:
    """
    KNN imputation using Aitchison distance, complete donors only.
    Control for comparing Hron-2a (flexible) vs complete-donor pool.
    """
    
    def __init__(self, k=8):
        self.k = k
        
    def fit(self, X_train, mask_train, row_ids):
        """Store complete donors (all D parts observed)."""
        self.complete_donors = X_train[mask_train.all(axis=1)]
        self.complete_donor_ids = row_ids[mask_train.all(axis=1)]
        return self
    
    def transform(self, X_query, mask_query):
        """Impute using k nearest complete donors."""
        # For each query with missing parts:
        #   1. Compute Aitchison distance to all complete donors
        #      using query's OBSERVED parts (subcomposition)
        #   2. Select k nearest
        #   3. Apply robust scale adjustment (median ratio)
        #   4. Impute missing parts
        pass
```

**Tests:**
```python
def test_aitchison_complete_vs_hron_same_donors():
    """When donors happen to be complete, Hron and Aitchison-complete match."""
    pass

def test_aitchison_complete_coverage():
    """Report fraction of queries with ≥k complete donors available."""
    pass
```

**GATE:**
- ✅ Implementation follows same logic as Hron except donor pool
- ✅ Coverage report shows sufficient complete donors (≥50% queries have k donors)

**Estimated time:** 2-3 days

---

### Task E1.3: Implement JSD-Complete (If Protocol K confirmed)

**Conditional on E0.1 result.**

```python
# src/initialization/jsd_complete.py

def jensen_shannon_divergence(p, q):
    """
    JSD with Tsagris 2026 convention: 2× standard definition.
    J_T(p,q) = Σ [p ln(2p/(p+q)) + q ln(2q/(p+q))]
    
    Zero contributions: 0·ln(0) = 0
    """
    # Validate against scipy.spatial.distance.jensenshannon
    # (note: scipy may use different convention, check carefully)
    pass

class JSDCompleteKNN:
    """Original Tsagris algorithm (requires Protocol K)."""
    
    def transform(self, X_query, mask_query, known_total):
        """
        1. Find k complete donors with smallest JSD on observed subcomp
        2. Take arithmetic mean of complete donor compositions
        3. Compute T_i = 1 - sum(observed_i) from known total
        4. Allocate T_i to missing parts proportional to mean
        """
        assert known_total is not None, "JSD-complete requires Protocol K"
        # Implementation per original plan §5.2
        pass
```

**Tests (T04, T05):**
```python
def test_jsd_properties():
    """Symmetry, J(p,p)=0, bounds [0, 2ln2], handles zeros."""
    pass

def test_tsagris_fixture():
    """Reproduce paper example: [0.2, NA, 0.3, 0.1, NA] → [0.2, 0.27, 0.3, 0.1, 0.13]."""
    pass

def test_jsd_mass_allocation():
    """Missing mass T_i allocated correctly, observed parts locked."""
    pass
```

**GATE:**
- ✅ Numerical parity with CompositionalNAimp R package (if available)
- ✅ Tsagris fixture matches (tolerance 1e-3)
- ✅ No negative allocations or invalid compositions

**Estimated time:** 3-4 days

---

### Task E1.4: Standalone Initializer Benchmark

**Before connecting to diffusion, evaluate initializers independently.**

```python
# scripts/benchmark_initializers.py

# For each fold, random-cell masks at severity=0.3:
#   1. Run Hron-2a
#   2. Run Aitchison-complete  
#   3. Run JSD-complete (if Protocol K)
#   4. Compute metrics: M2, CLR MAE (nonzero), JSD distance
#   5. Report donor coverage, fallback rate

# Output: artifacts/initializer_benchmark.csv
```

**Analysis questions:**
- Does Aitchison-complete vs Hron-2a show the donor flexibility effect?
- Does JSD-complete vs Aitchison-complete show the distance metric effect?
- Are error differences large enough to matter for downstream diffusion?

**GATE:**
- ✅ All methods produce valid outputs (no NaNs, finite, sum=1)
- ✅ Donor coverage documented
- ✅ At least one method shows feasibility (M2 < baseline Mean imputation)

**Estimated time:** 2-3 days

---

## Phase E2: Log-Ratio Transforms (After E1 passes)

### Task E2.1: Audit Existing CLR/ILR

**Check existing implementation:**
```bash
grep -r "def.*clr\|def.*ilr\|class.*CLR\|class.*ILR" src/
```

**Audit:**
- CLR: sum(z) = 0, inverse via softmax, handles scale invariance
- ILR: V orthonormal, V^T 1 = 0, forward/inverse consistent
- Positivity policy documented

**Tests (T09, T10):**
```python
def test_clr_sum_zero():
    assert abs(clr(x).sum()) < 1e-10

def test_clr_roundtrip():
    x_original = simplex(...)
    x_recovered = clr_inverse(clr(x_original))
    assert_allclose(x_recovered, x_original, atol=1e-8)

def test_ilr_algebra():
    V = get_ilr_basis(D=17)
    assert_allclose(V.T @ V, np.eye(16), atol=1e-10)
    assert_allclose(V.T @ np.ones(17), 0, atol=1e-10)

def test_ilr_aitchison_distance():
    """ILR distance = Aitchison distance."""
    d_ilr = np.linalg.norm(ilr(x) - ilr(y))
    d_aitchison = aitchison_distance(x, y)
    assert_allclose(d_ilr, d_aitchison, rtol=1e-6)
```

**GATE:**
- ✅ All algebra tests pass
- ✅ Roundtrip error <1e-8

**Estimated time:** 1-2 days

---

### Task E2.2: Implement HKGLR

```python
# src/core/log_ratio.py

class HKGLRTransform:
    """
    Housekeeping-Gene Log-Ratio (project adaptation).
    References: 5 variants with lowest missing rate in train.
    """
    
    def fit(self, X_train, mask_train, feature_names):
        """Select 5 references with lowest missing rate."""
        missing_rate = (~mask_train).mean(axis=0)
        self.reference_indices = np.argsort(missing_rate)[:5]
        self.reference_names = [feature_names[i] for i in self.reference_indices]
        return self
    
    def forward(self, X):
        """
        h_j = ln(x_j) - (1/5) * sum(ln(x_r) for r in references)
        Keeps all D=17 coordinates.
        """
        geom_mean_ref = np.exp(np.log(X[:, self.reference_indices]).mean(axis=1, keepdims=True))
        h = np.log(X) - np.log(geom_mean_ref)
        return h
    
    def inverse(self, h):
        """x = closure(exp(h))."""
        return softmax(h, axis=1)
    
    def to_clr(self, h):
        """HKGLR to CLR: h - mean(h) = CLR."""
        return h - h.mean(axis=1, keepdims=True)
```

**Tests (T11):**
```python
def test_hkglr_reference_sum_zero():
    """Mean of reference coordinates = 0."""
    h = hkglr.forward(x)
    assert_allclose(h[:, hkglr.reference_indices].mean(axis=1), 0, atol=1e-10)

def test_hkglr_to_clr():
    """h - mean(h) = CLR(x)."""
    h = hkglr.forward(x)
    clr_from_hkglr = h - h.mean(axis=1, keepdims=True)
    clr_direct = clr_transform.forward(x)
    assert_allclose(clr_from_hkglr, clr_direct, atol=1e-8)

def test_hkglr_roundtrip():
    x_recovered = hkglr.inverse(hkglr.forward(x))
    assert_allclose(x_recovered, x, atol=1e-8)
```

**GATE:**
- ✅ Algebra tests pass
- ✅ Equivalence to CLR verified

**Estimated time:** 2-3 days

---

### Task E2.3: Reference Selection Control

```python
# Implement 3 HKGLR variants:
# - HKGLR-actual: 5 lowest missing (current rule)
# - HKGLR-random: 5 random (3 seeds)
# - HKGLR-high: 5 highest missing

# Run pilot fold with all 3 variants + CLR + ILR
# Compare M2 and nonzero MAE

# If HKGLR-actual ≈ HKGLR-random:
#   → Selection rule doesn't matter (good, not biased)
# If HKGLR-actual > HKGLR-random:
#   → Low-missing references carry information (interpret carefully)
```

**GATE:**
- ✅ Reference selection effect quantified
- ✅ Results inform interpretation of Q4

**Estimated time:** 1-2 days

---

### Task E2.4: Dimensionality Control

```python
# Train 2 CLR models:
# - CLR-16: channels=16 (capacity-matched to ILR)
# - CLR-17: channels=17 (unconstrained)

# If CLR-17 >> CLR-16:
#   → Redundant coordinate hurts learning, dimensionality confounded
# If CLR-17 ≈ CLR-16:
#   → Architecture handles redundancy, fair comparison to ILR
```

**GATE:**
- ✅ Dimensionality effect documented

**Estimated time:** 1 day (piggybacks on E4 pilot)

---

## Phase E3: CSDI Core + Controls (After E2 passes)

### Task E3.1: CSDI Numerical Parity

```python
# Implement small synthetic fixture (D=3 or D=4, T=10)
# Compare forward noising, loss, reverse step with reference implementation
# Use same random seed and schedule

# Target: numerical agreement within 1e-5 for float32
```

**Tests (T15):**
```python
def test_csdi_forward_noising():
    """z_t = sqrt(ᾱ_t) z_0 + sqrt(1-ᾱ_t) ε matches reference."""
    pass

def test_csdi_loss_masking():
    """Loss only computed on hidden targets, not natural missing."""
    pass

def test_csdi_reverse_step():
    """One reverse step matches reference (same seed)."""
    pass
```

**GATE:**
- ✅ Numerical parity with reference CSDI on fixture
- ✅ Sampling produces finite outputs

**Estimated time:** 3-4 days

---

### Task E3.2: Leakage Tests

**Critical: Hidden targets must not leak into conditioning.**

```python
# tests/test_leakage.py (T12)

def test_poison_hidden_targets():
    """
    Replace hidden ground truth with NaN or extreme values.
    Conditioning, initializer, and fitted statistics must not change.
    Predictions may differ (they use labels), but conditioning path clean.
    """
    # Original data
    init1, cond1 = pipeline(X, mask, hidden_values=X_hidden_true)
    
    # Poisoned data
    X_poisoned = X.copy()
    X_poisoned[hidden_mask] = np.nan  # or 1e10
    init2, cond2 = pipeline(X_poisoned, mask, hidden_values=None)
    
    assert_allclose(cond1, cond2, atol=1e-10, 
                    err_msg="Conditioning depends on hidden truth → LEAKAGE")
```

**GATE:**
- ✅ T12-T14 leakage tests pass
- ✅ No information from hidden targets in condition encoder

**Estimated time:** 2-3 days

---

### Task E3.3: No-Init and Mask-Only Controls

```python
# Variant 1: No-Init
# Condition on: raw visible values + mask + time metadata
# No initializer estimates

# Variant 2: Mask-Only  
# Condition on: mask pattern + time metadata ONLY
# No observed values, no initializer
# Tests if model has memorized dataset statistics

# Expected: No-Init > Mask-Only >> Random
# If Mask-Only ≈ No-Init, model is memorizing, not conditioning
```

**GATE:**
- ✅ No-Init >> Mask-Only (validates conditioning is used)
- ✅ Both produce finite, valid outputs

**Estimated time:** 2-3 days

---

## Phase E4: Pilot Comparison (After E3 passes)

### Task E4.1: Run Pilot Grid

**Configuration:**
- 1 fold (e.g., fold 0, test=United States)
- 1 scenario (random-cell, severity=0.3)
- 1 mask seed
- 1 model seed

**Methods:**
```
Initializers: {Hron-2a, Aitchison-complete, [JSD-complete if K], None}
Transforms: {CLR-16, CLR-17, ILR-16, HKGLR-16-actual}
Baselines: {Mean, LOCF, Linear}

Grid: 3-4 init × 4 transform ≈ 12-16 diffusion methods
      + 3 no-init diffusion  
      + 4 init-only (no diffusion)
      + 3 baselines
Total: ~22-26 methods
```

**Estimated runtime:** 
- If current pilot (696 cells, 20 epochs, 20 steps) = 157 seconds
- ~26 methods × 180 seconds ≈ 1.2 hours

**GATE:**
- ✅ At least one diffusion method beats Init-only by >10% on M2
- ✅ Init+diffusion beats No-init (validates initialization helps)
- ✅ No-init beats Mask-only (validates conditioning is used)
- ✅ Choose ≤3 finalists for full CV based on validation M2

**Estimated time:** 2-3 days (including analysis)

---

## Phase E5: Full Cross-Validation (Finalists Only)

### Task E5.1: Run Full Grid for Finalists

**Configuration:**
- 5 folds (5 complete locations)
- 2 scenarios: random-cell (3 severities) + whole-variant (3 variant-sets)
- 3 model seeds per configuration

**Estimated grid:**
```
3 finalists × 5 folds × (3 severities + 3 variant-sets) × 3 seeds
= 3 × 5 × 6 × 3 = 270 runs
```

**Computational cost:**
- If each run ≈ 5 minutes (larger than pilot due to different folds)
- 270 × 5 min = 22.5 hours on single GPU
- Parallelize across 5 GPUs → ~4.5 hours

**GATE:**
- ✅ All runs complete without silent failures
- ✅ Results stable across seeds (CV <20%)
- ✅ Sign test shows consistent improvement (≥4/5 locations favor winner)

**Estimated time:** 5-8 days (including reruns, analysis, reporting)

---

## Decision Points and Off-Ramps

### After E0: Protocol K vs U

**If Protocol K confirmed (✅):**
→ Proceed with full plan including JSD methods

**If Protocol U (❌):**
→ Drop JSD-complete
→ Focus on Hron vs Aitchison-complete, LR comparisons
→ Adjust timeline: -3 days

### After E1: Donor Coverage

**If complete-donor coverage >70% (✅):**
→ Proceed with Aitchison-complete and JSD-complete

**If coverage 40-70% (⚠️):**
→ Report conditional metrics (stratify by donor availability)
→ Acknowledge limitation

**If coverage <40% (❌):**
→ Drop complete-donor methods
→ Only compare Hron-2a with LR transforms

### After E3: Conditioning Validation

**If No-init >> Mask-only (✅):**
→ Conditioning is working, proceed

**If No-init ≈ Mask-only (❌):**
→ Model is memorizing dataset
→ Redesign conditioning or reduce model capacity
→ Do not proceed to E4 until fixed

### After E4: Diffusion Feasibility

**If best diffusion > Init-only + 10% (✅):**
→ Proceed to full CV with top 3 finalists

**If best diffusion > Init-only + 5% (⚠️):**
→ Proceed but acknowledge marginal gains
→ May not justify computational cost

**If best diffusion < Init-only (❌):**
→ Diffusion is not helping
→ Report initializer-only results
→ Investigate: data size, architecture, conditioning design

---

## Timeline Summary

| Phase | Tasks | Time | Blocking | Cumulative |
|---|---|---|---|---|
| **E0** | Data audit, Protocol K/U, splits | 1-2 days | YES | 2 days |
| **E1** | Hron, Aitchison-complete, [JSD], benchmark | 6-10 days | YES | 12 days |
| **E2** | CLR/ILR audit, HKGLR, controls | 4-6 days | YES | 18 days |
| **E3** | CSDI core, leakage tests, controls | 5-7 days | YES | 25 days |
| **E4** | Pilot comparison, finalist selection | 2-3 days | Gate | 28 days |
| **E5** | Full CV (3 finalists × 5 folds × 6 configs × 3 seeds) | 5-8 days | - | 36 days |
| **E6** | Reporting, plots, interpretation | 2-3 days | - | 39 days |

**Total:** 25-39 research days (depending on off-ramps and parallelization)

**Critical path:** E0 → E1 → E2 → E3 → E4 gate → E5

**Parallelizable:** Within E5, runs can parallelize across GPUs/folds

---

## What to Do Right Now

### Immediate Priority (Next 2 Hours)

1. **Run E0.1: Composition closure check**
   ```bash
   python scripts/audit_composition_closure.py
   ```

2. **Read the decision from `artifacts/data_manifest.json`**
   - Protocol K → Can implement JSD
   - Protocol U → Skip JSD, adjust plan

3. **Run E0.2: Missingness patterns**
   ```bash
   python scripts/audit_missingness_patterns.py
   ```

4. **Confirm evaluation scenarios**
   - If patterns are constant → Whole-variant primary
   - If mixed → Reconsider

### This Week

1. Complete E0 (audit, splits, masks)
2. Start E1 (audit existing Hron, implement Aitchison-complete)
3. Write unit tests for initializers

### Next Two Weeks

1. Complete E1 (initializer benchmark)
2. Complete E2 (LR transforms + controls)
3. Start E3 (CSDI core)

### Week 3-4

1. Complete E3 (leakage tests, controls)
2. Run E4 pilot
3. Choose finalists

### Week 5-6

1. Run E5 full CV
2. Analyze results
3. Write report

---

## Success Criteria

**Minimum viable research output:**
- ✅ Protocol K/U determined with evidence
- ✅ Hron-2a benchmarked with donor diagnostics
- ✅ At least one LR transform (CLR) working end-to-end
- ✅ Diffusion vs initialization compared fairly
- ✅ Statistical limitations acknowledged (n=5)

**Full success:**
- ✅ All above
- ✅ JSD vs Aitchison comparison (if Protocol K)
- ✅ CLR vs ILR vs HKGLR with dimensionality controls
- ✅ Reference selection bias quantified
- ✅ Whole-variant and random-cell scenarios both evaluated
- ✅ Consistent improvement demonstrated (sign test)

**Exceptional output:**
- ✅ All above
- ✅ Alpha-Fréchet sensitivity
- ✅ Iterative refinement tested (challenges single-pass constraint)
- ✅ Open-source reproducible artifacts
- ✅ Clear recommendations for practitioners

---

## Red Flags to Watch For

🚩 **Leakage:** If poisoned-target test fails, stop and fix before continuing

🚩 **Memorization:** If Mask-only ≈ No-init, model is not using conditioning

🚩 **Divergence:** If any method produces NaN/Inf, debug before proceeding

🚩 **Confounds:** If dimensionality control shows large effect, interpretation changes

🚩 **Coverage failure:** If <40% queries have k donors, method not applicable

🚩 **Instability:** If results flip across seeds, need more seeds or different selection

🚩 **Statistical overreach:** If draft claims "p<0.05" with n=5, correct immediately

---

## Final Checklist Before Starting

- [ ] Read original plan thoroughly
- [ ] Read revised plan thoroughly  
- [ ] Read revision summary
- [ ] Understand blocking nature of E0
- [ ] Understand gates at each phase
- [ ] Have data file path confirmed
- [ ] Have artifact directory created
- [ ] Have compute resources allocated (GPU for E4-E5)
- [ ] Have backup plan if Protocol K fails
- [ ] Have timeline approved by collaborators

**If all boxes checked → Execute E0.1 now.**
