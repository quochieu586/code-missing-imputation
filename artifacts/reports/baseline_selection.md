# Baseline Selection Report

## Selection Rule (Section 6.3 / Nhánh A)

1. Discard methods violating invariants.
2. Rank by mean MSE on proportions at artificial masked observed cells (primary).
3. Tie within tolerance -> prefer simpler/faster method (JSD as tie-breaker).
5. Adaptive never selected if worse on time-block MSE.

## Results

| Method | MSE empirical | MSE time-block | MSE mean | JSD empirical | JSD time-block | JSD mean | Runtime (s) | Params |
|--------|--------------|----------------|----------|---------------|----------------|----------|-------------|--------|
| jsd_knn | 0.002896 | 0.015364 | 0.009130 | 0.020057 | 0.052782 | 0.036420 | 1.0 | {'k': 15, 'alpha': 1.0} |
| jsd_alpha_knn | 0.002850 | 0.015899 | 0.009374 | 0.018185 | 0.053017 | 0.035601 | 4.9 | {'k': 15, 'alpha': 0.5} |
| adaptive_jsd_alpha_knn | 0.003510 | 0.015899 | 0.009704 | 0.021784 | 0.053017 | 0.037400 | 429.4 | {'global_k': 15, 'global_alpha': 0.5, 'min_pattern_support': 30} |

## Champion: `jsd_knn`

- Params: {'k': 15, 'alpha': 1.0}
- Invariants OK: True
- Closure valid: True (violations: 0)
- Observed cells unchanged: True
- Seeds: [42, 123, 456, 789, 999]
