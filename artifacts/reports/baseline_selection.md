# Baseline Selection Report

## Selection Rule (Section 6.3 / Nhánh A)

1. Discard methods violating invariants.
2. Rank by mean MSE on proportions at artificial masked observed cells (primary).
3. Tie within tolerance -> prefer simpler/faster method (JSD as tie-breaker).
5. Adaptive never selected if worse on time-block MSE.

## Results

| Method | MSE empirical | MSE time-block | MSE mean | JSD empirical | JSD time-block | JSD mean | Runtime (s) | Params |
|--------|--------------|----------------|----------|---------------|----------------|----------|-------------|--------|
| jsd_knn | 0.022568 | 0.081693 | 0.052131 | 0.060553 | 0.050122 | 0.055337 | 1.6 | {'k': 15, 'alpha': 1.0} |
| jsd_alpha_knn | 0.067384 | 0.066614 | 0.066999 | 0.067384 | 0.039338 | 0.053361 | 6.4 | {'k': 5, 'alpha': 10.0} |
| adaptive_jsd_alpha_knn | 0.027062 | 0.042782 | 0.034922 | 0.060367 | 0.023364 | 0.041866 | 779.4 | {'global_k': 5, 'global_alpha': 10.0, 'min_pattern_support': 30} |

## Champion: `adaptive_jsd_alpha_knn`

- Params: {'global_k': 5, 'global_alpha': 10.0, 'min_pattern_support': 30}
- Invariants OK: True
- Closure valid: True (violations: 0)
- Observed cells unchanged: True
- Seeds: [42, 123, 456, 789, 999]
