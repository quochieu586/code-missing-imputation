# Baseline Selection Report

## Selection Rule (Section 6.3)

1. Discard methods violating invariants.
2. Rank by mean JSD of empirical-pattern and time-block splits.
3. Tie within tolerance -> prefer simpler/faster method.
4. Adaptive never selected if worse on time-block.

## Results

| Method | JSD empirical | JSD time-block | JSD mean | Runtime (s) | Params |
|--------|--------------|----------------|----------|-------------|--------|
| jsd_knn | 0.045187 | 0.011538 | 0.028363 | 0.7 | {'k': 7, 'alpha': 1.0} |
| jsd_alpha_knn | 0.045104 | 0.010266 | 0.027685 | 6.1 | {'k': 7, 'alpha': 1.0} |
| adaptive_jsd_alpha_knn | 0.044519 | 0.011538 | 0.028028 | 283.8 | {'global_k': 7, 'global_alpha': 1.0, 'min_pattern_support': 30} |

## Champion: `jsd_knn`

- Params: {'k': 7, 'alpha': 1.0}
- Invariants OK: True
- Closure valid: True (violations: 0)
- Observed cells unchanged: True
- Seeds: [42, 123]
