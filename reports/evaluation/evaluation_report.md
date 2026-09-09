# Evaluation Report — Compositional Longitudinal Imputation v2

**Generated:** 2026-09-09 21:14:15

---

## 1. Metric Comparison Across Pipeline Stages

| Stage | MSE (prop) | RMSE (prop) | MAE (prop) | JSD | Wasserstein (mean) |
|-------|-----------|-------------|-----------|-----|-------------------|
| tsagris | 0.0038083369961402595 | 0.06171172494866968 | 0.00978015825036453 | 0.057622257491697214 | 0.009780 |
| fused | 0.0038044168071796917 | 0.06167995466259433 | 0.00977250739772065 | 0.05757718057271585 | 0.009773 |
| gated_gan | 0.005283496365767198 | 0.07268766309193878 | 0.00977250739772065 | 0.05757718057271585 | 0.009773 |
| ungated_gan | 0.0039737290054697475 | 0.06303752061645308 | 0.00977250739772065 | 0.05757718057271585 | 0.009773 |

![Metric Comparison](reports\evaluation\plots\metric_comparison.png)

## 2. Distribution Fidelity (JSD by Variant)

| Variant | JSD |
|---------|-----|
| recombinant | 0.016676 |
| 20A | 0.032569 |
| 20B | 0.167834 |
| 20C | 0.275666 |
| 20E | 0.034616 |
| Beta | 0.021841 |
| Alpha | 0.000223 |
| Gamma | 0.051947 |
| Delta | 0.000000 |
| Kappa | 0.275734 |
| Epsilon | 0.181498 |
| Eta | 0.089467 |
| Iota | 0.177422 |
| Lambda | 0.122819 |
| Mu | 0.100398 |
| Omicron | 0.000000 |
| S:677 | 0.298812 |

![JSD by Variant](reports\evaluation\plots\jsd_by_variant.png)

## 3. Wasserstein Distance by Variant

| Variant | Wasserstein |
|---------|-------------|
| recombinant | 0.011715 |
| 20A | 0.019538 |
| 20B | 0.024146 |
| 20C | 0.023966 |
| 20E | 0.010440 |
| Beta | 0.003521 |
| Alpha | 0.000429 |
| Gamma | 0.009912 |
| Delta | 0.000000 |
| Kappa | 0.009646 |
| Epsilon | 0.010493 |
| Eta | 0.006570 |
| Iota | 0.005960 |
| Lambda | 0.005105 |
| Mu | 0.007095 |
| Omicron | 0.000000 |
| S:677 | 0.017595 |

![Wasserstein by Variant](reports\evaluation\plots\wasserstein_by_variant.png)

## 4. Temporal Fidelity

| Metric | Value |
|--------|-------|
| tsagris_roughness_true | 0.009189 |
| tsagris_roughness_imputed | 0.015227 |
| tsagris_lag1_true | 0.009189 |
| tsagris_lag1_imputed | 0.015227 |
| tsagris_tw_jsd_mean | 0.040030 |
| fused_roughness_true | 0.009189 |
| fused_roughness_imputed | 0.015217 |
| fused_lag1_true | 0.009189 |
| fused_lag1_imputed | 0.015217 |
| fused_tw_jsd_mean | 0.039999 |
| gated_gan_roughness_true | 0.009189 |
| gated_gan_roughness_imputed | 0.015086 |
| gated_gan_lag1_true | 0.009189 |
| gated_gan_lag1_imputed | 0.015086 |
| gated_gan_tw_jsd_mean | 0.039999 |
| ungated_gan_roughness_true | 0.009189 |
| ungated_gan_roughness_imputed | 0.015339 |
| ungated_gan_lag1_true | 0.009189 |
| ungated_gan_lag1_imputed | 0.015339 |
| ungated_gan_tw_jsd_mean | 0.039999 |

![Temporal Fidelity](reports\evaluation\plots\temporal_fidelity.png)

## 6. Compute Time

| Stage | Time (s) |
|-------|----------|
| tsagris | 4.2 |
| occurrence_gate | 0.3 |
| fused | 4.3 |
| gated_gan | 4.4 |
| ungated_gan | 4.1 |
| seed_stability | 1.4 |

![Compute Time](reports\evaluation\plots\compute_time.png)

## 7. Seed Variance

| Metric | Mean | Std | CV |
|--------|------|-----|-----|
| wasserstein_Gamma | 0.013843819656136716 | 0.0 | 0.0 |
| jsd_Omicron | 0.0 | 0.0 | inf |
| wasserstein_20C | 0.008742583872742066 | 0.0 | 0.0 |
| jsd_20C | 0.0674499506494951 | 0.0 | 0.0 |
| wasserstein_20B | 0.026082818915444006 | 0.0 | 0.0 |
| wasserstein_20A | 0.02106906889158655 | 0.0 | 0.0 |
| jsd_Lambda | 0.11433572591432528 | 0.0 | 0.0 |
| wasserstein_mean | 0.009772507397720651 | 0.0 | 0.0 |
| zero_precision | 1.0 | 0.0 | 0.0 |
| rmse_prop | 0.07268766309193878 | 0.0 | 0.0 |
| jsd_20E | 0.026122152639388704 | 0.0 | 0.0 |
| wasserstein_Mu | 0.01131423894476292 | 1.734723475976807e-18 | 1.5332215312456058e-16 |
| wasserstein_recombinant | 0.02278598612461034 | 3.469446951953614e-18 | 1.522623130278477e-16 |
| wasserstein_Kappa | 0.0016203241539912963 | 0.0 | 0.0 |
| zero_recall | 0.9556109585971203 | 0.0 | 0.0 |
| zero_f1 | 0.9773017014413118 | 0.0 | 0.0 |
| wasserstein_Alpha | 0.0010932264705441454 | 0.0 | 0.0 |
| jsd_recombinant | 0.03420851167498291 | 0.0 | 0.0 |
| jsd_Iota | 0.10701080417894378 | 1.3877787807814457e-17 | 1.296858566225517e-16 |
| mae_prop | 0.00977250739772065 | 0.0 | 0.0 |
| wasserstein_Epsilon | 0.012550163382092345 | 0.0 | 0.0 |
| jsd_Kappa | 0.05166461590816155 | 6.938893903907228e-18 | 1.3430650324085889e-16 |
| mse_prop | 0.005283496365767197 | 8.673617379884035e-19 | 1.6416434836753353e-16 |
| wasserstein_20E | 0.01152575585059997 | 1.734723475976807e-18 | 1.5050843506168028e-16 |
| mae_count | 1.1034993725019782 | 0.0 | 0.0 |
| wasserstein_Lambda | 0.007673953700669946 | 0.0 | 0.0 |
| jsd_Epsilon | 0.15974102337832063 | 0.0 | 0.0 |
| wasserstein_Omicron | 0.0 | 0.0 | inf |
| jsd | 0.05757718057271585 | 0.0 | 0.0 |
| jsd_Alpha | 0.00023812506293564807 | 0.0 | 0.0 |
| jsd_Gamma | 0.04671544758804305 | 0.0 | 0.0 |
| jsd_Beta | 0.018139134804413738 | 0.0 | 0.0 |
| wasserstein_Delta | 0.0 | 0.0 | inf |
| jsd_20B | 0.13605707545815987 | 0.0 | 0.0 |
| jsd_Eta | 0.07815801753725415 | 0.0 | 0.0 |
| jsd_S:677 | 0.13837297171393304 | 0.0 | 0.0 |
| wasserstein_Eta | 0.008464946427457624 | 0.0 | 0.0 |
| jsd_Delta | 0.0 | 0.0 | inf |
| jsd_20A | 0.022981986035330127 | 3.469446951953614e-18 | 1.5096375685809075e-16 |
| wasserstein_Iota | 0.003294726270972629 | 4.336808689942018e-19 | 1.3162880109799708e-16 |
| wasserstein_S:677 | 0.010154107596608031 | 0.0 | 0.0 |
| jsd_Mu | 0.09034250767298585 | 0.0 | 0.0 |
| wasserstein_Beta | 0.00591690550303248 | 0.0 | 0.0 |

## 8. Invariant Checks

| Check | Status |
|-------|--------|
| tsagris_closure | ✅ |
| tsagris_observed_preserved | ✅ |
| tsagris_nonnegative | ✅ |
| fused_closure | ✅ |
| fused_observed_preserved | ✅ |
| fused_nonnegative | ✅ |
| gated_gan_closure | ✅ |
| gated_gan_observed_preserved | ✅ |
| gated_gan_nonnegative | ✅ |

## 9. Ablation: Gated vs Ungated GAN

| Configuration | MSE (prop) | JSD |
|---------------|-----------|-----|
| gated_gan | 0.005283496365767198 | 0.05757718057271585 |
| ungated_gan | 0.0039737290054697475 | 0.05757718057271585 |

---

*Report generated by missing-imputation evaluate*