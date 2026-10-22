Seeds: 35; rows: 1750

### Mean test ideal accuracy per rule (average over seeds and the four budgets)

| rule | mean_accuracy |
|---|---|
| bald_decisive | 0.8467 |
| core_set | 0.8431 |
| ensemble_bald | 0.8477 |
| laplace_bald | 0.8480 |
| probcover_pairs | 0.8372 |
| random | 0.8441 |
| typiclust_pairs | 0.8421 |
| uncertainty | 0.8513 |

### Primary: test ideal-image accuracy, paired difference to Random (budgets averaged per seed; Holm over the 7 non-random rules)

| rule | n_seeds | mean_diff_vs_random | ci_lo | ci_hi | share_seeds_better | wilcoxon_p | holm_p | ci_excludes_0 |
|---|---|---|---|---|---|---|---|---|
| uncertainty | 35 | 0.0072 | -0.0020 | 0.0162 | 0.7143 | 0.0754 | 0.5281 | False |
| core_set | 35 | -0.0010 | -0.0085 | 0.0059 | 0.5143 | 0.9217 | 1.0000 | False |
| typiclust_pairs | 35 | -0.0020 | -0.0117 | 0.0089 | 0.3714 | 0.3009 | 1.0000 | False |
| laplace_bald | 35 | 0.0039 | -0.0042 | 0.0117 | 0.6286 | 0.1956 | 1.0000 | False |
| bald_decisive | 35 | 0.0026 | -0.0059 | 0.0122 | 0.5143 | 0.9935 | 1.0000 | False |
| ensemble_bald | 35 | 0.0036 | -0.0072 | 0.0134 | 0.6571 | 0.1928 | 1.0000 | False |
| probcover_pairs | 35 | -0.0068 | -0.0161 | 0.0024 | 0.4571 | 0.2014 | 1.0000 | False |

### Accuracy per budget (mean over seeds)

| rule | 10 | 20 | 40 | 60 |
|---|---|---|---|---|
| random | 0.8449 | 0.8422 | 0.8469 | 0.8422 |
| uncertainty | 0.8429 | 0.8510 | 0.8561 | 0.8551 |
| core_set | 0.8327 | 0.8367 | 0.8469 | 0.8561 |
| typiclust_pairs | 0.8449 | 0.8439 | 0.8449 | 0.8347 |
| laplace_bald | 0.8347 | 0.8510 | 0.8520 | 0.8541 |
| bald_decisive | 0.8459 | 0.8398 | 0.8520 | 0.8490 |
| ensemble_bald | 0.8449 | 0.8469 | 0.8459 | 0.8531 |
| probcover_pairs | 0.8388 | 0.8378 | 0.8408 | 0.8316 |

### Secondary: macro-F1, paired difference to Random

| rule | n_seeds | mean_diff_vs_random | ci_lo | ci_hi | share_seeds_better | wilcoxon_p | holm_p | ci_excludes_0 |
|---|---|---|---|---|---|---|---|---|
| uncertainty | 35 | 0.0064 | -0.0038 | 0.0161 | 0.6857 | 0.1002 | 0.7013 | False |
| core_set | 35 | -0.0012 | -0.0091 | 0.0060 | 0.4857 | 0.9163 | 1.0000 | False |
| typiclust_pairs | 35 | -0.0041 | -0.0149 | 0.0075 | 0.4286 | 0.2726 | 1.0000 | False |
| laplace_bald | 35 | 0.0031 | -0.0058 | 0.0116 | 0.5714 | 0.3507 | 1.0000 | False |
| bald_decisive | 35 | 0.0007 | -0.0085 | 0.0106 | 0.4857 | 0.8398 | 1.0000 | False |
| ensemble_bald | 35 | 0.0026 | -0.0094 | 0.0130 | 0.6571 | 0.2655 | 1.0000 | False |
| probcover_pairs | 35 | -0.0071 | -0.0175 | 0.0032 | 0.4857 | 0.2726 | 1.0000 | False |

### Secondary: held-out pair-group preference AUC, paired difference to Random

| rule | n_seeds | mean_diff_vs_random | ci_lo | ci_hi | share_seeds_better | wilcoxon_p | holm_p | ci_excludes_0 |
|---|---|---|---|---|---|---|---|---|
| uncertainty | 35 | -0.0121 | -0.0215 | -0.0040 | 0.4286 | 0.0191 | 0.1147 | True |
| core_set | 35 | 0.0077 | -0.0006 | 0.0153 | 0.7143 | 0.0263 | 0.1147 | False |
| typiclust_pairs | 35 | 0.0105 | 0.0027 | 0.0184 | 0.6286 | 0.0200 | 0.1147 | True |
| laplace_bald | 35 | 0.0121 | 0.0044 | 0.0196 | 0.7143 | 0.0070 | 0.0492 | True |
| bald_decisive | 35 | 0.0094 | 0.0016 | 0.0173 | 0.6571 | 0.0438 | 0.1315 | True |
| ensemble_bald | 35 | 0.0058 | -0.0017 | 0.0132 | 0.6286 | 0.1535 | 0.3070 | False |
| probcover_pairs | 35 | -0.0000 | -0.0078 | 0.0077 | 0.4857 | 0.9806 | 0.9806 | False |

### Secondary: held-out pair-group preference accuracy, paired difference to Random

| rule | n_seeds | mean_diff_vs_random | ci_lo | ci_hi | share_seeds_better | wilcoxon_p | holm_p | ci_excludes_0 |
|---|---|---|---|---|---|---|---|---|
| uncertainty | 35 | -0.0005 | -0.0107 | 0.0088 | 0.5714 | 0.7771 | 1.0000 | False |
| core_set | 35 | -0.0004 | -0.0091 | 0.0078 | 0.5429 | 0.9291 | 1.0000 | False |
| typiclust_pairs | 35 | 0.0098 | 0.0012 | 0.0185 | 0.6571 | 0.0495 | 0.2473 | True |
| laplace_bald | 35 | 0.0094 | -0.0009 | 0.0196 | 0.5714 | 0.1002 | 0.4007 | False |
| bald_decisive | 35 | 0.0131 | 0.0046 | 0.0219 | 0.6286 | 0.0096 | 0.0672 | True |
| ensemble_bald | 35 | 0.0115 | 0.0034 | 0.0195 | 0.6286 | 0.0124 | 0.0744 | True |
| probcover_pairs | 35 | 0.0026 | -0.0060 | 0.0115 | 0.5143 | 0.6505 | 1.0000 | False |

### Secondary: held-out pair-group preference log-loss (positive = better), paired difference to Random

| rule | n_seeds | mean_diff_vs_random | ci_lo | ci_hi | share_seeds_better | wilcoxon_p | holm_p | ci_excludes_0 |
|---|---|---|---|---|---|---|---|---|
| uncertainty | 35 | -0.0469 | -0.0914 | -0.0107 | 0.4286 | 0.0871 | 0.3483 | True |
| core_set | 35 | 0.0453 | 0.0092 | 0.0821 | 0.7429 | 0.0137 | 0.0958 | True |
| typiclust_pairs | 35 | 0.0484 | 0.0143 | 0.0826 | 0.6000 | 0.0151 | 0.0958 | True |
| laplace_bald | 35 | 0.0324 | -0.0023 | 0.0656 | 0.6857 | 0.0557 | 0.2783 | False |
| bald_decisive | 35 | 0.0302 | -0.0061 | 0.0665 | 0.6000 | 0.1535 | 0.4606 | False |
| ensemble_bald | 35 | 0.0076 | -0.0227 | 0.0366 | 0.5429 | 0.5545 | 1.0000 | False |
| probcover_pairs | 35 | 0.0033 | -0.0345 | 0.0398 | 0.4286 | 0.9935 | 1.0000 | False |

### No-selection references (initial 10 groups only; all candidate-pool groups)

| reference | n_seeds | test_ideal_accuracy | test_ideal_macro_f1 | pref_auc | pref_accuracy | pref_log_loss |
|---|---|---|---|---|---|---|
| initial_only | 35 | 0.8408 | 0.8240 | 0.8460 | 0.7901 | 0.6633 |
| all_pool | 35 | 0.8531 | 0.8351 | 0.9272 | 0.8465 | 0.3469 |
