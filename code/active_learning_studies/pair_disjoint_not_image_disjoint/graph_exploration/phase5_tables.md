
#### AUC: gain over Random, seeds 500-519 (mean [95% bootstrap CI] (share of seeds better, raw p))

| rule | A-single | A-sequential | B-single | B-sequential |
|---|---|---|---|---|
| typed_zeroshot_coverage_unc | +0.009 [+0.003, +0.016] (70%, p 0.017) | +0.015 [+0.005, +0.024] (70%, p 0.007) | +0.023 [+0.016, +0.031] (95%, p 0.000) | +0.025 [+0.015, +0.036] (90%, p 0.000) |
| typed_flat_unc_coverage_a99 | +0.004 [-0.002, +0.011] (60%, p 0.261) | +0.018 [+0.007, +0.029] (80%, p 0.002) | +0.014 [+0.006, +0.022] (80%, p 0.003) | +0.009 [-0.003, +0.022] (60%, p 0.231) |

Paired difference AUC (typed_zeroshot_coverage_unc minus typed_flat_unc_coverage_a99):

| | A-single | A-sequential | B-single | B-sequential |
|---|---|---|---|---|
| difference | +0.005 [-0.004, +0.015] (50%, p 0.349), Holm 0.698 | -0.003 [-0.013, +0.006] (45%, p 0.756), Holm 0.756 | +0.009 [+0.001, +0.017] (60%, p 0.165), Holm 0.495 | +0.015 [+0.009, +0.022] (80%, p 0.001), Holm 0.003 |

#### log-loss: gain over Random, seeds 500-519 (mean [95% bootstrap CI] (share of seeds better, raw p))

| rule | A-single | A-sequential | B-single | B-sequential |
|---|---|---|---|---|
| typed_zeroshot_coverage_unc | +0.010 [-0.032, +0.052] (50%, p 0.622) | +0.030 [-0.012, +0.072] (65%, p 0.245) | +0.046 [+0.015, +0.076] (75%, p 0.014) | +0.050 [+0.001, +0.102] (65%, p 0.083) |
| typed_flat_unc_coverage_a99 | +0.020 [-0.003, +0.044] (65%, p 0.143) | +0.085 [+0.036, +0.140] (65%, p 0.009) | +0.077 [+0.049, +0.108] (95%, p 0.000) | +0.054 [-0.004, +0.113] (65%, p 0.123) |

Paired difference log-loss (typed_zeroshot_coverage_unc minus typed_flat_unc_coverage_a99):

| | A-single | A-sequential | B-single | B-sequential |
|---|---|---|---|---|
| difference | -0.010 [-0.062, +0.042] (40%, p 0.648), Holm 1.000 | -0.055 [-0.108, -0.001] (40%, p 0.083), Holm 0.330 | -0.031 [-0.074, +0.009] (40%, p 0.294), Holm 0.883 | -0.003 [-0.043, +0.038] (40%, p 0.756), Holm 1.000 |

Primary criterion (paired AUC difference, Holm < .05 and positive in >= 2 of 4 blocks): significant blocks ['B-sequential'] -> NOT met
