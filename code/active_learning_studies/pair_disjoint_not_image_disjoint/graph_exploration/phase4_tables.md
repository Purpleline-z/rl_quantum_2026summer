
#### log-loss (secondary): gain over Random, seeds 480-499 (mean [95% bootstrap CI] (share of seeds better, raw p; Holm over the 4 blocks for the candidate))

| rule | A-single | A-sequential | B-single | B-sequential |
|---|---|---|---|---|
| typed_zeroshot_coverage_unc | +0.008 [-0.036, +0.049] (50%, p 0.648, Holm 1.000) | +0.015 [-0.023, +0.053] (60%, p 0.498, Holm 1.000) | +0.024 [-0.030, +0.072] (75%, p 0.349, Holm 1.000) | +0.024 [-0.044, +0.085] (60%, p 0.261, Holm 1.000) |
| typed_flat_coverage_a99 (control) | +0.032 [-0.008, +0.070] (75%, p 0.097) | +0.058 [+0.006, +0.108] (70%, p 0.048) | +0.015 [-0.014, +0.045] (55%, p 0.452) | +0.045 [+0.011, +0.078] (80%, p 0.044) |
| typed_zeroshot_coverage_unc_shuffled (control) | +0.043 [+0.006, +0.081] (70%, p 0.044) | +0.059 [+0.015, +0.102] (75%, p 0.006) | +0.069 [+0.036, +0.102] (75%, p 0.002) | +0.050 [-0.001, +0.106] (70%, p 0.076) |

n seeds per cell: [20]

#### AUC (PRIMARY): gain over Random, seeds 480-499 (mean [95% bootstrap CI] (share of seeds better, raw p; Holm over the 4 blocks for the candidate))

| rule | A-single | A-sequential | B-single | B-sequential |
|---|---|---|---|---|
| typed_zeroshot_coverage_unc | +0.013 [+0.001, +0.026] (75%, p 0.033, Holm 0.059) | +0.016 [+0.004, +0.029] (60%, p 0.030, Holm 0.059) | +0.019 [+0.007, +0.029] (80%, p 0.002, Holm 0.006) | +0.026 [+0.014, +0.038] (85%, p 0.001, Holm 0.003) |
| typed_flat_coverage_a99 (control) | +0.006 [-0.006, +0.019] (60%, p 0.409) | +0.009 [-0.005, +0.022] (60%, p 0.091) | +0.001 [-0.006, +0.010] (50%, p 0.927) | +0.010 [-0.000, +0.020] (70%, p 0.053) |
| typed_zeroshot_coverage_unc_shuffled (control) | +0.012 [+0.003, +0.021] (65%, p 0.024) | +0.014 [+0.007, +0.021] (95%, p 0.000) | +0.013 [+0.002, +0.023] (70%, p 0.036) | +0.012 [-0.002, +0.026] (65%, p 0.154) |

n seeds per cell: [20]

#### Paired differences, AUC gain (candidate minus control)

| candidate minus | A-single | A-sequential | B-single | B-sequential |
|---|---|---|---|---|
| flat_coverage_a99 | +0.007 [-0.004, +0.019] (p 0.349) | +0.007 [-0.004, +0.021] (p 0.349) | +0.018 [+0.003, +0.030] (p 0.011) | +0.017 [+0.005, +0.027] (p 0.014) |
| zeroshot_coverage_unc_shuffled | +0.001 [-0.009, +0.011] (p 0.674) | +0.002 [-0.009, +0.013] (p 0.701) | +0.006 [-0.008, +0.020] (p 0.245) | +0.015 [-0.003, +0.031] (p 0.114) |

#### Paired differences, log-loss gain (candidate minus control)

| candidate minus | A-single | A-sequential | B-single | B-sequential |
|---|---|---|---|---|
| flat_coverage_a99 | -0.024 [-0.064, +0.021] (p 0.090) | -0.043 [-0.091, +0.010] (p 0.036) | +0.008 [-0.052, +0.062] (p 0.522) | -0.021 [-0.075, +0.030] (p 0.571) |
| zeroshot_coverage_unc_shuffled | -0.035 [-0.080, +0.006] (p 0.216) | -0.043 [-0.091, +0.005] (p 0.114) | -0.046 [-0.102, +0.008] (p 0.216) | -0.026 [-0.103, +0.045] (p 0.546) |

Replication criterion (AUC: Holm p < .05 and positive in >= 2 blocks incl. both splits): significant blocks ['B-single', 'B-sequential'] -> NOT met
