
#### log-loss: gain over Random, seeds 460-479 (mean [95% bootstrap CI] (share of seeds better, raw p; Holm over 12 candidate tests))

| rule | A-single | A-sequential | B-single | B-sequential |
|---|---|---|---|---|
| typed_decisive_coverage_ul_k10a99 | +0.035 [+0.006, +0.066] (60%, p 0.076, Holm 0.765) | +0.041 [+0.005, +0.080] (70%, p 0.058, Holm 0.699) | +0.024 [-0.010, +0.057] (65%, p 0.189, Holm 1.000) | +0.027 [-0.012, +0.066] (60%, p 0.245, Holm 1.000) |
| typed_zeroshot_coverage_a9 | +0.032 [-0.010, +0.077] (60%, p 0.261, Holm 1.000) | +0.032 [-0.024, +0.092] (60%, p 0.498, Holm 1.000) | +0.038 [-0.015, +0.096] (65%, p 0.202, Holm 1.000) | +0.040 [-0.003, +0.086] (75%, p 0.070, Holm 0.765) |
| typed_zeroshot_coverage_unc | +0.043 [-0.003, +0.088] (75%, p 0.097, Holm 0.876) | +0.031 [-0.023, +0.093] (75%, p 0.114, Holm 0.912) | -0.022 [-0.075, +0.028] (45%, p 0.475, Holm 1.000) | -0.024 [-0.090, +0.033] (65%, p 1.000, Holm 1.000) |
| typed_flat_coverage_a99 (control/reference) | +0.050 [+0.022, +0.081] (75%, p 0.004) | +0.051 [+0.009, +0.098] (65%, p 0.058) | +0.048 [+0.003, +0.095] (70%, p 0.076) | +0.018 [-0.034, +0.067] (60%, p 0.312) |
| typed_flat_coverage_ul_k10 (control/reference) | +0.037 [+0.008, +0.065] (85%, p 0.014) | +0.038 [+0.001, +0.072] (75%, p 0.036) | +0.067 [+0.020, +0.115] (85%, p 0.001) | +0.059 [+0.014, +0.101] (85%, p 0.004) |
| typed_decisive_coverage_ul_k10a99_shuffled (control/reference) | +0.055 [+0.020, +0.093] (70%, p 0.011) | +0.055 [+0.005, +0.106] (70%, p 0.053) | +0.029 [-0.020, +0.078] (65%, p 0.105) | +0.034 [-0.004, +0.068] (75%, p 0.064) |
| typed_zeroshot_coverage_a9_shuffled (control/reference) | +0.011 [-0.026, +0.049] (60%, p 0.596) | +0.013 [-0.032, +0.062] (55%, p 0.701) | +0.039 [+0.001, +0.076] (70%, p 0.048) | +0.044 [+0.006, +0.084] (65%, p 0.058) |
| typed_zeroshot_coverage_unc_shuffled (control/reference) | +0.020 [-0.015, +0.052] (65%, p 0.058) | +0.069 [+0.035, +0.103] (80%, p 0.001) | -0.002 [-0.068, +0.056] (65%, p 0.546) | +0.017 [-0.042, +0.068] (70%, p 0.294) |
| typed_decisive_coverage (control/reference) | +0.009 [-0.046, +0.060] (70%, p 0.330) | +0.004 [-0.055, +0.059] (45%, p 0.674) | +0.062 [+0.024, +0.101] (75%, p 0.007) | +0.040 [-0.001, +0.082] (75%, p 0.070) |

n seeds per cell: [20]

#### AUC: gain over Random, seeds 460-479 (mean [95% bootstrap CI] (share of seeds better, raw p; Holm over 12 candidate tests))

| rule | A-single | A-sequential | B-single | B-sequential |
|---|---|---|---|---|
| typed_decisive_coverage_ul_k10a99 | +0.006 [-0.001, +0.012] (65%, p 0.090, Holm 0.348) | +0.010 [-0.000, +0.020] (70%, p 0.083, Holm 0.348) | +0.007 [-0.001, +0.015] (75%, p 0.053, Holm 0.319) | +0.012 [+0.003, +0.021] (75%, p 0.017, Holm 0.155) |
| typed_zeroshot_coverage_a9 | +0.006 [-0.004, +0.015] (55%, p 0.312, Holm 0.624) | +0.006 [-0.007, +0.020] (50%, p 0.622, Holm 0.624) | +0.008 [-0.001, +0.017] (70%, p 0.070, Holm 0.348) | +0.010 [+0.001, +0.018] (85%, p 0.030, Holm 0.213) |
| typed_zeroshot_coverage_unc | +0.017 [+0.006, +0.029] (80%, p 0.006, Holm 0.064) | +0.018 [+0.005, +0.034] (85%, p 0.006, Holm 0.061) | +0.012 [+0.003, +0.021] (65%, p 0.027, Holm 0.213) | +0.015 [+0.007, +0.023] (85%, p 0.002, Holm 0.024) |
| typed_flat_coverage_a99 (control/reference) | +0.009 [-0.002, +0.019] (75%, p 0.064) | +0.008 [-0.005, +0.023] (50%, p 0.388) | +0.002 [-0.012, +0.015] (50%, p 0.869) | -0.002 [-0.015, +0.010] (55%, p 0.869) |
| typed_flat_coverage_ul_k10 (control/reference) | -0.002 [-0.011, +0.006] (45%, p 0.869) | -0.001 [-0.010, +0.007] (55%, p 0.784) | +0.002 [-0.010, +0.013] (55%, p 0.674) | +0.003 [-0.008, +0.014] (65%, p 0.277) |
| typed_decisive_coverage_ul_k10a99_shuffled (control/reference) | +0.001 [-0.007, +0.010] (40%, p 0.596) | +0.002 [-0.011, +0.015] (45%, p 0.841) | -0.002 [-0.016, +0.009] (60%, p 0.388) | +0.000 [-0.010, +0.011] (65%, p 0.522) |
| typed_zeroshot_coverage_a9_shuffled (control/reference) | +0.004 [-0.007, +0.014] (60%, p 0.368) | +0.004 [-0.007, +0.017] (45%, p 0.985) | +0.006 [-0.004, +0.016] (60%, p 0.216) | +0.008 [-0.004, +0.019] (70%, p 0.133) |
| typed_zeroshot_coverage_unc_shuffled (control/reference) | +0.005 [-0.004, +0.013] (60%, p 0.261) | +0.015 [+0.007, +0.023] (75%, p 0.001) | -0.000 [-0.011, +0.012] (50%, p 0.956) | +0.003 [-0.008, +0.014] (60%, p 0.596) |
| typed_decisive_coverage (control/reference) | +0.007 [-0.002, +0.016] (65%, p 0.177) | +0.008 [-0.003, +0.019] (65%, p 0.202) | +0.011 [+0.002, +0.020] (60%, p 0.053) | +0.011 [+0.002, +0.020] (70%, p 0.033) |

n seeds per cell: [20]

#### Paired differences, log-loss gain (candidate minus control; positive = the candidate's ingredient helps)

| candidate minus control | A-single | A-sequential | B-single | B-sequential |
|---|---|---|---|---|
| decisive_coverage_ul_k10a99 minus decisive_coverage_ul_k10a99_shuffled | -0.019 [-0.062, +0.021] (p 0.522) | -0.014 [-0.046, +0.020] (p 0.277) | -0.005 [-0.045, +0.036] (p 0.674) | -0.007 [-0.040, +0.027] (p 0.927) |
| decisive_coverage_ul_k10a99 minus flat_coverage_ul_k10 | -0.002 [-0.030, +0.030] (p 0.546) | +0.003 [-0.030, +0.042] (p 0.729) | -0.043 [-0.082, -0.004] (p 0.058) | -0.032 [-0.076, +0.012] (p 0.231) |
| zeroshot_coverage_a9 minus zeroshot_coverage_a9_shuffled | +0.021 [-0.031, +0.077] (p 0.898) | +0.019 [-0.027, +0.067] (p 0.622) | -0.000 [-0.064, +0.071] (p 0.729) | -0.003 [-0.053, +0.050] (p 0.729) |
| zeroshot_coverage_a9 minus flat_coverage_a99 | -0.018 [-0.055, +0.019] (p 0.330) | -0.019 [-0.055, +0.020] (p 0.202) | -0.010 [-0.076, +0.051] (p 0.956) | +0.023 [-0.038, +0.081] (p 0.409) |
| zeroshot_coverage_unc minus zeroshot_coverage_unc_shuffled | +0.023 [-0.019, +0.063] (p 0.189) | -0.037 [-0.084, +0.012] (p 0.105) | -0.020 [-0.074, +0.035] (p 0.294) | -0.041 [-0.084, +0.003] (p 0.048) |
| zeroshot_coverage_unc minus flat_coverage_a99 | -0.008 [-0.051, +0.036] (p 0.674) | -0.020 [-0.066, +0.026] (p 0.452) | -0.070 [-0.147, +0.006] (p 0.123) | -0.041 [-0.139, +0.047] (p 0.729) |

#### Paired differences, AUC gain (candidate minus control; positive = the candidate's ingredient helps)

| candidate minus control | A-single | A-sequential | B-single | B-sequential |
|---|---|---|---|---|
| decisive_coverage_ul_k10a99 minus decisive_coverage_ul_k10a99_shuffled | +0.006 [-0.006, +0.016] (p 0.231) | +0.008 [-0.001, +0.017] (p 0.216) | +0.010 [-0.003, +0.022] (p 0.245) | +0.012 [+0.002, +0.021] (p 0.040) |
| decisive_coverage_ul_k10a99 minus flat_coverage_ul_k10 | +0.008 [+0.001, +0.016] (p 0.027) | +0.011 [+0.003, +0.020] (p 0.064) | +0.006 [-0.003, +0.015] (p 0.368) | +0.008 [-0.001, +0.018] (p 0.177) |
| zeroshot_coverage_a9 minus zeroshot_coverage_a9_shuffled | +0.002 [-0.009, +0.014] (p 0.841) | +0.002 [-0.006, +0.012] (p 0.927) | +0.002 [-0.009, +0.014] (p 0.812) | +0.002 [-0.010, +0.014] (p 0.701) |
| zeroshot_coverage_a9 minus flat_coverage_a99 | -0.003 [-0.012, +0.006] (p 0.475) | -0.002 [-0.010, +0.006] (p 0.648) | +0.007 [-0.008, +0.020] (p 0.294) | +0.012 [-0.002, +0.025] (p 0.083) |
| zeroshot_coverage_unc minus zeroshot_coverage_unc_shuffled | +0.012 [+0.001, +0.023] (p 0.033) | +0.004 [-0.008, +0.015] (p 0.522) | +0.012 [+0.000, +0.023] (p 0.076) | +0.012 [+0.004, +0.022] (p 0.007) |
| zeroshot_coverage_unc minus flat_coverage_a99 | +0.008 [-0.002, +0.019] (p 0.154) | +0.010 [-0.001, +0.021] (p 0.165) | +0.010 [-0.005, +0.025] (p 0.388) | +0.017 [+0.001, +0.034] (p 0.058) |

#### Success criterion (Holm p < .05 and positive gain in BOTH log-loss and AUC in >= 2 blocks incl. both splits)

- typed_decisive_coverage_ul_k10a99: blocks significant in both metrics [] -> NOT met
- typed_zeroshot_coverage_a9: blocks significant in both metrics [] -> NOT met
- typed_zeroshot_coverage_unc: blocks significant in both metrics [] -> NOT met
