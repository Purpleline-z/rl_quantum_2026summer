### Random's level by budget (mean over 35 seeds; Random = mean of its 5 draws)

| Split | condition | metric | schedule | 10 judgments | 20 judgments | 40 judgments | 60 judgments |
|---|---|---|---|---:|---:|---:|---:|
| Split A | single | log-loss | old | 0.666 | 0.634 | 0.552 | 0.496 |
| Split A | single | log-loss | new | 0.528 | 0.493 | 0.443 | 0.464 |
| Split A | single | AUC | old | 0.876 | 0.884 | 0.895 | 0.902 |
| Split A | single | AUC | new | 0.871 | 0.882 | 0.898 | 0.904 |
| Split A | single | accuracy (secondary) | old | 0.811 | 0.820 | 0.829 | 0.829 |
| Split A | single | accuracy (secondary) | new | 0.796 | 0.811 | 0.825 | 0.832 |
| Split A | sequential | log-loss | old | 0.659 | 0.615 | 0.556 | 0.508 |
| Split A | sequential | log-loss | new | 0.523 | 0.486 | 0.447 | 0.456 |
| Split A | sequential | AUC | old | 0.878 | 0.882 | 0.893 | 0.900 |
| Split A | sequential | AUC | new | 0.875 | 0.883 | 0.897 | 0.904 |
| Split A | sequential | accuracy (secondary) | old | 0.811 | 0.815 | 0.823 | 0.829 |
| Split A | sequential | accuracy (secondary) | new | 0.804 | 0.810 | 0.825 | 0.833 |
| Split B | single | log-loss | old | 0.596 | 0.547 | 0.527 | 0.470 |
| Split B | single | log-loss | new | 0.458 | 0.432 | 0.417 | 0.420 |
| Split B | single | AUC | old | 0.881 | 0.889 | 0.894 | 0.905 |
| Split B | single | AUC | new | 0.891 | 0.900 | 0.905 | 0.911 |
| Split B | single | accuracy (secondary) | old | 0.815 | 0.825 | 0.829 | 0.839 |
| Split B | single | accuracy (secondary) | new | 0.825 | 0.833 | 0.840 | 0.844 |
| Split B | sequential | log-loss | old | 0.627 | 0.607 | 0.560 | 0.514 |
| Split B | sequential | log-loss | new | 0.482 | 0.463 | 0.427 | 0.450 |
| Split B | sequential | AUC | old | 0.875 | 0.880 | 0.889 | 0.898 |
| Split B | sequential | AUC | new | 0.886 | 0.892 | 0.903 | 0.905 |
| Split B | sequential | accuracy (secondary) | old | 0.809 | 0.813 | 0.821 | 0.837 |
| Split B | sequential | accuracy (secondary) | new | 0.816 | 0.825 | 0.836 | 0.841 |

### Initial rows only (no acquired judgment), test metrics, mean over seeds

| Split | schedule | seeds | log-loss | AUC | accuracy |
|---|---|---:|---:|---:|---:|
| Split A | old | 35 | 0.715 | 0.872 | 0.813 |
| Split A | new | 35 | 0.563 | 0.866 | 0.798 |
| Split B | old | 35 | 0.667 | 0.865 | 0.804 |
| Split B | new | 35 | 0.513 | 0.872 | 0.812 |

### Holm-significant variants (p < 0.05 within a table of 32) under the old and the re-tuned schedule

Gain = improvement over Random averaged over budgets 10/20/40/60 (positive is better, also for log-loss). `n.s.` = Holm p >= 0.05 (value shown).

| Table | metric | variant | old gain (Holm p) | re-tuned gain (Holm p) | status |
|---|---|---|---|---|---|
| Split A, single | cal. log-loss | BALD x P(decisive) | -0.003 (1.000) | +0.022 (0.025) | only re-tuned schedule |
| Split A, single | accuracy (secondary) | Deep-ensemble BALD x P(decisive) | +0.015 (0.002) | +0.019 (0.001) | significant under both |
| Split A, single | accuracy (secondary) | BALD x P(decisive) | +0.009 (0.969) | +0.017 (0.049) | only re-tuned schedule |
| Split A, single | accuracy (secondary) | Deep-ensemble BALD (8 heads) | +0.012 (0.039) | +0.016 (0.011) | significant under both |
| Split A, sequential | cal. log-loss | BALD x P(decisive) | +0.035 (1.000) | +0.057 (0.019) | only re-tuned schedule |
| Split A, sequential | AUC | Uncertainty, all heads | -0.012 (0.331) | -0.017 (0.035) | only re-tuned schedule |
| Split A, sequential | AUC | Uncertainty, original code, all heads | -0.012 (0.331) | -0.017 (0.035) | only re-tuned schedule |
| Split A, sequential | accuracy (secondary) | Fisher D-optimal (Active Reward Modeling) | +0.024 (<0.001) | +0.016 (0.121) | only old schedule |
| Split A, sequential | accuracy (secondary) | Deep-ensemble BALD x P(decisive) | +0.020 (0.002) | +0.017 (0.006) | significant under both |
| Split A, sequential | accuracy (secondary) | Deep-ensemble BALD (8 heads) | +0.019 (0.010) | +0.014 (0.118) | only old schedule |
| Split A, sequential | accuracy (secondary) | FASS (pairs) | +0.016 (0.021) | +0.008 (0.890) | only old schedule |
| Split A, sequential | accuracy (secondary) | Image-coverage uncertainty | +0.013 (0.040) | +0.014 (0.009) | significant under both |
| Split B, single | log-loss | DPP (quality x diversity) | -0.104 (0.003) | -0.040 (0.053) | only old schedule |
| Split B, single | log-loss | Image-coverage uncertainty | -0.069 (0.243) | -0.051 (0.007) | only re-tuned schedule |
| Split B, single | log-loss | Uncertainty, all heads | -0.049 (0.102) | -0.045 (0.002) | only re-tuned schedule |
| Split B, single | log-loss | Uncertainty, original code, all heads | -0.049 (0.102) | -0.045 (0.002) | only re-tuned schedule |
| Split B, single | log-loss | Uncertainty | -0.047 (0.369) | -0.043 (0.002) | only re-tuned schedule |
| Split B, single | log-loss | Uncertainty + diversity, original code, all heads | -0.019 (1.000) | -0.038 (0.035) | only re-tuned schedule |
| Split B, single | AUC | Uncertainty, all heads | -0.026 (<0.001) | -0.021 (<0.001) | significant under both |
| Split B, single | AUC | Uncertainty, original code, all heads | -0.026 (<0.001) | -0.021 (<0.001) | significant under both |
| Split B, single | AUC | MC-dropout mutual info, original code, all heads | -0.014 (0.040) | -0.016 (0.003) | significant under both |
| Split B, single | AUC | MC-dropout variance, original code, all heads | -0.007 (1.000) | -0.016 (0.012) | only re-tuned schedule |
| Split B, single | AUC | Uncertainty + diversity, original code, all heads | -0.016 (0.053) | -0.015 (0.008) | only re-tuned schedule |
| Split B, single | AUC | Image-coverage uncertainty | -0.015 (0.529) | -0.013 (0.029) | only re-tuned schedule |
| Split B, single | AUC | Uncertainty | -0.013 (0.326) | -0.012 (0.012) | only re-tuned schedule |
| Split B, single | accuracy (secondary) | Largest predicted gap | -0.024 (<0.001) | -0.013 (0.629) | only old schedule |
| Split B, single | accuracy (secondary) | Gap + posterior std (DeltaUCB-style) | -0.019 (0.019) | -0.010 (1.000) | only old schedule |
| Split B, sequential | log-loss | Cluster-quota uncertainty, original code, all heads | +0.074 (<0.001) | +0.021 (1.000) | only old schedule |
| Split B, sequential | AUC | MC-dropout variance, original code, all heads | -0.007 (1.000) | -0.019 (0.014) | only re-tuned schedule |
| Split B, sequential | AUC | BALD x P(decisive) | +0.012 (0.227) | +0.017 (0.020) | only re-tuned schedule |
| Split B, sequential | AUC | Deep-ensemble BALD x P(decisive) | +0.014 (0.023) | +0.006 (1.000) | only old schedule |
| Split B, sequential | accuracy (secondary) | BALD x P(decisive) | +0.016 (0.117) | +0.027 (<0.001) | only re-tuned schedule |
| Split B, sequential | accuracy (secondary) | Deep-ensemble BALD x P(decisive) | +0.022 (0.001) | +0.016 (0.019) | significant under both |
| Split B, sequential | accuracy (secondary) | Fisher D-optimal (Active Reward Modeling) | +0.020 (0.006) | +0.017 (0.209) | only old schedule |
| Split B, sequential | accuracy (secondary) | Deep-ensemble BALD (8 heads) | +0.019 (0.031) | +0.015 (0.139) | only old schedule |
| Split B, sequential | accuracy (secondary) | Cluster-quota uncertainty | +0.016 (0.024) | +0.008 (1.000) | only old schedule |
| Split B, sequential | accuracy (secondary) | Uncertainty + diversity | +0.014 (0.398) | +0.016 (0.003) | only re-tuned schedule |
| Split B, sequential | accuracy (secondary) | BADGE (pairs) | +0.015 (0.013) | +0.014 (0.605) | only old schedule |

### Per table: significance counts and rank agreement of the 32 variants' gains (old vs re-tuned schedule)

| Table | metric | Holm<0.05 old | Holm<0.05 re-tuned | in both | nominal p<0.05 old / re-tuned | Spearman rho (p) | mean abs gain old / re-tuned |
|---|---|---:|---:|---:|---:|---|---|
| Split A, single | log-loss | 0 | 0 | 0 | 0 / 4 | +0.36 (0.044) | 0.014 / 0.011 |
| Split A, single | cal. log-loss | 0 | 1 | 0 | 3 / 4 | +0.55 (0.001) | 0.017 / 0.013 |
| Split A, single | AUC | 0 | 0 | 0 | 5 / 7 | +0.76 (<0.001) | 0.005 / 0.005 |
| Split A, single | accuracy (secondary) | 2 | 3 | 2 | 10 / 13 | +0.87 (<0.001) | 0.007 / 0.007 |
| Split A, sequential | log-loss | 0 | 0 | 0 | 4 / 6 | +0.28 (0.124) | 0.021 / 0.015 |
| Split A, sequential | cal. log-loss | 0 | 1 | 0 | 5 / 9 | +0.59 (<0.001) | 0.028 / 0.026 |
| Split A, sequential | AUC | 0 | 2 | 0 | 8 / 8 | +0.74 (<0.001) | 0.006 / 0.007 |
| Split A, sequential | accuracy (secondary) | 5 | 2 | 2 | 12 / 15 | +0.85 (<0.001) | 0.009 / 0.009 |
| Split B, single | log-loss | 1 | 5 | 0 | 8 / 18 | +0.77 (<0.001) | 0.028 / 0.024 |
| Split B, single | AUC | 3 | 7 | 3 | 11 / 14 | +0.83 (<0.001) | 0.008 / 0.008 |
| Split B, single | accuracy (secondary) | 2 | 0 | 0 | 4 / 6 | +0.73 (<0.001) | 0.006 / 0.007 |
| Split B, sequential | log-loss | 1 | 0 | 0 | 8 / 7 | +0.67 (<0.001) | 0.026 / 0.020 |
| Split B, sequential | AUC | 1 | 2 | 0 | 9 / 7 | +0.74 (<0.001) | 0.006 / 0.006 |
| Split B, sequential | accuracy (secondary) | 5 | 3 | 1 | 13 / 12 | +0.86 (<0.001) | 0.009 / 0.009 |

### Spearman correlation pooled over the four tables (128 variant-table gains per metric)

| metric | n | Spearman rho | p |
|---|---:|---:|---|
| log-loss | 128 | +0.58 | <0.001 |
| cal. log-loss | 64 | +0.55 | <0.001 |
| AUC | 128 | +0.80 | <0.001 |
| accuracy (secondary) | 128 | +0.84 | <0.001 |