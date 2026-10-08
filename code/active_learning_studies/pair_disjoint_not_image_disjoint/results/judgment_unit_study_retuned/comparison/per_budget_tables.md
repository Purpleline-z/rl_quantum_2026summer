**A single, log-loss (re-tuned schedule)**

| Budget (judgments) | top 3 by mean gain over Random (Holm p within the budget) | variants with Holm p < 0.05 |
|---:|---|---|
| 10 | Laplace BALD +0.029 (1.00); Core-set, relation-aware pairs +0.028 (1.00); ProbCover (pairs) +0.026 (1.00) | none |
| 20 | Core-set, relation-aware pairs +0.036 (1.00); Laplace BALD +0.034 (1.00); Fisher D-optimal (Active Reward Modeling) +0.032 (1.00) | none |
| 40 | Laplace BALD +0.043 (0.37); BALD x P(decisive) +0.039 (0.36); MaxHerding (pairs) +0.035 (1.00) | none |
| 60 | MaxHerding (pairs) +0.067 (0.02); Laplace BALD +0.043 (0.89); Graph cut (pairs) +0.038 (1.00) | MaxHerding (pairs) +0.067 |

**A single, AUC (re-tuned schedule)**

| Budget (judgments) | top 3 by mean gain over Random (Holm p within the budget) | variants with Holm p < 0.05 |
|---:|---|---|
| 10 | Laplace BALD +0.016 (0.57); Fisher D-optimal (Active Reward Modeling) +0.014 (0.16); BALD x P(decisive) +0.013 (0.57) | none |
| 20 | Deep-ensemble BALD x P(decisive) +0.015 (1.00); Fisher D-optimal (Active Reward Modeling) +0.015 (0.99); Laplace BALD +0.013 (1.00) | none |
| 40 | Laplace BALD +0.014 (0.29); BALD x P(decisive) +0.014 (0.34); Deep-ensemble BALD x P(decisive) +0.013 (1.00) | none |
| 60 | MaxHerding (pairs) +0.017 (0.11); Graph cut (pairs) +0.013 (0.78); BALD x P(decisive) +0.012 (1.00) | none |

**A sequential, log-loss (re-tuned schedule)**

| Budget (judgments) | top 3 by mean gain over Random (Holm p within the budget) | variants with Holm p < 0.05 |
|---:|---|---|
| 10 | Laplace BALD +0.024 (1.00); Core-set, relation-aware pairs +0.023 (1.00); ProbCover (pairs) +0.021 (1.00) | none |
| 20 | Laplace BALD +0.049 (0.27); ProbCover (pairs) +0.026 (1.00); BALD x P(decisive) +0.025 (1.00) | Uncertainty, all heads -0.050, Uncertainty, original code, all heads -0.050 |
| 40 | BALD x P(decisive) +0.047 (0.36); ProbCover (pairs) +0.042 (0.12); Laplace BALD +0.037 (0.89) | none |
| 60 | ProbCover (pairs) +0.044 (0.28); Core-set +0.043 (1.00); Core-set, relation-aware pairs +0.036 (1.00) | none |

**A sequential, AUC (re-tuned schedule)**

| Budget (judgments) | top 3 by mean gain over Random (Holm p within the budget) | variants with Holm p < 0.05 |
|---:|---|---|
| 10 | Laplace BALD +0.011 (1.00); Fisher D-optimal (Active Reward Modeling) +0.010 (1.00); BALD x P(decisive) +0.009 (1.00) | none |
| 20 | Laplace BALD +0.018 (0.20); Deep-ensemble BALD (8 heads) +0.015 (1.00); Fisher D-optimal (Active Reward Modeling) +0.012 (1.00) | Uncertainty + diversity, original code, all heads -0.019, Uncertainty, all heads -0.021, Uncertainty, original code, all heads -0.021 |
| 40 | ProbCover (pairs) +0.017 (0.27); Deep-ensemble BALD (8 heads) +0.016 (0.36); BALD x P(decisive) +0.014 (0.69) | none |
| 60 | ProbCover (pairs) +0.013 (0.34); TypiClust (pairs) +0.009 (1.00); Core-set +0.009 (1.00) | none |

**B single, log-loss (re-tuned schedule)**

| Budget (judgments) | top 3 by mean gain over Random (Holm p within the budget) | variants with Holm p < 0.05 |
|---:|---|---|
| 10 | BALD x P(decisive) +0.002 (1.00); Largest predicted gap -0.003 (1.00); Deep-ensemble BALD (8 heads) -0.004 (1.00) | Uncertainty + diversity, original code, all heads -0.037, Uncertainty, all heads -0.039, Uncertainty, original code, all heads -0.039, MC-dropout mutual info, original code, all heads -0.040, DPP (quality x diversity) -0.042, Uncertainty -0.043, Image-coverage uncertainty -0.048, MC-dropout variance, original code, all heads -0.058 |
| 20 | Core-set +0.010 (1.00); TypiClust (pairs) +0.009 (1.00); BALD x P(decisive) +0.007 (1.00) | Uncertainty -0.046, Uncertainty, original code, all heads -0.054, Uncertainty, all heads -0.054, MC-dropout variance, original code, all heads -0.057, MC-dropout mutual info, original code, all heads -0.066 |
| 40 | BALD x P(decisive) +0.030 (1.00); Core-set +0.027 (1.00); MaxHerding (pairs) +0.015 (1.00) | Uncertainty, all heads -0.042, Uncertainty, original code, all heads -0.042 |
| 60 | ProbCover (pairs) +0.017 (1.00); Gap + posterior std (DeltaUCB-style) +0.016 (1.00); MaxHerding (pairs) +0.011 (1.00) | Deep-ensemble BALD (8 heads) -0.072, Uncertainty + diversity -0.073, Image-coverage uncertainty -0.075, Deep-ensemble BALD x P(decisive) -0.080 |

**B single, AUC (re-tuned schedule)**

| Budget (judgments) | top 3 by mean gain over Random (Holm p within the budget) | variants with Holm p < 0.05 |
|---:|---|---|
| 10 | BALD x P(decisive) +0.003 (1.00); Graph cut (pairs) +0.001 (1.00); Deep-ensemble BALD x P(decisive) +0.001 (1.00) | MC-dropout mutual info, original code, all heads -0.019, Uncertainty, all heads -0.020, Uncertainty, original code, all heads -0.020, Uncertainty + diversity, original code, all heads -0.020, MC-dropout variance, original code, all heads -0.026 |
| 20 | TypiClust (pairs) +0.008 (1.00); BADGE (pairs) +0.003 (1.00); BALD x P(decisive) +0.001 (1.00) | Uncertainty, original code, all heads -0.022, Uncertainty, all heads -0.022, MC-dropout mutual info, original code, all heads -0.026, MC-dropout variance, original code, all heads -0.026 |
| 40 | BALD x P(decisive) +0.010 (1.00); Core-set +0.008 (1.00); BADGE (pairs) +0.007 (1.00) | Uncertainty, all heads -0.021, Uncertainty, original code, all heads -0.021 |
| 60 | BALD x P(decisive) +0.008 (1.00); ProbCover (pairs) +0.002 (1.00); MaxHerding (pairs) +0.002 (1.00) | none |

**B sequential, log-loss (re-tuned schedule)**

| Budget (judgments) | top 3 by mean gain over Random (Holm p within the budget) | variants with Holm p < 0.05 |
|---:|---|---|
| 10 | BALD x P(decisive) +0.026 (1.00); Largest predicted gap +0.021 (1.00); Deep-ensemble BALD (8 heads) +0.020 (1.00) | MC-dropout variance, original code, all heads -0.035 |
| 20 | BALD x P(decisive) +0.054 (0.91); Cluster-quota uncertainty, original code, all heads +0.040 (0.90); Gap + posterior std (DeltaUCB-style) +0.040 (1.00) | none |
| 40 | BALD x P(decisive) +0.056 (0.14); Core-set, relation-aware pairs +0.051 (0.24); TypiClust (pairs) +0.049 (0.11) | MC-dropout mutual info, original code, all heads -0.049 |
| 60 | BALD x P(decisive) +0.068 (0.06); Core-set, relation-aware pairs +0.054 (1.00); Cluster-Margin, original code, all heads +0.051 (1.00) | none |

**B sequential, AUC (re-tuned schedule)**

| Budget (judgments) | top 3 by mean gain over Random (Holm p within the budget) | variants with Holm p < 0.05 |
|---:|---|---|
| 10 | BALD x P(decisive) +0.008 (1.00); Graph cut (pairs) +0.006 (1.00); Deep-ensemble BALD x P(decisive) +0.006 (1.00) | MC-dropout variance, original code, all heads -0.020 |
| 20 | BALD x P(decisive) +0.017 (0.36); Graph cut (pairs) +0.012 (0.65); TypiClust (pairs) +0.010 (1.00) | none |
| 40 | BALD x P(decisive) +0.022 (0.02); Core-set, relation-aware pairs +0.017 (0.34); TypiClust (pairs) +0.015 (0.64) | BALD x P(decisive) +0.022, MC-dropout variance, original code, all heads -0.021, MC-dropout mutual info, original code, all heads -0.023 |
| 60 | BALD x P(decisive) +0.021 (0.00); Cluster-Margin, original code, all heads +0.013 (0.84); Graph cut (pairs) +0.012 (0.20) | BALD x P(decisive) +0.021 |

