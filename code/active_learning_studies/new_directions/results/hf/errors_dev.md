# HTR error analysis of `lap3_k10_st4` (dev seeds; 8 seeds x splits ['A', 'B'], 17424 stored predictions)

## 1. Accuracy / log-loss by type and split (row-weighted, bootstrap-by-seed 95% interval)

| split   | type   |   n_unique_judgments |   acc |    lo |    hi |    ll |
|:--------|:-------|---------------------:|------:|------:|------:|------:|
| A       | t1x1   |                   68 | 0.692 | 0.647 | 0.741 | 0.583 |
| A       | tc6x2  |                   46 | 0.85  | 0.828 | 0.872 | 0.447 |
| A       | t13    |                   81 | 0.961 | 0.938 | 0.978 | 0.221 |
| A       | htr    |                   38 | 0.934 | 0.9   | 0.967 | 0.194 |
| B       | t1x1   |                   43 | 0.706 | 0.572 | 0.819 | 0.535 |
| B       | tc6x2  |                   32 | 0.861 | 0.815 | 0.922 | 0.418 |
| B       | t13    |                   58 | 0.981 | 0.958 | 1     | 0.203 |
| B       | htr    |                   24 | 0.925 | 0.854 | 0.984 | 0.19  |

Comparison learner `aw8_all_st2lr0.5` (same label sets), HTR only:

| learner          | split   |   acc |    lo |    hi |    ll |
|:-----------------|:--------|------:|------:|------:|------:|
| aw8_all_st2lr0.5 | A       | 0.908 | 0.864 | 0.953 | 0.23  |
| aw8_all_st2lr0.5 | B       | 0.867 | 0.757 | 0.973 | 0.257 |
| lap3_k10_st4     | A       | 0.934 | 0.9   | 0.967 | 0.194 |
| lap3_k10_st4     | B       | 0.925 | 0.854 | 0.984 | 0.19  |

## 2. The no-learner rule 'the frame nearer to the ideal HTR images wins' on the same pooled HTR test rows

rule accuracy 0.974 vs learner 0.932 (same rows, both splits, all budgets).  By budget / split:

|           |   rule |   learner |   n |
|:----------|-------:|----------:|----:|
| ('A', 10) |  0.989 |     0.923 | 261 |
| ('A', 20) |  0.989 |     0.935 | 261 |
| ('A', 40) |  0.989 |     0.931 | 261 |
| ('A', 60) |  0.989 |     0.946 | 261 |
| ('B', 10) |  0.933 |     0.922 |  90 |
| ('B', 20) |  0.933 |     0.922 |  90 |
| ('B', 40) |  0.933 |     0.911 |  90 |
| ('B', 60) |  0.933 |     0.944 |  90 |

## 3. Per-judgment profile (decisive HTR judgments that appear in at least one test set; sorted by learner error rate)

39 distinct HTR judgments seen; error-rate distribution: always right (<5%) 36, mostly right (5-35%) 1, mixed (35-65%) 1, mostly wrong (65-95%) 0, always wrong (>95%) 1.

|   jid |   n_rows |   n_cells |   err_rate |   mean_margin |   mean_ll | rule_ok   |   sign | labeller   | conf      | cls_pair   | win   | lose   | same_run   |   temp1 |   temp2 |   d_pair |
|------:|---------:|----------:|-----------:|--------------:|----------:|:----------|-------:|:-----------|:----------|:-----------|:------|:-------|:-----------|--------:|--------:|---------:|
|    62 |       48 |         4 |      1     |        -1.612 |     1.801 | True      |     -1 | AJ         | none      | 1x1-c6x2   | c6x2  | 1x1    | False      |     165 |     620 |    1.787 |
|   101 |       96 |         8 |      0.479 |         0.033 |     0.679 | True      |     -1 | HL         | Confident | 1x1-RT13   | 1x1   | RT13   | False      |     443 |     857 |    0.736 |
|   322 |       24 |         2 |      0.083 |         0.601 |     0.447 | True      |     -1 | AJ         | none      | 1x1-HTR    | HTR   | 1x1    | True       |     249 |    1092 |    0.129 |
|    41 |       36 |         3 |      0     |         3.858 |     0.021 | True      |     -1 | HL         | Confident | HTR-RT13   | HTR   | RT13   | False      |     579 |     498 |    0.926 |
|    45 |       36 |         3 |      0     |         3.265 |     0.038 | True      |      1 | AJ         | none      | HTR-RT13   | HTR   | RT13   | False      |    1085 |     576 |    0.828 |
|    37 |       36 |         3 |      0     |         4.248 |     0.015 | True      |     -1 | AJ         | none      | HTR-RT13   | HTR   | RT13   | False      |     655 |     424 |    1.196 |
|    25 |       24 |         2 |      0     |         0.453 |     0.493 | True      |      1 | HL         | Confident | RT13-RT13  | RT13  | RT13   | False      |     873 |     981 |    0.324 |
|    91 |       36 |         3 |      0     |         2.194 |     0.107 | True      |     -1 | AJ         | Confident | HTR-RT13   | HTR   | RT13   | False      |     441 |    1077 |    1.092 |
|    66 |       48 |         4 |      0     |         1.99  |     0.13  | True      |     -1 | SY         | none      | HTR-RT13   | HTR   | RT13   | False      |     293 |    1024 |    1.038 |
|   176 |       48 |         4 |      0     |         1.039 |     0.306 | True      |     -1 | SY         | none      | 1x1-1x1    | 1x1   | 1x1    | False      |     610 |     871 |    0.47  |
|   153 |       36 |         3 |      0     |         3.178 |     0.041 | True      |      1 | AJ         | Confident | HTR-RT13   | HTR   | RT13   | False      |     767 |     536 |    0.793 |
|   200 |       48 |         4 |      0     |         3.194 |     0.041 | True      |      1 | SY         | none      | HTR-RT13   | HTR   | RT13   | False      |     960 |     668 |    1.058 |
|   221 |       36 |         3 |      0     |         3.824 |     0.022 | True      |     -1 | AJ         | Confident | 1x1-HTR    | HTR   | 1x1    | False      |     782 |     540 |    0.77  |
|   234 |       60 |         5 |      0     |         2.351 |     0.096 | True      |     -1 | AJ         | none      | 1x1-HTR    | HTR   | 1x1    | False      |     843 |     993 |    0.62  |
|   192 |       48 |         4 |      0     |         2.931 |     0.053 | True      |     -1 | AJ         | none      | HTR-RT13   | HTR   | RT13   | False      |     641 |     985 |    0.804 |
|   247 |       12 |         1 |      0     |         3.25  |     0.04  | True      |      1 | HL         | none      | 1x1-HTR    | HTR   | 1x1    | False      |     944 |     881 |    0.533 |
|   266 |       24 |         2 |      0     |         3.702 |     0.026 | True      |      1 | HL         | Confident | 1x1-HTR    | HTR   | 1x1    | False      |     621 |     935 |    0.381 |
|   291 |       36 |         3 |      0     |         4.002 |     0.019 | True      |     -1 | AJ         | none      | HTR-RT13   | HTR   | RT13   | False      |     938 |     766 |    1.598 |
|   275 |       48 |         4 |      0     |         4.119 |     0.017 | True      |     -1 | AJ         | none      | HTR-RT13   | HTR   | RT13   | False      |     915 |     184 |    1.199 |
|   294 |       48 |         4 |      0     |         4.243 |     0.015 | True      |      1 | AJ         | none      | HTR-RT13   | HTR   | RT13   | False      |     581 |     937 |    1.603 |

Share of all HTR errors contributed by the 5 / 10 worst judgments: 1.00 / 1.00.

## 4. Where do failures live? (HTR; accuracy by stratum)

### by `cls_pair`
| cls_pair   |   n_judgments |   n_rows |   rate |    lo |   hi |    ll |
|:-----------|--------------:|---------:|-------:|------:|-----:|------:|
| 1x1-1x1    |             1 |       48 |  1     | 1     | 1    | 0.306 |
| 1x1-HTR    |            13 |      360 |  0.994 | 0.984 | 1    | 0.124 |
| 1x1-RT13   |             1 |       96 |  0.521 | 0.333 | 0.62 | 0.679 |
| 1x1-c6x2   |             1 |       48 |  0     | 0     | 0    | 1.801 |
| HTR-HTR    |             1 |       24 |  1     | 1     | 1    | 0.459 |
| HTR-RT13   |            21 |      804 |  1     | 1     | 1    | 0.046 |
| RT13-RT13  |             1 |       24 |  1     | 1     | 1    | 0.493 |

### by `winner_is`
| winner_is       |   n_judgments |   n_rows |   rate |    lo |   hi |    ll |
|:----------------|--------------:|---------:|-------:|------:|-----:|------:|
| 1x1 beats 1x1   |             1 |       48 |  1     | 1     | 1    | 0.306 |
| 1x1 beats RT13  |             1 |       96 |  0.521 | 0.333 | 0.62 | 0.679 |
| HTR beats 1x1   |            13 |      360 |  0.994 | 0.984 | 1    | 0.124 |
| HTR beats HTR   |             1 |       24 |  1     | 1     | 1    | 0.459 |
| HTR beats RT13  |            21 |      804 |  1     | 1     | 1    | 0.046 |
| RT13 beats RT13 |             1 |       24 |  1     | 1     | 1    | 0.493 |
| c6x2 beats 1x1  |             1 |       48 |  0     | 0     | 0    | 1.801 |

### by `labeller`
| labeller   |   n_judgments |   n_rows |   rate |    lo |    hi |    ll |
|:-----------|--------------:|---------:|-------:|------:|------:|------:|
| AJ         |            23 |      804 |  0.938 | 0.886 | 0.983 | 0.178 |
| HL         |            11 |      408 |  0.887 | 0.846 | 0.931 | 0.249 |
| SY         |             5 |      192 |  1     | 1     | 1     | 0.134 |

### by `confidence`
| confidence   |   n_judgments |   n_rows |   rate |    lo |    hi |    ll |
|:-------------|--------------:|---------:|-------:|------:|------:|------:|
| Confident    |            13 |      456 |  0.899 | 0.854 | 0.943 | 0.221 |
| none         |            26 |      948 |  0.947 | 0.913 | 0.985 | 0.179 |

### by `same_run`
| same_run   |   n_judgments |   n_rows |   rate |    lo |   hi |    ll |
|:-----------|--------------:|---------:|-------:|------:|-----:|------:|
| False      |            32 |     1212 |  0.922 | 0.885 | 0.96 | 0.186 |
| True       |             7 |      192 |  0.99  | 0.972 | 1    | 0.233 |

### by `temp_bin`
| temp_bin    |   n_judgments |   n_rows |   rate |    lo |    hi |    ll |
|:------------|--------------:|---------:|-------:|------:|------:|------:|
| (0, 600]    |             2 |       48 |  1     | 1     | 1     | 0.031 |
| (600, 800]  |             6 |      216 |  0.778 | 0.643 | 0.923 | 0.424 |
| (800, 900]  |             9 |      360 |  0.872 | 0.797 | 0.93  | 0.267 |
| (900, 2000] |            22 |      780 |  0.997 | 0.993 | 1     | 0.104 |

### by `anchor_bin`
| anchor_bin    |   n_judgments |   n_rows |   rate |    lo |    hi |    ll |
|:--------------|--------------:|---------:|-------:|------:|------:|------:|
| (-1.0, 0.001] |            15 |      552 |  0.913 | 0.818 | 0.979 | 0.186 |
| (0.001, 0.01] |            12 |      396 |  1     | 1     | 1     | 0.111 |
| (0.01, 0.03]  |             6 |      192 |  1     | 1     | 1     | 0.127 |
| (0.03, 9.0]   |             6 |      264 |  0.818 | 0.708 | 0.871 | 0.376 |

### by `far_anchor_bin`
| far_anchor_bin   |   n_judgments |   n_rows |   rate |    lo |    hi |    ll |
|:-----------------|--------------:|---------:|-------:|------:|------:|------:|
| (-1.0, 0.001]    |             1 |       48 |  1     | 1     | 1     | 0.012 |
| (0.001, 0.01]    |             7 |      240 |  1     | 1     | 1     | 0.027 |
| (0.01, 0.03]     |             5 |      144 |  1     | 1     | 1     | 0.125 |
| (0.03, 9.0]      |            26 |      972 |  0.901 | 0.849 | 0.951 | 0.253 |

### by `d_htr_gap`
| d_htr_gap    |   n_judgments |   n_rows |   rate |    lo |   hi |    ll |
|:-------------|--------------:|---------:|-------:|------:|-----:|------:|
| (-1.0, 0.05] |             7 |      204 |  0.99  | 0.972 | 1    | 0.144 |
| (0.05, 0.1]  |             7 |      216 |  1     | 1     | 1    | 0.106 |
| (0.1, 0.2]   |             7 |      228 |  1     | 1     | 1    | 0.098 |
| (0.2, 0.4]   |             9 |      384 |  0.755 | 0.622 | 0.87 | 0.461 |
| (0.4, 9.0]   |             9 |      372 |  1     | 1     | 1    | 0.051 |

### by `rule_agree`
| rule_agree   |   n_judgments |   n_rows |   rate |    lo |    hi |    ll |
|:-------------|--------------:|---------:|-------:|------:|------:|------:|
| rule right   |            38 |     1368 |   0.93 | 0.897 | 0.964 | 0.197 |
| rule wrong   |             1 |       36 |   1    | 1     | 1     | 0.036 |

### by `budget_s`
|   budget_s |   n_judgments |   n_rows |   rate |    lo |    hi |    ll |
|-----------:|--------------:|---------:|-------:|------:|------:|------:|
|         10 |            39 |      351 |  0.923 | 0.891 | 0.956 | 0.196 |
|         20 |            39 |      351 |  0.932 | 0.894 | 0.968 | 0.19  |
|         40 |            39 |      351 |  0.926 | 0.893 | 0.96  | 0.196 |
|         60 |            39 |      351 |  0.946 | 0.915 | 0.978 | 0.188 |

### by `split`
| split   |   n_judgments |   n_rows |   rate |    lo |    hi |    ll |
|:--------|--------------:|---------:|-------:|------:|------:|------:|
| A       |            38 |     1044 |  0.934 | 0.9   | 0.967 | 0.194 |
| B       |            24 |      360 |  0.925 | 0.854 | 0.984 | 0.19  |

### learner vs rule (rows)
| learner correct   |   False |   True |
|:------------------|--------:|-------:|
| False             |       0 |     96 |
| True              |      36 |   1272 |

### label-neighbour conflict of the judgments (nearest other HTR comparison, oriented): conflict = opposite label
| err_rate      |   n |   conflicts |   median_nn_D |
|:--------------|----:|------------:|--------------:|
| (-0.01, 0.05] |  36 |           0 |         0.045 |
| (0.05, 0.35]  |   1 |           0 |         0.029 |
| (0.35, 0.65]  |   1 |           0 |         0.205 |
| (0.65, 1.0]   |   1 |           1 |         0.145 |
