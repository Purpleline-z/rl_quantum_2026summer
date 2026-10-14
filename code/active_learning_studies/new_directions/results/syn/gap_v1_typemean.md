#### Distances (1 - cos of centred features; medians) (bank `v1_typemean`)

| type | n_syn | syn->nearest unlabelled frame | syn->nearest ideal, same type | syn->nearest ideal, other type | REAL ideal->nearest syn (same type) | REAL ideal->nearest other ideal (same type) | REAL ideal->nearest unlabelled frame | n_real_ideal |
|---|---|---|---|---|---|---|---|---|
| 1x1 | 400 | 0.053 | 0.041 | 0.079 | 0.020 | 0.000 | 0.168 | 41 |
| c6x2 | 400 | 0.089 | 0.166 | 0.096 | 0.069 | 0.001 | 0.332 | 42 |
| RT13 | 400 | 0.045 | 0.089 | 0.078 | 0.039 | 0.001 | 0.063 | 30 |
| HTR | 400 | 0.113 | 0.140 | 0.102 | 0.035 | 0.000 | 0.144 | 19 |

Precision (synthetic -> nearest real reference, median) = 0.048; recall (real unlabelled frame -> nearest synthetic, median) = 0.034; for reference the median nearest-neighbour distance between real unlabelled frames is 0.0001.

**Real-vs-synthetic classifier** (logistic regression on 512-d features, 5-fold CV, 972 real [unlabelled frames + masked clean ideal images] vs 1600 synthetic): AUC = 0.971 (0.5 = indistinguishable).

**Type classifier trained on synthetic features only, applied to the 132 clean real ideal images (masked):** accuracy = 0.856 (majority-class rate 0.318); real-only 5-fold reference = 0.939.

| real ideal (true) | 1x1 | HTR | RT13 | c6x2 |
|---|---|---|---|---|
| 1x1 | 30 | 3 | 0 | 8 |
| HTR | 1 | 17 | 0 | 1 |
| RT13 | 0 | 3 | 27 | 0 |
| c6x2 | 0 | 0 | 3 | 39 |