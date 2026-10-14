#### Distances (1 - cos of centred features; medians) (bank `v1_mean`)

| type | n_syn | syn->nearest unlabelled frame | syn->nearest ideal, same type | syn->nearest ideal, other type | REAL ideal->nearest syn (same type) | REAL ideal->nearest other ideal (same type) | REAL ideal->nearest unlabelled frame | n_real_ideal |
|---|---|---|---|---|---|---|---|---|
| 1x1 | 400 | 0.015 | 0.049 | 0.220 | 0.032 | 0.000 | 0.168 | 41 |
| c6x2 | 400 | 0.050 | 0.262 | 0.129 | 0.219 | 0.001 | 0.332 | 42 |
| RT13 | 400 | 0.049 | 0.122 | 0.093 | 0.081 | 0.001 | 0.063 | 30 |
| HTR | 400 | 0.016 | 0.237 | 0.052 | 0.134 | 0.000 | 0.144 | 19 |

Precision (synthetic -> nearest real reference, median) = 0.031; recall (real unlabelled frame -> nearest synthetic, median) = 0.031; for reference the median nearest-neighbour distance between real unlabelled frames is 0.0001.

**Real-vs-synthetic classifier** (logistic regression on 512-d features, 5-fold CV, 972 real [unlabelled frames + masked clean ideal images] vs 1600 synthetic): AUC = 0.995 (0.5 = indistinguishable).

**Type classifier trained on synthetic features only, applied to the 132 clean real ideal images (masked):** accuracy = 0.470 (majority-class rate 0.318); real-only 5-fold reference = 0.939.

| real ideal (true) | 1x1 | HTR | RT13 | c6x2 |
|---|---|---|---|---|
| 1x1 | 38 | 3 | 0 | 0 |
| HTR | 2 | 17 | 0 | 0 |
| RT13 | 16 | 6 | 5 | 3 |
| c6x2 | 40 | 0 | 0 | 2 |