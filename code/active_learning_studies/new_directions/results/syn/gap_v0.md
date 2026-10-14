#### Distances (1 - cos of centred features; medians) (bank `t0`)

| type | n_syn | syn->nearest unlabelled frame | syn->nearest ideal, same type | syn->nearest ideal, other type | REAL ideal->nearest syn (same type) | REAL ideal->nearest other ideal (same type) | REAL ideal->nearest unlabelled frame | n_real_ideal |
|---|---|---|---|---|---|---|---|---|
| 1x1 | 40 | 0.046 | 1.574 | 0.060 | 0.618 | 0.000 | 0.168 | 41 |
| c6x2 | 40 | 0.069 | 0.096 | 0.091 | 0.444 | 0.001 | 0.332 | 42 |
| RT13 | 40 | 0.120 | 0.147 | 0.155 | 0.164 | 0.001 | 0.063 | 30 |
| HTR | 40 | 0.052 | 0.091 | 0.051 | 0.849 | 0.000 | 0.144 | 19 |

Precision (synthetic -> nearest real reference, median) = 0.070; recall (real unlabelled frame -> nearest synthetic, median) = 0.159; for reference the median nearest-neighbour distance between real unlabelled frames is 0.0001.

**Real-vs-synthetic classifier** (logistic regression on 512-d features, 5-fold CV, 972 real [unlabelled frames + masked clean ideal images] vs 160 synthetic): AUC = 1.000 (0.5 = indistinguishable).

**Type classifier trained on synthetic features only, applied to the 132 clean real ideal images (masked):** accuracy = 0.311 (majority-class rate 0.318); real-only 5-fold reference = 0.939.

| real ideal (true) | 1x1 | HTR | c6x2 |
|---|---|---|---|
| 1x1 | 41 | 0 | 0 |
| HTR | 19 | 0 | 0 |
| RT13 | 22 | 6 | 2 |
| c6x2 | 42 | 0 | 0 |