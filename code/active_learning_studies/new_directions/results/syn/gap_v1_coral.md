#### Distances (1 - cos of centred features; medians) (bank `v1_coral`)

| type | n_syn | syn->nearest unlabelled frame | syn->nearest ideal, same type | syn->nearest ideal, other type | REAL ideal->nearest syn (same type) | REAL ideal->nearest other ideal (same type) | REAL ideal->nearest unlabelled frame | n_real_ideal |
|---|---|---|---|---|---|---|---|---|
| 1x1 | 400 | 0.009 | 0.032 | 0.203 | 0.057 | 0.000 | 0.168 | 41 |
| c6x2 | 400 | 0.022 | 0.302 | 0.078 | 0.251 | 0.001 | 0.332 | 42 |
| RT13 | 400 | 0.022 | 0.145 | 0.060 | 0.066 | 0.001 | 0.063 | 30 |
| HTR | 400 | 0.010 | 0.218 | 0.033 | 0.170 | 0.000 | 0.144 | 19 |

Precision (synthetic -> nearest real reference, median) = 0.013; recall (real unlabelled frame -> nearest synthetic, median) = 0.016; for reference the median nearest-neighbour distance between real unlabelled frames is 0.0001.

**Real-vs-synthetic classifier** (logistic regression on 512-d features, 5-fold CV, 972 real [unlabelled frames + masked clean ideal images] vs 1600 synthetic): AUC = 0.750 (0.5 = indistinguishable).

**Type classifier trained on synthetic features only, applied to the 132 clean real ideal images (masked):** accuracy = 0.341 (majority-class rate 0.318); real-only 5-fold reference = 0.939.

| real ideal (true) | 1x1 | HTR | RT13 |
|---|---|---|---|
| 1x1 | 41 | 0 | 0 |
| HTR | 17 | 2 | 0 |
| RT13 | 17 | 11 | 2 |
| c6x2 | 41 | 1 | 0 |