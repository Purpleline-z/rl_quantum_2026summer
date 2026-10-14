#### Distances (1 - cos of centred features; medians) (bank `v2`)

| type | n_syn | syn->nearest unlabelled frame | syn->nearest ideal, same type | syn->nearest ideal, other type | REAL ideal->nearest syn (same type) | REAL ideal->nearest other ideal (same type) | REAL ideal->nearest unlabelled frame | n_real_ideal |
|---|---|---|---|---|---|---|---|---|
| 1x1 | 400 | 0.036 | 0.072 | 0.227 | 0.043 | 0.000 | 0.168 | 41 |
| c6x2 | 400 | 0.049 | 0.063 | 0.048 | 0.270 | 0.001 | 0.332 | 42 |
| RT13 | 400 | 0.051 | 0.065 | 0.076 | 0.037 | 0.001 | 0.063 | 30 |
| HTR | 400 | 0.055 | 0.201 | 0.084 | 0.104 | 0.000 | 0.144 | 19 |

Precision (synthetic -> nearest real reference, median) = 0.040; recall (real unlabelled frame -> nearest synthetic, median) = 0.064; for reference the median nearest-neighbour distance between real unlabelled frames is 0.0001.

**Real-vs-synthetic classifier** (logistic regression on 512-d features, 5-fold CV, 972 real [unlabelled frames + masked clean ideal images] vs 1600 synthetic): AUC = 1.000 (0.5 = indistinguishable).

**Type classifier trained on synthetic features only, applied to the 132 clean real ideal images (masked):** accuracy = 0.227 (majority-class rate 0.318); real-only 5-fold reference = 0.939.

| real ideal (true) | 1x1 | HTR | RT13 | c6x2 |
|---|---|---|---|---|
| 1x1 | 6 | 28 | 7 | 0 |
| HTR | 17 | 1 | 1 | 0 |
| RT13 | 2 | 3 | 23 | 2 |
| c6x2 | 0 | 1 | 41 | 0 |