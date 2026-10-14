#### Distances (1 - cos of centred features; medians) (bank `v1`)

| type | n_syn | syn->nearest unlabelled frame | syn->nearest ideal, same type | syn->nearest ideal, other type | REAL ideal->nearest syn (same type) | REAL ideal->nearest other ideal (same type) | REAL ideal->nearest unlabelled frame | n_real_ideal |
|---|---|---|---|---|---|---|---|---|
| 1x1 | 400 | 0.039 | 0.100 | 0.224 | 0.050 | 0.000 | 0.168 | 41 |
| c6x2 | 400 | 0.058 | 0.170 | 0.086 | 0.309 | 0.001 | 0.332 | 42 |
| RT13 | 400 | 0.053 | 0.066 | 0.080 | 0.028 | 0.001 | 0.063 | 30 |
| HTR | 400 | 0.045 | 0.296 | 0.097 | 0.101 | 0.000 | 0.144 | 19 |

Precision (synthetic -> nearest real reference, median) = 0.042; recall (real unlabelled frame -> nearest synthetic, median) = 0.062; for reference the median nearest-neighbour distance between real unlabelled frames is 0.0001.

**Real-vs-synthetic classifier** (logistic regression on 512-d features, 5-fold CV, 972 real [unlabelled frames + masked clean ideal images] vs 1600 synthetic): AUC = 1.000 (0.5 = indistinguishable).

**Type classifier trained on synthetic features only, applied to the 132 clean real ideal images (masked):** accuracy = 0.614 (majority-class rate 0.318); real-only 5-fold reference = 0.939.

| real ideal (true) | 1x1 | HTR | RT13 | c6x2 |
|---|---|---|---|---|
| 1x1 | 33 | 0 | 0 | 8 |
| HTR | 1 | 1 | 0 | 17 |
| RT13 | 4 | 2 | 7 | 17 |
| c6x2 | 1 | 0 | 1 | 40 |