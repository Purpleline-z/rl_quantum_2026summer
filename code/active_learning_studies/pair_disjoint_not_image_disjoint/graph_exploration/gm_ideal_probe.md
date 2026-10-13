| learner | (1x1) one-vs-rest AUC | c(6x2) one-vs-rest AUC | RT13 one-vs-rest AUC | HTR one-vs-rest AUC | RT13 score: RT13 vs HTR images AUC | seeds |
|---|---|---|---|---|---|---|
| aw8_all | 0.9953 | 0.9887 | 0.9588 | 0.9443 | 0.8957 | 20 |
| lap3_k10-like (lambda 3) | 0.9950 | 0.9884 | 0.9633 | 0.9648 | 0.9171 | 20 |
| uniform lambda 10 | 0.9944 | 0.9891 | 0.9449 | 0.9574 | 0.8571 | 20 |
| RT13 off (C1) | 0.9944 | 0.9894 | 0.9595 | 0.9483 | 0.8929 | 20 |
| shuffled graph (lambda 10) | 0.9722 | 0.9584 | 0.9122 | 0.8913 | 0.7829 | 20 |

(held-out outer-test ideal images only; budget 20, one random label set per seed; descriptive, development seeds)
