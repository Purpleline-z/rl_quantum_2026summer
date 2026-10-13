Mean over contexts: split A, seeds [9200, 9201, 9202].

| edge set | edges | share in kNN graph | median cos distance | p90 cos distance | share distance > 0.05 | share |dT| > 20C | median feature-neighbour rank |
|---|---|---|---|---|---|---|---|
| kNN graph (k=10), all edges | 6770 | 1.000 | 0.001 | 0.021 | 0.044 | 0.307 | 5.000 |
| chain +-1 | 883 | 0.938 | 0.000 | 0.005 | 0.013 | 0.064 | 2.000 |
| chain +-2 | 1762 | 0.920 | 0.000 | 0.008 | 0.019 | 0.091 | 3.000 |
| chain +-5 | 4380 | 0.833 | 0.001 | 0.021 | 0.050 | 0.213 | 5.000 |
| chain +-5, only edges NOT in the kNN graph | 733 | 0.000 | 0.012 | 0.217 | 0.240 | 0.453 | 17.667 |
| temperature-bin partners | 7522 | 0.205 | 0.489 | 1.721 | 0.704 | 0.000 | 339.000 |
| chain +-2 shuffled within runs (control) | 1762 | 0.033 | 0.257 | 1.793 | 0.774 | 0.936 | 191.500 |
| chain +-2 locally shuffled (control) | 1762 | 0.225 | 0.008 | 0.483 | 0.282 | 0.710 | 27.667 |
| random kNN edges, same number as chain +-2 (control) | 1762 | 1.000 | 0.001 | 0.021 | 0.044 | 0.320 | 5.333 |
