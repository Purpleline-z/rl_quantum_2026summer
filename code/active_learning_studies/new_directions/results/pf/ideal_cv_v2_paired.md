| clf | feature set | mean diff vs SimCLR (accuracy per image) | 95% CI | Wilcoxon p (uncorrected) | n images |
|---|---|---|---|---|---|
| 1nn | SimCLR (16-bit decoded) + pfa | -0.004 | [-0.039, +0.031] | +0.858 | 140 |
| 1nn | SimCLR + pfa (z-scored concat) | +0.012 | [-0.017, +0.042] | +0.405 | 140 |
| 1nn | SimCLR, 16-bit ideals decoded correctly | -0.032 | [-0.066, -0.000] | +0.050 | 140 |
| 1nn | physics pfa | -0.000 | [-0.044, +0.044] | +0.940 | 140 |
| 1nn | physics pfb (peak table) | +0.019 | [-0.028, +0.067] | +0.404 | 140 |
| 1nn | physics pfc (all blocks) | +0.024 | [-0.020, +0.068] | +0.276 | 140 |
| 1nn | random projection of SimCLR (dim of pfa) | -0.005 | [-0.027, +0.017] | +0.599 | 140 |
| logreg | SimCLR (16-bit decoded) + pfa | +0.012 | [-0.018, +0.043] | +0.280 | 140 |
| logreg | SimCLR + pfa (z-scored concat) | +0.014 | [-0.013, +0.041] | +0.300 | 140 |
| logreg | SimCLR, 16-bit ideals decoded correctly | -0.019 | [-0.055, +0.013] | +0.236 | 140 |
| logreg | physics pfa | -0.003 | [-0.037, +0.033] | +0.812 | 140 |
| logreg | physics pfb (peak table) | +0.009 | [-0.034, +0.051] | +0.599 | 140 |
| logreg | physics pfc (all blocks) | +0.016 | [-0.018, +0.052] | +0.393 | 140 |
| logreg | random projection of SimCLR (dim of pfa) | -0.043 | [-0.078, -0.012] | +0.010 | 140 |
