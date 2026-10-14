# FT development log (append-only; seeds 10300–10319; every learner variant that was run is listed)

## diag1 — seeds 10300–10303, split A, budget 20 (4 seeds; stopped before budget 40); gain vs `lap3_k10_st4` (family P, stage 2 = 40 mini-batch steps)
| learner | AUC | acc | far AUC | far acc |
|---|---|---|---|---|
| frozen control, aug, head lr×1 | −0.026 | −0.015 | −0.105 | −0.047 |
| frozen control, no aug, head lr×1 | +0.000 | −0.012 | −0.016 | −0.036 |
| frozen control, aug, head lr×0.2 | −0.054 | −0.091 | −0.202 | −0.186 |
| frozen control, no aug, head lr×0.2 | +0.004 | −0.003 | −0.036 | −0.020 |
| fine-tuned lr 3e-5, aug, head lr×0.2 | −0.013 | −0.029 | −0.155 | −0.157 |
| fine-tuned lr 3e-5, no aug, head lr×0.2 | +0.006 | +0.011 | −0.004 | +0.035 |
| fine-tuned lr 1e-4, aug, head lr×0.2 | +0.001 | +0.004 | −0.049 | −0.059 |

Reading: feature-map augmentation (flip + ±1-cell shift + intensity jitter on the 14×14 layer-3 maps) is clearly harmful for the head even with the encoder frozen (frozen control with augmentation loses 0.03–0.05 AUC and 0.10–0.20 far-AUC against the incumbent): augmented maps leave the clean-feature distribution that evaluation uses. Augmentation is OFF by default from here; head-lr multiplier 0.2 becomes the default (frozen no-aug control ≈ incumbent). Cost per fit (1 thread): fine-tuned ≈ 82 s, frozen stage 2 ≈ 26 s, incumbent ≈ 5 s.
