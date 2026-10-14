# Pre-registration GP2 (amendment, frozen before any cell of confirmation 2 exists) — ensemble of the calibrated GP and the calibrated `lap3_k10_st4`

Written 2026-10-14 (container clock UTC; see the commit time of this file). **Origin of the hypothesis (stated honestly):** `PREREGISTRATION_GP1.md` was frozen before confirmation 1 (seeds 11030–11064); after confirmation 1 had been analysed (GP1 verdicts: not met), the combination below was tried on the confirmation-1 cells as an *exploratory* analysis and looked consistently positive (log-loss +0.007 / +0.012, AUC +0.002 / +0.004 over the calibrated reference, splits A / B; p ≤ 0.005). It is therefore **not** counted as confirmed by confirmation 1. **Its pre-registered test is confirmation 2 only (seeds 11065–11099, splits A and B, the same cells that the GP1 chain produces; no confirmation-2 cell existed when this file was committed).** Specification: `GP2_spec.json`; code: `gp_tables.py ensemble` (variant `E` in `variants_of`).

## Candidate
**E** = 0.5 × (logit gap of `lap3_k10_st4` divided by the per-type constants of the calibration study (1x1 1.352, c(6x2) 0.682, RT13 0.293, HTR 0.630) + logit gap of the hand-set GP `gp_fix3` divided by its development constants (1x1 0.886, c(6x2) 0.515, RT13 0.358, HTR 0.421)); both constants sets are fixed numbers fitted on development seeds from out-of-fold logits of labelled judgments; same label sets and draws as GP1. Reference: the calibrated `lap3_k10_st4` (`lap3_k10_st4+Tcal`).

## Tests and success rule
Endpoint as in GP1 (splits A and B, budgets 10/20/40/60 × draws 0, 1, per-seed means, two-sided Wilcoxon over the 35 seeds). **Family (Holm over 4): log-loss and AUC of E minus the calibrated reference × splits A and B.** Success: all four gains positive, ≥ 3 of the 4 Holm p < 0.05, and the accuracy difference's lower 95 % bootstrap bound ≥ −0.01 in both splits. Descriptive: E against the calibrated GP, raw `lap3_k10_st4` and raw GP; anchor-far AUC / log-loss; ECE.
Expectation stated in advance: positive log-loss gain of 0.005–0.012 and AUC gain of 0.002–0.004 (the confirmation-1 values), with success more likely than not but not certain; accuracy ≈ 0.

## Limits
Same 168 pair groups re-split; the success of E would say that two partly complementary learners averaged after per-type calibration are slightly better than the calibrated Laplacian learner on this data set, not that either model generalises to new growth runs; the amount (≈ 0.01 log-loss, ≈ 0.003 AUC) is small. The ensemble costs one `lap3_k10_st4` fit (4 s) plus one GP fit (0.3 s).
