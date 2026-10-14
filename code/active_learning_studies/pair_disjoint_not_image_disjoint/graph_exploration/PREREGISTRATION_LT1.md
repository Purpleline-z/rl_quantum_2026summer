# Pre-registration LT1 (latent-type study) — committed before any confirmation seed (10730–10799) was run

Written 2026-10-14 ~05:30 UTC (container clock), about 2 h 45 min into the session — earlier than the 4.5 h of the schedule, because every development screen was flat (no further screen was expected to change the choice; the saved time goes to confirmation 2 and the time-blocked split). Code: `new_directions/lt_model.py`, `lt_learner.py`, `lt_run.py`; tables: `graph_exploration/lt_tables.py`; chain: `scripts_lt/lt_chain.sh`. Same 168 pair groups / 150 ideal images as all earlier work (re-split by seed) → p-values optimistic.

## What development (seeds 10700–10719; 20 seeds, splits A/B, 1 label set per budget) showed — observations, not evidence
Baseline `lap3_k10_st4`; AUC gain over it (standard A / B; anchor-far A / B), full tables in `PROGRESS_LOG_latent_type.md`:
- soft pseudo-anchors: `pa0.25_em` −0.006 / −0.002 (far −0.006 / +0.010); `pa1_em` −0.014 / −0.009; anchor-far and confidence-thresholded variants −0.001…−0.020. Hurts in the standard endpoint.
- answerability weighting `lik_em` +0.0005 / +0.0006 (far +0.004 / +0.008); `lik_prior` 0.000 / +0.003.
- auxiliary target `aux0.5_em` −0.003 / +0.002 (far −0.003 / +0.012, p = 0.011); `aux0.1_em` +0.000 / +0.002 (far +0.002 / +0.008); `aux1_em` −0.005 / +0.001; graph-wide `auxall` variants ≈ 0 (−0.004…+0.001; far B +0.004…+0.008).
- controls: random memberships hurt clearly (`aux0.5_random` −0.015 / −0.009); the graph-prior-only versions are about as good as the EM versions (`aux0.5_prior` −0.004 / +0.001); the shuffled prior gives ≈ 0 or slightly negative.
- inference-time use of the latent model (d + λ·log-odds from the table and the prior memberships of the two test images): worse than baseline for λ ≥ 0.5.
- leave-anchors-out (D2): type recovery of hidden anchors 76.3 % (graph prior) vs 76.6–77.0 % (prior + outcomes); random posteriors 75–76 %; shuffled prior 27 % (chance).
Expectation stated in advance: the success rule below is unlikely to be met; the confirmation quantifies a small or null effect and the comparison with the graph-prior-only control.

## Frozen candidates (4) and controls
Candidates (all = `lap3_k10_st4` + the stated use of the latent-type posterior, EM posteriors, ρ = 0.15, pseudo-count scale 1): **C1 `lt_aux0.5_em`**, **C2 `lt_aux0.1_em`**, **C3 `lt_lik_em`**, **C4 `lt_pa0.25_em`** (one per integration channel: auxiliary target ×2 weights, answerability weighting, soft pseudo-anchors).
Reference: `lap3_k10_st4`. Controls: graph-prior-only (no outcome likelihood, same downstream use) `lt_aux0.5_prior`, `lt_aux0.1_prior`, `lt_lik_prior`, `lt_pa0.25_prior`; random memberships `lt_aux0.5_random`; shuffled prior `lt_aux0.5_shufprior`; tuned graph-free `aw8_all_st2lr0.5` (also compared with `lap3_k10_st4`).

## Evaluation
Seeds: confirmation 1 = 10730–10764 (35), confirmation 2 = 10765–10799 (35, independent); each run once; splits A and B; budgets 10/20/40/60 judgments; 1 random label set per budget (identical for all arms); per seed the mean over the 4 budgets. Endpoints: held-out pair-group AUC, accuracy, log-loss (log-loss reported, not in the rule). Anchor-far stratum: test judgments whose two images are both > 0.01 cosine distance from every typed ideal image (`anc_all_min` of `rt_a1b.py`, the definition of PREREGISTRATION_RT1). Time-blocked split (`rt_blocked.py`, 100-frame blocks, 25-frame buffer; 7 arms: baseline, aw8 control, C1–C4, `lt_aux0.5_prior`) reported for AUC and accuracy. Per-seed paired gains over `lap3_k10_st4`, two-sided Wilcoxon over seeds, paired bootstrap CI.

## Test families and success rule (per candidate)
- Standard family: 4 candidates × {A, B} × {AUC, accuracy} = 16 tests, Holm-corrected. Anchor-far family: the same 16 tests on the far stratum, Holm-corrected separately.
- **Success of a candidate** (all must hold, in confirmation 1; confirmation 2 is reported with the same rule as the independent replication): (a) standard: all four gains > 0 and ≥ 3 of 4 Holm p < 0.05; (b) anchor-far: gain > 0 in both splits for AUC and accuracy (all four) — the question of the task is the far stratum; ≥ 3 of 4 Holm p < 0.05 in the far family is reported as the stronger reading; (c) better than its graph-prior-only control: all four paired gains (candidate − control; A/B × AUC/accuracy) > 0 (uncorrected p reported, not corrected); for C1 also better than the random-membership and shuffled-prior controls (same reading).
- Blocked split: 2 tests per candidate, reported, not part of the rule. Pooled 70-seed gains: reported as secondary.
- Verdict wording: **confirmed** if success in confirmation 1 and gain > 0 in the four standard cells of confirmation 2; **weakened** if (a) holds but not (b)/(c); **not supported** otherwise. "Gain only near anchors" is stated whenever the far gains are ≤ 0 while the overall gains are positive.
No re-tuning after the results; failures are reported with all variants tried (see PROGRESS_LOG).
