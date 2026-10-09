# New acquisition methods vs Random: overnight research log

Branch `claude/new-methods` (from `claude/frozen-encoder-strategies`). Started 2026-10-09 (user asleep; at least 8 hours requested). Graph-based methods are out of scope (another session explores them).

## Fixed protocol (not to be changed)
- Query unit: one (pair, type) judgment; budget in judgments (10/20/40/60); single-shot and sequential (rounds of 10); Splits A and B; groups-initial set; frozen SimCLR features, order-independent head;
  endpoints: held-out preference log-loss, calibrated log-loss (Split A), AUC, accuracy; statistics: per-seed gain over Random (mean of 5 draws) averaged over budgets, Wilcoxon + Holm.
- Head schedule: the **re-tuned per-budget schedule** (`schedule_judgment_unit.json`), because it gives a well-trained Random and prevents "wins" that only exploit an under-trained baseline.
- Seeds: **development 600-629** (30 seeds; all iteration happens here), **confirmation 700-734** (35 seeds, touched only once, after the final candidates are pre-registered in a committed file).
- Primary confirmatory hypotheses (written before the confirmatory run): for each of at most 3 pre-registered methods, one-sided Wilcoxon of the per-seed gain over Random pooled over the four (split x condition) cells and budgets, on log-loss and on AUC; Holm over the 2 x (#methods) tests. Per-cell results are secondary.

## Todo
1. [x] Dev baselines (Random, random_pair_type, bald_decisive, fisher_dopt, laplace_bald, core_set_relation, uncertainty) on seeds 600-629, both splits, both conditions (running).
2. [x] Literature round 1 (ARM-FI Fisher; A-optimality for logistic regression (Schein & Ungar); LAL (Konyushkova); abstention).
3. [x] Diagnostic: how predictable is a decisive outcome before asking (AUC 0.67-0.70 type+distance, 0.71-0.74 with anchor-head scores, 0.72-0.75 all; type only 0.57-0.58).
4. [ ] M1 pool-wide variance reduction (I/V-optimal), with and without P(decisive) -> dev seeds.
5. [ ] Inspect why candidates are mostly tie / not_apply; improve the outcome model.
6. [ ] Further ideas (see below), iterate, pre-register, confirm.

## Ideas queue
- I-optimal / V-optimal pool-wide predictive variance reduction (Schein & Ungar A-optimality for logistic regression; ARM-FI uses Fisher on last layer).
- Outcome-informativeness: a better P(decisive) from anchor-only head scores + pair statistics.
- Learned acquisition function (LAL, Konyushkova 2017) with group-wise cross-fitting.
- Type-quota / type-matching to the test distribution.
- Expected-error-reduction via linearised retraining.

## Literature notes (round 1)
- Active Reward Modeling (Shen et al., ICML 2025): Fisher information on the last layer, pairs with moderate reward differences + exploration of representation space; our `fisher_dopt` implements this and showed a replicated decisive-accuracy gain.
- Schein & Ungar, A-optimality for logistic regression: variance-reduction designs are the most likely AL methods to match/beat random in their evaluation; Yang & Loog benchmark: uncertainty sampling is strong but random is rarely overwhelmed. Motivates pool-wide predictive-variance reduction (`vopt`).
- Hacohen & Weinshall (ICML 2022) / Uncertainty Herding (Bae et al., ICLR 2025) / SelectAL: low budgets favour coverage/typicality, high budgets uncertainty; no single rule works at all budgets without adaptation.
- Tripp (blog, 2025): AL cannot beat random when all candidates are about equally informative; the smaller the set, the stronger random. In our data the main source of variation in informativeness is whether a judgment is decisive (51%) versus tie / not_apply (49%), plus type-specific heads.
- Direct Acquisition Optimization (Zhao et al. 2024): expected error reduction with influence functions for low budgets; our `vopt` is the linearised last-layer analogue.
- LAL (Konyushkova et al., NeurIPS 2017): learn the acquisition function as a regressor on state/candidate features; planned as idea 3 (cross-fitted over seeds).
- Tie models (Rao-Kupper, Davidson): no published tie-aware active selection found; here the outcome is {win, win, tie, not applicable} and only decisive outcomes count for the endpoint, so the informativeness of a query is P(decisive) x BT information.

## Diagnostics
- Predicting a decisive outcome before asking (group-wise 5-fold CV on the pool, 10 dev seeds): AUC type only 0.57-0.58; + pair distance/cosine 0.67-0.70; + anchor-only head scores 0.71-0.74; all 0.73-0.75 (`explore_decisive_predictability.py`).
- Smoke test of `vopt`: with only the 30 initial judgments the variance-reduction picks have a LOWER decisive share (0.42-0.45) than the pool (0.52), i.e. high-leverage pairs are disproportionately tie / not_apply.

## Dev results (seeds 600-609, Split A single-shot unless noted; mean gain over Random, uncorrected)
- vopt family (pool-wide predictive-variance reduction on the own-type last layer, greedy, Laplace posterior): AUC +0.007 to +0.013, decisive accuracy +0.015 to +0.028 (9-10 of 10 seeds better), log-loss +0.01 to +0.035 (n.s.).
- Ridge (0.3, 1, 3, 10) matters little; unit target weights (`vopt_u`, minimise the summed logit variance over the pool) is as good as or better than sensitivity-weighted targets and has no knobs.
- Outcome-informativeness factors P(decisive) (v1, v2) add little on top of the variance reduction itself (acc +0.01 to +0.02 relative to Random for both).
- Reference rules on 3 seeds only so far; full reference runs (600-629) are in progress.

## Dev, 30 seeds x 4 cells (seeds 600-629; mean gain over Random; Holm over 15-21 tests within the dev table, descriptive only)
vopt (I-optimal, sensitivity-weighted): AUC +0.0093, acc +0.0202, log-loss +0.0043; vopt_inf2: AUC +0.0114, acc +0.0205, ll +0.0170; vopt_u: AUC +0.0089, acc +0.0167 (97% of seeds), ll +0.0172; vopt_u_inf1: AUC +0.0101, acc +0.0143, ll +0.0214.
Earlier rules under the same re-tuned schedule on the same seeds: fisher_dopt AUC +0.0095, acc +0.0184, ll +0.0109; bald_decisive AUC +0.0089, acc +0.0129, ll +0.0159; laplace_bald AUC +0.0092, acc +0.0120, ll +0.0173; core_set_relation AUC +0.0066, acc +0.0050 (n.s.), ll +0.0233; plain uncertainty about 0 on all (-0.001 to +0.004).
=> The gain is a property of the whole last-layer optimal-design family (Fisher / Laplace-BALD / variance reduction), not specific to vopt; uncertainty sampling does not share it.
Mechanism (budget 60, share of picks): decisive 0.49-0.52 (Random 0.51-0.52), tie 0.09-0.11 (Random 0.13-0.15), not_apply 0.39-0.40 (Random 0.34-0.35); fewer distinct pairs (37 vs 42); type mix tilted away from (1x1) (0.18-0.20 vs 0.25) towards (sqrt13) (0.33 vs 0.28). So the gain does not come from choosing decisive judgments.

## CONFIRMATORY RESULT (pre-registered, seeds 700-734, 35 seeds, pooled over the 4 split x condition cells; one-sided Wilcoxon; Holm over 6 tests)
| method | metric | mean gain | seeds better | p | Holm |
| vopt_u | log-loss | +0.0074 | 49% | 0.37 | 0.74 |
| vopt_u | AUC | +0.0058 | 60% | 0.033 | 0.13 |
| vopt_u | accuracy | +0.0095 | 77% | 0.0001 | 0.0004 |
| vopt_u_inf1 | log-loss | +0.0046 | 51% | 0.39 | 0.74 |
| vopt_u_inf1 | AUC | +0.0049 | 60% | 0.061 | 0.18 |
| vopt_u_inf1 | accuracy | +0.0114 | 77% | <0.0001 | 0.0002 |
Pre-registered verdict: "partly" (accuracy significant after Holm for both; AUC and log-loss positive but not significant after Holm). Effects are smaller than on the development seeds (accuracy +0.017 -> +0.010, AUC +0.009 -> +0.006): winner's-curse / regression to the mean.

## Diagnostics (dev seeds 600-609)
- Oracle ceiling (selection by hindsight using the TEST labels; top-b judgments by single-addition test log-loss gain; Split A): log-loss gain over Random 0.16-0.20 and accuracy gain 0.038-0.054 at budgets 10-60; oracle picks are 55-73% decisive. The practical rules recover only about a third of the accuracy headroom (+0.017) and a tenth of the log-loss headroom.
- Value of knowing the outcome class (Split A, seeds 600-604, 10 draws per policy): random among TRULY decisive judgments (perfect knowledge) is NOT better than Random (log-loss -0.01 to -0.09, AUC about 0 to +0.01, accuracy 0 to +0.015); random among the top 25/50% by predicted P(decisive) is also not better. Conclusion: ties / not_apply judgments are as useful for this endpoint as decisive ones (they train the push-down / equality terms); the P(decisive) factors add nothing, which matches `vopt_u` vs `vopt_u_inf1`.
- Type quotas (stratified allocation) change nothing (`vopt_u_quota` vs `vopt_u`: all within noise).

## Confirmatory seeds 700-734: earlier rules on the same seeds (secondary, descriptive; pooled over 4 cells)
fisher_dopt: AUC +0.0073, acc +0.0140, ll +0.0038; bald_decisive: AUC +0.0083 (one-sided p 0.0003), acc +0.0134, ll +0.0130; core_set_relation: AUC +0.0005, acc +0.0023; plain uncertainty: AUC -0.0044, acc +0.0037, ll -0.0172.
=> `vopt_u` / `vopt_u_inf1` are NOT better than Fisher D-optimal or BALD x P(decisive); the whole last-layer optimal-design family gives ~ +0.006-0.008 AUC and +0.010-0.014 accuracy over Random; uncertainty and core-set do not.

## Wave 2: NTK I-optimal design (kernelised variance reduction over ALL head parameters; first-layer NTK part is 1.3x the last-layer part, so cross-type information through the shared first layer is not negligible)
Implemented as `ntk_vopt_t1`, `ntk_vopt_t300` (tau = prior precision); dev run on seeds 600-629 in progress.

## Wave 2 dev results (seeds 600-629, 30 seeds x 4 cells)
NTK I-optimal design (tau=1 / 300): AUC +0.0071 / +0.0071, acc +0.0141 / +0.0156, ll +0.0078 / -0.0001 -> not better than last-layer vopt_u (AUC +0.0089, acc +0.0167), Fisher D-optimal (AUC +0.0095, acc +0.0184) or BALD x P(dec). The extra first-layer part of the kernel does not help.
Hindsight utility diagnostic (Spearman between observable candidate features and the TRUE single-judgment test-log-loss utility, Split A seeds 600-607, 8 seeds): |rho| <= 0.06 for phi norm, p(1-p), leverage, EGL, predicted/true decisiveness, anchor scores, unseen images, type indicators. The true marginal utility of one judgment is not predictable from these features => there is no learnable per-candidate score (LAL-style learned acquisition is not promising); the family gains come from batch-level geometry (coverage of the representation), not from ranking individual judgments.

## Robustness of the pre-registered methods (seeds 700-719, 20 seeds, pooled over 4 cells)
- Old head schedule (lr 0.01, 100 steps): vopt_u AUC +0.0007, acc +0.0042 (n.s.); vopt_u_inf1 AUC +0.0055, acc +0.0082 (Holm 0.17 / 0.06). The benefit depends on a well-trained Random baseline / head.
- Cold start (10 RANDOM initial judgments instead of the 10 initial groups, about 30 judgments): vopt_u log-loss +0.063, AUC +0.023, acc +0.028 (85-95% of seeds, Holm <= 0.0002); vopt_u_inf1 log-loss +0.050, AUC +0.019, acc +0.027. Large effect when the starting model is weak (post hoc discovery -> pre-registration 3 on seeds 900-934).

## HTR-targeted acquisition (dev seeds 600-629; the lab's goal is HTR performance)
Per-type held-out metrics added (`new_methods_study.py`; HTR = head 4, about 13-16 decisive test judgments per seed). Gains over Random (all types), pooled over 4 cells: vopt_u (all types): HTR acc +0.045, AUC +0.035, ll +0.111; random_htr (random among HTR judgments): +0.044 / +0.037 / +0.134 (so most of the gain is label allocation); vopt_htr (I-optimal design within HTR): +0.065 / +0.047 / +0.181; fisher_htr: +0.060 / +0.046 / +0.172. Selection value on top of allocation (paired, vs random_htr): vopt_htr +0.020 acc (p 0.002), +0.010 AUC (0.004), +0.046 ll (0.001); fisher_htr +0.016 / +0.009 / +0.037. Other types pay a small price (13: -0.008, c6x2: -0.014, 1x1: -0.008 accuracy); the all-type accuracy is unchanged (+0.004), all-type AUC +0.009. Pre-registration 4 (seeds 1000-1034) is running.

## CONFIRMATION 2 (pre-registration 2, seeds 800-834, 35 seeds, pooled over 4 cells; Holm over 8 = 4 methods x {AUC, accuracy})
| method | AUC gain (seeds better, Holm) | accuracy gain (seeds better, Holm) | log-loss gain (secondary) |
| vopt_u | +0.0127 (74%, 0.0004) | +0.0167 (86%, <0.0001) | +0.0349 |
| vopt_u_inf1 | +0.0116 (66%, 0.0013) | +0.0156 (89%, <0.0001) | +0.0250 |
| fisher_dopt | +0.0071 (66%, 0.016) | +0.0160 (80%, 0.0001) | +0.0017 |
| bald_decisive | +0.0098 (66%, 0.0013) | +0.0153 (89%, <0.0001) | +0.0215 |
All 8 primary tests are significant after Holm: the family effect replicates on fresh splits. Post hoc pooling of seeds 700-734 and 800-834 (70 seeds): vopt_u AUC +0.0093, accuracy +0.0131, log-loss +0.0211 (p 0.002); vopt_u_inf1 AUC +0.0083, acc +0.0135, ll +0.0148; fisher AUC +0.0072, acc +0.0150, ll +0.0027 (n.s.); bald_decisive AUC +0.0091, acc +0.0143, ll +0.0173.
Same 168 groups in every seed -> sensitivity to split draw, not new data.
