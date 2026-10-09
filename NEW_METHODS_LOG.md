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
