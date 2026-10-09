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
