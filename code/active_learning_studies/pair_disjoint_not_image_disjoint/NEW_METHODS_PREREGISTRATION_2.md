# Pre-registration 2: replication of the confirmatory run on seeds 800-834

Written and committed before any cell of seeds 800-834 exists. Same protocol as `NEW_METHODS_PREREGISTRATION.md` (judgment unit, budgets 10/20/40/60, single-shot and sequential, Splits A and B, re-tuned head schedule, groups-initial set, Random = mean of 5 draws).

## Why
On seeds 700-734 the pre-registered methods `vopt_u` and `vopt_u_inf1` were significant after Holm on decisive accuracy (+0.0095, +0.0114) but not on AUC (+0.0058, +0.0049; nominal one-sided p 0.033 and 0.061) or log-loss. The earlier rules `fisher_dopt` and `bald_decisive` (not pre-registered in that run, run on the same seeds as comparators) gave similar gains (accuracy +0.014 and +0.013, AUC +0.007 and +0.008). This second run on new seeds tests whether the family-level gains replicate.

## Methods (four)
`vopt_u`, `vopt_u_inf1` (definitions in the first pre-registration), `fisher_dopt` (greedy D-optimal Fisher design of the own-type last layer; Active Reward Modeling), `bald_decisive` (BALD of the own head under the last-layer Laplace posterior x predicted probability of a decisive answer).

## Primary hypotheses (one-sided; 8 tests)
For each method, gain over Random (per-seed, averaged over budgets then over the four split x condition cells) > 0 for **AUC** and **decisive accuracy**; one-sided Wilcoxon signed-rank over 35 seeds; **Holm over the 8 tests**. Log-loss and the per-cell results are reported as secondary (uncorrected). Pooling seeds 700-734 and 800-834 (70 seeds) is a post hoc analysis and will be labelled so.

## Limitation
New random splits of the same 168 pair groups; not new data.

Pre-run commit: (recorded below)
