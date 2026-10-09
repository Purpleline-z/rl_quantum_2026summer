# Pre-registration 3: cold-start protocol (10 random initial judgments), seeds 900-934

Written and committed before any cell of seeds 900-934 exists.

## Why
The second confirmatory family result (seeds 700-734, groups-initial set of about 30 judgments chosen with a type-coverage rule) gave small gains (AUC +0.006, accuracy +0.01). In a robustness run on seeds 700-719 with a much weaker starting point (**10 random judgments** as the initial set, `NM_INITIAL=random`; the sensitivity variant already defined in the judgment-unit study) the same rules gave large gains: `vopt_u` log-loss +0.063, AUC +0.023, accuracy +0.028 (85-95% of seeds better, Holm < 0.001, 20 seeds); `vopt_u_inf1` log-loss +0.050, AUC +0.019, accuracy +0.027. That finding was discovered post hoc on seeds that had been used for another purpose, so it is **not** confirmatory; this pre-registration tests it on unused seeds.

## Protocol
Identical to the judgment-unit study with the re-tuned per-budget head schedule, except for the initial labelled set: 10 judgments drawn at random from the judgments of the initial and pool groups (the rest of those groups is the candidate pool); budgets 10, 20, 40, 60 acquired judgments; single-shot and sequential (rounds of 10); Splits A and B; Random = mean of 5 draws; 35 seeds 900-934.

## Methods (four) and primary hypotheses
`vopt_u`, `vopt_u_inf1`, `fisher_dopt`, `bald_decisive`. For each method and each metric in {log-loss, AUC, decisive accuracy}: the per-seed gain over Random (averaged over budgets, then over the four split x condition cells) is > 0; one-sided Wilcoxon signed-rank over 35 seeds; **Holm over the 12 tests**. Plain uncertainty sampling and relation-aware core-set are run as descriptive comparators (not part of the test family).

## Limitation
New random splits and initial draws of the same 168 pair groups; not new data. The cold-start protocol (10 random judgments, no type-coverage guarantee) is a different starting point from the main protocol and is reported as such.

Pre-run commit: (recorded below)
36777ecaff2f450c792ef2757a26dd29788b5209
