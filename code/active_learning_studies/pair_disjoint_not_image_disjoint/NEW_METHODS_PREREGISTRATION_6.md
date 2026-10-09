# Pre-registration 6: cross-world replication (train on one image-disjoint half of the data, test on the other), seeds 1200-1234

Written and committed before any cell of this run exists. Purpose: address the main limitation of the earlier runs (every seed resamples the same 168 pair groups) by making the world the rules are applied to and the world they are evaluated on **disjoint in images and pair groups**.

## Design
The 168 pair groups are split once into two image-disjoint halves H1 and H2 (connected components of the shared-image relation, 117 components; random assignment with a fixed seed, about 84 groups each; code `new_methods_study.py`, `NM_CROSS`). In direction 1 the initial set (10 groups, drawn with the study's type-coverage rule) and the candidate pool come from H1 and **all groups of H2 are the test set** (about 129-135 decisive test judgments); direction 2 swaps the halves. Protocol otherwise unchanged: one (pair, type) judgment per query, budgets 10/20/40/60, single-shot and sequential, re-tuned head schedule (calibrated earlier on the whole data, a mild limitation), Random = mean of 5 draws, no validation set. Seeds 1200-1234 vary the initial set, the pool order, the Random draws and the head initialisation; the test set of each direction is fixed.

## Methods and primary hypotheses
`vopt_u`, `vopt_u_inf1`, `fisher_dopt`, `bald_decisive`. For each method and metric in {log-loss, AUC, decisive accuracy}: the per-seed gain over Random (averaged over budgets, the two conditions and the two directions) is > 0; one-sided Wilcoxon signed-rank over 35 seeds; **Holm over the 12 tests**. Secondary: each direction separately (the gain should be positive in both), per-cell, per budget.

## Limitations stated in advance
The two worlds are image-disjoint but come from the same experiment sessions and labelling process; the seeds only resample the initial set and pool order within a world, so Wilcoxon p-values are optimistic and the two direction-level tests are the more informative check; the head hyper-parameters were calibrated on data that include both halves.

Pre-run commit: (recorded below)
