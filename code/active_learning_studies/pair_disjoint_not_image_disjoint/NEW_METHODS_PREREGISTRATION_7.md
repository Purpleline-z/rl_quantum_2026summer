# Pre-registration 7: gentler head schedule (lr 0.0003, 100 steps), seeds 1300-1334

Written and committed before any cell of this run exists.

## Why
The re-tuned schedule of `JUDGMENT_UNIT_RETUNED_RESULTS.md` sat at the corner of its grid (lr 0.001). A robustness run on seeds 700-714 (descriptive, 15 seeds) with lr 0.003 and lr 0.0003 (100 steps, all budgets) gave vopt_u gains over Random of AUC +0.011 and +0.017, accuracy +0.011 and +0.018, log-loss +0.032 and +0.037, against AUC +0.008, accuracy +0.010, log-loss +0.019 under the re-tuned schedule on the same seeds; and the absolute levels with lr 0.0003 are the best seen (Random log-loss 0.433, `vopt_u` log-loss 0.396 and AUC 0.916, against Random 0.448 and `vopt_u` 0.429 / 0.908 under the re-tuned schedule). The gain with the old schedule (lr 0.01) vanishes. This pre-registration tests whether the family gain over Random is larger and still significant with lr 0.0003, on unused seeds.

## Protocol
Identical to pre-registration 2 (judgment unit, budgets 10/20/40/60, single-shot and sequential, Splits A and B, groups-initial set, Random = mean of 5 draws, 35 seeds 1300-1334) except for the head schedule: lr 0.0003, 100 full-batch steps for the initial fit and every budget (file `results/new_methods/sched_lr0003_s100.json`). This is a sensitivity analysis of the protocol; the schedule of the main results (re-tuned) is not changed.

## Methods and primary hypotheses
`vopt_u`, `vopt_u_inf1`, `fisher_dopt`, `bald_decisive`; for each, log-loss, AUC and decisive accuracy: the per-seed gain over Random (budgets, cells averaged) is > 0; one-sided Wilcoxon over 35 seeds; **Holm over 12**. Secondary: absolute test metrics of Random and of each method under this schedule (is the gain relative to a weaker baseline?).

## Limitation
Same 168 groups, new splits; the schedule was not calibrated by validation (it was chosen from three values after seeing a 15-seed result), so no claim is made that lr 0.0003 is optimal.

Pre-run commit: (recorded below)
95fea602b6cce31912d503c98750603ff0d0b82c
