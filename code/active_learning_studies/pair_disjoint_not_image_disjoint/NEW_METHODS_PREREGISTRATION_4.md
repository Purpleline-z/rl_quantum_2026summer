# Pre-registration 4: type-targeted (HTR) acquisition, seeds 1000-1034

Written and committed before any cell of seeds 1000-1034 exists. Motivation: the lab's stated goal is to optimise HTR performance rather than all-type performance.

## Development evidence (seeds 600-629, not confirmatory)
Per-type held-out decisive metrics were added (HTR = head 4; about 13-16 decisive HTR test judgments per seed). Restricting the queries to HTR judgments and applying I-optimal design within them (`vopt_htr`) gave an HTR accuracy gain over Random (all types) of +0.065, HTR AUC +0.047, HTR log-loss +0.181 (dev, 30 seeds, 4 cells); random selection among HTR judgments only (`random_htr`) already gives +0.044 / +0.037 / +0.134, so most of the gain is label allocation; `vopt_htr` beats `random_htr` by +0.020 accuracy, +0.010 AUC, +0.046 log-loss (paired Wilcoxon p 0.002, 0.004, 0.001). The overall (all-type) endpoint is unchanged (accuracy +0.004).

## Protocol
Judgment-unit protocol, re-tuned head schedule, groups-initial set, budgets 10/20/40/60 judgments, single-shot and sequential, Splits A and B, Random = mean of 5 draws. 35 seeds 1000-1034. At budgets that exceed the number of remaining HTR judgments the rest is filled with `vopt_u` on the other types (the type-restricted rules are defined for any budget).

## Methods (two) and primary hypotheses
`vopt_htr` (I-optimal design restricted to HTR candidates; unit target weights; definition in `new_methods_strategies.py::_restrict(vopt_u, 4)`) and `fisher_htr` (D-optimal Fisher design restricted to HTR candidates). Per method, **six one-sided tests** on the per-seed gain (averaged over budgets then over the four split x condition cells), Wilcoxon signed-rank over 35 seeds: HTR decisive accuracy, HTR AUC and HTR log-loss, each against (a) Random (all types) and (b) `random_htr` (same label allocation). **Holm over the 12 tests.** Secondary: the all-type endpoints (accuracy, AUC, log-loss) of the same methods against Random (is the HTR gain paid for elsewhere?), the other single types, per-cell results.

## Limitation
New random splits of the same 168 pair groups; HTR test sets are small (about 13-16 decisive judgments per seed), so single-seed HTR metrics are very noisy; at budgets 40-60 the HTR candidates of the pool are nearly exhausted, so there is little room for selection.

Pre-run commit: (recorded below)
