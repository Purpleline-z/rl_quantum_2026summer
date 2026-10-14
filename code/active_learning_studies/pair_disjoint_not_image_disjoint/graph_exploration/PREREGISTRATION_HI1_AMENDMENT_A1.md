# Amendment A1 to PREREGISTRATION_HI1 — p-value of the re-assessment rule

Written 2026-10-14 ~04:15 UTC (commit time is authoritative), after the development block (seeds 11300–11319, 40 cells) was complete and analysed in a debugging run, and BEFORE any confirmation cell (seeds 11330–11399) was analysed (confirmation 1 was still running; nothing of it had been opened; no group-level number of any confirmation seed exists).

## What was found on development data
`hi_check_null.py` (output `results/hi/null_check_dev.txt`):
| comparison, split | null SD of the swap-randomisation statistic (formula) | same, from directly recomputed random swaps | empirical SE of the estimate (random half-splits of the 168 groups) | group-bootstrap SE |
|---|---|---|---|---|
| lap3_k10 vs aw8_all, A | 0.0070 | 0.0074 | 0.0045 | 0.0040 |
| lap3_k10_st4 vs aw8_all_st2lr0.5, A | 0.0070 | 0.0067 | 0.0055 | 0.0051 |
| aw8 vs default head, B | 0.0050 | 0.0048 | 0.0045 | 0.0045 |

The exact swap-randomisation test is implemented correctly (formula = direct swaps) and tests the sharp null that the two learners are exchangeable group by group. But when the two learners' scores are on different scales (the graph regulariser changes logit scale), swapping one group and not another compares scores of different learners across groups in the cross-group pairs of the AUC; this inflates the null variance 1.4–1.6x relative to the real sampling variability of the estimate (checked independently by half-splits). The test is valid but conservative for the question "is the AUC gain zero", and with Holm over 40 tests it labels almost everything WEAKER even when the group-level interval is far from 0 (e.g. development lap3_k10 vs aw8_all: interval [+0.003, +0.019], swap p = 0.13).

## Amendment (effective for all confirmation analyses)
1. The p-value of the rule and of the Holm families F1 / F2 is the **group-bootstrap p-value** `2·min(P*(b ≤ 0), P*(b ≥ 0))` with the +1 correction, from the same B = 10 000 replicates as the interval (the p at which the percentile interval just excludes 0), not the swap-randomisation p. Everything else in HI1 is unchanged (units, intervals, families, SUPPORTED / WEAKER / UNSUPPORTED definition with "Holm-adjusted p < 0.05 in both splits", annotations).
2. The swap-randomisation p-value (20 000 swaps) is still computed and reported next to it, and the corrected claims table shows, for every claim, the label under the **originally frozen rule** (swap p) in square brackets next to the label under the amended rule. The reader can therefore see where the amendment matters.
3. The bootstrap SE is ~8–12 % smaller than the half-split SE on two of three development checks, i.e. the percentile interval is slightly optimistic; this is why the cluster-pair (c0.03) and block (b42) intervals are reported as "robust to dependence" annotations and why the results document states the half-split check.
This amendment was motivated by development data under knowledge of how the labels would come out under both rules; it is stated here rather than hidden. It changes labels in the direction of the interval the user's own example rule asks for ("the interval for the AUC gain excludes 0 in both splits"), plus Holm.
