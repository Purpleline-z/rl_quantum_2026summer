# Pre-registration 5: priority-weighted I-optimal design (HTR weight 4), seeds 1100-1134

Written and committed before any cell of seeds 1100-1134 exists.

## Development evidence (seeds 600-629, not confirmatory)
`vopt_hw4` multiplies the gain of HTR candidates by 4 in the I-optimal greedy design (unit target weights, otherwise `vopt_u`), a soft version of the hard HTR restriction `vopt_htr` (pre-registration 4). Dev gains over Random (all types), pooled over the four split x condition cells: HTR accuracy +0.056, HTR AUC +0.045, HTR log-loss +0.145, all-type accuracy +0.014, all-type AUC +0.010, other types' accuracy +0.004 (13), -0.001 (c6x2), +0.008 (1x1); the hard restriction gives larger HTR gains (+0.065) but loses accuracy on the other types (-0.008 to -0.014).

## Protocol
Judgment-unit protocol, re-tuned head schedule, groups-initial set, budgets 10/20/40/60, single-shot and sequential, Splits A and B, Random = mean of 5 draws, 35 seeds 1100-1134.

## Method and primary hypotheses
`vopt_hw4` only. Five one-sided Wilcoxon tests of the per-seed gain over Random (averaged over budgets, then over the four cells), **Holm over the 5**: HTR decisive accuracy, HTR AUC, HTR log-loss, all-type decisive accuracy, all-type AUC. Secondary: all-type log-loss; the other types' accuracies (non-inferiority is judged descriptively: a mean gain not below -0.01); comparison with `random_htr` (same HTR allocation) and with `vopt_u` (no priority); per-cell results.

## Limitation
New splits of the same 168 groups; HTR test sets are small (about 13-16 decisive judgments per seed).

Pre-run commit: (recorded below)
