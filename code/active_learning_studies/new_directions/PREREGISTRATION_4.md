# Pre-registration 4 (HTR, the lab's priority type)

Written before the confirmatory cells exist. Exploratory breakdown of pre-registration 1's cells (`results/confirm1_per_type_exploratory.csv`): the anchor-weight gain on the held-out HTR judgments was small and not distinguishable from 0 with the training references only (aw8: accuracy +0.010 [-0.021, +0.041] in split A, +0.004 [-0.016, +0.023] in B) but clear when the class labels of all ideal images were used as anchors (aw8_all: +0.036 [+0.014, +0.059] / +0.026 [+0.005, +0.048]). This is a post-hoc observation on seeds already used, so it is tested again on fresh seeds.

Confirmatory seeds **5000-5034** (disjoint from all earlier seed sets). Random label sets only (3 draws per seed and budget), splits A and B, budgets 10 / 20 / 40 / 60 judgments, initial 10 groups as in pre-registration 1. Cells: random selection x learner in {baseline, aw8, aw8_all}.

Family **H1 (12 tests, Holm over 12; `prereg4_spec.json`, `nd_confirm.py`)**: for split in {A, B} and metric in {HTR accuracy, HTR log-loss} (decisive held-out judgments of type HTR only): aw8_all vs baseline, aw8 vs baseline, aw8_all vs aw8. Per-seed gain = mean over budgets and draws; two-sided Wilcoxon over 35 seeds; paired bootstrap interval. Seeds with no HTR decisive test judgment in a cell are dropped for that metric (reported as n_seeds).

Limits: HTR has few held-out decisive judgments per seed (about 10-16), so per-seed values are noisy; same 168 groups in every seed.
