# Pre-registration 3 (cold start)

Written before the confirmatory cells exist. Development (seeds 2000-2019, `results/sel5_cold`): with 10 RANDOM initial judgments instead of the 10 initial groups (about 30 judgments), the anchor-weight gain over the default head was larger (aw8 vs default, random labels: AUC +0.032 split A / +0.026 split B), and `vopt_u` kept a small advantage over random selection even with the aw8 head in split B (AUC +0.012 [+0.004, +0.020], accuracy +0.011, log-loss +0.024; nominal p 0.015, 20 seeds, no multiplicity correction) but not in split A (AUC +0.004 [-0.006, +0.011]).

Confirmatory seeds **4000-4034** (fresh, disjoint from the development seeds and from pre-registration 1's seeds 3000-3034). Condition: `ND_INITIAL=random` (10 random judgments drawn from the judgments of the initial and pool groups, as in the earlier cold-start study); everything else as in pre-registration 1 (splits A and B; budgets 10 / 20 / 40 / 60 judgments on top of the 10 initial ones; 3 random draws; single shot from the model trained on the initial judgments). Cells: selector in {random, vopt_u} x learner in {baseline (anchor weight 0.25), aw8}.

Tests (spec `prereg3_spec.json`, code `nd_confirm.py`; per-seed gain = mean over budgets and draws; two-sided Wilcoxon over the 35 seeds; Holm within a family):
- **G1 (6 tests):** (random, aw8) vs (random, baseline), AUC / accuracy / log-loss, splits A and B (does the anchor-weight gain hold from a cold start?).
- **G2 (4 tests):** (vopt_u, aw8) vs (random, aw8), AUC and accuracy, splits A and B (does selection still add to the tuned learner from a cold start?).
- **G3 (4 tests):** (vopt_u, baseline) vs (random, baseline), AUC and accuracy, splits A and B (replication of the earlier cold-start selection gain with the default learner).

Decision rule and limits as in pre-registration 1: a claim needs Holm < 0.05 in its family; same 168 groups and 144 ideal images in every seed; the test sets are small.
