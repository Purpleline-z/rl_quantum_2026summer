# Pre-registration 1 (anchor-weighted learning and its interaction with selection)

Written before any confirmatory cell exists. The development seeds 2000-2019 were used for all exploration (screens 1-9 in `OVERNIGHT_LOG.md`); the confirmatory seeds are **3000-3034**, disjoint from every seed of the earlier studies and from the development seeds (`nd_core.DEV_SEEDS / CONFIRM_SEEDS`, asserted in `test_new_directions.py`).

## What was found on the development seeds (why these hypotheses)
Raising the weight of the reference-anchor loss (the Bradley-Terry ranking loss among the labelled ideal images of the four types; repository default 0.25) to 8 improved the held-out preference prediction with randomly chosen judgments (20 dev seeds): split A AUC +0.024, accuracy +0.023, log-loss +0.069; split B AUC +0.009, accuracy +0.016, log-loss +0.039 (all vs the default). Weight 0 is clearly worse; the gain is flat between about 2 and 16 and gone at 32. Using the class labels of ALL ideal images of the seed's split (training references + utility-validation + outer-test ideal images; none of them is in the pair universe, checked by content identity) with weight 8 gave a further small gain (A AUC +0.027, B +0.014). The weight 8 was picked from the grid {0.25, 1, 2, 4, 8, 16, 32} on these dev seeds, so the dev effect sizes are optimistic (winner's curse); the confirmatory run is the evidence.

## Protocol (unchanged from the repository's judgment-unit study)
Frozen SimCLR features; one (pair, type) judgment per query; 10 initial groups; budgets 10 / 20 / 40 / 60 judgments; splits A and B; per-budget re-tuned head schedule; held-out image-disjoint test groups; endpoint = decisive held-out judgments (AUC, accuracy, log-loss). Single-shot selection from the model trained on the initial judgments. Random label sets: 3 draws per (seed, budget), identical for every learner. `vopt_u` = the repository's pool-wide I-optimal rule (greedy, unit target weights), chosen with the default-weight model; the learner trained on the chosen labels is the factor "learner".

Cells: selector in {random, vopt_u} x learner in {baseline (anchor weight 0.25), aw8 (anchor weight 8, training references only), aw8_all (anchor weight 8, all ideal images of the split as anchors)}.

## Hypotheses and tests (spec: `prereg1_spec.json`; code: `nd_confirm.py`)
Gain of a cell A over a cell B per seed = mean over budgets and draws of sign x (metric_A - metric_B), sign +1 for AUC / accuracy and -1 for log-loss; two-sided Wilcoxon signed-rank over the 35 seeds; Holm within a family; reported with a paired bootstrap 95% interval over seeds.
- **F1 (primary, 12 tests, Holm over 12):** (random, aw8) vs (random, baseline) and (random, aw8_all) vs (random, baseline), for AUC, accuracy, log-loss, in split A and split B.
- **F2 (4 tests, Holm over 4):** does selection still help with the stronger learner? (vopt_u, aw8) vs (random, aw8), AUC and accuracy, splits A and B.
- **F3 (4 tests, Holm over 4):** replication of the repository's earlier selection result: (vopt_u, baseline) vs (random, baseline), AUC and accuracy, splits A and B.
- **F4 (6 tests, Holm over 6; descriptive of the combined effect):** (vopt_u, aw8) vs (random, baseline), AUC, accuracy, log-loss, splits A and B.

## Decision rules
A claim is made for a split only if the corresponding Holm-adjusted p < 0.05 in its family; effects whose interval includes 0 are reported as not confirmed. A confirmed learner effect is called "confirmed on a re-split of the same 168 pair groups" - not on new data. Everything else (per budget, per type, calibrated log-loss in split A, cold start) is exploratory and labelled so.

## Known limits (stated in advance)
Same 168 pair groups in every seed, so Wilcoxon p-values are optimistic; the held-out sets are small (about 50-60 decisive judgments in split A, about 100 in split B); the anchor labels are those of 144 ideal images; the weight grid was explored on the development seeds only.
