# Judgment unit re-run with a re-calibrated head schedule

Branch `claude/judgment-unit-retuned-schedule` (from `claude/frozen-encoder-strategies`, 6714938). Question: the judgment-unit study (`JUDGMENT_UNIT_RESULTS.md`) reused the head hyper-parameters
(lr 0.01, 100 full-batch steps) that had been calibrated once on random batches of the group-unit design. With 10-60 judgments the optimum may differ. Here the schedule is
re-calibrated for the judgment unit, per budget, and the whole comparison is re-run under it. Nothing else changes: same 35 seeds, splits, features, strategy code (not modified),
endpoint, hold-out sets, statistics.

Code: `judgment_unit_calibrate.py` (calibration), `judgment_unit_study.py` (runner; new optional `JU_SCHEDULE` / `JU_RESULTS`, defaults unchanged, covered by tests in
`code_behavior_tests/test_judgment_unit.py`), `judgment_unit_aggregate.py` (unchanged), `judgment_unit_retuned_compare.py` (old versus re-tuned, every number below is generated from CSVs),
`judgment_unit_pack.sh`. Results: `results/judgment_unit_study_retuned/{A_groups,B_groups}/` (`analysis/`, `manifest_seed*.json`, `schedule_used.json`; raw cells in `cells_*.tar.xz`),
`results/judgment_unit_study_retuned/comparison/` (CSVs and markdown fragments), calibration in `results/pair_endpoint_study/{schedule_judgment_unit.json, calibration_judgment_unit_*.csv}`.
The old `schedule.json` and `results/judgment_unit_study/` are untouched and still reproducible (default behaviour of the runner is unchanged).

## 1. Plain-language summary

- **The old schedule was clearly too aggressive for few judgments.** On Split A's validation groups the best setting is the smallest learning rate and fewest steps of the grid (lr 0.001, 100 steps) at every budget except 60 (lr 0.003, 100 steps); the old lr 0.01 / 100 steps is much worse at 10-40 judgments (validation log-loss 0.75 vs 0.51 at 10 judgments, 0.55 vs 0.31 at 40) and about the same at 60. The optimum sits at the corner of the specified grid, so an even gentler schedule might be better still; I did not extend the grid.
- **Random's log-loss improves a lot, its AUC and accuracy hardly change.** Random's test log-loss falls by 0.03 to 0.14 at every budget in both splits and both conditions (e.g. Split A single, 10 judgments: 0.666 to 0.528); AUC changes by -0.005 to +0.011 and accuracy by about +-0.015. The model's *ranking* ability was already what it was; the old schedule mostly produced over-confident logits. (Section 4.)
- **The headline conclusion survives: no strategy is robustly better than Random.** Under the old schedule 6 variant-hits were Holm-significant on the three primary metrics (log-loss, calibrated log-loss, AUC), 2 of them in the better-than-Random direction; under the re-tuned schedule there are 18, of which 15 are *worse* than Random and 3 better (all three the same rule, BALD x P(decisive)). No variant is Holm-significantly better than Random on log-loss in any of the four tables.
- **What does not survive the re-tuning:** the one large positive log-loss result of the earlier judgment-unit note, the all-head cluster-quota rule in Split B sequential (+0.074, Holm p < 0.001), falls to +0.021 (Holm p = 1.0); deep-ensemble BALD x P(decisive) in Split B sequential AUC (+0.014, Holm 0.023) falls to +0.006 (Holm 1.0); DPP's significant log-loss loss in Split B single (-0.104) shrinks to -0.040 (Holm 0.053). These were partly schedule effects.
- **What gets sharper:** uncertainty-driven rules look worse than Random in Split B single-shot once Random is well trained: five variants are now Holm-significantly worse on log-loss (Image-coverage uncertainty -0.051, Uncertainty, all heads -0.045 and its identical original-code twin, Uncertainty -0.043, Uncertainty + diversity, original code, all heads -0.038) and seven on AUC. Deep-ensemble and Bayesian rules that choose the type with the model (Laplace BALD, BALD x P(decisive), Fisher) go from about zero to a small positive gain in Split A (e.g. Split A single log-loss: Laplace BALD -0.001 to +0.037, BALD x P(decisive) +0.004 to +0.030, 69-74% of seeds better; neither is Holm-significant, Holm p 0.25 and 0.31). BALD x P(decisive) is positive on all 14 (table, metric) cells of the comparison; it is Holm-significant in 3 of the primary ones (calibrated log-loss Split A single +0.022 and sequential +0.057, AUC Split B sequential +0.017) and in 2 accuracy cells. This is a hypothesis worth a pre-registered follow-up, not a finding: it is one rule out of 32, found after the fact, in 10 primary tables without correction across tables, on the same 35 seeds as the old run.
- **How much the schedule matters for the ranking of strategies:** the Spearman correlation between the 32 variants' gains under the two schedules is +0.58 (log-loss), +0.55 (calibrated log-loss), +0.80 (AUC), +0.84 (accuracy), pooled over the four tables. Per table the log-loss correlation ranges from +0.28 (Split A sequential, p = 0.12) to +0.77 (Split B single). So *which rule looks best by log-loss* is moderately schedule dependent; AUC and accuracy orderings are stable.

## 2. Calibration (Split A validation only; outer test never used)

Design, mirroring `run_pair_endpoint_study.py::calibrate`: seeds 42, 79, 123; grid lr {1e-3, 3e-3, 1e-2} x steps {100, 300, 1000} (full batch); training rows = the judgments of the 10 initial groups
(about 30) plus a deterministic random batch of `budget` pool judgments (`harness.reference_batch`, `random.Random(seed*1_000_003 + budget)`) for budgets 10, 20, 40, 60, and the initial rows alone ("initial rows");
evaluation on the 20 validation groups of Split A (decisive judgments, each row's own type head). **Selection metric: mean validation decisive Bradley-Terry log-loss
(`pair_preference_endpoint.evaluate_preferences(...)['decisive_log_loss']`, confidence-weighted), the same metric the old calibration used for `schedule.json`** (the older type-accuracy calibration of the frozen-encoder study is not what the pair-endpoint study used);
validation accuracy is stored but not used. Tie-break as before: lower log-loss, then fewer steps, then lower lr. Differences from the old calibration: judgment rows instead of group rows, and one schedule per budget instead of one schedule over all budgets.
Split B has no validation set, so it uses the schedule selected here on Split A.

Mean validation log-loss (3 seeds, one batch each; best cell per column in bold):

| lr | steps | initial rows | 10 | 20 | 40 | 60 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.001 | 100 | **0.501** | **0.507** | **0.376** | **0.307** | 0.316 |
| 0.001 | 300 | 0.530 | 0.634 | 0.454 | 0.415 | 0.302 |
| 0.001 | 1000 | 0.635 | 0.856 | 0.614 | 0.570 | 0.304 |
| 0.003 | 100 | 0.602 | 0.698 | 0.480 | 0.368 | **0.286** |
| 0.003 | 300 | 0.607 | 0.758 | 0.538 | 0.593 | 0.334 |
| 0.003 | 1000 | 0.763 | 0.910 | 0.669 | 0.635 | 0.358 |
| 0.01 | 100 | 0.755 | 0.750 | 0.568 | 0.549 | 0.331 |
| 0.01 | 300 | 0.588 | 0.787 | 0.693 | 0.599 | 0.401 |
| 0.01 | 1000 | 0.521 | 1.076 | 0.626 | 0.599 | 0.346 |

Mean validation accuracy (not used for selection):

| lr | steps | initial rows | 10 | 20 | 40 | 60 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.001 | 100 | 0.776 | 0.806 | 0.865 | 0.832 | 0.854 |
| 0.001 | 300 | 0.840 | 0.828 | 0.865 | 0.813 | 0.856 |
| 0.001 | 1000 | 0.817 | 0.794 | 0.845 | 0.803 | 0.877 |
| 0.003 | 100 | 0.819 | 0.784 | 0.816 | 0.811 | 0.877 |
| 0.003 | 300 | 0.842 | 0.828 | 0.819 | 0.803 | 0.866 |
| 0.003 | 1000 | 0.866 | 0.794 | 0.832 | 0.811 | 0.888 |
| 0.01 | 100 | 0.821 | 0.796 | 0.782 | 0.811 | 0.824 |
| 0.01 | 300 | 0.863 | 0.828 | 0.844 | 0.803 | 0.845 |
| 0.01 | 1000 | 0.874 | 0.804 | 0.845 | 0.792 | 0.888 |

**Chosen schedule** (`results/pair_endpoint_study/schedule_judgment_unit.json`; the old `schedule.json` is lr 0.01 / 100 steps for everything):

| rows trained on | lr | steps |
|---|---:|---:|
| initial rows only | 0.001 | 100 |
| initial + 10 judgments | 0.001 | 100 |
| initial + 20 judgments | 0.001 | 100 |
| initial + 40 judgments | 0.001 | 100 |
| initial + 60 judgments | 0.003 | 100 |

Pooled over all budgets (the old style of selection) the choice would have been lr 0.001 / 100 steps.

Stability check (not used for selection): the same grid on 5 further independent random batches per (seed, budget), 15 batches per cell in total:

| lr | steps | initial rows | 10 | 20 | 40 | 60 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.001 | 100 | **0.501** | **0.456** | **0.400** | **0.311** | **0.296** |
| 0.001 | 300 | 0.530 | 0.487 | 0.495 | 0.358 | 0.354 |
| 0.001 | 1000 | 0.635 | 0.570 | 0.633 | 0.489 | 0.501 |
| 0.003 | 100 | 0.602 | 0.540 | 0.505 | 0.336 | 0.315 |
| 0.003 | 300 | 0.607 | 0.545 | 0.605 | 0.446 | 0.407 |
| 0.003 | 1000 | 0.763 | 0.612 | 0.758 | 0.601 | 0.527 |
| 0.01 | 100 | 0.755 | 0.572 | 0.602 | 0.388 | 0.362 |
| 0.01 | 300 | 0.588 | 0.555 | 0.662 | 0.485 | 0.419 |
| 0.01 | 1000 | 0.521 | 0.696 | 0.727 | 0.593 | 0.534 |

lr 0.001 / 100 steps is best at every budget in this check, including 60 (0.296 versus 0.315 for 0.003 / 100). So the 3e-3 choice at 60 judgments comes from the pre-specified single batch per seed and is probably noise; I kept the pre-specified rule rather than choosing again after seeing results. A visible consequence: Random's log-loss is not monotone in the budget under the re-tuned schedule (Split A single 0.443 at 40, 0.464 at 60), most likely an effect of that switch (not verified by a separate run).

## 3. Re-run

- Design: the same as `JUDGMENT_UNIT_RESULTS.md`: 32 non-random variants + Random + `random_pair_type` (42 strategy names counting the 5 draws of each baseline), 35 seeds (42, 79, 123, 202, 303, 400-429), Splits A and B, initial set = judgments of the 10 initial groups, single-shot and sequential (rounds of 10), budgets 10/20/40/60. Strategy code was not changed.
- **How the schedule is applied.** Every head fit trains on the initial judgments plus n acquired judgments and uses the schedule of the smallest calibrated budget >= n (n = 0 uses "initial rows only"). Single-shot: the selection model (and the deep-ensemble members) is fit on the initial rows (schedule "initial"), the model evaluated at budget b is fit on initial + b judgments (schedule b). Sequential: the head is retrained after each round of 10, so after 10 / 20 / 30 / 40 / 50 / 60 acquired judgments it uses the schedule of budget 10 / 20 / 40 / 40 / 60 / 60; the model that selects round r+1 is the model retrained with that schedule; checkpoints at 10/20/40/60 are the models of those rounds. The ensemble members of the sequential condition are re-fit each round with the same rule.
- **Split B** uses the schedule selected on Split A's validation (above); Split B has no validation set.
- Cell counts (recounted from the files and by `judgment_unit_aggregate.py`): 4 runs (Split A / B x single / sequential) x 35 seeds x 42 strategy names x 4 budgets = **23,520 strategy-budget cells** (5,880 per run), in 5,950 files (1,505 single-shot files per split including `initial_only`, 1,470 sequential files per split) + 70 manifests; 35/35 seeds have all 42 strategies in every run; every cell revealed exactly `budget` judgments (asserted in the aggregation).
- Checks: `judgment_unit_verify.py` (extended to honour the schedule) re-derived the pool and initial sets from the manifest, confirmed that no validation/test image is reachable from the labelled set or the pool, that training rows = initial + revealed judgments with no sibling judgment, and reproduced the stored test log-loss, AUC and accuracy to 6 decimals for Split A seed 42: single-shot Laplace BALD at 40, sequential Fisher D-optimal at 60, single-shot Random at 20. The 12 new tests (schedule mapping, default behaviour unchanged, per-budget selection rule) and the existing judgment-unit, order-invariance, new-pair-strategy and label-free tests pass (the whole `code_behavior_tests` folder: 175 passed).
- Statistics: exactly as before (`judgment_unit_aggregate.py`): per-seed gain over Random averaged over the four budgets, Wilcoxon signed-rank over 35 seeds, Holm within each table over the 32 variants, Friedman over the 33 variants, per-budget tables with Holm within each budget. Friedman (log-loss / cal. log-loss / AUC), re-tuned schedule: Split A single chi2 = 35.0 (p = 0.33) / 50.1 (0.022) / 72.5 (<0.001); Split A sequential 51.7 (0.015) / 116.3 (<0.001) / 90.8 (<0.001); Split B single 74.2 (<0.001) / - / 98.7 (<0.001); Split B sequential 93.9 (<0.001) / - / 127.1 (<0.001) (the old log-loss values were p = 0.64, 0.005, <0.001, 0.004; the omnibus statistic is larger in three of the four tables and smaller in Split A sequential; the Friedman test only says that the rules differ, not in which direction).
- `random_pair_type` against Random (log-loss gain, not in the Holm family): Split A single +0.010 (Wilcoxon p 0.037), Split A sequential +0.020 (p 0.070), Split B single -0.006 (p 0.57), Split B sequential +0.016 (p 0.18); one nominal p < 0.05 of four.

## 4. Old schedule versus re-tuned schedule

### Random's level by budget (mean over 35 seeds; Random = mean of its 5 draws)

| Split | condition | metric | schedule | 10 judgments | 20 judgments | 40 judgments | 60 judgments |
|---|---|---|---|---:|---:|---:|---:|
| Split A | single | log-loss | old | 0.666 | 0.634 | 0.552 | 0.496 |
| Split A | single | log-loss | new | 0.528 | 0.493 | 0.443 | 0.464 |
| Split A | single | AUC | old | 0.876 | 0.884 | 0.895 | 0.902 |
| Split A | single | AUC | new | 0.871 | 0.882 | 0.898 | 0.904 |
| Split A | single | accuracy (secondary) | old | 0.811 | 0.820 | 0.829 | 0.829 |
| Split A | single | accuracy (secondary) | new | 0.796 | 0.811 | 0.825 | 0.832 |
| Split A | sequential | log-loss | old | 0.659 | 0.615 | 0.556 | 0.508 |
| Split A | sequential | log-loss | new | 0.523 | 0.486 | 0.447 | 0.456 |
| Split A | sequential | AUC | old | 0.878 | 0.882 | 0.893 | 0.900 |
| Split A | sequential | AUC | new | 0.875 | 0.883 | 0.897 | 0.904 |
| Split A | sequential | accuracy (secondary) | old | 0.811 | 0.815 | 0.823 | 0.829 |
| Split A | sequential | accuracy (secondary) | new | 0.804 | 0.810 | 0.825 | 0.833 |
| Split B | single | log-loss | old | 0.596 | 0.547 | 0.527 | 0.470 |
| Split B | single | log-loss | new | 0.458 | 0.432 | 0.417 | 0.420 |
| Split B | single | AUC | old | 0.881 | 0.889 | 0.894 | 0.905 |
| Split B | single | AUC | new | 0.891 | 0.900 | 0.905 | 0.911 |
| Split B | single | accuracy (secondary) | old | 0.815 | 0.825 | 0.829 | 0.839 |
| Split B | single | accuracy (secondary) | new | 0.825 | 0.833 | 0.840 | 0.844 |
| Split B | sequential | log-loss | old | 0.627 | 0.607 | 0.560 | 0.514 |
| Split B | sequential | log-loss | new | 0.482 | 0.463 | 0.427 | 0.450 |
| Split B | sequential | AUC | old | 0.875 | 0.880 | 0.889 | 0.898 |
| Split B | sequential | AUC | new | 0.886 | 0.892 | 0.903 | 0.905 |
| Split B | sequential | accuracy (secondary) | old | 0.809 | 0.813 | 0.821 | 0.837 |
| Split B | sequential | accuracy (secondary) | new | 0.816 | 0.825 | 0.836 | 0.841 |

### Initial rows only (no acquired judgment), test metrics, mean over seeds

| Split | schedule | seeds | log-loss | AUC | accuracy |
|---|---|---:|---:|---:|---:|
| Split A | old | 35 | 0.715 | 0.872 | 0.813 |
| Split A | new | 35 | 0.563 | 0.866 | 0.798 |
| Split B | old | 35 | 0.667 | 0.865 | 0.804 |
| Split B | new | 35 | 0.513 | 0.872 | 0.812 |

### Holm-significant variants (p < 0.05 within a table of 32) under the old and the re-tuned schedule

Gain = improvement over Random averaged over budgets 10/20/40/60 (positive is better, also for log-loss). `n.s.` = Holm p >= 0.05 (value shown).

| Table | metric | variant | old gain (Holm p) | re-tuned gain (Holm p) | status |
|---|---|---|---|---|---|
| Split A, single | cal. log-loss | BALD x P(decisive) | -0.003 (1.000) | +0.022 (0.025) | only re-tuned schedule |
| Split A, single | accuracy (secondary) | Deep-ensemble BALD x P(decisive) | +0.015 (0.002) | +0.019 (0.001) | significant under both |
| Split A, single | accuracy (secondary) | BALD x P(decisive) | +0.009 (0.969) | +0.017 (0.049) | only re-tuned schedule |
| Split A, single | accuracy (secondary) | Deep-ensemble BALD (8 heads) | +0.012 (0.039) | +0.016 (0.011) | significant under both |
| Split A, sequential | cal. log-loss | BALD x P(decisive) | +0.035 (1.000) | +0.057 (0.019) | only re-tuned schedule |
| Split A, sequential | AUC | Uncertainty, all heads | -0.012 (0.331) | -0.017 (0.035) | only re-tuned schedule |
| Split A, sequential | AUC | Uncertainty, original code, all heads | -0.012 (0.331) | -0.017 (0.035) | only re-tuned schedule |
| Split A, sequential | accuracy (secondary) | Fisher D-optimal (Active Reward Modeling) | +0.024 (<0.001) | +0.016 (0.121) | only old schedule |
| Split A, sequential | accuracy (secondary) | Deep-ensemble BALD x P(decisive) | +0.020 (0.002) | +0.017 (0.006) | significant under both |
| Split A, sequential | accuracy (secondary) | Deep-ensemble BALD (8 heads) | +0.019 (0.010) | +0.014 (0.118) | only old schedule |
| Split A, sequential | accuracy (secondary) | FASS (pairs) | +0.016 (0.021) | +0.008 (0.890) | only old schedule |
| Split A, sequential | accuracy (secondary) | Image-coverage uncertainty | +0.013 (0.040) | +0.014 (0.009) | significant under both |
| Split B, single | log-loss | DPP (quality x diversity) | -0.104 (0.003) | -0.040 (0.053) | only old schedule |
| Split B, single | log-loss | Image-coverage uncertainty | -0.069 (0.243) | -0.051 (0.007) | only re-tuned schedule |
| Split B, single | log-loss | Uncertainty, all heads | -0.049 (0.102) | -0.045 (0.002) | only re-tuned schedule |
| Split B, single | log-loss | Uncertainty, original code, all heads | -0.049 (0.102) | -0.045 (0.002) | only re-tuned schedule |
| Split B, single | log-loss | Uncertainty | -0.047 (0.369) | -0.043 (0.002) | only re-tuned schedule |
| Split B, single | log-loss | Uncertainty + diversity, original code, all heads | -0.019 (1.000) | -0.038 (0.035) | only re-tuned schedule |
| Split B, single | AUC | Uncertainty, all heads | -0.026 (<0.001) | -0.021 (<0.001) | significant under both |
| Split B, single | AUC | Uncertainty, original code, all heads | -0.026 (<0.001) | -0.021 (<0.001) | significant under both |
| Split B, single | AUC | MC-dropout mutual info, original code, all heads | -0.014 (0.040) | -0.016 (0.003) | significant under both |
| Split B, single | AUC | MC-dropout variance, original code, all heads | -0.007 (1.000) | -0.016 (0.012) | only re-tuned schedule |
| Split B, single | AUC | Uncertainty + diversity, original code, all heads | -0.016 (0.053) | -0.015 (0.008) | only re-tuned schedule |
| Split B, single | AUC | Image-coverage uncertainty | -0.015 (0.529) | -0.013 (0.029) | only re-tuned schedule |
| Split B, single | AUC | Uncertainty | -0.013 (0.326) | -0.012 (0.012) | only re-tuned schedule |
| Split B, single | accuracy (secondary) | Largest predicted gap | -0.024 (<0.001) | -0.013 (0.629) | only old schedule |
| Split B, single | accuracy (secondary) | Gap + posterior std (DeltaUCB-style) | -0.019 (0.019) | -0.010 (1.000) | only old schedule |
| Split B, sequential | log-loss | Cluster-quota uncertainty, original code, all heads | +0.074 (<0.001) | +0.021 (1.000) | only old schedule |
| Split B, sequential | AUC | MC-dropout variance, original code, all heads | -0.007 (1.000) | -0.019 (0.014) | only re-tuned schedule |
| Split B, sequential | AUC | BALD x P(decisive) | +0.012 (0.227) | +0.017 (0.020) | only re-tuned schedule |
| Split B, sequential | AUC | Deep-ensemble BALD x P(decisive) | +0.014 (0.023) | +0.006 (1.000) | only old schedule |
| Split B, sequential | accuracy (secondary) | BALD x P(decisive) | +0.016 (0.117) | +0.027 (<0.001) | only re-tuned schedule |
| Split B, sequential | accuracy (secondary) | Deep-ensemble BALD x P(decisive) | +0.022 (0.001) | +0.016 (0.019) | significant under both |
| Split B, sequential | accuracy (secondary) | Fisher D-optimal (Active Reward Modeling) | +0.020 (0.006) | +0.017 (0.209) | only old schedule |
| Split B, sequential | accuracy (secondary) | Deep-ensemble BALD (8 heads) | +0.019 (0.031) | +0.015 (0.139) | only old schedule |
| Split B, sequential | accuracy (secondary) | Cluster-quota uncertainty | +0.016 (0.024) | +0.008 (1.000) | only old schedule |
| Split B, sequential | accuracy (secondary) | Uncertainty + diversity | +0.014 (0.398) | +0.016 (0.003) | only re-tuned schedule |
| Split B, sequential | accuracy (secondary) | BADGE (pairs) | +0.015 (0.013) | +0.014 (0.605) | only old schedule |

### Per table: significance counts and rank agreement of the 32 variants' gains (old vs re-tuned schedule)

| Table | metric | Holm<0.05 old | Holm<0.05 re-tuned | in both | nominal p<0.05 old / re-tuned | Spearman rho (p) | mean abs gain old / re-tuned |
|---|---|---:|---:|---:|---:|---|---|
| Split A, single | log-loss | 0 | 0 | 0 | 0 / 4 | +0.36 (0.044) | 0.014 / 0.011 |
| Split A, single | cal. log-loss | 0 | 1 | 0 | 3 / 4 | +0.55 (0.001) | 0.017 / 0.013 |
| Split A, single | AUC | 0 | 0 | 0 | 5 / 7 | +0.76 (<0.001) | 0.005 / 0.005 |
| Split A, single | accuracy (secondary) | 2 | 3 | 2 | 10 / 13 | +0.87 (<0.001) | 0.007 / 0.007 |
| Split A, sequential | log-loss | 0 | 0 | 0 | 4 / 6 | +0.28 (0.124) | 0.021 / 0.015 |
| Split A, sequential | cal. log-loss | 0 | 1 | 0 | 5 / 9 | +0.59 (<0.001) | 0.028 / 0.026 |
| Split A, sequential | AUC | 0 | 2 | 0 | 8 / 8 | +0.74 (<0.001) | 0.006 / 0.007 |
| Split A, sequential | accuracy (secondary) | 5 | 2 | 2 | 12 / 15 | +0.85 (<0.001) | 0.009 / 0.009 |
| Split B, single | log-loss | 1 | 5 | 0 | 8 / 18 | +0.77 (<0.001) | 0.028 / 0.024 |
| Split B, single | AUC | 3 | 7 | 3 | 11 / 14 | +0.83 (<0.001) | 0.008 / 0.008 |
| Split B, single | accuracy (secondary) | 2 | 0 | 0 | 4 / 6 | +0.73 (<0.001) | 0.006 / 0.007 |
| Split B, sequential | log-loss | 1 | 0 | 0 | 8 / 7 | +0.67 (<0.001) | 0.026 / 0.020 |
| Split B, sequential | AUC | 1 | 2 | 0 | 9 / 7 | +0.74 (<0.001) | 0.006 / 0.006 |
| Split B, sequential | accuracy (secondary) | 5 | 3 | 1 | 13 / 12 | +0.86 (<0.001) | 0.009 / 0.009 |

### Spearman correlation pooled over the four tables (128 variant-table gains per metric)

| metric | n | Spearman rho | p |
|---|---:|---:|---|
| log-loss | 128 | +0.58 | <0.001 |
| cal. log-loss | 64 | +0.55 | <0.001 |
| AUC | 128 | +0.80 | <0.001 |
| accuracy (secondary) | 128 | +0.84 | <0.001 |

## 5. Full 32-variant tables (log-loss and AUC gain over Random under both schedules)

#### Split A, single: gain over Random, averaged over budgets (Holm p within the table of 32 in brackets)

| Variant | log-loss gain, old | log-loss gain, re-tuned | AUC gain, old | AUC gain, re-tuned | seeds better (log-loss), old / re-tuned |
|---|---:|---:|---:|---:|---:|
| Laplace BALD | -0.001 (1.000) | +0.037 (0.250) | +0.004 (1.000) | +0.013 (0.099) | 51% / 69% |
| MaxHerding (pairs) | +0.032 (1.000) | +0.034 (0.729) | +0.005 (1.000) | +0.010 (0.679) | 60% / 66% |
| BALD x P(decisive) | +0.004 (1.000) | +0.030 (0.314) | +0.007 (1.000) | +0.013 (0.059) | 49% / 74% |
| Core-set, relation-aware pairs | +0.041 (1.000) | +0.029 (1.000) | -0.007 (1.000) | +0.002 (1.000) | 57% / 54% |
| Fisher D-optimal (Active Reward Modeling) | +0.014 (1.000) | +0.019 (1.000) | +0.006 (1.000) | +0.011 (0.397) | 49% / 66% |
| ProbCover (pairs) | +0.004 (1.000) | +0.018 (0.547) | -0.002 (1.000) | +0.003 (1.000) | 54% / 69% |
| Gap + posterior std (DeltaUCB-style) | -0.023 (1.000) | +0.014 (1.000) | -0.012 (0.540) | -0.001 (1.000) | 40% / 51% |
| TypiClust (pairs) | -0.005 (1.000) | +0.009 (1.000) | -0.003 (1.000) | +0.003 (1.000) | 60% / 66% |
| Deep-ensemble BALD x P(decisive) | +0.016 (1.000) | +0.008 (1.000) | +0.010 (0.308) | +0.011 (0.587) | 51% / 51% |
| Cluster-Margin | +0.021 (1.000) | +0.008 (1.000) | -0.001 (1.000) | -0.000 (1.000) | 60% / 54% |
| Graph cut (pairs) | -0.022 (1.000) | +0.005 (1.000) | -0.001 (1.000) | +0.007 (0.715) | 46% / 63% |
| Deep-ensemble BALD (8 heads) | -0.005 (1.000) | +0.005 (1.000) | +0.007 (1.000) | +0.010 (1.000) | 57% / 57% |
| MC-dropout mutual info | -0.001 (1.000) | +0.004 (1.000) | +0.000 (1.000) | +0.002 (1.000) | 60% / 69% |
| Cluster-quota uncertainty, original code, all heads | +0.022 (1.000) | +0.004 (1.000) | -0.007 (1.000) | -0.002 (1.000) | 60% / 51% |
| Uncertainty + diversity, original code, all heads | +0.018 (1.000) | +0.002 (1.000) | -0.007 (1.000) | -0.008 (0.274) | 60% / 43% |
| MC-dropout variance | -0.008 (1.000) | +0.001 (1.000) | -0.002 (1.000) | -0.002 (1.000) | 54% / 51% |
| MC-dropout variance, original code, all heads | -0.012 (1.000) | -0.000 (1.000) | -0.008 (1.000) | -0.006 (1.000) | 49% / 46% |
| Largest predicted gap | -0.005 (1.000) | -0.000 (1.000) | -0.008 (1.000) | -0.006 (1.000) | 54% / 37% |
| Cluster-Margin, original code, all heads | +0.014 (1.000) | -0.001 (1.000) | -0.014 (0.547) | -0.009 (1.000) | 49% / 43% |
| FASS (pairs) | +0.011 (1.000) | -0.003 (1.000) | +0.005 (1.000) | +0.001 (1.000) | 60% / 57% |
| Cluster-quota uncertainty | +0.014 (1.000) | -0.005 (1.000) | +0.001 (1.000) | -0.001 (1.000) | 60% / 60% |
| Uncertainty, original code, all heads | +0.009 (1.000) | -0.005 (1.000) | -0.005 (1.000) | -0.007 (1.000) | 60% / 54% |
| Uncertainty, all heads | +0.009 (1.000) | -0.005 (1.000) | -0.005 (1.000) | -0.007 (1.000) | 60% / 54% |
| MC-dropout mutual info, original code, all heads | -0.014 (1.000) | -0.006 (1.000) | -0.012 (0.909) | -0.007 (1.000) | 46% / 43% |
| Uncertainty | +0.012 (1.000) | -0.007 (1.000) | +0.002 (1.000) | -0.000 (1.000) | 54% / 60% |
| Core-set | -0.010 (1.000) | -0.007 (1.000) | -0.007 (1.000) | -0.005 (1.000) | 54% / 49% |
| Uncertainty + diversity | +0.024 (1.000) | -0.008 (1.000) | +0.004 (1.000) | -0.000 (1.000) | 57% / 51% |
| DropQuery (pairs) | -0.008 (1.000) | -0.011 (1.000) | -0.006 (1.000) | -0.003 (1.000) | 43% / 51% |
| Image-coverage uncertainty | -0.018 (1.000) | -0.013 (1.000) | -0.000 (1.000) | +0.000 (1.000) | 51% / 60% |
| BADGE (pairs) | -0.023 (1.000) | -0.017 (1.000) | +0.003 (1.000) | +0.002 (1.000) | 51% / 40% |
| Graph facility location (uncertainty-weighted) | -0.007 (1.000) | -0.019 (1.000) | +0.005 (1.000) | -0.002 (1.000) | 66% / 51% |
| DPP (quality x diversity) | -0.019 (1.000) | -0.022 (1.000) | +0.001 (1.000) | +0.001 (1.000) | 40% / 60% |

#### Split A, sequential: gain over Random, averaged over budgets (Holm p within the table of 32 in brackets)

| Variant | log-loss gain, old | log-loss gain, re-tuned | AUC gain, old | AUC gain, re-tuned | seeds better (log-loss), old / re-tuned |
|---|---:|---:|---:|---:|---:|
| ProbCover (pairs) | +0.038 (1.000) | +0.033 (0.129) | +0.009 (0.750) | +0.011 (0.378) | 63% / 74% |
| Laplace BALD | -0.015 (1.000) | +0.028 (1.000) | +0.003 (1.000) | +0.011 (0.966) | 57% / 66% |
| Core-set, relation-aware pairs | +0.052 (0.566) | +0.028 (1.000) | +0.002 (1.000) | +0.003 (1.000) | 66% / 63% |
| BALD x P(decisive) | +0.001 (1.000) | +0.025 (1.000) | +0.003 (1.000) | +0.010 (1.000) | 57% / 69% |
| TypiClust (pairs) | +0.008 (1.000) | +0.017 (1.000) | +0.004 (1.000) | +0.007 (1.000) | 54% / 63% |
| Cluster-Margin | +0.034 (1.000) | +0.017 (1.000) | +0.009 (0.453) | +0.004 (1.000) | 66% / 66% |
| MaxHerding (pairs) | +0.008 (1.000) | +0.016 (1.000) | -0.001 (1.000) | +0.004 (1.000) | 51% / 69% |
| Core-set | +0.023 (1.000) | +0.013 (1.000) | -0.001 (1.000) | -0.002 (1.000) | 63% / 54% |
| Fisher D-optimal (Active Reward Modeling) | -0.010 (1.000) | +0.012 (1.000) | +0.010 (0.365) | +0.008 (1.000) | 54% / 57% |
| Deep-ensemble BALD (8 heads) | -0.008 (1.000) | +0.010 (1.000) | +0.006 (1.000) | +0.011 (1.000) | 49% / 60% |
| Cluster-quota uncertainty | +0.034 (1.000) | +0.004 (1.000) | +0.005 (1.000) | +0.000 (1.000) | 63% / 54% |
| Deep-ensemble BALD x P(decisive) | -0.004 (1.000) | +0.003 (1.000) | +0.011 (0.448) | +0.007 (1.000) | 60% / 57% |
| Gap + posterior std (DeltaUCB-style) | -0.037 (1.000) | +0.001 (1.000) | -0.016 (0.359) | -0.008 (1.000) | 37% / 51% |
| Largest predicted gap | -0.012 (1.000) | -0.001 (1.000) | -0.010 (1.000) | -0.009 (1.000) | 43% / 46% |
| DropQuery (pairs) | -0.011 (1.000) | -0.003 (1.000) | -0.003 (1.000) | +0.002 (1.000) | 40% / 49% |
| MC-dropout mutual info | -0.023 (1.000) | -0.003 (1.000) | -0.003 (1.000) | -0.004 (1.000) | 51% / 54% |
| Cluster-Margin, original code, all heads | +0.039 (1.000) | -0.004 (1.000) | -0.008 (1.000) | -0.010 (1.000) | 60% / 37% |
| MC-dropout variance | -0.005 (1.000) | -0.005 (1.000) | -0.002 (1.000) | -0.005 (1.000) | 51% / 57% |
| Graph facility location (uncertainty-weighted) | -0.037 (1.000) | -0.006 (1.000) | +0.000 (1.000) | +0.001 (1.000) | 49% / 54% |
| Image-coverage uncertainty | -0.016 (1.000) | -0.008 (1.000) | +0.002 (1.000) | +0.001 (1.000) | 51% / 49% |
| Cluster-quota uncertainty, original code, all heads | +0.016 (1.000) | -0.009 (1.000) | -0.008 (1.000) | -0.008 (1.000) | 54% / 46% |
| BADGE (pairs) | -0.031 (1.000) | -0.012 (1.000) | +0.004 (1.000) | +0.001 (1.000) | 40% / 49% |
| FASS (pairs) | +0.015 (1.000) | -0.014 (1.000) | +0.006 (1.000) | -0.002 (1.000) | 54% / 54% |
| Uncertainty + diversity | +0.043 (0.659) | -0.016 (1.000) | +0.007 (1.000) | -0.002 (1.000) | 66% / 51% |
| Uncertainty + diversity, original code, all heads | +0.017 (1.000) | -0.016 (1.000) | -0.012 (0.153) | -0.014 (0.080) | 63% / 40% |
| DPP (quality x diversity) | -0.062 (1.000) | -0.018 (1.000) | -0.004 (1.000) | +0.003 (1.000) | 34% / 43% |
| Uncertainty | -0.007 (1.000) | -0.020 (1.000) | +0.004 (1.000) | -0.002 (1.000) | 46% / 46% |
| Graph cut (pairs) | -0.057 (0.531) | -0.023 (1.000) | -0.000 (1.000) | +0.001 (1.000) | 31% / 46% |
| MC-dropout variance, original code, all heads | -0.003 (1.000) | -0.024 (1.000) | -0.005 (1.000) | -0.012 (0.593) | 51% / 40% |
| Uncertainty, original code, all heads | -0.006 (1.000) | -0.032 (0.366) | -0.012 (0.331) | -0.017 (0.035) | 49% / 34% |
| Uncertainty, all heads | -0.006 (1.000) | -0.032 (0.366) | -0.012 (0.331) | -0.017 (0.035) | 49% / 34% |
| MC-dropout mutual info, original code, all heads | +0.001 (1.000) | -0.039 (1.000) | -0.004 (1.000) | -0.018 (0.422) | 57% / 37% |

#### Split B, single: gain over Random, averaged over budgets (Holm p within the table of 32 in brackets)

| Variant | log-loss gain, old | log-loss gain, re-tuned | AUC gain, old | AUC gain, re-tuned | seeds better (log-loss), old / re-tuned |
|---|---:|---:|---:|---:|---:|
| BALD x P(decisive) | +0.013 (1.000) | +0.012 (1.000) | +0.006 (1.000) | +0.006 (1.000) | 60% / 51% |
| Core-set | +0.029 (1.000) | +0.007 (1.000) | -0.003 (1.000) | -0.003 (1.000) | 60% / 63% |
| Gap + posterior std (DeltaUCB-style) | +0.003 (1.000) | +0.005 (1.000) | -0.012 (0.058) | -0.006 (1.000) | 49% / 43% |
| MaxHerding (pairs) | +0.001 (1.000) | +0.003 (1.000) | -0.001 (1.000) | +0.001 (1.000) | 51% / 54% |
| ProbCover (pairs) | -0.014 (1.000) | +0.000 (1.000) | -0.002 (1.000) | -0.001 (1.000) | 46% / 57% |
| TypiClust (pairs) | -0.011 (1.000) | -0.002 (1.000) | -0.004 (1.000) | +0.001 (1.000) | 57% / 51% |
| Largest predicted gap | +0.001 (1.000) | -0.003 (1.000) | -0.014 (0.461) | -0.009 (1.000) | 54% / 40% |
| Core-set, relation-aware pairs | -0.006 (1.000) | -0.004 (1.000) | -0.005 (1.000) | -0.003 (1.000) | 54% / 57% |
| Cluster-Margin, original code, all heads | +0.014 (1.000) | -0.008 (1.000) | -0.006 (1.000) | -0.008 (0.736) | 57% / 34% |
| Laplace BALD | -0.015 (1.000) | -0.009 (1.000) | +0.000 (1.000) | +0.001 (1.000) | 46% / 46% |
| Graph cut (pairs) | -0.035 (1.000) | -0.013 (1.000) | -0.002 (1.000) | -0.001 (1.000) | 49% / 54% |
| BADGE (pairs) | -0.034 (0.926) | -0.014 (1.000) | +0.002 (1.000) | +0.002 (1.000) | 40% / 40% |
| Cluster-Margin | -0.039 (1.000) | -0.018 (1.000) | -0.011 (1.000) | -0.004 (1.000) | 43% / 43% |
| Cluster-quota uncertainty, original code, all heads | -0.005 (1.000) | -0.019 (0.501) | -0.011 (0.139) | -0.010 (0.421) | 43% / 34% |
| FASS (pairs) | -0.043 (0.929) | -0.019 (1.000) | -0.006 (1.000) | -0.003 (1.000) | 43% / 43% |
| Deep-ensemble BALD (8 heads) | -0.020 (1.000) | -0.024 (0.581) | +0.002 (1.000) | -0.001 (1.000) | 43% / 29% |
| DropQuery (pairs) | -0.018 (1.000) | -0.025 (0.488) | -0.007 (1.000) | -0.010 (0.364) | 49% / 34% |
| Cluster-quota uncertainty | -0.012 (1.000) | -0.032 (0.053) | -0.003 (1.000) | -0.008 (0.383) | 43% / 29% |
| Fisher D-optimal (Active Reward Modeling) | -0.031 (1.000) | -0.032 (0.273) | +0.003 (1.000) | -0.004 (1.000) | 46% / 34% |
| Deep-ensemble BALD x P(decisive) | -0.036 (1.000) | -0.033 (0.123) | +0.001 (1.000) | -0.002 (1.000) | 34% / 26% |
| MC-dropout variance, original code, all heads | -0.003 (1.000) | -0.034 (0.081) | -0.007 (1.000) | -0.016 (0.012) | 43% / 26% |
| Graph facility location (uncertainty-weighted) | -0.063 (0.069) | -0.035 (0.453) | -0.011 (0.326) | -0.005 (1.000) | 29% / 34% |
| Uncertainty + diversity, original code, all heads | -0.019 (1.000) | -0.038 (0.035) | -0.016 (0.053) | -0.015 (0.008) | 51% / 26% |
| MC-dropout variance | -0.019 (1.000) | -0.039 (0.212) | -0.007 (1.000) | -0.013 (0.185) | 46% / 34% |
| DPP (quality x diversity) | -0.104 (0.003) | -0.040 (0.053) | -0.013 (0.115) | -0.009 (0.421) | 14% / 34% |
| MC-dropout mutual info, original code, all heads | -0.027 (1.000) | -0.040 (0.080) | -0.014 (0.040) | -0.016 (0.003) | 37% / 31% |
| MC-dropout mutual info | -0.020 (1.000) | -0.040 (0.213) | -0.005 (1.000) | -0.014 (0.362) | 43% / 34% |
| Uncertainty + diversity | -0.048 (1.000) | -0.041 (0.212) | -0.009 (1.000) | -0.008 (1.000) | 43% / 40% |
| Uncertainty | -0.047 (0.369) | -0.043 (0.002) | -0.013 (0.326) | -0.012 (0.012) | 31% / 29% |
| Uncertainty, all heads | -0.049 (0.102) | -0.045 (0.002) | -0.026 (<0.001) | -0.021 (<0.001) | 29% / 23% |
| Uncertainty, original code, all heads | -0.049 (0.102) | -0.045 (0.002) | -0.026 (<0.001) | -0.021 (<0.001) | 29% / 23% |
| Image-coverage uncertainty | -0.069 (0.243) | -0.051 (0.007) | -0.015 (0.529) | -0.013 (0.029) | 26% / 29% |

#### Split B, sequential: gain over Random, averaged over budgets (Holm p within the table of 32 in brackets)

| Variant | log-loss gain, old | log-loss gain, re-tuned | AUC gain, old | AUC gain, re-tuned | seeds better (log-loss), old / re-tuned |
|---|---:|---:|---:|---:|---:|
| BALD x P(decisive) | +0.061 (0.255) | +0.051 (0.213) | +0.012 (0.227) | +0.017 (0.020) | 69% / 69% |
| Core-set, relation-aware pairs | +0.063 (0.279) | +0.038 (0.851) | +0.007 (1.000) | +0.009 (1.000) | 74% / 69% |
| Core-set | +0.063 (0.274) | +0.034 (0.320) | +0.006 (1.000) | +0.005 (1.000) | 66% / 66% |
| TypiClust (pairs) | +0.044 (1.000) | +0.033 (0.587) | +0.006 (1.000) | +0.010 (1.000) | 60% / 69% |
| Gap + posterior std (DeltaUCB-style) | +0.064 (0.511) | +0.031 (1.000) | -0.002 (1.000) | -0.002 (1.000) | 54% / 54% |
| Largest predicted gap | +0.051 (1.000) | +0.030 (1.000) | -0.004 (1.000) | -0.001 (1.000) | 60% / 54% |
| Cluster-quota uncertainty, original code, all heads | +0.074 (<0.001) | +0.021 (1.000) | +0.005 (1.000) | +0.003 (1.000) | 80% / 66% |
| Laplace BALD | +0.023 (1.000) | +0.021 (1.000) | +0.006 (1.000) | +0.007 (1.000) | 54% / 54% |
| Cluster-Margin, original code, all heads | +0.033 (1.000) | +0.020 (1.000) | -0.005 (1.000) | +0.000 (1.000) | 69% / 40% |
| Graph cut (pairs) | +0.009 (1.000) | +0.016 (1.000) | +0.010 (0.189) | +0.010 (0.193) | 63% / 57% |
| ProbCover (pairs) | -0.006 (1.000) | +0.014 (1.000) | +0.000 (1.000) | +0.005 (1.000) | 46% / 60% |
| MaxHerding (pairs) | +0.019 (1.000) | +0.011 (1.000) | -0.001 (1.000) | +0.002 (1.000) | 63% / 60% |
| DropQuery (pairs) | +0.009 (1.000) | +0.010 (1.000) | -0.000 (1.000) | +0.003 (1.000) | 60% / 57% |
| Cluster-Margin | +0.012 (1.000) | +0.004 (1.000) | +0.001 (1.000) | +0.001 (1.000) | 49% / 60% |
| Deep-ensemble BALD x P(decisive) | +0.013 (1.000) | +0.003 (1.000) | +0.014 (0.023) | +0.006 (1.000) | 60% / 54% |
| BADGE (pairs) | +0.016 (1.000) | -0.003 (1.000) | +0.011 (0.256) | +0.006 (1.000) | 51% / 46% |
| Deep-ensemble BALD (8 heads) | -0.000 (1.000) | -0.005 (1.000) | +0.004 (1.000) | +0.005 (1.000) | 49% / 49% |
| MC-dropout mutual info | +0.034 (1.000) | -0.005 (1.000) | +0.004 (1.000) | -0.004 (1.000) | 69% / 49% |
| MC-dropout variance | +0.024 (1.000) | -0.008 (1.000) | +0.001 (1.000) | -0.005 (1.000) | 60% / 51% |
| Cluster-quota uncertainty | +0.039 (1.000) | -0.011 (1.000) | +0.009 (0.575) | -0.003 (1.000) | 71% / 46% |
| Uncertainty + diversity | +0.003 (1.000) | -0.012 (1.000) | +0.002 (1.000) | -0.001 (1.000) | 57% / 49% |
| Graph facility location (uncertainty-weighted) | +0.009 (1.000) | -0.013 (1.000) | -0.000 (1.000) | +0.000 (1.000) | 49% / 43% |
| FASS (pairs) | -0.022 (1.000) | -0.013 (1.000) | -0.004 (1.000) | -0.003 (1.000) | 40% / 49% |
| Uncertainty + diversity, original code, all heads | +0.020 (1.000) | -0.014 (1.000) | -0.010 (1.000) | -0.012 (0.654) | 66% / 43% |
| Fisher D-optimal (Active Reward Modeling) | +0.003 (1.000) | -0.020 (1.000) | +0.011 (0.132) | +0.001 (1.000) | 49% / 43% |
| Uncertainty, all heads | +0.013 (1.000) | -0.021 (1.000) | -0.014 (0.303) | -0.015 (0.256) | 46% / 34% |
| Uncertainty, original code, all heads | +0.013 (1.000) | -0.021 (1.000) | -0.014 (0.303) | -0.015 (0.256) | 46% / 34% |
| DPP (quality x diversity) | -0.060 (1.000) | -0.021 (1.000) | -0.004 (1.000) | -0.001 (1.000) | 46% / 43% |
| Image-coverage uncertainty | -0.003 (1.000) | -0.024 (1.000) | -0.001 (1.000) | -0.006 (1.000) | 49% / 40% |
| Uncertainty | +0.024 (1.000) | -0.029 (0.679) | +0.003 (1.000) | -0.008 (1.000) | 57% / 43% |
| MC-dropout mutual info, original code, all heads | -0.017 (1.000) | -0.031 (0.359) | -0.012 (1.000) | -0.018 (0.071) | 46% / 34% |
| MC-dropout variance, original code, all heads | +0.002 (1.000) | -0.039 (0.213) | -0.007 (1.000) | -0.019 (0.014) | 51% / 34% |

## 6. Per budget, re-tuned schedule (best variants at each budget are the best of 32 and overstate; Holm only within one budget)

**A single, log-loss (re-tuned schedule)**

| Budget (judgments) | top 3 by mean gain over Random (Holm p within the budget) | variants with Holm p < 0.05 |
|---:|---|---|
| 10 | Laplace BALD +0.029 (1.00); Core-set, relation-aware pairs +0.028 (1.00); ProbCover (pairs) +0.026 (1.00) | none |
| 20 | Core-set, relation-aware pairs +0.036 (1.00); Laplace BALD +0.034 (1.00); Fisher D-optimal (Active Reward Modeling) +0.032 (1.00) | none |
| 40 | Laplace BALD +0.043 (0.37); BALD x P(decisive) +0.039 (0.36); MaxHerding (pairs) +0.035 (1.00) | none |
| 60 | MaxHerding (pairs) +0.067 (0.02); Laplace BALD +0.043 (0.89); Graph cut (pairs) +0.038 (1.00) | MaxHerding (pairs) +0.067 |

**A single, AUC (re-tuned schedule)**

| Budget (judgments) | top 3 by mean gain over Random (Holm p within the budget) | variants with Holm p < 0.05 |
|---:|---|---|
| 10 | Laplace BALD +0.016 (0.57); Fisher D-optimal (Active Reward Modeling) +0.014 (0.16); BALD x P(decisive) +0.013 (0.57) | none |
| 20 | Deep-ensemble BALD x P(decisive) +0.015 (1.00); Fisher D-optimal (Active Reward Modeling) +0.015 (0.99); Laplace BALD +0.013 (1.00) | none |
| 40 | Laplace BALD +0.014 (0.29); BALD x P(decisive) +0.014 (0.34); Deep-ensemble BALD x P(decisive) +0.013 (1.00) | none |
| 60 | MaxHerding (pairs) +0.017 (0.11); Graph cut (pairs) +0.013 (0.78); BALD x P(decisive) +0.012 (1.00) | none |

**A sequential, log-loss (re-tuned schedule)**

| Budget (judgments) | top 3 by mean gain over Random (Holm p within the budget) | variants with Holm p < 0.05 |
|---:|---|---|
| 10 | Laplace BALD +0.024 (1.00); Core-set, relation-aware pairs +0.023 (1.00); ProbCover (pairs) +0.021 (1.00) | none |
| 20 | Laplace BALD +0.049 (0.27); ProbCover (pairs) +0.026 (1.00); BALD x P(decisive) +0.025 (1.00) | Uncertainty, all heads -0.050, Uncertainty, original code, all heads -0.050 |
| 40 | BALD x P(decisive) +0.047 (0.36); ProbCover (pairs) +0.042 (0.12); Laplace BALD +0.037 (0.89) | none |
| 60 | ProbCover (pairs) +0.044 (0.28); Core-set +0.043 (1.00); Core-set, relation-aware pairs +0.036 (1.00) | none |

**A sequential, AUC (re-tuned schedule)**

| Budget (judgments) | top 3 by mean gain over Random (Holm p within the budget) | variants with Holm p < 0.05 |
|---:|---|---|
| 10 | Laplace BALD +0.011 (1.00); Fisher D-optimal (Active Reward Modeling) +0.010 (1.00); BALD x P(decisive) +0.009 (1.00) | none |
| 20 | Laplace BALD +0.018 (0.20); Deep-ensemble BALD (8 heads) +0.015 (1.00); Fisher D-optimal (Active Reward Modeling) +0.012 (1.00) | Uncertainty + diversity, original code, all heads -0.019, Uncertainty, all heads -0.021, Uncertainty, original code, all heads -0.021 |
| 40 | ProbCover (pairs) +0.017 (0.27); Deep-ensemble BALD (8 heads) +0.016 (0.36); BALD x P(decisive) +0.014 (0.69) | none |
| 60 | ProbCover (pairs) +0.013 (0.34); TypiClust (pairs) +0.009 (1.00); Core-set +0.009 (1.00) | none |

**B single, log-loss (re-tuned schedule)**

| Budget (judgments) | top 3 by mean gain over Random (Holm p within the budget) | variants with Holm p < 0.05 |
|---:|---|---|
| 10 | BALD x P(decisive) +0.002 (1.00); Largest predicted gap -0.003 (1.00); Deep-ensemble BALD (8 heads) -0.004 (1.00) | Uncertainty + diversity, original code, all heads -0.037, Uncertainty, all heads -0.039, Uncertainty, original code, all heads -0.039, MC-dropout mutual info, original code, all heads -0.040, DPP (quality x diversity) -0.042, Uncertainty -0.043, Image-coverage uncertainty -0.048, MC-dropout variance, original code, all heads -0.058 |
| 20 | Core-set +0.010 (1.00); TypiClust (pairs) +0.009 (1.00); BALD x P(decisive) +0.007 (1.00) | Uncertainty -0.046, Uncertainty, original code, all heads -0.054, Uncertainty, all heads -0.054, MC-dropout variance, original code, all heads -0.057, MC-dropout mutual info, original code, all heads -0.066 |
| 40 | BALD x P(decisive) +0.030 (1.00); Core-set +0.027 (1.00); MaxHerding (pairs) +0.015 (1.00) | Uncertainty, all heads -0.042, Uncertainty, original code, all heads -0.042 |
| 60 | ProbCover (pairs) +0.017 (1.00); Gap + posterior std (DeltaUCB-style) +0.016 (1.00); MaxHerding (pairs) +0.011 (1.00) | Deep-ensemble BALD (8 heads) -0.072, Uncertainty + diversity -0.073, Image-coverage uncertainty -0.075, Deep-ensemble BALD x P(decisive) -0.080 |

**B single, AUC (re-tuned schedule)**

| Budget (judgments) | top 3 by mean gain over Random (Holm p within the budget) | variants with Holm p < 0.05 |
|---:|---|---|
| 10 | BALD x P(decisive) +0.003 (1.00); Graph cut (pairs) +0.001 (1.00); Deep-ensemble BALD x P(decisive) +0.001 (1.00) | MC-dropout mutual info, original code, all heads -0.019, Uncertainty, all heads -0.020, Uncertainty, original code, all heads -0.020, Uncertainty + diversity, original code, all heads -0.020, MC-dropout variance, original code, all heads -0.026 |
| 20 | TypiClust (pairs) +0.008 (1.00); BADGE (pairs) +0.003 (1.00); BALD x P(decisive) +0.001 (1.00) | Uncertainty, original code, all heads -0.022, Uncertainty, all heads -0.022, MC-dropout mutual info, original code, all heads -0.026, MC-dropout variance, original code, all heads -0.026 |
| 40 | BALD x P(decisive) +0.010 (1.00); Core-set +0.008 (1.00); BADGE (pairs) +0.007 (1.00) | Uncertainty, all heads -0.021, Uncertainty, original code, all heads -0.021 |
| 60 | BALD x P(decisive) +0.008 (1.00); ProbCover (pairs) +0.002 (1.00); MaxHerding (pairs) +0.002 (1.00) | none |

**B sequential, log-loss (re-tuned schedule)**

| Budget (judgments) | top 3 by mean gain over Random (Holm p within the budget) | variants with Holm p < 0.05 |
|---:|---|---|
| 10 | BALD x P(decisive) +0.026 (1.00); Largest predicted gap +0.021 (1.00); Deep-ensemble BALD (8 heads) +0.020 (1.00) | MC-dropout variance, original code, all heads -0.035 |
| 20 | BALD x P(decisive) +0.054 (0.91); Cluster-quota uncertainty, original code, all heads +0.040 (0.90); Gap + posterior std (DeltaUCB-style) +0.040 (1.00) | none |
| 40 | BALD x P(decisive) +0.056 (0.14); Core-set, relation-aware pairs +0.051 (0.24); TypiClust (pairs) +0.049 (0.11) | MC-dropout mutual info, original code, all heads -0.049 |
| 60 | BALD x P(decisive) +0.068 (0.06); Core-set, relation-aware pairs +0.054 (1.00); Cluster-Margin, original code, all heads +0.051 (1.00) | none |

**B sequential, AUC (re-tuned schedule)**

| Budget (judgments) | top 3 by mean gain over Random (Holm p within the budget) | variants with Holm p < 0.05 |
|---:|---|---|
| 10 | BALD x P(decisive) +0.008 (1.00); Graph cut (pairs) +0.006 (1.00); Deep-ensemble BALD x P(decisive) +0.006 (1.00) | MC-dropout variance, original code, all heads -0.020 |
| 20 | BALD x P(decisive) +0.017 (0.36); Graph cut (pairs) +0.012 (0.65); TypiClust (pairs) +0.010 (1.00) | none |
| 40 | BALD x P(decisive) +0.022 (0.02); Core-set, relation-aware pairs +0.017 (0.34); TypiClust (pairs) +0.015 (0.64) | BALD x P(decisive) +0.022, MC-dropout variance, original code, all heads -0.021, MC-dropout mutual info, original code, all heads -0.023 |
| 60 | BALD x P(decisive) +0.021 (0.00); Cluster-Margin, original code, all heads +0.013 (0.84); Graph cut (pairs) +0.012 (0.20) | BALD x P(decisive) +0.021 |



## 7. Reading, caveats

1. **Schedule matters for levels, much less for conclusions.** The old schedule over-fit the very small training sets (higher loss, over-confident logits); Random's log-loss is 0.03-0.14 lower under the calibrated schedule and the initial-rows-only model's log-loss falls from 0.715 to 0.563 (Split A) and 0.667 to 0.513 (Split B), while AUC (a ranking measure) is nearly unchanged. Log-loss is the metric where the schedule matters most; the conclusions drawn from AUC and accuracy are almost the same under both schedules (rank correlations 0.80 and 0.84).
2. **No strategy is robustly better than Random in the primary log-loss endpoint under either schedule.** Several results that looked like effects under the old schedule (cluster-quota all-head in Split B sequential, deep-ensemble BALD x P(decisive) AUC in Split B sequential, DPP in Split B single) weaken or vanish; they were not stable under a change of the head schedule.
3. **The consistent positive signal under the new schedule is one rule family (BALD x P(decisive), and with smaller effects Laplace BALD / Fisher), and it is small** (log-loss +0.01 to +0.06 depending on table, AUC +0.006 to +0.017, 51-83% of seeds better), significant after Holm only on some metrics, absent on log-loss after Holm. Treat as a hypothesis.
4. **The negative signal is clearer than the positive one:** plain uncertainty sampling and its all-head and coverage variants are worse than Random in Split B single-shot (up to 7 Holm-significant variants on AUC, 5 on log-loss) and in a few per-budget cells elsewhere. This agrees with the earlier finding that uncertainty selects mostly non-decisive judgments (about 30% decisive against 50% for Random).
5. **What this does not show.** (a) Only one grid, with the optimum at its corner; an even smaller learning rate or fewer steps may help further. (b) The schedule for 60 judgments is probably a noisy choice (see section 2). (c) One schedule family per budget, shared by all strategies: a strategy-specific schedule (e.g. actively chosen batches may prefer a different one) was not tuned. (d) Same seeds, splits and test groups as the old run, so the two schedules' results are not independent. (e) Calibration used 3 seeds and 20 validation groups (about 60 decisive judgments) per seed; the grid differences at 60 judgments are small. (f) Multiple tables (10 primary tables, 14 including accuracy) are each Holm-corrected within the table only; across tables a few Holm hits are expected by chance.
6. The sensitivity run with 10 random initial judgments (`A_random`, `B_random` in the old study) was not repeated under the new schedule.

Reproduce: `python judgment_unit_calibrate.py` (about 30 minutes); `JU_RESULTS=judgment_unit_study_retuned JU_SCHEDULE=results/pair_endpoint_study/schedule_judgment_unit.json JU_SPLIT=A|B JU_INITIAL=groups JU_THREADS=1 python judgment_unit_study.py` (default five seeds) and again with `PAIR_STUDY_SEEDS=400-429`
(about 5.5 minutes per seed and split on one thread); `JU_RESULTS=judgment_unit_study_retuned python judgment_unit_aggregate.py A_groups|B_groups`; `python judgment_unit_retuned_compare.py`; `judgment_unit_pack.sh unpack` restores the raw cells.
