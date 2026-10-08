# Replication of the judgment-unit study on 35 new seeds (500-534)

Pre-registered in `JUDGMENT_UNIT_REPLICATION_PREREGISTRATION.md` (committed before any run; pre-run commit `6b6070dabd57515f49c933ca9499c144f5322637`). Code: `judgment_unit_study.py` (unchanged; run through `judgment_unit_replication_run.sh` with `PAIR_STUDY_SEEDS=500-534` and `JU_ONLY=<15 strategy keys>`), analysis `judgment_unit_replication_analysis.py`, tables `judgment_unit_replication_tables.py`. Results: `results/judgment_unit_study/replication/{A,B}_groups/` (cells, manifests) and `.../replication/analysis/` (CSVs, `report.txt`).
No strategy, training, split or evaluation code was changed; the existing runner already takes a seed range and a variant list. The only code edit is a path-parsing line in `judgment_unit_verify.py` (so that it accepts `replication/A_groups`; the old call form behaves as before). No bug was found or fixed.

## 1. Limitation (read first)

The new seeds draw new random splits (initial / validation / test / pool), new initial sets, new Random draws and new model initialisations, but the SAME 168 pair groups (521 judgments, 284 images). This replication therefore checks whether the findings are sensitive to the split and initial draw; it is not a test on new data, and test sets of the new and old seeds overlap (the same groups appear in many splits). Seeds are also not independent of each other (they resample the same groups), so every Wilcoxon p-value here and in the main run is somewhat optimistic.

## 2. What was run

- Seeds 500-534 (35), Split A and Split B, single-shot and sequential, budgets 10/20/40/60 judgments, initial set = judgments of the 10 initial groups, lr 0.01 / 100 steps, 2 torch threads.
- 15 strategy families = 13 variants + Random + random_pair_type; 23 strategy names per seed (Random and random_pair_type with 5 draws each).
- Cells, recounted from the files (asserted in the analysis script): 3,220 per (split, condition) = 35 seeds x 23 names x 4 budgets; 12,880 cells in total, stored as 4 x 805 cell files (35 x 23 each, one file per seed and name holding the four budgets) plus 70 `initial_only` files and 70 manifests; `n_revealed == budget` in every cell; all 35 seeds have all 23 names in every (split, condition). Committed size of the replication results: about 17 MB.
- Nothing failed or was skipped. The "Uncertainty, original code, all heads" key (`uncertainty_lf`) was not rerun (identical rule to `uncertainty_all_heads`, see the pre-registration).

## 3. Confirmatory result (seeds 500-534 only; 18 one-sided Wilcoxon tests, Holm over the 18)

Estimate = mean over seeds of the per-seed gain over Random (averaged over the four budgets; positive = better than Random, also for log-loss, where the gain is Random minus strategy). "Seeds in predicted direction" = share of the 35 seeds with gain > 0 (for "gain < 0" hypotheses: with gain < 0). The last column gives the estimate of the same test on the 35 main seeds (where the hypotheses came from).

| Hyp. | Variant | Cell | Metric | Direction | Estimate | Seeds in predicted direction | one-sided p | Holm p | Replicated? | Main-run estimate |
|---|---|---|---|---|---|---|---|---|---|---|
| H1 | Cluster-quota uncertainty, all heads (original code) | B seq. | log-loss | gain > 0 | +0.044 | 57% | 0.022 | 0.110 | no | +0.074 |
| H2 | Deep-ensemble BALD x P(decisive) | B seq. | AUC | gain > 0 | +0.014 | 69% | 0.002 | 0.019 | yes | +0.014 |
| H3 | Deep-ensemble BALD x P(decisive) | A single | accuracy | gain > 0 | +0.011 | 69% | 0.007 | 0.039 | yes | +0.015 |
| H3 | Deep-ensemble BALD x P(decisive) | A seq. | accuracy | gain > 0 | +0.019 | 77% | <0.001 | 0.003 | yes | +0.020 |
| H3 | Deep-ensemble BALD x P(decisive) | B single | accuracy | gain > 0 | +0.012 | 69% | 0.004 | 0.027 | yes | +0.006 |
| H3 | Deep-ensemble BALD x P(decisive) | B seq. | accuracy | gain > 0 | +0.023 | 83% | <0.001 | <0.001 | yes | +0.022 |
| H3 | Deep-ensemble BALD | A single | accuracy | gain > 0 | +0.008 | 60% | 0.059 | 0.237 | no | +0.012 |
| H3 | Deep-ensemble BALD | A seq. | accuracy | gain > 0 | +0.020 | 71% | <0.001 | 0.002 | yes | +0.019 |
| H3 | Deep-ensemble BALD | B single | accuracy | gain > 0 | +0.013 | 69% | 0.001 | 0.015 | yes | +0.010 |
| H3 | Deep-ensemble BALD | B seq. | accuracy | gain > 0 | +0.025 | 74% | <0.001 | <0.001 | yes | +0.019 |
| H3 | Fisher D-optimal | A single | accuracy | gain > 0 | +0.010 | 71% | 0.002 | 0.015 | yes | +0.011 |
| H3 | Fisher D-optimal | A seq. | accuracy | gain > 0 | +0.021 | 77% | <0.001 | <0.001 | yes | +0.024 |
| H3 | Fisher D-optimal | B single | accuracy | gain > 0 | +0.021 | 80% | <0.001 | <0.001 | yes | +0.011 |
| H3 | Fisher D-optimal | B seq. | accuracy | gain > 0 | +0.030 | 83% | <0.001 | <0.001 | yes | +0.020 |
| H4 | Core-set | B single | log-loss | gain > 0 | +0.002 | 60% | 0.358 | 0.358 | no | +0.029 |
| H4 | Core-set | B seq. | log-loss | gain > 0 | +0.029 | 60% | 0.070 | 0.237 | no | +0.063 |
| H5 | DPP | B single | log-loss | gain < 0 | -0.035 | 51% | 0.079 | 0.237 | no | -0.104 |
| H5 | Uncertainty, all heads | B single | AUC | gain < 0 | -0.015 | 71% | 0.001 | 0.015 | yes | -0.026 |

### Verdict per hypothesis (pre-registered rule: every one of its tests must have Holm p < 0.05 with the predicted sign)

| Hyp. | Claim | Tests | Replicated tests (Holm p < 0.05, right sign) | Tests with right sign | Verdict |
|---|---|---:|---:|---:|---|
| H1 | Cluster-quota uncertainty, all heads (original code) | 1 | 0 | 1 | not replicated |
| H2 | Deep-ensemble BALD x P(decisive) | 1 | 1 | 1 | replicated |
| H3 | Deep-ensemble BALD x P(decisive) | 4 | 4 | 4 | replicated |
| H3 | Deep-ensemble BALD | 4 | 3 | 4 | not replicated |
| H3 | Fisher D-optimal | 4 | 4 | 4 | replicated |
| H4 | Core-set | 2 | 0 | 2 | not replicated |
| H5 | DPP | 1 | 0 | 1 | not replicated |
| H5 | Uncertainty, all heads | 1 | 1 | 1 | replicated |
| H1 | ALL (whole hypothesis) | 1 | 0 | 1 | not replicated |
| H2 | ALL (whole hypothesis) | 1 | 1 | 1 | replicated |
| H3 | ALL (whole hypothesis) | 12 | 11 | 12 | not replicated |
| H4 | ALL (whole hypothesis) | 2 | 0 | 2 | not replicated |
| H5 | ALL (whole hypothesis) | 2 | 1 | 2 | not replicated |

| Hypothesis | Estimate (new seeds) | Holm p | Seeds in predicted direction | Outcome |
|---|---|---|---|---|
| H1 all-head Cluster-quota, log-loss, B sequential | +0.044 (main +0.074) | 0.110 (one-sided nominal 0.022) | 57% | NOT replicated (right sign, about 60% of the main effect, fails Holm over 18) |
| H2 deep-ensemble BALD x P(decisive), AUC, B sequential | +0.014 (main +0.014) | 0.019 | 69% | REPLICATED |
| H3 decisive accuracy, three variants x four cells | +0.008 to +0.030, all 12 positive | 11 of 12 below 0.05 (largest 0.237) | 60-83% | NOT replicated as stated: 11 of 12 cells pass; deep-ensemble BALD in A single (+0.008, nominal p 0.059, Holm 0.237) fails. Per variant: ensemble BALD x P(decisive) 4/4, Fisher D-optimal 4/4, ensemble BALD 3/4 |
| H4 core-set log-loss, B single and B sequential | +0.002 / +0.029 | 0.358 / 0.237 | 60% / 60% | NOT replicated (signs positive, effects small, not significant even nominally at p < 0.05 two-sided) |
| H5 DPP worse than Random (log-loss, B single) | -0.035 (main -0.104) | 0.237 | 51% worse | NOT replicated (only half of the seeds worse; the mean shrinks to a third) |
| H5 all-head uncertainty worse than Random (AUC, B single) | -0.015 (main -0.026) | 0.015 | 71% worse | REPLICATED |
| H6 plain Uncertainty no advantage (4 cells x log-loss, AUC) | all 8 estimates negative: -0.003 to -0.039 | all 1.000 | 34-51% better | SUPPORTED (no test rejected) |

### H6 null check (alternative: Uncertainty gain > 0; Holm over the 8 tests)

| Hyp. | Variant | Cell | Metric | Direction | Estimate | Seeds in predicted direction | one-sided p | Holm p | Replicated? | 
|---|---|---|---|---|---|---|---|---|---|
| H6 | Uncertainty (own type) | A single | log-loss | gain > 0 | -0.025 | 46% | 0.803 | 1.000 | no |
| H6 | Uncertainty (own type) | A single | AUC | gain > 0 | -0.005 | 51% | 0.683 | 1.000 | no |
| H6 | Uncertainty (own type) | A seq. | log-loss | gain > 0 | -0.024 | 43% | 0.874 | 1.000 | no |
| H6 | Uncertainty (own type) | A seq. | AUC | gain > 0 | -0.003 | 43% | 0.754 | 1.000 | no |
| H6 | Uncertainty (own type) | B single | log-loss | gain > 0 | -0.021 | 40% | 0.916 | 1.000 | no |
| H6 | Uncertainty (own type) | B single | AUC | gain > 0 | -0.006 | 46% | 0.928 | 1.000 | no |
| H6 | Uncertainty (own type) | B seq. | log-loss | gain > 0 | -0.039 | 34% | 0.978 | 1.000 | no |
| H6 | Uncertainty (own type) | B seq. | AUC | gain > 0 | -0.007 | 34% | 0.952 | 1.000 | no |

## 4. Plain-language reading

- What replicated: the AUC gain of deep-ensemble BALD x P(decisive) in Split B sequential (+0.014 both times); the decisive-accuracy gain of Fisher D-optimal design in all four cells and of deep-ensemble BALD x P(decisive) in all four cells (+0.010 to +0.030, 69-83% of seeds); deep-ensemble BALD without P(decisive) has the same positive sign in all four cells and is significant in three; and the loss of all-head uncertainty in AUC in Split B single-shot (-0.015). These are modest effects: 1-3 points of decisive accuracy, 0.01 AUC, and no log-loss effect that survives. They concern rules that choose the type with the model (ensemble BALD, Fisher), which is consistent with the main-run reading.
- What did not replicate: the headline log-loss result of the main run, the all-head cluster-quota gain in Split B sequential (+0.074 in the main run) shrinks to +0.044 and does not pass Holm (nominal one-sided p 0.022; only 57% of seeds better versus 80% before). Core-set in Split B (+0.029 / +0.063 in the main run) shrinks to +0.002 / +0.029 and is not significant. DPP being much worse than Random in Split B single-shot (-0.104 in the main run, 86% of seeds worse) shrinks to -0.035 (51% of seeds worse). Log-loss effects of individual selection rules are therefore much less stable across split draws than the accuracy and AUC effects.
- Plain Uncertainty has no advantage: all eight estimates are negative (-0.003 to -0.039), consistent with H6 (uncertainty selection mostly asks about ties and not_apply).
- Overall verdict under the pre-registered binary rule: H2 replicated; H3 not replicated as a whole (11 of 12 cells pass; two of its three variants pass in all four cells); H1 not replicated; H4 not replicated; H5 half replicated (all-head uncertainty yes, DPP no), hence not replicated as a whole; H6 supported. None of the conclusions is reversed (all signs agree with the main run in all 18 tests), but the sizes of the log-loss effects are not reliable. Given the shared 168 groups, even the replicated effects say only that the accuracy / AUC advantage of ensemble BALD and Fisher D-optimal design is not an accident of the 35 particular splits; whether it holds on new data remains untested.

## 5. Exploratory analyses (NOT pre-specified; nothing in the verdicts above depends on them)

### 5.1 Pooled 70 seeds (35 main + 35 new): the same 18 tests, Holm over the 18

The main seeds are those on which the hypotheses were formed, so pooled estimates are not an independent test.

| Hyp. | Variant | Cell | Metric | Direction | Estimate | Seeds in predicted direction | one-sided p | Holm p | Replicated? | 
|---|---|---|---|---|---|---|---|---|---|
| H1 | Cluster-quota uncertainty, all heads (original code) | B seq. | log-loss | gain > 0 | +0.059 | 69% | <0.001 | <0.001 | yes |
| H2 | Deep-ensemble BALD x P(decisive) | B seq. | AUC | gain > 0 | +0.014 | 70% | <0.001 | <0.001 | yes |
| H3 | Deep-ensemble BALD x P(decisive) | A single | accuracy | gain > 0 | +0.013 | 71% | <0.001 | <0.001 | yes |
| H3 | Deep-ensemble BALD x P(decisive) | A seq. | accuracy | gain > 0 | +0.019 | 79% | <0.001 | <0.001 | yes |
| H3 | Deep-ensemble BALD x P(decisive) | B single | accuracy | gain > 0 | +0.009 | 64% | 0.004 | 0.009 | yes |
| H3 | Deep-ensemble BALD x P(decisive) | B seq. | accuracy | gain > 0 | +0.023 | 83% | <0.001 | <0.001 | yes |
| H3 | Deep-ensemble BALD | A single | accuracy | gain > 0 | +0.010 | 66% | <0.001 | 0.001 | yes |
| H3 | Deep-ensemble BALD | A seq. | accuracy | gain > 0 | +0.019 | 74% | <0.001 | <0.001 | yes |
| H3 | Deep-ensemble BALD | B single | accuracy | gain > 0 | +0.012 | 63% | <0.001 | 0.002 | yes |
| H3 | Deep-ensemble BALD | B seq. | accuracy | gain > 0 | +0.022 | 76% | <0.001 | <0.001 | yes |
| H3 | Fisher D-optimal | A single | accuracy | gain > 0 | +0.010 | 71% | <0.001 | <0.001 | yes |
| H3 | Fisher D-optimal | A seq. | accuracy | gain > 0 | +0.022 | 79% | <0.001 | <0.001 | yes |
| H3 | Fisher D-optimal | B single | accuracy | gain > 0 | +0.016 | 70% | <0.001 | <0.001 | yes |
| H3 | Fisher D-optimal | B seq. | accuracy | gain > 0 | +0.025 | 83% | <0.001 | <0.001 | yes |
| H4 | Core-set | B single | log-loss | gain > 0 | +0.015 | 60% | 0.092 | 0.092 | no |
| H4 | Core-set | B seq. | log-loss | gain > 0 | +0.046 | 63% | 0.003 | 0.008 | yes |
| H5 | DPP | B single | log-loss | gain < 0 | -0.069 | 69% | <0.001 | <0.001 | yes |
| H5 | Uncertainty, all heads | B single | AUC | gain < 0 | -0.020 | 77% | <0.001 | <0.001 | yes |

Pooled verdicts (same rule):

| Hyp. | Claim | Tests | Replicated tests (Holm p < 0.05, right sign) | Tests with right sign | Verdict |
|---|---|---:|---:|---:|---|
| H1 | Cluster-quota uncertainty, all heads (original code) | 1 | 1 | 1 | replicated |
| H2 | Deep-ensemble BALD x P(decisive) | 1 | 1 | 1 | replicated |
| H3 | Deep-ensemble BALD x P(decisive) | 4 | 4 | 4 | replicated |
| H3 | Deep-ensemble BALD | 4 | 4 | 4 | replicated |
| H3 | Fisher D-optimal | 4 | 4 | 4 | replicated |
| H4 | Core-set | 2 | 1 | 2 | not replicated |
| H5 | DPP | 1 | 1 | 1 | replicated |
| H5 | Uncertainty, all heads | 1 | 1 | 1 | replicated |
| H1 | ALL (whole hypothesis) | 1 | 1 | 1 | replicated |
| H2 | ALL (whole hypothesis) | 1 | 1 | 1 | replicated |
| H3 | ALL (whole hypothesis) | 12 | 12 | 12 | replicated |
| H4 | ALL (whole hypothesis) | 2 | 1 | 2 | not replicated |
| H5 | ALL (whole hypothesis) | 2 | 2 | 2 | replicated |

Pooled H6: no test is significant (Holm p = 1.0 in all 8; estimates -0.034 to +0.0002).

| Hyp. | Variant | Cell | Metric | Direction | Estimate | Seeds in predicted direction | one-sided p | Holm p | Replicated? | 
|---|---|---|---|---|---|---|---|---|---|
| H6 | Uncertainty (own type) | A single | log-loss | gain > 0 | -0.006 | 50% | 0.564 | 1.000 | no |
| H6 | Uncertainty (own type) | A single | AUC | gain > 0 | -0.001 | 56% | 0.400 | 1.000 | no |
| H6 | Uncertainty (own type) | A seq. | log-loss | gain > 0 | -0.015 | 44% | 0.758 | 1.000 | no |
| H6 | Uncertainty (own type) | A seq. | AUC | gain > 0 | +0.000 | 50% | 0.327 | 1.000 | no |
| H6 | Uncertainty (own type) | B single | log-loss | gain > 0 | -0.034 | 36% | 0.997 | 1.000 | no |
| H6 | Uncertainty (own type) | B single | AUC | gain > 0 | -0.010 | 41% | 0.997 | 1.000 | no |
| H6 | Uncertainty (own type) | B seq. | log-loss | gain > 0 | -0.008 | 46% | 0.710 | 1.000 | no |
| H6 | Uncertainty (own type) | B seq. | AUC | gain > 0 | -0.002 | 47% | 0.683 | 1.000 | no |

Reading: pooled over 70 seeds, H1, H2, H3, H5 pass the rule and H4 does not (core-set in Split B single-shot: +0.015, p = 0.09; sequential +0.046, Holm 0.0075). Because the pooled numbers contain the data that suggested the hypotheses they cannot rescue the failed replication of H1 and H4 and DPP; they only show that, averaged over both sets of splits, the all-head cluster-quota gain is +0.059 (69% of seeds) and the DPP loss -0.069.

### 5.2 All 13 variants, new seeds against main seeds

Cell entries: mean gain on the new seeds (share of seeds better, `*` = Holm p < 0.05 two-sided over the 13 variants within that table; `random_pair_type` is not in the Holm family) / mean gain on the main seeds. Full tables with p-values: `analysis/full_table_*.csv`.

Log-loss gain:

| Variant | A single: new / main | A seq.: new / main | B single: new / main | B seq.: new / main |
|---|---:|---:|---:|---:|
| Cluster-quota uncertainty, all heads (original code) | +0.001 (43%) / +0.022 | +0.015 (54%) / +0.016 | +0.016 (57%) / -0.005 | +0.044 (57%) / +0.074 |
| Cluster-quota uncertainty (own type) | -0.015 (49%) / +0.014 | +0.020 (57%) / +0.034 | +0.017 (66%) / -0.012 | +0.024 (69%) / +0.039 |
| Deep-ensemble BALD x P(decisive) | +0.017 (57%) / +0.016 | +0.023 (66%) / -0.004 | +0.026 (69%) / -0.036 | +0.026 (57%) / +0.013 |
| Deep-ensemble BALD | +0.003 (51%) / -0.005 | +0.015 (63%) / -0.008 | +0.033* (80%) / -0.020 | +0.015 (57%) / -0.000 |
| Fisher D-optimal | +0.000 (57%) / +0.014 | +0.033 (60%) / -0.010 | +0.021 (66%) / -0.031 | +0.016 (57%) / +0.003 |
| BALD x P(decisive) | +0.019 (66%) / +0.004 | +0.043 (66%) / +0.001 | +0.034 (66%) / +0.013 | +0.020 (60%) / +0.061 |
| Laplace BALD | +0.017 (60%) / -0.001 | +0.039 (69%) / -0.015 | +0.014 (57%) / -0.015 | -0.021 (40%) / +0.023 |
| Core-set | +0.024 (63%) / -0.010 | +0.065* (71%) / +0.023 | +0.002 (60%) / +0.029 | +0.029 (60%) / +0.063 |
| Core-set, relation-aware pairs | +0.050 (69%) / +0.041 | +0.051 (54%) / +0.052 | +0.058* (77%) / -0.006 | +0.048 (69%) / +0.063 |
| DPP | -0.010 (57%) / -0.019 | -0.002 (43%) / -0.062 | -0.035 (49%) / -0.104 | -0.043 (43%) / -0.060 |
| Uncertainty (own type) | -0.025 (46%) / +0.012 | -0.024 (43%) / -0.007 | -0.021 (40%) / -0.047 | -0.039 (34%) / +0.024 |
| Uncertainty, all heads | -0.004 (54%) / +0.009 | -0.014 (46%) / -0.006 | -0.031 (40%) / -0.049 | +0.005 (49%) / +0.013 |
| Largest predicted gap | +0.015 (51%) / -0.005 | +0.036 (60%) / -0.012 | -0.027 (49%) / +0.001 | -0.015 (37%) / +0.051 |
| Random pair, random type | +0.002 (46%) / +0.003 | +0.026 (51%) / +0.015 | -0.006 (40%) / -0.012 | -0.007 (37%) / +0.024 |

AUC gain:

| Variant | A single: new / main | A seq.: new / main | B single: new / main | B seq.: new / main |
|---|---:|---:|---:|---:|
| Cluster-quota uncertainty, all heads (original code) | -0.005 (37%) / -0.007 | +0.000 (49%) / -0.008 | -0.008 (26%) / -0.011 | +0.003 (49%) / +0.005 |
| Cluster-quota uncertainty (own type) | -0.003 (46%) / +0.001 | +0.002 (43%) / +0.005 | +0.002 (60%) / -0.003 | +0.006 (71%) / +0.009 |
| Deep-ensemble BALD x P(decisive) | +0.007 (69%) / +0.010 | +0.012 (69%) / +0.011 | +0.010* (77%) / +0.001 | +0.014 (69%) / +0.014 |
| Deep-ensemble BALD | +0.006 (63%) / +0.007 | +0.011 (71%) / +0.006 | +0.011* (80%) / +0.002 | +0.015 (71%) / +0.004 |
| Fisher D-optimal | +0.005 (71%) / +0.006 | +0.014* (69%) / +0.010 | +0.011* (74%) / +0.003 | +0.015* (69%) / +0.011 |
| BALD x P(decisive) | +0.008 (71%) / +0.007 | +0.013 (57%) / +0.003 | +0.008 (66%) / +0.006 | +0.005 (60%) / +0.012 |
| Laplace BALD | +0.005 (66%) / +0.004 | +0.011 (69%) / +0.003 | +0.006 (63%) / +0.000 | +0.001 (49%) / +0.006 |
| Core-set | +0.001 (60%) / -0.007 | +0.011* (74%) / -0.001 | -0.010 (43%) / -0.003 | -0.002 (51%) / +0.006 |
| Core-set, relation-aware pairs | +0.009 (69%) / -0.007 | +0.006 (51%) / +0.002 | +0.004 (66%) / -0.005 | +0.003 (51%) / +0.007 |
| DPP | +0.003 (63%) / +0.001 | +0.007 (60%) / -0.004 | -0.004 (51%) / -0.013 | -0.002 (49%) / -0.004 |
| Uncertainty (own type) | -0.005 (51%) / +0.002 | -0.003 (43%) / +0.004 | -0.006 (46%) / -0.013 | -0.007 (34%) / +0.003 |
| Uncertainty, all heads | -0.011 (37%) / -0.005 | -0.015 (34%) / -0.012 | -0.015* (29%) / -0.026 | -0.011 (26%) / -0.014 |
| Largest predicted gap | -0.002 (46%) / -0.008 | +0.003 (51%) / -0.010 | -0.018* (26%) / -0.014 | -0.016* (23%) / -0.004 |
| Random pair, random type | -0.002 (40%) / -0.003 | +0.003 (49%) / +0.001 | -0.006 (29%) / -0.003 | -0.005 (26%) / +0.000 |

Decisive-accuracy gain:

| Variant | A single: new / main | A seq.: new / main | B single: new / main | B seq.: new / main |
|---|---:|---:|---:|---:|
| Cluster-quota uncertainty, all heads (original code) | +0.002 (60%) / -0.002 | +0.004 (54%) / -0.005 | +0.001 (57%) / -0.003 | +0.010 (69%) / +0.008 |
| Cluster-quota uncertainty (own type) | +0.002 (51%) / +0.007 | +0.008 (54%) / +0.007 | +0.004 (60%) / +0.007 | +0.016* (74%) / +0.016 |
| Deep-ensemble BALD x P(decisive) | +0.011 (69%) / +0.015 | +0.019* (77%) / +0.020 | +0.012 (69%) / +0.006 | +0.023* (83%) / +0.022 |
| Deep-ensemble BALD | +0.008 (60%) / +0.012 | +0.020* (71%) / +0.019 | +0.013* (69%) / +0.010 | +0.025* (74%) / +0.019 |
| Fisher D-optimal | +0.010* (71%) / +0.011 | +0.021* (77%) / +0.024 | +0.021* (80%) / +0.011 | +0.030* (83%) / +0.020 |
| BALD x P(decisive) | +0.011 (66%) / +0.009 | +0.017* (74%) / +0.012 | +0.012 (69%) / +0.008 | +0.011 (69%) / +0.016 |
| Laplace BALD | +0.009 (66%) / +0.009 | +0.014* (69%) / +0.009 | +0.010 (63%) / +0.004 | +0.010 (66%) / +0.013 |
| Core-set | +0.004 (51%) / -0.006 | +0.010 (63%) / +0.001 | -0.008 (46%) / -0.004 | +0.002 (46%) / +0.004 |
| Core-set, relation-aware pairs | +0.008 (51%) / -0.008 | +0.005 (49%) / +0.004 | +0.001 (57%) / -0.005 | +0.001 (57%) / +0.009 |
| DPP | +0.010 (71%) / +0.009 | +0.015* (71%) / +0.008 | +0.001 (49%) / -0.001 | +0.010 (63%) / +0.010 |
| Uncertainty (own type) | +0.002 (51%) / +0.010 | +0.007 (63%) / +0.014 | -0.002 (43%) / -0.003 | +0.004 (54%) / +0.011 |
| Uncertainty, all heads | -0.006 (46%) / -0.000 | -0.003 (49%) / -0.002 | -0.001 (54%) / -0.014 | +0.003 (60%) / +0.002 |
| Largest predicted gap | -0.009 (37%) / -0.011 | -0.004 (40%) / -0.011 | -0.024 (34%) / -0.024 | -0.020* (29%) / -0.015 |
| Random pair, random type | +0.001 (57%) / -0.002 | +0.004 (57%) / +0.003 | -0.003 (43%) / -0.000 | -0.003 (43%) / +0.004 |

Observations (exploratory; 156 variant-cell-metric entries, so a few nominal hits are expected by chance):
- Ranking consistency between the main and new seeds (Spearman correlation of the 13 variants' mean gains) is low for log-loss (0.03 / 0.34 / 0.36 / 0.65 in A single / A sequential / B single / B sequential) and high for decisive accuracy (0.64 / 0.88 / 0.89 / 0.87) and AUC (0.45 / 0.62 / 0.88 / 0.76). The per-rule ordering by log-loss is mostly noise at this test-set size; the accuracy ordering is stable.
- Several rules that were not significant in the main run look good in single cells of the new seeds, for example core-set (+0.065, Holm 0.022) in Split A sequential log-loss, core-set relation-aware pairs (+0.058, Holm 0.017) and ensemble BALD (+0.033, Holm 0.036) in Split B single log-loss; the corresponding main-run values are +0.023, -0.006 and -0.020. These flip sign between seed sets and should be treated as chance until re-tested.
- The accuracy gain of ensemble BALD / Fisher D-optimal is also visible for BALD x P(decisive), Laplace BALD (smaller) and DPP in Split A sequential, i.e. it is not specific to the three pre-specified variants; Largest predicted gap is worse than Random in Split B in AUC (-0.018 single-shot, -0.016 sequential; Holm 0.025 / 0.022) and accuracy (-0.024, Holm 0.057; -0.020, Holm 0.013); in the main run it was significantly worse in Split B single-shot accuracy.
- `random_pair_type` against Random (two-sided, uncorrected, 12 comparisons): indistinguishable in 10 of 12; in Split B it has slightly lower AUC (-0.006, p = 0.004 single-shot; -0.005, p = 0.019 sequential). In the main run the two baselines were indistinguishable everywhere. (`analysis/random_pair_type_vs_random.csv`.)

## 6. Checks

- Cell counts recounted from the files (see section 2): 12,880 cells, 805 cell files per (split, condition), 35 seeds x 23 names each, `n_revealed == budget` everywhere, asserted in `judgment_unit_replication_analysis.py`.
- Hand check with `judgment_unit_verify.py` (re-derives pool and initial set from the manifest, retrains the head from the stored selection, compares metrics, checks that no validation/test image is in the labelled set or the pool and that no unrevealed judgment of a selected pair is in the training rows): (Split A single seed 510 core-set budget 20: pool 202 judgments from 66 groups, 20 revealed from 20 distinct pairs, training rows 49 = 29 + 20, 0 shared held-out images, 0 hidden judgments in the training rows, log-loss 0.426513 / AUC 0.929894 / accuracy 0.872727 identical to six decimals to the stored cell), (Split A sequential seed 523 Fisher D-optimal budget 40: 70 = 30 + 40 rows, metrics 0.401692 / 0.915623 / 0.844961 identical) and (Split B sequential seed 517 deep-ensemble BALD x P(decisive) budget 40: pool 337 judgments from 108 groups, 73 = 33 + 40 rows, metrics 0.203438 / 0.977564 / 0.9 identical). The analysis was spot-checked by hand for one seed and cell (Split B sequential seed 503, Fisher D-optimal, accuracy: gain +0.060345 from the raw cell files, identical to the script).
- Tests: `pytest` in `code/` = 163 passed (same as before this work). Running the suite rewrites a few tracked files under `results/active_learning_v1.8_seed42/manifests/`; these were restored with `git checkout` and are not part of the commits.

## 7. Caveats

- Same 168 groups (section 1); one encoder, one schedule; test sets of about 54-63 decisive judgments per seed.
- Holm was applied over the pre-registered family (18 tests; H6: 8 tests), not over the 13 variants of the main report, so the confirmatory Holm p-values are not comparable with the main tables (which use 32 variants per table). Decisions were made before the run; the verdict labels are binary by design and the near-misses (H3 ensemble BALD A single, p = 0.059; H1 Holm 0.110) are labelled "not replicated" accordingly.
- Everything in section 5 is exploratory.
