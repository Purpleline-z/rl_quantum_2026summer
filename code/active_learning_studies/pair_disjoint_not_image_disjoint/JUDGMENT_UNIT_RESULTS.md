# Judgment-unit re-run: the query unit is one (pair, type) judgment

Branch `claude/judgment-unit-rerun` (from `claude/frozen-encoder-strategies`, ca96d5e). Code: `judgment_unit_study.py` (runner), `judgment_unit_strategies.py` (selectors),
`judgment_unit_aggregate.py` / `judgment_unit_tables.py` / `judgment_unit_composition.py` / `judgment_unit_sensitivity.py` / `judgment_unit_verify.py` (analysis), tests in
`code_behavior_tests/test_judgment_unit.py`. Results: `results/judgment_unit_study/{A_groups,B_groups,A_random,B_random}/` (`single/`, `sequential/` cells, `manifest_seed*.json`, `analysis/`).
The earlier group-unit study (`results/pair_endpoint_study/`) is untouched.

## 1. Plain-language summary

With the query unit changed to a single (pair, type) judgment and the budget counted in judgments (10/20/40/60), the strategy comparison again finds **no strategy that is robustly better than random selection**. Everything below is from 35 seeds in two splits (Split A: small split with validation; Split B: classifier2-style 20% hold-out) and two conditions (single batch; sequential rounds of 10), 33 strategy variants plus Random, 23,520 cells in the main run.

- **Primary endpoints (log-loss, calibrated log-loss, AUC): almost nothing survives Holm.** Split A, either condition: no variant beats Random. Split B, single batch: none. Split B, sequential: the all-head version of the original cluster-quota rule gains +0.074 log-loss (Holm p < 0.001, 80% of seeds better; its own-type counterpart is +0.039, not significant, and the same all-head rule is +0.016 to +0.022 and not significant in Split A, -0.005 in Split B single-batch) and deep-ensemble BALD x P(decisive) gains +0.014 AUC (Holm p = 0.023). Several rules are significantly *worse* than Random in Split B single-batch (DPP -0.104 log-loss; all-head uncertainty -0.026 AUC).
- **The significant results of the group-unit study did not replicate.** Laplace BALD and BALD x P(decisive) in Split A (earlier log-loss gain +0.038 and AUC gain +0.010, Holm significant) are now about +0.00 log-loss and +0.004 to +0.007 AUC and not significant; core-set in Split B (earlier +0.052 / +0.053, significant) keeps its sign (+0.029 single, +0.063 sequential, 60-66% of seeds better) but is not significant after Holm. So the earlier "which rule wins" statements depend on the query unit, as well as on the split.
- **A weak, secondary signal:** decisive-judgment *accuracy* (a metric the earlier study also computed, with no Holm-significant rule in any of its four cells) is now 1-2.5 points higher than Random for deep-ensemble BALD (with and without P(decisive)) and Fisher D-optimal design, Holm-significant for ensemble BALD x P(decisive) in three of the four main cells (Split A single +0.015, Split A sequential +0.020, Split B sequential +0.022) and for Fisher D-optimal in two (Split A and B sequential, +0.024 / +0.020). These rules choose the type with the model. The AUC gain is only +0.00 to +0.014, and with 20 seeds and 10 random initial judgments nothing is significant, though the sign of the accuracy gain stays positive in all four cells. I treat this as a hypothesis, not a finding.
- Uncertainty-driven rules (Uncertainty, MC-dropout, BADGE, DropQuery) are again not better than Random; plain uncertainty selects decisive judgments only 28-31% of the time (Random: 50%; the rest are tie / not_apply).
- The budgets and the number of labelled rows are not comparable with the old study: a budget of 60 old groups is about 190 judgments, and Random log-loss at 60 judgments (0.47-0.51) is far above that at 60 old groups (0.35-0.37).

## 2. What changed relative to the group-unit study

| | Group-unit study (earlier) | Judgment-unit study (this note) |
|---|---|---|
| query unit | an image pair; selecting it reveals ALL its recorded judgments (about 3.1 on average, one per type) | ONE (pair, type) judgment; only that row's outcome (1 / 2 / tie / not_apply) is revealed |
| budget | 10, 20, 40, 60 pair groups (= about 31, 62, 124, 186 judgments) | 10, 20, 40, 60 judgments |
| candidate pool | pool groups (64-77 in Split A, 99-113 in Split B); type = first row's type | every judgment row of the pool groups (see section 4); each candidate carries its own type |
| who chooses the type | the data (all types of the group are revealed) | the strategy (it may choose any type of any pair), or a random type for rules that cannot score a type |
| training rows | all rows of the labelled groups | exactly the revealed judgments (plus the initial set); an unselected judgment of a selected pair is NOT in the training set (test: `test_unselected_judgment_of_a_selected_pair_is_not_trained_on`) |
| splits, seeds, features, head, schedule, hold-out sets, endpoint, statistics | identical: seeds 42, 79, 123, 202, 303, 400-429 (35), `split_heldout` (Split A: 10 initial / 20 validation / 40 test groups) and `split_classifier2_style` (Split B: 20% of groups = test, no validation), SHA-256 identity-safe image disjointness, frozen SimCLR features, lr 0.01 / 100 steps from `schedule.json`; test and validation sets are the same groups and are evaluated on their decisive judgments | same (imported from the earlier modules, not copied) |

Conditions: **single-shot** (one batch of the budget size chosen with the model trained on the initial set) and **sequential** (rounds of 10 judgments, head retrained after each round, checkpoints at 10/20/40/60).
Budget 100 was not run (the pair-level rules below pick one judgment per pair, and Split A has only 64-77 pool pairs).
Random = uniform over the remaining candidate judgments, mean of 5 draws within a seed (as before). An additional baseline `random_pair_type` (uniform pair, then uniform type of that pair; 5 draws averaged) is reported separately and is not part of the Holm family.

**Initial labelled set (decision).** Main run: the judgments of the same 10 initial groups are revealed (about 30 judgments: mean 30.5 (range 14-38) in Split A, 30.2 (range 21-38) in Split B). Reasons: it keeps the cold-start data identical to the earlier study, so the first model, the validation/test groups and the pool differ from the old design only by the query unit; it is also what a labelling project would have before the first query. Sensitivity run (`A_random`, `B_random`): 10 random judgments drawn from the judgments of the initial and pool groups (20 seeds: 42, 79, 123, 202, 303, 400-414), the rest being the pool. I make no claim about cold-start type coverage (the random start does not guarantee that every type appears).
The budgets are therefore **not comparable** with the old ones: 10 old groups are about 31 judgments, and each old strategy also received free extra types it had not chosen. The old results are shown next to the new ones only for orientation.

Tie-breaking: candidate order is shuffled with the seed before each selection, so equal scores or identical pair vectors are broken at random, not by file order.

## 3. Strategy adaptations (every one of the 33 variants)

Families: **R** = row level, scored with the head of the candidate's own type (the strategy may pick any type); **P** = pair level (the rule works on the unique image pairs because its structure lives in pair space; one judgment of each chosen pair is revealed; the type is a random remaining type, or, for rules with an uncertainty quality score, the most uncertain remaining type, and the pair's quality is the largest own-head uncertainty among its remaining judgments); **L** = all-head variant, which by definition cannot score a type: the pair is chosen by the all-head score and a random remaining type of it is revealed.
If a pair-level rule is asked for more judgments than there are candidate pairs it is run again on what is left (not triggered at budgets up to 60).

| Strategy variant | Family | Adaptation |
|---|---|---|
| Random | - | uniform over remaining judgments |
| Uncertainty | R | original code; entropy of the candidate's own head (the original already used `type_idx`; now it is each row's own type) |
| Cluster-quota uncertainty | R | original rule (quota of the most uncertain per k-means cluster of the pair, then fill) re-implemented on judgments; own-head entropy; the original's one-per-image-pair de-duplication is dropped because the unit is the judgment |
| Uncertainty + diversity | R | original code; own-head entropy; diversity = distance in pair space to labelled and selected pairs (a sibling judgment of an already chosen pair has distance 0) |
| Cluster-Margin | R | original code; own-head margin; clusters from the pair |
| MC-dropout variance / mutual information | R | original code; own head |
| Core-set | P | original code on the unique pairs (farthest-first in pair-vector space), random type |
| Core-set, relation-aware pairs; TypiClust; ProbCover; MaxHerding; Graph cut | P | rule unchanged on unique pairs (no model score involved except where noted), random type |
| Graph facility location; DPP; FASS | P (uncertainty quality) | rule unchanged on unique pairs, with the uncertainty term = largest own-head uncertainty among the pair's remaining judgments; the most uncertain remaining type is revealed |
| DropQuery | R / P hybrid | first half of the budget: pair nearest each k-means centre (random type); second half: most uncertain own-head judgments |
| BADGE | R | gradient embedding of the own head only (block of the candidate's type), k-means++ |
| Fisher D-optimal | R | gain log(1 + w phi' M_k^-1 phi) for the candidate's own head k; a revealed judgment updates only its own head's information matrix |
| Laplace BALD | R | BALD of the own head under the last-layer Laplace posterior; a revealed judgment shrinks only its own head's posterior |
| BALD x P(decisive) | R | as Laplace BALD, times P(decisive) of (own type, pair features) from the logistic model fitted on the revealed judgments' outcomes |
| Deep-ensemble BALD (8 heads), x P(decisive) | R | mutual information of the own head across the 8 differently initialised heads (optionally x P(decisive)) |
| Largest predicted gap; Gap + posterior std | R | own-head |logit| (plus posterior std, batch aware per head) |
| Image-coverage uncertainty | R | own-head uncertainty x (1 + number of unseen images) |
| Uncertainty, all heads | L | mean entropy over the four active heads, one random type of the chosen pair |
| Uncertainty / Cluster-quota / Uncertainty + diversity / Cluster-Margin / MC-dropout variance / MC-dropout mutual info, "original code, all heads" (`*_lf`) | L | original code with the type removed and the unused Twinned head silenced (as in the old study), run on the unique pairs; one random type of each chosen pair is revealed |

The all-head notion therefore no longer means "reveals everything about the group": it only means "scores the pair, not the type". Comparing "Uncertainty" with "Uncertainty, all heads" isolates the value of choosing the type with the model.
`Uncertainty, all heads` and `Uncertainty, original code, all heads` are the same rule here and give identical numbers (the old study gave identical numbers for them as well).

## 4. Pool sizes (per split; recounted from the manifests)

| Split / initial set | seeds | initial judgments (mean, range) | pool judgments (mean, range) | pool pairs (mean, range) | judgments per pool pair | validation decisive judgments | test decisive judgments (mean, range) |
|---|---:|---|---|---|---:|---:|---|
| A_groups | 35 | 30.5 (14–38) | 220.3 (195–250) | 70.8 (64–77) | 3.11 | 20 groups | 63.3 (51–79) |
| A_random | 20 | 10.0 (10–10) | 243.1 (224–265) | 81.1 (73–87) | 3.00 | 20 groups | 61.8 (51–79) |
| B_groups | 35 | 30.2 (21–38) | 329.2 (306–350) | 106.3 (99–113) | 3.10 | none | 53.5 (42–67) |
| B_random | 20 | 10.0 (10–10) | 353.1 (331–371) | 117.0 (110–122) | 3.02 | none | 53.0 (42–63) |

The old design's pools were the same groups: about 71 (Split A) and 106 (Split B) pool groups; the new pools are about 3.1 times larger in candidates. Budget 60 judgments is therefore about 27% (Split A) or 18% (Split B) of the pool, against 85% / 57% of the pool groups in the old design.
(Decisive test judgments: Split A about 63, Split B about 54, as before.)

## 5. Main results (per-seed gain over Random, averaged over budgets 10/20/40/60; Wilcoxon over 35 seeds, Holm within each table over the 32 non-random variants)

Gain = improvement over Random (positive is better; log-loss lower is better). "OLD" columns: the same strategy in the group-unit study (budgets in groups, not comparable) for the same split and condition. Rows ordered by log-loss gain. The Friedman omnibus test over the 33 variants and the per-seed forest data are in `results/judgment_unit_study/<run>/analysis/report.txt` and the CSVs.

### 5.1 Split A, single-shot
Friedman: log-loss chi2=28.5, p=0.6428; cal. log-loss chi2=63.0, p=0.0009; AUC chi2=67.0, p=0.0003 (33 variants, 35 seeds)

| Strategy | log-loss gain | Holm p | cal. log-loss gain | Holm p | AUC gain | Holm p | seeds better (log-loss) | OLD group-unit: log-loss gain | Holm p | AUC gain | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Core-set, relation-aware pairs | +0.041 | 1.000 | +0.001 | 1.000 | -0.007 | 1.000 | 57% | +0.036 | 0.176 | +0.006 | 1.000 |
| MaxHerding (pairs) | +0.032 | 1.000 | +0.014 | 1.000 | +0.005 | 1.000 | 60% | +0.031 | 0.122 | +0.007 | 0.601 |
| Uncertainty + diversity | +0.024 | 1.000 | +0.015 | 1.000 | +0.004 | 1.000 | 57% | +0.015 | 1.000 | +0.001 | 1.000 |
| Cluster-quota uncertainty, original code, all heads | +0.022 | 1.000 | -0.046 | 0.584 | -0.007 | 1.000 | 60% | +0.029 | 0.176 | +0.001 | 1.000 |
| Cluster-Margin | +0.021 | 1.000 | +0.022 | 1.000 | -0.001 | 1.000 | 60% | +0.033 | 0.285 | +0.002 | 1.000 |
| Uncertainty + diversity, original code, all heads | +0.018 | 1.000 | -0.056 | 1.000 | -0.007 | 1.000 | 60% | +0.002 | 1.000 | -0.006 | 1.000 |
| Deep-ensemble BALD x P(decisive) | +0.016 | 1.000 | -0.008 | 1.000 | +0.010 | 0.308 | 51% | +0.004 | 1.000 | +0.005 | 1.000 |
| Cluster-quota uncertainty | +0.014 | 1.000 | +0.015 | 1.000 | +0.001 | 1.000 | 60% | +0.038 | 0.033 | +0.005 | 1.000 |
| Fisher D-optimal (Active Reward Modeling) | +0.014 | 1.000 | +0.018 | 1.000 | +0.006 | 1.000 | 49% | +0.041 | 0.155 | +0.008 | 0.909 |
| Cluster-Margin, original code, all heads | +0.014 | 1.000 | -0.024 | 0.746 | -0.014 | 0.547 | 49% | +0.018 | 1.000 | -0.003 | 1.000 |
| Uncertainty | +0.012 | 1.000 | -0.003 | 1.000 | +0.002 | 1.000 | 54% | +0.007 | 1.000 | +0.001 | 1.000 |
| FASS (pairs) | +0.011 | 1.000 | +0.014 | 1.000 | +0.005 | 1.000 | 60% | +0.025 | 1.000 | +0.003 | 1.000 |
| Uncertainty, original code, all heads | +0.009 | 1.000 | -0.017 | 1.000 | -0.005 | 1.000 | 60% | +0.003 | 1.000 | -0.004 | 1.000 |
| Uncertainty, all heads | +0.009 | 1.000 | -0.017 | 1.000 | -0.005 | 1.000 | 60% | +0.003 | 1.000 | -0.004 | 1.000 |
| BALD x P(decisive) | +0.004 | 1.000 | -0.003 | 1.000 | +0.007 | 1.000 | 49% | +0.038 | 0.026 | +0.010 | 0.027 |
| ProbCover (pairs) | +0.004 | 1.000 | +0.014 | 1.000 | -0.002 | 1.000 | 54% | +0.016 | 1.000 | +0.005 | 1.000 |
| Laplace BALD | -0.001 | 1.000 | +0.006 | 1.000 | +0.004 | 1.000 | 51% | +0.038 | 0.022 | +0.010 | 0.031 |
| MC-dropout mutual info | -0.001 | 1.000 | -0.029 | 1.000 | +0.000 | 1.000 | 60% | +0.012 | 1.000 | +0.004 | 1.000 |
| Largest predicted gap | -0.005 | 1.000 | -0.013 | 1.000 | -0.008 | 1.000 | 54% | +0.017 | 1.000 | +0.005 | 1.000 |
| Deep-ensemble BALD (8 heads) | -0.005 | 1.000 | -0.003 | 1.000 | +0.007 | 1.000 | 57% | +0.013 | 1.000 | +0.005 | 1.000 |
| TypiClust (pairs) | -0.005 | 1.000 | +0.011 | 1.000 | -0.003 | 1.000 | 60% | +0.018 | 1.000 | +0.006 | 1.000 |
| Graph facility location (uncertainty-weighted) | -0.007 | 1.000 | +0.021 | 1.000 | +0.005 | 1.000 | 66% | +0.020 | 1.000 | +0.004 | 1.000 |
| DropQuery (pairs) | -0.008 | 1.000 | +0.003 | 1.000 | -0.006 | 1.000 | 43% | -0.016 | 1.000 | -0.004 | 1.000 |
| MC-dropout variance | -0.008 | 1.000 | -0.023 | 1.000 | -0.002 | 1.000 | 54% | +0.019 | 1.000 | +0.003 | 1.000 |
| Core-set | -0.010 | 1.000 | +0.004 | 1.000 | -0.007 | 1.000 | 54% | +0.023 | 1.000 | -0.002 | 1.000 |
| MC-dropout variance, original code, all heads | -0.012 | 1.000 | -0.053 | 1.000 | -0.008 | 1.000 | 49% | -0.019 | 1.000 | -0.007 | 1.000 |
| MC-dropout mutual info, original code, all heads | -0.014 | 1.000 | -0.030 | 1.000 | -0.012 | 0.909 | 46% | -0.015 | 1.000 | -0.007 | 1.000 |
| Image-coverage uncertainty | -0.018 | 1.000 | -0.003 | 1.000 | -0.000 | 1.000 | 51% | +0.010 | 1.000 | -0.000 | 1.000 |
| DPP (quality x diversity) | -0.019 | 1.000 | +0.019 | 1.000 | +0.001 | 1.000 | 40% | +0.028 | 0.047 | +0.005 | 1.000 |
| Graph cut (pairs) | -0.022 | 1.000 | +0.003 | 1.000 | -0.001 | 1.000 | 46% | -0.058 | 0.047 | -0.000 | 1.000 |
| BADGE (pairs) | -0.023 | 1.000 | +0.018 | 1.000 | +0.003 | 1.000 | 51% | +0.016 | 1.000 | +0.002 | 1.000 |
| Gap + posterior std (DeltaUCB-style) | -0.023 | 1.000 | -0.007 | 1.000 | -0.012 | 0.540 | 40% | -0.001 | 1.000 | +0.001 | 1.000 |

### 5.2 Split A, sequential
Friedman: log-loss chi2=56.3, p=0.0050; cal. log-loss chi2=81.9, p=0.0000; AUC chi2=79.7, p=0.0000 (33 variants, 35 seeds)

| Strategy | log-loss gain | Holm p | cal. log-loss gain | Holm p | AUC gain | Holm p | seeds better (log-loss) | OLD group-unit: log-loss gain | Holm p | AUC gain | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Core-set, relation-aware pairs | +0.052 | 0.566 | +0.051 | 1.000 | +0.002 | 1.000 | 66% | +0.023 | 1.000 | +0.002 | 1.000 |
| Uncertainty + diversity | +0.043 | 0.659 | +0.036 | 1.000 | +0.007 | 1.000 | 66% | -0.000 | 1.000 | -0.002 | 1.000 |
| Cluster-Margin, original code, all heads | +0.039 | 1.000 | +0.031 | 1.000 | -0.008 | 1.000 | 60% | +0.010 | 1.000 | -0.004 | 1.000 |
| ProbCover (pairs) | +0.038 | 1.000 | +0.054 | 1.000 | +0.009 | 0.750 | 63% | -0.004 | 1.000 | +0.002 | 1.000 |
| Cluster-Margin | +0.034 | 1.000 | +0.050 | 0.377 | +0.009 | 0.453 | 66% | +0.012 | 1.000 | -0.005 | 1.000 |
| Cluster-quota uncertainty | +0.034 | 1.000 | +0.048 | 1.000 | +0.005 | 1.000 | 63% | +0.015 | 1.000 | -0.002 | 1.000 |
| Core-set | +0.023 | 1.000 | +0.026 | 1.000 | -0.001 | 1.000 | 63% | +0.010 | 1.000 | -0.005 | 1.000 |
| Uncertainty + diversity, original code, all heads | +0.017 | 1.000 | +0.011 | 1.000 | -0.012 | 0.153 | 63% | +0.001 | 1.000 | -0.005 | 1.000 |
| Cluster-quota uncertainty, original code, all heads | +0.016 | 1.000 | -0.013 | 1.000 | -0.008 | 1.000 | 54% | -0.005 | 1.000 | -0.007 | 0.452 |
| FASS (pairs) | +0.015 | 1.000 | +0.051 | 1.000 | +0.006 | 1.000 | 54% | +0.013 | 1.000 | -0.001 | 1.000 |
| MaxHerding (pairs) | +0.008 | 1.000 | +0.024 | 1.000 | -0.001 | 1.000 | 51% | +0.021 | 1.000 | +0.004 | 1.000 |
| TypiClust (pairs) | +0.008 | 1.000 | +0.036 | 1.000 | +0.004 | 1.000 | 54% | -0.007 | 1.000 | +0.001 | 1.000 |
| BALD x P(decisive) | +0.001 | 1.000 | +0.035 | 1.000 | +0.003 | 1.000 | 57% | +0.017 | 1.000 | +0.004 | 1.000 |
| MC-dropout mutual info, original code, all heads | +0.001 | 1.000 | +0.026 | 1.000 | -0.004 | 1.000 | 57% | -0.032 | 1.000 | -0.010 | 1.000 |
| MC-dropout variance, original code, all heads | -0.003 | 1.000 | -0.005 | 1.000 | -0.005 | 1.000 | 51% | -0.025 | 1.000 | -0.009 | 1.000 |
| Deep-ensemble BALD x P(decisive) | -0.004 | 1.000 | +0.032 | 1.000 | +0.011 | 0.448 | 60% | -0.021 | 1.000 | -0.000 | 1.000 |
| MC-dropout variance | -0.005 | 1.000 | +0.018 | 1.000 | -0.002 | 1.000 | 51% | -0.000 | 1.000 | -0.001 | 1.000 |
| Uncertainty, original code, all heads | -0.006 | 1.000 | -0.015 | 1.000 | -0.012 | 0.331 | 49% | -0.023 | 1.000 | -0.009 | 0.145 |
| Uncertainty, all heads | -0.006 | 1.000 | -0.015 | 1.000 | -0.012 | 0.331 | 49% | -0.023 | 1.000 | -0.009 | 0.145 |
| Uncertainty | -0.007 | 1.000 | +0.023 | 1.000 | +0.004 | 1.000 | 46% | -0.011 | 1.000 | -0.005 | 1.000 |
| Deep-ensemble BALD (8 heads) | -0.008 | 1.000 | -0.031 | 1.000 | +0.006 | 1.000 | 49% | -0.009 | 1.000 | -0.001 | 1.000 |
| Fisher D-optimal (Active Reward Modeling) | -0.010 | 1.000 | +0.013 | 1.000 | +0.010 | 0.365 | 54% | +0.013 | 1.000 | +0.002 | 1.000 |
| DropQuery (pairs) | -0.011 | 1.000 | +0.029 | 1.000 | -0.003 | 1.000 | 40% | -0.039 | 0.491 | -0.008 | 1.000 |
| Largest predicted gap | -0.012 | 1.000 | +0.009 | 1.000 | -0.010 | 1.000 | 43% | +0.001 | 1.000 | +0.001 | 1.000 |
| Laplace BALD | -0.015 | 1.000 | +0.045 | 1.000 | +0.003 | 1.000 | 57% | +0.016 | 1.000 | +0.004 | 1.000 |
| Image-coverage uncertainty | -0.016 | 1.000 | -0.025 | 1.000 | +0.002 | 1.000 | 51% | -0.007 | 1.000 | -0.004 | 1.000 |
| MC-dropout mutual info | -0.023 | 1.000 | -0.003 | 1.000 | -0.003 | 1.000 | 51% | -0.009 | 1.000 | -0.003 | 1.000 |
| BADGE (pairs) | -0.031 | 1.000 | +0.043 | 1.000 | +0.004 | 1.000 | 40% | -0.009 | 1.000 | -0.001 | 1.000 |
| Gap + posterior std (DeltaUCB-style) | -0.037 | 1.000 | +0.017 | 1.000 | -0.016 | 0.359 | 37% | -0.020 | 1.000 | -0.005 | 1.000 |
| Graph facility location (uncertainty-weighted) | -0.037 | 1.000 | +0.036 | 1.000 | +0.000 | 1.000 | 49% | -0.000 | 1.000 | -0.001 | 1.000 |
| Graph cut (pairs) | -0.057 | 0.531 | +0.018 | 1.000 | -0.000 | 1.000 | 31% | -0.049 | 0.015 | -0.001 | 1.000 |
| DPP (quality x diversity) | -0.062 | 1.000 | +0.014 | 1.000 | -0.004 | 1.000 | 34% | -0.002 | 1.000 | -0.002 | 1.000 |

### 5.3 Split B (classifier2-style), single-shot
Friedman: log-loss chi2=67.0, p=0.0003; AUC chi2=100.5, p=0.0000 (33 variants, 35 seeds)

| Strategy | log-loss gain | Holm p | AUC gain | Holm p | seeds better (log-loss) | OLD group-unit: log-loss gain | Holm p | AUC gain | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Core-set | +0.029 | 1.000 | -0.003 | 1.000 | 60% | +0.052 | 0.012 | +0.003 | 1.000 |
| Cluster-Margin, original code, all heads | +0.014 | 1.000 | -0.006 | 1.000 | 57% | +0.012 | 1.000 | -0.008 | 1.000 |
| BALD x P(decisive) | +0.013 | 1.000 | +0.006 | 1.000 | 60% | +0.037 | 0.314 | +0.008 | 0.654 |
| Gap + posterior std (DeltaUCB-style) | +0.003 | 1.000 | -0.012 | 0.058 | 49% | +0.027 | 1.000 | +0.001 | 1.000 |
| Largest predicted gap | +0.001 | 1.000 | -0.014 | 0.461 | 54% | +0.015 | 1.000 | -0.001 | 1.000 |
| MaxHerding (pairs) | +0.001 | 1.000 | -0.001 | 1.000 | 51% | +0.014 | 1.000 | +0.004 | 1.000 |
| MC-dropout variance, original code, all heads | -0.003 | 1.000 | -0.007 | 1.000 | 43% | -0.020 | 1.000 | -0.007 | 1.000 |
| Cluster-quota uncertainty, original code, all heads | -0.005 | 1.000 | -0.011 | 0.139 | 43% | -0.010 | 1.000 | -0.011 | 0.219 |
| Core-set, relation-aware pairs | -0.006 | 1.000 | -0.005 | 1.000 | 54% | +0.042 | 0.391 | +0.004 | 1.000 |
| TypiClust (pairs) | -0.011 | 1.000 | -0.004 | 1.000 | 57% | +0.024 | 1.000 | +0.006 | 1.000 |
| Cluster-quota uncertainty | -0.012 | 1.000 | -0.003 | 1.000 | 43% | -0.015 | 1.000 | -0.008 | 1.000 |
| ProbCover (pairs) | -0.014 | 1.000 | -0.002 | 1.000 | 46% | +0.025 | 1.000 | +0.005 | 1.000 |
| Laplace BALD | -0.015 | 1.000 | +0.000 | 1.000 | 46% | +0.027 | 1.000 | +0.003 | 1.000 |
| DropQuery (pairs) | -0.018 | 1.000 | -0.007 | 1.000 | 49% | -0.001 | 1.000 | -0.006 | 1.000 |
| Uncertainty + diversity, original code, all heads | -0.019 | 1.000 | -0.016 | 0.053 | 51% | -0.001 | 1.000 | -0.007 | 0.654 |
| MC-dropout variance | -0.019 | 1.000 | -0.007 | 1.000 | 46% | -0.029 | 1.000 | -0.009 | 0.183 |
| MC-dropout mutual info | -0.020 | 1.000 | -0.005 | 1.000 | 43% | -0.018 | 1.000 | -0.006 | 0.819 |
| Deep-ensemble BALD (8 heads) | -0.020 | 1.000 | +0.002 | 1.000 | 43% | +0.021 | 1.000 | +0.007 | 0.654 |
| MC-dropout mutual info, original code, all heads | -0.027 | 1.000 | -0.014 | 0.040 | 37% | -0.006 | 1.000 | -0.005 | 1.000 |
| Fisher D-optimal (Active Reward Modeling) | -0.031 | 1.000 | +0.003 | 1.000 | 46% | +0.020 | 1.000 | +0.002 | 1.000 |
| BADGE (pairs) | -0.034 | 0.926 | +0.002 | 1.000 | 40% | +0.012 | 1.000 | +0.005 | 1.000 |
| Graph cut (pairs) | -0.035 | 1.000 | -0.002 | 1.000 | 49% | -0.048 | 0.909 | -0.002 | 1.000 |
| Deep-ensemble BALD x P(decisive) | -0.036 | 1.000 | +0.001 | 1.000 | 34% | +0.012 | 1.000 | +0.003 | 1.000 |
| Cluster-Margin | -0.039 | 1.000 | -0.011 | 1.000 | 43% | +0.014 | 1.000 | -0.003 | 1.000 |
| FASS (pairs) | -0.043 | 0.929 | -0.006 | 1.000 | 43% | +0.016 | 1.000 | +0.000 | 1.000 |
| Uncertainty | -0.047 | 0.369 | -0.013 | 0.326 | 31% | -0.012 | 1.000 | -0.007 | 0.493 |
| Uncertainty + diversity | -0.048 | 1.000 | -0.009 | 1.000 | 43% | -0.002 | 1.000 | -0.001 | 1.000 |
| Uncertainty, all heads | -0.049 | 0.102 | -0.026 | <0.001 | 29% | -0.012 | 1.000 | -0.013 | 0.175 |
| Uncertainty, original code, all heads | -0.049 | 0.102 | -0.026 | <0.001 | 29% | -0.012 | 1.000 | -0.013 | 0.175 |
| Graph facility location (uncertainty-weighted) | -0.063 | 0.069 | -0.011 | 0.326 | 29% | +0.020 | 1.000 | +0.004 | 1.000 |
| Image-coverage uncertainty | -0.069 | 0.243 | -0.015 | 0.529 | 26% | -0.008 | 1.000 | -0.010 | 0.091 |
| DPP (quality x diversity) | -0.104 | 0.003 | -0.013 | 0.115 | 14% | -0.016 | 1.000 | -0.007 | 1.000 |

### 5.4 Split B (classifier2-style), sequential
Friedman: log-loss chi2=57.7, p=0.0035; AUC chi2=102.9, p=0.0000 (33 variants, 35 seeds)

| Strategy | log-loss gain | Holm p | AUC gain | Holm p | seeds better (log-loss) | OLD group-unit: log-loss gain | Holm p | AUC gain | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cluster-quota uncertainty, original code, all heads | +0.074 | <0.001 | +0.005 | 1.000 | 80% | +0.001 | 1.000 | -0.006 | 1.000 |
| Gap + posterior std (DeltaUCB-style) | +0.064 | 0.511 | -0.002 | 1.000 | 54% | +0.041 | 0.922 | +0.005 | 1.000 |
| Core-set | +0.063 | 0.274 | +0.006 | 1.000 | 66% | +0.053 | 0.009 | +0.006 | 1.000 |
| Core-set, relation-aware pairs | +0.063 | 0.279 | +0.007 | 1.000 | 74% | +0.044 | 0.424 | +0.006 | 1.000 |
| BALD x P(decisive) | +0.061 | 0.255 | +0.012 | 0.227 | 69% | +0.043 | 0.877 | +0.011 | 0.114 |
| Largest predicted gap | +0.051 | 1.000 | -0.004 | 1.000 | 60% | +0.038 | 1.000 | +0.007 | 1.000 |
| TypiClust (pairs) | +0.044 | 1.000 | +0.006 | 1.000 | 60% | +0.015 | 1.000 | +0.009 | 0.689 |
| Cluster-quota uncertainty | +0.039 | 1.000 | +0.009 | 0.575 | 71% | -0.003 | 1.000 | -0.000 | 1.000 |
| MC-dropout mutual info | +0.034 | 1.000 | +0.004 | 1.000 | 69% | -0.018 | 1.000 | -0.004 | 1.000 |
| Cluster-Margin, original code, all heads | +0.033 | 1.000 | -0.005 | 1.000 | 69% | +0.037 | 0.498 | +0.001 | 1.000 |
| MC-dropout variance | +0.024 | 1.000 | +0.001 | 1.000 | 60% | -0.014 | 1.000 | -0.002 | 1.000 |
| Uncertainty | +0.024 | 1.000 | +0.003 | 1.000 | 57% | -0.021 | 1.000 | -0.006 | 1.000 |
| Laplace BALD | +0.023 | 1.000 | +0.006 | 1.000 | 54% | +0.037 | 1.000 | +0.006 | 1.000 |
| Uncertainty + diversity, original code, all heads | +0.020 | 1.000 | -0.010 | 1.000 | 66% | +0.010 | 1.000 | -0.001 | 1.000 |
| MaxHerding (pairs) | +0.019 | 1.000 | -0.001 | 1.000 | 63% | +0.015 | 1.000 | +0.006 | 1.000 |
| BADGE (pairs) | +0.016 | 1.000 | +0.011 | 0.256 | 51% | +0.019 | 0.870 | +0.007 | 0.274 |
| Deep-ensemble BALD x P(decisive) | +0.013 | 1.000 | +0.014 | 0.023 | 60% | +0.025 | 1.000 | +0.007 | 0.565 |
| Uncertainty, all heads | +0.013 | 1.000 | -0.014 | 0.303 | 46% | -0.026 | 0.922 | -0.011 | 0.102 |
| Uncertainty, original code, all heads | +0.013 | 1.000 | -0.014 | 0.303 | 46% | -0.026 | 0.922 | -0.011 | 0.102 |
| Cluster-Margin | +0.012 | 1.000 | +0.001 | 1.000 | 49% | +0.022 | 1.000 | +0.000 | 1.000 |
| Graph facility location (uncertainty-weighted) | +0.009 | 1.000 | -0.000 | 1.000 | 49% | +0.032 | 0.969 | +0.006 | 1.000 |
| DropQuery (pairs) | +0.009 | 1.000 | -0.000 | 1.000 | 60% | -0.018 | 1.000 | -0.008 | 0.359 |
| Graph cut (pairs) | +0.009 | 1.000 | +0.010 | 0.189 | 63% | -0.027 | 1.000 | +0.004 | 1.000 |
| Uncertainty + diversity | +0.003 | 1.000 | +0.002 | 1.000 | 57% | -0.012 | 1.000 | -0.002 | 1.000 |
| Fisher D-optimal (Active Reward Modeling) | +0.003 | 1.000 | +0.011 | 0.132 | 49% | +0.019 | 1.000 | +0.006 | 1.000 |
| MC-dropout variance, original code, all heads | +0.002 | 1.000 | -0.007 | 1.000 | 51% | -0.029 | 1.000 | -0.009 | 0.179 |
| Deep-ensemble BALD (8 heads) | -0.000 | 1.000 | +0.004 | 1.000 | 49% | +0.031 | 1.000 | +0.009 | 0.147 |
| Image-coverage uncertainty | -0.003 | 1.000 | -0.001 | 1.000 | 49% | -0.011 | 1.000 | -0.007 | 0.277 |
| ProbCover (pairs) | -0.006 | 1.000 | +0.000 | 1.000 | 46% | +0.032 | 1.000 | +0.008 | 0.380 |
| MC-dropout mutual info, original code, all heads | -0.017 | 1.000 | -0.012 | 1.000 | 46% | -0.006 | 1.000 | -0.004 | 1.000 |
| FASS (pairs) | -0.022 | 1.000 | -0.004 | 1.000 | 40% | -0.007 | 1.000 | -0.005 | 1.000 |
| DPP (quality x diversity) | -0.060 | 1.000 | -0.004 | 1.000 | 46% | -0.024 | 1.000 | -0.005 | 1.000 |

### 5.5 What is significant (Holm < 0.05 within a table)

| Table | metric | variants with Holm p < 0.05 (mean gain, share of seeds better) | nominal p < 0.05 (of 32) |
|---|---|---|---:|
| Split A, single | log-loss | none | 0 |
| Split A, single | cal. log-loss | none | 3 |
| Split A, single | AUC | none | 5 |
| Split A, single | accuracy (secondary) | Deep-ensemble BALD x P(decisive) +0.015 (p=0.002, 74%); Deep-ensemble BALD (8 heads) +0.012 (p=0.039, 71%) | 10 |
| Split A, sequential | log-loss | none | 4 |
| Split A, sequential | cal. log-loss | none | 5 |
| Split A, sequential | AUC | none | 8 |
| Split A, sequential | accuracy (secondary) | Fisher D-optimal (Active Reward Modeling) +0.024 (p=0.000, 80%); Deep-ensemble BALD x P(decisive) +0.020 (p=0.002, 80%); Deep-ensemble BALD (8 heads) +0.019 (p=0.010, 77%); FASS (pairs) +0.016 (p=0.021, 71%); Image-coverage uncertainty +0.013 (p=0.040, 71%) | 12 |
| Split B, single | log-loss | DPP (quality x diversity) -0.104 (p=0.003, 14%) | 8 |
| Split B, single | AUC | MC-dropout mutual info, original code, all heads -0.014 (p=0.040, 29%); Uncertainty, all heads -0.026 (p=0.000, 17%); Uncertainty, original code, all heads -0.026 (p=0.000, 17%) | 11 |
| Split B, single | accuracy (secondary) | Gap + posterior std (DeltaUCB-style) -0.019 (p=0.019, 23%); Largest predicted gap -0.024 (p=0.001, 26%) | 4 |
| Split B, sequential | log-loss | Cluster-quota uncertainty, original code, all heads +0.074 (p=0.000, 80%) | 8 |
| Split B, sequential | AUC | Deep-ensemble BALD x P(decisive) +0.014 (p=0.023, 71%) | 9 |
| Split B, sequential | accuracy (secondary) | Deep-ensemble BALD x P(decisive) +0.022 (p=0.001, 83%); Fisher D-optimal (Active Reward Modeling) +0.020 (p=0.006, 83%); Deep-ensemble BALD (8 heads) +0.019 (p=0.031, 77%); Cluster-quota uncertainty +0.016 (p=0.024, 74%); BADGE (pairs) +0.015 (p=0.013, 77%) | 13 |

## 6. Per budget (best variants at each budget are the best of 32 and overstate; Holm only within one budget)

**Split A, single, log-loss**

| Budget (judgments) | top 3 by mean gain over Random (Holm p within the budget) | variants with Holm p < 0.05 |
|---:|---|---|
| 10 | Core-set, relation-aware pairs +0.074 (0.51); Uncertainty + diversity +0.061 (0.44); Cluster-Margin, original code, all heads +0.048 (1.00) | none |
| 20 | Core-set, relation-aware pairs +0.102 (0.15); Cluster-Margin +0.057 (1.00); Fisher D-optimal (Active Reward Modeling) +0.046 (1.00) | none |
| 40 | MaxHerding (pairs) +0.045 (1.00); MC-dropout variance +0.023 (1.00); Cluster-quota uncertainty, original code, all heads +0.022 (1.00) | none |
| 60 | MaxHerding (pairs) +0.045 (0.61); Uncertainty + diversity, original code, all heads +0.043 (1.00); MC-dropout mutual info +0.035 (1.00) | none |

**Split A, single, AUC**

| Budget (judgments) | top 3 by mean gain over Random (Holm p within the budget) | variants with Holm p < 0.05 |
|---:|---|---|
| 10 | Deep-ensemble BALD x P(decisive) +0.015 (0.44); FASS (pairs) +0.011 (1.00); Fisher D-optimal (Active Reward Modeling) +0.010 (0.44) | none |
| 20 | Fisher D-optimal (Active Reward Modeling) +0.012 (1.00); Deep-ensemble BALD (8 heads) +0.011 (1.00); BALD x P(decisive) +0.008 (1.00) | Cluster-Margin, original code, all heads -0.022 |
| 40 | BADGE (pairs) +0.006 (1.00); Deep-ensemble BALD x P(decisive) +0.006 (1.00); MaxHerding (pairs) +0.006 (1.00) | none |
| 60 | Deep-ensemble BALD x P(decisive) +0.013 (0.12); BALD x P(decisive) +0.012 (0.39); Deep-ensemble BALD (8 heads) +0.011 (0.26) | none |

**Split A, sequential, log-loss**

| Budget (judgments) | top 3 by mean gain over Random (Holm p within the budget) | variants with Holm p < 0.05 |
|---:|---|---|
| 10 | Core-set, relation-aware pairs +0.066 (1.00); Uncertainty + diversity +0.054 (1.00); Cluster-Margin, original code, all heads +0.040 (1.00) | none |
| 20 | Cluster-Margin, original code, all heads +0.061 (1.00); ProbCover (pairs) +0.041 (1.00); Uncertainty + diversity +0.038 (1.00) | none |
| 40 | Cluster-quota uncertainty +0.058 (0.12); Uncertainty + diversity +0.056 (0.30); Core-set, relation-aware pairs +0.050 (1.00) | none |
| 60 | Core-set +0.078 (0.03); Cluster-Margin +0.069 (0.07); Core-set, relation-aware pairs +0.061 (0.20) | Core-set +0.078 |

**Split A, sequential, AUC**

| Budget (judgments) | top 3 by mean gain over Random (Holm p within the budget) | variants with Holm p < 0.05 |
|---:|---|---|
| 10 | Deep-ensemble BALD x P(decisive) +0.013 (0.70); FASS (pairs) +0.009 (1.00); Fisher D-optimal (Active Reward Modeling) +0.008 (1.00) | none |
| 20 | Fisher D-optimal (Active Reward Modeling) +0.017 (0.06); ProbCover (pairs) +0.013 (0.35); Deep-ensemble BALD x P(decisive) +0.013 (0.91) | none |
| 40 | Uncertainty + diversity +0.014 (0.25); ProbCover (pairs) +0.012 (0.73); Cluster-quota uncertainty +0.012 (0.37) | none |
| 60 | Cluster-Margin +0.015 (0.10); Core-set +0.014 (0.19); Fisher D-optimal (Active Reward Modeling) +0.009 (1.00) | none |

**Split B, single, log-loss**

| Budget (judgments) | top 3 by mean gain over Random (Holm p within the budget) | variants with Holm p < 0.05 |
|---:|---|---|
| 10 | Cluster-quota uncertainty, original code, all heads +0.028 (1.00); Gap + posterior std (DeltaUCB-style) +0.025 (1.00); Largest predicted gap +0.020 (1.00) | none |
| 20 | Core-set +0.049 (0.66); TypiClust (pairs) +0.016 (1.00); Cluster-Margin, original code, all heads +0.007 (1.00) | DPP (quality x diversity) -0.130 |
| 40 | Core-set +0.049 (1.00); Cluster-Margin, original code, all heads +0.039 (1.00); BALD x P(decisive) +0.031 (1.00) | none |
| 60 | BALD x P(decisive) +0.049 (1.00); ProbCover (pairs) +0.023 (1.00); Gap + posterior std (DeltaUCB-style) +0.013 (1.00) | Uncertainty -0.060 |

**Split B, single, AUC**

| Budget (judgments) | top 3 by mean gain over Random (Holm p within the budget) | variants with Holm p < 0.05 |
|---:|---|---|
| 10 | Fisher D-optimal (Active Reward Modeling) +0.013 (0.88); Deep-ensemble BALD x P(decisive) +0.008 (1.00); Graph cut (pairs) +0.006 (1.00) | none |
| 20 | TypiClust (pairs) +0.009 (1.00); BADGE (pairs) +0.007 (1.00); Fisher D-optimal (Active Reward Modeling) +0.005 (1.00) | none |
| 40 | BALD x P(decisive) +0.008 (1.00); Laplace BALD +0.008 (1.00); Cluster-quota uncertainty +0.006 (1.00) | Uncertainty, all heads -0.034, Uncertainty, original code, all heads -0.034 |
| 60 | BALD x P(decisive) +0.014 (0.23); Deep-ensemble BALD (8 heads) +0.003 (1.00); ProbCover (pairs) +0.003 (1.00) | Uncertainty -0.016 |

**Split B, sequential, log-loss**

| Budget (judgments) | top 3 by mean gain over Random (Holm p within the budget) | variants with Holm p < 0.05 |
|---:|---|---|
| 10 | Cluster-quota uncertainty, original code, all heads +0.059 (0.28); Gap + posterior std (DeltaUCB-style) +0.056 (0.82); Largest predicted gap +0.051 (1.00) | none |
| 20 | Cluster-quota uncertainty, original code, all heads +0.076 (0.28); Core-set +0.075 (1.00); Cluster-quota uncertainty +0.065 (1.00) | none |
| 40 | Core-set, relation-aware pairs +0.114 (0.02); Cluster-quota uncertainty, original code, all heads +0.106 (0.00); BALD x P(decisive) +0.083 (0.57) | Core-set, relation-aware pairs +0.114, Cluster-quota uncertainty, original code, all heads +0.106 |
| 60 | BALD x P(decisive) +0.098 (0.04); MC-dropout mutual info +0.068 (0.05); Core-set, relation-aware pairs +0.066 (1.00) | BALD x P(decisive) +0.098 |

**Split B, sequential, AUC**

| Budget (judgments) | top 3 by mean gain over Random (Holm p within the budget) | variants with Holm p < 0.05 |
|---:|---|---|
| 10 | Fisher D-optimal (Active Reward Modeling) +0.019 (0.12); Deep-ensemble BALD x P(decisive) +0.014 (0.18); Graph cut (pairs) +0.012 (0.27) | none |
| 20 | Deep-ensemble BALD x P(decisive) +0.015 (0.53); BADGE (pairs) +0.013 (0.57); Fisher D-optimal (Active Reward Modeling) +0.012 (1.00) | none |
| 40 | Deep-ensemble BALD x P(decisive) +0.019 (0.02); Core-set, relation-aware pairs +0.017 (0.42); BALD x P(decisive) +0.016 (0.92) | Deep-ensemble BALD x P(decisive) +0.019 |
| 60 | BALD x P(decisive) +0.020 (0.01); Core-set +0.011 (1.00); TypiClust (pairs) +0.011 (1.00) | BALD x P(decisive) +0.020 |


## 7. What the strategies reveal (budget 60, single-shot, means over 35 seeds)

| Strategy | Split A: distinct pairs touched | decisive share | tie share | Split B: distinct pairs touched | decisive share | tie share |
|---|---:|---:|---:|---:|---:|---:|
| Largest predicted gap | 37.3 | 69% | 5% | 41.4 | 69% | 5% |
| Gap + posterior std (DeltaUCB-style) | 34.2 | 67% | 4% | 37.6 | 68% | 4% |
| BALD x P(decisive) | 30.2 | 59% | 6% | 31.6 | 59% | 5% |
| Graph cut (pairs) | 60.0 | 54% | 18% | 60.0 | 57% | 17% |
| TypiClust (pairs) | 60.0 | 52% | 20% | 60.0 | 54% | 20% |
| ProbCover (pairs) | 60.0 | 51% | 18% | 60.0 | 54% | 17% |
| Core-set | 60.0 | 53% | 20% | 60.0 | 54% | 21% |
| Core-set, relation-aware pairs | 60.0 | 51% | 20% | 60.0 | 52% | 19% |
| Random pair, random type | 42.8 | 53% | 18% | 47.7 | 52% | 19% |
| MaxHerding (pairs) | 60.0 | 52% | 20% | 60.0 | 52% | 18% |
| Laplace BALD | 32.0 | 53% | 8% | 35.4 | 51% | 7% |
| Random | 42.3 | 50% | 14% | 47.6 | 50% | 14% |
| FASS (pairs) | 60.0 | 47% | 20% | 60.0 | 48% | 19% |
| Deep-ensemble BALD x P(decisive) | 39.5 | 47% | 7% | 43.4 | 47% | 7% |
| Deep-ensemble BALD (8 heads) | 40.1 | 45% | 8% | 45.0 | 45% | 7% |
| MC-dropout mutual info | 43.3 | 44% | 17% | 47.8 | 44% | 14% |
| MC-dropout variance | 42.2 | 43% | 18% | 47.6 | 44% | 15% |
| Fisher D-optimal (Active Reward Modeling) | 37.2 | 43% | 14% | 40.2 | 43% | 12% |
| MC-dropout mutual info, original code, all heads | 60.0 | 48% | 23% | 60.0 | 43% | 28% |
| BADGE (pairs) | 41.7 | 43% | 15% | 47.3 | 42% | 15% |
| Graph facility location (uncertainty-weighted) | 60.0 | 43% | 21% | 60.0 | 42% | 22% |
| Cluster-quota uncertainty, original code, all heads | 60.0 | 48% | 22% | 60.0 | 41% | 31% |
| MC-dropout variance, original code, all heads | 60.0 | 48% | 23% | 60.0 | 40% | 30% |
| DPP (quality x diversity) | 60.0 | 42% | 22% | 60.0 | 40% | 22% |
| Uncertainty + diversity, original code, all heads | 60.0 | 50% | 23% | 60.0 | 40% | 31% |
| Cluster-Margin, original code, all heads | 60.0 | 49% | 23% | 60.0 | 40% | 30% |
| DropQuery (pairs) | 44.9 | 39% | 25% | 49.4 | 37% | 28% |
| Uncertainty, original code, all heads | 60.0 | 47% | 23% | 60.0 | 37% | 33% |
| Uncertainty, all heads | 60.0 | 47% | 23% | 60.0 | 37% | 33% |
| Image-coverage uncertainty | 53.7 | 37% | 25% | 59.9 | 36% | 27% |
| Uncertainty + diversity | 49.0 | 34% | 25% | 58.2 | 35% | 27% |
| Cluster-Margin | 39.9 | 36% | 23% | 44.0 | 33% | 28% |
| Cluster-quota uncertainty | 43.7 | 35% | 22% | 47.4 | 31% | 27% |
| Uncertainty | 39.2 | 31% | 26% | 43.3 | 28% | 31% |

Reading: uncertainty-based rules (Uncertainty, Cluster-Margin, Cluster-quota, all-head variants) mostly reveal tie / not_apply judgments (decisive share 28-41% against about 50% for Random), which carry little preference information; Largest predicted gap, Gap + posterior std and BALD x P(decisive) reveal mostly decisive ones (59-69%). Pair-level rules reveal exactly 60 distinct pairs; Laplace BALD and BALD x P(decisive) concentrate on 30-35 pairs (several types of the same pair). Random touches about 47 (Split B) distinct pairs with 60 judgments.

## 8. Comparison with the group-unit results

Absolute levels are not comparable because the budgets differ in unit and the old strategies received about 3.1 times as many rows (Random, mean over 35 seeds):

**Split A, single-shot**

| Budget | NEW: judgments | Random log-loss | Random AUC | OLD: groups | Random log-loss | Random AUC |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 10 | 0.666 | 0.876 | 10 | 0.600 | 0.883 |
| 20 | 20 | 0.634 | 0.884 | 20 | 0.494 | 0.902 |
| 40 | 40 | 0.552 | 0.895 | 40 | 0.406 | 0.921 |
| 60 | 60 | 0.496 | 0.902 | 60 | 0.356 | 0.934 |

**Split A, sequential**

| Budget | NEW: judgments | Random log-loss | Random AUC | OLD: groups | Random log-loss | Random AUC |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 10 | 0.659 | 0.878 | 10 | 0.578 | 0.891 |
| 20 | 20 | 0.615 | 0.882 | 20 | 0.474 | 0.907 |
| 40 | 40 | 0.556 | 0.893 | 40 | 0.398 | 0.922 |
| 60 | 60 | 0.508 | 0.900 | 60 | 0.354 | 0.933 |

**Split B, single-shot**

| Budget | NEW: judgments | Random log-loss | Random AUC | OLD: groups | Random log-loss | Random AUC |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 10 | 0.596 | 0.881 | 10 | 0.574 | 0.886 |
| 20 | 20 | 0.547 | 0.889 | 20 | 0.484 | 0.905 |
| 40 | 40 | 0.527 | 0.894 | 40 | 0.392 | 0.924 |
| 60 | 60 | 0.470 | 0.905 | 60 | 0.359 | 0.930 |

**Split B, sequential**

| Budget | NEW: judgments | Random log-loss | Random AUC | OLD: groups | Random log-loss | Random AUC |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 10 | 0.627 | 0.875 | 10 | 0.560 | 0.888 |
| 20 | 20 | 0.607 | 0.880 | 20 | 0.481 | 0.902 |
| 40 | 40 | 0.560 | 0.889 | 40 | 0.407 | 0.916 |
| 60 | 60 | 0.514 | 0.898 | 60 | 0.366 | 0.928 |


Strategies that were Holm-significant in the old tables, and what they do now (new gain, new Holm p, share of seeds better; "old" in parentheses):

| Split / condition | Strategy | Old log-loss gain (Holm p) | New log-loss gain (Holm p, share of seeds) | Old AUC gain (Holm p) | New AUC gain (Holm p) |
|---|---|---|---|---|---|
| A single | Laplace BALD | +0.038 (0.022) | -0.001 (1.000, 51%) | +0.010 (0.031) | +0.004 (1.000) |
| A single | BALD x P(decisive) | +0.038 (0.026) | +0.004 (1.000, 49%) | +0.010 (0.027) | +0.007 (1.000) |
| A single | Cluster-quota uncertainty | +0.038 (0.033) | +0.014 (1.000, 60%) | +0.005 (1.000) | +0.001 (1.000) |
| A single | DPP | +0.028 (0.047) | -0.019 (1.000, 40%) | +0.005 (1.000) | +0.001 (1.000) |
| A single | Graph cut (worse than Random) | -0.058 (0.047) | -0.022 (1.000, 46%) | -0.000 | -0.001 |
| A sequential | Graph cut (worse than Random) | -0.049 (0.015) | -0.057 (0.531, 31%) | -0.001 | -0.000 |
| B single | Core-set | +0.052 (0.012) | +0.029 (1.000, 60%) | +0.003 | -0.003 |
| B sequential | Core-set | +0.053 (0.009) | +0.063 (0.274, 66%) | +0.006 | +0.006 |

Reading: the signs of core-set (Split B, both conditions) and of graph cut (worse, Split A) are retained, the magnitudes shrink or become noisier, and the Laplace-BALD / BALD x P(decisive) advantage of Split A single-batch disappears. New Holm-significant effects that were absent before: see section 5.5 (all-head cluster-quota log-loss and ensemble BALD x P(decisive) AUC in Split B sequential; accuracy gains of ensemble BALD / Fisher D-optimal; significantly worse DPP, all-head uncertainty and the gap rules in Split B single).
The new and old tables were produced with the same aggregation code (`aggregate_pair_endpoint.py`); the old columns are read from `results/pair_endpoint_study/*_vs_random.csv`.
The extra baseline `random_pair_type` (uniform pair, then uniform type of it) is statistically indistinguishable from Random (uniform judgment) on all metrics in all four cells (p-values between 0.075 and 1.0).

## 9. Sensitivity to the initial labelled set

Run on 20 seeds (42, 79, 123, 202, 303, 400-414), both splits, both conditions, with 10 random judgments as the initial set (drawn from the judgments of the initial and pool groups; the rest is the pool; splits, test and validation sets unchanged). Full tables: `results/judgment_unit_study/{A,B}_random/analysis/report.txt`.

| Split / condition | metric | Spearman correlation of the 32 variants' mean gains, initial groups vs random initial (same 20 seeds) | Holm < 0.05 (random initial) |
|---|---|---:|---|
| A single | log-loss / AUC | 0.24 / 0.69 | none |
| A sequential | log-loss / AUC | 0.63 / 0.74 | none |
| B single | log-loss / AUC | -0.01 / 0.47 | none |
| B sequential | log-loss / AUC | 0.10 / 0.48 | none |

No variant is significant on log-loss, AUC or accuracy under the random initial set (20 seeds, so lower power than the 35-seed main run). The ranking of variants by log-loss gain is only weakly related between the two initial sets (correlations -0.01 to 0.63) and the AUC ranking moderately (0.47-0.74), so the per-rule details of the main tables depend on the initial set as well as on the seeds. The accuracy gains of deep-ensemble BALD / Fisher D-optimal stay positive in all four cells (+0.006 to +0.022) but are not significant. Random levels are similar to the main run for AUC (about 0.86-0.90); Random's log-loss in Split B single-batch is higher with the random start (0.66 / 0.64 / 0.61 / 0.51 at budgets 10 / 20 / 40 / 60 against 0.59 / 0.54 / 0.52 / 0.47 with the initial groups, same 20 seeds).

## 10. Checks

- Cell counts were recounted from the files, not hard-coded (`judgment_unit_aggregate.py::check_counts`, asserted): per run and condition 35 seeds x 42 strategy names x 4 budgets = 5,880 cells (A_groups, B_groups; 20 seeds = 3,360 for the sensitivity runs); 42 names = Random (5 draws) + 32 other variants + `random_pair_type` (5 draws); 34 strategy families = 32 non-random variants + Random + `random_pair_type`. Main run: 4 x 5,880 = 23,520 cells; sensitivity: 4 x 3,360 = 13,440 cells (plus one initial-only cell per seed and run).
  (The earlier study had 37 names per seed: 8 + 17 + 6 + 2 variants with 4 extra Random draws; here 32 non-random variants + 5 Random + 5 `random_pair_type`.)
- Every cell asserts that exactly `budget` distinct judgments, all from the pool, were revealed (`n_revealed == budget`); every seed asserts that no validation/test image appears in the initial set or the pool, and that the initial set and the pool are disjoint.
- Hand check (`judgment_unit_verify.py`): for (Split A single seed 42 Core-set budget 20), (Split B sequential seed 400 Laplace BALD budget 40) and (Split A sequential seed 303 Random draw 2 budget 60) the pool size (225 / 350 / 233 judgments, 3.1-3.2 per group), the number revealed (20 / 40 / 60), the training rows (initial + revealed: 49 / 62 / 88), the absence of any validation/test image from the labelled set and the pool (0 shared images), the absence of unselected judgments of the selected pairs in the training rows (0), and the metrics after retraining (log-loss, AUC, accuracy identical to six decimals to the stored cell) were confirmed. Laplace BALD touched 25 distinct pairs with its 40 judgments in that cell.
- Tests: `code_behavior_tests/test_judgment_unit.py` (35 tests: every selector returns the budget of distinct judgments reproducibly, pair-level rules reveal one judgment per pair, the training rows contain exactly the revealed judgments and not their unrevealed siblings, the group path of `train_model` is unchanged, integration on real data for both splits). The full suite passes (163 tests: the 128 earlier ones, including the 56 frozen-encoder/strategy tests, plus these 35).

## 11. Caveats

- Test sets are small (Split A about 63, Split B about 54 decisive judgments per seed); the per-seed gains have standard deviations of 0.08-0.18 in log-loss.
- Holm is applied within one table (one split, one condition, one metric) over 32 variants; it is not applied across metrics, conditions, splits or budgets. Accuracy is a secondary metric and is shown here only because it is significant for some rules; with four metrics, two splits and two conditions, a few nominal hits are expected by chance. Per-budget "best" rows are the best of 32 and overstate.
- Hyper-parameters (lr 0.01, 100 steps) were calibrated once on random batches of the group-unit design and reused; with 10-60 judgments the optimum might differ. Nothing was tuned in this study; strategy parameters are those of the earlier study.
- Several adaptations are my own decisions (family P/L: one judgment per chosen pair with a random or most-uncertain type; own-head quality in place of the average over heads; coverage computed in pair space, so a sibling judgment of a labelled pair counts as covered). Other reasonable adaptations exist and could change individual rows. In particular the pair-level rules cannot concentrate budget on one pair, the row-level rules can (Laplace BALD / BALD x P(decisive) touch only about 30-35 distinct pairs with 60 judgments).
- The type of a (pair, type) candidate is observable to the strategy (it is the queried type); the outcome is not. No claim is made about cold-start type coverage.
- The old and new numbers are for different budgets in different units; old strategies received about 3.1 times as many labelled rows.
- One seed set, one encoder (frozen SimCLR), one head schedule.

## 12. Conclusion in plain language

Counting the budget in judgments, and letting each strategy choose which type of which pair to ask about, does not change the main message of the earlier study: there is no strategy that is clearly better than random selection, and the cases where a rule looks good in one split or one condition are not reproduced in the others. Under the group-unit design a few rules were Holm-significant on log-loss and AUC; in the judgment-unit design almost none are (one log-loss result for an all-head cluster-quota variant and one AUC result for deep-ensemble BALD x P(decisive), both only in Split B sequential), and the earlier winners (Laplace BALD, BALD x P(decisive) in Split A; core-set in Split B) are no longer significant. The earlier design therefore did not hide a big advantage of some strategy that appears when the type is chosen by the model: the data do not show one on log-loss or AUC. They do show a small, consistent advantage in decisive accuracy for deep-ensemble BALD and Fisher D-optimal design (1-2.5 points) that was absent before; because accuracy is a secondary metric, the effect is small, and it is not significant under the random-initial-set sensitivity run with fewer seeds, it should be re-tested rather than adopted. Rules that rely on model uncertainty alone ask mostly about judgments that experts answer with tie or not_apply, and these perform no better than random.
In practice: with this encoder and about 30 initial judgments, a few dozen additional judgments are best spent with a cheap rule (none of the 32 rules is shown to be reliably better than uniformly random judgments); if a rule is wanted, deep-ensemble BALD x P(decisive) or Fisher D-optimal design are the ones worth testing on a second encoder and on new data. All statements are for this data set, a frozen SimCLR encoder, small test sets (about 54-63 decisive judgments) and 35 seeds.
