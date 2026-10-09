# Pre-registration: pool-wide variance-reduction acquisition (confirmatory run on seeds 700-734)

Written and committed BEFORE any cell of the confirmatory run exists (the commit hash of this file is recorded at the end). Code: `new_methods_strategies.py`, runner `new_methods_study.py` / `new_methods_run.sh`, analysis `new_methods_confirm.py`.

## Background (development, seeds 600-629; not evidence)
On 30 development seeds (both splits, both conditions, budgets 10/20/40/60 judgments) rules that reduce the pool-wide predictive variance of the own-type last layer (Laplace posterior, greedy, rank-one updates) were better than Random on AUC (+0.009 to +0.011) and decisive accuracy (+0.014 to +0.020) in all four cells and weakly on log-loss. Development choices (rule form, ridge, weighting) were explored on these seeds only. The seeds 700-734 have not been used for anything.

## Protocol (unchanged from the judgment-unit study)
Query unit: one (pair, type) judgment; budget in judgments (10, 20, 40, 60); single-shot (one batch from the model trained on the initial set) and sequential (rounds of 10, retrained); Split A (10 initial / 20 validation / 40 test groups) and Split B (classifier2-style 20% hold-out); initial set = judgments of the 10 initial groups; frozen SimCLR features; order-independent head; the **re-tuned per-budget head schedule** `results/pair_endpoint_study/schedule_judgment_unit.json`; Random = mean of 5 draws within a seed; endpoint = held-out decisive preference prediction (log-loss, AUC, accuracy; calibrated log-loss is secondary and Split A only).

## Pre-registered methods (exactly two; no others will be tested confirmatorily on these seeds)
- **M1 `vopt_u`**: greedy I-optimal design. For each active head k, with posterior covariance Sigma_k = (ridge I + sum_i w_i phi_i phi_i')^-1 (ridge 1, w = p(1-p), phi = hidden-layer difference, revealed judgments inform only their own head), the gain of candidate i is w_i/(1 + w_i phi_i' Sigma phi_i) * sum_j (phi_j' Sigma phi_i)^2 over the candidate pool judgments j of the same head (unit target weights); greedy with rank-one updates. No outcome of an unrevealed judgment is used.
- **M2 `vopt_u_inf1`**: M1 with the gain multiplied by the predicted probability that the judgment is decisive (regularised logistic regression on type, pair distance and cosine, fitted on the outcomes revealed so far; 0.5 if only one class has been seen).

## Primary hypotheses (one-sided)
For each M in {M1, M2} and each metric in {log-loss, AUC, decisive accuracy}: the mean over seeds of the per-seed gain over Random (gain averaged over the four budgets, then averaged over the four split x condition cells) is > 0. Test: one-sided Wilcoxon signed-rank over the 35 seeds. **Holm correction over these 6 tests.** A method "beats Random" if its AUC, accuracy and log-loss tests are all significant at Holm 0.05; "partly" if at least one is. The results are reported whatever they are.

## Secondary analyses (uncorrected, labelled secondary)
Per-cell gains and p-values; per-budget gains; calibrated log-loss (Split A); comparison of M1 and M2 with the strongest earlier rules (Fisher D-optimal, BALD x P(decisive), relation-aware core-set, plain uncertainty) run on the same seeds (paired Wilcoxon, one-sided in favour of M1/M2, uncorrected); fraction of decisive judgments chosen; robustness runs: (a) old head schedule (lr 0.01, 100 steps), (b) random initial set (10 random judgments), each on a subset of seeds as time permits.

## Limitations stated in advance
The seeds are new random splits of the same 168 pair groups (521 judgments), so this checks sensitivity to the split and initial draw, not new data; Wilcoxon p-values are optimistic because seeds resample the same groups; test sets are small (about 54-63 decisive judgments); the head schedule was calibrated on three seeds; strategy adaptations of pair-space rules are not part of this test.

Pre-run commit: (recorded below)
