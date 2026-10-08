# Pre-registration: replication of the judgment-unit study on fresh seeds

Written and committed BEFORE any run on the new seeds. No result for seeds 500-534 exists at the time of writing (checked: no file with a `seed5xx` name under `results/`).
Pre-run commit hash: see the line "Pre-run commit" at the very end of this file (it names the commit that contains this text; it is appended in a second commit that changes nothing else).

Base code: branch `claude/frozen-encoder-strategies`, commit `6714938` (the judgment-unit study as reported in `JUDGMENT_UNIT_RESULTS.md` and report section 5.14). No strategy, training, split or evaluation code is modified for this replication.

## 1. Purpose and limitation

The judgment-unit study (35 seeds: 42, 79, 123, 202, 303, 400-429) found few Holm-significant effects and several of them look fragile. This replication re-runs the variants named below on 35 NEW seeds (500-534) and tests, with a fixed rule, whether the effects that appeared in the main run appear again.

Limitation (stated up front): a seed determines the random split of the 168 pair groups into initial / validation / test / pool, the initial set, the random draws and the model initialisations. The 168 pair groups (521 judgments, 284 images) are the SAME data as in the main run. The replication therefore checks the sensitivity of the conclusions to the split and initial draw (and to the training randomness); it does NOT provide new data, and the test sets of the new seeds overlap with those of the old seeds (the same groups appear in many different splits). The new seeds are not independent of the old ones in the data sense; they are independent in the random-draw sense only.

## 2. Design (identical to the main judgment-unit run)

- Seeds 500-534 (35 seeds): `PAIR_STUDY_SEEDS=500-534`.
- Splits: Split A (`JU_SPLIT=A`: 10 initial / 20 validation / 40 test groups) and Split B (`JU_SPLIT=B`: classifier2-style 20% test, no validation).
- Initial labelled set: the judgments of the 10 initial groups (`JU_INITIAL=groups`), as in the main run.
- Conditions: single-shot and sequential (rounds of 10) (`JU_MODE=both`).
- Budgets: 10, 20, 40, 60 judgments.
- Schedule: learning rate 0.01, 100 steps (read from `results/pair_endpoint_study/schedule.json`, unchanged), frozen SimCLR features, same head and endpoint code; `torch.set_num_threads(2)` (`JU_THREADS=2`).
- Runner: `judgment_unit_study.py`, unchanged, restricted to the variants below with `JU_ONLY` (comma list of strategy keys); the runner already accepts a seed list/range and a variant list through environment variables, so no code change is needed. Results go to `results/judgment_unit_study/replication/{A,B}_groups/`.
- Random = uniform over the remaining judgments, mean of its 5 draws within a seed (`random`, `random_r1`..`random_r4`), exactly as before. `random_pair_type` (5 draws) is the second baseline.

## 3. Variants (display name -> exact strategy key in `judgment_unit_strategies.py` / `judgment_unit_study.py`)

| # | Variant | Key | Family |
|---|---|---|---|
| 1 | Random (baseline) | `random` (+ `random_r1`..`random_r4`) | - |
| 2 | Random pair, random type (baseline) | `random_pair_type` (+ `_r1`..`_r4`) | - |
| 3 | Cluster-quota uncertainty, original code, all heads | `cluster_quota_uncertainty_lf` | L |
| 4 | Cluster-quota uncertainty (own type) | `cluster_quota_uncertainty` | R |
| 5 | Deep-ensemble BALD x P(decisive) | `ensemble_bald_decisive` | R |
| 6 | Deep-ensemble BALD (8 heads) | `ensemble_bald` | R |
| 7 | Fisher D-optimal (Active Reward Modeling) | `fisher_dopt` | R |
| 8 | BALD x P(decisive) | `bald_decisive` | R |
| 9 | Laplace BALD | `laplace_bald` | R |
| 10 | Core-set | `core_set` | P |
| 11 | Core-set, relation-aware pairs | `core_set_relation` | P |
| 12 | DPP (quality x diversity) | `dpp_pairs` | P |
| 13 | Uncertainty (own type) | `uncertainty` | R |
| 14 | Uncertainty, all heads | `uncertainty_all_heads` | L |
| 15 | Largest predicted gap | `delta_gap` | R |

13 non-random variants + 2 baselines = 23 strategy names per seed (Random 5 + random_pair_type 5 + 13). `uncertainty_lf` ("Uncertainty, original code, all heads") is numerically identical to `uncertainty_all_heads` in the main run (same rule) and is not run again; "Uncertainty all heads" in H5 means key `uncertainty_all_heads`.
Expected cells: 35 seeds x 23 names x 4 budgets = 3,220 per (split, condition); 4 x 3,220 = 12,880 cells in total (plus one `initial_only` file per seed and split).

## 4. Endpoints and statistic

Held-out decisive-judgment endpoint, as in the main run (computed by `pair_preference_endpoint.evaluate_full`, columns): log-loss `test_decisive_log_loss` (lower is better), AUC `test_decisive_auc`, decisive accuracy `test_decisive_accuracy` (higher is better).

Gain over Random (positive = better than Random; sign convention of the main run): for log-loss, `Random - strategy`; for AUC and accuracy, `strategy - Random`. Per seed: gain at each budget, then averaged over the four budgets 10/20/40/60. The per-seed value is the unit of analysis (35 values per variant and cell). Reported: mean gain (the estimate), share of seeds with gain > 0, one-sided p.

Test: Wilcoxon signed-rank test (`scipy.stats.wilcoxon`, default zero handling) on the 35 per-seed gains, ONE-SIDED in the direction of the hypothesis (alternative "greater" for "gain > 0", "less" for "worse than Random"). Holm correction over the confirmatory family defined in section 6 only (not over all 13 variants).

Cells: split (A, B) x condition (single-shot, sequential). Test and validation groups are drawn per seed, so the new seeds give new splits of the same 168 groups.

## 5. Pre-specified hypotheses (all one-sided)

- **H1.** `cluster_quota_uncertainty_lf` has log-loss gain > 0 in Split B, sequential. (Main run: +0.074, Holm p < 0.001, 80% of seeds.)
- **H2.** `ensemble_bald_decisive` has AUC gain > 0 in Split B, sequential. (Main: +0.014, Holm p = 0.023.)
- **H3.** Decisive-accuracy gain > 0 for each of `ensemble_bald_decisive`, `ensemble_bald`, `fisher_dopt`, in ALL FOUR split x condition cells (A single, A sequential, B single, B sequential). 12 tests; each variant's hypothesis is the conjunction over its four cells. (Main: +0.006 to +0.024; Holm-significant in some cells only.)
- **H4.** `core_set` has log-loss gain > 0 in Split B single-shot and in Split B sequential. 2 tests; conjunction. (Main: +0.029 and +0.063, not Holm-significant, 60-66% of seeds.)
- **H5.** In Split B single-shot, `dpp_pairs` is WORSE than Random on log-loss (gain < 0), and `uncertainty_all_heads` is WORSE than Random on AUC (gain < 0). 2 tests (alternative "less"). (Main: -0.104 and -0.026.)
- **H6 (null check).** `uncertainty` (plain, own type) has no advantage over Random: gain <= 0 in every cell, for log-loss and for AUC. Tested as 8 one-sided tests of "gain > 0" (4 cells x 2 metrics); H6 is supported if none of the 8 is rejected (Holm over these 8, alpha 0.05). The point estimates are reported; H6 is not a claim that the estimates are negative, but that no significant positive effect exists. (Main: log-loss gains were +0.012 / -0.007 / -0.047 / +0.024, so a positive estimate in a cell would not contradict H6 unless it is significant.)

The other variants (`cluster_quota_uncertainty`, `bald_decisive`, `laplace_bald`, `core_set_relation`, `delta_gap`, `random_pair_type`) are run as pre-specified comparison variants and baselines. They carry no confirmatory hypothesis; their results are reported in a descriptive table and are exploratory. `random_pair_type` vs Random is reported as a check on the baseline (two-sided Wilcoxon, all three metrics, four cells).

## 6. Decision rule

Confirmatory family F = {H1 (1 test), H2 (1), H3 (12), H4 (2), H5 (2)} = 18 one-sided Wilcoxon tests. Holm correction over these 18 at alpha = 0.05. H6 is a separate null-check family of 8 tests with its own Holm correction.

A test is "replicated" if its Holm-adjusted p < 0.05 AND the estimate has the hypothesised sign. A hypothesis is:
- **replicated** if all of its tests are replicated (H1, H2: the single test; H3: for each of the three variants, all four cells; H4: both cells; H5: both tests, with each of the two claims reported separately as well);
- **not replicated** otherwise.
For H3 and H5 (several claims) the report gives the verdict per variant / claim and the number of cells with the right sign and with Holm p < 0.05 (descriptive). The rule is binary on purpose; no "partially replicated" label is used for the verdict (it may be used as plain description).

Nothing is tuned: no strategy parameter, schedule, seed list, metric or threshold may be changed after seeing results. If a bug is found the run is stopped and reported; old behaviour stays reproducible.

## 7. Planned analyses

Confirmatory (per sections 4-6): the 18 + 8 tests above on seeds 500-534 only.

Exploratory (labelled as such in the results file; nothing in the verdict depends on it):
- E1. Pooled analysis over the 35 main seeds and the 35 new seeds (70 seeds): the same 18 + 8 tests, Holm as above. The main seeds are those on which the hypotheses were formed, so the pooled estimates are not an independent test.
- E2. Full table of all 13 variants x 4 cells x 3 metrics on the new seeds (two-sided Wilcoxon, Holm over the 13 variants within a table, as in the main report), and the main-run values next to them.
- E3. Consistency of the variant ranking (Spearman correlation of mean gains between main and new seeds, per cell and metric, over the 13 variants).
- Anything else added after seeing results is labelled exploratory as well.

## 8. Checks planned

Recount cells from files (expected 12,880 + 70 initial-only files); assert `n_revealed == budget` in every cell; recompute one cell by hand with `judgment_unit_verify.py` (retrain from the stored selection and compare metrics); run the full existing test suite (`pytest` in `code/`; all tests must pass); keep the committed result size modest.

---
Pre-run commit: `6b6070dabd57515f49c933ca9499c144f5322637` (contains the pre-registration text above; no run on seeds 500-534 had been started at that commit; this line is the only later change to the file).
