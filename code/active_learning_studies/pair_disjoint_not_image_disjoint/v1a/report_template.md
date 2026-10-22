# V1a report — classifier2 loss and ideal-image evaluation on the frozen SimCLR encoder

Branch `claude/v1a-20261022` (from `claude/integration-20261013`, af8413325). All code is new under `v1a/`; no existing file or result was modified. Tables below are inserted by `make_report.py` from `results/*.csv` (nothing typed by hand).

## 1. What was run
* **Model / training.** Cached 512-d SimCLR features (encoder frozen) → repository `BTModel` head Linear(512,256)–ReLU–Dropout–Linear(256,5), **132,613 parameters** (checked in `test_v1a.py`). Full-batch AdamW (weight decay 1e-4, clip 1.0), the experiment seed's own initial head at every retrain (ensemble members: head seeds 1000+k, repository behaviour), 300 steps at lr 0.01 (frozen in `FROZEN.md`, committed before the main run).
* **Loss** `L = 0.60 L_pair + 0.25 L_ideal + 0.15 L_bad` in expectation form (`v1a_core.py: pair_loss, ideal_loss, bad_loss, v1a_loss`). |T| = **1087** in every seed (1124 trajectory images in the cache minus **37** whose SHA-256 equals that of an ideal image; the 154 ideal files contain 148 distinct contents).
* **Data/protocol (reused).** classifier2-style split B (20 % of the pair groups = 34 test groups, no validation groups), unit of acquisition = one judgment, 30 initial judgments (10 groups), pool 306–350 judgments, rounds of 10 judgments with retraining after every round, budgets 10/20/40/60, 35 seeds (42, 79, 123, 202, 303, 400–429). Ideal split per seed 88 reference / 28 validation / 28 test, 18 bad images (verified for seed 42 and by test for every fit; the test images are never in references, validation images or the bad set).
* **Rules (8):** Random, Uncertainty, Core-set, TypiClust, Laplace-BALD, BALD×P(decisive), Ensemble-BALD (8 heads), ProbCover. The last five were picked because they were the best rules in V2 on the *preference* endpoint (post-hoc selection of rules; this report does not repeat that selection and does not treat the rule set as unbiased).
* **Primary endpoint:** accuracy of the win-rate classifier on the 28 test ideal images (`win_rate_predict`; its accuracy equals `frozen_encoder_reward_head.evaluate_model` exactly, asserted in `test_v1a.py` for test and validation images). Secondary (reported only): macro-F1 and the held-out pair-group preference AUC / accuracy / log-loss of V2.
* **Runtime / environment:** Python 3.13.16, torch 2.14.1 (PyPI wheel, run on CPU only; micromamba was not needed), 4 cores, one thread per process. Tuning: 135 fits in about 5.5 min per seed on 3 cores. Main run: %%SEED_SECONDS%% s per seed on average (Ensemble-BALD ≈ %%ENS%% s of it), %%CPU_H%% CPU-hours in total on 4 shards. Unit tests: 7 passed (`python3 -m pytest v1a/test_v1a.py`).

## 2. Hyper-parameter tuning (validation ideal images only)
Seeds 42/79/123, label sets = initial 30 judgments + {0,10,20,40,60} random pool judgments, 15 fits per setting (`tune.py`; label draws are mine, the prompt did not specify them). Chosen: lr 0.01, 300 steps. Differences between neighbouring cells are about 0.005 and below the noise of 15 fits × 28 images; the choice is a weak preference.

%%TUNING%%

## 3. Main results (35 seeds)
%%TABLES%%

## 4. Reading
* **Primary endpoint: no rule is distinguishable from Random.** The largest mean difference is Uncertainty +0.007 (95 % CI −0.002 … +0.016; Wilcoxon p 0.075, Holm over 7 rules 0.53); all seven confidence intervals contain 0 and all Holm p ≥ 0.53. With 28 test images per seed (binomial SD ≈ 0.07 per seed) and 35 seeds the interval half-width is ≈ ±0.01, so effects below about 0.01 cannot be claimed or excluded.
* **Pair information barely moves this endpoint.** The no-selection references give 0.841 (10 initial groups only) and 0.853 (all candidate-pool groups, an upper bound of the pair information) — a total headroom of ≈ 0.012, similar to the interval half-width; the reference anchors dominate ideal-image accuracy (earlier finding of the repository: anchors alone ≈ 0.85). Any acquisition rule can gain at most about that much on this endpoint, so the endpoint cannot rank rules.
* **Secondary (reported only; the Holm correction is within each metric over 7 rules, not across the secondary family):** on preference AUC, Laplace-BALD +0.012 (Holm 0.049), TypiClust +0.011, BALD×P(decisive) +0.009 are positive with CIs above 0 but only Laplace-BALD passes Holm; Uncertainty is −0.012 (CI below 0, Holm 0.11). On preference accuracy BALD×P(decisive), Ensemble-BALD and TypiClust are about +0.010 to +0.013 (CIs above 0, Holm 0.07–0.25). These directions agree with V2's post-hoc ranking (which selected these rules), so they are not independent confirmation. Ideal-image accuracy and preference quality do not move together here (Uncertainty is nominally best on ideal accuracy and worst on preference AUC).
* **Per-budget table:** no consistent ordering across budgets; differences are within about ±0.01–0.02 and are not tested.

## 5. Expectation form versus classifier2 sampling (cross-check of each term against `train_unified.py`)
* `__getitem__` (lines 153–168) picks the source by one uniform draw: < 0.60 pair, < 0.85 ideal, else bad. `compute_loss` (lines 313–356) normalises each outcome class as mean × count / batch size, i.e. a sum / batch size, so the loss is the mixture 0.60 / 0.25 / 0.15 of per-source means; V1a uses the expectation (full batch, no sampling noise; the 500 samples per epoch of the original are one noisy estimate of it).
* **L_pair:** classifier2 draws a valid pair row uniformly and swaps the image order with probability ½; BT, tie (|d|) and not_apply terms are symmetric under the swap, so the expectation equals (1/n) Σ c_j l_j exactly. Difference: here n counts only revealed judgments (active learning), classifier2 uses all training rows. Confidence weight: `confidence_weight` of the repository loader (`pairwise_active_learning_pipeline.py:242`: Confident 1.0, Somewhat sure 0.7, missing 1.0 — same map as classifier2's `CONFIDENCE_MAP`).
* **L_ideal (`_get_ideal_sample`, lines 203–249):** type uniform among types with images, image uniform within the type, 70 % trajectory image uniform, 30 % other type uniform then image uniform. V1a's closed form is therefore the mean over t of [0.7 · mean over (x,u) + 0.3 · **mean over o of** mean over (x,x′)] (mean of means over o; the prompt's wording "mean over other types o ≠ t, x, x′" could also be read as a pooled mean weighted by |R_o|; I followed classifier2's sampler; the effect of the other reading is unverified). `test_ideal_and_bad_expectation_equal_classifier2_sampling_in_the_limit` checks the closed form against 60,000 samples of this sampler (agreement within 2 %). Differences from classifier2: references only (88 images, four types; classifier2 samples from all ideal images of five types including Twinned and uses all trajectory images), T without byte-identical copies, no image augmentation (features are cached), ideal always "wins" (same).
* **L_bad (`_get_bad_sample`, lines 251–281):** bad image vs uniform trajectory, output index uniform among the 5 outputs, trajectory wins: matches the closed form (τ over the 5 outputs including Twinned).
* **Dropout:** classifier2 trains in train mode (dropout 0.1 in its own model class); the repository's frozen-feature `fit_head` trains with the head in eval mode (dropout inert, order-independent full batch) and V1a follows the repository. Effect on the result: unverified.

## 6. Reused repository code (file:line, at af8413325)
| need | reused function |
|---|---|
| ideal split, pair split, candidate pool, held-out set | `judgment_unit_study.py:84-103` (`Context`), `run_pair_endpoint_study.py:40-43` (`setup`), `pair_preference_endpoint.py:36-54` (`split_classifier2_style`) |
| judgments / rows | `judgment_unit_study.py:74-81` |
| sequential protocol (rounds of 10, retrain every round) | `judgment_unit_study.py:185-198` (`trajectory`), `:129-157` (`make_candidates`, `choose`), `:160-164` (`metrics_for`); Random: `:138-144` |
| feature cache / frozen encoder | `build_feature_cache.py`, `new_directions/nd_temporal.py build`; `frozen_encoder_reward_head.py:33-57` (`FrozenFeatures`), `:83-104` (`train_model` — initial-head and head-seed handling copied into `fit_v1a`) |
| win-rate evaluation | `frozen_encoder_reward_head.py:107-121` (`evaluate_model`; identical rule; `win_rate_predict` re-implements it only to return predictions for macro-F1) |
| secondary endpoint | `pair_preference_endpoint.py:83-94` (`evaluate_full`) |
| Uncertainty | `judgment_unit_strategies.py:254,265-282` → `pairwise_active_learning_pipeline.py:579` → `pair_acquisition_methods.py:135-138` → `score_uncertainty` `:75-` (type-aware: own-type head entropy) |
| Core-set | `judgment_unit_strategies.py:274` (pair-level wrapper `:193-`) → `pairwise_active_learning_pipeline.py:567-575` → `pair_acquisition_shared_calculations.py:52-62` |
| TypiClust, ProbCover | `new_pair_strategies.py:80-98`, `:179-` via `judgment_unit_strategies.py` pair-level wrapper |
| Laplace-BALD, BALD×P(decisive), Ensemble-BALD | `judgment_unit_strategies.py:97-105`, `:122-128` (ensemble size 8, `judgment_unit_study.py:38,150`); `new_pair_strategies.py:342` (`_decisive_probability`) |
Note on Core-set and TypiClust/ProbCover: in the judgment-unit protocol they act on unique image pairs and reveal one random judgment of each chosen pair (`pair_level`, `judgment_unit_strategies.py:193-`), not on individual judgments; this is the repository's behaviour in V2 and was not changed.

## 7. Deviations from the prompt, and what is unverified
1. **Random = mean of 5 draws** (repository convention of V2: `random`, `random_r1…r4`) instead of a single draw; the 4 extra draws are in `all_rows.csv`.
2. Tuning label sets (my fixed random draws) and the use of split-B contexts for tuning are my choices.
3. Fits on identical label sets are shared between rules (memoised in `V1aContext.fit`; deterministic, so results are unchanged).
4. **T contains the images of the held-out pair groups** (as specified: all trajectory images except ideal copies). The ideal-accuracy endpoint is unaffected, but the *secondary preference endpoint is not identity-safe* with respect to unlabelled T (labels are never used, images are). The preference numbers should not be compared with the identity-safe V2/graph studies without this caveat.
5. Rebuilt feature caches (`build_feature_cache.py`, `nd_temporal.py build`) are **bit-identical (max abs difference 0.0)** to the caches that already existed in this container; those had been built earlier in the same container by the same scripts, so this checks determinism, not an independent reproduction. No cache from another machine was available.
6. Dropout inert in training (Section 5); eval-mode training differs from classifier2's train-mode behaviour; effect unverified.
7. The ideal split equals V2's only to the extent that the same harness (`make_experiment`) and seeds were used; V2 result files were not re-read to compare the split element by element (unverified).
8. Confirmed absence of evidence is not evidence of absence: the primary endpoint has a ±0.01 resolution and a ≈0.012 headroom; "no difference" means "not detectable here".
9. The Twinned output (index 1) enters only L_bad (as in classifier2) and is never used for classification.

## 8. Files
`v1a_core.py` (loss, fit, evaluation, context), `tune.py`, `run_v1a.py` + `run_all.sh`, `aggregate.py`, `make_report.py`, `test_v1a.py`, `FROZEN.md` / `frozen_setting.json`, `results/{tuning_grid,tuning_cells,all_rows}.csv`, `results/by_seed/seed*.csv` (one file per seed, rows = rule × budget), `results/tables.md`, `logs/`.
