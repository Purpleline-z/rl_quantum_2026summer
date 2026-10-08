# Hand-off: frozen-encoder study and new acquisition strategies (branch `claude/frozen-encoder-strategies`)

Written at the end of an unattended session. Everything below ran on CPU; nothing needs a GPU yet.

## What was done
1. **Branch tidy-up** (earlier, on `claude/intelligent-franklin-a4xn5m`): merged the seed 202/303 work, copied the missing seed 202/303 Task 3c cells from Drive with `colab/copy_task3c_cells_from_drive.py`, updated the report to five seeds.
2. **Diagnosis of the low Task 3c accuracy**: no implementation bug found. Causes: very small labelled sets for end-to-end fine-tuning, training that depends on the order of the selected pairs, a 28-image outer test, and at budget 100 all strategies acquire the identical 100 groups.
3. **Frozen encoder** (`frozen_encoder_reward_head.py`, tests in `code_behavior_tests/test_frozen_encoder_order_invariance.py`): cached SimCLR features, head trained full-batch on a sum of per-row losses, so the result depends on the set of pairs and not their order.
4. **Type accuracy cannot compare strategies** (`explore_anchor_vs_pairs.py`): reference anchors alone give 0.848 with zero pair groups; pair labels alone give 0.37–0.55.
5. **New endpoint**: held-out preference prediction (`pair_preference_endpoint.py`: image-disjoint validation/test groups; accuracy, log-loss, AUC, calibrated log-loss).
6. **25 strategies** (8 original + 17 in `new_pair_strategies.py`), 35 seeds, single-shot and sequential (`run_pair_endpoint_study.py`, `run_pair_endpoint_sequential.py`), aggregated with Wilcoxon + Holm and Friedman (`aggregate_pair_endpoint.py`), figure `paper_assets/endpoint_gain_over_random.png`.
7. **Report**: new §5.11 (why freeze, results) and §5.12 (held-out preference endpoint); abstract and conclusion rewritten. Literature notes: `notes/literature_and_new_strategy_ideas.md`.

## Main findings (see §5.11–5.12 and the notes)
- Single-shot, 35 seeds: Laplace BALD and BALD x P(decisive) beat Random on log-loss and AUC after Holm correction (about -0.04 log-loss, +0.01 AUC); Cluster-quota uncertainty and DPP on log-loss only; graph cut worse. Sequential: nothing beats Random; graph cut worse. Uncertainty-driven rules never beat Random.
- Two earlier five-seed impressions did not survive 35 seeds (Cluster-Margin / core-set advantage; type-covering initial set), and are stated as non-replications.

## Things to be aware of
- The eight original strategies score a candidate with the type of the group's *first judgment row* (`Experiment.candidates_with_clusters`), which the author confirmed is the type the system queries (the annotator judges one type per query), so this is legitimate. The 17 new strategies instead aggregate over the four heads (tested), and `_lf` variants of the originals do the same. Remaining mismatch: a selected group yields all its judgments (about 3.1) while the budget counts groups.
- Hyper-parameters (lr 0.01, 100 steps) were tuned on random batches on seeds 42/79/123, not per strategy.
- The feature cache `results/frozen_encoder_task3/simclr_feature_cache.pt` (SimCLR encoder, 444 images) is committed so CPU experiments start instantly; delete it and run `build_feature_cache.py` to rebuild.
- Running the experiment scripts rewrites manifests under `results/active_learning_v1.8_seed*/manifests`; the scripts now send manifests to a temporary directory, but older scripts do not.

## Suggested GPU experiments for tomorrow
1. **Second encoder** for the held-out-preference study (ImageNet ResNet-18, DINOv2 or another self-supervised model): re-run `build_feature_cache.py` with that encoder and the same two study scripts; checks whether the Fisher/Laplace-posterior advantage is encoder-specific.
2. **End-to-end fine-tuning with order-independent training** (full-batch via gradient accumulation) on the held-out-preference endpoint, to see whether the frozen-head conclusions transfer to the fine-tuned model that the original Task 3 used.
3. **Strategy-specific hyper-parameters** (learning rate/steps per strategy) for the top five strategies, selected on validation only.
4. **More candidates**: re-split with a larger candidate pool by moving groups from the unused part of the 168 (or use the 58 pair groups that the original Task 3 pool left unused) to test budgets beyond 60.
5. (Done in this session, `_lf` variants; see §5.12.) Re-deriving the original rules natively for four heads rather than silencing the Twinned head.
