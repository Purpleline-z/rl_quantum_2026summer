# Handoff — recipe search around `lap3_k10_st4` (branch `claude/recipe-20261013`)

## State
Development (seeds 9300–9319), plan `PREREGISTRATION_RC0_development_plan.md` (+ amendment RC0b), frozen pre-registration `PREREGISTRATION_RC1.md` / `RC1_spec.json` (committed before any confirmation seed), confirmation on seeds 9350–9384 (run once). Results: `RESULTS_recipe.md` (generated), `SUMMARY_ZH_recipe.md`, `GPU_TASKS_recipe.md`, literature `notes_recipe_literature.md`.

## Code map
- `new_directions/nd_recipe.py`: `Spec` dataclass (every knob, default = `lap3_k10_st4`), `fit_recipe`, `VARIANTS` (all `rc_*` learners incl. the unscreened ones), `combine` / `resolve` (`rcx_<a>+<b>` combinations on the fly). Equivalence with `lap3_k10_st4` is unit-tested.
- `new_directions/rc_run.py`: runner (paired learners on identical label sets; stores per-judgment scores, anchor distances, standard / anchor-far / per-type metrics; resumable). `graph_exploration/scripts_rc/run_rc.sh` (4 shards), `autocommit.sh`.
- `graph_exploration/rc_tables.py` (per-seed gains, Wilcoxon, bootstrap, Holm, MDE, per-type, strata), `rc_confirm.py` (the pre-registered families and verdicts), `rc_report.py` + `results_narrative_recipe.md` → `RESULTS_recipe.md` (edit the narrative, never the generated file).
- `nd_core.py`: seed block `RC_DEV_SEEDS` 9300–9319, `RC_CONFIRM` 9350–9384 (separate lines with assertions; **do not reuse**).
- Tests: `code_behavior_tests/test_recipe.py`, `test_rc_tables.py`. Results: `new_directions/results/rc/{dev1,dev2,conf}/seed*_{A,B}.json`.

## Pitfalls
- The feature caches (`results/feature_cache_all.pt`, `simclr_feature_cache.pt`) are not committed: `python3 build_feature_cache.py` then `python3 nd_temporal.py build` (≈ 1 min each, torch + torchvision + sklearn + scipy needed).
- The repository trains its head in eval mode, so the head's `Dropout(0.2)` is inert; `rc_do0.3` switches it on in training (and the scores shrink) — not comparable to "dropout in the incumbent".
- ≥ 95 % of the judgments have `confidence_weight` 1.0 (rest 0.7): confidence-weight knobs are no-ops.
- The anchor-far stratum has 4–15 judgments per cell; its AUC is undefined for single-class cells (skipped).
- `rc_run.py` appends to existing files; never start two processes on the same (seed, split) file (shards are by seed index).
- Old tests rewrite `results/active_learning_v1.8_seed42/manifests` (`git checkout --` before committing).

## What I would do next
See `GPU_TASKS_recipe.md`; on CPU: the 7 unscreened knobs (implemented: `rc_aw4, rc_w512, rc_w128, rc_d3, rc_dn0.3, rc_b99, rc_noclip`), then a joint random search with a nested-seed protocol.
