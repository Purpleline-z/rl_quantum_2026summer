# Handoff — calibration of `lap3_k10_st4` (branch `claude/calibration-20261013`)

## State
Development screens `dev1`, `dev2` (seeds 9100–9119) and development side runs (baseline learners, RT13-off learner), frozen pre-registration (`PREREGISTRATION_CAL1.md`, `CAL1_spec.json`; commit ef0410d0f, before any confirmation cell), confirmation (seeds 9150–9184, splits A/B, run once, `results/cal/conf1`; reference learner `conf1_aw8`), generated results `RESULTS_calibration.md`, `SUMMARY_ZH_calibration.md`, `GPU_TASKS_calibration.md`, literature `notes_calibration_literature.md`, log `PROGRESS_LOG_calibration.md`. **Result in one place: `RESULTS_calibration.md`, section 1.**

## Code map (`new_directions/` unless stated)
- `nd_calibration.py`: stage 1 (`label_set_record`: final head + K cross-fitting heads by pair group with the full set's lr/steps → out-of-fold logits, test logits, geometry features; optional extra heads / label smoothing), `LEARNER_SPECS` (registry-equal specs for `lap3_k10_st4`, `lap3_k10`, `aw8_all`, `aw8_all_st2lr0.5`, plus the dev-only `lap3_k10_st4_rt13off`), stage 2 calibrators (`fit_global_t`, `fit_type_t`, `fit_prior`, `fit_conf_t`, `fit_shrink_far`, `fit_davidson`, `variants`) and `scores` (log-loss, accuracy, AUC, per-type, anchor-far/near, reliability-bin sums, ECE).
- `cal_run.py`: runner (resumable, one gzip-pickle per (seed, split)); `cal_tables.py`: `dev`, `oracle`, `budget`, `transfer`, `learners`, `confirm` (Holm, success rule) modes; `cal_figures.py`: reliability diagrams; `graph_exploration/cal_report.py` + `results_narrative_calibration.md` → `RESULTS_calibration.md` (edit the narrative, never the generated file); `graph_exploration/scripts_cal/{run_conf.sh,autocommit.sh}`.
- Tests: `code_behavior_tests/test_calibration.py` (9: seed blocks, fold construction, calibrators on synthetic data, equality of the stage-1 head with the registry learners, invariance of every variant to test labels).
- Seeds: `nd_core.py` `CAL_*` lines (dev 9100–9119, confirmation 9150–9184; 9120–9149 and 9185–9199 unused).

## Pitfalls
- Caches are not in git: `python3 pair_disjoint_not_image_disjoint/build_feature_cache.py` and `python3 new_directions/nd_temporal.py build` (needs torch + torchvision; a CPU install via `pip` works, the container had none).
- Do not use `pkill -f <pattern>` from the shell (it kills the calling shell).
- `variants()` in `nd_calibration.py` computes every variant for every record: adding a variant changes nothing stored; the stage-1 cells are all that is needed to re-score. Never change a frozen candidate's definition after the confirmation was seen.
- `fit_prior_test` uses TEST labels (development diagnostic only, `T_typetest_fixed`, `CF_typetest_fixed`): never use it in a confirmation.
- ECE is pooled per seed over label sets; the number depends on that choice.
- Old tests rewrite `results/active_learning_v1.8_seed42/manifests` (`git checkout --` before committing; this study's tests do not).

## What I would do next
1. The per-type constants need an independent calibration set (new growth runs; GPU_TASKS item 3) and a check with a better encoder (item 1).
2. A learner-side fix that removes the RT13/c(6x2) score compression without losing accuracy (per-type Laplacian weight in the end-to-end model), then calibrate.
3. Calibration under shift: harsh splits of `claude/graph-endpoint-1013` (held-out growth run, held-out temperature) with these calibrators; the anchor-far stratum is only a proxy.
4. Selection rules under the calibrated learner (not examined).
