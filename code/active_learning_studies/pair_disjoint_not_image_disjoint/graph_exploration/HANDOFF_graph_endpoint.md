# Handoff for the next Claude — graph-endpoint study (branch `claude/graph-endpoint-1013`)
Read first: `RESULTS_graph_endpoint.md` (generated; section 1 is the answer), `SUMMARY_ZH_graph_endpoint.md`, `PREREGISTRATION_GE0.md` / `GE1.md`, `PROGRESS_LOG_graph_endpoint.md`, `notes_graph_endpoint_literature.md`, `GPU_TASKS_graph_endpoint.md`. Common rules: `prompts/COMMON_RULES.md`.

## State
Finished: dev screens, frozen pre-registration, confirmation 1 (8500–8534) and confirmation 2 (8550–8584), exploratory mechanism runs, generated tables/results, tests. Results in one line each: (a) classification — graph methods do not beat 1-NN/logistic regression attributably to the graph; the Laplacian reward head is worse; (b) dose-response — trend test formally met but it is the p = 0 → 10% step only, flat afterwards; (c) harsh splits — `lap3_k10` keeps +0.02 AUC / +0.015–0.03 accuracy, controls fail, rule met in both blocks, but test sets are fixed and tiny (bootstrap intervals include 0).

## Code map (`code/active_learning_studies/new_directions/`)
`ge_classify.py` (controls + graph classifiers + graph builders for the classification endpoint), `ge_heads.py` (`fit_head` = `nd_graph_learner.fit_mlp_graph` with a reference-only-anchor switch, unit-tested equal to `aw8_all`/`lap3_k10`; `make_graph` with node subsets, dropped frames, held-out pair images), `ge_harsh.py` (held-out run / temperature folds from file-name metadata, identity assertions), runners `ge_run_class.py`, `ge_run_dose.py`, `ge_run_harsh.py` (`--save-d` stores per-judgment scores for the test-group bootstrap), `ge_run_lam.py`, `ge_run_nodes.py`, `ge_run_lamclass.py`; analysis `ge_analyze.py`, `ge_report.py` (all tables and verdicts; `--template ... --doc ...` builds the results file); scripts `ge_conf.sh`/`ge_conf_all.sh` (idempotent confirmation chain), `ge_dev_*.sh`, `ge_autocommit.sh`. Tests: `code_behavior_tests/test_graph_endpoint.py` (8 tests). Seeds: own block in `nd_core.py` (dev 8400–8419, 8500–8534, 8550–8584); the sibling selection session used 8000–8479 (overlap with dev 8400–8419 only).

## Pitfalls
- `aw8_all` and every `*_all` learner train on the outer-test ideal images (all-ideal anchors): they are invalid for the classification endpoint; use `ge_heads.fit_head(all_ideal=False)`.
- The all-image feature cache (`results/feature_cache_all.pt`, 1278 images) is not committed: rebuild with `python3 nd_temporal.py build` (a few minutes) — graph runs fail without it.
- The harsh splits' test judgments are identical across seeds; never read their seed-level p-values as test-set uncertainty.
- Container restarts kill background jobs; `ge_conf_all.sh` and the `ge_dev_*.sh` scripts are idempotent (unfinished seeds are redone).
- Old tests rewrite `results/active_learning_v1.8_seed42/manifests` (restore with `git checkout --`).

## What I would do next
GPU/lab list in `GPU_TASKS_graph_endpoint.md`; above all, more independent test groups for the harsh splits, and a graph from a better encoder.
