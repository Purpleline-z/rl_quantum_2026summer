# Handoff — graph-mechanism branch `claude/graph-mechanism-20261013`

## State
Development (seeds 8000–8019, 60+ variants over rounds dev1–dev13: dose-response, constructions, placement, per-type, non-graph arm, λ/k sweeps, LDA / ImageNet / hidden-space / purified graphs), frozen pre-registration (`PREREGISTRATION_GM1.md`, committed at 0f33c6f65 before any confirmation seed), confirmation 1 (8100–8134) and confirmation 2 (8150–8184) are finished; results in `RESULTS_graph_mechanism.md` (generated), `SUMMARY_ZH_graph_mechanism.md`, `GPU_TASKS_graph_mechanism.md`, `PROGRESS_LOG_graph_mechanism.md`. Seeds 8000–8184 are now used; 8200–8599 belong to parallel sessions; use >= 8600 for new tests.

## Results in one line
(a) graph-quality dose-response at λ=10: CONFIRMED twice; (b) `gmc_t_no_rt13` (kNN k=10, λ=10, penalty off for RT13) better than `lap3_k10`: CONFIRMED twice (+0.009–0.010 AUC, +0.005–0.007 accuracy); `gm_lap10_n0` (uniform λ=10): confirmed in confirmation 2 only (tiny effect) -> not robust; no better graph construction than plain cosine kNN (k=10) found; non-graph smoothing does not reproduce the gain.

## Code map
- `new_directions/nd_graph_mechanism.py`: all learners (`LEARNERS`, names `gm_*`, `gm10_*`, `gm20_*`, `gmc_*` = confirmation candidates and controls); `build_edges` (constructions), `graph_quality`, `fit_gm` (aw8_all + Laplacian / non-graph terms). With the default spec it reproduces `nd_graph_learner`'s `lap<L>_k10` and `..._n<σ>` exactly (tests in `code_behavior_tests/test_graph_mechanism.py`). Graph quality of every trained graph is written to `new_directions/results/gm/quality/seed<S>_<split>.json`.
- `new_directions/nd_gm_layers.py` (intermediate SimCLR layers, `python3 nd_gm_layers.py build`), `nd_run_gm.py` (nd_run with the gm registry), `nd_core.py` (seed blocks GM_*, own lines at the end of the seed section).
- `graph_exploration/`: `gm_tables.py` (dev / dose / dosetest / confirm / qgain / diff / curve / types / budget), `gm_report.py` + `results_narrative_graph_mechanism.md` -> `RESULTS_graph_mechanism.md` (edit the narrative, never the generated file), `gm_edge_stats.py` (+ csv), `scripts_gm/{run_gm,chain_confirm,autocommit}.sh`, `GM1_spec.json`, `GM1_dose_{primary,secondary}.json`.
- Results: `new_directions/results/gm/{dev1..dev9,confirm1,confirm2,quality}`; `dev7_INVALID_cache_key_collision` is a first run of round 7 with a bug (cache key lacked r0/frac), kept for transparency, not used.

## Development-only rounds after the freeze (do not change the verdicts)
dev8 ImageNet topology (worse), dev9 fairer non-graph arm (mixup consistency still harmful, noise consistency ~0), dev10 per-type strengths (plateau), dev11 hidden-space graph (no gain), dev12 label-based purification (no gain; adding same-class ideal edges hurts), dev13 placement (accuracy gain comes from edges touching ideal images), dev14 λ of the candidate (5/10/20 equal), dev_cold (candidate still +0.006/+0.011 AUC over `lap3_k10`), dev17 local rewiring (50 nearest; AUC gain stays positive, so long-range shortcuts are what destroys the gain), dev16 random rewiring of 10–75 % of the edges (very steep: −0.04 AUC already at 10 %), dev15 L1/Huber penalties (L1 catastrophic on degraded graphs) and partial RT13 weights (0.1/0.3 as good as off). `gm_integrity.py` checks the confirmation runs (all pass).

## Re-running
`bash graph_exploration/scripts_gm/run_gm.sh OUT LEARNERS SEEDS A,B 4` (idempotent; 1 torch thread per process; 4 shards; ≈ 70 min per 35 seeds for 17 learners). Caches (not in git): `python3 pair_disjoint_not_image_disjoint/build_feature_cache.py`, `python3 new_directions/nd_temporal.py build`, `python3 new_directions/nd_gm_layers.py build` (PYTHONPATH=code/active_learning_program). Never rerun confirmation seeds with changed candidates.

## Pitfalls
- The graph cache in `gm_graph` is keyed by `_KEYS`: if you add a graph parameter, add it there (a missing key silently reuses another variant's graph — happened once).
- `chain_confirm.sh` waits for ALL `nd_run_gm.py` processes (also unrelated dev runs).
- The old tests rewrite `results/active_learning_v1.8_seed42/manifests`: restore with `git checkout --`.
- Quality for LDA topologies is circular; quality for placement variants is undefined (no ideal-ideal edges).

## What I would do next
1. The encoder lever (GPU list): rebuild features with a better encoder, rerun `PREREGISTRATION_GM1` candidates and the dose-response on new seeds.
2. A cross-fitted learned graph (metric learned on ideal images, quality scored on held-out ideal images).
3. Type-specific smoothing strength (RT13 off, others on, was a hand switch; a learned per-type λ with proper nesting).
4. Prospective test on new growth data.
