# Handoff — temporal-graph study (branch `claude/temporal-graph-20261013`, cut from `claude/redteam-20261013`)

## State
Finished: development (seeds 9200–9219; 18 variants over three rounds + the blocked split), frozen pre-registration (`PREREGISTRATION_TG1.md`, commit ce128c184, before any confirmation cell), confirmation (35 seeds 9250–9284, splits A/B, one draw, run once; plus the red-team blocked split), exploratory time-proximity audit on spare seeds 9285–9299. Results: `RESULTS_temporal_graph.md` (generated), `SUMMARY_ZH_temporal_graph.md`, `GPU_TASKS_temporal_graph.md`, literature `notes_temporal_graph_literature.md`, development tables `TG1_dev_tables.md`, log `PROGRESS_LOG_temporal_graph.md`.

**Result in one line:** temporal edges (chain ±5 decayed, gentle chain, second-difference penalty) do not improve `lap3_k10_st4`: no candidate has a Holm-significant test (gains ±0.002 AUC, ±0.004 accuracy), none passes the success rule; blocked split null; the only nominal effects are anchor-near AUC gains and a worse log-loss for the curvature penalty.

## Seeds (do not reuse)
Block 9200–9299 is this study's: development 9200–9219, **confirmation 9250–9284 used**, spare 9220–9249 unused, 9285–9299 used by the exploratory audit (`results/tg/audit_time`). 8000–8199, 8400–8799, 8885–8999 belong to parallel conversations. `nd_core.py` has the `TG_*` lines with assertions.

## Code map
- `new_directions/nd_temporal_graph.py`: edge builders (`chain_edges`, `tbin_edges`, `curvature_triples`), controls (`within_run_permutation` full/local, random-kNN-edges), strict rule (`near_heldout`), the learner `fit_tg` (reproduces `lap3_k10_st4` exactly with no temporal term, unit-tested), registry `LEARNERS`/`SPEC_OF` (`tg_<variant>[_mu<x>][_shuf|_lshuf|_krand|_strict]`).
- `new_directions/tg_run.py` (runner; standard / `--splits blocked`; writes anchor-far/near and in-block strata), `tg_audit_time.py` (per-judgment predictions with time-gap and anchor distance, spare seeds only).
- `graph_exploration/tg_tables.py` (dev / confirm with Holm + verdicts / budget / audit / edges), `tg_report.py` + `results_narrative_temporal_graph.md` -> `RESULTS_temporal_graph.md` (edit the narrative, never the generated file), `tg_graph_stats.py`, `tg_run_stats.py`, `scripts_tg/{run_tg.sh,chain_confirm.sh,autocommit.sh}`, `TG1_spec.json`.
- Tests: `code_behavior_tests/test_temporal_graph.py` (10): seed block, chain/curvature semantics, permutation controls, held-out images never nodes, strict rule, blocked-context buffer for in-block test images, exact baseline equivalence.
- Results: `new_directions/results/tg/{dev1,dev2,dev_pool,devb,confirm,confirm_blocked,audit_time}` (`dev1` also holds the far-edge re-weighting learners; `dev_pool` is a filtered copy of dev1+dev2 used for the pooled table).

## Environment pitfalls
- No torch in the container image: PyPI's torch needs ~3 GB of CUDA libraries (slow proxy); use conda-forge instead: micromamba from `conda.anaconda.org/conda-forge/linux-64/micromamba-2.0.5-0.tar.bz2`, env `rl` = python 3.13 + `pytorch=*=cpu*` + torchvision + scikit-learn + scipy + pandas (`/home/user/mm/env.sh`, not in git). Caches (`simclr_feature_cache.pt`, `feature_cache_all.pt`) are not in git: rebuild with `build_feature_cache.py` and `nd_temporal.py build` (features may differ in the last digits from the original torch build).
- `pkill -f <pattern>` kills your own shell when the pattern is in the command line; `tg_tables.load` pairs per seed and requires all budgets (cells are written after every budget).
- A temporal term with weight 3 costs ≈ 1.4× a plain `lap3_k10_st4` fit (≈ 10 s); a confirmation with 7 learners × 70 cells took ≈ 80 min on 4 cores.

## Findings worth keeping (details and caveats in the results document)
1. 84–94 % of the chain ±1…±5 edges are already kNN edges; the others link feature-distant frames and smoothing across them hurts (AUC, log-loss, anchor-far).
2. Temperature is not monotone in time (runs heat up and cool down; Spearman 0.30–0.63); the physics hypothesis was not needed or supported.
3. The red-team blocked split leaks buffer-wise through cross-block partner images (55–70 % of graph nodes within 25 frames of some held-out image); `inb_*` is the clean stratum.
4. The strict 25-frame rule removes ≈ 98 % of temporal edges in the standard splits.
5. Removing the ≈ 4 % cross-run / far-in-time kNN edges costs ≈ 0.006–0.007 AUC (development); up-weighting them does nothing.

## What I would do next
1. Item 1 of `GPU_TASKS_temporal_graph.md` (frame-level labels along a run) — it is the only way to learn whether temporal neighbours share labels, and everything else here is a proxy.
2. Temporal terms in a representation that is not already invariant (end-to-end), see GPU list.
3. If anyone revisits the time-blocked evaluation: make the held-out set a union of whole blocks (drop groups with a partner outside the blocks) so that the 25-frame buffer holds for every test image.
