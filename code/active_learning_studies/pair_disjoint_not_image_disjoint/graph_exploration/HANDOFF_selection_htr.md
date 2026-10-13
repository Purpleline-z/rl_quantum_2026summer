# Handoff — selection × Laplacian learner × HTR (branch `claude/selection-htr-20261013`)

## State
Development screens sdev1–sdev6 (seeds 8200–8219), frozen pre-registration (`PREREGISTRATION_SH1.md`, commit 232d41d11, before any confirmation cell), confirmation 1 (8300–8334), confirmation 2 (8350–8384), secondary selection-under-long-schedule run (`c1_sel4`), cold-start addendum (`PREREGISTRATION_SH2_cold_start.md`, seeds 8385–8399). Results: `RESULTS_selection_htr.md` (generated), `SUMMARY_ZH_selection_htr.md`, `GPU_TASKS_selection_htr.md`, literature `notes_selection_htr_literature.md`.

## Code map
- `new_directions/nd_select_graph.py`: graph-aware selectors (`gdens_unc`, `gcover`, `gcover_dec`, `gvopt_node{1,3,10}`, `vopt_htr{2,4}`, `vopt_bridge{1,3}`, each graph rule with `_shuf`); registered in `nd_select.py`, which now also registers the earlier typed/`gvopt_*` selectors, the graph learners, and records the type composition of every chosen set (`n_type0/2/3/4`).
- `new_directions/nd_graph_learner.py`: new learner options `lap_w` (per-type Laplacian weight), `anchor_w` (per-type reference-anchor weight), `lr_scale`, `steps_scale`; names `lap3_k10_st4` (frozen candidate), `aw8_all_st{2,3,4}`, `aw8_all_st2lr0.5`, `lap3h10_k10`, `lap3_ha{2,4}_k10`, `lap3_k10_st4_shuf`, … (`aw8_all_st<x>` is the Laplacian-free learner with the same schedule knobs; equivalence with `aw8_all` at x = 1 is tested).
- `new_directions/nd_core.py`: `SH_*` seed blocks (8200–8219, 8300–8334, 8350–8384, 8385–8399; **do not reuse**; 8000–8199 and 8400–8599 belong to the two parallel studies).
- `graph_exploration/sh_tables.py` (dev / comp / types / learners / confirm tables and verdicts), `sh_report.py` (assembles `RESULTS_selection_htr.md` from `results_narrative_selection_htr.md`; edit the narrative, never the generated file), `SH1_spec_*.json`, `SH2_spec_cold.json`, `scripts_sh/{run_sel.sh,chain_*.sh,autocommit.sh}`.
- Tests: `code_behavior_tests/test_selection_htr.py` (6) + `test_graph_learner.py` (8).
- Results: `new_directions/results/sh/{sdev1..sdev6,c1_sel,c1_learn,c2_sel,c2_learn,c1_sel4,cold_learn}/seed*_{A,B}.json` (`sdev6` is partial: stopped early; the tables use only completed cells).

## Pitfalls
- A selector name missing from `nd_select.SELECTORS` kills the shard at the first selection and the chain script then moves on silently (happened once: `typed_decisive_bald_shuffled`, fixed). Always smoke-test every name of a chain.
- `lap3_k10_st4` costs ≈ 4× the compute of `lap3_k10` per fit; a cell with 7 selectors takes minutes. 4 cores, one process per shard.
- Cells are written after every budget, so a file existing does not mean the cell is finished; `sh_tables.load` skips files with fewer records than the fullest file of the directory.
- Old tests rewrite `results/active_learning_v1.8_seed42/manifests` (`git checkout --` before committing).

## What I would do next
1. GPU: Laplacian term + long schedule in the end-to-end model (see `GPU_TASKS_selection_htr.md`).
2. Try to break the learner result: a stronger graph-free baseline (other regularisers at the long schedule: dropout/mixup/weight decay/SWA), a different k or graph quality at the long schedule (the shuffled and noisy graphs were far worse).
3. Selection from a cold start and in sequential rounds with `lap3_k10_st4` (not examined).
4. HTR: more HTR references/pairs (data), prospective HTR-weighted arm.
