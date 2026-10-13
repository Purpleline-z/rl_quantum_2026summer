# Handoff — regularisers on unlabelled images, graph vs graph-free (branch `claude/regularisers-20261013`, cut from `claude/redteam-20261013`)

## State
Done: development rounds `dev1` / `dev2` (seeds 9000–9011, 20 variants, `results/rg/dev1`, `dev2`), frozen pre-registration `PREREGISTRATION_G2.md` + `G2_spec.json` (commit d32ccbb3c, before any confirmation cell), confirmation `conf` (6 learners × seeds 9050–9084 × splits A/B × 4 budgets × 2 draws) and `conf_ctl` (two shuffled-graph controls, draw 0), results `RESULTS_regularisers.md` (generated), `SUMMARY_ZH_regularisers.md`, `GPU_TASKS_regularisers.md`, literature `notes_regularisers_literature.md`. Unused: seeds 9012–9019 (dev block). **Do not reuse 9000–9084.**

## Code map (`new_directions/` and `graph_exploration/`)
- `nd_regularisers.py`: `_train` / `fit_reg` (aw8_all head + one extra term/switch: `cons`, `noise`, `mirror`, `mix`, `rdrop`, `hdrop`, `wd`, `ema`, `swa`, `inner`, `lap`, `lap_hidden`); registry `LEARNERS` (`rgcons<w>_st<x>`, `rgnoise…`, `rgcn_c0.3_n1_st<x>`, `rglap3_cons0.3_k10_st<x>` + `_shuf_`, …). Equivalences with `aw8_all`, `cons<L>_st<x>` and `lap3_k10_st<x>` are unit-tested.
- `rg_run.py`: runner (idempotent; records have `selector = "random"`, standard metrics + `*_far` / `*_near` + per-judgment `d`); anchor-far = both images > 0.03 from every typed ideal image (`anchor_min_distance`, same definition as `rt_a1b.py`).
- `rg_tables.py` (dev / confirm / budgets; Holm, non-inferiority, superiority, verdict rule), `rg_report.py` (builds `RESULTS_regularisers.md` from `results_narrative_regularisers.md` — edit the narrative, never the generated file), `scripts_rg/{run_rg.sh,autocommit.sh,chain_conf_ctl.sh}`.
- Tests: `code_behavior_tests/test_regularisers.py` (8).
- Seeds: `nd_core.py` (`G2_DEV_SEEDS`, `G2_CONFIRM`, assertions).

## Pitfalls
- The feature caches are not committed (`*.pt` is gitignored; the original protocol cache was missing from the checkout): create an empty protocol cache `torch.save({}, single_study.CACHE)`, then run `python3 nd_temporal.py build` and `python3 nd_build_mirror.py` (a few minutes each). With torch 2.14 the committed selection-htr cells of seed 8300 split A are reproduced to four decimals.
- `pkill -f` / `pgrep -f` patterns match your own shell command line (killed my shell twice): use PIDs.
- A shard writes a file after every budget; `rg_tables.load` skips files with fewer records than the fullest file of a directory.
- Wilcoxon over seeds on the same 168 pair groups: optimistic p-values; the far stratum has ≈ 12–15 judgments per seed.
- Manifold mixup in the hidden layer is vacuous for the one-hidden-layer head (tested); do not re-try it.

## What I would do next
1. GPU: image-space consistency (see `GPU_TASKS_regularisers.md`).
2. Get data far from the anchors (new growth runs) — the only way to settle whether any of this helps away from near-duplicates of ideal images.
3. Tune the consistency family on more seeds / weights per budget (weight 0.3–1 and 4×–6× steps were nearly equivalent; accuracy gap to the Laplacian persisted).
4. Selection on top of the new learners was not examined here (the selection-htr study found ≈ 0).
