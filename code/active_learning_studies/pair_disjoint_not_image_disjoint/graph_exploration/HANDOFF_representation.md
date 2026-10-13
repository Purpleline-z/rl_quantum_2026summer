# Handoff — representation study (branch `claude/representation-20261013`)

## State
Literature notes + network check → development plan R0 (`PREREGISTRATION_R0_development_plan.md`) → development stage A (8600–8609, 12 learners), ensemble control, stage B1 (graph-free head on every representation, 8600–8619), stage B2 (learning-rate re-tuning, stopped early) → frozen `PREREGISTRATION_R1.md` (commit c29e696c3, before any confirmation cell) → confirmation 1 (8650–8684, splits A,B) → **not confirmed** (no candidate improves `lap3_k10_st4`; see `RESULTS_representation.md`) → post-hoc controls for the HTR signal of `rp_gs4_his` on development seeds (`devH1`, `devH2`). Seeds 8600–8619 and 8650–8684 are used; **do not reuse**; 8000–8199, 8400–8599 and the other blocks of 8600–8999 belong to parallel studies.

## Code map
- `new_directions/nd_core.py`: `RP_DEV_SEEDS`, `RP_CONFIRM1`, `RP_SEEDS_ALL` (own lines, with assertions; merged into `USED_SEEDS_BY_EARLIER_WORK`).
- `new_directions/nd_rep_build.py`: encodes 7 deterministic views per image with the frozen SimCLR encoder (`results/rep/simclr_views.pt`; asserts view 0 equals the cached feature).
- `new_directions/nd_representation.py`: representations (`raw_rep`, `whiten_rep`, `ctx_rep`: s4, s3, i4, mir, mc, s34, is, s3is, cap, w64/w128), `rep_learner(head, graph, lap, steps_scale, shuffle, ensemble, lr_scale)` and the `LEARNERS` registry (`rp_base` ≡ `lap3_k10_st4`; `rp_<rep>` / `_shuf`; `rp_g<G>_h<H>`; `rp_gf_*` graph-free; `rp_*_lr*`; `rp_cap`; `rp_ens5`). To add an encoder: add a `path -> tensor` table in `_blocks()`, a branch in `raw_rep`, a learner line.
- `new_directions/nd_run_rp.py`: runner (extends `nd_run.registry`); `graph_exploration/scripts_rp/run_rp.sh OUT LEARNERS SEEDS SPLITS [nshards]` (idempotent; results to `new_directions/results/rp/OUT`, logs in `new_directions/logs_rp`); `scripts_rp/autocommit.sh` (8-minute commit of results; stop with `touch /tmp/rp_stop_autocommit`).
- `graph_exploration/rp_tables.py` (dev / confirm / diff / types / budget), `rp_report.py` (assembles `RESULTS_representation.md` from `results_narrative_representation.md`; edit the narrative, never the generated file), `R1_spec.json`, `rp_ens_table.py`.
- Tests: `code/active_learning_program/code_behavior_tests/test_representation.py` (8 tests: seed blocks, `rp_base` ≡ `lap3_k10_st4`, scale matching, views, whitening/identity safety, capacity control, learners train, `rp_gf_s4` ≡ `aw8_all_st2lr0.5`).
- Results: `new_directions/results/rp/{devA,devE,devB1,devB2,devH1,devH2,confirm1}/seed*_{A,B}.json` (`devB2` partial; tables use only complete cells).

## Pitfalls
- Network: HuggingFace / torch hub / download.pytorch.org → HTTP 403; arXiv/CVF blocked; only raw.githubusercontent.com and PyPI reachable. No new pretrained weights.
- A cell file exists after the first budget; `rp_tables.load_complete` skips files with fewer records than the fullest file of the directory.
- Cost: `lap3_k10_st4`-type learners ≈ 6 s per fit; confirmation 1 (6 learners x 35 seeds x 2 splits) took ≈ 90 min on 4 cores. Container restart kills background jobs: re-run the same `run_rp.sh` line.
- `_proxy(ctx, name, table)` shares a graph cache per (context, representation); the whitened representations depend on the allowed node set (`ctx_rep`).
- Old tests rewrite `results/active_learning_v1.8_seed42/manifests` (`git checkout --` before committing).

## What I would do next
1. A stronger or different frozen encoder (needs network or GPU; see `GPU_TASKS_representation.md`), run through the identical protocol and a fresh seed block.
2. Fine-tune the SimCLR encoder on all unlabelled growth images (the only representation lever excluded here).
3. HTR-specific: why does the ImageNet ⊕ SimCLR head improve HTR log-loss (section 6 of the results)? A new HTR-focused labelling batch is probably a better lever than representation.
4. Per-representation re-tuning of the head schedule on development seeds, then a pre-registered confirmation.
