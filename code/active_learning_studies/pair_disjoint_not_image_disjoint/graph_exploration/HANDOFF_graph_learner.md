# Handoff — graph-learner branch `claude/graph-learner-20261012`

## State
Everything planned in the prompt is finished and pushed: development screens (dev1–dev3, dev_cold), frozen pre-registration (`PREREGISTRATION_G1.md`, `G1_spec.json`, committed at 3999406bf before any confirmation seed ran), confirmation 1 (7100–7134), confirmation 2 (7200–7234), cold-start confirmation (7300–7334). Results: `RESULTS_graph_learner.md` (generated), `SUMMARY_ZH_graph_learner.md`, `GPU_TASKS_graph_learner.md`, `PROGRESS_LOG_graph_learner.md`.

## Code map
- `new_directions/nd_graph_learner.py`: `GraphData` (identity-safe kNN graph, label spreading, inductive attachment), `fit_mlp_graph` (aw8_all + `lap`/`rankq`/`cons`/`ridge` terms), `fit_gcn2`, learner registry `LEARNERS` (names like `lap3_k10`, `lap3_k10_shuf`, `lap3_k10_n3`, `lap3_rq3_k10`, `sgc_k10`, `gcn2_k10`, `ridge1`, `cons0.3`). `nd_run.py` imports it.
- `new_directions/nd_core.py`: seed blocks `GL_DEV_SEEDS`, `GL_CONFIRM1/2`, `GL_COLD` (added to `USED_SEEDS_BY_EARLIER_WORK`; 7000–7335 are now used — do not reuse them).
- `code_behavior_tests/test_graph_learner.py` (8 tests; `lap=rankq=cons=ridge=0` ≡ `aw8_all`).
- `graph_exploration/gl_tables.py` (dev/confirm/budget/types tables), `gl_report.py` (assembles `RESULTS_graph_learner.md` from `results_narrative_graph_learner.md`; edit the narrative, never the generated file), `gl_mechanism.py`, `scripts_gl/run_nd.sh`, `chain_confirm.sh`, `autocommit.sh`.
- Results: `new_directions/results/gl/{dev1,dev2,dev3,dev_cold,confirm1,confirm2,cold}/seed*_{A,B}.json`.

## Re-running
`PYTHONPATH=code/active_learning_program`, 1 torch thread per process, 4 shards; `run_nd.sh` is idempotent (restart after a container restart). Whole confirmation chain ≈ 3 h on 4 cores. Do **not** rerun the confirmation seeds with changed candidates; new variants need new seeds (e.g. 7400+, and add the block to `nd_core.py`).

## What I would do next
1. GPU tasks in `GPU_TASKS_graph_learner.md` (end-to-end Laplacian term; better encoders for the graph; prospective test with new growth images).
2. Try graph quality as the lever: the noisy-graph control suggests the gain depends on edge quality at larger λ. A graph from a better encoder is the cleanest test.
3. Check whether the learner gain survives with `vopt_u`-type selection (not tested; the learner was compared under random label sets only).
4. Gotchas: old tests rewrite `results/active_learning_v1.8_seed42/manifests` (restore with `git checkout --`); the shuffled-graph control is a low bar — always include the noisy-graph control; log-loss gets worse for λ ≥ 10.
