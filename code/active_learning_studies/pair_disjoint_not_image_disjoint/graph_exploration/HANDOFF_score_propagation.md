# Handoff — score propagation (branch `claude/score-propagation-20261014`)

## State
Development (seeds 11200–11219; `PREREGISTRATION_SP0_development_plan.md` + amendment SP0b), frozen pre-registration `PREREGISTRATION_SP1.md` (+ SP1b: confirmation 2 replicates only the headline candidate), confirmation 1 (11230–11264: standard A/B, ds shuffled control, time-blocked, cold start), confirmation 2 (11265–11299, headline only). Results: `RESULTS_score_propagation.md` (generated), `SUMMARY_ZH_score_propagation.md`, `GPU_TASKS_score_propagation.md`, literature `notes_score_propagation_literature.md`.

## Code map (`code/active_learning_studies/new_directions/`)
- `nd_propagate.py`: `PropData` (nodes + extra anchor / labelled nodes, edges, anchor index sets, test-image similarities; held-out content never becomes a node), `solve` (damped Newton, dense Cholesky, exact Hessian; separable per type column; hinges smoothed with softplus(10x)/10), `solve_lbfgs` (reference solver, tests only), `predict_d` (softmax kernel over the k nearest nodes; optional restriction to scored nodes), `fit_variant` (`fs_*`, `kn`, `ls`), `parse_name` (variant names such as `fs_lam10_rho0.001_sh1`).
- `sp_run.py`: runner (paired variants on identical label sets, per-judgment scores, anchor-far metrics, resumable). Kinds: head names, `fs/kn/ls`, `hy_*` (head + free residual), `ds_mu*_<fs tokens>` (distillation through `nd_recipe.fit_recipe(..., teacher=…)`, new `Spec.distill`), `st_*` (in-budget CV stacking), `rt_c*_<fs tokens>` / `bl_w*_…` (HTR route / blend). `Cell` caches the incumbent head per (context, label set). Splits `A`, `B`, `blocked`; cold start = `ND_INITIAL=random`.
- `graph_exploration/`: `sp_confirm.py` (pre-registered families, verdicts, interpolation-share table), `sp_offline.py` (route / blend scans from stored scores), `sp_mechanism.py` (new_directions; HTR gain by distance to labelled frames), `sp_report.py` + `results_narrative_score_propagation.md` → `RESULTS_score_propagation.md` (edit the narrative, never the generated file), `scripts_sp/{run_sp.sh,autocommit.sh}`, `SP1_spec.json`, `SP1b_spec.json`.
- `nd_core.py`: `SP_DEV_SEEDS` 11200–11219, `SP_CONFIRM1` 11230–11264, `SP_CONFIRM2` 11265–11299 (do not reuse). Tests: `code_behavior_tests/test_score_propagation.py` (8).
- Results: `new_directions/results/sp/{dev1,dev2,dev_blocked,dev_cold,conf1,conf1_ctl,conf1_blocked,conf1_cold,conf2}/seed*_{A,B,blocked}.json`.

## Pitfalls
- Feature caches are not committed: `python3 build_feature_cache.py` then `python3 nd_temporal.py build` (needs torch, torchvision, scikit-learn, scipy).
- Never start two `sp_run.py` processes on the same output directory with the same shard mapping (they rewrite whole files); extend a directory only after its previous processes have exited. Never use `pkill -f sp_run.py` from a shell whose own command line contains that string.
- A container restart kills background jobs; `run_sp.sh` is idempotent, `autocommit.sh` must be restarted.
- One held-out pair-group image is also a "bad" image in some splits; `PropData` drops typed/bad images whose content equals a held-out pair-group image (the head still trains on its bad-image features, as in every earlier study).
- A confirmation cell costs ≈ 50 s (≈ 9 Newton solves, a head fit and a distillation fit).

## What I would do next
See `GPU_TASKS_score_propagation.md`. On CPU: (i) a per-type route for the *best* free-score model chosen by cross-validation per type (HTR only here by dev evidence); (ii) free scores with a learned kernel on the head's hidden layer (graph-quality results suggest the geometry matters); (iii) prospective test on a new growth run (the free-score route relies on neighbouring labelled frames, and is null on the time-blocked split).
