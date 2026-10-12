# new_directions: overnight exploration of non-graph method directions

Start with `NEW_DIRECTIONS_RESULTS.md` (results, verdicts, caveats). `OVERNIGHT_LOG.md` is the chronological log. Pre-registrations: `PREREGISTRATION_1.md` (anchor-weighted learning, seeds 3000-3034), `PREREGISTRATION_2.md` (type accuracy), `PREREGISTRATION_3.md` (cold start, seeds 4000-4034), each with its spec / code.

| file | role |
|---|---|
| `nd_core.py` | harness: contexts of the judgment-unit study, generic head fit (same loss as `frozen_encoder_reward_head.fit_head`), held-out metrics, random label sets, seed sets |
| `nd_learners.py` | learner variants (anchor weights, all-ideal anchors, mirror-symmetric head, metadata, pseudo-labels, ties, Rao-Kupper, mixture anchors, cross-entropy anchors, ensembles, capacity) |
| `nd_bayes.py`, `nd_gp.py`, `nd_temporal.py` | prior-centred Bayes head (independent / type-coupled), GP preference learner, time-contrastive adapter |
| `nd_run.py` | learner screens on identical random label sets (`--learners ... --seeds ... --out`) |
| `nd_select.py` | selection screens (`random`, `vopt_u`, ambiguity-weighted, mirror consistency, GP variance, raw-space and stochastic design, committee) x learners; cold start with `ND_INITIAL=random` |
| `nd_analyze.py`, `nd_analyze_select.py`, `nd_make_report_tables.py` | paired gains with bootstrap intervals; `results/DEV_SCREENS.md` is generated |
| `nd_confirm.py`, `nd_confirm_markdown.py`, `nd_confirm_type.py`, `nd_type_accuracy.py` | pre-registered tests and their generated tables |
| `literature/` | search-verified literature notes (`lit_*.md`) and `LITERATURE_TABLE.md` |
| `results/` | all result JSON / CSV / tables (screen* = development seeds 2000-2019; confirm* = confirmatory seeds) |

Tests: `code/active_learning_program/code_behavior_tests/test_new_directions.py`. Environment: `PYTHONPATH=../../active_learning_program OMP_NUM_THREADS=1`; the SimCLR feature cache must exist (`build_feature_cache.py`), the all-image cache is built by `python3 nd_temporal.py build`, the mirror cache by `python3 nd_build_mirror.py`.
