# Handoff for the next Claude — anchor-exchange study (branch `claude/anchor-exchange-1013`)
Read first: `RESULTS_anchor_exchange.md` (generated; section 1 = answers), `SUMMARY_ZH_anchor_exchange.md`, `PREREGISTRATION_E0.md` / `E1.md`, `PROGRESS_LOG_anchor_exchange.md`, `notes_anchor_literature.md`, `GPU_TASKS_anchor_exchange.md`. Common rules: `prompts/COMMON_RULES.md`.

## State
Finished: literature notes, development screens (active rules, dose-response, lambda interaction, class-balance exploration), frozen pre-registration E1 (before any seed >= 8750), confirmation 1 (8750–8784: active rules, dose-response/exchange rate), generated results, tests. One line each: (a) active anchor rules at 36 anchors — not confirmed (none beats random anchors on the preference endpoint; diversity rules help the external type accuracy); (b) exchange rate — first 12 typed anchors ≈ 23 extra judgments (95% 14–35) ≈ 2 judgments per label at 10 extra judgments, 12→29 ≥ tens of judgments (extrapolated), saturation beyond ~30 anchors, HTR gains nothing in accuracy/AUC.

## Code map (`code/active_learning_studies/new_directions/`)
- `ae_core.py`: anchor universes (`no_test` = 116 non-test ideal images, `all144` = original), `fit_head` (= `nd_graph_learner.fit_mlp_graph` with an explicit anchor set; all 144 selected reproduces `lap3_k10_st4` exactly, unit-tested), external type-accuracy endpoint (selected anchors = reference set), active rules (`kcg`, `kcf`, `typi`, `amb`, `htrq`, `kcf_bal`, diagnostic `strat`; `*_shuf` use the permuted graph; none looks at unselected labels).
- `ae_run.py` (modes dose / active / lam; idempotent per seed x split file), `ae_analyze.py` (gains, Holm, dose tables, exchange-rate estimator `exchange_table`: log-axis model per anchor level + seed bootstrap), `ae_report.py` (all tables and the verdict; `--template ... --doc ...` builds the results file from `RESULTS_anchor_exchange.template.md` + `answers_anchor_exchange.md`), scripts `ae_dev.sh`, `ae_dev2.sh`, `ae_conf.sh`, `ae_autocommit.sh`.
- Tests: `code_behavior_tests/test_anchor_exchange.py`. Seeds: own block in `nd_core.py` (dev 8700–8719, confirmation 8750–8784; assertions against used and reserved ranges).
- Results: `new_directions/results/ae/{dev_active,dev_dose,dev_lam,dev_active2,conf_active,conf_dose}/seed*_{A,B}.json`.

## Pitfalls
- `core.anchors(..., all_ideal=True)` (the original `aw8_all`) includes the 28 OUTER-TEST ideal images as anchors: invalid for the external type-accuracy endpoint; this study uses the 116-image universe.
- Classes absent from a small random anchor set simply have no anchors (external accuracy cannot predict them; the anchor loss drops them): this is part of the dose-response, not a bug.
- The exchange-rate estimator extrapolates the log budget axis; intervals with upper end `inf` mean the bootstrap could not bound them; always read the 'beyond range' column.
- `lap3_k10_st4` fits take about 6 s; the confirmation chain takes about 3.5 h on 4 cores; container restarts kill jobs (scripts are idempotent).
- Old tests rewrite `results/active_learning_v1.8_seed42/manifests` (`git checkout --` before committing).

## What I would do next
GPU/lab list in `GPU_TASKS_anchor_exchange.md`: cost per label type, the prospective trajectory-frame experiment, end-to-end dose-response. Do not spend more CPU on anchor-selection rules for the preference endpoint: random anchors were as good as every rule in 14 + 35 seeds.
