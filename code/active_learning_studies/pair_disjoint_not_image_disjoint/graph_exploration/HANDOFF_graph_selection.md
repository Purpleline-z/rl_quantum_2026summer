# Handoff for the next Claude — graph selection under the tuned learner (Prompt 2)
Branch `claude/graph-selection-htr-1012` (from `claude/integration`). Read first: `RESULTS_graph_selection.md` (generated; the bottom line is section 1), `SUMMARY_ZH_graph_selection.md`, `PREREGISTRATION_S0/S1/S2/S3.md` (S1 has Amendment 1), `LOG_graph_selection.md`, `notes_graph_selection_literature.md`. Common rules: `prompts/COMMON_RULES.md`.

## State
- Pre-registered success criterion (A) (accuracy AND AUC Holm-significant in both splits for a graph rule over Random under `aw8_all`): **not met in S1 and S2, met in S3** (`qcov_alloc` regular start, `tdc` cold start; 40 fresh seeds per block) — 2 of 12 rule x block checks. Small gains of the typed-coverage rules (`tdc`, `qcov_alloc`): +0.005 to +0.012 in accuracy and AUC pooled over 105 seeds per start mode (descriptive). Graph-specific part (vs shuffled posterior): AUC difference positive in 11/12 block x split cells, p<0.05 in 4/12.
- Criterion (B) (HTR): met by `gvopt_lap3@aw`, `qcov_alloc` (and `vopt_u@aw` except its other-type side condition) — but non-graph controls gain as much or more, so it is a variance-reduction-design effect.
- Graph-specific effect that is solid: the type posterior lets typed coverage pick ~65% decisive judgments (Random 50%; shuffled/type-only below Random). It does not turn into a large endpoint gain: the learning curve under `aw8_all` is almost flat and an oracle decisive selector is no better (dev diagnostic).

## Code map (all in `code/active_learning_studies/new_directions/`)
- `gsel_select.py` — harness (single-shot selection; `REGISTRY` of 38 selector names; `@aw` = selection model fitted with `aw8_all`; `*_shuffled` / `*_typeonly` controls; `ORACLE_*` = diagnostic reading hidden outcomes, never a method). `gsel_seq.py` — sequential extension (rounds of 10). `gsel_analyze.py` — gains, Wilcoxon, Holm, composition. `gsel_report.py` — all generated tables (`--out TABLES_graph_selection.md`, `--template ... --doc ...` builds the results file). Tests: `code/active_learning_program/code_behavior_tests/test_graph_selection.py` (24 tests; run with `OMP_NUM_THREADS=1 python3 -m pytest ...`).
- Idempotent run scripts: `gsel_dev1..7.sh`, `gsel_conf.sh` (S1), `gsel_conf_s2.sh`, `gsel_conf_s3.sh`; `gsel_autocommit.sh` commits result cells while jobs run (the stop hook wants a clean tree). Container restarts kill background jobs: check `uptime`, `ps`, then relaunch the script (finished cells are skipped; `.done` files in `logs/` are not committed).
- Seeds 8000–8479 are registered as used in `nd_core.py` (dev 8000–8019; S1 8100–8134, 8200–8234, 8300–8334; S2 8340–8399; S3 8400–8479). Seeds 8335–8339 and 8480–8499 are unused.

## Things I would do next
1. The GPU/lab list in `GPU_TASKS_graph_selection.md` (end-to-end fine-tuning; prospective decisive-enrichment test).
2. If the group wants a graph claim: it needs a control that removes only the posterior while keeping everything else; `tdc` vs `tdc_shuffled` is the clean one (support is weak; see the specificity table).
3. Do not spend more CPU on selection variants under `aw8_all`: headroom is ~0.01 AUC. Learner-side ideas (Prompt 1 branch) have more room.

## Pitfalls
- `ImageGraph(shuffle=True)` only permutes the type posterior q; the kNN adjacency (propagated pair space) stays. `gvopt_*_shuffled` permutes the adjacency.
- `pick_decisive` in the result JSONs reads candidate outcomes (analysis only).
- Random rows in a block are identical across single-shot runs of the same seed (same draws, same learner seed), so Random is shared between arms of one run but not across blocks.
- Per-seed HTR test sets have 13–16 decisive judgments; never read a single seed's HTR number.
