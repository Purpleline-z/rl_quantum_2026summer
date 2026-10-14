# Handoff — latent-type (Dawid–Skene) study (branch `claude/latent-type-1014`)

**Top: result.** Treating each image's type as a latent variable and fitting a Dawid–Skene-style confusion table by EM on the labelled judgments (graph label-spreading prior, anchors fixing memberships) did **not** improve held-out preference prediction over `lap3_k10_st4` in any pre-registered test (confirmation 1: 0 of 16 standard tests and 0 of 16 anchor-far tests with Holm p < 0.05; see `RESULTS_latent_type.md` for confirmation 2 and the blocked split). The model itself works: the learned confusion table matches the verbal outcome definitions, random or shuffled memberships clearly hurt, and using the real outcome likelihood is slightly better than the graph prior alone (AUC +0.001…+0.003 for the auxiliary-target variant) — but the effect on the head is of the order of 0.001–0.003 AUC. Reason found: the graph prior already carries the answerability information (P(not_apply) falls from 61 % to 3 % as the prior mass on the type grows), and the outcomes can only move the memberships of the ≤ 100 images that occur in labelled judgments (leave-anchors-out type recovery 76.3 % → 77.0 %).

## Code map (`new_directions/`)
- `lt_model.py` latent-class model: confusion table `T[s_a, s_b, outcome]`, pairwise factors per image pair, loopy BP (E-step), expected counts + Dirichlet pseudo-counts (M-step), controls (`prior`, `random`), label spreading with soft sources.
- `lt_learner.py` `fit_lt` = `fit_mlp_graph` + posterior uses (`pa`, `lik`, `aux`, `auxall`, thresholds); learners `lt_*`; inference-time learners `lt_inf*`; `lt_off` equals `lap3_k10_st4` exactly (tested).
- `lt_run.py` idempotent runner (JSON metrics + per-judgment CSV per seed × split; splits A, B, blocked); `lt_diag.py` leave-anchors-out diagnostic; `test_lt.py` unit tests; `lt_autocommit.sh`.
- `graph_exploration/`: `lt_tables.py` (paired gains, far stratum, Holm), `lt_report.py` + `RESULTS_latent_type.template.md` (all tables), `scripts_lt/lt_chain2.sh` (idempotent confirmation chain), `PREREGISTRATION_LT0/LT1`, `PROGRESS_LOG_latent_type.md` (every variant tried), `notes_latent_type_literature.md`.
- Seed blocks `nd_core.LT_*`: dev 10700–10719, confirmation 10730–10764 and 10765–10799 (all used; do not reuse). Spare seeds 10720–10729 were not used.

## Pitfalls
- Never run `pkill -f` with a pattern that appears in your own command line (it kills the tool shell).
- Background waits are limited to 2 h; a 12-arm confirmation (35 seeds × A/B × 4 budgets × 1 label set) takes ≈ 2 h 10 min on 4 cores.
- Pseudo-anchor variants (`pa*`) hurt in split A (AUC −0.005…−0.02): the posterior is mostly the graph prior, and hard-ish supervision from it adds the prior's errors.
- `GraphData` is cached per ctx; the shuffled-prior control is built on `graph_for(ctx, 10, True)` (anchors' source indices are kept, node identities permuted).
- The learned table is pooled across the four types; with ≤ 90 judgments a per-type table was not attempted (listed below).

## Next steps (not done)
1. Ground-truth labels for ≈ 100 judgment frames to validate the confusion table and the posteriors directly (`GPU_TASKS_latent_type.md`, item 1).
2. Per-type confusion tables and an explicit 'mixed type' state once more judgments per image exist.
3. End-to-end fine-tuning with the auxiliary target; leave-a-run-out evaluation on a new growth run.
4. Using the model for selection (expected information on memberships) — only together with more judgments per image.
