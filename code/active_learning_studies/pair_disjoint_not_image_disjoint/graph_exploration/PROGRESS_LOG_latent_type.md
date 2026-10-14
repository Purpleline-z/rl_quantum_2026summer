# Progress log — latent-type study (branch claude/latent-type-1014)
Session start 2026-10-14 ~02:47 UTC (clock of the container).
- 02:50 read COMMON_RULES, CONSOLIDATED_RESULTS, redteam handoff/scripts, nd_core, nd_graph_learner; one `lap3_k10_st4` fit ≈ 7 s (1 thread), graph build 2 s.
- 02:55 seed block `LT_*` (10700–10719 / 10730–10764 / 10765–10799) added to nd_core.py (assertions pass).
- Exploration of the data (all 521 judgments, graph prior from all typed ideal images; descriptive, not an evaluation): P(not_apply) falls from 61% (both images with prior mass on type k ≤ 0.1) to 3% (max ≥ 0.9); tie rises 6% → 26%: the label-spreading prior already carries the answerability information.
- 02:58 `lt_model.py` (latent-class EM with BP), `lt_learner.py` (pa / lik / aux), unit test: `lt_off` == `lap3_k10_st4` to 0.0. First EM on seed 10700 learns tables consistent with the outcome definitions (both of type k: tie 71%, not_apply 3%; exactly one: typed image wins 89%; neither: not_apply 60%).
- 03:00 diagnostic D2 on seed 10700 A (25% of anchors hidden): prior acc 0.83; prior + outcome posteriors 0.83 (no change); shuffled prior 0.24.
- 03:00 D1 launched (`results/lt/dev1`, 9 learners, seeds 10700–10719, A,B, 1 draw).
