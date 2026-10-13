# Progress log — branch `claude/selection-htr-20261013` (UTC, 2026-10-13)

- 03:02 start (container restarted; rebuilt nothing — feature caches persisted). Read COMMON_RULES, nd_select, graph_vopt/typed code, handoffs. Seed blocks 8200–8399 added to `nd_core.py`.
- 03:05 sdev1 (earlier selectors × {aw8_all, lap3_k10}); literature notes by a search subagent; new selectors `nd_select_graph.py`; tests; PREREGISTRATION_SH0.
- 03:50–04:30 sdev2 (new graph/HTR selectors), sdev3 (per-type λ, steps, lr), sdev4 (HTR anchor weight: dropped).
- 05:26 sdev5: **longer head schedule helps lap3_k10, not aw8_all** → sdev6 (steps ×6/×8, λ, lr, controls; stopped early for time, partial cells).
- 06:19 froze SH1 (commit 232d41d11) and started confirmation 1; 06:20 found/fixed a missing selector registration (chain had skipped c1_sel); chain restarted.
- 07:08 c1_learn done (learner family confirmed); 07:27 c1_sel done (selection not confirmed); 07:42 c2_sel; 08:32 c2_learn (confirmed); 08:30–09:27 c1_sel4 (secondary, selection under the long schedule: not confirmed).
- 07:30 SH2 addendum (cold-start learner confirmation, seeds 8385–8399) committed before it ran; 09:49 done (confirmed).
- 09:50–10:10 sdev9 development check of a graph-free regulariser at the long schedule (`cons0.3_st4`): reaches most of the gain → recorded as a caveat in the results.
- 10:11 final tables generated (`sh_report.py`), docs written (RESULTS, SUMMARY_ZH, HANDOFF, GPU_TASKS).
