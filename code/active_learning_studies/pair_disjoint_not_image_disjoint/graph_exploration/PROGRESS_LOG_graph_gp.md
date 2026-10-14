# Progress log — graph-GP study (branch claude/graph-gp-20261014, cut from claude/integration-20261013)

Seeds: development 11000–11019, confirmation 1 = 11030–11064, confirmation 2 = 11065–11099 (`nd_core.py`, lines `GP_*`, asserted disjoint from every earlier block).
Clock = container UTC.

- 02:46 start (Oct 14). Read COMMON_RULES, CONSOLIDATED_RESULTS, nd_gp.py (RBF-only MAP GP of the first new-directions study), calibration harness. Rebuilt nothing: feature caches were still on disk.
- 03:1x model written: `nd_gpg.py` — GP over image scores per type, **semi-supervised deformed kernel** k~ = k − k_x^T (I + βLK)^-1 βL k_y on the kNN graph of identity-safe nodes (exact Gaussian-process version of "RKHS penalty + β f^T L f"), Bradley–Terry / tie / not_apply likelihood as the repository's loss, typed ideal images (aw8_all anchors) as ranking constraints with the head's weights, Laplace inference in the dual (Newton), Laplace variance of f(a) − f(b), probit-corrected probabilities. Fit time 0.3–6 s on one CPU thread.
- 03:2x coarse fixed-hyperparameter pre-screen (explore1: 3 length scales x 4 β x 5 amplitudes = 60 settings, 4 seeds x 2 splits x budgets 20/60, one draw; this is hand-tuning of the "hand-set" variant and is counted as such): the best fixed settings (ℓ = 0.25 x median, β = 0.3, amplitude 1–3) were +0.015 AUC / +0.010 accuracy over `lap3_k10_st4` on those 16 cells; β = 0 (no graph) +0.012 AUC.
- 03:3x runner `gp_run.py` (per-judgment mean/variance + geometry features, resumable), dev1 launched: lap3_k10_st4, aw8_all_st2lr0.5, gp_fixed, gp_ev (Laplace evidence), gp_evc (conditional evidence: evidence of all terms minus evidence of anchors alone), gp_cv (grouped 3-fold CV).
