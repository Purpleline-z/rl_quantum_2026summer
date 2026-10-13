# Progress log — calibration study (branch claude/calibration-20261013)

Seeds: development 9100–9119, confirmation 9150–9184 (`nd_core.py`, lines `CAL_*`; asserted disjoint from every earlier block and from the reserved 8000–8199, 8400–8799, 8885–8999).
Wall-clock notes use the container clock (UTC).

- 15:55 start. Read COMMON_RULES, SH1 pre-registration, handoff, RESULTS (selection-htr, graph-learner), code; read the parallel branches' summaries (graph-mechanism: RT13-off Laplacian candidate, dose-response in graph quality; graph-endpoint: classification endpoint, harsh splits; red-team RESULTS_redteam.md does not exist yet, rt_*.py code read).
- 16:05 environment: torch/torchvision/scikit-learn installed, both feature caches rebuilt (444 and 1278 images).
- 16:07 stage-1 harness `nd_calibration.py` + `cal_run.py` (candidate head identical to the registry learner `lap3_k10_st4`, max |d diff| = 0; 5 cross-fitting heads by pair group; 4 extra full-data heads; label smoothing 0.1 / 0.25), dev run `results/cal/dev1` started (20 seeds x 2 splits x 4 budgets x 1 draw, 4 shards, ~210 s per cell).
- 16:42 dev1 finished (20 seeds x 2 splits x 4 budgets x draw 0; K = 5 cross-fitting heads, 4 extra heads, label smoothing 0.1 / 0.25). First reading (leave-one-seed-out priors): a global temperature is unreliable (log-loss +0.008 in A, -0.022 in B); per-type temperatures help (within-run ridge-to-1: +0.018 / +0.013; development-prior-centred: +0.035 / +0.034 log-loss, AUC +0.008 / +0.012); the cross-fit ensemble of the K fold heads + final head adds +0.004 / +0.009 log-loss and +0.0005 / +0.0047 AUC; label smoothing hurts; Davidson tie model, far-from-anchor shrinkage and distance-weighted temperatures do not help. The raw head is UNDER-confident on RT13 / HTR / c(6x2) (test-optimal T = 0.3 / 0.7 / 0.6) and slightly over-confident on (1x1) (T about 1.1-1.4).
- 16:43 dev2 started (same seeds, draw 1, K = 5 only) for more stable development priors.
