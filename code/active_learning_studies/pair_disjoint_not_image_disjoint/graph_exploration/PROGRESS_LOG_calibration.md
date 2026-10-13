# Progress log — calibration study (branch claude/calibration-20261013)

Seeds: development 9100–9119, confirmation 9150–9184 (`nd_core.py`, lines `CAL_*`; asserted disjoint from every earlier block and from the reserved 8000–8199, 8400–8799, 8885–8999).
Wall-clock notes use the container clock (UTC).

- 15:55 start. Read COMMON_RULES, SH1 pre-registration, handoff, RESULTS (selection-htr, graph-learner), code; read the parallel branches' summaries (graph-mechanism: RT13-off Laplacian candidate, dose-response in graph quality; graph-endpoint: classification endpoint, harsh splits; red-team RESULTS_redteam.md does not exist yet, rt_*.py code read).
- 16:05 environment: torch/torchvision/scikit-learn installed, both feature caches rebuilt (444 and 1278 images).
- 16:07 stage-1 harness `nd_calibration.py` + `cal_run.py` (candidate head identical to the registry learner `lap3_k10_st4`, max |d diff| = 0; 5 cross-fitting heads by pair group; 4 extra full-data heads; label smoothing 0.1 / 0.25), dev run `results/cal/dev1` started (20 seeds x 2 splits x 4 budgets x 1 draw, 4 shards, ~210 s per cell).
