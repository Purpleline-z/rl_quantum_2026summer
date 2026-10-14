# SP0 — development plan of the score-propagation study (written 2026-10-14 ≈ 03:15, after the first three-seed smoke test and before any development result was read)

Seeds: development 11200–11219; confirmation 1 = 11230–11264, confirmation 2 = 11265–11299 (`nd_core.py`: `SP_*`). Reference = `lap3_k10_st4` (unchanged); tuned graph-free control `aw8_all_st2lr0.5`. Same protocol as all earlier studies (splits A/B, budgets 10/20/40/60, identical random label sets, held-out decisive preference judgments), plus the anchor-far stratum (both test images > 0.03 from every typed anchor), the time-blocked split (`rt_blocked.py`) and cold start (`ND_INITIAL=random`) as secondary evaluations.

## Models (all in `new_directions/nd_propagate.py`, runner `sp_run.py`)
- **Free scores (`fs_*`)**: one free score vector per graph node (+ extra nodes for anchors / labelled images that are not graph nodes), L-BFGS on [Bradley–Terry terms of the labelled judgments + anchor ranking terms (aw) + λ·graph Laplacian over all nodes + ρ·ridge]; test images (never nodes) get the kernel-weighted average of the scores of their `kt` nearest nodes (softmax(sim/τ)). No head.
- **Baselines**: `ls` (label spreading of the anchors' class one-hots, no judgment used), `kn` (free scores without Laplacian, kernel over scored nodes only = nearest neighbour in score space), `fs_lam0` (no penalty = memorisation), `fs_*_sh1` (shuffled graph).
- **Hybrids**: (a) `hy_*` head `lap3_k10_st4` + free residual, penalties on the residual; (b) `ds_mu*_<teacher>` the head with a pair-difference distillation (weight μ) of the free-score teacher on unlabelled nodes/edges; (c) stacking weight chosen by in-budget cross-validation (implemented only if (a)/(b) leave room).

## Development screens (all on seeds 11200–11219, tier 1 = 11200–11209 × splits A, B × 4 budgets × 2 label sets; extension to 20 seeds for the shortlisted)
1. Free-score hyper-parameters one at a time around λ = 30, aw = 8, ρ = 0.01, kt = 5, τ = 0.05 (λ, aw, ρ, kernel, extra anchor nodes) + the baselines and the shuffled control.
2. Hybrid (a) residual strength grid (λr, ρr, penalty on R or on S) and hybrid (b) μ grid (0.03 … 0.3) with the best free model as teacher.
3. Hybrid (c) stacking if time allows.
4. Secondary evaluations of the leading variants on the development seeds: anchor strata (from stored scores), blocked split, cold start.
Every variant that is run is listed in the results (generated tables; winner's-curse accounting).

## Shortlisting and freeze (rule fixed now)
- Ranking score of a variant = mean of its four gains over `lap3_k10_st4` (AUC and accuracy, splits A and B) on the development seeds. A pure non-parametric model is expected to be *worse*; it is still a candidate if it is the best of its kind (the question "how much of the head's gain is interpolation" needs its confirmed difference, positive or negative).
- Freeze ≤ 4 candidates before any confirmation seed: the best free-score model, the best residual hybrid, the best distilled hybrid and one of {stacking, second-best of the above}; controls run alongside: shuffled-graph version of each graph candidate, `fs_lam0`, `kn`, `ls`, `aw8_all_st2lr0.5`. PREREGISTRATION_SP1 will fix test family, margins (non-inferiority 0.003 AUC / 0.01 accuracy), Holm family and success rule; confirmation 1 on 11230–11264 (35 seeds, A and B, 3 label sets per budget), confirmation 2 on 11265–11299 only if time allows; each run once.
