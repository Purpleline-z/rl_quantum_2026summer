# Progress log — graph-mechanism branch (claude/graph-mechanism-20261013)

## 2026-10-13 03:00-03:40 UTC: setup, literature, round 1
- Fetched `claude/graph-learner-20261012`, branch `claude/graph-mechanism-20261013`. Rebuilt SimCLR caches (`build_feature_cache.py`, `nd_temporal.py build`) and the intermediate-layer features (`nd_gm_layers.py build`, layer-4 pooled == cached final feature, max abs diff 0).
- Own seed blocks in `nd_core.py` (GM_DEV 8000-8019, GM_CONFIRM1 8100-8134, GM_CONFIRM2 8150-8184, assertions; added to the used set). Literature notes written (`notes_graph_mechanism_literature.md`, all "(摘要)").
- `nd_graph_mechanism.py`: constructions (kNN, mutual, eps, SNN, Gaussian, local scale, b-matching, PCA-whitened, layers 2/3/3+4), noise dose, normalised Laplacian, edge placement, per-type weight, annealing, non-graph controls (noise-consistency, mixup consistency, weight-decay sweep); `gm_lap3_n0` and `gm_lap3_n3` reproduce `lap3_k10` / `lap3_k10_n3` exactly (max abs diff 0.0; unit tests, 15 tests).
- Round 1 (dev1, dose-response, 20 seeds x A,B): see RESULTS. Summary: at lambda=10 the AUC gain falls monotonically with noise (A: +0.020 at sigma 0, -0.020 at sigma 3; B: +0.021 -> -0.010), per-seed Spearman(gain, quality) +0.81 (A), +0.58 (B), p<0.001; at lambda=3 weaker (A +0.37 p 0.03; B +0.08 n.s.). Quality = share of same-class edges among ideal-ideal edges.

## 2026-10-13 03:40-06:20 UTC: development rounds 2-7 (seeds 8000-8019), freeze, confirmation 1 started
- Round 2 (11 constructions at lambda 10): none beats `lap3_k10` by more than +0.003 AUC; PCA-whitened (-0.025), eps graph (-0.018), Gaussian weights (-0.01) are worse. Across the 12 graphs the per-seed Spearman(quality, AUC gain) is +0.32 (A, p 0.0003) and +0.18 (B, p 0.007); not for accuracy.
- Round 3: non-graph smoothing does not reproduce the gain (input-noise consistency +0.002 AUC, weight decay 0, mixup consistency -0.13..-0.18); edge placement: unlabelled-unlabelled edges only or ideal-trajectory edges only give about a third of the full-graph gain; HTR-only smoothing gives the HTR accuracy gain, no-HTR smoothing gives the 1x1 gain; annealing no better.
- Round 4: lambda 5..40 and k 5..30 plateau at +0.017..+0.021 AUC; k=10 is best; larger lambda on lscale/mutual/SNN no better.
- Round 5 (per-type switches): Laplacian off for RT13 (`gm10_type_no_rt13`) is +0.009 AUC over `lap3_k10` in both splits (dev, uncorrected) -> candidate C1.
- Round 6: label-aware LDA topology: weight 1 equal, weight 3 harmful (-0.03 AUC) although its (circular) quality is 0.99.
- Round 7 (own idea: skip near-duplicate neighbours): first run INVALID (graph cache key lacked r0/frac, so ring6/drop40 reused ring3/drop20); fixed and rerun; no gain beyond +0.005. The invalid cells are kept in `results/gm/dev7_INVALID_cache_key_collision` and the two wrong quality entries were removed. No other variant shares a cache key with a different graph (checked).
- 49 variants in total over 7 rounds (plan said 4 rounds of <=12; extended after seeing results; stated in PREREGISTRATION_GM1).
- 05:14 UTC: `PREREGISTRATION_GM1.md` committed (0f33c6f65) before any confirmation seed; confirmation 1 (8100-8134) then confirmation 2 (8150-8184) run by `scripts_gm/chain_confirm.sh`.
