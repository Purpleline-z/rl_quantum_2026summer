# Progress log — graph-mechanism branch (claude/graph-mechanism-20261013)

## 2026-10-13 03:00-03:40 UTC: setup, literature, round 1
- Fetched `claude/graph-learner-20261012`, branch `claude/graph-mechanism-20261013`. Rebuilt SimCLR caches (`build_feature_cache.py`, `nd_temporal.py build`) and the intermediate-layer features (`nd_gm_layers.py build`, layer-4 pooled == cached final feature, max abs diff 0).
- Own seed blocks in `nd_core.py` (GM_DEV 8000-8019, GM_CONFIRM1 8100-8134, GM_CONFIRM2 8150-8184, assertions; added to the used set). Literature notes written (`notes_graph_mechanism_literature.md`, all "(摘要)").
- `nd_graph_mechanism.py`: constructions (kNN, mutual, eps, SNN, Gaussian, local scale, b-matching, PCA-whitened, layers 2/3/3+4), noise dose, normalised Laplacian, edge placement, per-type weight, annealing, non-graph controls (noise-consistency, mixup consistency, weight-decay sweep); `gm_lap3_n0` and `gm_lap3_n3` reproduce `lap3_k10` / `lap3_k10_n3` exactly (max abs diff 0.0; unit tests, 15 tests).
- Round 1 (dev1, dose-response, 20 seeds x A,B): see RESULTS. Summary: at lambda=10 the AUC gain falls monotonically with noise (A: +0.020 at sigma 0, -0.020 at sigma 3; B: +0.021 -> -0.010), per-seed Spearman(gain, quality) +0.81 (A), +0.58 (B), p<0.001; at lambda=3 weaker (A +0.37 p 0.03; B +0.08 n.s.). Quality = share of same-class edges among ideal-ideal edges.
