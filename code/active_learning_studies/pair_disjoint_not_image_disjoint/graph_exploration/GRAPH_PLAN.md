# Graph idea for active selection: prompt and to-do list (branch `claude/graph-active-selection`)

Branched from `claude/judgment-unit-rerun` (which contains all of `claude/frozen-encoder-strategies`). Read `SESSION_CONTEXT_FOR_NEXT_CLAUDE.md` at the repository root first.
The user (PhD student, writes Chinese, wants plain language, hedged and honest reporting, no unexplained jargon) wants to know whether a graph over images can improve
which (pair, type) judgments to label next. Nothing here is merged to `main`; no PR.

## Prompt (for the next Claude working on this branch)
Goal: test whether graph information improves acquisition for the Bradley–Terry reward head on the held-out preference endpoint (log-loss, AUC on held-out pair groups),
using the same splits (A and B), seeds (42, 79, 123, 202, 303, 400–429), budgets (10/20/40/60), conditions (single-shot, sequential) and Random baselines as the judgment-unit
study (`judgment_unit_study.py`, `judgment_unit_strategies.py`). Do not change the endpoint, the splits or the head schedule. Add graph strategies as new selectors only.
Report findings separately from hypotheses; use per-seed gain over Random, Wilcoxon + Holm and Friedman exactly as `aggregate_pair_endpoint.py` does; state every multiple-comparison caveat.

## What was established before this branch (see `graph_exploration/*.py`, all reproducible with plain numpy/scipy/sklearn)
- The comparison graph is nearly a matching: 168 pair groups (edges), 284 images, 117 components (largest 11 nodes); per reconstruction type the decisive directed graph has
  components of at most 4–5 nodes and 0–6 images that both won and lost. A GNNRank-style ranking GNN on comparison edges alone has nothing to recover. (The slide's "669 pairs, ~300 images" does not match the data: 638 rows, 168 groups, 284 images.)
- The feasible graph is over image similarity: nodes = 1278 images (1124 trajectory + 154 ideal), edges = kNN in SimCLR space (+ optional within-session temporal chain, which
  `data/temporal_constraints.json` marks as tentative, so keep it behind a flag), comparison edges as an extra layer. Labelled pairs are not near each other in feature space
  (median partner rank 518 of 1277, random 626), so comparison edges bridge kNN regions (algebraic connectivity 0.003 -> 0.067 with k=10 + temporal).
- Boundary structure: among ideal images, 112 of 770 five-nearest-neighbour edges cross types; HTR–RT13 is 52 of them.
- Quick signal test (`graph_signal_test.py`, linear BT head, 5-fold by pair group x 20 repeats, 264 decisive rows): raw features AUC 0.941; SGC-smoothed features 0.938–0.940; spectral embedding 0.909. No gain; the endpoint is near its ceiling. This does not test a trained GNN or any acquisition rule.
- Offline benchmark limit: only the 168 labelled pair groups can be selected; the extra nodes can only be context. ~100 new trajectories from Yao would add nodes, not labels.

## To-do
Status key: [ ] open, [x] done.
- [x] Feasibility diagnostics (structure, boundary, signal test) committed in `graph_exploration/`.
- [ ] Rebuild the feature cache (`build_feature_cache.py`; `simclr_feature_cache.pt` is not in the repository, the SimCLR checkpoint is) and run `pytest` on the existing tests to get a baseline (56 pass in the notes).
- [ ] Build `image_graph.py`: label-free graph over all images (kNN k in {5,10}, optional temporal chain flag), cached on disk, with tests (symmetry, no test/validation label information used; images of held-out groups may appear as unlabeled nodes only if that is label-free, but check the identity-safety assertion in `judgment_unit_study.Context`: the held-out images must not be reachable from the labelled set or candidate pool — decide whether graph context may include them and document it; the safe default is to build the graph from pool, labelled and ideal images only).
- [ ] Step 1 strategies in a new `graph_strategies.py` (row-level selectors with the existing signature `select(cands, labeled, model, cache, budget, seed)`):
  - `graph_centrality_uncertainty`: own-head uncertainty x PageRank/degree centrality of the pair's images in the kNN graph.
  - `graph_boundary_uncertainty`: uncertainty x boundary score (share of an image's kNN that belong to other types, using ideal images as typed anchors).
  - controls: the same with a shuffled graph (graph carries information?), and uncertainty alone (already `uncertainty`).
- [ ] Step 2: replace the mean-pair embedding of core-set/DPP/facility location by SGC-propagated embeddings (`kind="graph"` in `pair_features`), keep |a-b|.
- [ ] Register the new names in `judgment_unit_strategies.py` (`make_selector`, family list), run single-shot + sequential for both splits, aggregate with Holm/Friedman, per-budget tables.
- [ ] Only if steps 1–2 show a signal that survives both splits: Step 3, a trained 2-layer GCN encoder with per-type BT heads (watch over-fitting with 168 groups; early stopping on validation only).
- [ ] Write a short report section (separate findings from hypotheses; mention the 669-vs-168 discrepancy and the ceiling of the endpoint) and update `notes/literature_and_new_strategy_ideas.md` with the graph sources below.
- [ ] Open question for the user/prof: whether the new trajectories from Yao should be added as unlabeled graph nodes.

## Literature (checked via search results only where stated)
- Sequential GCN for Active Learning, Caramalau et al., CVPR 2021, https://arxiv.org/abs/2006.10219 (similarity graph over the pool, GCN separates labelled/unlabelled, CoreSet/uncertainty on its embeddings).
- GNNRank, He et al., ICML 2022, https://arxiv.org/abs/2202.00211 (global ranking from a directed comparison graph; needs dense comparisons).
- HodgeRank with Information Maximization, Xu et al., AAAI 2018, https://arxiv.org/abs/1711.05957; https://arxiv.org/pdf/1503.00164 (active sampling that maximises algebraic connectivity). Jiang–Lim–Yao–Ye 2011 cited from memory, verify.
- FeatProp https://arxiv.org/abs/1910.07567, GRAIN https://arxiv.org/html/2108.00219v1, Graph Policy Network https://arxiv.org/html/2006.13463 (node selection for GNNs; AGE and ANRMAB only seen in search summaries).
- Already in the notes: ASAP, Just Sort It!, Active Learning with Label Comparisons (2204.04670), Long et al. 2008 label propagation.
- No paper found that combines a graph-regularised Bradley–Terry model, a kNN graph and active choice of comparisons.
