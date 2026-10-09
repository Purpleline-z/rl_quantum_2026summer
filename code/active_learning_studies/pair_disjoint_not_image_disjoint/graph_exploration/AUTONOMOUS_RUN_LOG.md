# Autonomous run (started 2026-10-09, user asleep, at least 8 hours): make a graph-aware acquisition rule significantly better than Random

Source of truth for this run. Update it at every phase; a later context must be able to continue from this file alone.
Branch `claude/graph-active-selection`. Language to the user: Chinese, plain, hedged, no overclaiming.

## Prompt (to myself)
Goal: find a graph-aware (image-similarity graph, type-aware, possibly GCN-based) acquisition rule that is significantly better than Random on the held-out preference endpoint, in the judgment-unit
framework (query unit = one (pair, type) judgment, budgets 10/20/40/60 judgments, splits A and B, single-shot and sequential rounds of 10). Read the literature on how others do it; do not guess.
Protocol is FIXED and must not change: endpoint, splits, head schedule (`results/pair_endpoint_study/schedule.json`), seeds, budgets, Random = mean of 5 draws, Wilcoxon over seeds + Holm.
Only selectors (new rules in `graph_strategies.py`/new modules) may change. A selector may see only: candidate judgments, revealed judgments, the model, cached features of their images, and the typed
reference (ideal) images that are already training anchors in every strategy (`exp.references`; never `exp.test_images`/`utility_images`, never validation/test group images).
Honesty rules: report failure if it happens; no tuning on confirmation seeds; every number in reports generated from cells; list all rules tried (including failures); state multiple-comparison caveats.

## Selection-bias protocol (decided before any new method was run)
- DEV seeds 400-409 (both splits, both conditions): all design iteration and tuning happens here only.
- CONFIRM seeds 410-429 plus 42, 79, 123, 202, 303 (25 seeds): used once, after freezing at most 4 candidate rules and writing them into this file. Holm over those candidates (+ their controls reported separately).
- Caveat: the first-generation graph rules (centrality, bridge, graph_core_set; report §5.14) were already evaluated on all 35 seeds, so what they showed (weak split-A hint for centrality) informs design; their confirm-seed results are not independent evidence for new rules.
- Success criterion (fixed now): a frozen candidate has Holm p < 0.05 for log-loss gain over Random in at least two of the four blocks (split x condition), including both splits, and a non-negative AUC gain. Otherwise report "not achieved" with the numbers.
- Power note: per-seed sd of the gain is 0.09-0.14 log-loss, so with 25 seeds an effect below about 0.04-0.05 is unlikely to be significant.

## To-do (status: [ ] open, [~] in progress, [x] done)
- [ ] P0 literature: how graph-based active learning and preference active learning do it (Sequential GCN details, GRAIN/FeatProp, pairwise-preference AL, low-budget AL); write findings to `notes/graph_literature_round.md`.
- [ ] P1 implement the literature graph baselines faithfully on the image graph: UncertainGCN and CoreGCN (Caramalau et al. 2021), with typed reference nodes.
- [ ] P2 type-aware graph signals: label propagation of the reference-image type posterior over the kNN graph; per-judgment features (type-t posterior gap of the two images, graph distance to revealed judgments of the same type).
- [ ] P3 combined rules (coverage x predicted-decisive x own-head uncertainty on the graph); iterate on DEV seeds, log each round below.
- [ ] P4 freeze <= 4 candidates here, then run CONFIRM seeds, aggregate, Holm.
- [ ] P5 report section, tests, commit/push (restore any manifests rewritten by old tests before committing; add only specific paths).

## Log (append)
