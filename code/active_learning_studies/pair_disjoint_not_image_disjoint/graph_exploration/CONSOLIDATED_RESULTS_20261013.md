# Consolidated results of the parallel sessions (integration branch `claude/integration-20261013`, 2026-10-13)

Merged branches: redteam (⊃ selection-htr ⊃ graph-learner), graph-mechanism, graph-endpoint, paper-package, representation, regularisers, recipe, anchor-exchange, temporal-graph, calibration. Merge conflicts were only in `nd_core.py` seed blocks (both sides kept); the merged code imports (178 learners, 84 selectors registered) and the repository's behaviour tests pass (380 passed, 27 skipped). Source of every statement: the one-page summaries and result documents named below; this note does not re-derive them. All numbers are on the same 168 labelled pair groups (re-split by seed), frozen SimCLR features, MLP head; p-values optimistic; "confirmed" = pre-registered on previously unused seeds, run once.

## 1. The learner ladder (confirmed)
default head → anchor weight 0.25→8 (`aw8`) → all ideal images as anchors (`aw8_all`) → + Laplacian regulariser on the cosine-kNN graph of unlabelled images (`lap3_k10`, λ=3, k=10) → + 4× head training steps (`lap3_k10_st4`). Each step confirmed on fresh seeds; the whole recipe ≈ +0.02–0.03 AUC and ≈ +0.02–0.03 accuracy over `aw8_all`, and ≈ +0.016–0.027 AUC over the best graph-free schedule tuned (`aw8_all_st2lr0.5`). The effect replicates in cold start. (selection-htr, graph-learner, paper-package `claims_table.md`)

## 2. What was tried on top of it and did not help (confirmed negatives)
| Direction (branch) | Result |
|---|---|
| Selection rules (selection-htr) | 4 rules, 2 confirmations: none beats random on AUC/accuracy under `lap3_k10` / `lap3_k10_st4`; HTR-directed selection only improves HTR log-loss (+0.02) |
| Head recipe search (recipe) | 4 frozen candidates (cosine decay, per-type lr, anchor weight 16, decay+wd): all null; plateau within ±0.005; dropout 0.3 during training is harmful |
| Representation (representation) | SimCLR layer 3, ImageNet ResNet-18, ImageNet⊕SimCLR, view averaging: none improves (15 of 16 differences negative); ImageNet-containing candidates worse on accuracy; HuggingFace/torch hub unreachable (403) so no DINOv2/CLIP |
| Temporal edges (temporal-graph) | 3 candidates null (Holm min p 0.79); 84–94 % of adjacent-frame edges are already kNN edges; temperature is not monotone in time |
| Active anchor selection (anchor-exchange) | HTR quota, k-center (feature/graph), TypiClust at 36 anchors: none beats random anchors |
| Graph-free regularisers (regularisers) | graph-free consistency gets ≈ 70–90 % of the AUC gain but not the accuracy gain; no graph-free candidate non-inferior; "Laplacian + consistency" beats each graph-free candidate by +0.003–0.005 AUC (graph adds a small but real share) |
| Graph builders (graph-mechanism) | ~60 variants (mutual-kNN, ε, SNN, kernels, PCA-whitened, other layers, b-matching, λ/k grids): none beats cosine kNN k=10 by more than +0.003 AUC |

## 3. What was confirmed beyond the ladder
- **Graph quality matters** (graph-mechanism, two confirmations): gain falls with noise added to the topology (per-seed Spearman +0.63…+0.76 with graph quality at λ=10); a few random long-range wrong edges destroy the gain, local rewiring is harmless; no graph-free smoothing reproduces it.
- **RT13 off the graph penalty** (graph-mechanism, two confirmations): `gmc_t_no_rt13` (λ=10 on the other three types) beats `lap3_k10` by +0.009–0.010 AUC. Not yet compared against `lap3_k10_st4`.
- **Calibration** (calibration, pre-registered): per-type temperature (constants fitted on development seeds, or cross-fitted in-budget) lowers log-loss by 0.015–0.051 and ECE by 0.012–0.024 with AUC +0.003–0.013 and no loss in accuracy; the model is *under*-confident on RT13 (best temperature ≈ 0.3).
- **Exchange rate typed anchors vs pair judgments** (anchor-exchange, estimate only): the first ~12–29 typed anchors are each worth ≈ 2 pairwise judgments, saturating beyond ~30; HTR accuracy gets nothing from anchors; Laplacian and anchors are complementary.
- **Harsh splits** (graph-endpoint): holding out a whole run or the top temperature tercile keeps the gain (AUC ≈ +0.02) — fixed small test sets (50 / 26 judgments).
- **Paper endpoint** (graph-endpoint): on the 28 outer-test ideal images no graph method beats 1-NN or plain logistic regression in a way attributable to the graph.
- **More unlabelled images**: formal trend passes but all of it is the step 0 → 10 % of trajectory frames; nothing consistent from 10 % to 100 %.

## 4. What the red team (redteam) established
No implementation error (independent re-implementation matches to 2e-15); no path-level leak (one content-twin image in 3/10 audited cells). The confirmed learner gain is robust under graph de-duplication, time-blocked split and anchor-strict evaluation, but it is carried by judgments adjacent to typed anchors: on anchor-far judgments AUC gain ≈ 0 (accuracy +0.01–0.02, not Holm-significant). The anchor-weight-8 gain holds on anchor-far judgments when all anchors are present but vanishes when near-copy anchors are removed. Replicated independently by regularisers, recipe and temporal-graph branches (far-from-anchor gains small or absent; recipe's far stratum ≈ 15 % of judgments). The data are 284 frames of three runs; test images are near-duplicates of training frames.

## 5. Where the work stands (author's synthesis)
1. On this dataset and frozen features the head-side recipe is at a plateau: confirmed improvements are exhausted by the ladder, calibration and (probably) RT13-off; further knobs, representations, edges, selection rules give zero.
2. The scientific gap is **generalisation away from typed anchors / to new runs**: nothing so far improves anchor-far AUC, and nothing can test new runs without new data.
3. Untested combinations of confirmed parts: RT13-off × 4× steps × calibration × (Laplacian + consistency); each part was confirmed against a different baseline.
4. Levers that need resources not in the container: new labelled growth runs (prospective test), GPU (end-to-end fine-tuning, larger encoders; HuggingFace blocked).
5. Paper-package (`PAPER_PACKAGE/`, `make all`) holds the claims table and figures up to the red-team stage; later branches (representation … calibration) are not yet in its tables.

## 6. Branch map (documents to read)
| Branch | Read first |
|---|---|
| graph-learner | `RESULTS_graph_learner.md` |
| selection-htr | `RESULTS_selection_htr.md`, `HANDOFF_selection_htr.md` |
| redteam | `RESULTS_redteam.md`, `HANDOFF_redteam.md` |
| graph-mechanism | `SUMMARY_ZH_graph_mechanism.md`, `RESULTS_graph_mechanism.md` |
| graph-endpoint | `SUMMARY_ZH_graph_endpoint.md`, `RESULTS_graph_endpoint.md` |
| representation | `SUMMARY_ZH_representation.md` |
| regularisers | `SUMMARY_ZH_regularisers.md` (`PREREGISTRATION_G2.md`) |
| recipe | `SUMMARY_ZH_recipe.md` |
| anchor-exchange | `SUMMARY_ZH_anchor_exchange.md` |
| temporal-graph | `SUMMARY_ZH_temporal_graph.md` |
| calibration | `SUMMARY_ZH_calibration.md` |
| paper-package | `PAPER_PACKAGE/README.md`, `PAPER_PACKAGE/tables/claims_table.md` |
Seed blocks used (do not reuse): 7000–7335, 8000–8184 (graph-mechanism), 8200–8399 (selection-htr), 8400–8584 (graph-endpoint), 8600–8684 (representation), 8700–8784 (anchor-exchange), 8800–8884 (red team), 9000–9084 (regularisers), 9100–9184 (calibration), 9200–9299 (temporal-graph), 9300–9384 (recipe); earlier blocks listed in `nd_core.py`.
