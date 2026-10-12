# PREREGISTRATION_S1 — frozen candidates and tests (Prompt 2; written 2026-10-12 ~08:10 UTC, BEFORE any seed >= 8100 was run)
Everything below is fixed by this commit. Dev observations that led to it are in `LOG_graph_selection.md` and are not evidence.

## Setup (unchanged protocol)
Learner `aw8_all` for every arm; judgment-unit protocol; budgets 10/20/40/60; splits A and B; single-shot selection from the model fitted on the initial judgments; Random = mean of **5** draws (the protocol's replicate count; dev used 3); selectors only see candidates, revealed judgments, the model and candidate/labelled/reference images (`ImageGraph`). Harness `new_directions/gsel_select.py`, tables by `gsel_analyze.py`.
Gain over Random = per-seed mean over the four budgets of (arm − Random); for log-loss sign flipped. Test = two-sided Wilcoxon signed-rank over the 35 seeds of the block. 'Other-type accuracy' = mean of the t1x1, c(6x2), RT13 accuracies.

## Candidates (Holm family members; <= 4)
C1 `vopt_u@aw` (repository's pool-wide I-optimal design, model fitted with aw8_all — NOT a graph rule; the strongest non-graph reference);
C2 `gvopt_lap3@aw` (Laplacian-prior graph design, lambda = 3; frozen graph rule);
C3 `tdc` = `typed_decisive_coverage` (frozen graph rule);
C4 `htr_cov_amb` (new HTR-oriented graph rule: typed coverage in graph-propagated pair space, distance x P(decisive) x [0.1 + mean HTR-vs-RT13 label-spreading ambiguity of the two images for HTR/RT13 judgments; 0.1 for other types]).
Interpretation of "<= 4 candidates (incl. controls)": 4 rules enter the Holm families; the matching **comparator arms** (not tested against Random inside the family, compared with paired differences, uncorrected) are `tdc_shuffled`, `tdc_typeonly`, `gvopt_lap3_shuffled@aw`, `htr_cov_amb_shuffled`, plus two non-graph controls `vopt_ridge10@aw` (Laplacian prior replaced by a plain stronger ridge prior) and `htr_vopt_prio@aw` (class priority for HTR with no graph).

## Blocks
Confirmation 1: seeds 8100–8134, regular start (10 initial groups). Arms: random, C1, C2, C3, tdc_shuffled, tdc_typeonly, gvopt_lap3_shuffled@aw, vopt_ridge10@aw.
Confirmation 2: seeds 8200–8234, cold start (`ND_INITIAL=random`, 10 random initial judgments). Same arms.
Confirmation 3: seeds 8300–8334, regular start, HTR block. Arms: random, C1, C2, C3, C4, htr_cov_amb_shuffled, htr_vopt_prio@aw, vopt_ridge10@aw.
Each block is run once.

## Families and criteria
(A) Blocks 1 and 2 separately: family = {C1, C2, C3} x {A, B} x {acc, auc} = 12 tests per block, Holm. A rule 'beats Random' in a block if acc AND auc gains are Holm < 0.05 in BOTH splits. A GRAPH claim for C2/C3 additionally requires the paired difference to the matching shuffled control (and, for C3, the type-only control) to be positive in both splits and p < 0.05 uncorrected for AUC or accuracy; otherwise the result is reported as 'beats Random, graph not the ingredient'. C1 is not a graph rule; its result is reported as a non-graph reference.
(B) Block 3: family = {C1, C2, C3, C4} x {A, B} x {htr_acc, htr_auc} = 16 tests, Holm. An HTR rule succeeds if both HTR metrics are Holm < 0.05 in both splits AND the mean gain of other-type accuracy is >= −0.01 in both splits. Graph claims need C4/C2/C3 to beat their shuffled control as above.
Secondary (descriptive, no Holm): log-loss gains; HTR metrics in blocks 1–2; all-type per-type gains; composition and decisive share of the batches; Random-draw noise.
Anything not listed here (extra arms, other metrics, other budgets) is exploratory and will be labelled so.
Known limitation: all seeds re-split the same 168 pair groups, so p-values are optimistic; a significant result is evidence about this dataset's split variability, not about new data.
