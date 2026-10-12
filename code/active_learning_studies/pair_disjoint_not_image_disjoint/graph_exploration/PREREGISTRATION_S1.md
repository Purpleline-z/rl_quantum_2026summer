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

## Amendment 1 (2026-10-12 08:25 UTC; made BEFORE any confirmation seed was analysed)
Disclosure of the sequence: the first version of this file was committed (36f368f7) at ~08:10 while dev screen 5 (type-quota coverage, a mechanism test requested in the plan) was still running; the confirmation chain was set to start after it. When dev 5 finished (08:20) it showed that type-quota coverage `qcov_alloc` (shares t1x1 .25, c(6x2) .30, RT13 .10, HTR .35 chosen from dev1 composition/per-type gains: RT13 saturated) beat its shuffled-posterior control more clearly than any other rule (dev: acc +0.013/+0.010 vs +0.004/+0.004, AUC +0.010/+0.009 vs +0.006/+0.003 regular A/B; cold B acc +0.011 vs −0.009). The confirmation chain had started seconds earlier (seed 8100 only; no result was read; the partial files were deleted and `gs_conf1` restarted from nothing). I therefore amend:
- C4 `htr_cov_amb` is replaced by C4 `qcov_alloc` (typed coverage with fixed type quota ALLOC = {t1x1 .25, c(6x2) .30, RT13 .10, HTR .35}, P(decisive) from the label-spreading features). `htr_cov_amb` and the other HTR/RT13 ambiguity and bridge rules are NOT confirmed; their dev results (inconsistent across splits and start modes; vopt-based variants equal to the class-priority non-graph control) are reported as dev-only. This replaces the HTR-ambiguity candidate the plan asked for; reason: it did not show a robust graph-specific dev signal.
- Families grow to 4 candidates: family A = {C1, C2, C3, C4} x {A, B} x {acc, auc} = 16 tests per block; family B = the same four x {A, B} x {htr_acc, htr_auc} = 16 tests.
- Comparator arms (uncorrected paired differences) per block: `tdc_shuffled`, `tdc_typeonly`, `gvopt_lap3_shuffled@aw`, `vopt_ridge10@aw`, `qcov_alloc_shuffled`, `qcov_alloc_typeonly`; block 3 additionally `htr_vopt_prio@aw`. Block 3 has no shuffled/typeonly arm of its own beyond those listed (`qcov_alloc_shuffled`, `qcov_alloc_typeonly`, `gvopt_lap3_shuffled@aw`, `tdc_shuffled`).
- What 'shuffled' destroys: in `ImageGraph(shuffle=True)` the label-spreading type posterior q is permuted over the nodes; the kNN adjacency (used for the propagated pair space of the coverage rules) is kept. So the shuffled control tests the type-posterior ingredient, not graph propagation of image features; a type-only control (`*_typeonly`: no reference/graph features in P(decisive)) tests the posterior's gain over type identity. For `gvopt_lap3` the adjacency itself is permuted (`graph_vopt._shuffled`).
- ALLOC was tuned by eye on dev seeds (winner's curse expected): the effect sizes in dev are likely overstated.
Success criteria (A) and (B) are unchanged and apply to C1–C4 as written above; the HTR-ambiguity rule is no longer a candidate for (B).
