# Pre-registration HF: HTR failure analysis and targeted fix (branch `claude/htr-failures-analysis-6iw4qu`)

Written and committed **before any Stage-2 development cell and before any confirmation cell existed**.  Stage-1 (diagnosis) results are in `HTR_FAILURES_RESULTS.md`.

## What the data are (stated first because it limits what the statistics can mean)
* All studies of this repository use the same 168 labelled pair groups.  Only **45 decisive HTR judgments** exist (26 "1", 19 "2"; 83 distinct images; every (pair, type) was judged exactly once).  A seed re-draws the split, so each HTR
  judgment is a held-out test item in about a quarter of the (seed, split) cells; "pooled HTR judgments, bootstrap by seed" therefore tests whether a learner is better **on these 45 judgments across re-drawn splits and label sets**,
  not on new labels.  Seeds do not add labels.
* Seed blocks (all unused before; asserted in `nd_core.py`): development 10400-10419, confirmation 1 = 10430-10464, confirmation 2 = 10465-10499.

## Baseline and protocol
Baseline `lap3_k10_st4` (the incumbent).  Judgment-unit protocol of `nd_core` / `rt_a1.py`: Splits A and B, budgets 10/20/40/60 random judgments on top of the 10 initial groups, 3 label-set draws per budget, identical label sets for baseline
and candidate (so predictions are paired row by row), per-budget re-tuned head schedule, confidence-weighted held-out decisive-judgment metrics.

## Variants (at most 8, all fixed here; Stage-2 learners in `hf_learners.py`)
`hfb_g5`, `hfb_g10`, `hfb_g20` (post-hoc blend of the HTR logit gap with the anchor-similarity gap, top-1), `hfb3_g10` (top-3 mean), `hfs` (anchor-similarity features in the head input), `hfq1` (anchor-similarity teacher loss on the HTR
output), `hfc1` (cross-type coupling d_HTR - d_RT13), `hfcb10` (coupling + blend g = 10).  Only the HTR output changes for the blend / coupling variants (rows of other types are bit-identical).
Disclosure: before the variants were fixed I looked once at a post-hoc blend `d + g (dHTR(b) - dHTR(a))` with g in {2, 5, 10, 20} on the first 12 of the 20 development seeds (static similarities, identical to the learner up to the centring):
g = 2 was weaker and is not carried; g = 5, 10, 20 are.  That preview is part of the development data and is not evidence.  Diagnosis items not turned into variants because the diagnosis did not support them: hard-negative anchors /
RT13-boundary margin head (HTR-vs-RT13 comparisons are already 100 % correct), re-weighting of conflicting labels (only one conflicting label pair exists), per-type graphs (no sign that the graph is involved), per-type steps.

## Selection of the single candidate (rule fixed now)
On the development seeds, for each variant compute the HTR log-loss gain over the baseline in four conditions: {Split A, Split B} x {standard endpoint, anchor-strict endpoint with delta = 0.01} (anchor-strict = ideal images within
cosine distance 0.01 of any held-out image are removed from all anchor tables, for the variant and the baseline alike, as in the red-team audit rt_a9).  A variant is eligible if in all four conditions HTR accuracy gain >= 0, every other type's
accuracy changes by >= -0.01 and the all-type AUC changes by >= -0.002.  The candidate is the eligible variant with the largest **minimum** HTR log-loss gain over the four conditions; variants within 0.005 of the best are resolved in favour of the
one with the fewest fitted components (blend < coupling < features < teacher).  If no variant is eligible, no candidate is confirmed and that is reported.

## Confirmation (candidate frozen before the first confirmation cell is run)
Confirmation 1 on seeds 10430-10464, confirmation 2 on 10465-10499, both splits, baseline and candidate on identical label sets.  Per confirmation set, four primary tests: HTR accuracy gain and HTR log-loss gain (baseline - candidate), each in
Split A and Split B (confidence-weighted pooled over all HTR test rows of a seed's cells; the seed is the unit).  Inference: bootstrap over seeds (20000 resamples), two-sided bootstrap p, Holm over the four tests.
**Success (per confirmation set)** = all four Holm-adjusted p < 0.05 with positive gains, AND in both splits every other type (1x1, c6x2, RT13) loses <= 0.01 accuracy, AND all-type AUC is not worse by more than 0.002.  Success on both sets is
a replication.  Reported but not part of the success rule: all-type accuracy / log-loss, per-type changes, a sign-flip randomisation p as a cross-check, and the anchor-strict endpoint (delta = 0.003 and 0.01; budgets 20/40, 2 draws) on both
confirmation sets, because the red-team audit showed that gains of earlier learners concentrate on judgments near typed anchors.

## Interpretation rules (fixed in advance)
* If the candidate passes on the standard endpoint but its anchor-strict gain is not significant, the result is reported as "depends on near-copy anchors" and not as an improvement of the learner in general.
* The label-ceiling statement (HTR labels are consistent: leave-one-out nearest-comparison agreement and the label-free rule 'the frame nearer to the ideal HTR images wins' both reach 44/45) is a Stage-1 finding and is reported whatever Stage 2 does.
* No variant is added or tuned after the first confirmation cell.
