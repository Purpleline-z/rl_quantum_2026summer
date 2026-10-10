### 5.20 Graph rules under the re-tuned head schedule, and a graph-regularised variance-reduction design

**Why this section.** Every graph experiment of §5.16–5.18 used the head schedule of §5.12–5.13 (lr 0.01, 100 steps). §5.14 showed that this schedule is too aggressive for 10–60 judgments, and §5.15 that the pool-wide variance-reduction rules gain over Random only under the re-tuned schedule (lr 0.001, 100 steps; lr 0.003 at 60 judgments) or a gentler one (lr 0.0003). The graph rules had therefore not been tested in the regime in which other rules of this report show an effect. Everything here uses the judgment-unit protocol of §5.14 (one (pair, type) judgment per query, budgets of 10, 20, 40 and 60 judgments, splits A and B, single-shot and sequential selection, Random = mean of five draws, the held-out preference endpoint) with the per-budget head schedule named in each part; the hypotheses, the seed sets and the freezing of rules were written to `graph_exploration/OVERNIGHT2_PLAN.md` and `AUTONOMOUS_RUN_LOG.md` before the corresponding cells were run. Gains are per-seed differences from Random averaged over the four budgets; p-values are seed-level Wilcoxon signed-rank tests; Holm correction is over the frozen candidates within each block; paired differences to controls carry raw p-values and are not corrected for multiplicity.

**Part 1: the frozen graph rules under the re-tuned schedule (30 fresh seeds, 1500–1529).** The three rules frozen in §5.17 (`typed_decisive_coverage_unc`, `typed_decisive_coverage`, `typed_decisive_bald`) were run unchanged, with the type-only and shuffled-posterior controls of the plain rule and, as references, the pool-wide I-optimal design without a graph (`vopt_u`, §5.15) and relation-aware Core-set.

{TABLE_P1}

The plain rule has a positive log-loss gain over Random in all four blocks (Holm p between 0.002 and 0.009), with positive AUC and accuracy gains; the Laplace-BALD rule is significant in three blocks and the uncertainty-weighted rule in two. This differs from the old-schedule result of §5.17–5.18 and shows that the earlier null was conditional on the schedule. In contrast with the old schedule, the type posterior is also measurably useful: the plain rule is above its type-only control in AUC in all four blocks (+0.012 to +0.015, raw p between 0.003 and 0.025) and above its shuffled-posterior control in AUC in all four (+0.007 to +0.012, raw p between 0.014 and 0.096, only one below 0.05), and its log-loss differences to the controls are positive but mostly not significant. The rule is not distinguishable from `vopt_u` (log-loss differences −0.012 to +0.006, none significant), so the graph rule is comparable with, not better than, the best non-graph rule of §5.15.

**Part 2: a graph-regularised variance-reduction design (development on seeds 600–624).** The strongest non-graph rule of §5.15, `vopt_u`, is a pool-wide I-optimal design on the last layer of the own-type head under the Laplace posterior. Because the score of a judgment is f(a) − f(b) with f(x) = θ·h(x) and h the hidden layer, a smoothness prior for f over the image graph is a prior on the head parameters θ. `gvopt_lap` adds to the prior precision (ridge · I) the term λ · HᵀLH rescaled to mean diagonal 1, with H the hidden features of the images of the kNN graph (candidates, revealed judgments and typed reference images) and L its Laplacian; this is the Laplacian-regularised optimal design of He (IEEE TIP 2010) and Cai and He (IEEE TKDE 2012) applied to the last layer (`graph_exploration/notes_overnight2_literature.md`). Further variants weighted the pool judgments by the type posterior (`gvopt_type`), designed on propagated features (`gvopt_prop`), or used the Σ-optimal criterion of Ma, Garnett and Schneider (NeurIPS 2013) with and without the graph prior (`gvopt_sigma`, `gvopt_lapsigma`); λ was varied over 0.3, 1, 3 and 10. Development results (mean over the four blocks of the gain over Random; the last three columns are the differences to `vopt_u`):

```
{TABLE_P2}
```

In split A the Laplacian-prior rules clearly beat Random on the development seeds, in split B nothing beats Random on log-loss (all about 0) and only AUC and accuracy are positive; the Σ-optimal and propagated-feature variants are worse than `vopt_u`. `gvopt_lap3` (λ = 3) was frozen as the graph method because it was the only variant above `vopt_u` on all three metrics; λ was therefore chosen on the development seeds from four values and is a tuned setting.

**Part 3: confirmation of the frozen method under the re-tuned schedule (35 fresh seeds, 1400–1434).**

{TABLE_P3}

The method did not replicate. `gvopt_lap3` has no significant log-loss gain over Random in any block (−0.018, −0.014, +0.006, +0.011), its positive AUC (+0.008 to +0.009) and accuracy (+0.013 to +0.020) gains equal those of its shuffled-graph control and of `vopt_u`, and it is below `vopt_u` on log-loss in split A sequential (−0.032, raw p 0.046). The development advantage in split A was therefore selection among nine variants and four λ values on 25 seeds. The second candidate, the plain typed coverage rule, was significant in split B (Holm < 0.001 and 0.008) and not in split A, so the pre-registered criterion (Holm p < 0.05 in at least three of four blocks) was not met by either rule in this seed set.

**Part 4: the same frozen rules under a gentler schedule (lr 0.0003, seeds 1400–1434).**

{TABLE_P4}

Under the gentler schedule (the one for which §5.15 reports the largest gains of the variance-reduction rules) the picture reverses: the graph-regularised design beats Random in all four blocks (log-loss +0.033, +0.039, +0.044, +0.041, Holm p < 0.002, better in 74–97% of seeds; AUC +0.016 to +0.019; accuracy +0.016 to +0.028), and so does `vopt_u` (+0.031 to +0.035); the typed coverage rule is weaker (+0.012 to +0.019). The graph design is not distinguishable from `vopt_u` in split A (+0.002, +0.005) and is above it on log-loss in split B (+0.010 and +0.011, raw p 0.039 and 0.007, uncorrected; AUC +0.003 and +0.001). The shuffled-graph control was not part of this run.

{PARTS_5_6}
