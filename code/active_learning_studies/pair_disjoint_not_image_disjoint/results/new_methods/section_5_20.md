### 5.20 Graph rules under the re-tuned head schedule, and a graph-regularised variance-reduction design

**Why this section.** Every graph experiment of §5.16–5.18 used the head schedule of §5.12–5.13 (lr 0.01, 100 steps). §5.14 showed that this schedule is too aggressive for 10–60 judgments, and §5.15 that the pool-wide variance-reduction rules gain over Random only under the re-tuned schedule (lr 0.001, 100 steps; lr 0.003 at 60 judgments) or a gentler one (lr 0.0003). The graph rules had therefore not been tested in the regime in which other rules of this report show an effect. Everything here uses the judgment-unit protocol of §5.14 (one (pair, type) judgment per query, budgets of 10, 20, 40 and 60 judgments, splits A and B, single-shot and sequential selection, Random = mean of five draws, the held-out preference endpoint) with the per-budget head schedule named in each part; the hypotheses, the seed sets and the freezing of rules were written to `graph_exploration/OVERNIGHT2_PLAN.md` and `AUTONOMOUS_RUN_LOG.md` before the corresponding cells were run. Gains are per-seed differences from Random averaged over the four budgets; p-values are seed-level Wilcoxon signed-rank tests; Holm correction is over the frozen candidates within each block; paired differences to controls carry raw p-values and are not corrected for multiplicity.

**Part 1: the frozen graph rules under the re-tuned schedule (30 fresh seeds, 1500–1529).** The three rules frozen in §5.17 (`typed_decisive_coverage_unc`, `typed_decisive_coverage`, `typed_decisive_bald`) were run unchanged, with the type-only and shuffled-posterior controls of the plain rule and, as references, the pool-wide I-optimal design without a graph (`vopt_u`, §5.15) and relation-aware Core-set.

| Split | Condition | Rule | log-loss gain | raw p | Holm p (candidates) | AUC gain | accuracy gain | seeds better | n |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | single-shot | Type-aware graph coverage x P(decisive) | +0.032 | <0.001 | 0.002 | +0.0105 | +0.0088 | 73% | 30 |
| A | single-shot | ... x own-head uncertainty | +0.022 | 0.002 | 0.002 | +0.0110 | +0.0135 | 80% | 30 |
| A | single-shot | Laplace BALD x P(decisive), graph features | +0.040 | <0.001 | 0.002 | +0.0166 | +0.0157 | 73% | 30 |
| A | single-shot | (control) decisive predictor from the type only | +0.017 | 0.067 |  | -0.0017 | -0.0037 | 63% | 30 |
| A | single-shot | (control) type posterior shuffled over images | +0.027 | 0.040 |  | +0.0033 | +0.0016 | 63% | 30 |
| A | single-shot | (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.028 | 0.008 |  | +0.0113 | +0.0130 | 70% | 30 |
| A | single-shot | (reference) Core-set, relation-aware pairs | +0.025 | 0.073 |  | +0.0048 | -0.0017 | 67% | 30 |
| A | sequential | Type-aware graph coverage x P(decisive) | +0.027 | 0.003 | 0.006 | +0.0079 | +0.0150 | 73% | 30 |
| A | sequential | ... x own-head uncertainty | +0.025 | 0.011 | 0.011 | +0.0105 | +0.0246 | 70% | 30 |
| A | sequential | Laplace BALD x P(decisive), graph features | +0.034 | 0.002 | 0.006 | +0.0127 | +0.0218 | 70% | 30 |
| A | sequential | (control) decisive predictor from the type only | +0.001 | 0.598 |  | -0.0066 | -0.0042 | 57% | 30 |
| A | sequential | (control) type posterior shuffled over images | +0.015 | 0.229 |  | -0.0002 | +0.0056 | 57% | 30 |
| A | sequential | (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.021 | 0.119 |  | +0.0100 | +0.0229 | 57% | 30 |
| A | sequential | (reference) Core-set, relation-aware pairs | +0.024 | 0.040 |  | +0.0048 | +0.0136 | 63% | 30 |
| B | single-shot | Type-aware graph coverage x P(decisive) | +0.047 | <0.001 | 0.003 | +0.0134 | +0.0105 | 73% | 30 |
| B | single-shot | ... x own-head uncertainty | +0.026 | 0.124 | 0.124 | +0.0100 | +0.0200 | 60% | 30 |
| B | single-shot | Laplace BALD x P(decisive), graph features | +0.038 | 0.005 | 0.009 | +0.0149 | +0.0155 | 67% | 30 |
| B | single-shot | (control) decisive predictor from the type only | +0.025 | 0.050 |  | +0.0016 | +0.0037 | 63% | 30 |
| B | single-shot | (control) type posterior shuffled over images | +0.035 | 0.047 |  | +0.0067 | +0.0069 | 67% | 30 |
| B | single-shot | (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.058 | 0.001 |  | +0.0189 | +0.0150 | 73% | 30 |
| B | single-shot | (reference) Core-set, relation-aware pairs | +0.047 | 0.001 |  | +0.0115 | +0.0060 | 70% | 30 |
| B | sequential | Type-aware graph coverage x P(decisive) | +0.050 | 0.003 | 0.009 | +0.0140 | +0.0143 | 70% | 30 |
| B | sequential | ... x own-head uncertainty | +0.032 | 0.070 | 0.140 | +0.0071 | +0.0259 | 70% | 30 |
| B | sequential | Laplace BALD x P(decisive), graph features | +0.034 | 0.084 | 0.140 | +0.0121 | +0.0167 | 60% | 30 |
| B | sequential | (control) decisive predictor from the type only | +0.022 | 0.177 |  | -0.0007 | +0.0063 | 57% | 30 |
| B | sequential | (control) type posterior shuffled over images | +0.028 | 0.088 |  | +0.0025 | +0.0082 | 57% | 30 |
| B | sequential | (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.048 | 0.031 |  | +0.0163 | +0.0217 | 63% | 30 |
| B | sequential | (reference) Core-set, relation-aware pairs | +0.047 | 0.025 |  | +0.0088 | +0.0064 | 67% | 30 |

| Split | Condition | Paired difference | log-loss | raw p | AUC | raw p | accuracy | raw p |
|---|---|---|---:|---:|---:|---:|---:|---:|
| A | single-shot | Type-aware graph coverage x P(decisive) minus (control) decisive predictor from the type only | +0.0154 | 0.100 | +0.0123 | 0.004 | +0.0125 | 0.023 |
| A | single-shot | Type-aware graph coverage x P(decisive) minus (control) type posterior shuffled over images | +0.0051 | 0.503 | +0.0072 | 0.096 | +0.0072 | 0.198 |
| A | single-shot | Type-aware graph coverage x P(decisive) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.0037 | 0.919 | -0.0007 | 0.792 | -0.0042 | 0.265 |
| A | sequential | Type-aware graph coverage x P(decisive) minus (control) decisive predictor from the type only | +0.0257 | 0.022 | +0.0145 | 0.003 | +0.0192 | 0.003 |
| A | sequential | Type-aware graph coverage x P(decisive) minus (control) type posterior shuffled over images | +0.0124 | 0.198 | +0.0081 | 0.086 | +0.0094 | 0.068 |
| A | sequential | Type-aware graph coverage x P(decisive) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.0056 | 0.919 | -0.0021 | 0.715 | -0.0079 | 0.096 |
| B | single-shot | Type-aware graph coverage x P(decisive) minus (control) decisive predictor from the type only | +0.0215 | 0.245 | +0.0119 | 0.025 | +0.0068 | 0.239 |
| B | single-shot | Type-aware graph coverage x P(decisive) minus (control) type posterior shuffled over images | +0.0116 | 0.280 | +0.0067 | 0.064 | +0.0037 | 0.370 |
| B | single-shot | Type-aware graph coverage x P(decisive) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | -0.0115 | 0.349 | -0.0054 | 0.428 | -0.0045 | 0.781 |
| B | sequential | Type-aware graph coverage x P(decisive) minus (control) decisive predictor from the type only | +0.0282 | 0.031 | +0.0146 | 0.004 | +0.0080 | 0.133 |
| B | sequential | Type-aware graph coverage x P(decisive) minus (control) type posterior shuffled over images | +0.0229 | 0.061 | +0.0115 | 0.014 | +0.0060 | 0.198 |
| B | sequential | Type-aware graph coverage x P(decisive) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.0027 | 0.715 | -0.0023 | 0.730 | -0.0074 | 0.238 |

The plain rule has a positive log-loss gain over Random in all four blocks (Holm p between 0.002 and 0.009), with positive AUC and accuracy gains; the Laplace-BALD rule is significant in three blocks and the uncertainty-weighted rule in two. This differs from the old-schedule result of §5.17–5.18 and shows that the earlier null was conditional on the schedule. In contrast with the old schedule, the type posterior is also measurably useful: the plain rule is above its type-only control in AUC in all four blocks (+0.012 to +0.015, raw p between 0.003 and 0.025) and above its shuffled-posterior control in AUC in all four (+0.007 to +0.012, raw p between 0.014 and 0.096, only one below 0.05), and its log-loss differences to the controls are positive but mostly not significant. The rule is not distinguishable from `vopt_u` (log-loss differences −0.012 to +0.006, none significant), so the graph rule is comparable with, not better than, the best non-graph rule of §5.15.

**Part 2: a graph-regularised variance-reduction design (development on seeds 600–624).** The strongest non-graph rule of §5.15, `vopt_u`, is a pool-wide I-optimal design on the last layer of the own-type head under the Laplace posterior. Because the score of a judgment is f(a) − f(b) with f(x) = θ·h(x) and h the hidden layer, a smoothness prior for f over the image graph is a prior on the head parameters θ. `gvopt_lap` adds to the prior precision (ridge · I) the term λ · HᵀLH rescaled to mean diagonal 1, with H the hidden features of the images of the kNN graph (candidates, revealed judgments and typed reference images) and L its Laplacian; this is the Laplacian-regularised optimal design of He (IEEE TIP 2010) and Cai and He (IEEE TKDE 2012) applied to the last layer (`graph_exploration/notes_overnight2_literature.md`). Further variants weighted the pool judgments by the type posterior (`gvopt_type`), designed on propagated features (`gvopt_prop`), or used the Σ-optimal criterion of Ma, Garnett and Schneider (NeurIPS 2013) with and without the graph prior (`gvopt_sigma`, `gvopt_lapsigma`); λ was varied over 0.3, 1, 3 and 10. Development results (mean over the four blocks of the gain over Random; the last three columns are the differences to `vopt_u`):

```
rule             log_loss       auc  accuracy   | minus vopt_u: log_loss       auc  accuracy  blocks all-positive
vopt_u            +0.0133   +0.0076   +0.0164   |                  +0.0000   +0.0000   +0.0000  4/4
gvopt_lap         +0.0135   +0.0095   +0.0186   |                  +0.0002   +0.0019   +0.0022  2/4
gvopt_lap03       +0.0085   +0.0082   +0.0171   |                  -0.0048   +0.0006   +0.0007  2/4
gvopt_lap3        +0.0164   +0.0111   +0.0210   |                  +0.0031   +0.0034   +0.0046  3/4
gvopt_lap10       +0.0122   +0.0115   +0.0204   |                  -0.0011   +0.0039   +0.0040  2/4
gvopt_type        +0.0177   +0.0066   +0.0110   |                  +0.0045   -0.0010   -0.0054  4/4
gvopt_prop        +0.0082   +0.0055   +0.0106   |                  -0.0051   -0.0021   -0.0058  4/4
gvopt_sigma       +0.0029   +0.0031   +0.0062   |                  -0.0103   -0.0045   -0.0102  2/4
gvopt_lapsigma    -0.0009   +0.0045   +0.0100   |                  -0.0142   -0.0031   -0.0064  3/4
```

In split A the Laplacian-prior rules clearly beat Random on the development seeds, in split B nothing beats Random on log-loss (all about 0) and only AUC and accuracy are positive; the Σ-optimal and propagated-feature variants are worse than `vopt_u`. `gvopt_lap3` (λ = 3) was frozen as the graph method because it was the only variant above `vopt_u` on all three metrics; λ was therefore chosen on the development seeds from four values and is a tuned setting.

**Part 3: confirmation of the frozen method under the re-tuned schedule (35 fresh seeds, 1400–1434).**

| Split | Condition | Rule | log-loss gain | raw p | Holm p (candidates) | AUC gain | accuracy gain | seeds better | n |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | single-shot | Laplacian-regularised I-optimal design (lam 3) | -0.018 | 0.232 | 0.279 | +0.0082 | +0.0140 | 40% | 35 |
| A | single-shot | Type-aware graph coverage x P(decisive) | +0.014 | 0.140 | 0.279 | +0.0043 | +0.0038 | 57% | 35 |
| A | single-shot | (control) same, graph shuffled | -0.009 | 0.955 |  | +0.0042 | +0.0124 | 60% | 35 |
| A | single-shot | (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.001 | 0.903 |  | +0.0078 | +0.0164 | 54% | 35 |
| A | sequential | Laplacian-regularised I-optimal design (lam 3) | -0.014 | 0.342 | 0.685 | +0.0094 | +0.0202 | 46% | 35 |
| A | sequential | Type-aware graph coverage x P(decisive) | +0.008 | 0.471 | 0.685 | +0.0025 | +0.0043 | 54% | 35 |
| A | sequential | (control) same, graph shuffled | +0.003 | 0.441 |  | +0.0070 | +0.0174 | 57% | 35 |
| A | sequential | (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.018 | 0.154 |  | +0.0100 | +0.0141 | 54% | 35 |
| B | single-shot | Laplacian-regularised I-optimal design (lam 3) | +0.006 | 0.994 | 0.994 | +0.0092 | +0.0132 | 43% | 35 |
| B | single-shot | Type-aware graph coverage x P(decisive) | +0.038 | <0.001 | <0.001 | +0.0105 | +0.0099 | 86% | 35 |
| B | single-shot | (control) same, graph shuffled | +0.022 | 0.039 |  | +0.0118 | +0.0113 | 66% | 35 |
| B | single-shot | (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.025 | 0.023 |  | +0.0101 | +0.0102 | 60% | 35 |
| B | sequential | Laplacian-regularised I-optimal design (lam 3) | +0.011 | 0.213 | 0.213 | +0.0085 | +0.0160 | 66% | 35 |
| B | sequential | Type-aware graph coverage x P(decisive) | +0.028 | 0.004 | 0.008 | +0.0065 | +0.0047 | 69% | 35 |
| B | sequential | (control) same, graph shuffled | +0.012 | 0.219 |  | +0.0058 | +0.0095 | 63% | 35 |
| B | sequential | (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.021 | 0.017 |  | +0.0083 | +0.0158 | 66% | 35 |

| Split | Condition | Paired difference | log-loss | raw p | AUC | raw p | accuracy | raw p |
|---|---|---|---:|---:|---:|---:|---:|---:|
| A | single-shot | Laplacian-regularised I-optimal design (lam 3) minus (control) same, graph shuffled | -0.0091 | 0.471 | +0.0040 | 0.287 | +0.0016 | 0.612 |
| A | single-shot | Laplacian-regularised I-optimal design (lam 3) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | -0.0186 | 0.245 | +0.0004 | 0.831 | -0.0024 | 0.602 |
| A | single-shot | Laplacian-regularised I-optimal design (lam 3) minus Type-aware graph coverage x P(decisive) | -0.0315 | 0.040 | +0.0039 | 0.413 | +0.0102 | 0.081 |
| A | sequential | Laplacian-regularised I-optimal design (lam 3) minus (control) same, graph shuffled | -0.0177 | 0.385 | +0.0024 | 0.359 | +0.0029 | 0.280 |
| A | sequential | Laplacian-regularised I-optimal design (lam 3) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | -0.0320 | 0.046 | -0.0006 | 0.668 | +0.0061 | 0.172 |
| A | sequential | Laplacian-regularised I-optimal design (lam 3) minus Type-aware graph coverage x P(decisive) | -0.0224 | 0.207 | +0.0069 | 0.326 | +0.0159 | 0.006 |
| B | single-shot | Laplacian-regularised I-optimal design (lam 3) minus (control) same, graph shuffled | -0.0156 | 0.075 | -0.0026 | 0.422 | +0.0019 | 0.570 |
| B | single-shot | Laplacian-regularised I-optimal design (lam 3) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | -0.0190 | 0.154 | -0.0010 | 0.844 | +0.0031 | 0.411 |
| B | single-shot | Laplacian-regularised I-optimal design (lam 3) minus Type-aware graph coverage x P(decisive) | -0.0320 | 0.014 | -0.0013 | 0.481 | +0.0033 | 0.417 |
| B | sequential | Laplacian-regularised I-optimal design (lam 3) minus (control) same, graph shuffled | -0.0012 | 0.840 | +0.0027 | 0.368 | +0.0065 | 0.399 |
| B | sequential | Laplacian-regularised I-optimal design (lam 3) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | -0.0107 | 0.815 | +0.0002 | 0.351 | +0.0002 | 0.647 |
| B | sequential | Laplacian-regularised I-optimal design (lam 3) minus Type-aware graph coverage x P(decisive) | -0.0173 | 0.441 | +0.0020 | 0.656 | +0.0113 | 0.057 |

The method did not replicate. `gvopt_lap3` has no significant log-loss gain over Random in any block (−0.018, −0.014, +0.006, +0.011), its positive AUC (+0.008 to +0.009) and accuracy (+0.013 to +0.020) gains equal those of its shuffled-graph control and of `vopt_u`, and it is below `vopt_u` on log-loss in split A sequential (−0.032, raw p 0.046). The development advantage in split A was therefore selection among nine variants and four λ values on 25 seeds. The second candidate, the plain typed coverage rule, was significant in split B (Holm < 0.001 and 0.008) and not in split A, so the pre-registered criterion (Holm p < 0.05 in at least three of four blocks) was not met by either rule in this seed set.

**Part 4: the same frozen rules under a gentler schedule (lr 0.0003, seeds 1400–1434).**

| Split | Condition | Rule | log-loss gain | raw p | Holm p (candidates) | AUC gain | accuracy gain | seeds better | n |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | single-shot | Laplacian-regularised I-optimal design (lam 3) | +0.033 | <0.001 | 0.002 | +0.0179 | +0.0266 | 74% | 35 |
| A | single-shot | Type-aware graph coverage x P(decisive) | +0.014 | 0.046 | 0.046 | +0.0061 | +0.0091 | 63% | 35 |
| A | single-shot | (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.031 | <0.001 |  | +0.0193 | +0.0271 | 74% | 35 |
| A | sequential | Laplacian-regularised I-optimal design (lam 3) | +0.039 | <0.001 | <0.001 | +0.0193 | +0.0276 | 74% | 35 |
| A | sequential | Type-aware graph coverage x P(decisive) | +0.014 | 0.056 | 0.056 | +0.0040 | +0.0074 | 63% | 35 |
| A | sequential | (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.035 | <0.001 |  | +0.0170 | +0.0250 | 80% | 35 |
| B | single-shot | Laplacian-regularised I-optimal design (lam 3) | +0.044 | <0.001 | <0.001 | +0.0182 | +0.0161 | 97% | 35 |
| B | single-shot | Type-aware graph coverage x P(decisive) | +0.019 | <0.001 | <0.001 | +0.0104 | +0.0103 | 77% | 35 |
| B | single-shot | (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.033 | <0.001 |  | +0.0148 | +0.0103 | 77% | 35 |
| B | sequential | Laplacian-regularised I-optimal design (lam 3) | +0.041 | <0.001 | <0.001 | +0.0162 | +0.0196 | 94% | 35 |
| B | sequential | Type-aware graph coverage x P(decisive) | +0.012 | 0.019 | 0.019 | +0.0037 | +0.0063 | 74% | 35 |
| B | sequential | (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.031 | <0.001 |  | +0.0150 | +0.0179 | 83% | 35 |

| Split | Condition | Paired difference | log-loss | raw p | AUC | raw p | accuracy | raw p |
|---|---|---|---:|---:|---:|---:|---:|---:|
| A | single-shot | Laplacian-regularised I-optimal design (lam 3) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.0018 | 0.840 | -0.0014 | 0.891 | -0.0005 | 0.831 |
| A | single-shot | Type-aware graph coverage x P(decisive) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | -0.0171 | 0.078 | -0.0131 | 0.023 | -0.0180 | <0.001 |
| A | sequential | Laplacian-regularised I-optimal design (lam 3) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.0046 | 0.565 | +0.0023 | 0.576 | +0.0026 | 0.516 |
| A | sequential | Type-aware graph coverage x P(decisive) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | -0.0214 | 0.023 | -0.0130 | 0.011 | -0.0175 | <0.001 |
| B | single-shot | Laplacian-regularised I-optimal design (lam 3) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.0103 | 0.039 | +0.0034 | 0.287 | +0.0058 | 0.367 |
| B | single-shot | Type-aware graph coverage x P(decisive) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | -0.0149 | 0.219 | -0.0043 | 0.680 | -0.0000 | 0.957 |
| B | sequential | Laplacian-regularised I-optimal design (lam 3) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.0106 | 0.007 | +0.0012 | 0.743 | +0.0017 | 0.789 |
| B | sequential | Type-aware graph coverage x P(decisive) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | -0.0188 | 0.158 | -0.0113 | 0.078 | -0.0115 | 0.078 |

Under the gentler schedule (the one for which §5.15 reports the largest gains of the variance-reduction rules) the picture reverses: the graph-regularised design beats Random in all four blocks (log-loss +0.033, +0.039, +0.044, +0.041, Holm p < 0.002, better in 74–97% of seeds; AUC +0.016 to +0.019; accuracy +0.016 to +0.028), and so does `vopt_u` (+0.031 to +0.035); the typed coverage rule is weaker (+0.012 to +0.019). The graph design is not distinguishable from `vopt_u` in split A (+0.002, +0.005) and is above it on log-loss in split B (+0.010 and +0.011, raw p 0.039 and 0.007, uncorrected; AUC +0.003 and +0.001). The shuffled-graph control was not part of this run.

**Part 5: replication of the frozen graph-regularised design under the gentler schedule (40 new seeds, 1700–1739).** Because the frozen `gvopt_lap3` had been fixed before Part 4 and was not tuned on it, a pre-registered replication was run on 40 seeds not used before, with `vopt_u` and the shuffled-graph control `gvopt_lap3_shuffled` (permuted node identities of the graph, same degree sequence). H1: positive log-loss gain over Random in every block (Holm over the four blocks); H2: a positive paired difference to `vopt_u` and to the shuffled control, claimed only if significant after Holm in a split-B block.

| Split | Condition | Rule | log-loss gain | raw p | Holm p (candidates) | AUC gain | accuracy gain | seeds better | n |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | single-shot | Laplacian-regularised I-optimal design (lam 3) | +0.039 | <0.001 | <0.001 | +0.0159 | +0.0191 | 80% | 40 |
| A | single-shot | (control) same, graph shuffled | +0.038 | <0.001 |  | +0.0169 | +0.0182 | 88% | 40 |
| A | single-shot | (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.034 | <0.001 |  | +0.0148 | +0.0192 | 88% | 40 |
| A | sequential | Laplacian-regularised I-optimal design (lam 3) | +0.039 | <0.001 | <0.001 | +0.0174 | +0.0196 | 85% | 40 |
| A | sequential | (control) same, graph shuffled | +0.038 | <0.001 |  | +0.0182 | +0.0175 | 88% | 40 |
| A | sequential | (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.036 | <0.001 |  | +0.0179 | +0.0182 | 80% | 40 |
| B | single-shot | Laplacian-regularised I-optimal design (lam 3) | +0.038 | <0.001 | <0.001 | +0.0169 | +0.0175 | 80% | 40 |
| B | single-shot | (control) same, graph shuffled | +0.036 | <0.001 |  | +0.0163 | +0.0201 | 78% | 40 |
| B | single-shot | (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.038 | <0.001 |  | +0.0184 | +0.0212 | 78% | 40 |
| B | sequential | Laplacian-regularised I-optimal design (lam 3) | +0.031 | <0.001 | <0.001 | +0.0148 | +0.0184 | 85% | 40 |
| B | sequential | (control) same, graph shuffled | +0.029 | <0.001 |  | +0.0138 | +0.0195 | 78% | 40 |
| B | sequential | (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.028 | <0.001 |  | +0.0124 | +0.0184 | 78% | 40 |

| Split | Condition | Paired difference | log-loss | raw p | AUC | raw p | accuracy | raw p |
|---|---|---|---:|---:|---:|---:|---:|---:|
| A | single-shot | Laplacian-regularised I-optimal design (lam 3) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.0053 | 0.221 | +0.0011 | 0.493 | -0.0001 | 0.734 |
| A | single-shot | Laplacian-regularised I-optimal design (lam 3) minus (control) same, graph shuffled | +0.0012 | 0.879 | -0.0011 | 0.485 | +0.0009 | 0.739 |
| A | sequential | Laplacian-regularised I-optimal design (lam 3) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.0031 | 0.320 | -0.0005 | 0.765 | +0.0014 | 0.338 |
| A | sequential | Laplacian-regularised I-optimal design (lam 3) minus (control) same, graph shuffled | +0.0005 | 0.755 | -0.0008 | 0.444 | +0.0021 | 0.177 |
| B | single-shot | Laplacian-regularised I-optimal design (lam 3) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.0001 | 0.755 | -0.0015 | 0.675 | -0.0037 | 0.222 |
| B | single-shot | Laplacian-regularised I-optimal design (lam 3) minus (control) same, graph shuffled | +0.0018 | 0.675 | +0.0006 | 0.858 | -0.0026 | 0.379 |
| B | sequential | Laplacian-regularised I-optimal design (lam 3) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.0038 | 0.354 | +0.0024 | 0.301 | +0.0001 | 0.780 |
| B | sequential | Laplacian-regularised I-optimal design (lam 3) minus (control) same, graph shuffled | +0.0022 | 0.420 | +0.0010 | 0.538 | -0.0010 | 0.722 |

H1 is met: the graph-regularised design beats Random on log-loss in all four blocks (+0.039, +0.039, +0.038, +0.031; Holm p < 0.001; better in 80–85% of seeds), with AUC gains of +0.015 to +0.017 and accuracy gains of +0.018 to +0.019. H2 is not met: the differences to `vopt_u` (+0.005, +0.003, +0.000, +0.004) and to the shuffled-graph control (+0.001, +0.001, +0.002, +0.002) are within noise (raw p ≥ 0.22), and the shuffled-graph control and `vopt_u` have essentially the same gains over Random as the graph version (+0.036 to +0.038 and +0.028 to +0.038). The split-B advantage over `vopt_u` seen in Part 4 (+0.010, raw p 0.007 and 0.039) did not reproduce. The conclusion is that, under the gentler head schedule, variance-reduction design on the last layer beats Random reproducibly (this section and §5.15), and that adding a graph prior to the design neither helps nor hurts measurably.

**Part 6: the typed coverage rule on 40 further fresh seeds under the re-tuned schedule (seeds 1600–1639), and pooled over all fresh sets.** Because the plain typed coverage rule was significant in Part 1, in two blocks only in Part 3, and its own controls had been run only in Part 1, a pre-registered large-sample run (`OVERNIGHT2_PLAN.md`, P5) repeated it, unchanged, with the type-only and shuffled-posterior controls and `vopt_u` on 40 seeds not used before (primary test: log-loss gain over Random, Holm over the four blocks).

| Split | Condition | Rule | log-loss gain | raw p | Holm p (candidates) | AUC gain | accuracy gain | seeds better | n |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | single-shot | Type-aware graph coverage x P(decisive) | +0.023 | 0.008 | 0.008 | +0.0077 | +0.0131 | 68% | 40 |
| A | single-shot | (control) decisive predictor from the type only | -0.007 | 0.368 |  | -0.0051 | -0.0026 | 48% | 40 |
| A | single-shot | (control) type posterior shuffled over images | +0.010 | 0.092 |  | -0.0004 | +0.0050 | 62% | 40 |
| A | single-shot | (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.014 | 0.206 |  | +0.0076 | +0.0145 | 62% | 40 |
| A | sequential | Type-aware graph coverage x P(decisive) | +0.009 | 0.183 | 0.183 | +0.0033 | +0.0085 | 60% | 40 |
| A | sequential | (control) decisive predictor from the type only | -0.015 | 0.048 |  | -0.0073 | -0.0037 | 38% | 40 |
| A | sequential | (control) type posterior shuffled over images | -0.010 | 0.192 |  | -0.0026 | +0.0041 | 40% | 40 |
| A | sequential | (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.017 | 0.068 |  | +0.0093 | +0.0156 | 65% | 40 |
| B | single-shot | Type-aware graph coverage x P(decisive) | +0.029 | <0.001 | <0.001 | +0.0038 | +0.0076 | 70% | 40 |
| B | single-shot | (control) decisive predictor from the type only | +0.019 | 0.019 |  | -0.0040 | +0.0008 | 72% | 40 |
| B | single-shot | (control) type posterior shuffled over images | +0.024 | 0.019 |  | +0.0000 | +0.0044 | 62% | 40 |
| B | single-shot | (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.001 | 0.452 |  | +0.0018 | +0.0056 | 57% | 40 |
| B | sequential | Type-aware graph coverage x P(decisive) | +0.023 | 0.031 | 0.031 | +0.0025 | +0.0072 | 65% | 40 |
| B | sequential | (control) decisive predictor from the type only | +0.018 | 0.013 |  | -0.0023 | +0.0005 | 68% | 40 |
| B | sequential | (control) type posterior shuffled over images | +0.027 | 0.002 |  | +0.0004 | +0.0038 | 72% | 40 |
| B | sequential | (reference) pool-wide I-optimal design, no graph (vopt_u) | -0.007 | 0.921 |  | +0.0035 | +0.0173 | 57% | 40 |

| Split | Condition | Paired difference | log-loss | raw p | AUC | raw p | accuracy | raw p |
|---|---|---|---:|---:|---:|---:|---:|---:|
| A | single-shot | Type-aware graph coverage x P(decisive) minus (control) decisive predictor from the type only | +0.0300 | 0.001 | +0.0127 | <0.001 | +0.0158 | 0.001 |
| A | single-shot | Type-aware graph coverage x P(decisive) minus (control) type posterior shuffled over images | +0.0133 | 0.183 | +0.0080 | 0.017 | +0.0082 | 0.060 |
| A | single-shot | Type-aware graph coverage x P(decisive) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.0088 | 0.320 | +0.0000 | 0.375 | -0.0014 | 0.762 |
| A | sequential | Type-aware graph coverage x P(decisive) minus (control) decisive predictor from the type only | +0.0241 | 0.004 | +0.0105 | 0.001 | +0.0122 | 0.005 |
| A | sequential | Type-aware graph coverage x P(decisive) minus (control) type posterior shuffled over images | +0.0183 | 0.032 | +0.0058 | 0.037 | +0.0045 | 0.227 |
| A | sequential | Type-aware graph coverage x P(decisive) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | -0.0086 | 0.745 | -0.0060 | 0.397 | -0.0071 | 0.232 |
| B | single-shot | Type-aware graph coverage x P(decisive) minus (control) decisive predictor from the type only | +0.0107 | 0.237 | +0.0078 | 0.014 | +0.0068 | 0.141 |
| B | single-shot | Type-aware graph coverage x P(decisive) minus (control) type posterior shuffled over images | +0.0052 | 0.618 | +0.0038 | 0.354 | +0.0032 | 0.497 |
| B | single-shot | Type-aware graph coverage x P(decisive) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.0281 | 0.037 | +0.0020 | 0.705 | +0.0019 | 0.558 |
| B | sequential | Type-aware graph coverage x P(decisive) minus (control) decisive predictor from the type only | +0.0049 | 0.675 | +0.0048 | 0.216 | +0.0067 | 0.307 |
| B | sequential | Type-aware graph coverage x P(decisive) minus (control) type posterior shuffled over images | -0.0049 | 0.795 | +0.0021 | 0.581 | +0.0034 | 0.503 |
| B | sequential | Type-aware graph coverage x P(decisive) minus (reference) pool-wide I-optimal design, no graph (vopt_u) | +0.0293 | 0.079 | -0.0010 | 0.952 | -0.0100 | 0.077 |

The rule has a positive log-loss gain in all four blocks (+0.023, +0.009, +0.029, +0.023; raw p 0.008, 0.18, 0.001, 0.031; Holm over the four blocks 0.025, 0.18, 0.004, 0.062), so it is significant after correction in two of the four blocks (A single-shot and B single-shot), with positive AUC (+0.002 to +0.008) and accuracy (+0.008 to +0.013) gains throughout. The controls separate the graph from the type: the type-only decisive predictor is at −0.007 and −0.015 in split A and +0.019 and +0.018 in split B, and the rule is above it in AUC in all four blocks (+0.013, +0.011, +0.008, +0.005; raw p < 0.001, 0.001, 0.014, 0.22) and in log-loss in split A (+0.030, +0.024; raw p 0.001, 0.004). Against the shuffled-posterior control the AUC differences are +0.008, +0.006, +0.004 and +0.002 (raw p 0.017, 0.037, 0.35, 0.58) and the log-loss differences +0.013, +0.018, +0.005 and −0.005. In split B the gain over Random is the same for the real and the shuffled posterior (+0.029 against +0.024 single-shot; +0.023 against +0.027 sequential), so in split B the gain is not attributable to the graph; in split A it is, partly. `vopt_u` was at +0.014, +0.017, +0.001 and −0.007 on these seeds, so the typed rule is above it in split B (+0.028 and +0.029, raw p 0.037 and 0.079) and not different in split A, which shows how strongly the gain of the non-graph design varies between seed sets (its gains over the three fresh sets of this section (Parts 1, 3 and 6) range from −0.007 to +0.058).

Pooling all fresh confirmatory seed sets run under the re-tuned schedule (Parts 1, 3 and 6; 105 seeds; a post-hoc summary, not a pre-registered test), the rule beats Random in all four blocks:

```
block             n  log-loss   raw p   Holm4      AUC      acc  better | minus typeonly (n) LL / AUC p | minus shuffled LL / AUC p | minus vopt_u (n) LL p
A-single      105    +0.022  <0.001  <0.001  +0.0073  +0.0088     66% | +0.024/+0.0125 p <0.001/<0.001 (n=70) | +0.010/+0.0077 p 0.143/0.006 | +0.009 (n=105) p 0.240
A-sequential  105    +0.014   0.007   0.007  +0.0044  +0.0090     62% | +0.025/+0.0122 p <0.001/<0.001 (n=70) | +0.016/+0.0068 p 0.013/0.009 | -0.005 (n=105) p 0.568
B-single      105    +0.037  <0.001  <0.001  +0.0088  +0.0092     76% | +0.015/+0.0096 p 0.099/0.001 (n=70) | +0.008/+0.0050 p 0.255/0.048 | +0.012 (n=105) p 0.182
B-sequential  105    +0.032  <0.001  <0.001  +0.0071  +0.0084     68% | +0.015/+0.0090 p 0.084/0.003 (n=70) | +0.007/+0.0061 p 0.288/0.039 | +0.014 (n=105) p 0.275
```

(Columns: pooled log-loss gain over Random, raw p and Holm over four blocks, AUC and accuracy gain, share of seeds better; then the paired difference to the type-only control and to the shuffled-posterior control, both from the 70 seeds of Parts 1 and 6, as log-loss and AUC differences with raw p-values; then the difference to `vopt_u` over all 105 seeds.) The pooled gains are +0.014 to +0.037 in log-loss, with Holm p < 0.01 in each block. The type posterior adds to the type-only predictor in AUC in every block (+0.009 to +0.013, p ≤ 0.003) and in log-loss in split A (+0.024, +0.025, p < 0.001), and adds to the shuffled posterior in AUC (+0.005 to +0.008, raw p 0.006 to 0.048); the log-loss differences to the shuffled posterior are not significant in three of four blocks. These are the strongest signs in this report that the graph-derived type information carries something beyond the type itself, and they are small (AUC +0.005 to +0.013) and uncorrected for multiplicity.


**Summary of the graph work (§5.16–5.20).** (i) The comparison graph is nearly a matching, so a ranking network on it has nothing to recover. (ii) With the old head schedule no graph rule beat Random reproducibly (§5.16–5.18). (iii) With the re-tuned schedule the frozen type-aware graph rules beat Random in 4 of 4 blocks on 30 fresh seeds and in the two split-B blocks on 35 further seeds; the type posterior adds AUC over a type-only decisive predictor in Part 1, but not distinguishably from the non-graph variance-reduction rule `vopt_u`. (iv) With the gentler schedule the graph-regularised variance-reduction design beats Random in 4 of 4 blocks on two independent sets of 35 and 40 seeds, but its shuffled-graph control and `vopt_u` do the same, so there is no evidence that the graph is responsible for the gain. (v) Selection of a method on 25 development seeds repeatedly overstated its effect (the Laplacian prior in split A, Part 3), so only the pre-registered confirmations should be read as evidence. **Limitations:** one encoder (frozen SimCLR), the same 168 pair groups in every seed (so each run tests sensitivity to the split draw, not new data, and Wilcoxon p-values are optimistic), three head schedules (old, re-tuned, gentle) with conclusions that depend on the schedule, modest effects (log-loss +0.02 to +0.05, AUC +0.005 to +0.019), paired differences to controls that are not corrected for multiplicity, and a graph built from the same frozen features as the head, which limits how much new information it can add.

