### 5.14 Graph-aware acquisition on an image-similarity graph

**Question.** Pair groups share images and expert comparisons implicitly order the images, so a graph over images might carry information that the mean-pair embedding of the earlier strategies discards. We asked whether a graph-aware selection rule improves held-out preference prediction over Random and over the embedding-based rules.

**What the comparison graph looks like.** Taking images as nodes and the 168 usable pair groups as edges, the graph has 284 nodes and 117 connected components (largest 11 nodes); 237 images occur in exactly one pair group, 42 in two and 5 in three. Per reconstruction type the directed graph of decisive judgments (loser to winner) has 45–94 edges, connected components of at most 4–5 nodes, and only 0–6 images that both won and lost a comparison. The comparison graph is therefore nearly a matching, and a model that recovers a global ranking from it (such as GNNRank, He et al., ICML 2022) has no transitive structure to recover. (The working slides give 669 pairs and about 300 images; the data in this repository contain 638 judgment rows, 168 pair groups and 284 images.) The graph used below is instead built from image similarity.

**Graph and strategies.** At every selection step the graph has as nodes the images of the candidate judgments and of the judgments already revealed, and as edges the symmetrised ten nearest neighbours in the cached SimCLR feature space. It uses no labels and cannot contain a validation or test image, because a selector sees only these images. Three rules were implemented in `graph_strategies.py`: (i) own-head uncertainty multiplied by the percentile rank of the PageRank of the pair's two images, (ii) own-head uncertainty multiplied by a boundary score (the share of an image's neighbours that fall in another k-means cluster of eight), and (iii) farthest-first k-centre on [(a+b)/2, |a−b|, a·b] of features propagated twice over the graph (SGC). Each has a control in which the graph scores, or the propagated features, are permuted over the nodes. Reconstruction-type information (the ideal images as typed anchors) is not available to the selectors in this implementation, so the boundary score is an approximation of the cross-type boundary seen in the ideal images (112 of 770 five-nearest-neighbour edges among ideal images join different types, 52 of them HTR–RT13).

**Protocol.** The query unit is one (pair, type) judgment and the budget counts judgments (10, 20, 40, 60), as in the judgment-unit study; numbers are therefore not comparable with the group-budget tables of §5.12–5.13. Splits A and B, seeds, head schedule and endpoint are those of §5.12–5.13 (35 seeds, single-shot and sequential rounds of 10). The candidate pool holds 201–250 judgments in split A and 311–350 in split B. Gains are per-seed differences from the mean of five Random draws, averaged over budgets; the test is a Wilcoxon signed-rank test over seeds with Holm correction over the seven non-random rows of each block. There are eight blocks (two splits, two conditions, two metrics) and no correction across them.

**Diagnostic before the experiment.** As a check on whether graph smoothing changes what the features can predict, a linear Bradley–Terry head on frozen features was fit to the 264 decisive judgments (five folds by pair group, 20 repetitions). Raw features reach AUC 0.941; features smoothed over the similarity graph (1–4 steps, with or without a within-session temporal chain) reach 0.938–0.940, and a spectral embedding 0.909. This does not test a trained graph network or an acquisition rule.

| Split | Condition | Strategy | log-loss gain | Holm p | AUC gain | Holm p | seeds better (log-loss) |
|---|---|---|---:|---:|---:|---:|---:|
| A | single-shot | Uncertainty x PageRank of the pair's images | +0.034 | 0.106 | +0.006 | 0.588 | 77% (35) |
| A | single-shot | Core-set on propagated features, graph shuffled | +0.028 | 1.000 | +0.003 | 1.000 | 66% (35) |
| A | single-shot | Uncertainty (no graph) | +0.012 | 1.000 | +0.002 | 1.000 | 54% (35) |
| A | single-shot | Uncertainty x boundary score | +0.009 | 1.000 | -0.003 | 1.000 | 57% (35) |
| A | single-shot | Uncertainty x boundary score, graph shuffled | +0.007 | 1.000 | +0.004 | 1.000 | 51% (35) |
| A | single-shot | Core-set on graph-propagated features | +0.003 | 1.000 | -0.005 | 1.000 | 46% (35) |
| A | single-shot | Uncertainty x PageRank, graph shuffled | -0.030 | 1.000 | +0.000 | 1.000 | 46% (35) |
| A | sequential | Uncertainty x PageRank of the pair's images | +0.048 | 0.083 | +0.011 | 0.087 | 66% (35) |
| A | sequential | Core-set on graph-propagated features | +0.026 | 1.000 | +0.000 | 1.000 | 57% (35) |
| A | sequential | Core-set on propagated features, graph shuffled | +0.025 | 1.000 | +0.001 | 1.000 | 54% (35) |
| A | sequential | Uncertainty x boundary score | +0.002 | 1.000 | -0.002 | 1.000 | 60% (35) |
| A | sequential | Uncertainty (no graph) | -0.007 | 1.000 | +0.004 | 1.000 | 46% (35) |
| A | sequential | Uncertainty x boundary score, graph shuffled | -0.013 | 1.000 | +0.004 | 1.000 | 46% (35) |
| A | sequential | Uncertainty x PageRank, graph shuffled | -0.033 | 1.000 | +0.000 | 1.000 | 37% (35) |
| B | single-shot | Core-set on propagated features, graph shuffled | +0.012 | 1.000 | -0.001 | 0.883 | 51% (35) |
| B | single-shot | Core-set on graph-propagated features | +0.004 | 1.000 | -0.003 | 0.883 | 57% (35) |
| B | single-shot | Uncertainty x PageRank of the pair's images | -0.021 | 0.505 | -0.005 | 0.475 | 43% (35) |
| B | single-shot | Uncertainty x PageRank, graph shuffled | -0.045 | 0.018 | -0.014 | 0.019 | 29% (35) |
| B | single-shot | Uncertainty (no graph) | -0.047 | 0.055 | -0.013 | 0.065 | 31% (35) |
| B | single-shot | Uncertainty x boundary score, graph shuffled | -0.053 | 0.025 | -0.008 | 0.280 | 34% (35) |
| B | single-shot | Uncertainty x boundary score | -0.066 | 0.007 | -0.017 | 0.004 | 26% (35) |
| B | sequential | Core-set on propagated features, graph shuffled | +0.056 | 0.013 | +0.005 | 0.803 | 71% (35) |
| B | sequential | Core-set on graph-propagated features | +0.046 | 0.188 | +0.005 | 0.894 | 60% (35) |
| B | sequential | Uncertainty x PageRank, graph shuffled | +0.033 | 0.614 | +0.003 | 1.000 | 57% (35) |
| B | sequential | Uncertainty x PageRank of the pair's images | +0.032 | 0.228 | +0.005 | 0.894 | 74% (35) |
| B | sequential | Uncertainty (no graph) | +0.024 | 0.715 | +0.003 | 1.000 | 57% (35) |
| B | sequential | Uncertainty x boundary score, graph shuffled | -0.004 | 1.000 | -0.002 | 1.000 | 54% (35) |
| B | sequential | Uncertainty x boundary score | -0.017 | 1.000 | -0.009 | 0.812 | 49% (35) |

**Results.** No graph rule is significantly better than Random after Holm correction in any block. The one significant gain in the table belongs to a control: the shuffled-graph core-set in split B sequential (log-loss +0.056, Holm p = 0.013), which shows that the coverage gain of that rule does not come from the graph. Uncertainty multiplied by PageRank is the only rule that is nominally positive in split A in both conditions (log-loss +0.034 single-shot and +0.048 sequential, raw p of about 0.012–0.015 but Holm p of 0.11 and 0.08; AUC +0.006 and +0.011); it is not positive in split B single-shot (−0.021) and its sequential gain there (+0.032, Holm p = 0.23) is matched by its shuffled control (+0.033). The boundary-score rule is never better than Random and is significantly worse in split B single-shot (log-loss −0.066, Holm p = 0.007; AUC −0.017, Holm p = 0.004); uncertainty without a graph is also below Random there (−0.047, Holm p = 0.055), so the boundary score does not repair the weakness of uncertainty sampling in that split. Core-set on propagated features does not differ from its shuffled control in either split.

Comparison with the shuffled controls (paired over seeds, unadjusted p-values, twelve comparisons per metric):

| Split | Condition | Graph rule | log-loss: real − shuffled | p | AUC: real − shuffled | p |
|---|---|---|---:|---:|---:|---:|
| A | single-shot | Uncertainty x PageRank of the pair's images | +0.064 | 0.037 | +0.0058 | 0.081 |
| A | single-shot | Uncertainty x boundary score | +0.002 | 0.865 | -0.0072 | 0.168 |
| A | single-shot | Core-set on graph-propagated features | -0.026 | 0.404 | -0.0077 | 0.225 |
| A | sequential | Uncertainty x PageRank of the pair's images | +0.081 | <0.001 | +0.0111 | 0.031 |
| A | sequential | Uncertainty x boundary score | +0.016 | 0.752 | -0.0056 | 0.287 |
| A | sequential | Core-set on graph-propagated features | +0.001 | 0.929 | -0.0002 | 0.968 |
| B | single-shot | Uncertainty x PageRank of the pair's images | +0.024 | 0.168 | +0.0088 | 0.135 |
| B | single-shot | Uncertainty x boundary score | -0.013 | 0.512 | -0.0095 | 0.073 |
| B | single-shot | Core-set on graph-propagated features | -0.008 | 0.554 | -0.0016 | 0.884 |
| B | sequential | Uncertainty x PageRank of the pair's images | -0.001 | 0.840 | +0.0016 | 0.617 |
| B | sequential | Uncertainty x boundary score | -0.013 | 0.692 | -0.0069 | 0.310 |
| B | sequential | Core-set on graph-propagated features | -0.010 | 0.252 | -0.0001 | 0.777 |

The PageRank rule is better than its shuffled control in split A (log-loss +0.064 single-shot, p = 0.037; +0.081 sequential, p < 0.001), but the shuffled control is itself worse than Random there (−0.030 and −0.033), so part of this difference reflects the control rather than the graph, and the difference is absent in split B.

**What this does and does not show.** Under these splits and with this encoder, selection on a ten-nearest-neighbour similarity graph is not better than Random or than the rules of §5.12–5.13, and the one nominally positive rule does not replicate across splits. The experiment does not test a trained graph network, a graph built from all 1,124 trajectory images (840 of them unlabelled), ideal images as typed anchors, or temporal edges within a session (the temporal constraints file marks these as tentative); the effect of an informative graph could be larger than that of these simple versions. The tests are low-powered for effects of the size seen elsewhere in this report (log-loss gains of 0.03–0.05 against a per-seed standard deviation of 0.09–0.14).

