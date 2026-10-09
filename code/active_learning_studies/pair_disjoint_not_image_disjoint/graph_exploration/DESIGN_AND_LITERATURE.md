# Design of the type-aware graph acquisition rules, results so far, and how they relate to the literature

Status 2026-10-09 night: the development table in section 2 is superseded by the confirmation (25 unseen seeds) and the replication (30 fresh seeds, split B) in the report section 5.15 and in AUTONOMOUS_RUN_LOG.md: the pre-registered criterion was not met, the plain coverage rule has a small gain in split B (about +0.03 to +0.06 log-loss) that its type-only and shuffled controls reproduce, and split A shows no gain; treat the section-2 numbers as optimistic development estimates. Sources are cited only where a search result stated them; "(summary)" means only a search summary was read, not the paper.

## 1. What the method does (graph_typed.py)
Setting: each query is one (image pair, reconstruction type) judgment; the annotator may answer decisively (a wins / b wins), "tie", or "not_apply" (35% of judgments; 64% for HTR). The evaluated endpoint (held-out preference log-loss / AUC) only scores decisive judgments, while every outcome is used in training.

1. **Image graph (label-free).** Nodes = images of the candidate and revealed judgments plus the typed reference (ideal) images; edges = symmetrised cosine 10-nearest neighbours on mean-centred cached SimCLR features.
2. **Type posterior by label spreading.** The one-hot type of the reference images is propagated over the graph, F = (I - 0.9 S)^-1 Y with S the symmetric-normalised adjacency, then normalised per image: q_i(t). This is the label-spreading algorithm of Zhou et al., "Learning with Local and Global Consistency" (NeurIPS 2003) (summary).
3. **Per-judgment features** (8 numbers for pair (a, b) and queried type t): q_a(t), q_b(t), |q_a(t)-q_b(t)|, q_a(t)+q_b(t), min, max, ||q_a-q_b||_1, q_a . q_b.
4. **P(decisive).** Ridge logistic regression on the outcomes of the judgments revealed so far (type one-hot + the 8 features); 0.5 until both outcome classes were seen. A data-level diagnostic on all 521 judgments (5 folds by pair group, 10 repeats; design-informing, uses all labels) gave AUC for decisive 0.83 with these features versus 0.60 for type only and 0.80 for 1024-d mean/|a-b| features; for not_apply 0.90 versus 0.72 and 0.75.
5. **Selection.** Greedy farthest-first (k-centre) coverage in the space of SGC-propagated pair features [(a+b)/2, |a-b|, a*b] plus type one-hot, each candidate's distance multiplied by P(decisive) (`typed_decisive_coverage`) and additionally by (0.25 + own-head uncertainty) (`typed_decisive_coverage_unc`, the primary candidate). `typed_decisive_bald` multiplies the Laplace-BALD score by P(decisive).
6. **Controls.** `*_typeonly`: P(decisive) from the type one-hot only (no graph features); `*_shuffled`: the type posterior of every image replaced by that of a random other image. The difference between a rule and these controls is the contribution of the graph.

## 2. Results so far (DEV seeds 400-409, log-loss gain over Random; blocks A-single, A-seq, B-single, B-seq)
| Rule | A-single | A-seq | B-single | B-seq |
|---|---:|---:|---:|---:|
| typed_decisive_coverage_unc | +0.108 | +0.138 | +0.029 | +0.091 |
| ... type-only control | +0.119 | +0.056 | -0.017 | +0.056 |
| ... shuffled-posterior control | +0.103 | +0.096 | -0.043 | +0.022 |
| typed_decisive_coverage | +0.095 | +0.109 | +0.024 | +0.105 |
| typed_decisive_bald | +0.100 | +0.068 | -0.021 | +0.078 |
| Literature graph baselines: coregcn / uncertaingcn | +0.010 / +0.050 | +0.011 / +0.036 | -0.006 / -0.022 | +0.034 / +0.014 |
| Embedding baselines: core_set_relation / bald_decisive | +0.041 / -0.003 | +0.058 / +0.001 | -0.036 / +0.007 | +0.062 / +0.078 |
Reading (hedged, 10 seeds, per-seed sd of the gain 0.09-0.14): (i) most of the log-loss gain is already obtained by weighting with a **type-only** P(decisive) - the graph's own increment is about +0.03 to +0.08 log-loss in three of the four blocks and +0.01 to +0.03 AUC in all four; (ii) the labelled-vs-pool GCN rules of Caramalau et al. are not better than the embedding baselines here; (iii) decisive-weighted random sampling is worse than coverage (B-single -0.06 to -0.10), so coverage is needed.

## 3. Relation to the literature
| Component of our method | Closest published idea (as far as searched) | Difference / what to claim |
|---|---|---|
| Weight informativeness by the probability that the query is answerable | Active learning from an oracle with a knowledge blind spot (AAAI; summary): estimate the probability that an instance lies in the oracle's blind spot and combine it with a mutual-information criterion; Active Learning with Oracle Epiphany (Huang et al., NeurIPS 2016; summary): abstentions (I-don't-know) are modelled explicitly | Our "not_apply" and "tie" outcomes play the role of abstention; the new part is that the answerability is predicted per (pair, type) from a graph-propagated type posterior and that the endpoint counts only decisive outcomes. Do not claim the idea of answerability-weighting as new |
| Uncertainty x density | Settles & Craven (EMNLP 2008) density-weighted uncertainty (secondary summaries) | Our coverage x P(decisive) x uncertainty is a multiplicative combination of the same kind; graph-based propagated features define the density/coverage space |
| Coverage first, uncertainty later at small budgets | TypiClust (Hacohen et al., ICML 2022), Active Learning on a Budget: Opposite Strategies Suit High and Low Budgets; Doucet et al. hybrid TypiClust-then-Margin; CSAL-3D, CSCS pacing (summaries) | We use coverage with an uncertainty factor from the start; a pacing schedule between the two is untested |
| Graph over a feature space for selection | Sequential GCN (Caramalau et al., CVPR 2021; read in the authors' code), FeatProp, GRAIN, Patron (kNN graph for spacing, summary), Long et al. 2008 (label propagation) | Our graph is used for a type posterior and propagated pair features, not for a labelled-vs-pool discriminator; the discriminator version was implemented (coregcn/uncertaingcn) and did not help here |
| Preference acquisition | Active Reward Modeling (Shen et al., ICML 2025; Fisher information on the last layer, summary), Active Preference Learning for LLMs (arXiv 2402.08114; entropy + certainty), BALD-type rules | Laplace BALD x P(decisive) is our graph-informed version of this family; ties are not modelled in the acquisition of those papers (searched: none found) |
| Ties in preference models | Reward Learning From Preference With Ties (arXiv 2410.05328, summary): Bradley-Terry with ties | Our loss treats ties with an absolute-difference penalty; a tie-aware information gain is not implemented |
| Type-conditioned / multi-attribute preferences | RLHFlow multi-objective reward modelling (summary) | The search found no paper that chooses which attribute to query in active reward modelling; this is the angle the advisor suggested (type-aware acquisition); claim only "not found in this search" |

## 4. What would make the claim solid
- Confirmation on seeds not used for design (running); the report must say the gain is mostly type-aware answerability weighting and quantify the graph increment against the type-only and shuffled controls.
- A tie/abstention-aware expected-information formulation (outcome probabilities from the graph features, information value of each outcome) instead of the product heuristic.
- A second encoder (GPU task) and a held-out dataset or session, since all data come from one set of three trajectory sessions.
