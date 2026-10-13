# Literature notes: does selection still matter on a manifold-regularised learner, and what about a rare type (HTR)? (2026-10-13)

How this was gathered: WebSearch only. arXiv / ar5iv / CVF pages are blocked, so nothing here comes from reading a paper's full text.
Every item is "(abstract/snippet)" = I saw only a search-result snippet or summary (often a third-party aggregator such as Lacuna / liner / emergentmind, which I flag).
"UNVERIFIED" = I could not confirm the point. Authors / venue / year are given only where a search result showed them.
Statements labelled "Translation" are my own inference, not search-confirmed.

Already covered elsewhere and NOT repeated in detail (see code/active_learning_studies/new_directions/literature/*.md,
pair_disjoint_not_image_disjoint/notes/literature_and_new_strategy_ideas.md, graph_exploration/notes_overnight2_literature.md,
notes_graph_literature_round.md, notes_graph_learner_literature.md):
Munjal et al. CVPR 2022; Mittal et al. "Parting with Illusions" (arXiv 1912.05361); Simeoni et al. (arXiv 1911.08177); Luth et al. NeurIPS 2023;
Zhu-Lafferty-Ghahramani 2003; He 2010 Laplacian D-optimal; Chen et al. AAAI 2010 G-optimal; Cai & He 2012; Ji & Han 2012; Ma-Garnett-Schneider 2013 Sigma-optimality;
Long et al. 2008 label-propagation AL; TypiClust; ProbCover; TCM; Patron; Cold-start LP + density (arXiv 2201.10227); Cluster-Margin (Citovsky et al. 2021);
Active Reward Modeling (Shen, Sun, Ton; ICML 2025); Ioannidis et al. IJCAI 2018 BT design; AMPLe; Biyik et al. batch preference AL; Settles & Craven density-weighting
(mentioned in DESIGN_AND_LITERATURE.md and the survey is cited in lit_extra.md); Aggarwal et al. WACV 2020 and Ertekin et al. CIKM 2007 (named, one line, in
notes/literature_and_new_strategy_ideas.md "Sixth round"); Donmez & Carbonell (lit_extra.md). Topics 3 and 8 are therefore short.

---------------------------------------------------------------------------------------------------------------------------------

## 1. Active learning on top of semi-supervised / manifold-regularised learners

Already in the earlier notes: Mittal et al. 2019 (AL's advantage shrinks to near zero with strong augmentation + SSL), Munjal et al. 2022, Simeoni et al. 2019, Zhu et al. 2003.
New items:

1. **Reducing Label Effort: Self-Supervised meets Active Learning.** Bengar, van de Weijer, Twardowski, Raducanu. ICCV 2021 Workshops, pp. 1631-1639; arXiv 2108.11458. (abstract/snippet)
   Tests on CIFAR10, CIFAR100, Tiny ImageNet at several labelling budgets. The snippet reports that self-training cuts labelling need more than active learning does and that at low budgets active learning
   adds nothing on top of self-training; the snippet was cut off before the combined result, so the final claim about the combination is UNVERIFIED.
   Translation: the closest "does AL still help once the learner uses unlabelled data" result; it points the same way as our worry (the Laplacian learner may absorb most of the gain), but it is classification with CNNs, not preference pairs.

2. **Semi-Supervised Active Learning for Semi-Supervised Models: Exploit Adversarial Examples with Graph-Based Virtual Labels.** Guo, Shi, Kang, Kuang, Tang, Jiang, Sun, Wu, Zhuang. ICCV 2021, pp. 2896-2905. (abstract/snippet)
   Graph-based label propagation provides virtual labels for unlabelled samples; active learning picks samples near cluster boundaries (via adversarial examples) so that propagation works better; the authors claim better results than prior methods on classification and segmentation.
   (One aggregator, Lacuna, mislabels it as CVPR 2021 / "REVIVAL"; I trust the ICCV listing.)
   Translation: an explicit case where the acquisition rule is designed for the propagation learner (boundary-seeking), not borrowed from a supervised learner.

3. **Active Self-Semi-Supervised Learning for Few Labeled Samples (AS3L).** First author Ziting Wen (+2 authors, names not shown in my results); arXiv 2203.04560, later a Neurocomputing listing (pii S0925231224015431). (abstract/snippet)
   Prior pseudo-labels from label propagation over self-supervised features bootstrap a semi-supervised model; an active strategy picks which samples to label; the authors note that prior-pseudo-label accuracy depends on both feature quality and WHICH samples are labelled; reported average gain 5.4% over baseline on four image datasets (author-reported).
   Translation: the sentence "pseudo-label quality depends on which samples are labelled" is the mechanism by which selection could still matter on a graph learner; it is a claim from the authors, not an independent test.

4. **Label Propagation for Deep Semi-supervised Learning.** Iscen et al., CVPR 2019 (arXiv 1904.04717). (abstract/snippet) kNN graph on the network's own embeddings, propagate labels, retrain on pseudo-labels; gains are larger when labels are scarce. (Also in notes_graph_learner_literature.md as a learner reference; listed here only for the "scarce labels" claim.)

5. **How To Overcome Confirmation Bias in Semi-Supervised Image Classification By Active Learning** (arXiv 2308.08224; authors not shown in my results). (abstract/snippet)
   Reports that random sampling does not mitigate confirmation bias and in some cases is worse than supervised learning, and argues active selection can help; it also says that choosing the best AL method for a real scenario is not trivial.
   Translation: a counter-voice to item 1; evidence on "AL vs SSL" is mixed.

6. **Analysis of active semi-supervised learning** (ACM SAC 2023; authors and findings not visible) and **Combination of Active Learning and Semi-Supervised Learning under a Self-Training Scheme** (MDPI Entropy 21(10):988, 2019; reports beating supervised, SSL and AL baselines on 55 UCI datasets, author-reported). (abstract/snippet) Titles only confirmed; UNVERIFIED beyond that.

Not found: a controlled study that isolates the acquisition effect on top of a Laplacian / graph-regularised learner at ~10-60 labels (search summary said the same for deep SSL).
Settles survey (Active Learning Literature Survey, Computer Sciences TR 1648, UW-Madison, 2009): existence and the "density-weighted methods" section confirmed from search results; I did not see the section text. (abstract/snippet)

---------------------------------------------------------------------------------------------------------------------------------

## 2. Density-weighted / representativeness-aware uncertainty sampling; graph-density selection

1. **An Analysis of Active Learning Strategies for Sequence Labeling Tasks.** Settles & Craven. EMNLP 2008, pp. 1070-1079. (abstract/snippet + third-party slides/summary)
   Information density (ID) weights an uncertainty score by the instance's average similarity to the rest of the unlabelled pool, with an exponent controlling density vs informativeness. Motivation quoted in the full-text snippet: uncertainty sampling and query-by-committee can pick outliers that are informative to the learner in isolation but say little about the rest of the distribution.
   Secondary summaries (Lacuna, slides) say ID gave the highest average performance across tasks; I did not verify against the paper.
   Translation: a multiplicative density x uncertainty score is the standard fix for "picks outliers"; on a graph learner, density could be the kNN-graph degree or mean cosine similarity of the two images.

2. **Active Learning for Graph Embedding (AGE).** Cai, Zheng, Chang; arXiv 1705.05085 (2017). (abstract/snippet + secondary)
   Score = weighted sum of entropy, density (distance to a k-means cluster centre in embedding space) and PageRank centrality percentiles; time-sensitive weights shift emphasis from graph criteria to embedding criteria as learning proceeds. Reported gains over random on Citeseer, Cora, Pubmed (author-reported; I did not see the full numbers beyond MacroF1 +4.9 to +7.6 points).
   Critique seen in a related-work passage of the FeatProp paper: Cai et al. and Gao et al. only combine scores by a linear weighted sum, and centrality ignores node features.
   Translation: AGE's "graph score early, model score late" schedule is a pre-specifiable alternative to TCM-style switching; but its centrality term makes little sense for a near-matching pair graph.

3. **Active Learning for Graph Neural Networks via Node Feature Propagation (FeatProp).** Yuexin Wu et al.; NeurIPS 2019 Graph Representation Learning Workshop; arXiv 1910.07567. (abstract/snippet)
   Propagate node features through the graph, then K-Medoids on the propagated features to choose nodes; theory relates classification loss to the geometry of propagated features; the authors argue this is less sensitive to a poorly trained model.
   Translation: selecting on graph-smoothed features (the same idea as the "coverage on graph-smoothed features" in notes_graph_literature_round.md) needs no model uncertainty at all, which matters at 10 labels.

4. **Hierarchical Sampling for Active Learning.** Dasgupta & Hsu. ICML 2008, pp. 208-215. (abstract/snippet)
   Works on a hierarchical clustering tree; keeps a pruning of the tree and samples from the least pure nodes (third-party summary), with valid confidence intervals regardless of how the tree was built. Dasgupta's "Two faces of active learning" says cluster-then-label-one-per-cluster has no clean characterisation of when it beats random.
   Translation: purity-driven cluster sampling is a "boundary-in-the-graph" idea that needs no learner at all.

---------------------------------------------------------------------------------------------------------------------------------

## 3. Laplacian-regularised optimal design / graph experimental design (only what is new)

All main papers (He 2010; Chen et al. 2010; Cai & He 2012; Ji & Han 2012; Ma et al. 2013; Zhu et al. 2003) are in notes_overnight2_literature.md. New from this round:

1. **Uncertainty for Active Learning on Graphs.** Fuchsgruber, Wollschlager, Charpentier, Oroz, Gunnemann. ICML 2024 (PMLR 235, pp. 14275-14307); arXiv 2405.01462. (abstract/snippet)
   Benchmarks uncertainty estimators against traditional AL for node classification; finds uncertainty sampling (and many other approaches) fall short of random; derives ground-truth uncertainty from the data-generating process and argues only EPISTEMIC uncertainty sampling aligns with AL; proposes Expected Single Pseudo-Label (ESP). A secondary summary (liner.com) quotes CoraML numbers (ESP 71.45 vs 68.16 best other vs 62.51 random AUC); I could not verify them.
   Another search summary of the same benchmark: except GEEM (only tractable for SGC), traditional AL does not significantly outperform random on graphs.
   Translation: the node-classification version of "random is hard to beat" on a graph learner; and the epistemic/aleatoric split matters for our tie / not_apply-heavy labels.

2. **Open-World Graph Active Learning for Node Classification** (ACM TKDD; authors not shown). (abstract/snippet) Uses label propagation to find nodes far from labelled ones in topology space; the snippet says this discovers novel-class nodes AND boundary nodes.

3. A snippet from an OpenReview paper ("A new way to graph active learning with soft ..."; authors/venue UNVERIFIED) argues uncertainty sampling cannot guarantee the largest entropy reduction on a single node and proposes selecting by aggregated neighbourhood uncertainty.

Not found (again): graph-regularised design combined with a Bradley-Terry head (same gap as in notes_overnight2_literature.md).

---------------------------------------------------------------------------------------------------------------------------------

## 4. Class-priority / rare-class-aware / cost-sensitive active learning

1. **Active Learning for Imbalanced Datasets.** Aggarwal, Popescu, Hudelot. WACV 2020, pp. 1417-1426. (abstract/snippet)
   Two adaptations: acquisition functions using a model pretrained on a source domain, and a balancing step in acquisition; works "as long as knowledge from the source domain is transferable". A secondary summary (Lacuna) says the balancing step picks unlabelled samples geometrically close to the rarest classes in the labelled set, and that entropy AL can be worse than random under imbalance (not verified).
   Same authors: **Minority Class Oriented Active Learning for Imbalanced Datasets**, ICPR 2021 (title only seen).

2. **Class-Balanced Active Learning for Image Classification.** Bengar, van de Weijer, Lopez-Fuentes, Raducanu. WACV 2022; arXiv 2110.04543 (page numbers conflict between sources). (abstract/snippet)
   General optimisation framework that adds class balance to existing AL rules (informative and representative); Entropy-CB gains over 1% over Entropy after 4 cycles regardless of imbalance factor (author-reported).
   Translation: class balance as a constraint on top of the base score, i.e. a quota, not a new score.

3. **Algorithm Selection for Deep Active Learning with Imbalanced Datasets (TAILOR).** Zhang, Shao, Verma, Nowak. NeurIPS 2023 (one source shows an ICML 2023 DMLR header); arXiv 2302.07317. (abstract/snippet)
   Bandit over candidate AL algorithms with reward functions that favour class-balanced AND informative examples; comparable or better than the best single baseline on 9 of 10 datasets.

4. **Improving class and group imbalanced classification with uncertainty-based active learning.** Tifrea, Hill, Yang. NeurIPS 2023 Workshop ReALML. (abstract/snippet)
   Uncertainty-based AL often does not beat passive learning on average accuracy, but is competitive on WORST-subpopulation accuracy under heavy imbalance and beats reweighting methods designed for that goal (author-reported; workshop paper).
   Translation: average-metric and rare-group-metric comparisons of the same selector can disagree; report both.

5. **Learning on the Border: Active Learning in Imbalanced Data Classification.** Ertekin, Huang, Bottou, Giles. CIKM 2007, pp. 127-136. (abstract/snippet) Active learning gives the learner more balanced classes by selecting near-border instances from a small pool (SVM). Already named in notes/literature_and_new_strategy_ideas.md.

6. **Rare Class Discovery Based on Active Learning (ALICE / MALICE).** He & Carbonell. ISAIM 2008 (builds on their NIPS 2007 NNDM paper). (abstract/snippet)
   Local-topology (nearest-neighbour) density estimation finds rare-class examples even when overlapping the majority class; MALICE excludes unlabelled points within a class-specific radius of a newly labelled one. Reported on a synthetic set: ~4 labels vs ~68 random vs ~382 Interleave; on an image set 49 vs 112 vs 662 for five rare classes (excerpt, author-reported).
   He & Carbonell note that locating members of an already-identified class is unusual (closest thing to "targeted"); I did not find a paper named "Targeted active learning" (searched; none returned).
   Also: **Finding Rare Classes: Active Learning with Generative and Discriminative Models** (Hospedales, Gong, Xiang; IEEE TKDE 2011) and **Active Rare Class Discovery and Classification Using Dirichlet Processes** (Haines & Xiang; IJCV, doi 10.1007/s11263-013-0630-3): they note discovery and classification have CONFLICTING query criteria. (abstract/snippet)
   Translation: all of this is rare-class DISCOVERY; in our setting the rare type (HTR) is known and the question is evaluation/learning precision for it, so the discovery papers are only loosely relevant.

7. **Active Cost-Sensitive Learning.** Margineantu, IJCAI 2005. (abstract/snippet) Fixed K x K loss matrix; queries the instance with largest expected gain, trading expected misclassification loss against labelling cost, so costly-to-misclassify classes get more weight. **Active Learning for Cost-Sensitive Classification** (Krishnamurthy et al., JMLR 2019, arXiv 1703.01014; COAL): regress costs per label and query only labels that could be best. (abstract/snippet)
   Translation: a per-type weight w_HTR on the acquisition score is the simplest cost-sensitive reading; neither paper is about pairwise preferences.

8. Other results, titles/snippets only (UNVERIFIED beyond that): Rarity-Aware Stratified Active Learning for Class-Imbalanced Industrial Object Detection (MDPI Applied Sciences 16(3):1236, 2026; adds a stratified querying step to force controlled rare-class inclusion); Dynamic Class-Aware Active Learning for Unbiased Satellite Image Segmentation (arXiv 2604.08965; per-class performance adapts sampling weights each cycle); When Imbalance Comes Twice (arXiv 2601.06209; a simulation study reporting entropy and core-set remain efficient even for highly imbalanced data - a dissenting result).

Searched but no hit: active learning for PAIRWISE comparison data with class/type imbalance. Results were only active-ranking papers (Jamieson & Nowak; Heckel et al.; Ailon 2012) without any imbalance aspect.

---------------------------------------------------------------------------------------------------------------------------------

## 5. Preference / comparison AL with several query types or attributes (only new items)

Already covered: Shen et al. (ARM), Ioannidis et al., AMPLe (Oh et al., ACL 2025), Biyik et al., Liang & Grauman 2014, BALD-style, Deep Bayesian AL for preference (NeurIPS 2024).
New facts from this round:
1. **Comparison-based Active Preference Learning for Multi-dimensional Personalization (AMPLe).** ACL 2025 long paper (aclanthology 2025.acl-long.1590); arXiv 2411.00524; code github.com/ml-postech/AMPLe. (abstract/snippet) Multi-dimensional reward = per-attribute reward models; attributes e.g. harmlessness / helpfulness / humor; query chosen to halve the preference space ("vol"); the vol-mo variant is reported to recover the user profile with limited feedback, up to four attributes (author-reported). Closest in structure to "one reward head per reconstruction type", but the unit being learned is a user mixing weights over attributes, not per-attribute heads from scratch.
2. **Active Preference-Based Learning of Reward Functions** (Sadigh et al., RSS 2017; PDF seen at Berkeley) and **Batch Active Preference-Based Learning of Reward Functions** (Biyik & Sadigh, CoRL 2018, PMLR 87:519-528): the search says the CoRL paper does NOT contain the DPP method; DPPs came in Biyik, Wang, Anari, Sadigh, arXiv 1906.07975 and the 2024 ACM THRI journal version (arXiv 2402.15757). (abstract/snippet) A previous note in the repo cites DPPs with Biyik 2019, which matches.
3. **Active preference-based Gaussian process regression for reward learning and optimization** (Biyik, Huynh, Kochenderfer, Sadigh; IJRR, SAGE 2024). (title/snippet only)
4. **Preference Elicitation for Multi-objective Combinatorial Optimization with Active Learning and Maximum Likelihood Estimation** (arXiv 2503.11435). (title only)
5. ARM result details seen this round (ICML 2025, "Active Reward Modeling: Adaptive Preference Labeling for LLM Alignment"): D-optimal on last-layer Fisher information favours pairs with moderate reward differences and diverse embeddings; reported about 1,000 selected comparisons matching a 20,000-comparison random baseline; ablation suggests cross-prompt pairing helps (author-reported).
Not found: any paper allocating a labelling budget across several preference types / attribute heads with unequal importance.

---------------------------------------------------------------------------------------------------------------------------------

## 6. Stratified / quota-based budget allocation across groups

1. **Active Testing: Sample-Efficient Model Evaluation.** Kossen, Farquhar, Gal, Rainforth. ICML 2021 (PMLR 139:5753-5763); arXiv 2103.05331; code github.com/jlko/active-testing. (abstract/snippet)
   Chooses which TEST points to label; corrects selection bias with an importance-sampling estimator (LURE); naive active selection gives biased error estimates; efficiency gains ~4x on Fashion-MNIST/CIFAR-10, ~2x on CIFAR-100 (author-reported).
   Translation: if rare-type test pairs are chosen or filtered non-uniformly, an inverse-selection-probability weight is needed to keep the HTR test estimate unbiased.
2. **Stratified random sampling for neural network test input selection (SSOA).** Wu, Wang, Chen, You, Yan (author list from the ScienceDirect contribution section; journal not confirmed; pii S0950584923001866). (abstract/snippet)
   Strata from model confidence, per-stratum accuracy variance estimated, budget split by optimum (Neyman-type) allocation; authors report >26.14% lower MSE of the accuracy estimate. They flag that the per-stratum variance cannot be observed without labels (the usual obstacle).
3. **Active Sampling for Min-Max Fairness** (arXiv 2006.06879; authors not shown). (abstract/snippet) Each step samples from the currently worst-off group; convergence proved for convex problems; can be applied (no guarantee) to neural nets.
4. Fair Robust Active Learning by Joint Inconsistency (JIN; picks the worst validation group, ranks within it; assumes sensitive attributes known for the pool; via Lacuna summary only) and Fairness Without Harm (influence-guided; arXiv 2402.12789): titles/snippets only.
5. Tifrea et al. (section 4, item 4) is the group-imbalance case with uncertainty selection.
6. **Two faces of active learning** (Dasgupta; PDF seen). Cluster, query one label per cluster, treat it as representative; no clean characterisation of when it beats random.
Not found: a paper on explicit quotas per group inside deep AL for preference data.

---------------------------------------------------------------------------------------------------------------------------------

## 7. Boundary / "bridge" selection in graph label propagation

1. **Graph-Based Active Learning Based on Label Propagation.** Long, Yin, Zhao, Zhu. MDAI 2008 (Springer LNCS 5285). (abstract/snippet) Trains a label-propagation classifier, queries the instance with the largest expected entropy reduction. Already in earlier notes. I found NO source using a margin score on propagated values specifically (search summary said the same).
2. scikit-learn example "Label Propagation digits: Active learning" (docs 1.8.0). (snippet) Starts with 10 labelled points and repeatedly adds the five most uncertain by LabelSpreading; a practical, entropy-style baseline.
3. Open-World Graph AL (section 3, item 2): propagation-distance finds boundary nodes as well as novel-class nodes (snippet).
4. Guo et al. ICCV 2021 (section 1, item 2): boundary-seeking selection so propagation is better.
5. Dasgupta & Hsu 2008 (section 2, item 4): sample impure clusters, i.e. the places where the label map changes.
6. Fuchsgruber et al. ICML 2024 (section 3): plain uncertainty on graphs does not beat random; epistemic part is what aligns.
Not found: a "bridge node between two classes" criterion (betweenness / bottleneck) for label propagation.

---------------------------------------------------------------------------------------------------------------------------------

## 8. Cold-start AL (only what is not already in earlier notes)

TypiClust, ProbCover, TCM, Patron, k-means baseline (Pourahmadi), Foundation-model clustering init (arXiv 2402.02561), PEAL, and the foundation-model-era revisit (arXiv 2401.14555) are already covered. Only additions:
- Dasgupta & Hsu 2008 / "Two faces" (section 2, 6) as the older cluster-then-label foundation.
- AS3L (section 1): self-supervised features -> propagated pseudo-labels, with selection chosen to improve them (snippet).
- Bengar et al. ICCVW 2021 (section 1): at low budgets AL adds nothing over self-training (snippet).

---------------------------------------------------------------------------------------------------------------------------------

## What this implies for design (author's inference, NOT search-confirmed)

1. The literature on "AL on top of SSL" (Bengar 2021, Mittal 2019, Munjal 2022) leans toward "little residual benefit" at low budgets; the only controlled-looking contrary claims are author-reported (Guo 2021, AS3L 2022, 2308.08224). Expect small average gains from selection on the Laplacian learner and pre-register that expectation.
2. If selection is to help a graph learner, the candidates most supported by the snippets are (a) density x uncertainty (Settles-Craven style) with density taken from the same kNN graph, (b) boundary/impure-cluster seeking (Guo 2021; Dasgupta-Hsu), and (c) epistemic rather than total uncertainty (Fuchsgruber 2024). Coverage-only rules need no model and are safest at 10 labels.
3. For HTR, treat the rare-type objective as a separate metric and, if testable, a quota or per-type weight on top of the base score (class-balanced framing of Bengar 2022 / Aggarwal 2020), not as a new acquisition score; rare-class DISCOVERY papers (He & Carbonell) address a different problem.
4. Report average and HTR-only metrics side by side: Tifrea et al. suggest a selector can look neutral on average and different on a rare subgroup.
5. With few HTR reference/test pairs, the HTR estimate is high-variance; any non-uniform choice of test or reference pairs should carry inverse-probability weights (Active Testing), and per-type budgets might use stratified allocation (SSOA), with the caveat that per-stratum variance is unknown without labels.
6. Graph-score-then-model-score schedules (AGE) and TCM-style switching are pre-specifiable; a bandit over rules (TAILOR, ALBL) is probably too noisy at 10-60 judgments.
7. Because the labelled pair graph is nearly a matching, centrality or connectivity terms (AGE) are unlikely to carry signal; use the image-similarity graph only.
8. Gaps found in searching (so claims of novelty are unverified): no graph-regularised design with a Bradley-Terry head, no pairwise-preference AL with per-type importance or imbalance, no margin-on-propagated-score acquisition.
