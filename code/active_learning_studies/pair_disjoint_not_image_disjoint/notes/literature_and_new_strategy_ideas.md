# Literature notes and new acquisition ideas (written during the unattended session)

Searched on 2026-10-08. Only what a search result stated is attributed; interpretation is marked as ours.

## Sources and what we take from them

| Source | Idea | How we use it |
|---|---|---|
| [BADGE, Ash et al., ICLR 2020](https://arxiv.org/abs/1906.03671) | Gradient embedding of the loss under the model's own prediction, then k-means++ seeding; no uncertainty/diversity trade-off weight | `badge_pairs`: gradient of the Bradley–Terry loss w.r.t. the head's last layer, with the model's predicted preference as the hallucinated label |
| [TypiClust, Hacohen et al., ICML 2022](https://arxiv.org/abs/2202.02794) | At low budgets uncertainty fails; pick typical (dense) points, one per cluster, from self-supervised features | `typiclust_pairs`: cluster pair vectors, take the densest pair in the largest uncovered cluster. Directly relevant: our budgets are 10–100 pairs |
| [ASAP, Mikhailiuk et al.](https://www.cl.cam.ac.uk/~rkm38/pdfs/mikhailiuk2020asap.pdf) | Information gain for pairwise comparisons; batches built as a spanning tree over the comparison graph | Motivates treating images as graph nodes and pairs as edges (`image_coverage_uncertainty`) |
| [Just Sort It!, Maystre & Grossglauser, ICML 2017](https://proceedings.mlr.press/v70/maystre17a/maystre17a.pdf) | Simple, effective active preference learning | Reference point for sequential/graph-aware pair choice |
| [Active Learning with Label Comparisons, 2022](https://arxiv.org/abs/2204.04670) | A label-neighbourhood graph: comparing neighbouring classes is sufficient | Suggests the HTR/√13 boundary (see §5.4 of the report) is where comparisons carry the most information |
| [Graph-based active learning with label propagation, Long et al., 2008](https://link.springer.com/chapter/10.1007/978-3-540-88269-5_17) | Entropy reduction under label propagation on a graph | Motivates `graph_facility_location` (submodular coverage on a kNN graph of pairs) |

D-optimal design: see the fourth round (Active Reward Modeling, ICML 2025), which the search did return; `fisher_dopt` follows it.

## Representation of a pair (the "mean" limitation)
The original core-set uses (a+b)/2. That is symmetric, but it throws away how the two images relate: two pairs with the same mean can be
"two similar images" or "two very different images", and only the latter says much about a decision boundary. `core_set_relation` and
`graph_facility_location` use [(a+b)/2, |a-b|, a*b] (z-scored, plus a one-hot of the reconstruction type).

## New strategies (all in new_pair_strategies.py)
- `core_set_relation`: k-center in the relation-aware pair space.
- `typiclust_pairs`: TypiClust on pair vectors.
- `badge_pairs`: BADGE gradient embeddings for a Bradley–Terry head.
- `fisher_dopt`: greedy log-det (D-optimal) on the head's last-layer Fisher information, per reconstruction-type head.
- `image_coverage_uncertainty`: graph view with images as nodes; uncertain pairs touching images that have no labelled edge are boosted.
- `graph_facility_location`: kNN graph over pairs; greedy uncertainty-weighted facility location.

## Second round of sources and strategies
| Source | Idea | Strategy |
|---|---|---|
| [Batch AL with DPPs, Bıyık et al., 2019](https://arxiv.org/abs/1906.07975) | Kernel with diagonal = quality (uncertainty), off-diagonal = similarity; repulsive batches | `dpp_pairs` (greedy MAP via Schur complements) |
| [BatchBALD, Kirsch et al., 2019](https://arxiv.org/abs/1906.08158) | Joint mutual information of a batch; independent top-k picks redundant points | Motivates the batch-aware update inside `laplace_bald` |
| [ProbCover, Yehuda et al., NeurIPS 2022](https://arxiv.org/abs/2205.11320) | δ-ball coverage of self-supervised features, low-budget regime | `probcover_pairs` (δ = largest radius with ≥90% purity in the observable reconstruction type) |
| [Generalized coverage / MaxHerding, ECCV 2024](https://arxiv.org/html/2407.12212v2) | ProbCover is sensitive to its radius; smooth kernel coverage is more robust | `maxherding_pairs` (kernel coverage on the pair graph) |
| [BALD, Houlsby et al., 2011](https://arxiv.org/abs/1112.5745) and [GP preference learning, Chu & Ghahramani, ICML 2005](https://icml.cc/Conferences/2005/proceedings/papers/018_Preference_ChuGhahramani.pdf) | Information gain about the latent preference function | `laplace_bald`: BALD under a Laplace posterior on the linear last layer, with batch fantasy updates |
| [Batch active learning of reward functions from human preferences, Bıyık et al.](https://liralab.usc.edu/pdfs/publications/biyik2024batch.pdf) | Batch preference queries | Closest prior art for the whole task; read in full before writing related work |

No RHEED-specific active learning or contrastive-pretraining paper turned up in the search; the nearest are the Peak Sequence Transformer (already used in `paper_replicate`) and CNN pattern classifiers.

## What the pair data looks like as a graph (analyze_pair_graph.py, seed-independent)
- 168 usable pair groups (edges) over 284 images (nodes); 237 images occur in exactly one group, 42 in two, 5 in three.
- 117 connected components, the largest with 11 images: the graph is almost a perfect matching, so "cover more images" cannot separate
  strategies much (`image_coverage_uncertainty` is expected to behave close to plain uncertainty). Graph structure over *images* is not informative;
  structure over *pairs* (similarity graph in feature space) is what `graph_facility_location` and `maxherding_pairs` use.
- A pair group holds 1–4 judgments, one per reconstruction type (103 of 168 have all four types). Outcomes: 125 "1", 139 "2", 184 "not_apply", 73 "tie".
  Only 264 of 521 rows are decisive preferences; 35% are "not_apply".
- 38% of pairs have both images in the same trajectory folder (same day/run).

## Implementation observation (existing strategies)
`Experiment.candidates_with_clusters` attaches `type_idx = groups[pair_id].iloc[0].type_idx`, the type of the *first label row* of the group, and the
original uncertainty/MC-dropout rules score only that head. Because a group usually has several types, this (a) reads label information during selection and
(b) ignores most of the group's judgments. The new selectors aggregate over the four active heads and do not read `type_idx` (tested).

## Third round: what the literature says about frozen features and tiny budgets
- [Revisiting Active Learning in the Era of Vision Foundation Models, 2024](https://arxiv.org/html/2401.14555v2): on frozen DINOv2 features with a linear classifier, uncertainty sampling is competitive from the first iteration, and in ultra-low-budget settings starting from representative samples (centroids) improved accuracy by more than 20% over random; the proposed DropQuery combines centroid initialisation with uncertainty-style querying. -> `dropquery_pairs`.
- [Parameter-Efficient Active Learning for Foundational Models (PEAL), 2024](https://arxiv.org/html/2406.09296v1): linear-probing active learning with Entropy and feature-distance selection underperformed random sampling, because frozen features make distance-based diversity less effective. This is the same pattern we see (uncertainty not beating random with a head on frozen features) and a caution that "frozen" can remove the signal diversity rules rely on.
- [Foundation Model Makes Clustering A Better Initialization For Cold-Start Active Learning, 2024](https://arxiv.org/abs/2402.02561): the initial labelled set matters; random initial sets fluctuate. Our initial 10 groups are a random type-covering draw, so initial-set selection is an open lever (not yet tested).
- [Bridging Diversity and Uncertainty in Active Learning with Self-Supervised Pre-Training, 2024](https://arxiv.org/html/2403.03728v1): names the cold-start weakness of uncertainty-based methods.
- Annotation cost: [Settles et al.](https://burrsettles.com/pub/settles.nips08ws.pdf) report that dividing utility by predicted annotation cost did not beat random on several tasks; the search found nothing on oracles that abstain ("not_apply", 35% of our rows) in active learning, so an abstention-aware selector is untested territory.
- Ties: [Rao & Kupper (1967)](https://www.tandfonline.com/doi/abs/10.1080/01621459.1967.10482901) and Davidson (1970) model ties inside Bradley–Terry; our loss treats ties with an absolute-difference penalty instead. No source linked tie-aware models to active pair selection.

## Fourth round: preference-specific selection and a caution about uncertainty
- [ActiveUltraFeedback, 2026](https://arxiv.org/html/2603.09692v2): dueling-bandit selectors transfer poorly to preference data generation; the proposed Double Reverse Thompson Sampling and DeltaUCB prefer pairs with a *large* predicted quality gap because they give a lower-noise training signal than ambiguous comparisons. (Experiments use an LLM judge; transfer to human labels is flagged as open.) -> `delta_gap`, `delta_ucb`. The abstract reports comparable or better results with as little as one-sixth of the annotations; we have not reproduced that.
- [Active Reward Modeling, Shen, Sun & Ton, ICML 2025 (arXiv 2502.04354)](https://arxiv.org/abs/2502.04354): selects comparisons by D-optimality of the Bradley–Terry Fisher information, treating the penultimate-layer output as the input of a linear model; each comparison contributes the covariance of embedding differences times p(1-p). Reported to beat six other selectors (entropy sampling is the baseline) on 1-Spearman error and best-of-N reward with lower variance across seeds, on LLM reward models. A past-aware variant conditions on already-collected data. Our `fisher_dopt` implements the past-aware D-optimal rule on the 256-d hidden layer of the reward head, one information matrix per active head. The paper's own caveat (via a third-party summary): the theory is exact only for a linear reward.
- Statistical RLHF survey, [arXiv 2604.02507](https://arxiv.org/html/2604.02507v1): preference collection links to experimental design; uncertainty estimates matter for active querying.
- [Batch Active Learning at Scale (Cluster-Margin), Citovsky et al., NeurIPS 2021](https://arxiv.org/pdf/2107.14263): margin pre-filter, then round-robin over clusters; the abstract reports needing about 40% of the labels of the next best method on its benchmarks. Our `cluster_margin_pairwise` follows it with 20 flat k-means clusters rather than hierarchical agglomerative clustering, a simplification worth stating in the paper.
- [Active Learning with Imperfect Labels, 2025](https://arxiv.org/html/2512.12870): uncertainty-based selection picks samples more likely to receive noisy labels; diversity-only selection without noise awareness degrades under high noise. Relevant because 49% of our rows are tie/not_apply.

## Can the outcome of a judgment be predicted from the images? (explore_predict_decisive.py)
Group-wise (pair-level) 5-fold cross-validated AUC on all 521 rows, logistic regression, frozen SimCLR features:

| Target | type only | type + distance + cosine | type + mean + abs-difference (1024-d) |
|---|---:|---:|---:|
| decisive (winner 1/2) | 0.60 | 0.70 | 0.79 |
| not_apply | 0.72 | 0.74 | 0.75 |
| tie | 0.65 | 0.84 | 0.87 |

Outcome mix: 51% decisive, 14% tie, 35% not_apply. HTR judgments are 64% not_apply and c(6x2) 46%; (1x1) and (√13x√13) are mostly decisive or tie.
The relation features (|a-b|) carry the signal, which supports putting pair relations into the representation. `bald_decisive` multiplies BALD by a predicted
probability of a decisive answer fitted on the labelled judgments only (so it is label-honest, but with 10 initial groups the fit is crude).
Related work found: [tie-aware DPO](https://arxiv.org/html/2409.17431v1) (Rao–Kupper/Davidson inside the loss), a [2026 preprint on active query synthesis](https://arxiv.org/html/2605.26072v1)
with a confidence-aware response model (pairs of nearly identical or entirely dissimilar items give ambiguous answers), and cold-start work
([TypiClust](https://arxiv.org/abs/2202.02794), [k-means baseline](https://arxiv.org/pdf/2110.12033)) showing representative seeds beat random at tiny budgets.

## Mechanism checks on the first five seeds (single-shot, analyze_selected_pairs.py and analyze_coverage_mechanism.py)
- Uncertainty-type selectors do favour ambiguous comparisons: tie share 0.24–0.28 (random 0.14), decisive share 0.34–0.44 (random 0.51). Large-gap selectors (`delta_gap`, `delta_ucb`) pick 0.64–0.65 decisive.
- But the decisive share does not predict held-out log-loss across cells (correlation -0.07), and the best five-seed strategy (Cluster-Margin) has a *low* decisive share (0.44, tie 0.20).
- Coverage does not explain it either: the mean distance from held-out pairs to the nearest selected pair (relation-aware pair space) correlates 0.055 with log-loss within (seed, budget).
- So the five-seed ranking has no mechanistic story yet; the extension to 30 more seeds tests whether the ranking itself is real.

## Fifth round: submodular and optimal-design foundations
- [Experimental Design under the Bradley–Terry Model, Guo et al., IJCAI 2018](https://www.ijcai.org/proceedings/2018/0304.pdf): treats pair selection as active learning and evaluates mutual information, entropy, D-optimal (covariance) and Fisher-information objectives, proving they are submodular so greedy gives near-optimal batches. This is the theory behind `fisher_dopt` and `laplace_bald` (both greedy with batch-aware updates).
- [Submodularity in Data Subset Selection and Active Learning, Wei et al., ICML 2015](https://proceedings.mlr.press/v37/wei15.pdf): greedy 1-1/e guarantee; combines informativeness and representativeness; its FASS filters by uncertainty then covers the filtered set. -> `fass_pairs`; `graphcut_pairs` implements the graph-cut objective.
- Facility location as low-budget selection with no labels needed (Kaushal et al., WACV 2019, via search summary; not opened). Our `maxherding_pairs` and `graph_facility_location` are of this family.
- No source compared pair-feature constructions (concatenate / difference / product) for active selection; our relation-aware representation is therefore tested only empirically (`core_set` vs `core_set_relation`).

## Cold start: which 10 groups to start from (run_initial_set_study.py, 35 seeds, held-out pair endpoint)
**Correction.** A first version of this section, written from five seeds, claimed that the default initial set (random draw with greedy reconstruction-type coverage)
was clearly better than label-free image-space selection and explained it by per-head type coverage. That did not replicate over 35 seeds and is withdrawn.

Mean over 35 seeds; "added" = random pool groups added after the 10 initial ones.

| Initial set | AUC (0 added) | log-loss (0 / 30 added) | calibrated log-loss (0 / 30 added) | accuracy (0 added) |
|---|---:|---:|---:|---:|
| default (random draw, greedy type coverage) | 0.872 | 0.715 / 0.429 | 0.482 / 0.403 | 0.813 |
| farthest-first in relation-aware pair space | 0.884 | 0.567 / 0.410 | 0.488 / 0.466 | 0.818 |
| k-means representatives | 0.882 | 0.599 / 0.399 | 0.503 / 0.392 | 0.805 |
| random | 0.867 | 0.745 / 0.405 | 0.518 / 0.416 | 0.799 |
| TypiClust on pair vectors | 0.873 | 0.656 / 0.411 | 0.506 / 0.416 | 0.804 |

Paired over seeds, the default set's AUC minus the others' (0 added): farthest -0.011 (p = 0.25), k-means -0.009 (p = 0.50), random +0.005 (p = 1.0), TypiClust -0.001 (p = 0.81).
Label-free representative sets give a lower raw log-loss with no initial acquisition (farthest-first 0.567 vs 0.715), but the calibrated log-loss is about the same for all
sets, so the raw difference is mostly score scale. After 30 added groups all initial sets are within noise. Conclusion: with this endpoint and frozen features the first
10 groups do not matter much; the type-coverage story is not supported.

## Sixth round: coverage across reward heads, domain context
- Class coverage in active learning: [Active Learning for Imbalanced Datasets, Aggarwal et al., WACV 2020](https://openaccess.thecvf.com/content_WACV_2020/papers/Aggarwal_Active_Learning_for_Imbalanced_Datasets_WACV_2020_paper.pdf), [Learning on the Border, Ertekin et al.](https://clgiles.ist.psu.edu/pubs/CIKM-2007-learning-border.pdf) and VaB-AL address skewed class coverage in the acquisition step. Our setting is per-*head* rather than per-class coverage (each reconstruction type has its own reward head, and a pair group is judged for several types); the initial-set study above found no clear advantage for type-covering initial sets over label-free ones over 35 seeds, so this is hypothesis, not finding. No source treated multiple per-class reward heads; this is our own framing.
- [Batch Active Learning of Reward Functions from Human Preferences, Bıyık et al.](https://arxiv.org/html/2402.15757v1): the batch preference-query setting; closest prior art for the single-shot batch condition.
- [Active Query Selection for Crowd-Based RL, 2025](https://arxiv.org/html/2508.19132) and Crowd-BT-style reliability weighting: judge reliability could matter if labels came from several people.
- Domain: [RHEED pattern classification with a CNN](https://www.researchgate.net/publication/401711251_RHEED_pattern_classification_by_a_convolutional_neural_network_for_the_growth_of_chalcogenide_thin_films_and_nanostructures), [machine-learning on-the-fly RHEED analysis, JVST A 43 (2025)](https://pubs.aip.org/avs/jva/article/43/3/032702/3341018/Machine-learning-enabled-on-the-fly-analysis-of-RHEED-patterns-during-thin-film-deposition-by-molecular-beam-epitaxy), and a [Nano Letters study](https://pubs.acs.org/doi/10.1021/acs.nanolett.4c04500) that extracts RHEED features from ~10 expert-labelled examples. The search found no RHEED work on active selection of preference labels, so the question studied here appears unaddressed in that literature (search-based statement, not an exhaustive survey).
