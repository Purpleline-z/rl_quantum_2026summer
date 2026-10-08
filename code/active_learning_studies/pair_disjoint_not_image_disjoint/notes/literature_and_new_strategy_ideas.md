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

D-optimal design: the search did not return a source for batch D-optimal design in the Bradley–Terry model, so `fisher_dopt` rests on standard
experimental-design reasoning (the Fisher information of a Bradley–Terry comparison is p(1-p) phi phi^T) and has not been checked against a primary paper.

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
