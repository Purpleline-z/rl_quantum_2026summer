# Literature round for the autonomous run (2026-10-09)

Sources actually read (search summaries unless marked "code"); arxiv.org / ar5iv / openaccess.thecvf.com are blocked by the network proxy, so paper text could not be fetched; GitHub raw files could.

## Sequential GCN for Active Learning (Caramalau et al., CVPR 2021, arXiv 2006.10219)
Verified from the authors' code (github.com/razvancaramalau/Sequential-GCN-for-Active-Learning, `selection_methods.py`, `config.py`, `models/query_models.py`):
- Nodes = the candidate pool subset (10,000 of the pool, sampled per cycle) plus the labelled set; features = learner features, L2-normalised.
- Adjacency: S = X X^T (cosine), A = D^-1 (S - I) + I (row-normalised, self-loops added back). Dense, not a kNN graph.
- GCN: gc1 (nfeat -> nhid) + ReLU + dropout, then gc3 (nhid -> 1) + sigmoid (a third layer exists but is not used in forward). Targets: labelled = 1, pool = 0.
- Loss (BCEAdjLoss): -mean log s(labelled) - lambda * mean log(1 - s(pool)), lambda = 1.0, 200 full-batch steps, lr 1e-3, weight decay 5e-4.
- UncertainGCN: score = |s - s_margin|; the code sorts by -score and returns the full order (the caller takes a slice; the file does not show which end). Intuition in the paper (via search summaries): select unlabelled nodes the GCN is least sure are labelled-like, i.e. far from the labelled set.
- CoreGCN: k-centre greedy on the hidden layer (after gc1 + ReLU) of the GCN, centres = labelled nodes.
- Selection runs on image classification with CNN features; budgets of 1000 per cycle. Not a pairwise-preference setting.

## Pairwise-preference active learning (search summaries)
- Setwise active learning for relative attributes (Liang & Grauman, CVPR 2014): diversity via k-means clusters + low rank margins to avoid unreliable comparisons.
- Active Preference Learning for Ordering Items In- and Out-of-sample, arXiv 2405.03059: uncertainty-aware (aleatoric + epistemic) greedy selection bounding expected ordering error; BALD as a baseline; repeated queries of high-uncertainty pairs can help.
- Deep Bayesian Active Learning for Preference Modeling in LLMs (NeurIPS 2024): BALD plus an entropy term over the acquired batch for diversity; 33%-68% less feedback than random on text preferences.
- Active ranking / Hodge-based methods (earlier notes): graph connectivity of the comparison graph, not applicable to a near-matching.

## Low-budget / cold-start selection (search summaries)
- TypiClust (Hacohen et al., ICML 2022): dense, typical points, one per cluster, from self-supervised features; large gains over random at ~10 labels on CIFAR-10. ProbCover (Yehuda et al., NeurIPS 2022). Both are density/coverage rules, not uncertainty.
- Cold-start with label propagation + density sampling (arXiv 2201.10227): density-based sampling suits a label-propagation model; its cluster-based variants showed no significant gain over random on that benchmark.
- Patron (arXiv 2209.06995): kNN graph to control spacing between selected samples; uncertainty propagation over the graph; best overall vs cold-start baselines.
- Graph-based AL with label propagation (Long et al., 2008): expected entropy reduction under label propagation.

## What we take for this run
1. Implement CoreGCN / UncertainGCN faithfully on the image graph as the "graph literature baseline" (dense normalised cosine adjacency, labelled-vs-pool BCE, k-centre on the first hidden layer).
2. Use typed reference (ideal) images as extra nodes for a type-posterior label propagation (graph-based, like Long et al. / label propagation for cold start), to make selection type-aware.
3. Combine coverage (what works in split B), predicted-decisive weighting (bald_decisive) and uncertainty; coverage on graph-smoothed features (Patron-like spacing).
