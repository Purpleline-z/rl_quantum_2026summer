# Literature notes: graph-augmented learners (branch claude/graph-learner-20261012)

Scope: Manifold regularisation, label propagation / spreading, GCN / SGC, semi-supervised ranking and preference learning with unlabelled items, negative results and controls. Items already covered in `new_directions/literature/` (ArmoRM, Laplace Redux, GP preference learning, BALD, time-contrastive learning, DAFT / FiLM, SSRM / FixMatch pseudo-labelling, mixup, active learning by learning, TypiClust, ...) are not repeated.

Verification status (important): the pages could not be opened (arXiv, ar5iv and CVF are blocked by the proxy; a subagent used web search only). Every entry below is confirmed only through search-result titles, authors, venues and URLs and by the abstract-level summaries shown there; "(abstract)" marks such statements. Items marked UNVERIFIED must be checked before citing. The raw subagent output follows unchanged after the design notes.

## How the literature shapes the design (my reading, not claims of the papers)
- Manifold regularisation / neural graph learning / semi-supervised embedding (Belkin et al. 2006; Bui et al. 2018; Weston et al. 2008) justify variant 1: a neighbour-similarity penalty on the head's outputs over ALL nodes. Szummer & Yilmaz (CIKM 2011) is the closest ranking precedent (preference loss + regulariser making similar items similar in preference, including unlabelled items). I found no published Laplacian-regularised Bradley-Terry model with item features: unpublished or UNVERIFIED.
- Nadler, Srebro & Zhou (NeurIPS 2009) warn that Laplacian regularisation degenerates when labels are few and unlabelled data many: relevant here (about 170 labelled groups, 1.2k nodes), a reason to keep lambda small and to read a null result as plausible.
- Label spreading (Zhou et al. 2003), Poisson learning (Calder et al. 2020) and Iscen et al. (CVPR 2019, propagated pseudo-labels from a kNN graph of embeddings) are the precedents for variant 2; it differs from SSRM / FixMatch self-training because the target comes from the reference labels diffused over the graph, not from the model's own predictions.
- GCN (Kipf & Welling 2017) and SGC (Wu et al. 2019) motivate variant 3; SGC on frozen features is a parameter-free smoothing followed by the head. Over-smoothing papers (Li et al. 2018; Chen et al. 2020) limit it to 1-2 hops. PMLP (Yang et al., ICLR 2023: train as an MLP, use message passing only at test time), Graph-MLP (graph only as a training loss) and "All we have is low-pass filters" (features already informative, the graph mainly denoises) motivate the "no-edge" and "no-graph regulariser" controls. I found no canonical shuffled-graph control paper: the shuffled-graph control is my own design.
- Negative-result literature: Oliver et al. (NeurIPS 2018, realistic evaluation of semi-supervised learning) and Tian et al. (ECCV 2020, a good fixed embedding is enough) say that a well-tuned supervised baseline is hard to beat with unlabelled data; our baseline `aw8_all` is such a baseline.

---

# Raw literature file (subagent output)

# Literature: graph / manifold learning on frozen features for a Bradley-Terry head

Method note: all entries were found with WebSearch (standard mode). WebFetch was not used, so nothing was read in full text. Every claim below comes from search-result summaries (abstract-level). "(abstract)" marks claims taken only from an abstract or summary. Page numbers differ between indexes for several ICML papers and are omitted unless stated. Nothing already in the project literature files is repeated. Each URL appeared in a search result.

## T1. Manifold / Laplacian regularisation

1. **Manifold Regularization: A Geometric Framework for Learning from Labeled and Unlabeled Examples.** Belkin, Niyogi, Sindhwani. JMLR 7:2399-2434, 2006. https://jmlr.org/papers/v7/belkin06a.html (PDF: https://www.jmlr.org/papers/volume7/belkin06a/belkin06a.pdf)
   - Search summary (abstract): adds a data-geometry regulariser built on labelled and unlabelled points; RLS and SVM are special cases (LapRLS / LapSVM); Representer theorems; works both transductively and inductively. Caveat stated: it inverts a dense Gram matrix, cubic in n.
   - Application: the formal basis for idea (1). Penalise f^T L f over all 1278 nodes alongside the BT loss. At n≈1.3k the cubic cost does not matter.
   - Not confirmed from search results: any LapRLS-specific experiments.

2. **Neural Graph Learning: Training Neural Networks Using Graphs** (arXiv title: "Neural Graph Machines: Learning Neural Networks Using Graphs"). Bui, Ravi, Ramavajjala. WSDM 2018 (arXiv 1703.04818, 2017). https://dl.acm.org/doi/10.1145/3159652.3159731 ; https://arxiv.org/abs/1703.04818
   - Search summary (abstract): graph-regularised objective for neural nets, trainable by SGD, runtime linear in edges; reported to beat existing methods on several tasks, with and without node features.
   - Application: the closest template for putting a neighbour-similarity penalty on a neural head's outputs on unlabelled nodes. It acts on network outputs or embeddings, not on a fixed Laplacian solve.

3. **Deep Learning via Semi-Supervised Embedding.** Weston, Ratle, Collobert. ICML 2008 (ACM lists 3 authors; Semantic Scholar BibTeX adds Mobahi). https://dl.acm.org/doi/10.1145/1390156.1390303 ; PDF https://ronan.collobert.com/pub/2008_deep_icml.pdf
   - Search summary (abstract): embedding-based semi-supervised regulariser applied at the output layer or at every layer of a deep net; results competitive with shallow and deep SSL methods.
   - Application: supports a Laplacian-style pairwise penalty on the head's output (a scalar score here) as well as on hidden layers.

4. Supporting, for the "fixed-label-count, many-unlabelled" regime (also T2): **Statistical Analysis of Semi-Supervised Learning: The Limit of Infinite Unlabelled Data.** Nadler, Srebro, Zhou. NeurIPS 2009 (22), pp. 1330-1338. https://proceedings.neurips.cc/paper/2009/hash/68ce199ec2c5517597ce0a4d89620f55-Abstract.html
   - Search summary (abstract): Laplacian regularisation with a fixed number of labels and growing unlabelled data is ill-posed for d>=2 and degenerates to a near-constant function with spikes at labelled points.
   - Application: a caution for idea (1). With ~170 labelled groups and 1278 nodes the effect is a degeneracy risk, not a theorem that applies directly. Fix proposed by Zhou & Belkin 2011 (iterated Laplacian; AISTATS, https://proceedings.mlr.press/v15/zhou11b.html; title/authors from search snippet only, UNVERIFIED beyond the URL).

Also seen (survey): Zhu, **Semi-Supervised Learning Literature Survey**, TR 1530, U. Wisconsin-Madison, 2005 (updated to 2008). https://pages.cs.wisc.edu/~jerryzhu/pub/ssl_survey.pdf. Zhu's thesis: https://pages.cs.wisc.edu/~jerryzhu/pub/thesis.pdf.

## T2. Label propagation / spreading, low label rates

1. **Learning from Labeled and Unlabeled Data with Label Propagation** (CMU archive lists title as "Learning from Labels and Unlabeled Data with Label Propagation"). Zhu, Ghahramani. CMU-CALD-02-107 tech report, 2002. https://mlg.eng.cam.ac.uk/zoubin/papers/CMU-CALD-02-107.pdf (the PDF returned garbled text in the search tool; the abstract page is http://reports-archive.adm.cs.cmu.edu/anon/cald/abstracts/02-107.html)
   - (abstract) iterative propagation of labels through high-density regions defined by unlabelled data; relation to other methods; parameter learning by MST heuristic / entropy minimisation.
   - Application: idea (2). Run it from the 154 labelled reference images with their type labels and read soft posteriors on trajectory frames.

2. **Semi-Supervised Learning Using Gaussian Fields and Harmonic Functions.** Zhu, Ghahramani, Lafferty. ICML 2003, pp. 912-919. https://aaai.org/papers/icml03-118-semi-supervised-learning-using-gaussian-fields-and-harmonic-functions/ ; PDF https://mlg.eng.cam.ac.uk/zoubin/papers/zgl.pdf
   - (abstract) Gaussian random field on a weighted graph, field mean is harmonic; links to random walks and electric networks; class-prior incorporation.
   - Application: the "Laplace learning" baseline. It also yields the ability to fold class priors in, which matters if trajectory frames are not class balanced.

3. **Learning with Local and Global Consistency.** Zhou, Bousquet, Lal, Weston, Schölkopf. NIPS 2003 (vol. 16), pp. 321-328. https://papers.nips.cc/paper/2506-learning-with-local-and-global-consistency
   - (abstract) classifier varies smoothly with respect to structure revealed by labelled and unlabelled points; simple algorithm.
   - Application: label spreading with symmetric normalisation (D^-1/2 W D^-1/2). It is the version implemented as scikit-learn LabelSpreading (the sklearn page https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelSpreading.html appeared in search; I did not read what it says about the algorithm). Soft outputs can be the ranking target.

4. **Poisson Learning: Graph Based Semi-Supervised Learning At Very Low Label Rates.** Calder, Cook, Thorpe, Slepčev. ICML 2020 (PMLR 119). http://proceedings.mlr.press/v119/calder20a/calder20a.pdf ; arXiv 2006.11184 https://arxiv.org/abs/2006.11184
   - (abstract) replaces Laplace learning's label-value constraints with sources/sinks at labelled points and solves a Poisson equation; reported better than other low-label-rate methods on MNIST, FashionMNIST, CIFAR-10; Poisson MBO variant can use class-size priors.
   - Application: the 154 reference labels over 1278 nodes (about 12%) is not "very low", so Laplace/spreading is probably fine, but Poisson is the relevant fix if the per-class reference count is tiny (e.g. a handful per type).
   - Search-result caveat: page numbers disagree across indexes (1306-1316 vs 8588-8598); not cited here.

5. **Label Propagation for Deep Semi-supervised Learning.** Iscen, Tolias, Avrithis, Chum. CVPR 2019, pp. 5070-5079. https://openaccess.thecvf.com/content_CVPR_2019/html/Iscen_Label_Propagation_for_Deep_Semi-Supervised_Learning_CVPR_2019_paper.html ; arXiv 1904.04717 https://arxiv.org/abs/1904.04717
   - Search summary: transductive label propagation over a kNN graph built from the network's own embeddings gives pseudo-labels, which then train the network; alternates graph building and training; graph built with FAISS.
   - Application: direct precedent for idea (2) (propagated pseudo-labels as training targets over a kNN graph of embeddings). Different from SSRM/FixMatch pseudo-labelling because the pseudo-label comes from graph diffusion, not the model's own softmax. Their graph uses a trained network's embedding, not frozen features.

## T3. GCN / SGC and kNN graphs on pretrained features

1. **Semi-Supervised Classification with Graph Convolutional Networks.** Kipf, Welling. ICLR 2017 (arXiv 1609.02907). https://arxiv.org/abs/1609.02907 ; https://openreview.net/forum?id=SJU4ayYgl
   - (abstract) first-order approximation of spectral convolutions; cost linear in edges; hidden layers encode both graph structure and node features; gains on citation graphs.
   - Application: idea (3) baseline. Caveat: designed for given graphs, not graphs built from the same features.

2. **Simplifying Graph Convolutional Networks.** Wu, Souza, Zhang, Fifty, Yu, Weinberger. ICML 2019 (PMLR 97), pp. 6861-6871. https://proceedings.mlr.press/v97/wu19e.html ; arXiv 1902.07153 https://arxiv.org/abs/1902.07153 ; code https://github.com/Tiiiger/SGC
   - (abstract) removes nonlinearities, collapses weights; the result is a fixed low-pass filter followed by a linear classifier; accuracy kept, up to ~100x faster than FastGCN.
   - Application: SGC is S^K X followed by a head, so on frozen features it is precomputable smoothing (K hops of kNN averaging) before the BT head. Cheap to test, and also a clean ablation of "graph smoothing" vs "graph regulariser".

3. **Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning.** Li, Han, Wu. AAAI 2018. https://arxiv.org/abs/1801.07606 ; https://ojs.aaai.org/index.php/AAAI/article/view/11604
   - (abstract) graph convolution is Laplacian smoothing; deep GCNs over-smooth so clusters become indistinguishable; proposes co-training and self-training for shallow GCNs.
   - Application: warns against more than 1-2 hops, which matters for a kNN graph on tightly clustered RHEED features where sensitive differences within a cluster may be what the ranking head needs.

4. **Measuring and Relieving the Over-Smoothing Problem for Graph Neural Networks from the Topological View.** Chen, Lin, Li, Li, Zhou, Sun. AAAI 2020, pp. 3438-3445. https://ojs.aaai.org/index.php/AAAI/article/view/5747 ; https://arxiv.org/abs/1909.03211
   - (abstract) MAD / MADGap smoothness metrics; over-smoothing linked to low information-to-noise ratio of received messages, depends on graph topology; MADReg and AdaEdge.
   - Application: MAD can be computed on post-smoothing features as a diagnostic for whether the kNN graph is collapsing type distinctions.

5. **Laplacian Regularized Few-Shot Learning (LaplacianShot).** Ziko, Dolz, Granger, Ben Ayed. ICML 2020 (PMLR 119). https://proceedings.mlr.press/v119/ziko20a.html
   - (abstract) transductive inference on a fixed embedding: unary prototype term plus pairwise Laplacian term over query samples; applied at inference after cross-entropy pre-training; reported consistent gains over SOTA, with compute close to inductive inference.
   - Application: strongest precedent in the few-shot area for exactly "frozen embedding plus kNN-graph Laplacian over unlabelled items" (idea 1).

6. **Learning to Propagate Labels: Transductive Propagation Network for Few-shot Learning.** Liu, Lee, Park, Kim, Yang, Hwang, Yang. ICLR 2019 (arXiv 1805.10002). https://openreview.net/forum?id=SyVuRiC5K7 ; https://arxiv.org/abs/1805.10002
   - (abstract) learns graph construction and propagates labels to the whole test set, trained end to end; SOTA on miniImageNet / tieredImageNet at that time.
   - Application: shows the graph itself can be learned (edge weights from a small network). Probably too many parameters for ~170 labelled groups; mention as an upper-end option.

7. **Rethinking Few-Shot Image Classification: A Good Embedding Is All You Need?** Tian, Wang, Krishnan, Tenenbaum, Isola. ECCV 2020, pp. 266-282. https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590256.pdf ; https://link.springer.com/chapter/10.1007/978-3-030-58568-6_16
   - (abstract) a linear classifier on a fixed learned embedding beats meta-learning methods; does not use test-data information.
   - Application: counter-weight to ideas (2)/(3) when the frozen feature is strong. (The abstract says a good embedding suffices; it does not say graph methods never help.)

Gap: I did not find a paper that directly tests label propagation or GCN on frozen SimCLR features of a small scientific-image dataset versus a linear probe. The search for that returned only the scikit-learn pages and Iscen et al.

## T4. Semi-supervised ranking / preference learning; negative results

1. **Ranking on Data Manifolds.** Zhou, Weston, Gretton, Bousquet, Schölkopf. NIPS 2003 (vol. 16), pp. 169-176. https://papers.nips.cc/paper_files/paper/2003/hash/2c3ddf4bf13852db711dd1901fb517fa-Abstract.html ; PDF https://proceedings.neurips.cc/paper_files/paper/2003/file/2c3ddf4bf13852db711dd1901fb517fa-Paper.pdf
   - (abstract) ranks items by the manifold structure of the whole dataset rather than local distance, spreading a query's score through a graph (a secondary summary calls it PageRank-style); reported encouraging results on synthetic, image and text data.
   - Application: query-based (rank by similarity to a query/reference). It supports using reference images as queries to score trajectory frames (idea 2 as a ranking target), but it is not a pairwise-preference model.

2. **Semi-supervised Learning to Rank with Preference Regularization.** Szummer, Yilmaz. CIKM 2011, pp. 269-278 (pages from citing papers). https://dl.acm.org/doi/10.1145/2063576.2063620 ; https://www.microsoft.com/en-us/research/wp-content/uploads/2011/10/SzummerYilmaz-semisupervised-ranking-cikm11.pdf
   - Search summary (abstract-level; assembled from fragments): learns from pairwise preferences or absolute labels plus unlabelled items, handles groups (queries) containing only unlabelled items; adds a regulariser that makes similar items similar in preference; used in SSLambdaRank; rank-sensitive variant; reported improvement over LambdaRank.
   - Application: the closest published match to idea (1) in a ranking setting (preference loss plus similarity regulariser over unlabelled items). Also fits the "unlabelled groups" setting here.

3. **Learning to Rank with Partially-Labeled Data.** Duh, PhD thesis, University of Washington, 2009. https://www.cs.jhu.edu/~kevinduh/papers/duh09phd.pdf
   - Only seen as a search hit; the snippets say it covers semi-supervised learning for ranking. Venue/year details are from my memory of the PDF host, so mark UNVERIFIED for the university and year.

4. **Realistic Evaluation of Deep Semi-Supervised Learning Algorithms.** Oliver, Odena, Raffel, Cubuk, Goodfellow. NeurIPS 2018 (arXiv 1804.09170). https://proceedings.neurips.cc/paper/2018/hash/c1fea270c48e8079d8ddf7d06d26ab52-Abstract.html ; https://arxiv.org/abs/1804.09170
   - (abstract) supervised baselines are often under-tuned/under-reported; SSL methods are sensitive to label and unlabelled amounts; performance can degrade substantially when unlabelled data contain out-of-class examples.
   - Application: the main negative-results citation. The trajectory frames may include states not covered by the 4 type heads, which is the out-of-class risk. Mandates a well-tuned no-graph baseline with the same selection protocol.

Not found: any paper on a graph-Laplacian-regularised Bradley-Terry model with item features. The search turned up only a survey (https://arxiv.org/html/2601.14727v2, "Recent advances in the Bradley-Terry Model: theory, algorithms, and applications", authors not shown in results) that notes a Laplacian prior appears in spatial-smoothing examples, and that the BT Hessian/Fisher information is a graph Laplacian over the comparison graph (a different graph, the comparison graph, not the feature kNN graph). Treat the Laplacian-BT combination as novel here or UNVERIFIED as published.

## T5. Controls: shuffled/random graph, graph vs features

1. **Revisiting Graph Neural Networks: All We Have is Low-Pass Filters.** NT, Maehara. arXiv 1905.09550, 2019 (venue not shown in results; UNVERIFIED if published elsewhere). https://arxiv.org/abs/1905.09550
   - (abstract) GNNs mainly act as low-pass filters on node features; benchmark feature vectors are already informative, so the graph mainly denoises them.
   - Application: the mechanism for "graph gives little when features are strong", which is the likely situation with SimCLR features.

2. **Graph Neural Networks are Inherently Good Generalizers: Insights by Bridging GNNs and MLPs (PMLP).** Yang, Wu, Wang, Yan. ICLR 2023 (arXiv 2212.09034). https://arxiv.org/abs/2212.09034 ; https://openreview.net/forum?id=dqnNW2omZL6 ; code https://github.com/chr26195/PMLP
   - (abstract) PMLP is trained as a plain MLP and uses GNN message passing only at test time; matches or exceeds GNN counterparts while training much faster; the claim is that GNN gains come from generalisation, not expressivity.
   - Application: a control: train the BT head with no graph and apply smoothing only at inference (kNN averaging of scores). Cheap and avoids train-time graph leakage.

3. **Graph-MLP: Node Classification without Message Passing in Graph.** Hu, You, Wang, Wang, Zhou, Gao. arXiv 2106.04051, 2021 (venue not shown in results; UNVERIFIED). https://arxiv.org/abs/2106.04051 ; code https://github.com/yanghu819/Graph-MLP
   - (abstract) MLP only, graph used through a neighbouring contrastive (NContrast) loss at training; no adjacency at inference; comparable to or better than GNNs.
   - Application: matches idea (1) closely (graph enters only as a training-time loss). A later paper (arXiv 2412.08310, Edge-Splitting MLP) notes it relies solely on homophily (summary from search result).

4. **Pitfalls of Graph Neural Network Evaluation.** Shchur, Mumme, Bojchevski, Günnemann. NeurIPS 2018 Relational Representation Learning Workshop; arXiv 1811.05868. https://arxiv.org/abs/1811.05868
   - (abstract) identical splits and unequal training details preclude fair comparison; simpler architectures can beat more sophisticated ones when tuned fairly. The search results did NOT state that an MLP baseline was used, so I do not claim this.
   - Application: protocol warning: tune every variant equally, use many random splits (relevant with ~170 groups).

5. Graph structure learning benchmarks: **GSLB: The Graph Structure Learning Benchmark.** Li, Wang, Sun, et al. NeurIPS 2023 Datasets and Benchmarks (author list truncated in the results). https://proceedings.neurips.cc/paper_files/paper/2023/file/60bc87f3cf5257579435d92ec12c761b-Paper-Datasets_and_Benchmarks.pdf ; and **OpenGSL** https://arxiv.org/pdf/2306.10280 (authors/venue not shown in results, UNVERIFIED).
   - Search summary (abstract-level): GSL generally helps on homophilous graphs and with scarce labels; on heterophilous graphs many methods do not beat kNN-graph + standard message passing baselines; learned graphs get denser but homophily does not reliably rise. A 2024 paper (https://arxiv.org/html/2411.07672v1, "Rethinking Structure Learning For Graph Neural Networks") is quoted as citing Luo et al. 2024 and Platonov et al. 2023 for the finding that GSL does not consistently beat plain GNNs under the same tuning (I did not see those two papers themselves).
   - Application: supports using a plain kNN graph and spending effort on controls rather than learned graphs.

6. Weak evidence, flag as such: arXiv 2512.06638 ("The Impact of Data Characteristics on GNN Evaluation for Detecting Fake News", Dec 2025; authors not shown) https://arxiv.org/html/2512.06638 — the search summary says shuffling edges left performance essentially unchanged while shuffling node features collapsed it, and an MLP on features matched GCN models on GossipCop. Different domain; one dataset family; I did not read it. And arXiv 2402.04621 ("Feature Distribution on Graph Topology Mediates the Effect of Graph Convolution: Homophily Perspective") https://arxiv.org/html/2402.04621v2, which reports (summary) shuffling node features within class improved GCN while MLP and label propagation stayed flat.

Gap: I found no canonical, citable "shuffled-graph control" paper. The practice (rewire edges / shuffle node-to-image assignment and see whether the gain disappears) is described only as a suggested check in a review note on pith.science (https://pith.science/paper/2607.25284), which is not a result. Recommend citing the PMLP / low-pass / GSLB papers for the motivation and describing the shuffle control as our own design.

## Summary of support

Well supported (multiple confirmed items): manifold/Laplacian regularisation (Belkin 2006, Neural Graph Machines, Weston 2008); label propagation/spreading and low-label-rate degeneration (Zhu 2002/2003, Zhou 2003, Nadler 2009, Poisson learning); GCN/SGC and the over-smoothing caveats; Szummer & Yilmaz as a direct semi-supervised ranking precedent with a preference regulariser; Oliver et al. for negative results; PMLP / low-pass papers for "graph adds little when features are already good".

Thin: (a) no confirmed paper with Laplacian-regularised Bradley-Terry on features; (b) no confirmed study of label propagation or GCN on frozen SimCLR features vs linear probe; (c) no canonical shuffled-graph control paper (only low-quality hits); (d) Duh thesis and several arXiv-only items (Graph-MLP, low-pass) have unconfirmed venue; (e) everything is from abstracts/summaries, not full text.
