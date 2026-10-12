# Literature search: anchor/auxiliary class-label loss weighting in pairwise (Bradley-Terry) preference learning

Method note: all entries come from WebSearch result titles/snippets. WebFetch failed (getaddrinfo ENOTFOUND arxiv.org), so no full texts were read. Bibliographic details (title, authors, venue, year) were seen in search results; claims about content are from search-result summaries unless noted. Nothing below directly tests "8x weight good, 32x bad" for absolute-class-label anchor loss with 10-60 pairs; support is indirect.

## T17. Multi-task / auxiliary-loss weighting

1. Kendall, A., Gal, Y., Cipolla, R. "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics." CVPR 2018 (pp. 7482-7491); arXiv 1705.07115.
   URL: https://arxiv.org/abs/1705.07115 ; https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
   Relevance: argues performance depends strongly on relative task-loss weights and that hand-tuning is costly; learns weights via homoscedastic uncertainty. Motivates treating the anchor-loss weight as a sensitive hyperparameter (non-monotone effect is consistent with their claim).
2. Chen, Z., Badrinarayanan, V., Lee, C.-Y., Rabinovich, A. "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks." ICML 2018 (PMLR 80); arXiv 1711.02257.
   URL: https://arxiv.org/abs/1711.02257 ; https://proceedings.mlr.press/v80/chen18a.html
   Relevance: task imbalance shows up as gradient-norm imbalance; GradNorm matches or beats exhaustive grid search over static weights. Supports the framing that a fixed weight > 1 on a small-but-informative task can be right, and too large unbalances gradients.
3. Liu, S., Johns, E., Davison, A. J. "End-to-End Multi-Task Learning with Attention." CVPR 2019 (arXiv 1803.10704). Introduces Dynamic Weight Average (DWA).
   URL: https://ar5iv.labs.arxiv.org/html/1803.10704 ; https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.pdf
   Relevance: DWA reweights tasks from loss-rate-of-change, loss values only; a cheap adaptive alternative to a hand-set 8x.
4. Shi, B., Hoffman, J., Saenko, K., Darrell, T., Xu, H. "Auxiliary Task Reweighting for Minimum-data Learning." NeurIPS 2020; arXiv 2010.08244.
   URL: https://proceedings.neurips.cc/paper/2020/hash/4f87658ef0de194413056248a00ce009-Abstract.html ; https://arxiv.org/pdf/2010.08244
   Relevance: best match for "auxiliary weights matter in low-label regimes": reweights auxiliary tasks (as a surrogate prior) to cut main-task labelled-data needs; reports gains with very few main-task labels (e.g. 1 image per class per search summary).
5. (Supporting, lower priority) Kurin, V. et al. "In Defense of the Unitary Scalarization for Deep Multi-Task Learning." NeurIPS 2022; arXiv 2201.04122.
   URL: https://arxiv.org/abs/2201.04122
   Relevance: counterpoint: fixed unit weights plus regularization match specialised weighting methods in their (not low-label) settings. Cite as a caveat that weight tuning benefits are not universal.
6. (Supporting) "Auxiliary Learning by Implicit Differentiation" (AuxiLearn; arXiv 2007.02693; authors and venue UNVERIFIED, not shown in results). URL: https://arxiv.org/abs/2007.02693 (seen as ar5iv/arxiv.org html in results). Relevance: learns auxiliary-loss weights on a small held-out set and evaluates in a low-data regime (5% of CIFAR). First author and venue UNVERIFIED.

Gap: no paper found that systematically measures sensitivity of auxiliary weight vs. number of labels. Search tool itself said none found. Claim "weights matter most in low-label regimes" is supported by ARML (and AuxiLearn) only indirectly.

## T18. Combining absolute and pairwise labels

1. Guo, Y., Tian, P., et al. (11 authors incl. S. Ioannidis). "Experimental Design under the Bradley-Terry Model." IJCAI 2018, pp. 2198-2204, doi 10.24963/ijcai.2018/304. (Search summary: "Yuan Guo, Peng Tian, and nine coauthors (including Stratis Ioannidis)"; full list not seen.)
   URL: https://www.ijcai.org/proceedings/2018/304 ; https://www.ijcai.org/proceedings/2018/0304.pdf
   Relevance: setting is existing absolute (noisy) class labels plus a budget of expert pairwise comparisons under BT; direct precedent for mixing both label types. Results section not seen.
2. Wang, H., Xiong, W., Xie, T., Zhao, H., Zhang, T. "Interpretable Preferences via Multi-Objective Reward Modeling and Mixture-of-Experts" (ArmoRM). Findings of EMNLP 2024, pp. 10582-10592; arXiv 2406.12845.
   URL: https://arxiv.org/abs/2406.12845 ; https://aclanthology.org/2024.findings-emnlp.620/
   Relevance: frozen backbone + small head trained with regression on ABSOLUTE ratings, then a small gating MLP trained with BT loss on pairs; same overall design (frozen features, small head, absolute + pairwise). Stages are sequential, not a weighted sum.
3. Yang, R., Ding, R., Lin, Y., Zhang, H., Zhang, T. "Regularizing Hidden States Enables Learning Generalizable Reward Model for LLMs" (GRM). NeurIPS 2024; arXiv 2406.10216.
   URL: https://arxiv.org/abs/2406.10216
   Relevance: loss = (1 - alpha) * L_reward + alpha * L_reg with an auxiliary loss on shared features; shows the auxiliary-weight trade-off in reward modelling (generalisation gains; weight must be balanced). Auxiliary is SFT text loss, not absolute labels, so analogous only.
4. Zhu, H. et al. "Adaptive Image Quality Assessment via Teaching Large Multimodal Model to Compare" (Compare2Score). NeurIPS 2024 (Spotlight); arXiv 2405.19298.
   URL: https://arxiv.org/abs/2405.19298 ; https://github.com/Q-Future/Compare2Score
   Relevance: trains on pairs, infers on single images by comparing against a set of ANCHOR images (selected by quality-interval partitioning) and converting the preference matrix to scores via Thurstone Case V MAP. Closest published instance of anchor/reference-based scoring from comparisons in images.
5. Zhang, W., Ma, K., Zhai, G., Yang, X. "Uncertainty-Aware Blind Image Quality Assessment in the Laboratory and Wild" (UNIQUE). IEEE TIP 2021; arXiv 2005.13983.
   URL: https://arxiv.org/abs/2005.13983
   Relevance: learns a quality scorer purely from pairwise probabilities (fidelity loss) derived from absolute MOS; shows that absolute labels can be converted to pairwise supervision, the converse of your anchor loss.
6. Possible: "Learning Ordinal Probabilistic Reward from Preferences" (arXiv 2602.12660; authors/venue UNVERIFIED). URL: https://arxiv.org/html/2602.12660. Search summary says it combines pairwise ranking with absolute semantic constraints ("Region Flooding"). Not read; do not cite without checking.
7. Possible: "Beyond Binary Preferences: A Principled Framework for Reward Modeling with Ordinal Feedback" (arXiv 2603.02232; authors/venue UNVERIFIED). URL: https://arxiv.org/html/2603.02232. Derives losses from ordinal regression instead of ad-hoc margin/scale terms.
8. Possible caution: "Pointwise or Pairwise: When Do Pairwise Losses Help Reward Learning, Provably?" (arXiv 2609.37209, Sept 2026, first author Junghyun Lee per search; other details UNVERIFIED). URL: https://arxiv.org/abs/2609.37209. Theory comparing pointwise value regression vs pairwise difference regression.
9. Possible: Jensen, B. S., Nielsen, J. B. "Pairwise judgements and absolute ratings with Gaussian process priors." DTU technical report, 2011 -- seen only as a reference string in Biyik et al. 2024 (IJRR); NOT confirmed directly, UNVERIFIED. Joint model of both feedback types.
10. (Weak, applied) "Improving Multi Task Recommendations via Cross User Learning with a Hybrid Pointwise and Pairwise Ranking Loss." The Web Conference 2026 (ACM DL doi 10.1145/3774904.3792823); authors UNVERIFIED. URL: https://dl.acm.org/doi/10.1145/3774904.3792823. Pointwise BCE + weighted pairwise term in production ranking.

Gap: no source found for an explicit "anchor loss" with a weight sweep in pairwise RLHF reward models, nor for absolute-label calibration with weight ablation. Search for BT scale/shift non-identifiability gave only general background (e.g. "Rethinking Bradley-Terry Models in Preference-Based Reward Modeling", arXiv 2411.04991, authors/venue UNVERIFIED, https://arxiv.org/html/2411.04991v1); this supports the intuition that absolute labels supply the otherwise unidentified scale/anchor.

## T19. Class-labelled prototypes/references as a prior; semi-supervised use of reference sets

1. Snell, J., Swersky, K., Zemel, R. S. "Prototypical Networks for Few-shot Learning." NIPS/NeurIPS 2017; arXiv 1703.05175.
   URL: https://arxiv.org/abs/1703.05175
   Relevance: classify by distance to class prototypes (mean embeddings) in a fixed embedding; foundational for few-label reference-based scoring with frozen features.
2. Xu, Y., Balakrishnan, S., Singh, A., Dubrawski, A. (author names from memory, UNVERIFIED in search results) "Nonparametric Regression with Comparisons: Escaping the Curse of Dimensionality with Ordinal Information" (Ranking-Regression, R^2). ICML 2018; journal version JMLR 21 (2020), paper 19-505. (Only title/venue seen in search results.)
   URL: https://arxiv.org/html/1806.03286v1 ; https://jmlr.org/papers/volume21/19-505/19-505.pdf
   Relevance: semi-supervised combination of a very small labelled set with comparison/ordinal information on many unlabelled points; isotonic regression on labelled points anchors the ranking. Best theory/empirical match for "few absolute labels + comparisons".
3. Zhu, H. et al. Compare2Score (see T18.4): scoring a test image by comparison probabilities against labelled anchor images is a "win-rate against reference set" scorer.
4. Li, W. et al. "OrdinalCLIP: Learning Rank Prompts for Language-Guided Ordinal Regression." NeurIPS 2022. URL: https://openreview.net/pdf?id=JpxsSAecqq (first author name UNVERIFIED). Relevance: per-rank prototypes (text embeddings) with contrastive matching; reports gains in few-shot settings.
5. "An Algorithm for Ordinal Classification Based on Pairwise Comparison" (PairCode), Journal of Classification, c. 2019/2020 (authors UNVERIFIED). URL: https://link.springer.com/article/10.1007/s00357-019-9311-4. Ordinal classes handled through pairwise comparisons for small samples.
6. Duh, K. K. PhD thesis "Learning to Rank with Partially-Labeled Data" (Univ. Washington, 2009; seen at cs.jhu.edu). URL: https://www.cs.jhu.edu/~kevinduh/papers/duh09phd.pdf. Semi-supervised ranking; background only.

Gap: no paper found that uses "win rate against a class-labelled reference set" as a classification rule in the Bradley-Terry few-pair regime; the closest are Compare2Score and R^2.

## Summary of support
- T17: moderately supported (uncertainty weighting, GradNorm, DWA are confirmed; ARML supports low-label weighting; no direct low-label sensitivity study).
- T18: moderately supported conceptually (Guo et al. 2018, ArmoRM, GRM, Compare2Score, UNIQUE); thin for an explicit anchor-loss weight ablation in reward models.
- T19: thin; prototype and semi-supervised-comparison precedents (ProtoNets, R^2, Compare2Score) exist but none matches the exact setup.
