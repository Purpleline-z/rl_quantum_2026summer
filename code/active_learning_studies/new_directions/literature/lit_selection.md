# Literature search: active preference learning for a BT reward model on RHEED (T9-T12)

Verification note: WebFetch could not resolve any external host in this sandbox (DNS failure), so I could NOT open pages directly. Every entry below was confirmed from WebSearch result titles/URLs/snippets (usually 2+ independent hits: arXiv/venue page plus a mirror). Title, first author and venue/year are reliable where marked VERIFIED; items marked PARTIAL have one missing detail; UNVERIFIED items should be checked before citing. Today's date is 2026-10-10.

Support summary: T10 strong, T11 strong for "random is hard to beat / evaluation pitfalls" but thin for "winner's curse across seeds" and "pre-registration in active learning" (only generic ML sources), T9 moderate (direct SSRM hit, but nothing on pairs-with-ties or pairwise FixMatch), T12 thin (prototype+SSL good; "mixture of ideal classes" acquisition essentially absent).

---
## T9. Semi-supervised / pseudo-label / self-training for pairwise preference and reward models

1. **Semi-Supervised Reward Modeling via Iterative Self-Training** (SSRM). Yifei He, Haoxiang Wang, Ziyan Jiang, Alexandros Papangelis, Han Zhao. Findings of EMNLP 2024, pp. 7365-7377. arXiv 2409.06903. https://arxiv.org/abs/2409.06903 ; https://aclanthology.org/2024.findings-emnlp.434.pdf . VERIFIED (search).
   Loop: pseudo-label unlabeled prompt-response pairs with current RM, keep only those above a confidence threshold, retrain. Direct template for confidence-thresholded pseudo-pairs; for us, pseudo-label candidate pairs (winner/tie) from the MLP head, threshold on |s_a - s_b| or P_BT. Authors cite confidence filtering as the error-propagation guard.

2. **FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence.** Kihyuk Sohn, David Berthelot, et al. NeurIPS 2020. arXiv 2001.07685. https://arxiv.org/abs/2001.07685 . VERIFIED.
   Weak-augmentation pseudo-label, keep if confident, train on strong augmentation. Adaptation: with frozen SimCLR features, "strong augmentation" = feature dropout/noise or alternate augmented views embedded through the frozen encoder; apply the threshold to the BT pair probability. (No FixMatch-for-pairs paper found; this is our adaptation.)

3. **Pseudo-Labeling and Confirmation Bias in Deep Semi-Supervised Learning.** Eric Arazo, Diego Ortego, Paul Albert, Noel E. O'Connor, Kevin McGuinness. IJCNN 2020. arXiv 1908.02983. https://arxiv.org/abs/1908.02983 . VERIFIED.
   Key failure-mode reference: with few labels, naive pseudo-labeling fits its own wrong pseudo-labels (confirmation bias); mitigations are mixup and a minimum number of labeled samples per mini-batch (relevant at 168 labeled groups).

4. **Two Minds Better Than One: Collaborative Reward Modeling for LLM Alignment.** Jiazheng Zhang, ... Qi Zhang (12 authors). arXiv 2505.10597 (2025; venue not verified - an OpenReview entry exists: https://openreview.net/forum?id=BB1aypUDAF). https://arxiv.org/abs/2505.10597 . VERIFIED existence, venue UNVERIFIED.
   Argues self-loss filtering of noisy preferences inherits self-training confirmation bias; uses two RMs that peer-review each other's selections (with a "Self Review" ablation showing cumulative error). Adaptable as co-training of two heads or two feature views for pseudo-pair selection.

5. **Neighborhood-Regularized Self-Training for Learning with Few Labels** (NeST). Ran Xu, Yue Yu, Hejie Cui, ..., Carl Yang. AAAI 2023, 37(9):10611-10619. arXiv 2301.03726. https://arxiv.org/abs/2301.03726 ; https://ojs.aaai.org/index.php/AAAI/article/view/26260 . VERIFIED.
   Neighbourhood-based pseudo-label selection plus aggregating predictions across rounds to curb noise propagation; natural fit for SimCLR feature space (agreement of a pair's pseudo-label with its kNN neighbours' labels).

6. **Semi-Supervised Document Retrieval** (SSRank). Ming Li, Hang Li, Zhi-Hua Zhou. Information Processing & Management, 2009. https://ai.nju.edu.cn/lim/publications/ip&m09.pdf . PARTIAL (venue/year inferred from URL "ip&m09" and snippet).
   Closest classical semi-supervised learning-to-rank paper: pairwise (RankNet/RankSVM-style) learners, pseudo-labeling unlabeled data; two-view variants beat single-view. Also: Kevin Duh, PhD thesis "Learning to Rank with Partially-Labeled Data" (UW 2009; https://www.cs.jhu.edu/~kevinduh/papers/duh09phd.pdf), pairwise formulation with graph regularization. PARTIAL.

7. (Context, not core) **SemiReward: A General Reward Model for Semi-supervised Learning.** ICLR 2024, arXiv 2310.03013, https://arxiv.org/abs/2310.03013 . VERIFIED but this "reward" is a learned pseudo-label filter, not a preference model; useful only as an alternative to fixed thresholds.

Thin spots: no paper found applying FixMatch-style thresholding to pairs with an explicit tie class; no paper on pseudo-labeling for BT models with small labeled sets (SSRM uses LLMs with large labeled seeds, e.g. 1/16 of data).

---
## T10. Meta-strategies / adaptive mixing of diversity and uncertainty; small-budget evidence

1. **Active Learning by Learning** (ALBL). Wei-Ning Hsu, Hsuan-Tien Lin. AAAI 2015, pp. 2659-2665, DOI 10.1609/AAAI.V29I1.9597. https://ojs.aaai.org/index.php/AAAI/article/view/9597 ; PDF https://www.csie.ntu.edu.tw/~htlin/paper/doc/aaai15albl.pdf . VERIFIED.
   Treats each acquisition rule as a bandit arm (modified EXP4.P), rewarding by importance-weighted performance on labeled data. Direct basis for adaptively choosing among {random, uncertainty, TypiClust, coverage, pair-margin}. Caveat for us: reward estimate from ~168 labels is very noisy.

2. **Active Learning on a Budget: Opposite Strategies Suit High and Low Budgets** (TypiClust). Guy Hacohen, Avihu Dekel, Daphna Weinshall. ICML 2022, PMLR 162:8175-8195. arXiv 2202.02794. https://proceedings.mlr.press/v162/hacohen22a.html ; https://arxiv.org/abs/2202.02794 . VERIFIED.
   Low budget: pick typical/dense examples from clusters of self-supervised (SimCLR) features; high budget: pick uncertain/atypical. States random often beats deep AL when labeled set is tiny due to poor uncertainty estimates. Matches our frozen-SimCLR setting directly.

3. **Bridging Diversity and Uncertainty in Active Learning with Self-Supervised Pre-Training** (TCM). Paul Doucet, Benjamin Estermann, Till Aczel, Roger Wattenhofer. ICLR 2024 workshop (PML4LRS); arXiv 2403.03728. https://arxiv.org/abs/2403.03728 ; https://mlanthology.org/iclrw/2024/doucet2024iclrw-bridging/ . VERIFIED. (Workshop paper, not main track.)
   TypiClust first, then switch to Margin; with a self-supervised backbone (SimCLR on CIFAR) the switch point is early, so no elaborate switching schedule is needed. Simple, pre-specifiable schedule for us.

4. **Active Learning Through a Covering Lens** (ProbCover). Ofer Yehuda, Avihu Dekel, Guy Hacohen, Daphna Weinshall. NeurIPS 2022. arXiv 2205.11320. https://arxiv.org/abs/2205.11320 . VERIFIED.
   Low-budget selection by maximizing probability mass covered by delta-balls in self-supervised feature space; an alternative diversity arm.

5. **Adaptive Diversity-Uncertainty Active Learning with Redundancy Control for Bioacoustic Event Classification** (ADU-MMR). Gabriel Dubus et al. arXiv 2607.04868 (July 2026 preprint). https://arxiv.org/abs/2607.04868 . PARTIAL: seen in search results only; author list beyond first author and peer-review status not confirmed. Reweights diversity vs uncertainty by mean pool entropy, then greedy MMR for batch redundancy; 500-label budget, beats CoreSet/margin/random in AULC on BirdSet terrestrial, only competitive on marine. Cite cautiously (recent preprint).

6. (Supporting) **Cold-start Active Learning through Self-supervised Language Modeling** (ALPS). Michelle Yuan et al. EMNLP 2020. https://aclanthology.org/2020.emnlp-main.637/ . VERIFIED but NLP. Also **A Simple Baseline for Low-Budget Active Learning**, Kossar Pourahmadi et al., arXiv 2110.12033, https://arxiv.org/pdf/2110.12033 (self-supervised features + k-means selection beats random at small budgets). VERIFIED (search).

Evidence for small budgets: TypiClust, ProbCover, TCM, and Pourahmadi all report diversity/typicality-first beating random and uncertainty at low budget on image benchmarks. No evidence for pairwise/BT-style acquisition in this body of work; all are class-label settings.

Pair-specific active selection (bonus): **Reviving The Classics: Active Reward Modeling in Large Language Model Alignment.** Yunyi Shen, Hao Sun, Jean-Francois Ton. arXiv 2502.04354 (2025; venue not verified). https://arxiv.org/abs/2502.04354 . VERIFIED existence. Benchmarks 8 pair-scoring rules for BT reward models; reports that classical experimental-design criteria (D-/Fisher-information-style) applied to the final linear feature layer are state of the art and stable. Directly applicable to our MLP head on frozen features.

---
## T11. Active learning evaluation pitfalls; random is hard to beat

1. **Towards Robust and Reproducible Active Learning Using Neural Networks.** Prateek Munjal, Nasir Hayat, Munawar Hayat, Jamshid Sourati, Shadab Khan. CVPR 2022 (arXiv 2002.09564). https://arxiv.org/abs/2002.09564 ; https://openaccess.thecvf.com/content/CVPR2022/html/Munjal_Towards_Robust_and_Reproducible_Active_Learning_Using_Neural_Networks_CVPR_2022_paper.html . VERIFIED.
   Under identical, well-regularized training, uncertainty/diversity/committee AL gains over random are inconsistent and often marginal; variance attributed to uncontrolled randomness (seeds, init sets). Code: TorchAL.

2. **Navigating the Pitfalls of Active Learning Evaluation: A Systematic Framework for Meaningful Performance Assessment.** Carsten T. Luth, Till J. Bungert, Lukas Klein, Paul F. Jaeger. NeurIPS 2023. arXiv 2301.10625. https://arxiv.org/abs/2301.10625 ; https://proceedings.neurips.cc//paper_files/paper/2023/hash/1ed4723f12853cbd02aecb8160f5e0c9-Abstract-Conference.html . VERIFIED.
   Five pitfalls (data distribution, starting budget, query size, classifier configuration, ...) and the finding that apparent gains often come from poorly tuned random baselines. Relevant: tune the random baseline's head (lr, wd, epochs) equally; fix protocol before seeing results. (Detailed per-pitfall list from a secondary summary; the fifth is unnamed there.)

3. **Randomness Is the Root of All Evil: More Reliable Evaluation of Deep Active Learning.** Ji et al. WACV 2023. https://openaccess.thecvf.com/content/WACV2023/papers/Ji_Randomness_Is_the_Root_of_All_Evil_More_Reliable_Evaluation_WACV_2023_paper.pdf . PARTIAL (first author surname and venue from URL only; full title from URL slug).
   Varies initial-set and model-init seeds jointly; some methods are highly seed-sensitive; recommends many (init set x init weights) combinations. Supports same-seed paired designs plus many seeds.

4. **Parting with Illusions about Deep Active Learning.** Sudhanshu Mittal, Maxim Tatarchenko, Ozgun Cicek, Thomas Brox. arXiv 1912.05361 (2019; preprint). https://arxiv.org/abs/1912.05361 . VERIFIED.
   With strong augmentation and semi-supervised learning, AL's advantage over random shrinks to near zero; in low-budget CIFAR-100, nothing beat SSL + random sampling. Recommends strong SSL baseline and low-budget focus. Also: Simeoni, Budnik et al., **Rethinking deep active learning: Using unlabeled data at model training**, arXiv 1911.08177, https://arxiv.org/abs/1911.08177 (unlabeled data in training matters more than acquisition choice). VERIFIED (search; full author list not captured).

5. **Realistic Evaluation of Deep Active Learning for Image Classification and Semantic Segmentation.** International Journal of Computer Vision, 2025. https://link.springer.com/article/10.1007/s11263-025-02372-z . PARTIAL (authors not captured; claims per search snippet: AL barely beats random without augmentation + SSL).

6. Winner's curse / selection bias (generic, NOT active-learning-specific):
   - **On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation.** Gavin C. Cawley, Nicola L. C. Talbot. JMLR 11:2079-2107, 2010. https://www.jmlr.org/papers/v11/cawley10a.html . VERIFIED. Choosing the best configuration on the same data used to report performance biases the estimate; selection effects can be as large as differences between algorithms. Justifies nested/held-out confirmation for the chosen acquisition rule.
   - **A Flexible Defense Against the Winner's Curse.** Tijana Zrnic, William Fithian. arXiv 2411.18569. https://arxiv.org/abs/2411.18569 . VERIFIED (arXiv id/authors from search; statistics paper, not ML-AL). Valid inference for the top-selected candidate among several.
   I found NO paper that explicitly studies winner's curse across acquisition functions or seeds in AL; the claim must be framed by analogy to these.

7. Pre-registration (generic ML): **NeurIPS 2020 and 2021 Workshops on Pre-registration in Machine Learning** (organizers incl. Samuel Albanie, Joao F. Henriques, Luca Bertinetto). PMLR vol. 148 (2020 workshop, published 2021) and vol. 181 (2021 workshop, published 2022). http://proceedings.mlr.press/v148/ ; http://proceedings.mlr.press/v181/ ; https://preregister.science/ . VERIFIED (existence). Precedent for committing protocol before results. No AL-specific pre-registration paper found.

8. Hyperparameter-sensitivity (bonus): **Survey of Active Learning Hyperparameters: Insights from a Large-Scale Experimental Grid**, arXiv 2506.03817 (2025), https://arxiv.org/abs/2506.03817 . PARTIAL (authors not captured). Proposes a reduced grid reproducing method rankings.

---
## T12. Mixture / ambiguity-aware acquisition; prototype/anchor classifiers as priors

Thin area. No paper found that does acquisition for samples that are mixtures of ideal classes (mixed-membership) in images. Closest evidence:

1. **Prototypical Networks for Few-shot Learning.** Jake Snell, Kevin Swersky, Richard Zemel. NeurIPS 2017. arXiv 1703.05175. https://arxiv.org/abs/1703.05175 . VERIFIED.
   Class = mean embedding (prototype); nearest-prototype classification. Use: init per-reconstruction-type anchors from labeled "ideal" images in SimCLR space as a prior/regulariser for the MLP head, and as a nonparametric scorer when labels are very few.

2. **Meta-Learning for Semi-Supervised Few-Shot Classification.** Mengye Ren, Eleni Triantafillou, Sachin Ravi, Jake Snell, Kevin Swersky, et al. ICLR 2018. arXiv 1803.00676. https://arxiv.org/abs/1803.00676 ; https://openreview.net/pdf?id=HJcSzz-CZ . VERIFIED.
   Extends prototypes with soft k-means over unlabeled data, plus a distractor cluster / masked variant for out-of-class samples (relevant: mixtures and "not_apply" images behave like distractors). Also includes an active adaptation demo in the earlier workshop version (arXiv 1711.10856, "Semi-Supervised and Active Few-Shot Learning with Prototypical Networks", https://arxiv.org/html/1711.10856v2 ; PARTIAL).

3. **Improved prototypical network for active few-shot learning** (AC-FSL). Pattern Recognition Letters, 172:188, 2023 (ADS 2023PaReL.172..188W). https://www.sciencedirect.com/science/article/abs/pii/S0167865523001940 . PARTIAL (authors not captured). Prototype classifier plus loss-prediction module selecting samples to label.

4. **Predictive Uncertainty Estimation via Prior Networks.** Andrey Malinin, Mark Gales. NeurIPS 2018. arXiv 1802.10501. https://arxiv.org/abs/1802.10501 . VERIFIED.
   Dirichlet outputs separating data (aleatoric) uncertainty from distributional uncertainty. Relevant idea: a mixed-membership image has high data uncertainty that more labels will not remove; acquisition should target epistemic/distributional, not ambiguity.

5. **Deep Active Learning / epistemic vs aleatoric acquisition:** Nguyen et al., "How to measure uncertainty in uncertainty sampling for active learning", Machine Learning 2022, https://link.springer.com/article/10.1007/s10994-021-06003-9 (VERIFIED existence; authors from memory: Vu-Linh Nguyen, Mohammad Hossein Shaker, Eyke Hullermeier - PARTIAL); and the epistemic-uncertainty-sampling preprint arXiv 1909.00218 (https://pith.science/paper/1909.00218; PARTIAL). Caveats from search summaries: epistemic scoring is only weakly better than entropy empirically, and noise-chasing is a documented trap.

6. **Revisiting Active Learning under (Human) Label Variation.** Cornelia Gruber et al. arXiv 2507.02593 (2025). https://arxiv.org/abs/2507.02593 . PARTIAL (first author name from memory of search snippet; check). Argues entropy sampling conflates model uncertainty with legitimate annotator disagreement; proposes acquisitions based on divergence between model entropy and predicted annotator entropy. NLP, not images.
   Also: Baumler et al., "Which Examples Should be Multiply Annotated? Active Learning When Annotators May Disagree" (DAAL), https://ctbaumler.github.io/files/DAAL.pdf . PARTIAL (venue not captured). Finds that soft vs hard labels matters more than the query strategy by end of training.

7. **Active Label Distribution Learning.** Neurocomputing, 2021 (ScienceDirect: https://www.sciencedirect.com/science/article/abs/pii/S0925231220320464). PARTIAL (authors not captured; paywalled). Active instance selection when each sample has a label distribution (mixture-like targets), query-by-committee. Closest to "mixed membership" acquisition I found.

Not found: any work specifically on mixed-membership acquisition for diffraction/materials images, or on pair acquisition under ties (tie = both samples near-equal mixture). Treat the mixture-aware acquisition rule as a novel contribution to be justified empirically, with epistemic-vs-aleatoric papers (4, 5) as motivation.

---
## Practical take-aways for the project
- Default pre-registered schedule: TypiClust/ProbCover on SimCLR features (adapted to pairs: choose pairs of images from distinct dense clusters / nearest to anchors), then switch to margin or Fisher-information pair scoring (Shen et al. 2025) after a fixed number of labels (TCM-style). Compare to random with equally tuned head (Luth et al.).
- ALBL meta-selection is justifiable but its bandit reward is noisy at ~168 labels; report it as secondary.
- If adding SSRM-style pseudo-pairs, use high threshold, minimum labeled fraction per batch, and a second-view/neighbourhood agreement check (Arazo; NeST; CRM); compare to "SSL + random" baseline as in Mittal et al.
- Evaluation: many paired seeds (init set and model seed), fixed hyperparameters chosen only on initial labeled data, held-out confirmation for the chosen rule (Cawley and Talbot for selection bias).
