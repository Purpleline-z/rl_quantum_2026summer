# Literature notes: mixture-based synthetic pairs / monotone scoring (T20-T22)

Method note: WebFetch was NOT used. All entries were confirmed only through WebSearch result titles, URLs and snippets (abstract-level; no full texts read). Items I could not confirm in the results are marked UNVERIFIED. Quoted numbers are authors' claims as relayed by search snippets.

## T20. mixup, Manifold Mixup, mixup for regression / ordinal / ranking, mixup in feature space

1. Zhang, H., Cisse, M., Dauphin, Y. N., Lopez-Paz, D. "mixup: Beyond Empirical Risk Minimization." ICLR 2018 (arXiv 1710.09412).
   URL: https://arxiv.org/abs/1710.09412 (also https://iclr.cc/virtual/2018/poster/177)
   Relevance: trains on convex combinations of input pairs and their labels (vicinal risk minimisation), encouraging linear behaviour between samples. It is the direct precedent for "label of a mixture = same mixture of labels", i.e. the mixing-share to score assumption.

2. Verma, V., Lamb, A., Beckham, C., Najafi, A., Mitliagkas, I., Lopez-Paz, D., Bengio, Y. "Manifold Mixup: Better Representations by Interpolating Hidden States." ICML 2019, PMLR 97, pp. 6438-6447 (arXiv 1806.05236).
   URL: https://proceedings.mlr.press/v97/verma19a/verma19a.pdf (also https://arxiv.org/abs/1806.05236)
   Relevance: interpolates hidden representations rather than inputs; supports mixing in a learned feature space. Caveat: it interpolates inside a network being trained, not a frozen encoder.

3. Yao, H., Wang, Y., Zhang, L., Zou, J., Finn, C. "C-Mixup: Improving Generalization in Regression." NeurIPS 2022 (arXiv 2210.05775).
   URL: https://arxiv.org/abs/2210.05775 (also https://proceedings.neurips.cc/paper_files/paper/2022/hash/1626be0ab7f3d7b3c639fbfd5951bc40-Abstract-Conference.html)
   Relevance: shows vanilla mixup on continuous targets can give wrong labels, and fixes it by sampling mixing partners with similar labels (Gaussian kernel). Warns that the label-linearity assumption is not free for regression; in our case it is a physical assumption instead.

4. Noh, J., Park, H., Lee, J., Ham, B. "RankMixup: Ranking-Based Mixup Training for Network Calibration." ICCV 2023, pp. 1358-1368 (arXiv 2308.11990).
   URL: https://arxiv.org/abs/2308.11990 (also https://openaccess.thecvf.com/content/ICCV2023/papers/Noh_RankMixup_Ranking-Based_Mixup_Training_for_Network_Calibration_ICCV_2023_paper.pdf)
   Relevance: argues mixed labels may not reflect the true distribution of mixed samples, so it supervises only the ORDER (raw sample vs mixup sample, and among multiple mixup samples, via an NDCG-style loss). Closest precedent for using mixup as an ordinal/ranking signal rather than a numeric target.

Additional, frozen / embedding-space mixup:

5. Venkataramanan, S., Kijak, E., Amsaleg, L., Avrithis, Y. "Embedding Space Interpolation Beyond Mini-Batch, Beyond Pairs and Beyond Examples" (MultiMix). NeurIPS 2023 (arXiv 2311.05538).
   URL: https://arxiv.org/abs/2311.05538 (also https://proceedings.neurips.cc/paper_files/paper/2023/hash/c3532dd633e600e9f6db57aa7ae0c858-Abstract-Conference.html)
   Relevance: interpolates in embedding space with many mixtures and multi-example (not just pairwise) combinations; relevant to mixing more than two reference types.

6. Gadermayr, M., Koller, L., Tschuchnig, M. E., Stangassinger, L. M., Kreutzer, C., Couillard-Despres, S., Oostingh, G. J., Hittmair, A. "MixUp-MIL: A Study on Linear & Multilinear Interpolation-Based Data Augmentation for Whole Slide Image Classification." arXiv 2311.03052 (2023). Journal version UNVERIFIED. A shorter MICCAI 2023 version exists as "MixUp-MIL: Novel Data Augmentation for Multiple Instance Learning and a Study on Thyroid Cancer Diagnosis" (arXiv 2211.05862).
   URL: https://arxiv.org/abs/2311.03052
   Relevance: tests mixup on features from a fixed pretrained feature extractor. Search snippet says linear interpolation helped in 10 of 16 settings, so benefit is dataset-dependent. Snippet-level, so check before citing the count.

7. Wu, Y., Dong, Z., Chen, C., Zhou, W., Zhou, J. H. "SupReMix: Supervised contrastive learning for medical imaging regression with mixup." Medical Image Analysis (2025; DOI 10.1016/j.media.2025.103909, inferred from a search record, so volume/pages UNVERIFIED). Preprint arXiv 2309.16633 (earlier title "Mixup Your Own Pairs").
   URL: https://arxiv.org/abs/2309.16633
   Relevance: embedding-level mixup builds ordinal-aware hard positives/negatives for regression. Author list taken from the repo BibTeX shown in search results.

Gap: I found no paper that studies mixup in a FROZEN self-supervised feature space (SimCLR/DINO) with a ranking or Bradley-Terry head. Item 6 is the nearest. Note also that mixup of raw images and mixup of frozen features are not equivalent for a nonlinear encoder; the encoder of a pixel mixture is generally not the mixture of encodings. State this explicitly.

## T21. Linear-mixture / unmixing assumptions; ML trained on synthetic mixtures

1. Long, C. J. [first initial UNVERIFIED], et al. "Rapid identification of structural phases in combinatorial thin-film libraries using x-ray diffraction and non-negative matrix factorization." Review of Scientific Instruments 80(10), 103902 (2009).
   URL: https://pubmed.ncbi.nlm.nih.gov/19895071/ (also https://pubs.aip.org/aip/rsi/article-abstract/80/10/103902/354187/)
   Relevance: classic NMF treatment: each measured XRD pattern is a non-negative combination of basis patterns, and NMF estimates each basis pattern's contribution. Full author list not confirmed in results.

2. Stanev, V., et al. "Unsupervised phase mapping of X-ray diffraction data by nonnegative matrix factorization integrated with custom clustering." npj Computational Materials 4, 43 (2018) (arXiv 1802.07307). Volume/article number from memory, UNVERIFIED; the result snippets showed npj Comput. Mater. 2018 and the DOI-style slug s41524-018-0099-2.
   URL: https://www.nature.com/articles/s41524-018-0099-2 (also https://arxiv.org/abs/1802.07307)
   Relevance: mixes of non-negative end-member patterns with non-negative weights, plus handling of peak-shifted variants (violation of strict linearity). Same group (Stanev, Takeuchi) wrote the RHEED paper below.

3. Liang, H., Stanev, V., Kusne, A. G., Tsukahara, Y., Itou, A., Takahashi, R., Lippmaa, M., Takeuchi, I. "Application of machine learning to reflection high-energy electron diffraction images for automated structural phase mapping." Physical Review Materials 6, 063805 (2022).
   URL: https://link.aps.org/pdf/10.1103/PhysRevMaterials.6.063805 (also https://www.nist.gov/publications/application-machine-learning-reflection-high-energy-electron-diffraction-images)
   Relevance: the closest RHEED precedent. Combines supervised and unsupervised learning (U-Net segmentation of spots/streaks, grouping) to classify RHEED pattern types and extract phase-composition information. Does not appear to use a pairwise-preference or SimCLR approach. Details are from abstract-level text only.

4. Lee, J.-W., Park, W. B., Lee, J. H., Singh, S. P., Sohn, K.-S. "A deep-learning technique for phase identification in multiphase inorganic compounds using synthetic XRD powder patterns." Nature Communications 11, 86 (2020).
   URL: https://www.nature.com/articles/s41467-019-13749-3
   Relevance: trains CNNs on about 1.8 million synthetic patterns created by combinatorially mixing simulated single-phase patterns. Reports about 86% accuracy on three-step phase-fraction quantification on real data; the authors note exact fractions are hard. Direct precedent for mixtures-of-references as synthetic training data, but the targets are coarse.

5. Simonnet, T., Grangeon, S., Claret, F., Maubec, N., Fall, M. D., Harba, R., Galerne, B. "Phase quantification using deep neural network processing of XRD patterns." IUCrJ 11, 859-870 (2024).
   URL: https://journals.iucr.org/m/issues/2024/05/00/zx5030/ (also https://pmc.ncbi.nlm.nih.gov/articles/PMC11364039/)
   Relevance: regresses phase proportions from synthetic-only training mixtures (four minerals). Reported error is 0.5% on synthetic and 6% on experimental data (authors' numbers), a quantified synthetic-to-real gap that mixture-trained models should expect.

Optional (electron-diffraction unmixing, mixing premise stated explicitly):
6. "Nonnegative matrix factorization incorporating domain specific constraints for four dimensional scanning transmission electron microscopy." Scientific Reports (2025). Authors UNVERIFIED (not in search results).
   URL: https://www.nature.com/articles/s41598-025-23541-7
   Relevance: states that an experimental diffraction pattern is often a linear combination of diffraction from overlapping domains, and notes unconstrained NMF can give unphysical results (negative intensities).

Other items seen but not used: Enhancing deep-learning training for phase identification in powder X-ray diffractograms (IUCrJ 2021, https://journals.iucr.org/m/issues/2021/03/00/fc5051/) builds weighted-sum mixtures plus background/noise/orientation augmentations (authors UNVERIFIED); AIP Advances 2026 RHEED CNN classification paper (https://pubs.aip.org/aip/adv/article/16/3/035317/3382644/) is single-pattern classification, authors UNVERIFIED.

Gap: no RHEED paper found that uses synthetic LINEAR mixtures of ideal reconstruction patterns as training data, and none using pairwise preference / Bradley-Terry for RHEED. Two caveats supported by the literature: peak shifts (Stanev) and nonlinear dynamical-scattering effects mean strict linearity is an approximation; and real-data accuracy for fractions is lower than on synthetic data (Lee, Simonnet).

## T22. Monotonicity / order-consistency as inductive bias

1. Sill, J. "Monotonic Networks." NeurIPS (NIPS 10) 1997, pp. 661-667.
   URL: http://papers.neurips.cc/paper/1358-monotonic-networks.pdf (also https://papers.nips.cc/paper/1997/hash/83adc9225e4deb67d7ce42d58fe5157c-Abstract.html)
   Relevance: min-max networks of positive-weight hyperplanes give guaranteed monotonicity and are universal approximators of monotone functions. Original architecture for a score head that is monotone in its inputs (though monotone in feature coordinates, not in "mixing share" of a type unless features track it).

2. You, S., Ding, D., Canini, K., Pfeifer, J., Gupta, M. "Deep Lattice Networks and Partial Monotonic Functions." NeurIPS 2017 (arXiv 1709.06680).
   URL: https://arxiv.org/abs/1709.06680 (also https://proceedings.neurips.cc/paper/2017/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html)
   Relevance: deep models monotone in a chosen subset of inputs via calibrators and lattices. Partial monotonicity is the relevant notion if only some features/types are constrained.

3. Runje, D., Shankaranarayana, S. M. "Constrained Monotonic Neural Networks." ICML 2023, PMLR 202, pp. 29338-29353 (arXiv 2205.11775).
   URL: https://arxiv.org/abs/2205.11775 (also https://proceedings.mlr.press/v202/runje23a/runje23a.pdf)
   Relevance: weight-sign constraints plus extra activations avoid the convexity limitation; easy to attach to an MLP head on frozen features (code: https://github.com/airtai/monotonic-nn).

Pairwise / order-consistency with synthetic ordered pairs:

4. Liu, X., van de Weijer, J., Bagdanov, A. D. "RankIQA: Learning from Rankings for No-reference Image Quality Assessment." ICCV 2017 (arXiv 1707.08347).
   URL: https://arxiv.org/abs/1707.08347
   Relevance: Siamese network trained on synthetic pairs whose order is known by construction (more distortion = worse), then fine-tuned to absolute scores. Strong analogue to pairs ordered by mixing share.

5. Ma, K., Liu, W., et al. "dipIQ: Blind Image Quality Assessment by Learning-to-Rank Discriminable Image Pairs." IEEE Trans. Image Processing (2017). Author list inconsistent across search results (one lists Ma, Liu, Liu, Wang, Tao; another cites Ma, Liu, Zhang, Duanmu, Wang, Zuo), so treat authors beyond Ma and Liu, and volume/pages, as UNVERIFIED.
   URL: https://ece.uwaterloo.ca/~z70wang/publications/TIP_dipIQ.pdf (also https://dl.acm.org/doi/abs/10.1109/tip.2017.2708503)
   Relevance: RankNet trained on automatically generated discriminable pairs; only confidently ordered pairs are used. Supports excluding near-tie synthetic pairs.

6. Cao, W., Mirjalili, V., Raschka, S. "Rank consistent ordinal regression for neural networks with application to age estimation" (CORAL). Pattern Recognition Letters 140, 325-331 (2020) (arXiv 1901.07884).
   URL: https://arxiv.org/abs/1901.07884
   Relevance: guarantees rank-monotone outputs by construction (shared weights, ordered biases). Relevant if types' mixing shares are discretised into ordinal levels.

7. Zha, K., Cao, P., Son, J., Yang, Y., Katabi, D. "Rank-N-Contrast: Learning Continuous Representations for Regression." NeurIPS 2023 (spotlight) (arXiv 2210.01189).
   URL: https://arxiv.org/abs/2210.01189
   Relevance: contrastive loss that orders representations by label rank; supports the idea that features should be ordered by a continuous target (here mixing share).

Also seen: Igel, "Smooth Min-Max Monotonic Networks" (ICML 2024; arXiv 2306.01147, https://arxiv.org/abs/2306.01147), which fixes zero-gradient issues in Sill-style networks. Authors beyond Igel and exact ICML 2024 listing UNVERIFIED.

Gap: I did not find a paper that combines a Bradley-Terry/pairwise preference model with mixup-synthesised ordered pairs where order = mixing coefficient (nearest: RankMixup, RankIQA). Claiming this combination as novel is plausible but rests on a limited search.

## Support summary
- T20: well supported for mixup, Manifold Mixup, regression/ordinal/ranking variants (C-Mixup, RankMixup, SupReMix). Thin for FROZEN-feature mixup specifically (MixUp-MIL only, mixed results).
- T21: well supported for the linear-mixture/NMF premise in XRD and 4D-STEM and for ML on synthetic XRD mixtures (Lee 2020, Simonnet 2024). Thin for RHEED (only Liang 2022 for phase mapping; nothing on synthetic RHEED mixtures).
- T22: well supported for monotone network architectures and synthetic ranked-pair training (RankIQA, dipIQ). Thin for the exact combination of ordered mixture pairs with a monotone scorer.
