# Literature notes: why gains concentrate near typed anchors (RHEED preference learning)

Method: WebSearch only (arXiv/CVF/ar5iv blocked; no full-text reading). Everything below is taken from
search-result snippets/abstracts, so every item is "(abstract/snippet)" unless marked UNVERIFIED.
"Transfer" sentences are the note-taker's inference, not search-confirmed.

Setting: BT reward head on frozen SimCLR features; ~150 typed "ideal" anchor images (ranking loss: type-c
images score higher on output c than other types); Laplacian smoothness on a kNN graph of unlabelled
images. Gain only for test judgments near a typed anchor; none for judgments far from every anchor.

---------------------------------------------------------------------------------------------------
## 1. Interpolation consistency / mixup (and failure modes)

- Verma, Kawaguchi, Lamb, Kannala, Solin, Bengio, Lopez-Paz. "Interpolation Consistency Training for
  Semi-Supervised Learning". IJCAI 2019; Neural Networks 145:90-106 (2022). arXiv 1903.03825. (abstract/snippet)
  Shows: student prediction at a mix of two unlabelled inputs is trained to match the same mix of a
  mean-teacher's (EMA weights) predictions; perturbation is toward another random unlabelled sample (no
  gradient computation); reported to move decision boundaries to low-density regions; SOTA claims on
  CIFAR-10/SVHN; authors argue it is data-adaptive regularisation reducing overfitting to labelled points.
  Transfer: interpolating between unlabelled frames would enforce linearity of the reward between frames,
  but its effect on far-from-anchor judgments is not shown by this work.
- Zhang, Cisse, Dauphin, Lopez-Paz. "mixup: Beyond Empirical Risk Minimization". ICLR 2018. arXiv 1710.09412.
  (abstract/snippet) Shows: training on convex combinations of pairs of examples and labels favours simple
  linear behaviour between training examples; reports better generalisation, less memorisation of corrupt
  labels, robustness to adversarial examples. Transfer: a linearity prior between examples, not an
  anchor-distance mechanism.
- Verma et al. "Manifold Mixup: Better Representations by Interpolating Hidden States". ICML 2019 (PMLR 97,
  pp. 6438-6447). arXiv 1806.05236. (abstract/snippet) Shows: mixes hidden representations (single fixed
  layer per network); encourages less confident predictions on interpolations; learns flatter class
  representations (fewer directions of variance). Transfer: on frozen features, mixing at the feature level
  is the only available form (the input-level variant is not applicable).
- Noh, Park, Lee, Ham. "RankMixup: Ranking-Based Mixup Training for Network Calibration". ICCV 2023.
  arXiv 2308.11990. (abstract/snippet) Shows: replaces matching blended labels with a ranking constraint
  (mixed sample should have lower confidence than the raw one, margin-based "MRL"; "M-NDCG" extends to
  several mixes so confidence follows the mixing coefficient). Transfer: a ordinal/ranking constraint along
  an interpolation path is closest in form to a BT/ranking loss; no preference-learning use is shown.
- Guo, Mao, Zhang. "MixUp as Locally Linear Out-of-Manifold Regularization" (AdaMixUp). AAAI 2019. arXiv
  1809.02499. (abstract/snippet) Shows: "manifold intrusion" = under-fitting when synthetic mixed samples/
  labels conflict with real examples; arises when mixing policy is poorly tuned; AdaMixUp learns the mixing
  policy. Transfer: interpolating between distant frames in a quasi-1D RHEED trajectory could place
  synthetic points on top of real frames of another type.
- Arazo, Ortego, Albert, O'Connor, McGuinness. "Pseudo-Labeling and Confirmation Bias in Deep
  Semi-Supervised Learning". IJCNN 2020 (per GitHub repo); arXiv 1908.02983. (abstract/snippet) Shows: naive
  pseudo-labelling latches onto wrong labels (confirmation bias); mixup plus a minimum number of labelled
  samples per mini-batch reduces it; with very few labels a random batch may hold almost no labelled
  samples so training is driven by the model's own possibly wrong guesses. Transfer: this is the closest
  stated failure mode for self-generated (detached) targets with a tiny labelled seed.
- Mixup with soft labels / mixup for preference or reward models: UNVERIFIED. Search found no work applying
  mixup to BT/preference reward-model training. Only snippet: MergeMix (arXiv 2510.23479, multimodal LLM
  alignment) builds preference pairs of mixed vs raw images with a ranking-style loss; I did not verify
  details or venue. (abstract/snippet)
- Specific "detached target" failure mode of ICT (beyond confirmation bias above): UNVERIFIED, not found.

---------------------------------------------------------------------------------------------------
## 2. Label propagation / confidence / pseudo-label reliability

- Zhou, Bousquet, Lal, Weston, Scholkopf. "Learning with Local and Global Consistency". NIPS 2003
  (proceedings 2004, vol. 16, pp. 321-328). (abstract/snippet) Shows: classifier smooth over the graph
  structure of labelled+unlabelled points ("label spreading"; a later paper describes it as relaxing hard
  clamping of labelled points to a trade-off parameter applied to all points). Transfer: soft (non-clamped)
  anchors let anchor reliability be tuned; no distance decay is described in the snippet.
- Zhu, Ghahramani, Lafferty. "Semi-Supervised Learning Using Gaussian Fields and Harmonic Functions".
  ICML 2003, pp. 912-919. (abstract/snippet) Shows: Gaussian random field on a similarity graph; mean is a
  harmonic function (random-walk / electric-network interpretation); extensions for class priors and
  parameter learning by entropy minimisation. Transfer: harmonic solution is the "clamped" regime whose
  low-label degeneracy is discussed in topic 4.
- Iscen, Tolias, Avrithis, Chum. "Label Propagation for Deep Semi-supervised Learning". CVPR 2019. arXiv
  1904.04717. (abstract/snippet) Shows: alternates kNN graph on network embeddings, label propagation,
  retraining; pseudo-labels weighted per example by certainty (secondary write-ups say entropy-based) and
  per class by inverse population, claimed robust to wrong pseudo-labels. Exact formulas not seen
  (UNVERIFIED). Transfer: certainty weighting exists, but a dependence on distance to labelled seeds was
  not found in the snippets.
- Sohn et al. "FixMatch". NeurIPS 2020, arXiv 2001.07685. (abstract/snippet) Shows: pseudo-label from weak
  augmentation kept only above a high confidence threshold, trained against strong augmentation.
- Zhang et al. "FlexMatch: Boosting SSL with Curriculum Pseudo Labeling". NeurIPS 2021, arXiv 2110.08263.
  (abstract/snippet) Shows: per-class flexible thresholds from estimated learning status; strong when
  labels are extremely scarce; noted weakness under class imbalance. Transfer: confidence thresholds are
  model-confidence based, not distance-to-seed based.
- Calder, Cook, Thorpe, Slepcev. "Poisson Learning: Graph Based Semi-Supervised Learning At Very Low Label
  Rates". ICML 2020 (PMLR 119), arXiv 2006.11184. (abstract/snippet) Shows: Laplacian learning fails at
  very low label rates; replaces label-value constraints by sources/sinks and solves a graph Poisson
  equation; claimed provably more stable/informative; MNIST, FashionMNIST, CIFAR-10.
- Confidence-fragility: a 2026 arXiv paper ("Confidence-Variance Theory for Pseudo-Label Selection", arXiv
  2601.11670; authors/venue not seen) argues confident predictions can be wrong and informative
  low-confidence samples are discarded. (abstract/snippet)
- "Why pseudo-labels from graph propagation of a small seed are unreliable far from seeds": no source found
  stating this directly. UNVERIFIED. Closest support: Calder-Slepcev / Nadler et al. in topic 4 (solutions
  degrade toward uninformative far from labels) and the generic remark that label propagation struggles
  with sparse labels or poorly built graphs (overview blog snippet).
- Distance-decay / seed-reliability-weighted propagation: UNVERIFIED, no dedicated paper found.

---------------------------------------------------------------------------------------------------
## 3. Prototype / anchor methods and distance-aware uncertainty

- Snell, Swersky, Zemel. "Prototypical Networks for Few-shot Learning". NeurIPS 2017, arXiv 1703.05175.
  (abstract/snippet) Shows: class prototype = mean of support embeddings; classify by nearest prototype;
  Euclidean distance reported to beat cosine; Bregman-divergence justification. Transfer: the typed ideals
  are natural prototypes; distance to nearest prototype is a built-in far-from-anchor indicator.
- Papernot, McDaniel. "Deep k-Nearest Neighbors: Towards Confident, Interpretable and Robust Deep
  Learning". arXiv 1803.04765 (2018). (abstract/snippet) Shows: kNN over intermediate-layer representations
  of training data; neighbour-label agreement used as a conformity/confidence score, said to flag
  out-of-distribution inputs.
- Liu, Lin, Padhy, Tran, Bedrax-Weiss, Lakshminarayanan. "Simple and Principled Uncertainty Estimation with
  Deterministic Deep Learning via Distance Awareness" (SNGP). NeurIPS 2020, arXiv 2006.10108.
  (abstract/snippet; author list beyond "Liu and five co-authors" not in snippet, my listing of other
  names is from memory - UNVERIFIED) Shows: input distance awareness is a necessary condition for good
  uncertainty; SNGP = spectral normalisation + GP output layer; competitive with deep ensembles.
- van Amersfoort, Smith, Teh, Gal. "Uncertainty Estimation Using a Single Deep Deterministic Neural
  Network" (DUQ). ICML 2020, arXiv 2003.02037. (abstract/snippet) Shows: RBF-centroid model; feature
  collapse = OOD inputs mapped onto in-distribution feature regions; gradient penalty on input Jacobian
  counters it (ablation: CIFAR-10 vs SVHN AUROC 0.861 without penalty, per a secondary summary).
- Mukhoti, Kirsch, van Amersfoort, Torr, Gal. "Deep Deterministic Uncertainty: A Simple Baseline" (arXiv
  2102.11582; CVPR 2023 as "A New Simple Baseline"). (abstract/snippet) Shows: well-regularised softmax
  feature space (residual + spectral norm) + GDA density; names feature collapse as a reason feature-space
  density methods underperform.
- Transfer (all of topic 3): a frozen SimCLR feature space is not trained for sensitivity to anything
  outside the anchor set; distance to nearest typed ideal / labelled frame could serve as an explicit
  reliability gauge, but none of these works addresses a frozen encoder.

---------------------------------------------------------------------------------------------------
## 4. Spatially varying graph regularisation; low-label Laplacian degeneracy

- Nadler, Srebro, Zhou. "Statistical Analysis of Semi-Supervised Learning: The Limit of Infinite Unlabelled
  Data". NeurIPS 2009, pp. 1330-1338. (abstract/snippet) Shows: with fixed number of labelled points and
  growing unlabelled data, Laplacian regularisation is not well-posed for d >= 2 and degenerates to a
  non-informative function. (PDF copy is titled "Semi-Supervised Learning with the Graph Laplacian".)
- Zhou, Belkin. "Semi-supervised Learning by Higher Order Regularization". AISTATS 2011, pp. 892-900.
  (abstract/snippet) Shows: summarises Nadler et al. as near-constant function with spikes at labelled
  points (d >= 2); proposes iterated Laplacian (higher-order Sobolev seminorm). Condition 2m > d is a
  secondary-source claim (UNVERIFIED).
- El Alaoui, Cheng, Ramdas, Wainwright, Jordan. "Asymptotic behavior of lp-based Laplacian regularization
  in semi-supervised learning". COLT 2016 (PMLR 49:879-906), arXiv 1603.00564. (abstract/snippet) Shows:
  fixed labelled count, growing vertices; degenerate solutions when p <= d; phase transition in smoothness
  at p = d+1; p = infinity is insensitive to the data distribution; suggests p = d+1 as a compromise.
- Slepcev, Thorpe. "Analysis of p-Laplacian Regularization in Semisupervised Learning". SIAM J. Math.
  Anal. 2019. (abstract/snippet) Shows: for p > d, consistency needs graph scale eps_n to shrink fast
  enough; otherwise the problem still degenerates with spikes.
- Calder, Slepcev. "Properly-Weighted Graph Laplacian for Semi-supervised Learning". Appl. Math. Optim.
  82:1111-1159 (2020), arXiv 1810.04351. (abstract/snippet) Shows: reweights the Laplacian with heavier
  gradient penalisation near labelled points (weight built from distance to labelled set / min distance
  between labelled points), so the solution attains labelled values continuously rather than collapsing
  to a constant. Exact weight formula not seen (UNVERIFIED). This is the closest source to
  "regularisation strength depends on distance to labelled points".
- Calder et al. Poisson learning (see topic 2) targets the same low-label degeneracy.
- Zelnik-Manor, Perona. "Self-Tuning Spectral Clustering". NIPS 2004 (published 2005). (abstract/snippet)
  Shows: per-point scale sigma_i (distance to K'th neighbour; K=7 in the original per a secondary source),
  affinity exp(-d^2/(sigma_i sigma_j)). Transfer: local scaling adapts to density, not to distance from
  labels.
- Transfer: with ~150 anchors and a kNN graph, the Laplacian term may be uninformative (flat) away from
  anchors by exactly the Nadler/El Alaoui mechanism; pulling smoothness strength up near/down away from
  anchors is only partly supported by Calder-Slepcev.
- "Adaptive Laplacian weights learned jointly with the reward head" etc.: UNVERIFIED, not found.

---------------------------------------------------------------------------------------------------
## 5. Domain / run-level normalisation of features

- Li, Wang, Shi, Liu, Hou. "Revisiting Batch Normalization For Practical Domain Adaptation" (AdaBN). arXiv
  1603.04779; ICLR 2017 (workshop/ICLR listing). (abstract/snippet) Shows: label knowledge lives in weights,
  domain knowledge in BN statistics; re-estimating BN statistics on target data adapts the model; no
  parameters to tune.
- Schneider, Rusak, Eck, Bringmann, Brendel, Bethge. "Improving robustness against common corruptions by
  covariate shift adaptation". NeurIPS 2020, arXiv 2006.16971. (abstract/snippet) Shows: replacing training
  BN statistics with statistics of corrupted test images improves robustness across 25 models (ResNet-50
  ImageNet-C mCE 76.7% to 62.2%); even a single sample helps, 32 suffice to improve SOTA for ResNet-50;
  recommend reporting adapted results.
- Wang, Shelhamer, Liu, Olshausen, Darrell. "Tent: Fully Test-time Adaptation by Entropy Minimization".
  ICLR 2021, arXiv 2006.10726. (abstract/snippet) Shows: re-estimates normalisation statistics and tunes
  channel-wise affine parameters online by entropy minimisation.
- Chang, You, Seo, Kwak, Han. "Domain-Specific Batch Normalization for Unsupervised Domain Adaptation".
  CVPR 2019, arXiv 1906.03950. (abstract/snippet) Shows: separate BN branches per domain, shared other
  weights.
- Sun, Saenko. "Deep CORAL". ECCV-W 2016, arXiv 1607.01719. (abstract/snippet) Shows: aligns second-order
  statistics (covariances) of source/target activations.
- Cell-image batch effects: "Evaluating batch correction methods for image-based cell profiling" (Nature
  Communications 2024; authors not seen) includes per-batch mean-centering (BMC) as a baseline and reports
  Harmony and Seurat RPCA as top methods. A 2026 live-cell SSL paper (PubMed 41501105, authors not seen)
  keeps a separate DINO centre per batch and reports that without cross-batch training the model focused
  on technical variation. Cell-DINO (PubMed 41460955) applies sphering to well-level features. All
  (abstract/snippet).
- Transfer: for frozen SimCLR features, per-session/run mean-centring (and optionally per-run variance
  scaling) is the cheap analogue of AdaBN statistics; risk is removing real between-run differences if the
  runs differ in type composition (not addressed by these sources).

---------------------------------------------------------------------------------------------------
## 6. Evaluating away from training data; near-duplicate leakage

- Barz, Denzler. "Do We Train on Test Data? Purging CIFAR of Near-Duplicates". J. Imaging 6(6):41 (2020);
  arXiv 1902.00423. (abstract/snippet) Shows: ~3.3% of CIFAR-10 and ~10% of CIFAR-100 test images have
  near-duplicates in the training set (found by nearest neighbours in CNN feature space plus manual
  labelling); replacing them (ciFAIR) gave relative accuracy drops of 9% to 14%; they did not check
  duplicates within the training set. Transfer: nearest-neighbour search in feature space is the accepted
  way to find leakage and to stratify test items.
- Recht, Roelofs, Schmidt, Shankar. "Do ImageNet Classifiers Generalize to ImageNet?" ICML 2019
  (PMLR 97), arXiv 1902.10811. (abstract/snippet; author list from memory, UNVERIFIED) Shows: new test
  sets give 11-14% accuracy drops, attributed to distribution gap; follow-up experiments reported (via a
  secondary summary) little evidence that near-duplicate removal explains the drop.
- Roberts et al. "Cross-validation strategies for data with temporal, spatial, hierarchical, or
  phylogenetic structure". Ecography 40(8):913-929 (2017). (abstract/snippet) Shows: random CV on
  dependent data seriously underestimates prediction error; recommends block CV whenever dependence
  exists; blocks can unintentionally force extrapolation if too large; blocking in predictor space can
  help when extrapolation is the goal; a 2025 Frontiers in Remote Sensing study notes separation is
  necessary but not sufficient (block size tied to autocorrelation range).
- Bergmeir, Benitez. "On the use of cross-validation for time series predictor evaluation". Information
  Sciences 191:192-213 (2012). (abstract/snippet) Shows: recommends a blocked form of CV for time series;
  found no practical consequences of theoretical flaws in their study; a later note says plain CV is
  problematic mainly for heavily misspecified models.
- Video-derived datasets: "Find the Leak, Fix the Split: Cluster-Based Method to Prevent Leakage in
  Video-Derived Datasets" (arXiv 2511.13944; authors not seen). (abstract/snippet) Shows: random frame
  splits leak near-identical frames and inflate performance; non-consecutive frames of the same scene also
  leak; propose clustering visually similar frames before partitioning; practitioner advice is group
  splits (video/session ID). Trade-off noted: group splits reduce training variety.
- Oliver, Odena, Raffel, Cubuk, Goodfellow. "Realistic Evaluation of Deep Semi-Supervised Learning
  Algorithms". NeurIPS 2018, arXiv 1804.09170. (abstract/snippet) Shows: baselines often under-reported;
  SSL performance drops when unlabelled data contain out-of-distribution examples; re-run baselines under
  one setup.
- Stratifying test accuracy by nearest-training-neighbour distance: UNVERIFIED as a named, cited
  methodology; search returned no specific paper (only a general statement that networks can behave
  poorly far from training inputs, arXiv cs/0505021, and distance-based splitting like Kennard-Stone/SPXY
  in a Springer study). The binning recipe itself is general knowledge, not search-confirmed.

---------------------------------------------------------------------------------------------------
## What this implies for design (author's inference, NOT search-confirmed)

1. Report test judgments binned by feature-space distance to the nearest typed anchor AND to the nearest
   labelled frame (and drop near-duplicate pairs); treat the anchor-near gain and far-null as a distance
   curve, not a binary (supported by the Barz-Denzler and Roberts-style recommendations only in spirit).
2. Expect the Laplacian term to be weak far from anchors (Nadler/El Alaoui degeneracy); try a
   Calder-Slepcev-style distance-to-labelled weighting or Poisson-style source terms rather than a uniform
   lambda.
3. If pseudo-targets from graph propagation are used, weight them by a seed-distance/certainty score and
   guarantee labelled/anchor samples in every batch (Arazo et al., Iscen et al.); do not trust far-from-seed
   targets.
4. Feature-level mixup between unlabelled frames (ICT/Manifold-Mixup style) with a ranking-style constraint
   (RankMixup-like) is a candidate to extend smoothness beyond anchors, but guard against intrusion
   (AdaMixUp) by restricting mixes to graph neighbours.
5. Use anchor-distance as an explicit input/uncertainty signal (prototype/DDU-style) since a frozen SimCLR
   encoder gives no distance awareness (SNGP/DUQ concern).
6. Add per-run/session mean (and possibly variance) centring of features before the head and compare to the
   unnormalised baseline; check it does not erase type-relevant run differences.
7. Split by run/session (grouped/blocked CV) rather than by judgment or frame; verify no near-duplicate
   image crosses folds even when pairs are disjoint (pair-disjoint is not image-disjoint).
8. Anchor placement: the evidence suggests gains track anchor coverage, so adding anchors (or choosing
   them to cover far regions) is likely a stronger lever than further regularisation tuning.
