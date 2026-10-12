# Literature search: T5-T8 (active preference learning for a BT reward model on RHEED images)

Method note: every entry below was found through WebSearch result pages (title, authors, venue, URL all seen there). WebFetch could not open arxiv.org (egress proxy denied the host), so I could not open full texts or abstracts myself. "Verified" means title, authors and venue agreed across several search results (arXiv, publisher, ML Anthology, etc.). Application notes marked "[my suggestion]" are my own ideas, not claims made by the papers. Support strength per topic: T5 strong, T8 moderate, T6 moderate for the method and thin for science, T7 thin for symmetry-specific work.

---

## T5. Goal-oriented / transductive active learning and Bayesian experimental design

**Support: strong.**

1. **Transductive Active Learning: Theory and Applications** (earlier arXiv title: "Information-based Transductive Active Learning"). Jonas Hubotter, Bhavya Sukhija, Lenart Treven, Yarden As, Andreas Krause. NeurIPS 2024. https://arxiv.org/abs/2402.15898 and https://proceedings.neurips.cc/paper/2024/hash/e17fe6fe9990fffb637b42c98c005515-Abstract-Conference.html
   - Introduces ITL (mutual information between the queried points and a specified target set) and VTL (a variance-based rule). Both come with convergence guarantees to the irreducible uncertainty.
   - Fit: take the target set to be the 154 ideal reference images (or the 1124 trajectory frames) and the sample space to be candidate pairs. The learner then picks the pair judgments that most reduce uncertainty about the BT scores on that target set. The paper's finding that sampling outside the target can sometimes be more informative also applies. [my suggestion] ITL needs a posterior over the head. A last-layer Laplace or ensemble approximation would give the required covariance.

2. **Prediction-Oriented Bayesian Active Learning** (EPIG). Freddie Bickford Smith, Andreas Kirsch, Sebastian Farquhar, Yarin Gal, Adam Foster, Tom Rainforth. AISTATS 2023 (PMLR 206:7331-7348). https://arxiv.org/abs/2304.08151 and https://proceedings.mlr.press/v206/bickfordsmith23a/bickfordsmith23a.pdf
   - Argues BALD scores information about parameters, not predictions. Proposes expected predictive information gain, which is measured on a target input distribution. The search result confirmed the first author, the venue and the arXiv ID. The remaining author names are from my memory of the paper and are not confirmed by the search results.
   - Fit: this is the "target distribution" version of BALD. Using the trajectory-frame distribution as the target would measure held-out preference loss on the frames we actually care about.

3. **Making Better Use of Unlabelled Data in Bayesian Active Learning.** Freddie Bickford Smith, Adam Foster, Tom Rainforth. AISTATS 2024 (PMLR 238:847-855). https://proceedings.mlr.press/v238/bickford-smith24a.html and https://arxiv.org/abs/2404.17249
   - Pairs a pretrained encoder with semi-supervised learning and EPIG. Directly relevant to the frozen-SimCLR plus 1124 unlabelled frames setup.

4. **Direct Acquisition Optimization for Low-Budget Active Learning.** Zhuokai Zhao, Yibo Jiang, Yuxin Chen. arXiv 2402.06045 (2024). A NeurIPS 2024 workshop version (BDU) is listed on ML Anthology, per search results. https://arxiv.org/abs/2402.06045 and https://openreview.net/forum?id=ODjjSqlH8j
   - Chooses points by estimated expected reduction in true loss, using influence functions. Reports that it beats other methods in low-budget regimes.
   - Fit: an expected-error-reduction criterion at the 168-label scale. It is a workshop paper and its published scope is classification.

5. **Gone Fishing: Neural Active Learning with Fisher Embeddings** (BAIT). Jordan T. Ash, Surbhi Goel, Akshay Krishnamurthy, Sham Kakade. NeurIPS 2021, pp. 8927-8939. https://arxiv.org/abs/2106.09675
   - Selects batches by minimising a bound on error that is expressed through Fisher information. This is the same family as V-optimal / I-optimal design.
   - Fit: a closed-form, last-layer Fisher criterion for a small MLP head. [my suggestion] For BT the per-pair Fisher information is p(1-p) (phi_i - phi_j)(phi_i - phi_j)^T, which makes an I/V-optimal criterion straightforward to compute.

**Preference-specific active learning (useful context for T5):**
- **Bayesian Active Learning for Classification and Preference Learning** (original BALD). Neil Houlsby, Ferenc Huszar, Zoubin Ghahramani, Mate Lengyel. arXiv 1112.5745 (2011). https://arxiv.org/abs/1112.5745. Covers GP preference learning.
- **Active Learning for Direct Preference Optimization.** Branislav Kveton, Xintong Li, Julian McAuley, Ryan Rossi, Jingbo Shang, Junda Wu, Tong Yu. arXiv 2503.01076 (2025). https://arxiv.org/abs/2503.01076. Linearises the last layer and uses D-optimal design to pick which preference feedback to collect. This is the closest analogue to a BT head on frozen features.
- **Deep Bayesian Active Learning for Preference Modeling in Large Language Models** (BAL-PM). Luckeciano C. Melo et al. NeurIPS 2024. https://arxiv.org/abs/2406.10023. It reports that naive BALD gave no gain over random selection on preference data, and that adding feature-space diversity helped. This is a warning that uncertainty-only acquisition may not beat random with so few labels.
- **Active Preference Learning for Large Language Models.** William Muldrew, Peter Hayes, Mingtian Zhang, David Barber. ICML 2024 (PMLR 235:36577-36590). https://proceedings.mlr.press/v235/muldrew24a.html. Combines predictive entropy with the certainty of the implicit preference model.

Caveat: all of these are LLM or classification settings. I found no paper on active BT learning from small pair-group labels on scientific images.

---

## T6. Time-contrastive / temporal-neighbour representation learning

**Support: moderate for general methods. Thin for scientific image sequences, and nothing found for RHEED contrastive adaptation.**

1. **Time-Contrastive Networks: Self-Supervised Learning from Video.** Pierre Sermanet, Corey Lynch, Yevgen Chebotar, Jasmine Hsu, Eric Jang, Stefan Schaal, Sergey Levine. ICRA 2018 (arXiv 1704.06888). https://arxiv.org/abs/1704.06888. An earlier version, "...from Multi-View Observation", appeared at CVPR 2017 Workshops: https://openaccess.thecvf.com/content_cvpr_2017_workshops/w5/papers/Sermanet_Time-Contrastive_Networks_Self-Supervised_CVPR_2017_paper.pdf
   - Important correction: in TCN the positives are simultaneous views from different cameras. Temporal neighbours that look alike are used as negatives. It is therefore not a "nearby frames are positives" method.
   - Fit: only the idea of a metric loss over temporal structure carries over. There are no multi-view pairs in the RHEED data.

2. **Unsupervised Feature Extraction by Time-Contrastive Learning and Nonlinear ICA.** Aapo Hyvarinen, Hiroshi Morioka. NeurIPS 2016. https://arxiv.org/abs/1605.06336
   - Trains a classifier to predict the time segment of a sample, and proves identifiability for nonlinear ICA under nonstationarity. The segment-labelling idea fits "which run / temperature band / time segment is this frame from?" as a pretext task. [my suggestion] The theory assumes nonstationary independent sources and probably does not apply to RHEED frames.

3. **Unsupervised Learning of Visual Representations using Videos.** Xiaolong Wang, Abhinav Gupta. ICCV 2015. https://arxiv.org/abs/1505.00687
   - Triplet ranking loss where patches linked by a track are positives and random patches from other videos are negatives.

4. **Slow and Steady Feature Analysis: Higher Order Temporal Coherence in Video.** Dinesh Jayaraman, Kristen Grauman. CVPR 2016. https://arxiv.org/abs/1506.04714
   - A temporal-coherence regulariser that penalises both feature change and feature acceleration over frame tuples. Useful when the frame index is known and growth is smooth.

5. **DynaCLR: Contrastive Learning of Cellular Dynamics with Temporal Regularization.** Eduardo Hirata-Miyasaki et al. (13 authors). arXiv 2410.11281 (2024). The v1 title was "Contrastive learning of cell state dynamics in response to perturbations". https://arxiv.org/abs/2410.11281. I did not find the final venue.
   - Closest scientific example: time-lapse microscopy, with nearby time points of the same tracked entity as positives. It reports smoother trajectories and few-label downstream use with temporal regularisation.
   - Fit: positives are frames from the same run within a short index window (the filename gives run id and frame index). [my suggestion] Because the SimCLR encoder is frozen, the cheapest version is a temporal-neighbour loss on a small adapter or the MLP head rather than the full encoder.

Others seen in results but not checked in detail:
- Temporal Neighborhood Coding (Tonekaboni et al.). Time-series windows in a neighbourhood are positives. Citation details are not verified here.
- TCLR (Dave et al., arXiv 2101.07974, https://arxiv.org/pdf/2101.07974), temporal contrastive learning for video. Details not checked.

**RHEED / in-situ sequence work (not contrastive, but relevant context):**
- **Machine-learning-enabled on-the-fly analysis of RHEED patterns during thin film deposition by molecular beam epitaxy.** Tiffany C. Kaspar et al. J. Vac. Sci. Technol. A 43(3), 032702 (2025). https://pubs.aip.org/avs/jva/article/43/3/032702/3341018/Machine-learning-enabled-on-the-fly-analysis-of. Uses a pretrained CNN to featurize one frame per second, then changepoint detection on the feature sequence. This is the closest thing to "pretrained features plus temporal order for RHEED".
- **Monitoring MBE Substrate Deoxidation via RHEED Image-Sequence Analysis by Deep Learning.** Khaireh-Walieh, Arnoult, Plissard, Wiecha. Crystal Growth & Design 23(2), 892-898 (2023). https://arxiv.org/abs/2210.03430. An autoencoder feeds a sequence classifier. The abstract says no temperature or rotation angle is needed.
- **Machine-learning-assisted and real-time-feedback-controlled growth of InAs/GaAs quantum dots.** Chao Shen et al. (first-name initial from memory). Nature Communications 2024. https://www.nature.com/articles/s41467-024-47087-w. A 3D ResNet-50 on RHEED video (not single frames).

Gap: I found no paper that applies temporal-neighbour contrastive fine-tuning to RHEED or to in-situ growth images. That can be stated as novelty, hedged as "we did not find".

---

## T7. Symmetry / invariance in active learning and consistency-based acquisition

**Support: moderate for augmentation-consistency acquisition. Thin for symmetry-specific or equivariance-based active learning. I found no active-learning paper that uses physical symmetries (such as mirror symmetry of diffraction) as a prior.**

1. **Consistency-Based Semi-supervised Active Learning: Towards Minimizing Labeling Cost.** Mingfei Gao, Zizhao Zhang, Guo Yu, Sercan O. Arik, Larry S. Davis, Tomas Pfister. ECCV 2020 (arXiv 1910.07153). https://arxiv.org/abs/1910.07153
   - Selects samples whose predictions are most inconsistent across augmentations, which is cheap and needs no posterior.
   - Fit: for a pair (i, j), compute the variance of the BT score difference across mirror flips and small shifts. This acquisition score is nearly free once the features are cached. [my suggestion] Because the encoder is frozen, the augmented-view features should be precomputed for every frame.

2. **Deep Active Learning with Augmentation-based Consistency Estimation.** SeulGi Hong, Heonjin Ha, Junmo Kim, Min-Kook Choi. arXiv 2011.02666 (2020), no venue found. https://arxiv.org/abs/2011.02666. Uses cutout and cutmix prediction variance for selection.

3. **Towards Controlled Data Augmentations for Active Learning** (CAMPAL). Jianan Yang, Haobo Wang, Sai Wu, Gang Chen, Junbo Zhao. ICML 2023 (PMLR 202:39524-39542). https://proceedings.mlr.press/v202/yang23p.html. Computes an acquisition score on each augmented counterpart and aggregates them, while controlling augmentation strength. Directly relevant to "augmentations as a restricted set of known-valid transforms".

4. **LADA: Look-Ahead Data Acquisition via Augmentation for Deep Active Learning.** Yoon-Yeong Kim, Kyungwoo Song, JoonHo Jang, Il-Chul Moon. NeurIPS 2021, pp. 22919-22930. https://proceedings.neurips.cc/paper_files/paper/2021/hash/c1b70d965ca504aa751ddb62ad69c63f-Abstract.html. Scores the augmented, not-yet-labelled samples at acquisition time.

5. **Test-time Data Augmentation for Estimation of Heteroscedastic Aleatoric Uncertainty in Deep Neural Networks.** Murat Seckin Ayhan, Philipp Berens. MIDL 2018. https://openreview.net/forum?id=rJZz-knjz. Test-time augmentation spread as an uncertainty estimate. Medical imaging, not active learning.

6. **Deep Bayesian Active Learning with Image Data.** Yarin Gal, Riashat Islam, Zoubin Ghahramani. ICML 2017 (arXiv 1703.02910). https://arxiv.org/pdf/1703.02910. MC-dropout BALD baseline. The author list is from my memory; the search snippet did not show it.

Symmetry as a prior (indirect support only):
- **E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials** (NequIP). Simon Batzner et al. Nature Communications 13, 2453 (2022). https://www.nature.com/articles/s41467-022-29939-5. Evidence that building in symmetry greatly reduces the labels needed in a physical-science task. It is not active learning.
- **Equivariance and Augmentation for Bayesian Neural Networks.** Miaowen Dong, Axel Flinth, Jan E. Gerken. arXiv 2606.26273 (June 2026), preprint. https://arxiv.org/abs/2606.26273. Studies data augmentation as a route to equivariance in BNNs. I only saw the abstract and secondary summaries, and the paper is very recent.
- Quantum Active Learning (arXiv 2405.18230, https://arxiv.org/html/2405.18230) was surfaced as combining equivariant models with uncertainty acquisition, but it is a quantum-ML paper and I did not check it. Treat it as a pointer only.

Caveats: (a) Augmentation-consistency acquisition is also standard in semi-supervised learning, and the empirical gains reported for it are mostly on CIFAR-scale classification. (b) BAL-PM (T5) and Gal-style BALD results suggest that uncertainty scores can be unreliable at tiny label counts. (c) Whether RHEED left-right mirror symmetry is exact depends on the geometry. The papers cannot settle this, and it should be checked on the data. Approximate-equivariance papers suggest exact symmetry may not hold.
- [my suggestion, uncited] The BT pair itself has an exact symmetry: swapping the two images flips the label. Enforcing antisymmetry of the score difference, and using swap augmentation, is free.

---

## T8. Fusing tabular process metadata with image features

**Support: moderate. The fusion methods are well documented in medical imaging. For RHEED I found no paper that fuses temperature (or other process parameters) with images in a deep model.**

1. **Combining 3D Image and Tabular Data via the Dynamic Affine Feature Map Transform.** Sebastian Polsterl, Tom Nuno Wolf, Christian Wachinger. MICCAI 2021, LNCS 12905, pp. 688-698 (arXiv 2107.05990). https://arxiv.org/abs/2107.05990
   - DAFT rescales and shifts the feature maps of a conv layer, conditioned on both the image and the tabular data.

2. **DAFT: A universal module to interweave tabular data and 3D images in CNNs.** Tom Nuno Wolf, Sebastian Polsterl, Christian Wachinger. NeuroImage 260, 119505 (2022). https://www.sciencedirect.com/science/article/pii/S1053811922006218. Journal extension with an ablation study. Code: https://github.com/ai-med/DAFT.
   - Note the author order differs between the two DAFT papers (Polsterl first on the MICCAI paper, Wolf first on the NeuroImage paper).
   - Fit: with frozen SimCLR feature vectors there are no spatial feature maps, so DAFT reduces to FiLM on MLP hidden units. [my suggestion] Using temperature, frame index and normalised time to generate a per-unit scale and shift for the MLP head is a lightweight version.

3. **FiLM: Visual Reasoning with a General Conditioning Layer.** Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, Aaron Courville. AAAI 2018 (pp. 3942-3951). https://arxiv.org/abs/1709.07871. Feature-wise affine modulation driven by conditioning input.

4. **HyperFusion: A Hypernetwork Approach to Multimodal Integration of Tabular and Medical Imaging Data for Predictive Modeling.** Daniel Duenias, Brennan Nichyporuk, Tal Arbel, Tammy Riklin Raviv. Medical Image Analysis 102, 103503 (2025). https://arxiv.org/abs/2403.13319. Argues that concatenation only mixes high-level descriptors and reports gains over it. It can serve as a more expressive alternative to DAFT.

Evidence on concatenation versus conditioning: a review snippet, https://pmc.ncbi.nlm.nih.gov/articles/PMC10288577/ ("Deep multimodal fusion of image and non-image data in disease diagnosis and prognosis: a review"), says DAFT outperformed concatenation and attention baselines, and also says the best fusion method is task- and data-dependent. The evidence comes mostly from single papers and I found no controlled multi-dataset comparison. With about 168 labels, a simple baseline (concatenate, or add a learned per-run offset) should be included, and FiLM/DAFT could overfit.

RHEED / MBE with process parameters:
- I did not find a deep model that uses substrate temperature as an input alongside RHEED images. The Khaireh-Walieh abstract states their method does not need temperature or rotation angle.
- **RHEED pattern classification by a convolutional neural network for the growth of chalcogenide thin films and nanostructures.** Nathan Muetzel et al. AIP Advances 16, 035317 (2026). https://arxiv.org/abs/2602.18243. Image-only CNN. The authors say it could eventually support temperature and shutter automation.
- **Classification of Reflection High-Energy Electron Diffraction Pattern Using Machine Learning.** Jinkwan Kwoen, Yasuhiko Arakawa. Crystal Growth & Design 20(8), 5289-5293 (2020). https://pubs.acs.org/doi/10.1021/acs.cgd.0c00506. CNN classification of GaAs RHEED patterns with a dataset of growth conditions.
- **Machine-learning-assisted and real-time-feedback-controlled growth of InAs/GaAs quantum dots** (Shen et al., Nature Communications 2024), listed under T6. RHEED video only.
- **Machine-learning-assisted thin-film growth** (arXiv 1908.00739, https://arxiv.org/pdf/1908.00739) surfaced in results, but I did not check it.

---

## Summary of gaps
- T5: well supported (ITL/VTL, EPIG, BAIT, DAO, plus preference-specific active learning). No source covers pairwise-judgment active learning on scientific images.
- T6: the generic temporal-contrastive literature is real, but TCN uses temporal neighbours as negatives and multi-view pairs as positives. DynaCLR (microscopy) is the best scientific analogue. Nothing found for RHEED, so this would be a novel adaptation.
- T7: augmentation-consistency acquisition is supported (Gao et al., CAMPAL, LADA). Physical-symmetry priors in active learning: nothing direct found, only indirect evidence (NequIP, BNN-equivariance preprint).
- T8: DAFT/FiLM/HyperFusion are solid for methods. No RHEED + temperature deep-fusion paper found.
