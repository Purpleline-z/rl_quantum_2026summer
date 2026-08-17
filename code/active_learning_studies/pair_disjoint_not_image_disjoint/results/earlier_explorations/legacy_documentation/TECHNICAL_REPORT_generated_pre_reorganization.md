# RHEED Active Learning and Reconstruction Classification: Technical Report

**Status:** Generated from current code configuration and result artifacts.
**Purpose:** A reproducible technical reference for research-group review, GitHub, and Google Colab runs.
**Primary report generator:** `active learning code/generate_technical_report.py`

## 1. Project Goal

The project studies whether, under a limited human-label budget, an active-learning selector can choose **pairwise RHEED comparison candidates** that most improve downstream classification of ideal RHEED reconstruction types. A candidate is an unordered pair of trajectory images; acquiring it reveals its human preference rows for Bradley-Terry training. The downstream metric is fixed ideal-image classification accuracy.

For a selected pair $p$ and currently labelled set $L$, batch utility is evaluated as:

$$U(p \mid L) = \operatorname{Acc}(L \cup \{p\}) - \operatorname{Acc}(L).$$

This is deliberately distinct from pairwise preference accuracy. The latter measures fit to human comparison rows; the former measures the downstream ideal-image task.

## 2. Project Layout and Reproducibility

```text
project_root/
+-- data/                         stable images, labels, and configuration
+-- code/                         code, SimCLR checkpoint, documents, tests, results
    +-- active learning code/
    +-- classifier 2 context/
    +-- result/
```

`data/` is immutable input during ordinary runs. `code/result/` contains all generated manifests, metrics, plots, logs, and reports. The runtime data path is the sibling `../data` by default and can be overridden with `RHEED_DATA_ROOT` or `--data-root`; generated files never use the data directory.

For Colab, keep `data/` once in Drive, unpack only `code.zip` into `/content`, and set:

```bash
export RHEED_DATA_ROOT=/content/drive/MyDrive/rheed_project/data
```

The T4 validation runner writes large resumable model/optimizer checkpoints only when `--max-runtime-minutes > 120`. The default 50-minute job records lightweight JSON state and results but no `.pth` checkpoint.

## 3. Data

### 3.1 Sources

| Asset | Current count | Source artifact |
| --- | ---: | --- |
| Pairwise comparison rows | 678 | `data/original data/Quantum Label Data - Pairwise_Comparisonv1.8.csv` |
| Valid pairwise rows in revision audit | 669 | `result/professor_revision/technical_report.md` |
| Unique unordered pairs in revision audit | 179 | `result/professor_revision/technical_report.md` |
| Dataset images | 1,278 | `result/pca_tsne_dataset_exploration/manifest.csv` |
| Labeled ideal images | 154 | `result/pca_tsne_dataset_exploration/manifest.csv` |
| Unlabeled trajectory images | 1,124 | `result/pca_tsne_dataset_exploration/manifest.csv` |

### 3.2 Pairwise comparison rows by reconstruction type

| type | comparison rows |
| --- | --- |
| Root-13 (sqrt(13) x sqrt(13)) | 157 |
| (1 x 1) | 140 |
| HTR | 137 |
| c(6 x 2) | 120 |
| Twinned(2 x 1) | 115 |
| Other | 9 |

Rows that cannot be mapped to a supported reconstruction type, have unsupported winners, or reference missing images are excluded by the active-learning loader. Pair partitions are disjoint by unordered pair, but images may appear in multiple different pairs; this is audited rather than treated as image-disjoint data.

## 4. Model and Training

```text
grayscale RHEED image
        | resize 224 x 224; repeat to RGB; normalize (mean 0.5, std 0.25)
        v
frozen-initialized SimCLR ResNet-18 encoder (512-dimensional embedding)
        v
full-model fine-tuning
        v
Bradley-Terry reward head: Linear(512, 256) -> ReLU -> Dropout -> Linear(256, 5)
        v
one reconstruction-specific reward per image
```

The encoder checkpoint is `classifier 2 context/simclr_resnet18_encoder.pth`. The encoder is initialized from this checkpoint and the full model is fine-tuned with AdamW (learning rate 1e-4, weight decay 1e-4, batch size 16). Current controlled runs fix epoch count at 3. Tolerance is not an exposed parameter; all model layers are trainable.

Known downstream fine-tuning augmentation is random affine rotation +/-5 degrees, translation +/-5%, scale 0.95-1.05, and brightness/contrast jitter 0.2 (`classifier 2 context/train_unified.py`). The SimCLR pretraining recipe itself was not located in the current repository, so it is intentionally not inferred.

For winner labels, the loss is Bradley-Terry negative log likelihood for image 1/image 2 wins, absolute reward difference for ties, and a non-positive reward penalty for not-applicable labels. Ideal reference anchors set an absolute reward scale; reference images never overlap ideal test images.

## 5. Evaluation

An ideal test image is classified by comparing its reconstruction-specific reward against held-out ideal reference images of each class, using Bradley-Terry win rates. It is not classified by raw reward argmax. This evaluates ideal reconstruction recognition only; it does not establish accuracy on real online images with haze, mixture, changing substrate angle, electron-beam energy, or exposure settings.

## 6. Exploratory Representation Analysis

| Representation | 5-NN CV accuracy | Full-space silhouette | PCA variance (2D) |
| --- | --- | --- | --- |
| raw_pixels | 0.832 | -0.135 | 0.965 |
| imagenet_resnet18 | 0.896 | 0.071 | 0.374 |
| rheed_simclr_resnet18 | 0.884 | 0.092 | 0.762 |

The domain-specific SimCLR representation achieved 0.884 5-NN cross-validation accuracy on the ideal subset, compared with 0.896 for ImageNet ResNet-18 and 0.832 for raw pixels. PCA/t-SNE are exploratory visualization evidence only: t-SNE can exaggerate apparent separation and does not validate downstream classifier performance.

Main figure: `result/pca_tsne_dataset_exploration/figures/rheed_simclr_resnet18_tsne.png`.
Metrics: `result/pca_tsne_dataset_exploration/metrics/representation_metrics.csv`.

## 7. Active-Learning Protocols

### 7.1 Historical exploratory sequential runs

The early sequential experiments used 70 initial pairs, a 100-pair candidate pool, a 30-image ideal test set, and two-pair acquisition batches. They are exploratory, use several epoch/budget configurations, and are not pooled with controlled single-shot results.

### 7.2 Professor fixed-protocol validation

The validation endpoint uses 50 initial pairs, a 120-pair pool, one 100-pair single-shot acquisition, a fixed ideal test/reference policy, and three seeds (42, 79, 123). Epoch 3 was selected before the final strategy comparison; epoch is a training configuration, not a scientific result axis.

| Strategy | Seeds | Post-test accuracy | Batch utility |
| --- | --- | --- | --- |
| uncertainty | 3 | 0.744 +/- 0.077 | 0.311 +/- 0.038 |
| cluster_diverse | 3 | 0.522 +/- 0.139 | 0.089 +/- 0.252 |
| random | 3 | 0.511 +/- 0.077 | 0.078 +/- 0.168 |
| uncertainty_diversity | 3 | 0.544 +/- 0.107 | 0.111 +/- 0.184 |
| core_set | 3 | 0.478 +/- 0.139 | 0.044 +/- 0.234 |

Under this three-seed protocol, uncertainty had the strongest directional result. This is not a definitive significance claim because three seeds are insufficient for reliable statistical inference.

Source: `result/professor_validation/final_comparison/summary.csv`; figure: `result/professor_validation/final_comparison/selection_plot.png`.

### 7.3 Fifteen-seed endpoint extension

| Strategy | Seeds | Post-test accuracy | Batch utility |
| --- | --- | --- | --- |
| random | 15 | 0.460 +/- 0.130 | 0.082 +/- 0.208 |
| uncertainty_diversity | 15 | 0.489 +/- 0.159 | 0.111 +/- 0.248 |
| cluster_diverse | 15 | 0.427 +/- 0.121 | 0.049 +/- 0.222 |
| core_set | 15 | 0.422 +/- 0.179 | 0.044 +/- 0.205 |
| uncertainty | 15 | 0.522 +/- 0.188 | 0.144 +/- 0.255 |

At the 100-pair endpoint, uncertainty has the highest mean post-test accuracy (0.522 +/- 0.188) and batch utility (0.144 +/- 0.255). Large standard deviations remain central to interpretation: this supports an average advantage, not a settled strategy ranking for every run.

Source: `result/professor_followup_analysis/fifteen_seed_extension/fifteen_seed_summary.csv`; figure: `result/professor_followup_analysis/fifteen_seed_extension/fifteen_seed_strategy_comparison.png`; paired differences: `result/professor_followup_analysis/fifteen_seed_extension/paired_differences.png`.

## 8. Controlled Performance-versus-Budget Revision

The professor revision fixes epoch 3, learning rate 1e-4, full-model fine-tuning, five seeds (42, 79, 123, 202, 303), a 50-pair initial set, and a 120-pair candidate pool. It compares random and uncertainty at acquired-pair budgets 10, 25, 50, 75, and 100. Within each seed, the initial set and candidate pool are held fixed across methods and budgets.

| Budget | Strategy | Seeds | Post-test accuracy | Batch utility |
| --- | --- | --- | --- | --- |
| 10 | random | 5 | 0.413 +/- 0.065 | -0.013 +/- 0.102 |
| 10 | uncertainty | 5 | 0.387 +/- 0.084 | -0.040 +/- 0.172 |
| 25 | random | 5 | 0.433 +/- 0.127 | 0.007 +/- 0.207 |
| 25 | uncertainty | 5 | 0.440 +/- 0.126 | 0.013 +/- 0.202 |
| 50 | random | 5 | 0.327 +/- 0.116 | -0.100 +/- 0.127 |
| 50 | uncertainty | 5 | 0.300 +/- 0.113 | -0.127 +/- 0.192 |
| 75 | random | 5 | 0.293 +/- 0.043 | -0.133 +/- 0.141 |
| 75 | uncertainty | 5 | 0.380 +/- 0.141 | -0.047 +/- 0.177 |
| 100 | random | 5 | 0.493 +/- 0.076 | 0.067 +/- 0.176 |
| 100 | uncertainty | 5 | 0.647 +/- 0.202 | 0.220 +/- 0.258 |

The current curve is noisy and non-monotonic, so it must not be interpreted as evidence that larger budgets inherently reduce performance. At budget 100, uncertainty has a higher mean than random in this five-seed curve, but the standard deviation is large. Results are single-shot acquisitions and should not be combined with the historical sequential results or the 15-seed endpoint extension.

Per-seed results: `result/professor_revision/per_seed_results.csv`.
Summary: `result/professor_revision/budget_summary.csv`.
Figures: `result/professor_revision/performance_vs_budget.png` and `result/professor_revision/utility_vs_budget.png`.
Completed run artifacts: 50 `result.json` files.

## 9. Diversity, Coverage, and Mixture Diagnostics

`cluster_quota_uncertainty` is the previous custom cluster-diverse quota heuristic, not a standard Cluster-Margin algorithm. `cluster_margin_pairwise` is exploratory: it clusters frozen SimCLR pair embeddings, prefilters low Bradley-Terry-margin pairs, then selects round-robin from smaller nonempty clusters. It is not expected to outperform uncertainty in this low-budget regime.

Core-set uses full-dimensional frozen pair embeddings and farthest-first geometric coverage. Its behavior is not contradicted by overlap in a two-dimensional PCA plot, because PCA is a global projection whereas core-set optimizes local high-dimensional distance. Current redundancy analysis found no clear redundancy-utility relationship; lower image reuse or pair similarity alone does not establish that a diversity constraint will improve downstream utility.

Sources: `result/professor_followup_analysis/redundancy_analysis/redundancy_utility_accuracy.csv`, `result/professor_followup_analysis/core_set_analysis/interpretation.md`.

## 10. Deployment-Oriented Extensions and Limits

| Item | Current status | Interpretation boundary |
| --- | --- | --- |
| Feature-space NMF mixture surrogate | Implemented as an exploratory diagnostic | Entropy is not a physical mixture fraction; residual is a separate off-basis signal. |
| Metadata fusion | Scaffold implemented; no populated CSV supplied | Fusion is disabled without data; metadata OOD acquisition is not feature fusion. |
| Symmetry preprocessing | `none`, `left_half_mirror`, and `symmetric_average` supported | Must be compared consistently across training, scoring, anchors, and evaluation. |
| Temporal rules | Stored as tentative, disabled constraints | Physics team confirmation is required before applying them to inference/acquisition. |
| Stream-based system | Design direction only | Future single-image interception must form pairs from temporally adjacent frames in the same trajectory. |

The most important deployment limitation is realism: simulator-generated ideal images can enlarge an evaluation set, but must be compared explicitly with online experimental images that contain haze and multiple reconstruction components. The downstream metric should ultimately measure usefulness for growth control, not only pristine ideal-image classification.

## 11. Next Steps

1. Audit the simulated-ideal versus online-image domain gap before expanding the test set.
2. Complete and replicate the five-seed budget curve; use per-seed lines and sample standard deviations.
3. Evaluate a populated metadata-fusion dataset using the existing data contract.
4. Audit whether reference-only NMF components have meaningful class coverage before enabling mixture-driven acquisition.
5. Run matched symmetry ablations at budget 100.
6. Confirm proposed temporal constraints with the physics team before constraining a classifier or selector.
7. Define a deployment metric aligned with growth control, then discuss stream-based/RL-copilot design with the broader team.

## 12. Artifact Index

| Purpose | Artifact |
| --- | --- |
| Main report | `TECHNICAL_REPORT.md` |
| Existing revision audit | `result/professor_revision/technical_report.md` |
| Representation report | `result/pca_tsne_dataset_exploration/report.md` |
| Fixed-protocol summary | `result/professor_validation/final_comparison/summary.csv` |
| Fifteen-seed summary | `result/professor_followup_analysis/fifteen_seed_extension/fifteen_seed_summary.csv` |
| Revision manifests and results | `result/professor_revision` |
