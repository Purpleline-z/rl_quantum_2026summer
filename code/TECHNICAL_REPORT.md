# Technical Report: RHEED Pairwise Active Learning

## Purpose and scope

This is the maintained project report for review and reproduction. It inventories legacy materials and current controlled studies instead of pooling incompatible experiments. Every current value has a corresponding artifact under [`results/published`](results/published). Historical job states, caches, and checkpoints remain locally in `results/local_runs`.

The question is: under a limited label budget, which candidate **pairwise RHEED comparisons** should be revealed so retraining improves classification of ideal reconstruction types? For labeled pairs \(L\) and candidate \(p\), utility is \(U(p\mid L)=Acc(L\cup\{p\})-Acc(L)\), where accuracy is fixed ideal-image test accuracy after retraining. Selection is pairwise; evaluation is image classification.

## Repository, data, model, and evaluation

The stable `data/` directory is a sibling of this `code/` directory. `src/project_paths.py` defaults to `../data` and supports `RHEED_DATA_ROOT` or `--data-root`; generated work always writes to `results/local_runs`.

The current exploration report records 1,278 images: 154 labeled ideal images in five reconstruction folders and 1,124 trajectory images. The controlled-study audit records 669 valid pairwise rows and 179 unique unordered pairs: Root 13 157, 1x1 140, HTR 137, c(6x2) 120, and Twin 115.

The model is a pretrained SimCLR ResNet-18 encoder plus a 512-to-256-to-5 Bradley-Terry reward model. Controlled studies use full-model AdamW fine-tuning (learning rate 1e-4, weight decay 1e-4, batch size 16, three epochs). Tolerance is not exposed. Known downstream augmentation in `classifier2/downstream_classifier_scripts/train_unified.py` is affine rotation +/-5 degrees, translation +/-5%, scale 0.95-1.05, and brightness/contrast jitter 0.2. The checkpoint is `classifier2/simclr_resnet18_encoder.pth`; no SimCLR pretraining routine was found, so its recipe is not inferred.

Pair partitions are disjoint by unordered pair, but an image can recur across pairs. Immutable manifests audit pair/image/test overlap, source hashes, configuration, and selected IDs. Pair-disjoint therefore does not mean image-disjoint.

## Result families and reconciliation

| Family | Protocol and role | Current conclusion? | Curated evidence |
|---|---|---:|---|
| Classifier2 | Older data/model/evaluation version; implementation context only. | No | `classifier2/TECHNICAL_REPORT.md` |
| Representation exploration | PCA/t-SNE and representation diagnostics. | Context only | `results/published/representation_exploration/` |
| Sequential exploration | 70 initial pairs, 100-pair pool, 30-image test, two-pair batches. | No | `results/published/legacy_context/` |
| Fixed protocol, 3 seeds | 50 initial pairs, 120-pair pool, single 100-pair acquisition. | Directional evidence | `results/published/fixed_protocol_3seed/` |
| Endpoint extension, 5 seeds | Same controlled budget-100 endpoint over five seeds. | Primary endpoint evidence | `results/published/endpoint_extension_5seed/` |
| Endpoint extension, 15 seeds | Same endpoint over a larger seed population. | Supporting evidence | `results/published/endpoint_extension_15seed/` |
| Budget curve, 5 seeds | Fixed protocol across budgets 10, 25, 50, 75, 100. | Primary budget evidence | `results/published/budget_curve_5seed/` |
| Diversity and coverage | Redundancy and projection diagnostics. | Explanatory only | `results/published/diversity_and_coverage/` |

The five-seed budget-100 point is shared between the endpoint extension and budget curve: it is one result shown in two views, not an independent replicate. The 15-seed endpoint is a larger seed population, not a contradiction. Smoke jobs, audits, manifests, and utility caches are reproducibility artifacts, not evidence. Classifier2 is not numerically comparable with the newer studies.

## Representation exploration

On the ideal-image subset, 5-nearest-neighbor cross-validation accuracy was 0.832 for raw pixels, 0.896 for ImageNet ResNet-18 features, and 0.884 for domain-specific SimCLR features. Sources: [`representation_metrics.csv`](results/published/representation_exploration/tables/representation_metrics.csv) and [`simclr_resnet18_tsne.png`](results/published/representation_exploration/figures/simclr_resnet18_tsne.png).

PCA/t-SNE are exploratory representation evidence only. t-SNE can exaggerate separation and does not establish classifier performance, selection utility, or significance.

## Controlled active-learning results

At epoch 3, the three-seed fixed protocol gave uncertainty 0.744 +/- 0.077 post-test accuracy and +0.311 +/- 0.038 batch utility. Random gave 0.511 +/- 0.077 and +0.078 +/- 0.168. The custom K-means quota heuristic, uncertainty plus diversity, and core-set had mean accuracy 0.522 +/- 0.139, 0.544 +/- 0.107, and 0.478 +/- 0.139. Sources: [`strategy_summary.csv`](results/published/fixed_protocol_3seed/tables/strategy_summary.csv), [`per_seed_outcomes.csv`](results/published/fixed_protocol_3seed/tables/per_seed_outcomes.csv), and [`post_test_accuracy_by_epoch.png`](results/published/fixed_protocol_3seed/figures/post_test_accuracy_by_epoch.png). Epoch 3 was selected because uncertainty peaked before later epochs showed overfitting signs; epoch is a model setting, not an acquisition axis. Three seeds are directional, not definitive significance evidence.

At the shared five-seed budget-100 endpoint, uncertainty achieved 0.647 +/- 0.202 accuracy and +0.220 +/- 0.258 utility, versus random at 0.493 +/- 0.076 and +0.067 +/- 0.176. Source: [`endpoint_5seed_strategy_summary.csv`](results/published/endpoint_extension_5seed/tables/endpoint_5seed_strategy_summary.csv).

At 15 seeds, uncertainty remained higher on average: 0.522 +/- 0.188 accuracy and +0.144 +/- 0.255 utility, versus random at 0.460 +/- 0.130 and +0.082 +/- 0.208. Sources: [`endpoint_15seed_strategy_summary.csv`](results/published/endpoint_extension_15seed/tables/endpoint_15seed_strategy_summary.csv) and [`post_test_accuracy_by_strategy.png`](results/published/endpoint_extension_15seed/figures/post_test_accuracy_by_strategy.png). The large sample standard deviations are central: the evidence supports an average advantage, not a fixed ordering in every run.

The five-seed curve is reported directly in [`strategy_performance_by_acquisition_budget.csv`](results/published/budget_curve_5seed/tables/strategy_performance_by_acquisition_budget.csv), [`post_test_accuracy_by_acquisition_budget.png`](results/published/budget_curve_5seed/figures/post_test_accuracy_by_acquisition_budget.png), and [`batch_utility_by_acquisition_budget.png`](results/published/budget_curve_5seed/figures/batch_utility_by_acquisition_budget.png). Its intermediate budgets are noisy and non-monotonic; it does not show that every additional pair improves every seed.

## Diversity, coverage, and limitations

The custom K-means quota selector is named `cluster_quota_uncertainty`; it is not a standard algorithm. The exploratory Cluster-Margin selector is also not claimed to beat uncertainty in this low-budget setting.

The redundancy analysis found no clear similarity/reuse–utility relationship. Thus redundancy alone neither explains the utility/accuracy mismatch nor establishes that diversity constraints help. PCA overlap does not validate core-set selection: core-set uses full-space local Euclidean coverage, while PCA is a global 2D projection. Sources: [`redundancy_utility_accuracy.csv`](results/published/diversity_and_coverage/tables/redundancy_utility_accuracy.csv), [`pair_similarity_vs_batch_utility.png`](results/published/diversity_and_coverage/figures/pair_similarity_vs_batch_utility.png), and [`core_set_interpretation.md`](results/published/diversity_and_coverage/core_set_interpretation.md).

Simulated ideal images are pristine and may not represent online RHEED with haze, mixtures, substrate-angle changes, beam-energy changes, or exposure changes. Ideal-image accuracy therefore is not deployment proof for growth control.

## Exploratory extensions and next steps

Feature-space NMF uses frozen encoder activations, excludes ideal-test images, and fits a rank-four nonnegative basis using allowed ideal references. Coefficient entropy is a mixture surrogate and reconstruction residual is a separate off-basis signal; neither is an expert label or physical mixture fraction. Metadata fusion requires a populated CSV; symmetry modes are ablations. Tentative temporal rules are stored as configuration and are not applied without physics confirmation.

Under the controlled single-shot protocol, uncertainty sampling is the most promising selector. Open questions are robustness across budgets/runs, the simulation-to-online gap, and whether metadata, mixture structure, symmetry, and temporally adjacent live-image pairs improve the real growth-control objective. No stream-based or generative active-learning method has been selected.
