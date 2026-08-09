# Technical Report: RHEED Pairwise Active Learning

## Purpose and evidence boundary

This report documents the current RHEED active-learning evidence and directly answers the review questions raised about it. The canonical publication repository is `Purpleline-z/rl_quantum_2026summer`. Curated figures and tables are under [`results/completed_experiments`](results/completed_experiments); complete historical job artifacts remain under `results/local_runs`.

The active-learning unit is an unordered pair of trajectory images with human preference rows. The downstream quantity is ideal-image classification accuracy after retraining. For labeled pairs \(L\) and candidate pair \(p\), the experimental utility is

\[
U(p\mid L)=Acc(L\cup\{p\})-Acc(L).
\]

Selection is pairwise; evaluation is image classification. This report does not claim that ideal-image accuracy establishes real online growth-control performance.

## Direct Responses to Active-Learning Review Questions

### 1. What is the main active-learning graph?

The main result is performance against **acquisition budget**, not epoch:

- [`post_test_accuracy_by_acquisition_budget.png`](results/completed_experiments/budget_curve_5seed/figures/post_test_accuracy_by_acquisition_budget.png)
- [`batch_utility_by_acquisition_budget.png`](results/completed_experiments/budget_curve_5seed/figures/batch_utility_by_acquisition_budget.png)
- [`strategy_performance_by_acquisition_budget.csv`](results/completed_experiments/budget_curve_5seed/tables/strategy_performance_by_acquisition_budget.csv)
- [`per_seed_performance_by_acquisition_budget.csv`](results/completed_experiments/budget_curve_5seed/tables/per_seed_performance_by_acquisition_budget.csv)

The five budgets are 10, 25, 50, 75, and 100 acquired pair groups. Each point aggregates reproducibility seeds 42, 79, 123, 202, and 303. The controlled protocol fixes epoch count at 3 and learning rate at `1e-4`; epoch is a model-training setting, not the active-learning treatment axis. Seed is a reproducibility setting, not a scientific treatment. Tolerance is not an exposed parameter, and the full model is trainable.

The result is noisy and non-monotonic. Mean post-test accuracy for random/uncertainty was respectively 0.413/0.387 at budget 10, 0.433/0.440 at 25, 0.327/0.300 at 50, 0.293/0.380 at 75, and 0.493/0.647 at 100. These are descriptive multi-seed means, not evidence that each additional pair must improve every run.

### 2. What does “pretraining” mean here?

There are two distinct stages.

**SimCLR encoder pretraining** is self-supervised contrastive learning on RHEED images, not pairwise preference learning. The downstream model starts from [`classifier2/simclr_resnet18_encoder.pth`](classifier2/simclr_resnet18_encoder.pth). The local Classifier2 documentation describes the encoder as SimCLR-pretrained on unlabeled RHEED images, but no executable SimCLR training entry point or run manifest is available in this repository.

The upstream commit `bc19e167586da25ac314b5aa7c116e8d48141723` of Justin Meng's `ymeng3/Quantum` repository was also audited. It contains the encoder checkpoint and downstream scripts, but no SimCLR training entry point in Classifier1 or Classifier2. The detailed audit is [`classifier2/pretraining/UPSTREAM_SOURCE.md`](classifier2/pretraining/UPSTREAM_SOURCE.md). Therefore the training script may define a possible input population in another unprovided location, but the exact image population used to produce the shipped checkpoint is not established by the available checkpoint provenance. No image count, SimCLR augmentation recipe, optimizer, epochs, or batch size is asserted here.

The local checkpoint has 120 sequential ResNet-18 tensor keys and is structurally compatible with the encoder consumed by the downstream Bradley--Terry model. This is an architecture/state-dictionary compatibility check only; it does not prove checkpoint provenance.

**Downstream initial pairwise training** is separate. Every budget-curve run starts from 50 labeled unordered pair groups; the Bradley--Terry model is then fine-tuned using those preference rows plus the newly acquired rows. The number of initial revealed rows is seed-dependent rather than a universal “100+ pairs” number. For example, the seed-42 budget-10 run records 50 initial pair groups and 193 initial revealed preference rows; the selected 10 pair groups add 35 rows. Exact run-level values are stored in each `result.json` under [`results/local_runs/budget_curve_study/jobs`](results/local_runs/budget_curve_study/jobs).

The downstream architecture is SimCLR ResNet-18 encoder plus a 512-to-256-to-5 Bradley--Terry reward head. It uses full-model AdamW fine-tuning (learning rate `1e-4`, weight decay `1e-4`, batch size 16, three epochs). Known downstream augmentation in `classifier2/downstream_classifier_scripts/train_unified.py` is affine rotation +/-5 degrees, translation +/-5%, scale 0.95-1.05, and brightness/contrast jitter 0.2. This downstream augmentation must not be presented as the missing SimCLR pretraining recipe.

### 3. Which pairs were held out, and do pair partitions overlap?

The source audit records 669 valid pairwise preference rows and 179 unique unordered pair groups. In each seed, a reproducible random ordering creates 50 initial labeled pair groups and a 120-pair candidate pool. The remaining 9 pair groups are not in that run's initial set or candidate pool. At the stated budget, the selector chooses 10, 25, 50, 75, or 100 candidate pair groups and only then reveals their human preference rows.

Initial, candidate, and selected sets are disjoint **by unordered pair**. Candidate labels are hidden from the selector. They are not image-disjoint: a trajectory image can appear in several different pair groups. This is a limitation, not a leakage pass. The full pair IDs are in the run-local `pairwise_for_pretrain/pretrain_pairs.csv`, `pairwise_for_candidate_pool/candidate_pairs.csv`, and `selected_pairs.csv` files.

| Audit item | Recorded controlled-protocol value |
|---|---:|
| Valid pairwise rows / unique unordered pairs | 669 / 179 |
| Initial labeled groups / candidate groups / unused groups | 50 / 120 / 9 |
| Selected groups | 10, 25, 50, 75, or 100 |
| Exact initial-pool overlap / initial-selected overlap | 0 / 0 |
| Pairwise-image overlap with ideal outer test | 0 |
| Ideal reference--outer-test overlap | 0 |
| Pairwise partitions image-disjoint? | No |

The concrete overlap counts vary by seed and selected batch. For example, seed 42 at budget 10 has 28 images reused between the initial and candidate pair partitions, and 1 image reused between the initial and selected partitions. The authoritative audit for that run is [`result.json`](results/local_runs/budget_curve_study/jobs/seed-42_budget-10_strategy-random_symmetry-none/result.json). The same audit fields are available for every seed, budget, and strategy; therefore a reviewer can reproduce any row without relying on an aggregate plot.

### 4. What is validation, and what is the final test?

There is **no separate pairwise validation partition** in the controlled active-learning protocol. The three relevant data roles are:

1. Pairwise Bradley--Terry training rows: initial revealed rows plus the acquired candidate rows.
2. Ideal-image utility-validation images: used internally to estimate candidate utility, separate from reference and outer-test images.
3. Ideal-image reference anchors and an untouched ideal-image outer test: used for the final post-acquisition evaluation.

The existing budget-curve audits record 30 outer-test images and 120 reference images. The outer-test class allocation in the existing manifest is 8 `(1 x 1)`, 8 `c(6 x 2)`, 8 `RT13`, and 6 `HTR`; the reference allocation is 33, 34, 30, and 23 respectively. See [`ideal_test.csv`](results/local_runs/budget_curve_study/jobs/seed-42_budget-10_strategy-random_symmetry-none/manifests/ideal_for_test/ideal_test.csv).

The historical manifest referenced above unfortunately omits the utility-validation rows despite the current runner writing them for new runs. Consequently, its exact historical utility-validation count and per-class allocation cannot be recovered from the preserved manifest alone; this is an audit limitation, not a value to infer. Future runs write `utility_validation` entries into the same ideal split manifest, and the disjointness audit records reference/test, utility/test, and reference/utility overlap. The recorded outer-test and reference overlap is zero, and the recorded pairwise-image/outer-test overlap is zero.

### 5. Why can budget 10, especially random selection, change so much?

Budget 10 adds only ten pair groups to an initial set of 50. Pair groups contain varying counts of preference rows, random selections differ across reproducibility seeds, and full-model fine-tuning introduces run-to-run variation. In addition, the outer ideal test contains 30 images, so accuracy changes in increments of `1/30 = 0.0333`. A one-image difference is visibly large on this metric. These mechanisms make variation plausible; they do not provide a causal decomposition.

The budget curve shows why accuracy should not be interpreted as monotonic in budget. The intermediate means decrease for both strategies at some budgets, while the budget-100 uncertainty mean is highest. No p-values, confidence intervals, causal attribution, tolerance sweep, or trainable-layer ablation is claimed.

## Result families and reconciliation

| Family | Protocol and role | Included in current conclusion? | Evidence |
|---|---|---:|---|
| Classifier2 | Older data/model/evaluation version; implementation context. | No | `classifier2/TECHNICAL_REPORT.md` |
| Representation exploration | PCA/t-SNE and kNN representation diagnostics. | Context only | `results/completed_experiments/representation_exploration/` |
| Sequential exploration | 70 initial groups, 100-pair pool, two-pair batches, 30-image test. | No | `results/completed_experiments/legacy_context/` |
| Fixed protocol, 3 seeds | 50 initial groups, 120-pair pool, single budget-100 acquisition. | Directional evidence | `results/completed_experiments/fixed_protocol_3seed/` |
| Endpoint extension, 5 seeds | Controlled budget-100 endpoint. | Primary endpoint evidence | `results/completed_experiments/endpoint_extension_5seed/` |
| Endpoint extension, 15 seeds | Same endpoint with more seeds. | Supporting endpoint evidence | `results/completed_experiments/endpoint_extension_15seed/` |
| Budget curve, 5 seeds | Budgets 10--100, fixed protocol. | Primary result | `results/completed_experiments/budget_curve_5seed/` |
| Diversity and coverage | Redundancy and PCA diagnostics. | Explanatory only | `results/completed_experiments/diversity_and_coverage/` |

The five-seed budget-100 endpoint is shared between the endpoint-extension and budget-curve views; it is not an independent replication. The 15-seed endpoint is a larger seed population, not a contradiction. Smoke jobs, caches, and manifests are reproducibility artifacts rather than scientific evidence. Sequential exploration is not pooled with controlled evidence.

## Other evidence and limitations

On the ideal-image subset, 5-nearest-neighbor cross-validation accuracy was 0.832 for raw pixels, 0.896 for ImageNet ResNet-18 features, and 0.884 for domain-specific SimCLR features. See [`representation_metrics.csv`](results/completed_experiments/representation_exploration/tables/representation_metrics.csv). PCA/t-SNE are exploratory only and do not prove classifier performance.

At the three-seed fixed protocol, uncertainty had mean post-test accuracy 0.744 +/- 0.077 versus 0.511 +/- 0.077 for random. At the shared five-seed budget-100 endpoint, uncertainty had 0.647 +/- 0.202 versus 0.493 +/- 0.076 for random. At 15 seeds it had 0.522 +/- 0.188 versus 0.460 +/- 0.130. These results support an average uncertainty advantage under the controlled protocol, with substantial run-to-run variability; they do not establish a settled ranking for every run.

Simulated ideal images are pristine and may not represent online RHEED with haze, mixed reconstruction types, substrate-angle changes, beam-energy changes, or exposure changes. The NMF mixture score, metadata fusion, symmetry modes, and temporal constraints remain exploratory extensions rather than validated deployment conclusions.
