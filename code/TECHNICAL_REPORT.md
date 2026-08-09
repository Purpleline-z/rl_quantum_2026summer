# Technical Report: Result-Oriented RHEED Active Pair Selection

## Research objective

The objective is to select **more informative pairwise RHEED image groups** under a limited labeling budget, then improve downstream classification of ideal reconstruction types after retraining. This is not simply a representation study or a report on epochs. The central outcome is a reproducible performance-versus-acquisition-budget comparison of selection methods and input preprocessing.

For an acquired pair group \(p\) and currently revealed groups \(L\), the experimental utility is

\[
U(p \mid L) = Acc(L \cup \{p\}) - Acc(L),
\]

where `Acc` is fixed ideal-image outer-test accuracy after downstream Bradley--Terry retraining. A pair group may contain multiple preference rows. Selection is pairwise; final evaluation is image classification.

## Controlled benchmark protocol

The result-oriented benchmark fixes 50 initial labeled unordered pair groups, a 120-group candidate pool, single-shot acquisition, epoch 3, learning rate `1e-4`, full-model fine-tuning, and the existing ideal reference/outer-test policy. Five reproducibility seeds are 42, 79, 123, 202, and 303. The planned complete grid uses budgets 10, 25, 50, 75, and 100, all five selectors, and all three symmetry modes.

The five selectors are:

1. `random` — random candidate-pair selection.
2. `uncertainty` — selects pairs whose current Bradley--Terry predictions are least decisive.
3. `uncertainty_diversity` — uncertainty plus embedding-space diversity, with diversity weight 0.5.
4. `cluster_quota_uncertainty` — custom K-means quota heuristic followed by uncertainty ranking. Historical artifacts call this `cluster_diverse`; it is not claimed to be a standard algorithm.
5. `core_set` — greedy pair-embedding coverage relative to currently labeled pairs.

The three input modes are `none`, `left_half_mirror`, and `symmetric_average`. They are classifier/preprocessing ablations, not claims that experimental RHEED is exactly symmetric.

## Completed selection evidence

### Fixed budget-100 comparison: all five selectors

The completed endpoint comparison contains all five selectors. It is important evidence and is not omitted merely because the original five-budget curve initially focused on random and uncertainty.

| Selector | 3-seed post-test accuracy, mean ± SD | 3-seed utility, mean ± SD | 15-seed post-test accuracy, mean ± SD | 15-seed utility, mean ± SD |
|---|---:|---:|---:|---:|
| Random | 0.511 ± 0.077 | +0.078 ± 0.168 | 0.460 ± 0.130 | +0.082 ± 0.208 |
| Uncertainty | **0.744 ± 0.077** | **+0.311 ± 0.038** | **0.522 ± 0.188** | **+0.144 ± 0.255** |
| Uncertainty + diversity | 0.544 ± 0.107 | +0.111 ± 0.184 | 0.489 ± 0.159 | +0.111 ± 0.248 |
| Cluster quota uncertainty* | 0.522 ± 0.139 | +0.089 ± 0.252 | 0.427 ± 0.121 | +0.049 ± 0.222 |
| Core-set | 0.478 ± 0.139 | +0.044 ± 0.234 | 0.422 ± 0.179 | +0.044 ± 0.205 |

\*Historical CSV label: `cluster_diverse`.

Evidence:

- [`strategy_summary.csv`](results/completed_experiments/fixed_protocol_3seed/tables/strategy_summary.csv)
- [`post_test_accuracy_by_epoch.png`](results/completed_experiments/fixed_protocol_3seed/figures/post_test_accuracy_by_epoch.png) (epoch-selection context; not the active-learning x-axis)
- [`endpoint_15seed_strategy_summary.csv`](results/completed_experiments/endpoint_extension_15seed/tables/endpoint_15seed_strategy_summary.csv)
- [`post_test_accuracy_by_strategy.png`](results/completed_experiments/endpoint_extension_15seed/figures/post_test_accuracy_by_strategy.png)
- [`paired_differences_vs_random.csv`](results/completed_experiments/endpoint_extension_15seed/tables/paired_differences_vs_random.csv)

The current endpoint conclusion is therefore specific: **uncertainty has the highest average post-test accuracy and utility among the five tested selectors at budget 100; uncertainty plus diversity is second on the 15-seed endpoint.** The standard deviations are large, so this does not establish a universal per-run ordering or prove the same ranking at every budget.

### Completed five-seed budget curve: random versus uncertainty

The existing main curve varies acquisition budget but currently covers only two selectors. Its purpose is to show the budget-dependent behavior of the two primary baselines; it is not the complete selector comparison.

| Budget | Random post-test accuracy | Uncertainty post-test accuracy |
|---:|---:|---:|
| 10 | 0.413 ± 0.065 | 0.387 ± 0.084 |
| 25 | 0.433 ± 0.071 | 0.440 ± 0.126 |
| 50 | 0.327 ± 0.098 | 0.300 ± 0.113 |
| 75 | 0.293 ± 0.113 | 0.380 ± 0.141 |
| 100 | 0.493 ± 0.076 | 0.647 ± 0.202 |

Evidence:

- [`post_test_accuracy_by_acquisition_budget.png`](results/completed_experiments/budget_curve_5seed/figures/post_test_accuracy_by_acquisition_budget.png)
- [`batch_utility_by_acquisition_budget.png`](results/completed_experiments/budget_curve_5seed/figures/batch_utility_by_acquisition_budget.png)
- [`strategy_performance_by_acquisition_budget.csv`](results/completed_experiments/budget_curve_5seed/tables/strategy_performance_by_acquisition_budget.csv)
- [`per_seed_performance_by_acquisition_budget.csv`](results/completed_experiments/budget_curve_5seed/tables/per_seed_performance_by_acquisition_budget.csv)

Budget-10 performance is especially variable and uncertainty is not yet superior there. The active Stage 1 benchmark is designed to determine whether the endpoint ordering persists for all five selectors across the full curve, rather than assuming it does.

## Benchmark status matrix

`Completed` means a five-seed cell has results. `Reused` means protocol-compatible existing evidence will be normalized with provenance rather than rerun. `Planned` means an immutable job manifest is generated but no downstream result is claimed.

| Selector | `none`: 10/25/50/75 | `none`: 100 | `left_half_mirror`: all budgets | `symmetric_average`: all budgets |
|---|---|---|---|---|
| Random | Completed | Completed | Planned | Planned |
| Uncertainty | Completed | Completed | Planned | Planned |
| Uncertainty + diversity | Planned Stage 1 | Reused endpoint | Planned Stage 2 | Planned Stage 2 |
| Cluster quota uncertainty | Planned Stage 1 | Reused endpoint | Planned Stage 2 | Planned Stage 2 |
| Core-set | Planned Stage 1 | Reused endpoint | Planned Stage 2 | Planned Stage 2 |

Stage 1 has 60 new jobs: three selectors × four missing budgets × five seeds, all with `symmetry=none`. Stage 2 has 250 new jobs: two new symmetry modes × five selectors × five budgets × five seeds. The Stage 2 aggregation will join the fully completed `none` benchmark rather than rerun it.

Two additional selector studies are planned but have no results yet: a 30-job MC-dropout screening at budgets 10 and 25, and a 25-job Cluster-Margin pairwise curve across budgets 10–100. They are reported as planned methods only until their immutable manifests are complete and aggregated.

## Reproducibility, splits, and interpretation limits

The pairwise CSV contains 669 valid preference rows organized into 179 unique unordered pair groups. Every controlled run starts with 50 initial groups, uses 120 candidate groups, and leaves 9 groups unused. Initial, candidate, and selected partitions are disjoint by unordered pair, but they are **not image-disjoint**: an image can occur in several pair groups. Candidate preference labels are hidden until selection. The outer ideal test has 30 images and therefore accuracy changes in discrete increments of 1/30.

There is no separate pairwise validation partition. Pairwise rows train Bradley--Terry preferences; ideal reference anchors and the untouched ideal outer test evaluate downstream reconstruction classification. Existing run-local manifests and selected-pair CSVs are retained under `results/local_runs/` for audit.

The model is a SimCLR ResNet-18 encoder with a 512-to-256-to-5 Bradley--Terry reward head. SimCLR is image-only self-supervised pretraining, not pairwise preference learning. The supplied checkpoint is architecture-compatible with the downstream encoder, but the exact pretraining population and recipe that produced it are not proven by a run manifest. Known downstream augmentation is affine rotation ±5°, translation ±5%, scale 0.95–1.05, and brightness/contrast jitter 0.2.

PCA/t-SNE and kNN diagnostics remain representation context only. The representation report is in [`representation_exploration`](results/completed_experiments/representation_exploration/); it does not establish active-learning performance.

## Separate improvement tracks

Metadata fusion is implemented as a model scaffold, but the stable data folder presently contains only [`process_metadata_template.csv`](../data/metadata/process_metadata_template.csv), not real process metadata. It must not be reported as an experiment until populated metadata with normalized paths is available.

Feature-space NMF mixture acquisition is disabled: its reference-only diagnostic did not obtain one-to-one class coverage and had negative silhouette. It remains a diagnostic, not a paper result.

Temporal statements such as “HTR comes last” remain tentative physics hypotheses. The completed filename-order audit found three parseable trajectories totaling 1,124 frames, but no temporal constraint is used in selection or training. A later stream policy may only form pairs from temporally adjacent frames within a trajectory after physics confirmation; it will be compared independently against the unrestricted-pool benchmark.

## Commands and outputs

Run from `code/src` with the stable Google Drive data root. All outputs go under `code/results/local_runs/selection_benchmark/`; no input data are modified.

```bash
python run_selection_benchmark.py generate-stage1-manifest
python run_selection_benchmark.py prepare-reused-evidence --manifest ../results/local_runs/selection_benchmark/stage1_selector_curves_none/study_manifest.json
python run_selection_benchmark.py run-job --manifest ../results/local_runs/selection_benchmark/stage1_selector_curves_none/study_manifest.json --job-id <job-id> --data-root /content/drive/MyDrive/rheed_project/data
python run_selection_benchmark.py aggregate-stage1 --manifest ../results/local_runs/selection_benchmark/stage1_selector_curves_none/study_manifest.json

python run_selection_benchmark.py generate-stage2-manifest
python run_selection_benchmark.py run-job --manifest ../results/local_runs/selection_benchmark/stage2_symmetry_factorial/study_manifest.json --job-id <job-id> --data-root /content/drive/MyDrive/rheed_project/data
python run_selection_benchmark.py aggregate-stage2 --manifest ../results/local_runs/selection_benchmark/stage2_symmetry_factorial/study_manifest.json --stage1-aggregate ../results/local_runs/selection_benchmark/stage1_selector_curves_none/aggregate
```

Default jobs use the 50-minute runtime limit; large resumable checkpoints are disabled. Each new job writes its immutable manifest, source hashes, overlap audit, selected pairs, result JSON, and lightweight logs.
