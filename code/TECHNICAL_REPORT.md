# Technical Report: RHEED Active Pair Selection

## 1. Executive scientific conclusion

This project asks a practical active-learning question: **under a limited labeling budget, which unordered RHEED image-pair groups should be revealed so that retraining improves downstream classification of ideal reconstruction types?** The selected object is a pair group; the primary outcome is post-acquisition accuracy on an untouched ideal-image outer test set.

The completed controlled evidence shows that selection is **budget- and preprocessing-dependent**. In the five-selector, no-symmetry benchmark, the highest mean outer-test accuracy at budgets 10, 25, 50, 75, and 100 was respectively random (0.413), uncertainty (0.440), cluster-quota uncertainty (0.460), uncertainty + diversity (0.513), and uncertainty (0.647). Each is a five-seed mean and every comparison has substantial run-to-run variation; these are directional results, not significance claims. [Stage 1 aggregate CSV](results/local_runs/selection_benchmark/stage1_selector_curves_none/aggregate/strategy_budget_summary.csv)

The completed symmetry factorial confirms that neither a selector nor a symmetry mode is a universal winner. Symmetry preprocessing changes the ranking at particular budgets, but it does not establish that real experimental patterns should be forcibly symmetrized. The system is therefore not yet a deployment-ready growth-control classifier. Its most defensible present contribution is a reproducible benchmark that identifies promising, conditional acquisition strategies and exposes the current downstream limitations. [Stage 2 interaction table](results/local_runs/selection_benchmark/stage2_symmetry_factorial/aggregate/selector_symmetry_budget_interaction.csv)

**Conclusion.** The evidence supports continuing result-oriented selector development, but the next claims must be tied to a stronger and more realistic downstream evaluation—not to a single endpoint or representation plot.

## 2. Scientific objective and controlled pipeline

For a candidate pair group \(p\) and currently revealed set \(L\), utility is defined operationally as

\[
U(p \mid L) = \operatorname{Acc}(L \cup \{p\}) - \operatorname{Acc}(L),
\]

where accuracy is fixed ideal-image outer-test accuracy after downstream Bradley–Terry retraining. A pair group can contain multiple CSV preference rows. The controlled studies use **single-shot acquisition**: train on the initial pair groups, rank the candidate pool without revealing candidate preferences, acquire a budgeted set of groups, reveal their rows, retrain, and evaluate once.

The active-learning x-axis is acquired-pair-group budget (10, 25, 50, 75, or 100), not epoch. Epoch is a downstream-training setting. Seed is a reproducibility control, not a scientific treatment variable. The controlled benchmark fixes epoch 3, learning rate 1e-4, full-model fine-tuning, 50 initial pair groups, a 120-group candidate pool, and the ideal reference/test policy. Tolerance is not an exposed parameter. [Stage 1 manifest](results/local_runs/selection_benchmark/stage1_selector_curves_none/study_manifest.json)

The five evaluated selectors are random, uncertainty, uncertainty + diversity (lambda = 0.5), cluster-quota uncertainty, and core-set. `cluster_quota_uncertainty` is a custom K-means quota heuristic; historical artifacts may call it `cluster_diverse`, and it is not presented as a standard named algorithm. The additional Cluster-Margin method and three MC-dropout scores are exploratory selector studies.

**Conclusion.** The protocol isolates the value of the first selected batch under a fixed downstream model; it does not yet measure sequential acquisition interactions or online deployment behavior.

## 3. Dataset, splits, leakage, and evaluation contract

The source preference data contain 669 valid rows representing 179 unique unordered pair groups. In each controlled seed, 50 groups form the initially revealed pairwise-training set, 120 groups form the candidate pool, and 9 groups are unused. Candidate preference labels are hidden from the selector until acquisition. [Stage 2 manifest](results/local_runs/selection_benchmark/stage2_symmetry_factorial/study_manifest.json)

| Partition / check | Controlled protocol meaning |
|---|---|
| Initial pairwise training | 50 unordered groups; the number of revealed preference rows is seed-dependent because groups contain different numbers of rows. |
| Candidate pool | 120 unordered groups; only image content and permitted model scores are visible before acquisition. |
| Selected set | A budgeted subset of the candidate groups; exactly pair-disjoint from the initial set. |
| Unused groups | 9 groups outside that seed’s initial/pool partition. |
| Pairwise validation | None. There is no separate pairwise validation partition in the controlled protocol. |
| Ideal reference anchors | Fixed ideal images used by the downstream evaluation contract. |
| Utility-validation images | An ideal-image split used only where a diagnostic explicitly selects a hyperparameter. |
| Outer test | Untouched ideal images used for the main post-acquisition result; job artifacts report 30 images per run. |

Pair-disjoint is not image-disjoint: an image can occur in different unordered pairs. The manifests explicitly audit exact-pair overlap, image overlap between pairwise partitions, pairwise-image overlap with ideal test images, and reference/test overlap. The raw manifests are the authoritative per-run audit because those counts vary with seed and selected budget. [Example audited run](results/local_runs/selection_benchmark/stage1_selector_curves_none/jobs/seed-42_budget-10_strategy-uncertainty_diversity_symmetry-none/manifests/audit.json)

This distinction answers the key split question: SimCLR pretraining and downstream pairwise training are different stages. The current controlled run does **not** begin with 100+ labeled pair groups; it begins with 50 unordered groups. A budget of 10 is consequently a small, variable addition of ten groups, not ten fixed CSV rows. The small 30-image outer test also makes accuracy change in discrete increments. Together with seed-dependent group composition and full-model fine-tuning, this explains why low-budget and random results can vary substantially; it does not prove a causal mechanism or monotonic gain with budget.

**Conclusion.** The benchmark prevents exact-pair leakage across initial, candidate, and selected sets, but it does not claim image-disjoint partitions. All split and overlap claims must be read as pair-level unless a manifest says otherwise.

## 4. Model, SimCLR provenance, and training protocol

The downstream system uses a ResNet-18 image encoder with a 512-dimensional embedding and a 512-to-256-to-5 reward head. Each image receives reconstruction-specific rewards; Bradley–Terry comparisons train the relative preference model, while ideal reference anchors support ideal-image evaluation. The full model is trainable in the controlled protocol. [Current model implementation](src/active_learning_pipeline.py)

**SimCLR is image-only self-supervised pretraining, not pairwise preference learning.** Pairwise labels are used only after encoder initialization to train the downstream Bradley–Terry model. The known downstream augmentation is affine rotation ±5°, translation ±5%, scale 0.95–1.05, and brightness/contrast jitter 0.2. [Downstream augmentation implementation](classifier2/downstream_classifier_scripts/train_unified.py)

The supplied encoder checkpoint is auditable by hash in run manifests, but the exact image population and recipe that produced that checkpoint are not established by a matching pretraining run manifest. The provenance audit records that the currently vendored upstream material contains downstream Bradley–Terry scripts, not an executable self-supervised SimCLR training entry point. It must therefore not be claimed that the available source produced the shipped checkpoint. [Pretraining provenance audit](classifier2/pretraining/UPSTREAM_SOURCE.md)

Epoch 3 was selected in an earlier epoch sweep because uncertainty peaked before later epochs showed overfitting signs. That historical model-selection result fixes a downstream setting; it is not an active-learning curve. [Epoch-selection figure](results/completed_experiments/fixed_protocol_3seed/figures/post_test_accuracy_by_epoch.png)

**Conclusion.** The pipeline has a well-defined downstream architecture and known fine-tuning augmentation, but checkpoint-level SimCLR provenance remains incomplete and should not be overstated.

## 5. Representation context and earlier Classifier2 evidence

The representation exploration contains 1,278 images: 154 labeled ideal images and 1,124 unlabeled trajectory images. On the ideal subset, 5-nearest-neighbor cross-validation accuracy was 0.884 for the domain SimCLR representation, 0.896 for ImageNet ResNet-18, and 0.832 for raw pixels. PCA and t-SNE are exploratory support only: t-SNE can visually exaggerate separation and neither visualization proves downstream classifier performance. [Representation metrics](results/completed_experiments/representation_exploration/tables/representation_metrics.csv) and [SimCLR t-SNE](results/completed_experiments/representation_exploration/figures/simclr_resnet18_tsne.png)

The older Classifier2 report uses a different data/model/evaluation version. Its measured 83.3% and 71.4% values are pairwise-holdout results; 66.7% is a reported single-image result. Its 88–92% value is a projection, not an achieved result. These numbers cannot be pooled with the current ideal-image outer-test benchmark. [Classifier2 report](classifier2/TECHNICAL_REPORT.md)

A completed bridge gives useful but limited context: the legacy pairwise-only bridge obtained 0.824 (28/34) after 30 epochs, while the current pipeline recorded 0.832 on 137 rows after 3 epochs. This is not a strict head-to-head comparison because denominators and legacy supported-class handling differ, and no comparable ideal-image evaluation was run. [Bridge comparison](results/local_runs/protocol_diagnostics/classifier2_protocol_bridge/pairwise_bridge_comparison.csv) and [compatibility audit](results/local_runs/protocol_diagnostics/classifier2_protocol_bridge/compatibility_audit.md)

**Conclusion.** The representation and legacy results motivate the present project, but they neither validate nor contradict the controlled active-selection conclusions because their evaluation contracts differ.

## 6. Main result: five-selector performance versus budget without symmetry

This is the primary selector benchmark under the unmodified input mode. Values are mean ± sample SD outer-test accuracy; utility is the corresponding mean batch utility. All cells have five seeds. [Complete Stage 1 table](results/local_runs/selection_benchmark/stage1_selector_curves_none/aggregate/strategy_budget_summary.csv)

| Budget | Random | Uncertainty | Uncertainty + diversity | Cluster-quota uncertainty | Core-set |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.413 ± 0.065 / -0.013 | 0.387 ± 0.084 / -0.040 | 0.340 ± 0.177 / -0.007 | 0.360 ± 0.098 / +0.013 | 0.307 ± 0.134 / -0.040 |
| 25 | 0.433 ± 0.127 / +0.007 | **0.440 ± 0.126 / +0.013** | 0.387 ± 0.214 / +0.040 | 0.380 ± 0.185 / +0.033 | 0.347 ± 0.139 / 0.000 |
| 50 | 0.327 ± 0.116 / -0.100 | 0.300 ± 0.113 / -0.127 | 0.380 ± 0.065 / +0.033 | **0.460 ± 0.123 / +0.113** | 0.353 ± 0.107 / +0.007 |
| 75 | 0.293 ± 0.043 / -0.133 | 0.380 ± 0.141 / -0.047 | **0.513 ± 0.126 / +0.167** | 0.407 ± 0.220 / +0.060 | 0.420 ± 0.061 / +0.073 |
| 100 | 0.493 ± 0.076 / +0.067 | **0.647 ± 0.202 / +0.220** | 0.513 ± 0.090 / +0.087 | 0.440 ± 0.150 / +0.013 | 0.420 ± 0.173 / -0.007 |

Each entry is `post-test accuracy / batch utility`. The curve is non-monotonic: more acquired groups do not always improve a method’s mean result. The completed figures show per-seed lines as well as the aggregate summaries. [Accuracy figure](results/local_runs/selection_benchmark/stage1_selector_curves_none/aggregate/post_test_accuracy_by_budget.png) and [utility figure](results/local_runs/selection_benchmark/stage1_selector_curves_none/aggregate/batch_utility_by_budget.png)

**Conclusion.** At no symmetry, uncertainty is strongest at budget 100, while cluster-quota uncertainty and uncertainty + diversity lead at intermediate budgets. There is no defensible global ranking across every budget.

## 7. Completed symmetry factorial

Stage 2 completed the 5-selector × 3-symmetry × 5-budget × 5-seed factorial. It contains 250 newly run left-half-mirror/symmetric-average jobs plus 125 reused protocol-compatible `none` cells, for 375 completed selector–symmetry–budget cells. [Aggregation manifest](results/local_runs/selection_benchmark/stage2_symmetry_factorial/aggregate/aggregation_manifest.json) and [full interaction CSV](results/local_runs/selection_benchmark/stage2_symmetry_factorial/aggregate/selector_symmetry_budget_interaction.csv)

The best mean accuracy in each mode at budgets 10/25/50/75/100 was:

| Input mode | 10 | 25 | 50 | 75 | 100 |
|---|---|---|---|---|---|
| `none` | random, 0.413 | uncertainty, 0.440 | cluster-quota, 0.460 | uncertainty + diversity, 0.513 | uncertainty, 0.647 |
| `left_half_mirror` | uncertainty + diversity, 0.447 | uncertainty, 0.427 | core-set, 0.527 | uncertainty + diversity, 0.493 | random, 0.553 |
| `symmetric_average` | core-set, 0.533 | random, 0.507 | core-set and uncertainty + diversity, 0.460 | random, 0.460 | uncertainty + diversity, 0.453 |

The full CSV reports mean ± sample SD accuracy and utility for every cell. The paired output is available for direct comparisons with random under the same seed/mode/budget. [Accuracy figure](results/local_runs/selection_benchmark/stage2_symmetry_factorial/aggregate/post_test_accuracy_by_budget.png), [utility figure](results/local_runs/selection_benchmark/stage2_symmetry_factorial/aggregate/batch_utility_by_budget.png), and [paired differences](results/local_runs/selection_benchmark/stage2_symmetry_factorial/aggregate/paired_differences_vs_random.csv)

**Conclusion.** Symmetry preprocessing can improve individual cells, but it changes selector rankings and has no uniform benefit. This is evidence about the current classifier/selector system, not physical proof that every RHEED image should be symmetrized.

## 8. Secondary selector evidence

The completed lambda sweep selected diversity weight 0.5 within its limited endpoint scope; it should not be treated as a universal optimum. [Lambda-sweep summary](results/completed_experiments/lambda_sweep/summary.csv) and [choice record](results/completed_experiments/lambda_sweep/choice.json)

MC-dropout was intentionally screened only at budgets 10 and 25, which is why its plot has few points. Reward variance was the strongest of the three MC scores within that screen (0.360 at budget 10 and 0.333 at budget 25), but it did not exceed the matching random or uncertainty means. It is not a current winner. [MC-dropout table](results/local_runs/selection_benchmark/mc_dropout_screen_low_budget/aggregate/strategy_budget_summary.csv) and [MC accuracy figure](results/local_runs/selection_benchmark/mc_dropout_screen_low_budget/aggregate/post_test_accuracy_by_budget.png)

Cluster-Margin was evaluated across the full five-budget curve. Its budget-100 result was 0.533 ± 0.097 with +0.187 mean utility, below no-symmetry uncertainty at that budget (0.647 ± 0.202). It remains an exploratory alternative rather than a replacement for the primary selectors. [Cluster-Margin table](results/local_runs/selection_benchmark/cluster_margin_curve_none/aggregate/strategy_budget_summary.csv)

**Conclusion.** The added selectors expand the search space and document negative or conditional findings; they do not justify a new universal acquisition rule.

## 9. Downstream protocol diagnostics

Learning-rate calibration used utility-validation only—without outer-test access—to compare 1e-5, 3e-5, 1e-4, and 3e-4 for frozen random and uncertainty budget-100 selections over seeds 42, 79, and 123. The predeclared rule was highest mean utility-validation accuracy, then lower sample SD, then lower learning rate. It locked 3e-4. [LR calibration table](results/local_runs/protocol_diagnostics/learning_rate_calibration/aggregate/learning_rate_utility_validation_summary.csv)

The independent five-seed outer-test confirmation at that locked setting was 0.527 ± 0.201 for random and 0.513 ± 0.107 for uncertainty. [LR confirmation table](results/local_runs/protocol_diagnostics/learning_rate_confirmation/aggregate/outer_test_confirmation_summary.csv)

Encoder screening likewise used utility validation only. ImageNet initialization outperformed the shipped SimCLR checkpoint in this screen: random 0.556 ± 0.164 versus 0.389 ± 0.019, and uncertainty 0.611 ± 0.038 versus 0.300 ± 0.088. ImageNet was therefore locked for confirmation. [Encoder screen](results/local_runs/protocol_diagnostics/encoder_initialization_screen/aggregate/encoder_utility_validation_summary.csv) The five-seed outer-test confirmation then gave 0.593 ± 0.098 for random and 0.593 ± 0.101 for uncertainty. [Encoder confirmation](results/local_runs/protocol_diagnostics/encoder_initialization_confirmation/aggregate/outer_test_confirmation_summary.csv)

**Conclusion.** These diagnostics improve calibration of the current downstream protocol but do not prove universal optimizer or encoder superiority. They must not be numerically pooled with selection studies that use different frozen selections or evaluation settings.

## 10. Historical-result reconciliation

| Evidence family | Role | Status | Pool with main curve? |
|---|---|---|---|
| Older Classifier2 | Earlier architecture/evaluation context | Complete | No |
| Early sequential exploration | Exploratory sequential acquisition | Complete | No |
| Fixed 3-seed endpoint | Initial controlled five-selector direction | Complete | No; different seed population |
| 15-seed endpoint | Larger budget-100 endpoint population | Complete | No; endpoint only |
| Original random/uncertainty curve | Historical two-selector curve | Complete | Shared with later no-symmetry evidence where provenance matches |
| Stage 1 no-symmetry curve | Main five-selector budget comparison | Complete | Yes, primary evidence |
| Lambda sweep | Diversity-weight screen | Complete | No; limited endpoint scope |
| MC-dropout screen | Low-budget exploratory screen | Complete | No; two budgets only |
| Cluster-Margin curve | Exploratory full curve | Complete | Compare explicitly, not pooled implicitly |
| Stage 2 symmetry factorial | Controlled selector × preprocessing interaction | Complete | Yes, with matching `none` provenance |
| Classifier2 bridge | Pairwise-contract context | Complete | No |
| LR / encoder diagnostics | Downstream calibration | Complete | No |

The five-seed budget-100 endpoint is shared evidence between endpoint and curve views, not an independent replication. The 15-seed endpoint is a larger seed population, not a contradiction of the five-seed curve. Early sequential results are not pooled with the controlled single-shot evidence. [15-seed endpoint table](results/completed_experiments/endpoint_extension_15seed/tables/endpoint_15seed_strategy_summary.csv) and [fixed 3-seed table](results/completed_experiments/fixed_protocol_3seed/tables/strategy_summary.csv)

**Conclusion.** Results that look inconsistent often use different data versions, seed populations, selection schedules, or evaluation contracts. The report preserves them as evidence with defined scope rather than discarding completed work or combining incomparable numbers.

## 11. Limits, deployment relevance, and next experiments

Simulator-generated ideal images provide clean labels and scalable test material, but they do not contain the haze and multi-reconstruction mixtures common in online RHEED. High ideal-image accuracy is therefore not sufficient evidence of growth-control usefulness. The deployment metric may need to be revised once the growth-control objective and experimental ground truth are specified.

Metadata fusion is implemented as a scaffold only; no populated normalized metadata CSV is available, so no metadata result is claimed. The feature-space NMF mixture surrogate remains diagnostic because its reference audit did not establish one-to-one component coverage and had negative silhouette. Temporal observations such as “HTR comes last,” “starts at 1×1,” and possible 1×1 → bad → other sequences remain tentative physics hypotheses, not current labels, losses, or constraints.

The next high-value studies are: (1) establish a realistic downstream ceiling and simulated-to-online gap, (2) carry forward the best validated protocol configuration into an explicitly comparable selector study, (3) test metadata fusion only after real metadata arrive, and (4) add temporal constraints only after physics confirmation and a descriptive trajectory audit. Future stream-based acquisition should form candidate pairs only from temporally adjacent frames in the same trajectory.

**Conclusion.** The immediate research goal is not more unstructured endpoints; it is a stronger downstream evaluation and controlled comparisons that can support a paper-level statement about informative pair selection.

## 12. Reproducibility and handoff

`results/completed_experiments/` holds compact historical professor-facing evidence. `results/local_runs/` holds the immutable study manifests, job results, selected-pair records, audit files, source hashes, and current aggregate outputs. A study is reportable only when its immutable manifest, all expected results, aggregate CSVs, aggregate PNGs, and provenance record are present.

For Colab, clone the repository so its sibling `data/` directory is available, then run from `code/src`. Generated evidence belongs under `code/results/`; data are immutable inputs unless deliberately updated. Use the repository README for current commands; this report intentionally contains no credentials, personal paths, or branch troubleshooting.

**Conclusion.** The report is a manual, evidence-linked handoff document. It is designed to let a professor or another AI trace every stated result to a compact artifact before inspecting raw per-job records.
