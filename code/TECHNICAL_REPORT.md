# Technical Report: RHEED Active Pair Selection and Reconstruction Prediction

## Executive interpretation

This project uses human pairwise RHEED preferences to train an image reward model and asks which additional pair groups are most useful to label. The completed five-selector benchmark does not identify one selector that wins at every label budget. It shows that the useful acquisition rule depends on budget and image preprocessing. The strongest unmodified-image endpoint in the completed five-seed curve is uncertainty at budget 100 (outer-test accuracy `0.647 ± 0.202`, mean batch utility `+0.220`). At budget 75, the project-specific uncertainty-plus-diversity rule is strongest (`0.513 ± 0.126`, `+0.167`). At budget 50, the project-specific cluster-quota uncertainty rule is strongest (`0.460 ± 0.123`, `+0.113`). [Stage 1 table](active_learning_studies/pair_disjoint_not_image_disjoint/results/selection_benchmark/stage1_selector_curves_none/aggregate/strategy_budget_summary.csv)

The practical implication is not that more labels always improve the current model. It is that the old single schedule—three epochs and one learning rate at all budgets—can confound selection quality with training behavior. The active work therefore calibrates learning rate and epochs with validation data separately by budget before producing the next budget curve. That Task 3a calibration is currently **583/600 cells complete**: Accounts 1–3 have 150 cells each and Account 4 has 133, leaving 17 cells. No budget-aware final curve exists yet.

## 1. What is selected and what is evaluated

The acquisition object is an **unordered pair group**. A group may include several preference rows for the same two images and reconstruction type. Before selection, the model can see candidate images and permitted model scores, but not the candidate preference outcome. Once selected, the hidden preference rows are revealed and used to retrain a Bradley--Terry reward model.

The downstream endpoint is not pairwise agreement alone. After retraining, the model predicts reconstruction type on held-out ideal images. This separates the question “which human comparisons are informative?” from the question “does the resulting model classify reconstruction images better?” The controlled studies use single-shot acquisition: select one budgeted batch, reveal it, retrain from scratch, and evaluate once.

## 2. Data, split, and model contract

The v1.8 preference source has 669 valid rows representing 179 unordered pair groups. In the completed controlled benchmark, 50 groups form the initial labeled set, 120 form the candidate pool, and 9 are unused. Initial and candidate groups are pair-disjoint. They are **not image-disjoint**: an image may appear in an initial pair and in a different candidate pair. Ideal reference, utility-validation, and outer-test images are separately partitioned. There is no separate pairwise validation partition: preference data are allocated to the initial labeled set, candidate pool, or unused pairs. [Stage 1 manifest](active_learning_studies/pair_disjoint_not_image_disjoint/results/selection_benchmark/stage1_selector_curves_none/study_manifest.json)

The downstream model is a ResNet-18 encoder followed by a 512-to-256-to-5 reward head. For each reconstruction type, the model learns relative rewards for the two images; Bradley--Terry losses train the preferred ordering, while ideal reference images support reconstruction evaluation. The completed curves use full-model fine-tuning, batch size 16, learning rate `1e-4`, and three epochs. [Model implementation](active_learning_program/pairwise_active_learning_pipeline.py)

The shipped SimCLR checkpoint is image-only self-supervised initialization; it is not trained on the pairwise preference labels. SimCLR is image-only self-supervised pretraining, not pairwise preference learning. Pairwise labels enter only in downstream reward-model training. The checkpoint hash is recorded in manifests, but the repository does not contain a matching full pretraining manifest. This matters when comparing it with ImageNet initialization: a result can compare downstream behavior of the two initializations, but cannot reconstruct every upstream pretraining difference.

## 3. Completed unmodified-image budget curve

The following completed curve is the most direct evidence for selection under the original no-symmetry input mode. Values are mean outer-test accuracy over five seeds; utility is the mean change from the pre-acquisition model.

| Budget | Highest accuracy arm | Accuracy | Interpretation | Research decision |
|---:|---|---:|---|---|
| 10 | Random | 0.413 | With very few labels, the selector scores are not yet more useful than a random draw. | Keep random as a required paired baseline. |
| 25 | Uncertainty | 0.440 | Model uncertainty begins to identify useful comparisons, but the difference from random is small relative to seed variation. | Re-evaluate under budget-aware training. |
| 50 | Cluster-quota uncertainty | 0.460 | Combining uncertainty with K-means coverage can help at a middle budget. This is a project heuristic, not a standard named acquisition method. | Preserve it as a documented comparator; test whether its gain survives the updated protocol. |
| 75 | Uncertainty + diversity | 0.513 | A weighted combination of uncertainty and embedding diversity performed best at this budget. This is a project-specific rule, not a literature-standard algorithm. | Keep the exact formula and weight visible; compare it with standard methods rather than relabeling it. |
| 100 | Uncertainty | 0.647 | The largest completed endpoint favors uncertainty, with positive mean utility. | Make budget-100 uncertainty a principal reference in the budget-aware rerun. |

The curve is non-monotonic within several arms. In this dataset, increasing the number of acquired groups also changes how much data each fixed-epoch run processes; the next protocol explicitly separates that effect with validation-selected epochs and a fixed-total-update control. [Per-budget aggregate](active_learning_studies/pair_disjoint_not_image_disjoint/results/selection_benchmark/stage1_selector_curves_none/aggregate/strategy_budget_summary.csv) and [per-seed accuracy figure](active_learning_studies/pair_disjoint_not_image_disjoint/results/selection_benchmark/stage1_selector_curves_none/aggregate/post_test_accuracy_by_budget.png)

## 4. What the custom and standard selectors mean

`random` samples candidate groups without model scoring. `uncertainty` prioritizes pairs where the current reward model is least decisive. **Core-set** (`core_set`) seeks coverage in embedding space. **Cluster-quota uncertainty** (`cluster_quota_uncertainty`) first allocates selections across K-means clusters and then prioritizes uncertain pairs within that allocation. **Uncertainty + diversity** (`uncertainty_diversity`) combines normalized uncertainty and a diversity score with a configured weight. The last two are implemented project methods; their observed gains show that candidate-pool coverage can matter, not that they have an established universal name or guarantee.

The completed lambda screen supports weight `0.5` for the project uncertainty-diversity rule at its tested endpoint: `0.544 ± 0.107` at lambda `0.5`, compared with `0.478 ± 0.102` at `0.25` and `0.300 ± 0.153` at `0.75`. This suggests that forcing diversity too strongly can discard useful uncertain pairs, while no diversity component can leave redundant acquisitions. [Lambda screen](active_learning_studies/pair_disjoint_not_image_disjoint/results/selected_summaries/lambda_sweep/summary.csv)

Cluster-Margin reached `0.533 ± 0.097` at budget 100, below uncertainty's no-symmetry endpoint. It remains a useful alternative mechanism for selecting boundaries between clusters, but the next comparison should prioritize uncertainty, the two documented project rules, random, and one coverage baseline under the same budget-aware training protocol. [Cluster-Margin curve](active_learning_studies/pair_disjoint_not_image_disjoint/results/selection_benchmark/cluster_margin_curve_none/aggregate/strategy_budget_summary.csv)

## 5. Completed symmetry factorial: image preprocessing changes the answer

Stage 2 completed the factorial across five selectors, three image modes, five budgets, and five seeds. `left_half_mirror` reconstructs a full image from the left half; `symmetric_average` averages the image with its horizontal reflection. These are not cosmetic augmentations: they alter the information supplied to the encoder.

At budget 10, symmetric-average core-set reached `0.533`, whereas no-symmetry random reached `0.413`; at budget 100, left-half-mirror random reached `0.553`, whereas no-symmetry uncertainty reached `0.647`. The meaning is that symmetry assumptions can help particular selector/budget combinations but can also remove discriminatory asymmetry. The next step is not to globally enable a symmetry transform; it is to keep image mode as an explicit factor when a physical symmetry assumption is being tested. [Full selector-by-symmetry table](active_learning_studies/pair_disjoint_not_image_disjoint/results/selection_benchmark/stage2_symmetry_factorial/aggregate/selector_symmetry_budget_interaction.csv)

## 6. Training and encoder diagnostics

Validation-only learning-rate screening over three seeds selected `3e-4` in the earlier fixed protocol: random validation accuracy was `0.544` and uncertainty was `0.444` at that LR. This selected setting is an observation about that fixed budget-100 setup, not evidence that `3e-4` is optimal at every acquisition budget. [LR calibration](active_learning_studies/pair_disjoint_not_image_disjoint/results/protocol_diagnostics/learning_rate_calibration/aggregate/learning_rate_utility_validation_summary.csv)

In the completed encoder screen, ImageNet initialization exceeded the shipped SimCLR checkpoint on validation for random (`0.556` versus `0.389`) and uncertainty (`0.611` versus `0.300`). The associated outer-test confirmation reported `0.593 ± 0.098` for random and `0.593 ± 0.101` for uncertainty. The useful research conclusion is to compare initializations under identical partitions and training budgets, rather than treat an encoder name as a result by itself. [Encoder screen](active_learning_studies/pair_disjoint_not_image_disjoint/results/protocol_diagnostics/encoder_initialization_screen/aggregate/encoder_utility_validation_summary.csv) and [confirmation](active_learning_studies/pair_disjoint_not_image_disjoint/results/protocol_diagnostics/encoder_initialization_confirmation/aggregate/outer_test_confirmation_summary.csv)

Representation analysis gives compatible context: five-nearest-neighbor accuracy on labeled ideal images was `0.884` for the shipped SimCLR features, `0.896` for ImageNet features, and `0.832` for raw pixels. This means both learned encoders organize the ideal-image population more effectively than raw pixels in that diagnostic; it does not substitute for the preference-trained downstream evaluation. [Representation metrics](active_learning_studies/image_representation_analysis/results/selected_summaries/representation_exploration/tables/representation_metrics.csv)

## 7. Task 3a, 3b, and 3c status

Task 3a is a validation-only calibration grid: five seeds × two encoder initializations × five budgets × four learning rates × three epoch counts = 600 cells. Each cell trains on the initial ten pair groups plus a deterministic random reference acquisition for that budget, evaluates utility validation only, and never evaluates outer test. The resulting table will choose LR/epochs separately for each encoder and budget.

Task 3b aggregate validation grid into a protocol table. Task 3c then uses that table to run paired final curves with five strategies, five budgets, five seeds, two encoder initializations, and both epoch-based and fixed-update controls. This sequence directly tests whether an apparent selection gain survives a protocol that lets training effort vary appropriately with the amount of labeled data.

## 8. Related research directions

The simulator study tests a different question: whether labeled simulator images improve a real-image, strictly image-disjoint three-class evaluation after real fine-tuning. Its synthetic-only arm measures domain gap; its transfer arm measures whether synthetic supervised pretraining helps. It cannot be merged with five-class historical Classifier2 numbers because it supports only Twinned, c(6 x 2), and root-13 classes. [Simulator study](active_learning_studies/simulator_augmented_classifier2_preliminary/study_description.md)

The process-metadata study tests whether causal monitor context improves image prediction. Its available bundle has one session and no image bytes, so it currently establishes a data interface: explicit image mapping, causal sensor joins, image/run-disjoint splitting, and image-only/metadata-only/fusion controls. The next requirement is multiple image-resolved growth sessions. [Metadata study](active_learning_studies/process_metadata_fusion_for_rheed_prediction/study_description.md)

Trajectory ordering is a third, separate direction. It converts physics-team statements such as “HTR comes last” into a soft whole-trajectory decoder for externally supplied classifier probabilities. It does not append temperature as an image feature and it does not create labels. [Trajectory study](active_learning_studies/rheed_trajectory_ordering_analysis/study_description.md)

## 9. Evidence map and next step

Historical pair-disjoint curves, the symmetry factorial, LR/encoder diagnostics, and representation analysis are completed evidence families. Task 3a is an incomplete but directly actionable protocol-calibration result. The simulator, metadata, and trajectory studies are implemented research paths with distinct data requirements.

The immediate active-learning next step is to finish the 17 Task 3a cells, aggregate the validation-only protocol, and run Task 3c. The immediate laboratory-data next step is to collect image-resolved session bundles for the metadata study. The immediate simulator next step is to run its preflight and then compare real-only, synthetic-only, and transfer arms on the fixed real outer test. These are complementary experiments, not substitutes for each other.
