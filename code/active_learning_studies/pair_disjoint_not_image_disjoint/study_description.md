# Pair-Disjoint, Not Image-Disjoint Active-Learning Study

## Scientific question

The study asks which additional human pairwise RHEED judgments should be acquired when labels are limited. The candidate is an unordered pair group, not a single image. After a selector chooses groups, their hidden preference rows are revealed, a Bradley--Terry image reward model is trained from scratch, and the resulting model is evaluated on held-out ideal reconstruction images.

This makes the study relevant to laboratory labeling effort: it evaluates whether a proposed set of comparisons improves a downstream reconstruction predictor, rather than whether a selector merely agrees with its own uncertainty score.

## Dataset and split meaning

The v1.8 source has 669 valid preference rows in 179 unordered groups. The completed controlled protocol uses 50 initially labeled groups, a 120-group candidate pool, and 9 unused groups per seed. Initial and candidate sets have no identical unordered pair ID.

The split is **pair-disjoint, not image-disjoint**. The same physical image may occur in an initial pair and again in a candidate pair with a different partner. This matches the historical Classifier2-style pairwise setting and makes the study useful for comparing candidate pair choices inside that setting. It does not answer the stricter question of generalizing to entirely unseen pairwise images. Ideal reference, utility-validation, and outer-test images are partitioned separately from each other and from pairwise-image use.

## Model and acquisition contract

The current downstream model is a ResNet-18 image encoder, a 512-to-256-to-5 reconstruction reward head, and Bradley--Terry pairwise losses. The five output dimensions correspond to `(1 x 1)`, `Twinned(2 x 1)`, `c(6 x 2)`, root-13, and HTR. Candidate preference labels are hidden while ranking; a selector may use images, already revealed training rows, and predeclared model scores only.

The completed benchmark compares random sampling, uncertainty, core-set coverage, `cluster_quota_uncertainty`, and `uncertainty_diversity`. The last two are project-defined methods: the former imposes K-means cluster quotas before uncertain selection; the latter combines normalized uncertainty and diversity with a configured weight. Their results are retained because they test concrete hypotheses about coverage and redundancy, but their names do not imply that they are standard published algorithms.

## Completed evidence and its meaning

The no-symmetry five-seed curve finds different leading methods at different budgets: random at 10 labels, uncertainty at 25 and 100, cluster-quota uncertainty at 50, and uncertainty-diversity at 75. The high budget-100 uncertainty result (`0.647 ± 0.202`) suggests that model indecision can locate useful comparisons once the initial preference model has enough information. The budget-50 and budget-75 results suggest that coverage or diversity can matter when an uncertain candidate pool contains redundant pairs. [Aggregate curve](results/selection_benchmark/stage1_selector_curves_none/aggregate/strategy_budget_summary.csv)

The preprocessing factorial shows that horizontal-symmetry assumptions change the ranking. It should be read as an interaction between physical/image assumptions and selection, not as proof that one symmetry preprocessing method is always better. [Interaction table](results/selection_benchmark/stage2_symmetry_factorial/aggregate/selector_symmetry_budget_interaction.csv)

## Task 1–3 workflow and current status

Task 1 generated fixed initial-10 calibration cells over learning rates, epochs, seeds, and encoder initialization. Task 2 aggregates those diagnostic cells. Task 3a replaces a global locked schedule with validation-only calibration for every encoder and acquisition budget. Its full grid is 600 cells; 583 are currently Git-tracked, with 17 Account 4 cells remaining. The calibration output will select one LR/epoch setting per encoder and budget without opening the outer test.

Task 3b aggregates the completed grid into the budget-aware protocol table. Task 3c uses that table for the final paired five-strategy curves, with a second control that fixes optimizer updates so larger acquired datasets do not automatically receive more parameter updates merely because an epoch contains more batches. No Task 3b/3c result exists until Task 3a reaches 600/600.

## Result ownership and next research decision

`results/` contains the completed historical curves, diagnostics, and account-level calibration JSONs. Each result folder records the protocol it owns; it must not be pooled with another folder merely because a selector name is shared.

The immediate research decision is to complete the missing validation-only cells and produce the budget-aware comparison. The resulting curve will answer a sharper question than the historical fixed-epoch curve: whether the observed budget-dependent selector behavior remains when every budget has an appropriate validation-selected training schedule.
