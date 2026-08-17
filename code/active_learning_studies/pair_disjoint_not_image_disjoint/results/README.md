# Pair-Disjoint Study Results

This directory is the evidence store for the pair-disjoint study. A result folder owns the data, configuration, figures, and interpretation for one protocol family. It is not a single pooled table.

## Current result families

| Folder | Evidence | Meaning for research |
|---|---|---|
| `selection_benchmark/` | Completed five-selector curves, symmetry factorial, MC-dropout screen, and Cluster-Margin curve | Shows that selector ranking depends on budget and image preprocessing. |
| `protocol_diagnostics/` | Learning-rate, encoder, and Classifier2 bridge diagnostics | Identifies training/initialization hypotheses that require protocol-matched confirmation. |
| `selected_summaries/` | Focused endpoint, epoch, lambda, and representation summaries | Documents why particular follow-up choices were made. |
| `budget_aware_validation_calibration_account_*/` | Raw Task 3a validation-only cells | Incomplete protocol-calibration evidence: 583/600 cells are available. |
| `budget_curve_study/` | Earlier fixed-LR/fixed-epoch curve report | Historical reference for how the initial curve was produced. |
| `earlier_explorations/` | Sequential, pre-reorganization, and prototype records | Historical context; their data/split/training settings differ from the current controlled workflow. |
| `active_learning_v1.8_seed42/`, `baseline_run_seed42/`, and other seed folders | Local-style manifests retained in Git | Provenance examples, not a second independent benchmark. |

## How to read a result

Read the manifest or configuration first, then the per-seed CSV/JSON, then the aggregate figure/table. The same strategy label can occur under different initial set sizes, image preprocessing, epochs, encoder initialization, or split definitions. Those outputs answer different questions and should not be averaged together.

For completed selection curves, `post_test_accuracy_mean` is the mean held-out ideal-image accuracy after acquisition and retraining. `batch_utility_mean` is the mean change from the pre-acquisition model within that job. A positive utility indicates improvement from that acquisition relative to its own starting point; it does not by itself establish superiority over another strategy unless the paired seed comparison is inspected.

## Current next step

The immediate new evidence will come from the budget-aware protocol, not from adding another fixed-three-epoch curve. Finish the 17 missing Account 4 Task 3a cells, aggregate the complete 600-cell validation grid, then compare final strategies under selected schedules and fixed-update control. That experiment will show whether the historical budget-dependent rankings persist after the training schedule is allowed to match the amount of labeled data.
