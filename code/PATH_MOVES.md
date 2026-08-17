# Local Path Moves

This local-only cleanup replaces generic folders with names that describe the scientific purpose of their contents. No source data were deleted and no Git commit or push was made.

| Previous location | New location | Reason | Reference updates required |
|---|---|---|---|
| `src/` | `active_learning_program/` | Current active-learning implementation and executable studies are now named by purpose. | Python imports, tests, README, and report links. |
| `src/tests/` | `active_learning_program/code_behavior_tests/` | Pass/fail checks of implementation behavior. | Pytest discovery and test imports. |
| `results/local_runs/selection_benchmark/` | `active_learning_studies/pair_disjoint_not_image_disjoint/results/selection_benchmark/` | Results now belong to the study that produced them. | Report links and runner result root. |
| `results/local_runs/protocol_diagnostics/` | `active_learning_studies/pair_disjoint_not_image_disjoint/results/protocol_diagnostics/` | Learning-rate, epoch, encoder, and bridge comparisons belong to the pair-disjoint study. | Report links and runner result root. |
| `results/completed_experiments/` | `active_learning_studies/*/results/selected_summaries/` | Compact summaries are stored beside the study they summarize. | Report links. |
| `results/local_runs/representation_exploration/` | `active_learning_studies/image_representation_analysis/results/representation_exploration/` | Representation analysis is a separate question from pair acquisition. | Representation runner output root and report links. |
| `results/local_runs/temporal_constraint_audit/` | `active_learning_studies/rheed_trajectory_ordering_analysis/results/temporal_constraint_audit/` | Trajectory ordering is a separate descriptive study. | Trajectory runner output root. |
| `more_epochs_and_learning_rates/` | `active_learning_studies/strict_image_disjoint/` | Strict image-disjoint reproduction is a parallel active-learning study. | Strict-study README and launcher paths. |
| `classifier2/downstream_classifier_scripts/` | `classifier2/classifier_training_code/` | The folder contains downstream classifier training code. | Pipeline, bridge, report, and README links. |
| `classifier2/pretraining/` | `classifier2/pretraining_source_notes/` | The folder contains source/provenance notes, not executable pretraining. | Report links. |
| `classifier2/simclr_resnet18_encoder.pth` | `classifier2/simclr_encoder_checkpoint/simclr_resnet18_encoder.pth` | The checkpoint is named and separated from implementation code. | Model loading paths. |
