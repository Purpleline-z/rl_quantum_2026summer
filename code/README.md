# RHEED Pairwise Active Learning

This repository studies how labelled RHEED image-pair comparisons can train and improve a downstream reconstruction classifier.

## Repository map

- `active_learning_program/`: the current reusable implementation, named study launchers, and automated code-behavior tests.
- `active_learning_studies/pair_disjoint_not_image_disjoint/`: the current controlled pair-level active-learning evidence and its manifests.
- `active_learning_studies/strict_image_disjoint/`: a separate strict image-level direction - not yet implemented.
- `active_learning_studies/image_representation_analysis/`: PCA, t-SNE, and nearest-neighbor representation analysis.
- `active_learning_studies/rheed_trajectory_ordering_analysis/`: filename-order audit plus a prepared, soft higher-order trajectory decoder; it is separate from metadata embedding and inactive until an externally calibrated downstream classifier score table is available.
- `active_learning_studies/process_metadata_fusion_for_rheed_prediction/`: a registered, causal process-metadata fusion architecture and session-bundle audit. It is not a completed performance study and requires multiple image-resolved growth sessions before training is permitted.
- `classifier2/`: the reference Classifier2 code, report, checkpoint, and source notes.
- `TECHNICAL_REPORT.md`: the main interpretation of the evidence.

The data directory is a sibling of `code/` by default. Set `RHEED_DATA_ROOT` or pass `--data-root` when using a different location.

Each study owns its own results. Do not create or rely on a generic central results folder.
