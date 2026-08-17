# RHEED Pairwise Active Learning

This repository studies how labelled RHEED image-pair comparisons can train and improve a downstream reconstruction classifier.

## Repository map

- `active_learning_program/`: the current reusable implementation, named study launchers, and automated code-behavior tests.
- `active_learning_studies/pair_disjoint_not_image_disjoint/`: the current controlled pair-level active-learning evidence and its manifests.
- `active_learning_studies/strict_image_disjoint/`: a separate strict image-level reproduction tool; do not pool its results with the pair-disjoint study.
- `active_learning_studies/image_representation_analysis/`: PCA, t-SNE, and nearest-neighbor representation analysis.
- `active_learning_studies/rheed_trajectory_ordering_analysis/`: descriptive trajectory-ordering analysis.
- `classifier2/`: the reference Classifier2 code, report, checkpoint, and source notes.
- `TECHNICAL_REPORT.md`: the main interpretation of the evidence.

The data directory is a sibling of `code/` by default. Set `RHEED_DATA_ROOT` or pass `--data-root` when using a different location.

Each study owns its own results. Do not create or rely on a generic central results folder.
