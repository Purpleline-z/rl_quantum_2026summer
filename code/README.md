# RHEED Research Code and Evidence Map

## Research question

The central active-learning question is: given a limited number of human pairwise RHEED judgments, which unordered image-pair groups should be labeled so that retraining improves reconstruction prediction on held-out ideal images? The answer requires keeping three objects distinct:

- a **pair group** is the acquisition unit and contains one or more preference rows;
- an **image reward model** assigns five reconstruction-specific scores to an image;
- an **ideal-image outer test** measures the downstream prediction result after retraining.

The repository also contains studies that test different ways to improve the image model before or alongside active selection: representation analysis, simulated pretraining, process-metadata fusion, and trajectory-level constraints.

## Where to find each component

| Location | What it contains | How to interpret it |
|---|---|---|
| `active_learning_program/` | Reusable data loading, ResNet-18/Bradley--Terry model, acquisition methods, launchers, and behavior tests | Source code; it is not itself a result claim. |
| `active_learning_studies/pair_disjoint_not_image_disjoint/` | Historical five-strategy curves, preprocessing factorial, calibration artifacts, and Task 3a/3b/3c workflow | Main active-learning evidence. Pair groups are disjoint; images may repeat across pair groups. |
| `active_learning_studies/image_representation_analysis/` | PCA, t-SNE, nearest-neighbor, and encoder-space diagnostics | Explains representation geometry; it does not replace downstream evaluation. |
| `active_learning_studies/rheed_trajectory_ordering_analysis/` | Filename-order audit and soft, physics-informed trajectory decoder | A trajectory-level inference direction, separate from metadata fusion and training. |
| `active_learning_studies/simulator_augmented_classifier2_preliminary/` | Protocol for synthetic supervised pretraining followed by real-image evaluation | Tests whether simulator labels transfer; its three-class result is not interchangeable with five-class historical results. |
| `active_learning_studies/process_metadata_fusion_for_rheed_prediction/` | Causal monitor-log interface and late-fusion architecture | Requires multiple image-resolved session bundles before it can produce a performance result. |
| `classifier2/` | PhD-authored reference code, documentation, SimCLR checkpoint, and source notes | Read-only historical reference; do not alter these documents. |

## Current evidence and next research decision

The completed pair-disjoint curves show that selector performance depends on acquisition budget and image preprocessing. The next active-learning result should therefore use the budget-aware validation protocol rather than assume one epoch/LR setting fits every budget. That calibration is currently 583/600 cells complete and cannot yet support a new final budget curve.

The process-metadata and simulator studies have different purposes: metadata tests whether causal laboratory context changes prediction; synthetic pretraining tests whether clean generated labels transfer to real images. Both use image-disjoint real evaluation plans and should be interpreted separately from pair-disjoint selector curves.

## Data and reproducibility

The data directory is normally a sibling of `code/`. Set `RHEED_DATA_ROOT` or pass `--data-root` for a different location. Results are stored only beneath the study that generated them. Raw laboratory session ZIPs, images, Drive checkpoints, and Colab working copies are inputs or temporary state, not Git evidence.
