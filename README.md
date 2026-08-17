# RHEED Active Learning and Reconstruction Prediction

This repository studies how RHEED images, pairwise expert preferences, simulator-generated images, process-monitor metadata, and growth-trajectory knowledge can contribute to reconstruction prediction and experimental decision support.

The project is organized as separate studies because the inputs answer different questions. Pairwise preference data asks which image-pair labels are worth acquiring. Ideal reconstruction images measure downstream classification. Simulator images test whether clean synthetic labels transfer to real RHEED. Growth-monitor logs test whether causal experimental context improves an image model. Trajectory constraints encode laboratory knowledge about the order of states. Results are linked where comparable and kept separate where their data, split, or endpoint differs.

## Start here

- [Main technical report](code/TECHNICAL_REPORT.md) explains the completed pair-selection results, their meaning, and the next decisions they motivate.
- [Active-learning program](code/active_learning_program/) contains the reusable Bradley--Terry model, acquisition methods, data loading, and behavior tests.
- [Pair-disjoint active-learning study](code/active_learning_studies/pair_disjoint_not_image_disjoint/) is the principal completed selection benchmark and the in-progress budget-aware calibration.
- [Process-metadata fusion study](code/active_learning_studies/process_metadata_fusion_for_rheed_prediction/) defines the causal image-plus-monitor architecture and the data interface needed before an experiment can run.
- [Simulator-augmented Classifier2 study](code/active_learning_studies/simulator_augmented_classifier2_preliminary/) defines a real-image evaluation of synthetic supervised pretraining.
- [Trajectory-ordering analysis](code/active_learning_studies/rheed_trajectory_ordering_analysis/) separates filename chronology and physics-informed sequence decoding from frame-level image prediction.

`code/classifier2/` is a reference implementation and report authored for the earlier Classifier2 work. Those documents are preserved unchanged. The sibling data directory is not committed; it contains the RHEED images and source CSV files.

## Evidence status

The pair-disjoint benchmark has completed historical selector curves and preprocessing comparisons. Task 3a, the validation-only budget-aware calibration intended to replace a single fixed training schedule, currently has 583 of 600 cells complete; the remaining 17 Account 4 cells must finish before its final protocol table and new budget curves can be generated. The simulator and process-metadata folders contain implemented, auditable protocols but no performance result yet.

Every study owns its results. Large images, raw laboratory logs, resumable checkpoints, and Colab working directories are intentionally not committed.
