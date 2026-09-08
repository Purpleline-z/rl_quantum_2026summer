# Static Peak-Aware RHEED Reward Model

## What this study reproduces

Peak-Sequence Transformer (PS-Transformer) represents each RHEED video frame with a compact, diffraction-peak-focused representation and then uses the ordered sequence of frames to recognize changes during growth. Time matters in that setting: a transition is identified partly from where it sits in the history of the video.

This directory adapts the peak-focused part of that idea to this repository's STO data. The available observations are static images, not videos, so there is no frame sequence on which a Transformer could learn growth dynamics. The model therefore combines two complementary measurements from a single RHEED image:

- A ResNet-18 image representation captures the full spatial appearance of the diffraction pattern.
- A compact profile representation records horizontal and vertical intensity structure, peak strength, contrast, and variation. These are simple measurements of the streak/spot distribution that physicists inspect in a RHEED pattern.

The output is trained from the project's pairwise expert labels with Bradley--Terry learning. For a chosen reconstruction type, the model assigns each image a reward. The difference between two rewards becomes the probability that the first image is the better example of that reconstruction. This directly supplies the uncertainty signal needed when choosing the next pair to label.

## Difference from the paper

| Paper | This study |
|---|---|
| Ferroelectric nitride MBE RHEED videos | Static STO RHEED images |
| Peak-aware features followed by a temporal Transformer | Peak-aware features fused with a ResNet image embedding |
| Frame labels refined through video pseudolabeling | Expert pairwise winners, ties, and not-applicable labels |
| Real-time video throughput and power measurements | Compact image inference for active-pair selection |

The study is designed to answer a useful local question: do peak-aware image measurements improve reconstruction reward prediction and the information available to the pair selector beyond the existing image encoder? It does not evaluate the paper's video throughput or temporal-transition performance.

## Expected outputs

Each trained model returns five reconstruction rewards in this order: `(1 x 1)`, `Twinned(2 x 1)`, `c(6 x 2)`, `(√13 x √13)`, and `HTR`; a separate quality score; and a 512-dimensional image embedding. For a candidate pair it exports reward margins, per-type preference probabilities, mean entropy, and concatenated embeddings. Existing uncertainty, diversity, hybrid, cluster, core-set, and MC-dropout selectors can use these values without changing their acquisition rule.

The definitive study uses five fixed image-identity folds. In each fold, reward-model fine-tuning sees only pairs with two training images. Validation pairs contain at least one validation image and no test image. Test pairs contain at least one held-out image, giving a useful sample of predictions involving a new RHEED pattern while preserving an unseen endpoint in every test comparison. Test images and labels do not participate in reward-head fine-tuning, validation, calibration, or architecture selection.

The laboratory RHEED-SimCLR ResNet-18 is frozen and supplies an in-domain representation learned without pairwise preference labels. It may have self-supervised exposure to the image corpus, which is appropriate for active-learning representation pretraining but is not a fully inductive unseen-pixel benchmark. Both image-only and image-plus-peak heads are trained over three seeds in every outer fold. Reports include decisive-pair accuracy, all-label macro-F1 (including `tie` and `not_apply`), calibration, per-reconstruction-type counts, and active-learning embeddings/reward margins/entropy. Temperature and tie/not-applicable thresholds are fitted with validation predictions only.

## Latest five-fold unseen-image result

<!-- FIVE_FOLD_UNSEEN_IMAGE_RESULTS_START -->

No five-fold classifier result has been published yet. The earlier session-held-out audit and random image-identity artifacts remain implementation records only; they are excluded from architecture selection and from this result section.

<!-- FIVE_FOLD_UNSEEN_IMAGE_RESULTS_END -->

Each completed training JSON records the RHEED-SimCLR checkpoint SHA-256, 120 loaded tensors, frozen-backbone policy, and the fact that pairwise preference labels were not used for pretraining.

Committed experiment evidence belongs in `paper_replicate/results/<self_explanatory_run_name>/`. The publication command copies compact tables, JSON records, plots, and written interpretations from Drive while excluding raw images, temporary checkpoints, optimizer state, model weights, and files larger than 15 MB. Each published run contains a SHA-256 manifest so a later reader can identify exactly which Drive outputs support the reported result.

## Layout

- `peak_aware_static_rheed_reward_model.py`: model and peak-profile extractor.
- `pairwise_and_absolute_label_dataset.py`: CSV discovery, image resolution, and identity-disjoint partition creation.
- `train_peak_aware_reconstruction_reward_model.py`: checkpointed Bradley--Terry training.
- `evaluate_image_disjoint_reconstruction_reward_model.py`: sealed test-pair evaluation.
- `export_active_learning_pair_selection_features.py`: selector-ready reward, uncertainty, and embedding export.
- `run_five_fold_unseen_image_reward_classifier_study.py`: one-T4 five-fold unseen-image training, evaluation, aggregation, and deployment export.
- `run_resumable_paper_replicate_task_queue.py`: legacy image-identity task queue retained only for earlier run reproducibility.
- `publish_drive_results_to_github.py`: filtered Drive-to-GitHub result publication.

See [COLAB_EXECUTION_COMMANDS.md](COLAB_EXECUTION_COMMANDS.md) for the one-T4 command, Drive output path, resume behavior, and publication procedure.
