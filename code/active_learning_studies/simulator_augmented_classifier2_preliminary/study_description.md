# Simulator-Augmented Classifier2 Preliminary Study

## Question

The simulator can generate large numbers of RHEED-like raster images with known reconstruction labels. This study tests whether those labels improve a Classifier2-compatible model on real images, rather than assuming that more synthetic images are automatically useful.

## Available synthetic data

The schema-v6 `p656_aligned_peak` archive contains 2,250 images: three views of 250 latent surfaces for each of Twinned(2 x 1), c(6 x 2), and root-13. The archive marks every image `synthetic_training_only`. Its three views of the same latent surface are correlated; the study partitions by `same_surface_group`, never by individual PNG.

The simulator does not supply `(1 x 1)` or HTR labels. Its result therefore uses a real-image, three-class outer test and cannot be numerically pooled with historical five-class Classifier2 results.

## Model and comparison arms

All arms use the shipped SimCLR ResNet-18 initialization and the compatible five-output reward head. Synthetic supervised loss is applied only to the three output dimensions the simulator supports. Real fine-tuning keeps the existing pairwise, ideal-anchor, and Bad-image losses.

The five paired seeds compare:

1. **real-only baseline** — what real labeled data achieves without simulated images;
2. **synthetic-only transfer diagnostic** — how a simulator-trained model behaves on real outer-test images before real fine-tuning;
3. **synthetic pretrain then real fine-tune** — whether synthetic labels provide a useful starting representation once the model adapts to real data.

The synthetic-only arm is essential. A low score measures the simulator-to-real gap; a transfer gain despite that gap would show that some features still transfer after real fine-tuning.

## Split and evaluation policy

Real images are partitioned by image identity before training into mutually exclusive train, validation, and outer-test sets. Synthetic partitions preserve latent-surface groups. Hyperparameters use validation only. The primary endpoint is real-only, image-disjoint outer-test accuracy with per-class counts, especially the small Twinned count, reported alongside aggregate performance.

## Current status and research decision

The folder contains a preflight validator, strict real split builder, resumable queue, aggregation code, and behavior tests. No performance result is currently committed. The first experimental decision will come from the paired real outer test: if transfer consistently exceeds real-only across seeds, simulator pretraining becomes a justified augmentation direction; if synthetic-only and transfer both underperform, the next research task is to inspect domain mismatch rather than increase synthetic image count blindly.
