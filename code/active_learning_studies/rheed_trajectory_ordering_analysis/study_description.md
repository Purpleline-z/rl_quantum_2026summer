# RHEED Trajectory Ordering Analysis

## Question

This study has two deliberately separated stages:

1. a descriptive audit of chronological ordering recoverable from trajectory filenames; and
2. a prepared, **soft higher-order decoding** analysis for externally supplied six-state classifier probabilities.

The decoder is not metadata embedding and does not train or modify an image model. It applies three physics-team trajectory preferences only at inference: start at `(1 x 1)`; pass through `Bad` before another reconstruction; and do not leave HTR after it appears. It reports weak, moderate, and strong penalty sensitivities alongside raw framewise predictions.

## Data, split, and model version

The completed filename audit uses trajectory image paths and filename-derived ordering fields only. There is no train/validation/test split, no preference label, and no trained model in that stage.

The prepared decoding stage requires an external, calibrated six-state probability table. No downstream Classifier2 checkpoint is currently present locally or in the reviewed public repository, so it has not scored any real trajectory. When a checkpoint is supplied, scoring and decoding will be inference-only; they will not train on any trajectory image. Therefore train/validation/test/candidate image-disjoint requirements are not applicable to this inference-only study. Any future checkpoint reproduction must document an image-disjoint training/validation/test partition separately.

## Use in the technical report

Its completed outputs support only cautious descriptive statements about observed frame ordering. The new configuration, literature review, decoder, and synthetic behavior tests are implementation evidence only. They are not used in the current active-learning conclusions and cannot support an accuracy claim until independently annotated trajectories and a frozen scoring model are available.

## Comparability limits

Filename order is an observation, not a physical-state annotation. The physics statements are laboratory priors with possible exceptions, not labels. The decoder therefore uses soft penalties and exposes every decoded change from raw model predictions. This study is not comparable to either active-learning protocol, and it must not be used to report active-learning performance.

The full rationale and alternatives are in `higher_order_trajectory_constraint_literature_review.md`; the exact, versioned parameters are in `higher_order_trajectory_constraint_configuration.json`.
