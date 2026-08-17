# Simulator-Augmented Classifier2 Preliminary Study

## Question

Does labelled, simulator-generated RHEED data improve a Classifier2-compatible model on a real-image, strictly image-disjoint three-class outer test after real-data fine-tuning?

## Data, split, and model version

The preliminary synthetic source is the schema-v6 `p656_aligned_peak` archive. It contains only Twinned(2 x 1), c(6 x 2), and (√13 x √13) synthetic images and is marked `synthetic_training_only` in its manifest. The archive is never evaluation data.

Real image IDs are partitioned before training into disjoint train, validation, and outer-test sets. Synthetic views are partitioned by `same_surface_group`, not by individual PNG. The model is a five-output ResNet-18 Bradley--Terry reward model initialized from the shipped SimCLR encoder. Synthetic pretraining applies standard cross-entropy only to the three simulator-supported output dimensions; real fine-tuning uses the same pairwise, ideal-anchor, and Bad-image losses as the Classifier2-style protocol.

## Use in the technical report

This is a preliminary sim-to-real study. Its primary endpoint is three-class real outer-test macro accuracy, reported per seed and with paired differences from a real-only baseline. It cannot revise historical five-class Classifier2 results.

## Comparability limits

The simulator does not provide 1x1 or HTR labels. The outer test contains few real Twinned ideal images, so all class counts and uncertainty intervals must accompany aggregate values. Synthetic-only real performance is a domain-gap diagnostic, not independent validation evidence.
