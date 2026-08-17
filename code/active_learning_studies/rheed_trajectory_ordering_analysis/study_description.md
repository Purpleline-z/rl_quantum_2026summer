# RHEED Trajectory Ordering Analysis

## Physical question

Laboratory observations describe growth trajectories rather than isolated images: a run commonly begins in `(1 x 1)`, can pass through a disordered `Bad` regime before another reconstruction, and HTR generally appears late. These statements are higher-order constraints on a whole sequence. They are not the same as frame metadata such as temperature, and they should not be inserted as if they were image labels.

## Two separate stages

The completed stage is a filename/order audit. It extracts trajectory IDs, leading indices, frame tokens, and any temperature token present in filenames. It reports missing indices, duplicates, and parsing ambiguity. A numerical filename index gives an ordering observation; it does not prove a physical state transition.

The implemented next stage is a soft decoder for externally supplied six-state frame probabilities: `(1 x 1)`, `Bad`, Twinned, c(6 x 2), root-13, and HTR. A Viterbi-style path score combines those frame probabilities with penalties for violating the three laboratory preferences. Weak, moderate, and strong penalties are evaluated as sensitivity settings so the physics assumption is visible rather than hidden inside a classifier.

## Why decoding is separate from model training

The decoder acts after a framewise classifier has produced calibrated probabilities. It does not train the image encoder, alter active selection, append process metadata, or manufacture new labels. This separation lets a reader ask two clear questions: what did the image model score for each frame, and how did a stated trajectory preference alter the final sequence?

## Current evidence and next requirement

The repository has filename-order evidence and decoder code, but no saved calibrated downstream six-state classifier probability table that includes a validated `Bad` state. Therefore there is no decoded trajectory result to interpret yet. The next concrete input is a table containing trajectory ID, ordered frame ID, and six calibrated class probabilities from an externally validated downstream classifier. The decoder can then report raw paths, constrained paths, changed frames, and rule-specific penalties for each run.

## Relation to other studies

This study is complementary to process-metadata fusion. Metadata changes a per-frame predictor using measurements available at image time; trajectory decoding uses a sequence-level physical preference after prediction. Neither result should be merged with pair-disjoint active-learning curves, which select human comparison labels rather than decode growth trajectories.
