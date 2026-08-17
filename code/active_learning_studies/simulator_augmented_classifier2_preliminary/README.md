# Simulator-Augmented Classifier2: Research and Reproduction Guide

## Why this study exists

The RHEED simulator produces images with absolute synthetic labels, while the real dataset is small and contains experimental variation that a simulator may not reproduce. The relevant question is therefore transfer to real held-out images, not synthetic classification accuracy. This folder implements that comparison without modifying the PhD-authored documents under `classifier2/`.

## Protocol at a glance

| Arm | Training history | Evaluation question |
|---|---|---|
| Real-only | Real pairwise/ideal/Bad training only | What does the available real training data achieve? |
| Synthetic-only | Synthetic supervised three-class pretraining only | How large is the simulator-to-real domain gap before adaptation? |
| Synthetic then real | Synthetic supervised pretraining followed by unchanged real fine-tuning | Do synthetic labels supply features that remain useful after real adaptation? |

All three arms use the same fixed real image partitions and the shipped SimCLR initialization for each seed. The outer test contains only real images and is image-disjoint from real train/validation data. The synthetic archive is never evaluation data.

## Data contract

The expected ZIP is `classifier-sto-v6-dual-raster__p656_aligned_peak.zip`. It contains 2,250 656×492 PNGs, arranged as three views for each of 250 latent surfaces per supported reconstruction class. The preflight validator checks archive hash, manifest version, image hashes, labels, dimensions, training-only designation, and latent-surface grouping.

The supported simulator classes are Twinned, c(6 x 2), and root-13. The study's real outer test is therefore restricted to those three classes. `(1 x 1)` and HTR are not synthetic targets and no result here answers five-class performance.

## Architecture and losses

The model retains the five-output Classifier2-compatible reward head. During synthetic pretraining, cross-entropy is applied only to the three supported output dimensions. During real fine-tuning, the model uses the existing pairwise preference, ideal-anchor, and Bad-image objectives. This design asks whether the synthetic raster features help real training while preserving the real-data supervision contract.

## Output and interpretation

Each seed produces compact JSON with real outer-test accuracy, per-class metrics, confusion information, training provenance, and the selected arm. Aggregation reports raw per-seed values, paired transfer-minus-real-only differences, bootstrap intervals, and figures. A positive synthetic-to-real claim requires transfer to show a consistent paired advantage on the untouched real outer test. A strong synthetic-only result is not the target; it is a domain-gap diagnostic.

## Operational behavior

The preflight tools and queue write compact audits/results to the caller's output location. A queue cell writes a checkpoint each epoch, writes final JSON atomically on completion, and deletes only its own completed checkpoint. Re-running a queue skips completed JSONs. Extracted simulator images remain outside the repository and must not be committed.

## Current research decision

No arm has been run to completion in the committed evidence. The next step is the preflight audit followed by the three-arm paired comparison. Its outcome will determine whether to invest in simulator augmentation, inspect domain mismatch, or keep synthetic images out of downstream training.
