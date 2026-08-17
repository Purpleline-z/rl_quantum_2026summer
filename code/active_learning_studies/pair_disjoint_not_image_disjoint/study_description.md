# Pair-Disjoint, Not Image-Disjoint Study

## What this study tests

This is the existing controlled active-learning study. It compares methods for selecting additional unordered RHEED image-pair groups after a fixed initial labelled set.

## Split policy

Initial and candidate partitions are disjoint by unordered pair ID. They are not required to be disjoint by image ID: the same image may occur in an initial pair and a different candidate pair. Ideal reference, utility-validation, and outer-test images are partitioned separately by the current pipeline.

## Data and model

The current evidence family uses dataset version v1.8, a ResNet-18 encoder, a 512-to-256-to-5 Bradley--Terry reward head, and full-model fine-tuning. Individual study manifests are the source of truth for seeds, budgets, selectors, epochs, learning rate, and initialization.

## Use in the technical report

This study supplies the current active-learning results. Its pair-level split must be stated whenever results are interpreted; it does not establish strict image-level generalization across initial and candidate partitions.

## Comparability limits

Results are comparable only within matching manifest-defined seeds, candidate pools, budgets, preprocessing modes, and evaluation contracts. They are not automatically comparable to Classifier2 pairwise-holdout results or to the strict-image-disjoint study.
