# Strict Image-Disjoint Study

## What this study tests

This study is a proposed reproduction tool. It tests whether active-learning conclusions remain similar when image IDs, rather than only pair IDs, are partitioned before initial labels and candidate pairs are selected.

## Split policy

Each pair is eligible for initial training or acquisition only when both images belong to the training image partition. Validation and outer-test partitions use separately assigned images. The runner writes leakage audits for every seed and fails when the declared split contract is violated.

## Data and model

The fixed settings file declares the candidate encoder initializations, learning-rate and epoch grid, label budgets, seed list, and training controls.

## Use in the technical report

This study is not part of the current technical-report evidence unless a completed, audited run is explicitly cited. It must not be pooled with pair-disjoint results.

## Comparability limits

Strict image partitioning reduces the number of eligible pairs. Requested budgets may therefore be unavailable, and the actual acquired budget must be reported.
