# Strict Image-Disjoint Study

## What this study tests

This study is a proposed reproduction tool. It tests whether active-learning conclusions remain similar when image IDs, rather than only pair IDs, are partitioned before initial labels and candidate pairs are selected.

## Split policy

Images are assigned before pairs are constructed to four mutually exclusive partitions: initial labelled training, candidate acquisition, validation, and outer test. A pair is eligible only when both images belong to its assigned partition. The runner writes leakage audits for every seed and fails when the declared split contract is violated.

## Data and model

The fixed settings file declares the candidate encoder initializations, learning-rate and epoch grid, label budgets, seed list, and training controls.

## Use in the technical report

This study is not part of the current technical-report evidence unless a completed, audited run is explicitly cited. It must not be pooled with pair-disjoint results.

## Comparability limits

Strict image partitioning reduces the number of eligible pairs. The completed capacity audit found common capacity for 3 initial labelled groups and 69 candidate groups across the five declared seeds; the fixed shared acquisition budgets are therefore 10, 25, 50, and 69. No seed-specific budget substitution is allowed.
