# Reproduction Protocol Report

## Objective

This experiment tests whether active selection improves downstream performance over paired Random selection under an image-disjoint evaluation contract. It does not attempt to reproduce or pool the legacy pair-disjoint results.

## Leakage contract

Image identifiers are split before pair groups are formed. Initial labels and the candidate pool contain only pairs whose two images are in the training partition. Validation and outer pairwise holdouts contain only pairs internal to their respective partitions. The ideal-image reference, validation, and outer-test partitions are mutually exclusive and are checked against all pairwise image identifiers. Each split fails closed on any image or exact-pair overlap.

## Training contract

The same ResNet-18 Bradley--Terry architecture, augmentation, AdamW optimizer, batch size, weights, and seed policy are used across comparisons. SimCLR and ImageNet initialization are each calibrated over the declared learning-rate and epoch grid using validation ideal-image accuracy only. The outer ideal-image test is not read during calibration. A matched fixed-optimizer-update control runs exactly the configured number of optimization steps for every budget.

## Planned analyses

The primary analysis contains ten paired seeds, Random, Uncertainty, and Uncertainty + Diversity, and requested budgets of 10, 25, 50, 75, and 100. The actual acquired budget is retained when strict eligibility makes a requested budget unreachable. Outer ideal-image accuracy is the primary endpoint. Per-class outcomes, validation accuracy, pairwise-holdout accuracy where available, seed-level results, bootstrap confidence intervals, paired permutation tests, and Holm--Bonferroni decisions are generated without combining the sensitivity analysis with the primary analysis.

## Preliminary feasibility audit

The completed `split_capacity.csv` audit found 111--119 eligible training pairs across the ten planned seeds. With 50 initial pairs, only 61--69 candidate pairs remain. Therefore requests for 75 and 100 acquisitions are structurally unreachable under this strict split and will be reported with their actual budget rather than substituted with pair-disjoint data. Some pairwise validation/holdout cells are sparse; ideal-image validation and outer-test partitions are therefore the preregistered selection and primary endpoint, while pairwise holdout metrics remain auxiliary and may be unavailable for individual seeds.

## Claim rules

A selector may be described as superior only when its strict image-disjoint paired result against Random survives Holm--Bonferroni correction. SimCLR may be described as inferior only after a matched completed comparison consistently supports that conclusion. The legacy Classifier2 accuracy remains non-comparable unless full data, split, model, training, and metric equivalence is demonstrated.
