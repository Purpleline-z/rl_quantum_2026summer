# Historical Fixed-Protocol Budget-Curve Report

## What this result family is

This folder records an earlier controlled budget curve in which acquisition budget varied while downstream learning rate and epoch count were fixed. It is useful because it exposed the original non-monotonic behavior and motivated the current budget-aware calibration. It is not the final protocol for future claims.

## Historical protocol

The historical runs used single-shot pair acquisition, pair-disjoint initial/candidate groups, a SimCLR-initialized ResNet-18 Bradley--Terry reward model, learning rate `1e-4`, and three fine-tuning epochs. The main endpoint was held-out ideal-image accuracy after retraining. Pair groups did not overlap between the initial set and candidate pool, but images could recur in different groups.

## What was observed

The budget curves showed that mean accuracy and batch utility did not move monotonically as the acquired-group budget increased. Some selectors improved at particular budgets and fell at others. This was the evidence that made “more selected pairs” an inadequate explanation for performance: selection quality, pair redundancy, optimization schedule, and the number of updates per epoch were entangled.

## What it means

The fixed protocol was a valid first comparison because every strategy at a given budget received the same training recipe. Its limitation is across-budget interpretation: a 100-group training set contains more minibatches per epoch than a 10-group set, while three epochs may be insufficient or overly aggressive depending on the amount and composition of acquired data. Therefore a non-monotonic curve cannot be attributed to a selector alone.

## Research decision produced by this result

The follow-up is Task 3a/3b/3c in the parent study. Task 3a performs validation-only LR/epoch calibration by budget and encoder. Task 3c will run both the selected-epoch control and a fixed-total-update control. If a selector remains ahead under both controls, the result points to informative label selection rather than an accidental advantage from the historical fixed schedule.

## Provenance

The source tables and figures in this folder are retained as historical evidence. They must be read with their own manifests and should not be pooled with the completed symmetry factorial, Task 3a calibration, or later image-disjoint studies.
