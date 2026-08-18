# Leakage-Safe Active Selection from Pairwise Image Preferences

## An evidence-aware computational study

### Abstract

This study examines how to select a limited number of image-pair preference labels for training a Bradley--Terry reward model. The practical question is not simply whether a model can fit already labelled comparisons; it is whether a selection rule chooses comparisons that improve a downstream reconstruction classifier. We implement a ResNet-18 reward model, compare uncertainty and coverage-aware acquisition rules, and evaluate a fixed outer set of ideal images. The report deliberately separates historical diagnostic results from final evidence: a later SHA-256 audit found byte-identical images crossing prior partitions, so those results cannot be treated as leakage-free final estimates. The revised implementation uses content identity rather than path names to construct and audit splits, removes every outer-test identity from pairwise/unlabelled trajectory images and negative anchors, and records epoch-wise validation metrics for validation-only early stopping. Thus, the main contribution is a reproducible evaluation protocol for learning from pairwise image preferences under a label budget, not a claim of final state-of-the-art accuracy.

## 1. Introduction

Pairwise labels ask a simple question: *given two images, which one is preferred for a specified reconstruction type?* They are often easier for experts to provide than absolute scores. However, expert time is limited, so an active-learning system must choose which hidden comparisons to reveal.

We study the following computational question: among candidate **pair groups**, which groups should be labelled and added to a preference-trained image model? The final endpoint is an ideal-image reconstruction prediction task, not agreement with the model's own acquisition score. This distinction prevents a selector from being judged solely by a quantity it helped define.

Figure 1 shows the intended evaluation firewall. Validation can select training settings and a stopping epoch; the outer test is used only after these choices are frozen.

![Pipeline and evaluation protocol](active_learning_studies/pair_disjoint_not_image_disjoint/paper_assets/pipeline_and_evaluation_protocol.svg)

*Figure 1. Computational pipeline and evaluation firewall.*

## 2. Method

### 2.1 Preference reward model

Each image \(x\) is passed through a ResNet-18 encoder and a reward head that returns one score per reconstruction class, \(r_\theta(x,t)\). For a labelled pair \((x_i,x_j)\) of type \(t\), the Bradley--Terry model assigns the probability that the first image is preferred as

\[
P_\theta(x_i \succ x_j \mid t) = \sigma\!\left(r_\theta(x_i,t)-r_\theta(x_j,t)\right),
\qquad \sigma(z)=\frac{1}{1+e^{-z}}.
\]

In plain language, the model is confident that the first image should win when its learned score is much larger than the second image's score. For a first-image win, the loss is

\[
\mathcal L_{BT} = -\frac{1}{|D|}\sum_{(i,j,t)\in D} w_{ij}\log P_\theta(x_i \succ x_j\mid t),
\]

where \(w_{ij}\) is an optional annotator-confidence weight. The implementation also handles reversed wins, ties, and not-applicable labels using the corresponding loss terms in `pairwise_active_learning_pipeline.py`.

The model is initialized with an image encoder and fine-tuned with AdamW. Ideal reference images serve as training anchors only; they are neither validation nor outer-test images.

### 2.2 Active selection

Candidate comparisons retain their images but hide their human winner labels. A simple uncertainty score is

\[
u(i,j,t)=1-\left|2\sigma(r_\theta(x_i,t)-r_\theta(x_j,t))-1\right|.
\]

It is largest when the model is close to a 50--50 preference, so uncertainty acquisition asks for labels on comparisons it cannot decide. The project also tests random sampling, core-set coverage, cluster-quota uncertainty, and an uncertainty--diversity combination. For the latter, candidates are ranked by

\[
s(i,j)=\lambda\,\widetilde u(i,j)+(1-\lambda)\,\widetilde d(i,j),
\]

where \(\widetilde u\) and \(\widetilde d\) are normalized uncertainty and embedding-space diversity, and \(\lambda\) controls their balance. This is a project-specific heuristic, not a claim of a universal standard.

### 2.3 Algorithms

```text
Algorithm 1: Build leakage-safe partitions
Input: ideal images, pairwise rows, seed
1. Compute SHA-256 identity for every image file.
2. Assign each unique ideal identity to exactly one of {reference, validation, outer test}.
3. Collect identities assigned to outer test.
4. Remove every pairwise/unlabelled-trajectory row and negative anchor whose image identity is in outer test.
5. Form pair-disjoint initial and candidate pair groups; image reuse across those pair groups is allowed.
6. Audit and save every path/hash overlap count; fail a run if any test-identity overlap remains.
Output: pair-disjoint training/candidate groups and identity-disjoint ideal partitions.
```

```text
Algorithm 2: Train with validation-only early stopping
Input: labelled pairs, reference images, utility-validation set, maximum epochs E, patience p
best_metric = -infinity; stale = 0
for epoch = 1,...,E:
    optimize Bradley--Terry and permitted reference-anchor losses for one epoch
    record mean training loss
    metric = reconstruction_accuracy(utility-validation)
    record metric; never evaluate outer test here
    if metric improves best_metric: save model; best_metric = metric; stale = 0
    else: stale += 1
    if stale >= p: stop
return saved model with highest validation metric
```

```text
Algorithm 3: Single-shot active pair selection
Input: initial labelled groups L, label-hidden candidate groups C, budget b
train a reward model on L
score only permitted candidate images/embeddings
select b distinct pair groups using a declared acquisition rule
reveal winner labels only for selected groups
retrain from scratch on L plus selected groups
evaluate once on the frozen outer test after protocol choices are fixed
```

## 3. Data Protocol and Leakage Prevention

The unit of acquisition is an unordered pair group; initial and candidate groups are pair-disjoint. The same image may occur in different non-test pair groups because image-disjoint pair partitions are unnecessarily restrictive for this application. In contrast, no outer-test identity may appear in training pairs, candidate images, reference anchors, utility validation, or Bad-image anchors.

![Leakage-safe split protocol](active_learning_studies/pair_disjoint_not_image_disjoint/paper_assets/leakage_safe_split_protocol.svg)

*Figure 2. Required split protocol. Identity means SHA-256 file content, not filename.*

The implementation enforces this rule in `active_learning_program/pairwise_active_learning_pipeline.py`. It retains audit counts for path and content-identity overlaps. This avoids the failure mode in which duplicated image bytes appear under different paths and silently leak from a train-time pool into the test set.

## 4. Implementation and Reproducibility

The reusable program is in `active_learning_program/`. `pairwise_active_learning_pipeline.py` loads preference CSVs, creates ideal partitions, trains the reward model, and executes selectors. `resumable_model_training.py` now stores per-epoch training loss and utility-validation accuracy and restores the best validation checkpoint. The budget-aware runner uses `utility_validation` for training decisions and records `outer_test_not_evaluated: true` during calibration.

Each run should save: configuration; data hashes; pair manifests; split audit; per-epoch metrics; selected pair IDs; seed; and all aggregate CSVs used in figures. Every paper number must trace to one of these saved artifacts.

## 5. Results and Evidence Status

### 5.1 Historical selector curves: diagnostic only

The repository contains five-seed, fixed-schedule curves across acquisition budgets. Figure 3 is generated directly from `results/selected_summaries/budget_curve_5seed/tables/strategy_performance_by_acquisition_budget.csv`.

![Historical budget curve](active_learning_studies/pair_disjoint_not_image_disjoint/paper_assets/historical_selector_budget_curve.png)

*Figure 3. Historical budget curves. They describe previous runs but are not final leakage-safe estimates.*

These curves suggest that selector rankings depend on budget and preprocessing; they do not support one selector as universally best. They also do not isolate selection quality from the amount of optimization performed, because a fixed number of epochs processes different amounts of acquired data at different budgets.

### 5.2 No historical evidence for epoch 10

![Missing epoch-wise evidence](active_learning_studies/pair_disjoint_not_image_disjoint/paper_assets/missing_historical_epoch_curve.svg)

*Figure 4. Evidence status for historical epoch choices.*

The archived outputs compare endpoint settings such as 1, 2, 3, 5, and 10 epochs, but do not provide validation accuracy after every epoch for a common leakage-safe protocol. Therefore, the report makes **no claim** that ten epochs is optimal or that early stopping would stop at epoch 10. A future curve must plot mean training loss and utility-validation accuracy by epoch, optionally with seed variation, and choose the epoch before one outer-test evaluation.

### 5.3 Evidence categories

| Category | Status | Permitted interpretation |
|---|---|---|
| Historical fixed-schedule budget curves | Completed, but affected by identity audit | Diagnostic context only |
| Path-based budget-aware calibration | Completed artifact | Do not use for final test claims |
| SHA-256-safe split + epoch logging | Implemented | Required basis for new experiments |
| Leakage-safe calibration and final selector curve | Planned | Required before final performance conclusion |

## 6. Limitations and Next Experiment

The outer ideal-image endpoint is small and may have high seed variation. The historical data-identity leakage means previously reported outer-test numbers must not be presented as final generalization estimates. The next experiment must rebuild the split by identity, train using only permitted data, select hyperparameters and stopping epochs from utility validation, freeze those choices, and then run a single final outer-test comparison across declared strategies and seeds.

## 7. Conclusion

Pairwise active learning can be evaluated rigorously only when the acquisition unit, training data, validation decisions, and final test set are explicitly separated. This repository now provides the necessary computational safeguards: pair-disjoint acquisition groups, content-identity exclusion of outer-test images, validation-only early stopping, and artifact-level auditing. The historical curves are useful motivation; the leakage-safe rerun is the necessary evidence for a final result.

## References

1. R. A. Bradley and M. E. Terry. *Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons*. Biometrika, 1952.
2. B. Settles. *Active Learning Literature Survey*. University of Wisconsin--Madison, 2009.
3. K. He, X. Zhang, S. Ren, and J. Sun. *Deep Residual Learning for Image Recognition*. CVPR, 2016.
4. T. Chen et al. *A Simple Framework for Contrastive Learning of Visual Representations*. ICML, 2020.

## Appendix A. Figure Provenance

- Figure 1: generated by `active_learning_studies/pair_disjoint_not_image_disjoint/generate_academic_report_assets.py`.
- Figure 2: generated by the same script from the implemented split contract.
- Figure 3: generated by the script from the cited historical CSV; it introduces no new numerical results.
- Figure 4: generated by the script because no qualifying historical epoch-wise validation log exists.
