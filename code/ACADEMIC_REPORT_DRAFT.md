# Active Selection from Pairwise Image Preferences

### Abstract

This study examines how to select a limited number of image-pair preference labels for training the Bradley--Terry reward model, in order to improve the downstream reconstruction type classifier. We implement a ResNet-18 reward model, compare uncertainty and diversity-aware acquisition rules, and evaluate against a test set of ideal images with absolute labels. One thing to note is that an SHA-256 audit found byte-identical images crossing prior partitions (1-3 ideal images in each seed were included as unlabelled images in "trajectory" folder), so those results cannot be treated as leakage-free final estimates. The revised implementation uses content identity rather than path names to construct and audit splits, removes every outer-test identity from pairwise/unlabelled trajectory images and negative anchors, and records epoch-wise validation metrics for validation-only early stopping.

## 1. Introduction

Pairwise labels ask a simple question: *given two images, which one is preferred for a specified reconstruction type?* They are often easier for experts to provide than absolute scores. However, expert time is limited, so an active-learning system must choose which hidden comparisons to reveal.

We study the following computational question: among **candidate pairs**, which pairs should be labelled and added to a preference-trained image model? 

Figure 1 shows the intended evaluation firewall. Validation can select training settings and a stopping epoch; the outer test is used only after these choices are frozen.

![Pipeline and evaluation protocol](active_learning_studies/pair_disjoint_not_image_disjoint/paper_assets/pipeline_and_evaluation_protocol.svg)

*Figure 1. Computational pipeline and evaluation firewall.*

## 2. Method

### 2.1 Preference reward model

Each image $x$ is passed through a ResNet-18 encoder and a reward head that returns one score per reconstruction class, $r_\theta(x,t)$. For a labelled pair $(x_i,x_j)$ of type $t$, the Bradley--Terry model assigns the probability that the first image is preferred as

$$
P_\theta(x_i \succ x_j \mid t) = \sigma\!\left(r_\theta(x_i,t)-r_\theta(x_j,t)\right),
\qquad \sigma(z)=\frac{1}{1+e^{-z}}.
$$

In plain language, the model is confident that the first image should win when its learned score is much larger than the second image's score. For a first-image win, the loss is

$$
\mathcal L_{BT} = -\frac{1}{|D|}\sum_{(i,j,t)\in D} w_{ij}\log P_\theta(x_i \succ x_j\mid t),
$$

where \(w_{ij}\) is an optional annotator-confidence weight. The implementation also handles reversed wins, ties, and not-applicable labels using the corresponding loss terms in `pairwise_active_learning_pipeline.py`.

The model is initialized with an image encoder and fine-tuned with AdamW. Ideal reference images serve as training anchors only; they are neither validation nor outer-test images.

### 2.2 Active selection

Candidate comparisons retain their images but hide their human winner labels. A simple uncertainty score is

$$
u(i,j,t)=1-\left|2\sigma(r_\theta(x_i,t)-r_\theta(x_j,t))-1\right|.
$$

It is largest when the model is close to a 50--50 preference, so uncertainty acquisition asks for labels on comparisons it cannot decide. The project also tests random sampling, core-set coverage, cluster-quota uncertainty, and an uncertainty--diversity combination. For the latter, candidates are ranked by

$$
s(i,j)=\widetilde u(i,j)+\lambda\,\widetilde d(i,j),
$$

where $\widetilde u$ and $\widetilde d$ are normalized uncertainty and embedding-space diversity, and $\lambda$ is the additional weight assigned to diversity. This is a project-specific heuristic, not a claim of a universal standard.

### 2.3 Acquisition strategies and their roles

All methods select complete pair groups, never individual images. The candidate view contains image paths and permitted model outputs, but not the human winner label.

| Strategy | Selection rule | Why it may help | Main tuning or failure mode |
|---|---|---|---|
| Random | Uniform sample of candidate groups | Essential unbiased baseline | High variance at small budgets |
| Uncertainty | Largest $u(i,j,t)$ | Requests comparisons near the model decision boundary | Can repeatedly select visually similar pairs |
| Core-set | Farthest-first coverage in embedding space | Covers underrepresented image regions | Depends on encoder geometry and distance metric |
| Cluster-quota uncertainty | Spread selections over clusters, then rank by uncertainty | Avoids spending all labels in one region | Cluster count and quota can be mismatched to the data |
| Uncertainty + diversity | Rank by $s(i,j)$ | Balances difficult and nonredundant pairs | Requires validation choice of $\lambda$ |
| Cluster-Margin | Prefer informative pairs from locally sparse/boundary clusters | Tests a different coverage mechanism | Historical evidence is a separate extension, not pooled with the five-strategy curve |
| MC-dropout variance / mutual information | Rank disagreement across dropout predictions | Models epistemic uncertainty | Requires dropout probability and Monte-Carlo sample-count calibration |

The common strategy dispatcher is [`Experiment.select`](https://github.com/Purpleline-z/rl_quantum_2026summer/blob/main/code/active_learning_program/pairwise_active_learning_pipeline.py#L536). Each pseudocode block below links to the exact function that implements the selection rule.

#### Algorithm 3a: Random baseline

Implementation: [`random_sampling`](https://github.com/Purpleline-z/rl_quantum_2026summer/blob/main/code/active_learning_program/pair_acquisition_methods.py#L110).

```text
Input: candidate pair groups C, budget b, random seed s
initialize a deterministic random-number generator with s
return b distinct groups sampled uniformly from C
```

#### Algorithm 3b: Predictive uncertainty

Implementation: [`uncertainty_sampling`](https://github.com/Purpleline-z/rl_quantum_2026summer/blob/main/code/active_learning_program/pair_acquisition_methods.py#L116), which calls [`score_uncertainty`](https://github.com/Purpleline-z/rl_quantum_2026summer/blob/main/code/active_learning_program/pair_acquisition_methods.py#L75).

```text
Input: candidates C, trained reward model f, budget b
for each pair (xi, xj) in C:
    p = sigmoid(f(xi,t) - f(xj,t))
    uncertainty = mean BernoulliEntropy(p) across reward heads
return the b pairs with largest uncertainty
```

#### Algorithm 3c: Core-set coverage

Implementation: [`core_set_select`](https://github.com/Purpleline-z/rl_quantum_2026summer/blob/main/code/active_learning_program/pair_acquisition_shared_calculations.py#L52); pair embeddings are created through [`pair_vector`](https://github.com/Purpleline-z/rl_quantum_2026summer/blob/main/code/active_learning_program/pair_acquisition_shared_calculations.py#L20).

```text
Input: candidate pair embeddings C, labelled-pair embeddings L, budget b
selected = empty
repeat b times:
    choose c in C that has the greatest distance to L union selected
    add c to selected
return selected
```

#### Algorithm 3d: Cluster-quota uncertainty

Implementation: [`cluster_quota_uncertainty_sampling`](https://github.com/Purpleline-z/rl_quantum_2026summer/blob/main/code/active_learning_program/pair_acquisition_methods.py#L122).

```text
Input: candidates C with cluster IDs, reward model f, budget b
compute uncertainty for every candidate
allocate a proportional integer quota, with a minimum of one before budget is exhausted
within each cluster quota, choose the most uncertain remaining candidate
fill any unused budget with globally most uncertain remaining candidates
return selected groups
```

#### Algorithm 3e: Uncertainty plus diversity

Implementation: [`uncertainty_diversity_select`](https://github.com/Purpleline-z/rl_quantum_2026summer/blob/main/code/active_learning_program/pair_acquisition_shared_calculations.py#L29).

```text
Input: uncertainty-scored candidates C, labelled embeddings L, budget b, lambda
for each candidate c:
    u = normalized predictive uncertainty of c
    d = normalized distance from c to L and already selected pairs
    score(c) = u + lambda*d
repeat until b groups are selected:
    add remaining candidate with largest score and update diversity distances
return selected groups
```

#### Algorithm 3f: Cluster-Margin

Implementation: [`cluster_margin_pairwise_sampling`](https://github.com/Purpleline-z/rl_quantum_2026summer/blob/main/code/active_learning_program/pair_acquisition_methods.py#L145).

```text
Input: candidates C with clusters, reward model f, budget b
compute each pair's distance from probability 0.5 (its margin)
prefilter the lowest-margin candidates
round-robin across prefiltered clusters, visiting smaller prefiltered clusters first
return b selected groups
```

#### Algorithm 3g: MC-dropout probability variance

Implementation: [`score_mc_dropout`](https://github.com/Purpleline-z/rl_quantum_2026summer/blob/main/code/active_learning_program/monte_carlo_dropout_uncertainty.py#L33) and [`select_mc_dropout`](https://github.com/Purpleline-z/rl_quantum_2026summer/blob/main/code/active_learning_program/monte_carlo_dropout_uncertainty.py#L77).

```text
Input: candidates C, reward model f with dropout active, b, M stochastic passes
for m = 1,...,M:
    score every candidate with a different dropout mask
for each candidate:
    compute variance of its predicted preference probability across M passes
return b candidates with largest probability variance
```

#### Algorithm 3h: MC-dropout mutual information

Implementation: the same [`score_mc_dropout`](https://github.com/Purpleline-z/rl_quantum_2026summer/blob/main/code/active_learning_program/monte_carlo_dropout_uncertainty.py#L33) and [`select_mc_dropout`](https://github.com/Purpleline-z/rl_quantum_2026summer/blob/main/code/active_learning_program/monte_carlo_dropout_uncertainty.py#L77) functions, dispatched with metric `mc_dropout_mutual_information`.

```text
Input: candidates C, reward model f with dropout active, b, M stochastic passes
obtain M stochastic preference distributions per candidate
for each candidate:
    predictive_entropy = entropy(mean probability)
    expected_entropy = mean entropy(stochastic probabilities)
    mutual_information = predictive_entropy - expected_entropy
return b candidates with largest mutual_information
```

### 2.4 Leakage-safe training algorithms

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

The historical five-strategy curve has a useful qualitative pattern: random is strongest at budget 10 (0.413), uncertainty is slightly strongest at budget 25 (0.440), cluster-quota uncertainty is strongest at budget 50 (0.460), uncertainty--diversity is strongest at budget 75 (0.513), and uncertainty is strongest at budget 100 (0.647). These are not final generalization claims because the historical split failed the later identity audit. They do show why random must remain a baseline and why the acquisition rule should not be selected independently of budget.

### 5.2 Symmetry preprocessing is an experimental factor

The repository evaluates three input modes: `none`, `left_half_mirror`, and `symmetric_average`. The latter two encode a symmetry assumption by reconstructing an image from its left half or averaging it with its horizontal reflection. They are not treated as harmless augmentations: they can remove discriminative asymmetry as well as reduce nuisance variation.

![Historical symmetry factorial](active_learning_studies/pair_disjoint_not_image_disjoint/paper_assets/historical_symmetry_factorial.png)

*Figure 4. Historical strategy--budget--symmetry interaction, generated from the Stage 2 aggregate CSV. It is diagnostic only.*

The historical factorial shows that preprocessing changes the ranking rather than delivering a universal gain. For example, at low budget some coverage-based methods improved under symmetric averaging, while at budget 100 the no-symmetry uncertainty endpoint remained stronger than the compared mirrored endpoint. The leakage-safe rerun must retain symmetry mode as a factor and report the strategy × budget × preprocessing interaction instead of choosing one transform globally.

### 5.3 SimCLR versus ImageNet initialization

![Historical encoder-screen validation accuracy](active_learning_studies/pair_disjoint_not_image_disjoint/paper_assets/historical_encoder_screen.png)

*Figure 5. Historical validation-only encoder screen; error bars are standard deviations over three seeds.*

SimCLR is an image-only self-supervised initialization, whereas ImageNet initialization starts from supervised natural-image features. In the completed path-based validation screen, ImageNet exceeded SimCLR for random (0.556 vs. 0.389) and uncertainty (0.611 vs. 0.300). This is evidence about that particular split, encoder screen, and selected endpoint—not evidence that ImageNet is intrinsically superior for every budget or selector. The leakage-safe calibration should repeat encoder × budget × strategy comparisons with identical seeds and report paired differences.

### 5.4 Metadata fusion: a separate, currently data-limited study

Metadata fusion asks a different question from active pair selection: whether causal process-monitor variables improve image prediction beyond image features alone. It must not be pooled with the pairwise selector curves. Its minimum experiment contains three matched arms: image-only, metadata-only, and image-plus-metadata late fusion. The data contract requires image-to-record mappings, temporally causal joins, handling for missing values, and image/run-disjoint evaluation.

There is no completed performance claim for fusion in the current repository because the available bundle lacks sufficiently broad, image-resolved growth sessions. The next valid fusion result therefore begins with a data-availability audit, then evaluates all three arms with accuracy, macro-F1, per-class scores, and cross-run generalization.

### 5.5 No historical evidence for epoch 10

![Missing epoch-wise evidence](active_learning_studies/pair_disjoint_not_image_disjoint/paper_assets/missing_historical_epoch_curve.svg)

*Figure 6. Evidence status for historical epoch choices.*

The archived outputs compare endpoint settings such as 1, 2, 3, 5, and 10 epochs, but do not provide validation accuracy after every epoch for a common leakage-safe protocol. Therefore, the report makes **no claim** that ten epochs is optimal or that early stopping would stop at epoch 10. A future curve must plot mean training loss and utility-validation accuracy by epoch, optionally with seed variation, and choose the epoch before one outer-test evaluation.

### 5.6 Evidence categories

| Category | Status | Permitted interpretation |
|---|---|---|
| Historical fixed-schedule budget curves | Completed, but affected by identity audit | Diagnostic context only |
| Path-based budget-aware calibration | Completed artifact | Do not use for final test claims |
| SHA-256-safe split + epoch logging | Implemented | Required basis for new experiments |
| Leakage-safe calibration and final selector curve | Planned | Required before final performance conclusion |

## 6. Reproducible Next Experiments

The required experiment sequence is deliberately ordered so that the outer test cannot influence a decision.

1. **Identity-safe split capacity audit.** Rebuild every seed by SHA-256 identity, confirm zero outer-test overlap with pairwise rows, candidate images, unlabeled trajectories, references, utility validation, and Bad anchors, and record how many rows are excluded.
2. **Validation-only training calibration.** For every encoder and budget, use utility validation to screen learning rate $[10^{-5},3\cdot10^{-5},10^{-4},3\cdot10^{-4}]$, weight decay $[10^{-5},10^{-4},10^{-3}]$, maximum epochs $[3,10,30]$, and early-stopping patience $[3,5,8]$. Save loss and validation accuracy at every epoch.
3. **Leakage-safe strategy comparison.** Freeze the selected schedule and evaluate random, uncertainty, core-set, cluster-quota uncertainty, uncertainty--diversity, Cluster-Margin, and eligible MC-dropout rules at budgets $[10,25,50,75,100]$ across the declared seeds. Sweep uncertainty--diversity $\lambda \in [0,0.25,0.5,0.75,1]$, cluster count, dropout probability, and MC sample count on validation only.
4. **Controls and reporting.** Run both budget-specific early-stopped training and a fixed-total-optimizer-update control. Only after all choices are frozen, run outer-test evaluation and report mean, standard deviation, per-seed values, and paired differences versus random.
5. **Independent metadata-fusion study.** Obtain multiple image-resolved sessions, audit causal alignment, then run image-only, metadata-only, and fusion baselines under run-disjoint splits.

## 7. Limitations and Next Experiment

The outer ideal-image endpoint is small and may have high seed variation. The historical data-identity leakage means previously reported outer-test numbers must not be presented as final generalization estimates. The next experiment must rebuild the split by identity, train using only permitted data, select hyperparameters and stopping epochs from utility validation, freeze those choices, and then run a single final outer-test comparison across declared strategies and seeds.

## 8. Conclusion

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
