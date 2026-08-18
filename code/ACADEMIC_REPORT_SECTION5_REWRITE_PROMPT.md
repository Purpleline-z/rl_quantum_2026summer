# Prompt: Rewrite Section 5 as an Evidence-Based Results and Interpretation Section

You are revising the English GitHub Markdown paper `code/ACADEMIC_REPORT_DRAFT.md` in the repository `Purpleline-z/rl_quantum_2026summer`. First fetch the latest `main` branch and treat its version of the report as the source of truth. Preserve its valid method, split, and figure content. Rewrite **Section 5 and its figure captions**, and add only the diagnostics needed to support the revised text.

## Goal and audience

Write for a reviewer who understands scientific research but may not have a machine-learning background. The section must make each result useful: a reader should be able to answer, for every figure/table, **what changed, what that means about the data/model, what practical decision follows, and what experiment tests the remaining alternative explanation**.

Use concise academic English. Define a technical term in plain language the first time it appears. For example, explain an embedding as “a numerical location assigned to an image so visually similar images tend to lie near each other.”

Do **not** fill the paper with generic caveats such as “not definitive,” “not rigorous,” “cannot be promised,” “should be interpreted cautiously,” or “does not establish a conclusion.” Use an evidence boundary only when it names a concrete consequence, such as: “the historical split placed identical image bytes in both the trajectory pool and the held-out ideal set; rerun this comparison after SHA-256 exclusion before using its numeric ranking to choose a final acquisition rule.” Place the complete split qualification once in a short **Evidence Status** paragraph/table; do not repeat it after every result.

## Evidence and source artifacts

Use only saved repository artifacts and link each numerical claim to its source.

- Five-strategy budget curve: `code/active_learning_studies/pair_disjoint_not_image_disjoint/results/selection_benchmark/stage1_selector_curves_none/aggregate/strategy_budget_summary.csv`
- Symmetry factorial: `code/active_learning_studies/pair_disjoint_not_image_disjoint/results/selection_benchmark/stage2_symmetry_factorial/aggregate/strategy_budget_summary.csv`
- Encoder screen: `code/active_learning_studies/pair_disjoint_not_image_disjoint/results/protocol_diagnostics/encoder_initialization_screen/aggregate/encoder_utility_validation_summary.csv`
- Representation coordinates: `code/active_learning_studies/image_representation_analysis/results/representation_exploration/coordinates/`
- Representation manifest and metrics: `code/active_learning_studies/image_representation_analysis/results/representation_exploration/manifest.csv` and `metrics/representation_metrics.csv`
- Existing figure assets: `code/active_learning_studies/image_representation_analysis/results/representation_exploration/figures/`

Never infer a reconstruction label for an unlabeled trajectory image. Never use an outer-test result to select an encoder, preprocessing mode, acquisition strategy, epoch, or hyperparameter.

## Required diagnostic artifacts before writing PCA/t-SNE interpretation

Create a reproducible analysis script under `code/active_learning_studies/image_representation_analysis/` and save outputs in a new, clearly named `results/.../section5_diagnostics/` directory. Analyze both frozen SimCLR and frozen ImageNet coordinate files. Use the full-space feature metrics if feature vectors are available; otherwise explicitly label coordinate-space distance analyses as two-dimensional diagnostics.

Produce and save:

1. `per_class_knn_metrics.csv`: leave-one-out or stratified cross-validated 5-NN recall/precision per labelled ideal reconstruction type, support count, and macro average.
2. `knn_confusion_matrix.csv` and a legible heatmap: rows are true ideal classes and columns are predicted classes; identify which pairs of types are most often confused.
3. `class_separation_metrics.csv`: per class, within-class median distance, nearest-other-class median distance, their ratio, and a clearly defined separation score. State the metric formula in the file header or companion README.
4. `trajectory_neighborhood_coverage.csv`: for each embedding, measure each unlabeled trajectory image’s nearest labelled-ideal distance; choose the “near” threshold from labelled ideal leave-one-out nearest-neighbour distances, not arbitrarily. Report the near/far fraction and save the threshold.
5. One figure per encoder combining the labelled-class separation view with trajectory-neighborhood coverage, or clearly annotated existing plots plus a new coverage figure.

Use deterministic seeds and record input paths, row counts, coordinate columns, distance metric, k, and thresholds in a JSON manifest. Include a direct link from every report claim to the exact generated CSV/figure.

## Required structure for rewritten Section 5

Use the following pattern in every subsection:

1. **Question.** State the precise research question.
2. **Finding.** Report the comparison, sample/seed count, and exact numerical result.
3. **Meaning.** Explain what feature of the candidate pool, learned representation, or optimization process plausibly produced this pattern. Make clear whether this is a measured fact or an interpretation.
4. **Decision.** State the concrete experiment or protocol decision that follows.

### 5.1 Budget-dependent active selection

Explain the historical five-strategy pattern with the actual winning arms: random at budget 10, uncertainty at budget 25, cluster-quota uncertainty at budget 50, uncertainty-diversity at budget 75, and uncertainty at budget 100. Do not merely list the winners. Explain in accessible language why scarce labels may favor a stable baseline, why coverage can matter at intermediate budgets, and why uncertainty can become more useful when a larger acquisition budget can include several decision-boundary regions.

Do not claim those mechanisms are proven. Turn each mechanism into a discriminating experiment: the identity-safe rerun must hold seeds and selection budget fixed, choose training schedule on utility validation, compare paired per-seed differences against random, and include a fixed-total-optimizer-update control.

### 5.2 Symmetry preprocessing

Explain that `left_half_mirror` and `symmetric_average` change the input representation rather than merely adding more data. Add a compact strategy × budget × preprocessing table that reports the best and worst relevant comparisons with mean and standard deviation.

For every cited improvement or decline, say what it means operationally: a gain after averaging is consistent with nuisance asymmetry being reduced for that condition; a loss is consistent with discriminative asymmetric structure being removed. Do not invent a physical explanation. The next experiment must retain preprocessing as a factorial variable and use the identity-safe protocol.

### 5.3 SimCLR versus ImageNet

Explain that an encoder determines the coordinate system used by uncertainty, pair similarity, clustering, and core-set distance. Translate the existing validation values for random and uncertainty into that idea: the observed difference says one initialization produced a more useful starting geometry under this protocol, not that one pretraining method is universally better.

Specify the discriminating follow-up: same identity-safe seeds, identical optimizer and stopping rules, encoder × budget × strategy factorial, paired differences, and per-class outcomes.

### 5.4 PCA/t-SNE representation diagnostics

Start with a two-sentence plain-language explanation:

- PCA is a two-axis summary of the directions along which embeddings vary most; report the retained-variance fraction so readers know how much of the original geometry is visible.
- t-SNE preserves nearby neighborhoods better than global distances; it is useful for local overlap/outliers, but not for reading literal gap sizes.

Then interpret the newly computed diagnostics, not visual impressions alone. Explicitly name:

- reconstruction types with the highest and lowest per-class 5-NN recall;
- the class pairs with the largest confusion counts;
- types with compact/isolated neighborhoods versus broad/overlapping or fragmented neighborhoods, supported by the separation metrics;
- the near/far fraction of unlabeled trajectory images and what this says about coverage of the labelled-ideal feature space.

For non-ML readers, explain the implication of each result. Example: “High 5-NN recall means that when an ideal image is surrounded by its five closest images in this feature space, those neighbours usually have the same reconstruction label; this makes the feature space suitable for local similarity and coverage-based acquisition.” Conversely, explain that frequent confusion means appearance-only frozen features do not reliably distinguish those types, so the next experiment should acquire boundary pairs involving that class pair or test additional permitted features.

Do not describe PCA/t-SNE as a classifier result. Use them to motivate a concrete next experiment: targeted boundary-pair acquisition, encoder comparison, a preprocessing ablation, or an out-of-distribution trajectory audit—whichever follows from the measured class/coverage pattern.

### 5.5 Metadata fusion

State the precise question: whether temporally causal process-monitor variables add prediction information beyond image features. Specify the three controls: image-only, metadata-only, and image-plus-metadata fusion. State the missing requirement—multiple image-resolved sessions with causal sensor-to-image mapping—and the resulting next experiment: run-disjoint evaluation with accuracy, macro-F1, per-class results, and missing-data handling reported.

### 5.6 Training curves and stopping

Explain what an epoch-wise training-loss and utility-validation curve would decide: whether additional optimization is still improving generalization on the validation split, and where to stop before spending the outer test. Do not call an epoch value optimal unless the corresponding curve exists. State the planned evidence: per-epoch mean and seed variation, patience sweep, validation-selected checkpoint, then a single outer-test evaluation after settings are frozen.

## Captions, claim audit, and final checks

Rewrite every Section 5 caption so it answers:

1. What comparison is shown?
2. What is the one takeaway the reader should observe?
3. Which saved artifact generated it?

Add a final `### 5.x Claim Audit` table with columns: `Claim`, `Measured evidence`, `Interpretation`, `Concrete next test`. Do not include a claim unless all four cells are specific.

Before finishing, verify:

- Every number in Section 5 has a linked source CSV/figure.
- Every “separated,” “overlapping,” “compact,” or “fragmented” statement is supported by the generated per-class diagnostics.
- Every proposed mechanism has a distinct test that could support or refute it.
- No empty disclaimer remains.
- GitHub math uses `$...$` and `$$...$$`, never `\(...\)` or `\[...\]`.
- All Markdown links and image links resolve within the repository.

Return a concise summary listing changed files, generated diagnostics, and the evidence-backed decisions now available to a reviewer.
