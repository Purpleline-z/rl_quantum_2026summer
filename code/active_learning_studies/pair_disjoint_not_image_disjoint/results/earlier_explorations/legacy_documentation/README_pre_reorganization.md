# Historical Pre-Reorganization Active-Learning Guide

## How to use this document

This file records the code and protocol that existed before the current study reorganization. It is retained because it explains where early selector, metadata, mixture, and sequential-exploration ideas came from. It is not the current execution guide and its result tables must not be combined with the current pair-disjoint benchmark without checking the recorded data version, split, initial set, acquisition mode, and endpoint.

### What this historical work contributed

- It established that pairwise human labels can be treated as an acquisition resource rather than a fixed dataset.
- It introduced audits for CSV hashes, pair overlap, and candidate-label hiding.
- It explored metadata novelty, mixture surrogates, symmetry transformations, sequential acquisition, and several selectors.
- It also exposed why later studies separate pair-disjoint from image-disjoint evaluation, validation from outer test, and fixed-protocol curves from budget-aware calibration.

### What replaced it

The current implementation is under `code/active_learning_program/`. The current pair-disjoint study and its owned result folders are under `code/active_learning_studies/pair_disjoint_not_image_disjoint/`. The simulator, causal metadata, and trajectory paths each now have a separate study directory. Read their current descriptions before acting on any command or conclusion below.

The remaining sections preserve the historical technical details and raw-result interpretation that motivated those later designs.

---

# Original pre-reorganization README

## Stage-A audit protocol (v1.8 and v5.7)

This directory selects the next **pairwise human-labeling tasks** for Stage A;
it is not an MBE-control or Stage-B RL policy. Every run names its input version
explicitly and writes hashes, split manifests, overlap counts, and leakage checks
under a version-specific result directory. `v1.8` is historical; `v5.7` is the
GitHub Stage-A baseline. Never pool the two versions' result tables.

```powershell
# fast protocol check (uses the selected data version, no scientific conclusion)
python "active learning code/active_learning_pipeline.py" --dataset-version v1.8 --smoke-test

# create the preregistered 15-seed Stage-A manifest; run one bounded job at a time
python "active learning code/run_professor_validation.py" --stage epoch_sweep --dataset-version v5.7 --generate-manifest
python "active learning code/run_professor_validation.py" --stage epoch_sweep --dataset-version v5.7 --job-id <job-id>
python "active learning code/run_professor_validation.py" --stage epoch_sweep --dataset-version v5.7 --aggregate
```

Candidate records deliberately contain only a pair ID and two image paths.
The preference rows are a hidden simulation oracle and are revealed only after a
pair is selected. A `utility` strategy is trained only from the separate
`utility_validation` ideal split; the outer ideal test split is final-reporting
only. Pair partitions are disjoint by unordered pair, but images may recur in
different pairwise comparisons; this overlap is logged rather than hidden.

Run from the `code/` directory. Data defaults to the sibling `../data/` directory;
set `RHEED_DATA_ROOT` or pass `--data-root` when code and data are separated.

```powershell
python "active learning code/active_learning_pipeline.py" --smoke-test
python "active learning code/active_learning_pipeline.py" --epochs 30 --budget 50 --batch-size 5
python "active learning code/active_learning_pipeline.py" --strategies mc_dropout_mutual_information --mc-samples 20 --dropout-p 0.2
```

## What is labelled

An unordered pair of RHEED images is one candidate.  Acquiring it reveals every
human-preference CSV row for that pair; these rows are used in the
Bradley--Terry loss.  This is not the label used to judge active learning.

The oracle's utility label is a controlled counterfactual: fixed ideal-test
accuracy after training with the candidate minus accuracy after identical
training without it.  Both models start from the same SimCLR checkpoint and
share seed, data, optimizer, fixed ideal test/reference split, and synthetic
ideal-versus-other-type anchor loss. Reference anchors are never test images.
This avoids using a BT prediction as an oracle or confusing preference accuracy
with the downstream classification goal.  Results are cached in
`result/active_learning_seed*/utility_cache.json`; utility measurements are
expensive because each requires two complete retraining runs.

## Leakage controls and splits

Ideal images are split per class into fixed test images and distinct reference
images.  Test images are never anchors or pairwise training data. Pairwise rows
sharing an unordered image pair remain together.  The default is 70 initial
pairs and 100 hidden candidates, stratified greedily to cover every type.
Pairwise images can recur across different pairs, so the script reports that
overlap instead of falsely claiming it is image-disjoint. `Other` rows are
excluded. Twinned is excluded from downstream evaluation by default because its
ideal reference sample is small; use `--include-twinned` to include it.

## Evaluation and strategies

The BT model has one reward head per reconstruction type. Single-image class is
computed with cross-type Bradley--Terry win rates against the held-out reference
images, not raw reward-score argmax. Random, entropy uncertainty, and
cluster-diverse uncertainty are compared under the same split and seed. Cluster
features are mean frozen SimCLR embeddings of each pair. The optional `utility`
strategy fits a random-forest regressor only to past controlled utility labels;
until enough measurements exist it explicitly falls back to uncertainty.

## MC-dropout epistemic uncertainty

The model's reward head includes configurable dropout (`--dropout-p`, default
`0.2`). During acquisition scoring, the SimCLR encoder and all batch-normalization
layers remain in evaluation mode; only reward-head dropout is enabled. Candidate
image embeddings are built once with the existing cache, then the reward head is
evaluated `--mc-samples` times (default `20`). This avoids repeated image encoding.

Three directly comparable strategies are available:

- `mc_dropout_probability_variance`: average variance of per-head BT preference
  probabilities across stochastic passes.
- `mc_dropout_mutual_information`: average predictive entropy minus expected
  entropy across heads; this is the recommended primary epistemic measure.
- `mc_dropout_reward_variance`: average variance of raw per-head BT reward
  differences.

Every MC strategy writes all three candidate scores, selected flags, runtime,
sample count, and dropout rate to `mc_dropout_scores.csv`. The future ensemble
extension should use these exact three score definitions, replacing stochastic
dropout passes with independently trained model members, so comparisons remain
fair. No ensemble is trained in this implementation.

## Deployment-oriented optional ablations

The baseline remains unchanged: `--symmetry-mode none` and the existing image-only
strategies. All image transforms happen in memory; no BMP/PNG source image is
rewritten. Use the same mode for a complete training/selection/evaluation run:

```powershell
python "active learning code/active_learning_pipeline.py" --symmetry-mode left_half_mirror --strategies random,uncertainty --acquisition-mode single-shot
python "active learning code/active_learning_pipeline.py" --symmetry-mode symmetric_average --strategies random,uncertainty --acquisition-mode single-shot
```

`left_half_mirror` reconstructs the image from its left half; `symmetric_average`
averages each image with its horizontal reflection. Compare these to `none` with
identical seed/split/budget. The chosen mode is saved in `config.json` and used
for training, anchors, evaluation, and embedding extraction.

### Metadata-aware uncertainty

Pass `--metadata-csv` only for a CSV with one `path`, `image_path`, or `image_id`
column containing resolved image paths and numeric process columns (for example
`temperature`, `pressure`, `elapsed_time`, `exposure_ms`, or `beam_energy`). Values
are fit using the initial labelled split only. Unmatched or duplicate keys fail
clearly; labels are never read as metadata.

```powershell
python "active learning code/active_learning_pipeline.py" --metadata-csv process_metadata.csv --strategies metadata_ood,uncertainty_plus_metadata --metadata-weight 0.5
```

`metadata_ood` selects images far from the initial labelled metadata distribution;
`uncertainty_plus_metadata` combines normalized BT entropy and metadata novelty.
Coverage is written to `metadata_coverage.csv`.

### Mixture-aware uncertainty

Mixture score is a model surrogate, not a physical ground-truth mixture label. It
uses normalized reward-head evidence across reconstruction types: high entropy
and a low top-two margin indicate more mixed/ambiguous evidence. Candidate score
records include class probabilities, entropy, margin, and `mixture_score`.

```powershell
python "active learning code/active_learning_pipeline.py" --strategies mixture_only,uncertainty_plus_mixture --mixture-weight 0.5
```

Use manual expert mixture labels for validation separately; do not interpret the
surrogate itself as a measured mixture fraction.

Outputs include split manifests, logs/checkpoints, and plots under
`result/active_learning_seed*/`; nothing generated is written under `data/`.
The logs
contain test and training metrics, selected clusters, revealed pair IDs, and
per-pair measured utilities. The small dataset and retraining noise mean that
uncertainty may not beat random; that outcome is scientifically meaningful.

## Professor-guided validation protocol

The professor-validation runner separates *selection quality* from effects of
many sequential acquisition rounds. A single-shot job trains the initial model,
selects 100 pairs once, retrains once with the complete selection, and evaluates
once on the same fixed ideal test set. Its realised batch utility is the final
accuracy change for the complete batch. This is different from individual
utility: individual pair gains are conditional counterfactuals and can interact,
so their sum need not equal the realised batch gain.

First generate and run the epoch-sweep jobs. It compares random and uncertainty
over epochs 1, 2, 3, 5, and 10 for seeds 42, 79, and 123. The aggregate report
chooses a fixed epoch by uncertainty's mean test accuracy, preferring a smaller
epoch when error bars overlap. Then generate the lambda sweep, followed by the
five-strategy final comparison.

```powershell
python "active learning code/run_professor_validation.py" --stage epoch_sweep --generate-manifest
python "active learning code/run_professor_validation.py" --stage epoch_sweep --job-id 000_stage-epoch_sweep_epoch-1_seed-42_strategy-random_budget-100
python "active learning code/run_professor_validation.py" --stage epoch_sweep --aggregate
python "active learning code/run_professor_validation.py" --stage lambda_sweep --generate-manifest
python "active learning code/run_professor_validation.py" --stage lambda_sweep --job-id <job-id>
python "active learning code/run_professor_validation.py" --stage lambda_sweep --aggregate
python "active learning code/run_professor_validation.py" --stage final_comparison --generate-manifest
python "active learning code/run_professor_validation.py" --stage final_comparison --job-id <job-id>
python "active learning code/run_professor_validation.py" --stage final_comparison --aggregate
python "active learning code/diagnose_batch_interactions.py" --stage final_comparison
```

Use `--smoke` while generating a small two-seed, two-epoch, budget-four manifest
before launching the full study. Per-seed and aggregate CSVs, choice files, and
plots are stored under `result/professor_validation/`.

## Diversity and core-set strategies

`uncertainty_diversity` greedily combines normalized BT entropy uncertainty with
distance from the labeled/previously selected pairs: `uncertainty + lambda ×
diversity`. Lambda is swept over 0.25, 0.5, and 0.75 after the epoch is fixed.
The selected-pair logs retain the raw and normalized ingredients, combined score,
cluster ID, and greedy rank. `cluster_diverse` remains the cluster-quota version
of uncertainty sampling: high-uncertainty pairs are chosen within each cluster.

`core_set` is a separate geometric coverage baseline. It uses deterministic
farthest-first k-center selection on frozen SimCLR pair embeddings and does not
use BT uncertainty. Comparing it with uncertainty methods distinguishes model
uncertainty from simple representation-space coverage.

## T4 checkpoints and resumable jobs

Each manifest entry is one bounded job for one stage, epoch, lambda when needed,
seed, and strategy. It always saves lightweight `job_state.json` phase updates.
Large model/optimizer checkpoints are enabled only when
`--max-runtime-minutes` is greater than 120; they overwrite one file per phase
rather than accumulating epoch files. A heartbeat reports the active phase at
least once per minute.

Jobs default to a 50-minute limit, so they intentionally do not save large
checkpoints. If one pauses, rerun it from the beginning. `--resume` is available
only for jobs configured above two hours:

```powershell
python "active learning code/run_professor_validation.py" --stage epoch_sweep --job-id <job-id> --max-runtime-minutes 180 --checkpoint-minutes 1
python "active learning code/run_professor_validation.py" --stage epoch_sweep --job-id <job-id> --max-runtime-minutes 180 --resume
```

The interaction diagnostic uses available individual-utility records only. It
reports interaction gap, redundancy, cluster concentration, embedding similarity,
per-class deltas, and train/test change; it explicitly flags unavailable utility
data and the noise caused by the small fixed ideal test set.

## Professor revision: auditable budget curves

The revised controlled study writes only to `result/professor_revision/`. It
keeps each seed's 50-pair initial set and 120-pair candidate pool fixed across
budgets and methods, then retrains once after a single acquisition. The primary
comparison is random versus uncertainty at budgets 10, 25, 50, 75, and 100 over
seeds 42, 79, 123, 202, and 303. Epoch 3 and learning rate 1e-4 are fixed
configuration choices, not result axes.

```powershell
python "active learning code/run_professor_revision.py" generate-manifest --smoke
python "active learning code/run_professor_revision.py" run-job --manifest result/professor_revision/smoke_manifest.json --job-id seed-42_budget-2_strategy-random_symmetry-none
python "active learning code/run_professor_revision.py" aggregate
python "active learning code/run_professor_revision.py" generate-manifest
python "active learning code/run_professor_revision.py" generate-manifest --symmetry
python "active learning code/run_professor_revision.py" generate-manifest --exploratory
```

Each job stores immutable local split CSVs (`pretrain_pairs.csv`,
`candidate_pairs.csv`, `selected_pairs.csv`, `ideal_test.csv`), source/checkpoint
hashes, overlap audit, configuration, selected ordering, and result metrics.
`technical_report.md` is generated from these artifacts by `aggregate`, never
manually maintained. The curve plot includes faint per-seed trajectories and
mean plus sample standard deviation; the previous 15-seed budget-100 endpoint
is kept separate.

`cluster_quota_uncertainty` is the old custom cluster-diverse quota heuristic.
`cluster_margin_pairwise` is a separate exploratory method: it prefilters by
lowest BT margin, then takes low-margin pairs round-robin from the smallest
nonempty K-means clusters. It is deliberately not presented as an established
Cluster-Margin result in this low-budget setting.

Feature-space NMF is fitted only to allowed ideal-reference embeddings from the
four evaluated classes; ideal-test images are never supplied. Its entropy and
off-basis residual are diagnostics, not physical mixture fractions. The
`mixture_only` and `uncertainty_plus_mixture` jobs stop unless the reference
component audit passes. For metadata fusion, populate
`data/metadata/process_metadata_template.csv` and pass its path with
`--metadata-csv`; the image-plus-metadata network otherwise remains inactive.
Tentative temporal rules are versioned in `data/temporal_constraints.json` and
are not applied to the current pool-based classifier.

## Colab + Google Drive layout

`data/` is immutable experiment input: original images, labels, the metadata
template, and tentative temporal-constraint configuration. Re-zip it only when
you intentionally add or correct dataset/metadata assets. Everything else,
including model checkpoints, code, documents, tests, manifests, and results,
lives in `code/` and is the folder to update routinely.

The project root contains only these folders:

```text
project_root/
  data/
  code/
```

Create an update archive in PowerShell from `project_root`:

```powershell
Compress-Archive -Path .\code -DestinationPath .\code.zip -Force
```

In Colab, mount Drive, unpack the update archive, and point code at the stable
Drive dataset. The `--data-root` form is equivalent to setting the environment
variable and is useful for explicit notebook commands.

```python
from google.colab import drive
drive.mount('/content/drive')
!unzip -q -o "/content/drive/MyDrive/rheed_project/code.zip" -d /content
%env RHEED_DATA_ROOT=/content/drive/MyDrive/rheed_project/data
!python "/content/code/active learning code/active_learning_pipeline.py" --smoke-test
# Or: !python "/content/code/active learning code/active_learning_pipeline.py" --data-root "/content/drive/MyDrive/rheed_project/data" --smoke-test
```

All runners honor `RHEED_DATA_ROOT`; their generated files always remain under
the unpacked `code/result/` tree.

## Professor follow-up: core-set, PCA, and redundancy

The follow-up analysis keeps the original professor-validation results read-only
and writes only to `result/professor_followup_analysis/`. It tests whether
core-set selects embedding outliers or sparse clusters that are not relevant to
reconstruction classification. This does not automatically conflict with PCA:
PCA is a global two-dimensional projection, whereas core-set uses local
full-dimensional pair-embedding distance and has no downstream-utility signal.

The redundancy analysis reports absolute post-acquisition accuracy separately
from within-seed batch utility. If diversity methods lower image reuse or
embedding similarity but do not improve either outcome, pair redundancy is not
by itself a sufficient explanation for utility/accuracy mismatch.

## Five-seed follow-up experiment

The follow-up aggregates existing seeds `42`, `79`, and `123`, then creates only
the missing final-comparison jobs for seeds `202` and `303`. It compares random,
uncertainty, cluster-diverse, uncertainty-diversity, and core-set using matched
seed-level differences versus random. Results use mean plus sample standard
deviation; five seeds show direction and uncertainty but do not justify strong
significance claims.

## Running the follow-up analysis on T4

```powershell
python "active learning code/run_followup_analysis.py" generate-manifest
python "active learning code/run_followup_analysis.py" run-job --job-id followup_seed-202_strategy-random --max-runtime-minutes 50 --checkpoint-minutes 1
python "active learning code/run_followup_analysis.py" run-job --job-id followup_seed-202_strategy-random --resume
python "active learning code/run_followup_analysis.py" aggregate
python "active learning code/run_followup_analysis.py" analyze-coreset
python "active learning code/run_followup_analysis.py" analyze-redundancy
```

Every follow-up job reuses the existing resumable training implementation: it
checkpoints model and optimizer state every epoch, writes phase state atomically,
emits at least one heartbeat per minute, and pauses safely near the runtime
limit. Read the resulting conclusions in
`result/professor_followup_analysis/reports/followup_report.md`.

## Fifteen-seed follow-up experiment

The follow-up now uses the fixed seed set `42, 79, 123, 202, 303, 404, 505,
606, 707, 808, 909, 1010, 1111, 1212, 1313`. Expanding from five to fifteen
seeds reduces uncertainty in strategy-level means and paired comparisons, while
still reporting sample standard deviation and avoiding significance claims based
only on this finite sample. Completed results are reused; the manifest schedules
only missing seed/strategy combinations. Both absolute post-acquisition accuracy
and paired within-seed batch utility remain required outputs.

```powershell
python "active learning code/run_followup_analysis.py" generate-manifest
python "active learning code/run_followup_analysis.py" run-job --job-id followup_seed-404_strategy-random --max-runtime-minutes 50 --checkpoint-minutes 1
python "active learning code/run_followup_analysis.py" run-job --job-id followup_seed-404_strategy-random --resume
python "active learning code/run_followup_analysis.py" aggregate
python "active learning code/run_followup_analysis.py" analyze-coreset
python "active learning code/run_followup_analysis.py" analyze-redundancy
```

The 15-seed manifest is stored at
`result/professor_followup_analysis/manifests/missing_jobs_15_seed.json`; its
aggregate tables and plots are placed in `fifteen_seed_extension/`.

## PCA and t-SNE representation-baseline analysis

`explore_pca_tsne.py` explores representation structure without training a
classifier. It uses all five labelled `STO_ideal*` folders for reconstruction
colours and metrics, while displaying all trajectory images only as unlabeled
gray context. It compares raw 224-by-224 grayscale pixels, a frozen off-domain
ImageNet ResNet-18, and the frozen domain-specific RHEED SimCLR encoder. No
reconstruction or pairwise label is read during feature extraction.

Raw-pixel non-separation is a testable expectation rather than a prewritten
conclusion. PCA/t-SNE figures are exploratory; the report also includes
full-representation, ideal-only k-NN and separation metrics because t-SNE can
visually exaggerate apparent clusters. Results, cached embeddings, coordinates,
figures, and the provenance report are written under
`result/pca_tsne_dataset_exploration/` without touching previous results.

```powershell
python "active learning code/explore_pca_tsne.py" --all
python "active learning code/explore_pca_tsne.py" --representation raw_pixels
python "active learning code/explore_pca_tsne.py" --reuse-cache --all
```

If official ImageNet weights cannot be loaded or downloaded, that baseline is
explicitly marked skipped; the script never substitutes random ResNet weights.
