# Pair-Disjoint Active-Learning Evidence and Workflow

## What this folder measures

This study selects additional **unordered image-pair groups** for human preference labeling. A selected group exposes its existing hidden preference rows; those rows train a five-output Bradley--Terry image reward model. The reported endpoint is reconstruction accuracy on held-out ideal images after retraining.

The study is pair-disjoint: a pair cannot be in both the initial labeled set and candidate pool. It is not image-disjoint: an image can appear in two different pair groups. This is intentional historical protocol compatibility. A later SHA-256 audit also found that path-disjoint ideal partitions contain byte-identical copies of some pairwise and ideal images; that identity-level leakage is an error to repair before Task 3c, not a feature of the historical protocol. Read [study_description.md](study_description.md) before comparing its outputs with image-disjoint studies.

## Evidence map

| Result family | Question answered | What the result suggests |
|---|---|---|
| `selection_benchmark/stage1_selector_curves_none/` | Which selector helps under the unmodified-image, fixed-epoch protocol? | The leading selector changes with budget; uncertainty is strongest at 100, while project coverage/diversity rules lead at some intermediate budgets. |
| `selection_benchmark/stage2_symmetry_factorial/` | Does horizontal-symmetry preprocessing alter the selection result? | Yes. Symmetry choice is an experimental factor, not a universal preprocessing default. |
| `protocol_diagnostics/` | How did LR and encoder screens behave in their recorded protocols? | ImageNet and `3e-4` screened better in selected fixed settings; budget-aware validation is needed before using one schedule everywhere. |
| `selected_summaries/lambda_sweep/` | Which diversity weight worked at the tested endpoint? | The recorded lambda `0.5` was the best of the tested weights; it remains a project-specific design choice. |
| `budget_aware_validation_calibration_account_*/` and `budget_aware_protocol/` | Which LR/epoch setting is appropriate for each budget and encoder? | Complete path-based calibration: ImageNet is higher at four budgets, SimCLR at budget 50. It must be rerun after identity-level split repair. |

## Task sequence

### Task 1: initial calibration diagnostic

The initial grid screened encoder initialization, learning rate, and epoch count on utility-validation images. Its purpose was to identify whether the historical three-epoch schedule and default initialization were plausible starting points. It is a diagnostic, not the final answer to how every acquisition budget should be trained.

### Task 2: initial calibration aggregation

Task 2 reads compact per-cell JSON records, checks that expected specifications are present once, and writes a validation-only summary. It does not start training and does not use outer-test accuracy to choose a protocol.

### Task 3a: budget-aware validation calibration

Task 3a is the corrective calibration. It spans five seeds, two encoders, five budgets, four learning rates, and three epoch counts: 600 validation-only cells. Each cell uses the initial ten pair groups plus a deterministic random-reference acquisition at the specified budget. It chooses no candidate preference labels while building that reference set.

All 600 cells are complete and Task 3b has written `results/budget_aware_protocol/`. Within the shared Task 3a protocol, ImageNet has the higher selected validation mean at budgets 10, 25, 75, and 100; SimCLR has the higher selected validation mean at budget 50. This is a fair encoder comparison within this pipeline, but it uses a different downstream protocol from historical Classifier2 experiments.

### Task 3b: budget-aware protocol aggregation

The aggregation action selected a learning rate and epoch count for each `(encoder, budget)` from utility validation only and wrote `results/budget_aware_protocol/`. The result is retained because it explains the observed encoder difference, but it is not a final selection protocol: a SHA-256 audit found approximately one to two duplicated image identities per seed on average (one to three across seeds) between pairwise data and the ideal outer test, plus duplicate identities across ideal reference/validation/test partitions.

### Task 3c: final paired budget curves

Task 3c must not start from the current Task 3b table. First rebuild all ideal and pairwise partitions around SHA-256 image identity, remove or jointly partition duplicate copies, and confirm capacity for ten initial plus 100 candidate pair groups. Then rerun validation calibration. Only the repaired protocol may be used to compare approved strategies at budgets 10, 25, 50, 75, and 100. Every final cell will start a fresh acquisition model and fresh final model; the standard and fixed-update controls will then separate “selected better labels” from “ran more updates because the acquired set was larger.”

## Persistence and result handling

Each queue writes an epoch checkpoint to its supplied Drive output and writes a compact final JSON when a cell completes. Completed JSON is the durable record; the cell checkpoint is deleted only after that JSON is written. A re-run of the identical queue command skips completed JSONs. Different Colab accounts should use different Drive folders and push their completed JSONs into their matching account folder one at a time after pulling `main`.

Only compact JSON/CSV/figure outputs belong in this repository. Do not commit Drive checkpoints, extracted image data, Colab clones, or raw laboratory data. The aggregate scripts read Git-tracked result JSONs so a later recovery runtime can continue without sharing a Drive mount.
