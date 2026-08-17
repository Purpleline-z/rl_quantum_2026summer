# Pair-Disjoint Active-Learning Study

This is the active-learning main study. Unordered pair-group IDs are disjoint
between initial labelled training and the candidate pool. Image IDs may repeat
across groups; this matches Classifier2's pair-disjoint, not image-disjoint,
design. No custom selector is included.

## Persistent Colab setup

Run this on every independent Colab account. The repository clone is
disposable; results and every epoch checkpoint are written directly to Drive.

```python
from google.colab import drive
drive.mount('/content/drive')
!git -c http.version=HTTP/1.1 clone --depth 1 https://github.com/Purpleline-z/rl_quantum_2026summer.git
%cd /content/rl_quantum_2026summer/code
ACCOUNT_NAME = 'account_1'  # Change for each account.
DRIVE_OUTPUT = f'/content/drive/MyDrive/rheed_pair_disjoint/{ACCOUNT_NAME}'
!mkdir -p "$DRIVE_OUTPUT"
```

## Task 0: capacity audit

Run once on any T4 runtime. It does no model training and normally completes
in under 10 minutes.

```python
!python active_learning_studies/pair_disjoint_not_image_disjoint/run_pair_disjoint_segmented_calibration.py audit_pair_disjoint_capacity --drive-output "$DRIVE_OUTPUT" --device cuda
```

The result is `$DRIVE_OUTPUT/pair_disjoint_audits/capacity_summary.csv`. It
must show zero exact pair overlap and at least 10 initial plus 100 candidate
pair groups for every seed. Image overlap is reported but permitted.

## Task 1: complete calibration grid

The full grid is 120 cells: 5 seeds x 2 encoders x 4 learning rates x 3 epoch
counts. The runner gives each independent account 30 cells. Every 30-epoch
cell is internally split into three 10-epoch checkpoints; every completed cell
writes its final JSON to Drive before the queue advances.

```python
!python active_learning_studies/pair_disjoint_not_image_disjoint/run_pair_disjoint_segmented_calibration.py run_account_calibration_queue --drive-output "$DRIVE_OUTPUT" --device cuda --account-index 0
```

Use the same command on Accounts 2, 3, and 4, replacing `--account-index 0`
with `1`, `2`, and `3` respectively. The queue prints completed/total progress
and skips completed cells after interruption. Its actual elapsed time is
measured from this run; no unmeasured 10–30 minute estimate is claimed.

## Task 2: summarize the initial-10-pair diagnostic

After all four Task 1 queues finish and their small result JSON files have been
pushed to GitHub, run this once on any account after pulling `main`. It
summarizes LR/epoch sensitivity at the initial ten labelled pair groups. It
does **not** lock a single LR/epoch protocol for every acquisition budget.

```python
!python active_learning_studies/pair_disjoint_not_image_disjoint/run_pair_disjoint_segmented_calibration.py aggregate_initial_ten_pair_calibration_diagnostic --drive-output "$DRIVE_OUTPUT" --device cuda
```

The four accounts keep separate Drive backups. After Task 1, copy only each
account's small `calibration_results/*.json` files into the repository and
push them to GitHub one account at a time. This diagnostic reads those
Git-tracked raw results; do not share Drive folders or push checkpoints.

## Task 3a: budget-aware validation calibration

This replacement calibration has 600 cells: five seeds, two encoders, five
acquisition budgets, four learning rates, and three epoch counts. For each
budget it trains on the initial ten pair groups plus a deterministic random
reference acquisition of that budget. It evaluates validation images only;
it never evaluates outer test. Each account owns 150 cells. To keep a Colab
session bounded, one invocation runs at most five cells and then stops safely.
Re-run the identical command until it reports zero remaining cells.

```python
!python active_learning_studies/pair_disjoint_not_image_disjoint/run_budget_aware_pair_disjoint_active_learning.py run_bounded_validation_calibration_queue --drive-output "$DRIVE_OUTPUT" --device cuda --account-index 0 --maximum-cells 5
```

Use account indices 1, 2, and 3 on the other accounts. Every epoch checkpoint
and every complete result is saved directly under `$DRIVE_OUTPUT`. After each
account finishes, copy only `budget_aware_validation_calibration/*.json` to
`results/budget_aware_validation_calibration_account_N/` and push it to GitHub.
Do not copy Drive checkpoints.

## Task 3b: aggregate the budget-aware protocol

After all 600 JSON files are pushed and every account has pulled `main`, run
this once. It writes Git-trackable validation summaries and one selected
LR/epoch setting for each `(encoder, acquisition budget)`. It refuses to run
if any expected cell is missing or duplicated.

```python
!python active_learning_studies/pair_disjoint_not_image_disjoint/run_budget_aware_pair_disjoint_active_learning.py aggregate_budget_aware_validation_calibration
```

Commit and push `results/budget_aware_protocol/`, then pull `main` on all four
accounts.

## Task 3c: final five-strategy budget curves

The final experiment has 500 cells: five seeds, two encoders, five approved
strategies, five budgets, and two training controls. At each budget, all
strategies share that budget's validation-selected LR/epoch setting. The
second control fixes both the acquisition and final-model training at 30
optimizer updates, preventing larger training sets from receiving more updates
merely because they have more batches per epoch. Each final cell trains a fresh
acquisition model and a fresh final model and then evaluates outer test.

```python
!python active_learning_studies/pair_disjoint_not_image_disjoint/run_budget_aware_pair_disjoint_active_learning.py run_bounded_final_strategy_queue --drive-output "$DRIVE_OUTPUT" --device cuda --account-index 0 --maximum-cells 5
```

Use account indices 1, 2, and 3 on the other accounts and rerun the identical
command until it reports zero remaining cells. Results save under
`$DRIVE_OUTPUT/budget_aware_final_strategy_cells/`; copy only those JSON files
to a corresponding `results/` directory before committing them.
