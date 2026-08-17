# Strict Image-Disjoint Active-Learning Study

This is the new four-way image-disjoint study. It is not the earlier
pair-disjoint benchmark, and the two result sets must never be pooled.

## What is runnable now

Use only `run_literature_backed_strict_image_disjoint_budget_curve.py`. The
previous runner and its `uncertainty_diversity` experiment were removed; they
are not part of this study.

The runnable script currently supports exactly these stages:

1. a CPU-only four-way image-disjoint capacity and leakage audit;
2. writing a validation learning-rate/epoch task manifest; and
3. one independently runnable validation training cell.

The final five-strategy budget curve, weight-decay comparison, aggregation,
and figures are **not implemented yet**. Do not start a large GPU sweep until
those stages are added and reviewed.

## Four independent Colab accounts

The four accounts do not share a filesystem, process, GPU, or Drive mount.
They are not GPU numbers `0`, `1`, `2`, and `3` on one machine. Each account
writes to its own Google Drive directory.

Run this first cell separately in each account. It clones the code into
temporary Colab storage and puts every result and checkpoint on Google Drive.
Use a fresh runtime; do not clone into Drive.

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/Purpleline-z/rl_quantum_2026summer.git
%cd /content/rl_quantum_2026summer/code

# Change only this account-specific label.
ACCOUNT_NAME = 'account_1'  # Use account_1, account_2, account_3, or account_4.
DRIVE_OUTPUT = f'/content/drive/MyDrive/rheed_strict_image_disjoint/{ACCOUNT_NAME}'
!mkdir -p "$DRIVE_OUTPUT"
```

The clone under `/content/` is disposable. Every command below passes
`--drive-output "$DRIVE_OUTPUT"`, so completed cell results are saved under
Drive immediately and epoch checkpoints are also written there. A Colab
timeout can remove `/content/`, but it does not remove these Drive outputs.
Nothing in this repository pushes results automatically.

## Task 0: capacity and leakage gate (one account only)

```python
!python active_learning_studies/strict_image_disjoint/run_literature_backed_strict_image_disjoint_budget_curve.py \
  audit_four_way_image_disjoint_capacity \
  --drive-output "$DRIVE_OUTPUT" \
  --device cpu
```

Expected time: **under 10 minutes on CPU**. Saved to:
`$DRIVE_OUTPUT/audit/four_way_capacity_summary.csv` and one JSON audit per
seed. It must show zero overlap and capacity for 10 initial plus 100 acquired
pair groups for every seed. If it fails, stop: do not use a GPU and do not
weaken image-disjointness.

## Task 1: write validation tasks (one account only)

After Task 0 passes:

```python
!python active_learning_studies/strict_image_disjoint/run_literature_backed_strict_image_disjoint_budget_curve.py \
  write_validation_learning_rate_epoch_task_manifest \
  --drive-output "$DRIVE_OUTPUT" \
  --device cpu
```

Expected time: **under 1 minute on CPU**. Saved to:
`$DRIVE_OUTPUT/validation_learning_rate_epoch_cells.json`. It lists 120
independent cells: 5 seeds x 2 encoders x 4 learning rates x 3 epoch counts.

## Task 2: timing pilots before scheduling the grid

Run these four cells in parallel, one on each independent account, only after
Task 0 passes. They measure the actual T4 time for the longest calibration
case.

```python
# Account 1
!python active_learning_studies/strict_image_disjoint/run_literature_backed_strict_image_disjoint_budget_curve.py run_validation_cell --drive-output "$DRIVE_OUTPUT" --device cuda --seed 42 --encoder simclr --learning-rate 0.0001 --epochs 30 --weight-decay 0.0001

# Account 2
!python active_learning_studies/strict_image_disjoint/run_literature_backed_strict_image_disjoint_budget_curve.py run_validation_cell --drive-output "$DRIVE_OUTPUT" --device cuda --seed 42 --encoder imagenet --learning-rate 0.0001 --epochs 30 --weight-decay 0.0001

# Account 3
!python active_learning_studies/strict_image_disjoint/run_literature_backed_strict_image_disjoint_budget_curve.py run_validation_cell --drive-output "$DRIVE_OUTPUT" --device cuda --seed 79 --encoder simclr --learning-rate 0.0001 --epochs 30 --weight-decay 0.0001

# Account 4
!python active_learning_studies/strict_image_disjoint/run_literature_backed_strict_image_disjoint_budget_curve.py run_validation_cell --drive-output "$DRIVE_OUTPUT" --device cuda --seed 79 --encoder imagenet --learning-rate 0.0001 --epochs 30 --weight-decay 0.0001
```

Expected time: a **pilot measurement**, not a fabricated estimate. A cell over
25 minutes must be split into 10-epoch resumable segments before a full grid.
Saved directly to `$DRIVE_OUTPUT/resumable_checkpoints/validation/` and
`$DRIVE_OUTPUT/validation_cells/`. The current implementation prints a
heartbeat; epoch-level progress will be added before any full grid is run.

## Approved selectors for the eventual curve

The planned set in `literature_backed_strict_image_disjoint_budget_curve_settings.json` is:

- `random`
- `predictive_entropy`
- `core_set_k_center`
- `mc_dropout_mutual_information` (BALD)
- `mc_dropout_probability_variance`

There is no custom diversity-combination strategy.

## Required saved outputs

Each account keeps its own study-specific Drive root:

- `audit/`: image IDs, pair IDs, counts, and overlap assertions;
- `validation_cells/`: one raw result per GPU cell;
- `resumable_checkpoints/validation/`: restart state for long cells.

Do not put these in a generic shared `results/` directory. Once all accounts
finish, combine their four Drive directories only with an explicit future
non-GPU aggregation command.
