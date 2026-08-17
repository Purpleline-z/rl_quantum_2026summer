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

## Task 2: lock calibration and generate the complete final curve

After all four Task 1 queues finish, run this once on any account. It selects
one LR/epoch protocol separately for SimCLR and ImageNet using validation only,
then writes the 250-cell final manifest: 5 seeds x 2 encoders x 5 approved
strategies x 5 budgets.

```python
!python active_learning_studies/pair_disjoint_not_image_disjoint/run_pair_disjoint_segmented_calibration.py lock_calibration_and_write_final_manifest --drive-output "$DRIVE_OUTPUT" --device cuda
```

The four accounts keep separate Drive backups. After Task 1, copy only each
account's small `calibration_results/*.json` files into the repository and
push them to GitHub one account at a time. A later merge command reads these
Git-tracked raw results; do not share Drive folders or push checkpoints.

## Task 3: complete final strategy/budget curve

After Task 2, run one queue per account using its own `DRIVE_OUTPUT`:

```python
!python active_learning_studies/pair_disjoint_not_image_disjoint/run_pair_disjoint_segmented_calibration.py run_account_final_queue --drive-output "$DRIVE_OUTPUT" --device cuda --account-index 0
```

Use account indices 1, 2, and 3 on the other accounts. Each account receives
about 63 final cells. Each final cell retrains from scratch and saves its own
JSON under `$DRIVE_OUTPUT/final_budget_curve_cells/` before the next begins.
