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
!git clone https://github.com/Purpleline-z/rl_quantum_2026summer.git
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

## Task 1: bounded calibration segments

Every GPU segment contains at most 10 epochs. It saves an epoch checkpoint to
`$DRIVE_OUTPUT/resumable_checkpoints/pair_disjoint_calibration/` and prints
epoch progress. A 3-epoch task has one segment, a 10-epoch task has one, and a
30-epoch task has three segments. Start with these four timing/calibration
tasks after Task 0 passes:

```python
# Account 1: expected 10–30 minutes
!python active_learning_studies/pair_disjoint_not_image_disjoint/run_pair_disjoint_segmented_calibration.py run_calibration_segment --drive-output "$DRIVE_OUTPUT" --device cuda --seed 42 --encoder simclr --learning-rate 0.0001 --weight-decay 0.0001 --total-epochs 10 --segment-number 1

# Account 2: expected 10–30 minutes
!python active_learning_studies/pair_disjoint_not_image_disjoint/run_pair_disjoint_segmented_calibration.py run_calibration_segment --drive-output "$DRIVE_OUTPUT" --device cuda --seed 42 --encoder imagenet --learning-rate 0.0001 --weight-decay 0.0001 --total-epochs 10 --segment-number 1

# Account 3: expected 10–30 minutes
!python active_learning_studies/pair_disjoint_not_image_disjoint/run_pair_disjoint_segmented_calibration.py run_calibration_segment --drive-output "$DRIVE_OUTPUT" --device cuda --seed 79 --encoder simclr --learning-rate 0.0001 --weight-decay 0.0001 --total-epochs 10 --segment-number 1

# Account 4: expected 10–30 minutes
!python active_learning_studies/pair_disjoint_not_image_disjoint/run_pair_disjoint_segmented_calibration.py run_calibration_segment --drive-output "$DRIVE_OUTPUT" --device cuda --seed 79 --encoder imagenet --learning-rate 0.0001 --weight-decay 0.0001 --total-epochs 10 --segment-number 1
```

For a 30-epoch task, run the identical command with `--segment-number 1`,
then `2`, then `3`. No segment is longer than 10 epochs. The final segment
writes the result JSON to `$DRIVE_OUTPUT/calibration_results/`.
