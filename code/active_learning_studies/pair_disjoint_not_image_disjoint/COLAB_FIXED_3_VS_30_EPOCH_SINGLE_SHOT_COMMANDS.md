# Colab commands: fixed 3-versus-30 epoch single-shot comparison

This is a separate fixed-learning-rate experiment. It does not overwrite the
validation-selected Task 3 outputs. Both accounts must mount Drive separately,
clone their own checkout, and use their own Drive output directory.

The queue uses the shipped SimCLR checkpoint, `lr=1e-4`, three paired seeds,
and only Random and standard Uncertainty. It rejects any SHA-256 image identity
that occurs in pairwise training/candidate data and any ideal reference,
utility-validation, or outer-test partition.

## Account 1: setup and queue

Expected T4 time: approximately 30--50 minutes. Every epoch writes the current
checkpoint to Drive. Every completed cell writes a compact JSON and deletes its
own final checkpoint. Re-run the same queue command after an interruption.

```python
from google.colab import drive
drive.mount('/content/drive')

REPOSITORY_ROOT = '/content/rheed_fixed_epoch_account_1'
CODE_ROOT = f'{REPOSITORY_ROOT}/code'
DRIVE_OUTPUT = '/content/drive/MyDrive/rheed_fixed_epoch_single_shot/account_1'

!git clone https://github.com/Purpleline-z/rl_quantum_2026summer.git "$REPOSITORY_ROOT"
%cd {REPOSITORY_ROOT}
!git pull --rebase origin main
%cd {CODE_ROOT}
!mkdir -p "$DRIVE_OUTPUT"
!nvidia-smi
```

```bash
python active_learning_studies/pair_disjoint_not_image_disjoint/run_fixed_3_epoch_vs_30_epoch_single_shot_comparison.py \
  verify_identity_safe_input_protocol \
  --drive-output "$DRIVE_OUTPUT" \
  --device cpu
```

```bash
python active_learning_studies/pair_disjoint_not_image_disjoint/run_fixed_3_epoch_vs_30_epoch_single_shot_comparison.py \
  run_automatic_queue \
  --drive-output "$DRIVE_OUTPUT" \
  --device cuda \
  --worker-index 0 \
  --worker-count 2
```

## Account 2: setup and queue

Expected T4 time: approximately 30--50 minutes. Account 2 has a different
Drive directory and a non-overlapping deterministic shard.

```python
from google.colab import drive
drive.mount('/content/drive')

REPOSITORY_ROOT = '/content/rheed_fixed_epoch_account_2'
CODE_ROOT = f'{REPOSITORY_ROOT}/code'
DRIVE_OUTPUT = '/content/drive/MyDrive/rheed_fixed_epoch_single_shot/account_2'

!git clone https://github.com/Purpleline-z/rl_quantum_2026summer.git "$REPOSITORY_ROOT"
%cd {REPOSITORY_ROOT}
!git pull --rebase origin main
%cd {CODE_ROOT}
!mkdir -p "$DRIVE_OUTPUT"
!nvidia-smi
```

```bash
python active_learning_studies/pair_disjoint_not_image_disjoint/run_fixed_3_epoch_vs_30_epoch_single_shot_comparison.py \
  verify_identity_safe_input_protocol \
  --drive-output "$DRIVE_OUTPUT" \
  --device cpu
```

```bash
python active_learning_studies/pair_disjoint_not_image_disjoint/run_fixed_3_epoch_vs_30_epoch_single_shot_comparison.py \
  run_automatic_queue \
  --drive-output "$DRIVE_OUTPUT" \
  --device cuda \
  --worker-index 1 \
  --worker-count 2
```

## CPU-only behavior test

Run this once on any CPU runtime after cloning. Expected time: under 10 minutes.

```bash
python active_learning_studies/pair_disjoint_not_image_disjoint/run_fixed_3_epoch_vs_30_epoch_single_shot_comparison.py \
  run_behavior_tests \
  --device cpu
```

## Push compact results after a queue finishes

Run this only after its account queue reports completion. Replace `account_1`
with `account_2` on Account 2. Checkpoints are deliberately not copied or
committed.

```python
%cd {REPOSITORY_ROOT}
!git pull --rebase origin main

RESULT_DESTINATION='code/active_learning_studies/pair_disjoint_not_image_disjoint/results/simclr_three_seed_identity_safe_task3/fixed_epoch_3_and_30_single_shot_cells/account_1'
!mkdir -p "$RESULT_DESTINATION"
!rsync -a --include='*.json' --exclude='*' \
  "$DRIVE_OUTPUT/fixed_epoch_3_and_30_single_shot_cells/" \
  "$RESULT_DESTINATION/"

!git add "$RESULT_DESTINATION"
!git config user.name "Purpleline-z"
!git config user.email "purpleline@uchicago.edu"
!git commit -m "Add account 1 fixed epoch single-shot results"
```

```python
import getpass
GITHUB_PAT = getpass.getpass('GitHub PAT: ')
!git remote set-url origin https://{GITHUB_PAT}@github.com/Purpleline-z/rl_quantum_2026summer.git
!git push origin main
```

## Aggregate after both accounts have pushed

Run this in either checkout after `git pull --rebase origin main`. It is CPU
only and takes about 5--10 minutes. It refuses to aggregate fewer than all 60
unique cells.

```python
%cd {REPOSITORY_ROOT}
!git pull --rebase origin main
%cd {CODE_ROOT}
!python active_learning_studies/pair_disjoint_not_image_disjoint/run_fixed_3_epoch_vs_30_epoch_single_shot_comparison.py \
  aggregate_completed_results \
  --device cpu
```

Commit and push the generated `fixed_epoch_3_and_30_single_shot_aggregate/`
directory using the same identity and PAT steps above.
