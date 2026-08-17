# One-T4 Simulator-Augmented Classifier2 Preliminary Study

This study tests whether synthetic **training-only** images improve a real-image, strictly image-disjoint three-class evaluation. It does not modify the PhD-authored documents under `classifier2/`, and it does not compare its result directly with historical five-class Classifier2 numbers.

## What is run

The automatic queue completes 15 resumable cells in this order for seeds 42, 79, 123, 202, and 303:

1. real-only Classifier2-compatible baseline;
2. synthetic-only transfer diagnostic evaluated on real outer-test images; and
3. synthetic pretraining followed by real Classifier2-compatible fine-tuning.

Each epoch prints progress and atomically overwrites one small resumable checkpoint on Drive. A completed cell writes a final JSON then removes its checkpoint. Re-run the identical queue command after an interruption; completed JSON files are skipped.

## Before Colab

Upload this single archive to Google Drive exactly once:

```text
MyDrive/rheed_simulator_assets/classifier-sto-v6-dual-raster__p656_aligned_peak.zip
```

Do not extract it into Drive. It is about 312 MB; Colab extracts it to ephemeral `/content` instead. Drive retains only JSON/CSV/figures and a single currently active checkpoint.

## Main T4 account: setup

Run this one cell first. It is safe to re-run after a runtime reset.

```python
from google.colab import drive
drive.mount('/content/drive')

REPOSITORY_ROOT = '/content/rl_quantum_2026summer'
CODE_ROOT = f'{REPOSITORY_ROOT}/code'
DRIVE_OUTPUT = '/content/drive/MyDrive/rheed_simulator_classifier2_preliminary/account_1'
SYNTHETIC_ARCHIVE = '/content/drive/MyDrive/rheed_simulator_assets/classifier-sto-v6-dual-raster__p656_aligned_peak.zip'

!test -f "$SYNTHETIC_ARCHIVE"
!if [ ! -d "$REPOSITORY_ROOT/.git" ]; then git clone https://github.com/Purpleline-z/rl_quantum_2026summer.git "$REPOSITORY_ROOT"; fi
%cd "$REPOSITORY_ROOT"
!git pull --rebase origin main
%cd "$CODE_ROOT"
!mkdir -p "$DRIVE_OUTPUT"
!nvidia-smi
```

Expected time: 2–5 minutes, excluding Drive mounting and the initial clone.

## CPU-only preflight

Use a CPU runtime if available. Each command writes only compact files to:

```text
/content/drive/MyDrive/rheed_simulator_classifier2_preliminary/account_1/preflight_audits/
```

Run these in order. The first task streams all 2,250 image hashes and usually takes 3–10 minutes. The other three each take under 5 minutes.

```python
%cd "$CODE_ROOT"
!python active_learning_studies/simulator_augmented_classifier2_preliminary/validate_p656_synthetic_archive.py --synthetic-archive "$SYNTHETIC_ARCHIVE" --drive-output "$DRIVE_OUTPUT"
!python active_learning_studies/simulator_augmented_classifier2_preliminary/build_strict_real_image_partitions.py --drive-output "$DRIVE_OUTPUT"
!python active_learning_studies/simulator_augmented_classifier2_preliminary/audit_real_pairwise_image_capacity.py --drive-output "$DRIVE_OUTPUT"
!python active_learning_studies/simulator_augmented_classifier2_preliminary/run_simulator_study_behavior_tests.py --drive-output "$DRIVE_OUTPUT"
!python active_learning_studies/simulator_augmented_classifier2_preliminary/create_real_and_synthetic_contact_sheet.py --synthetic-archive "$SYNTHETIC_ARCHIVE" --drive-output "$DRIVE_OUTPUT"
```

The GPU queue fails closed if any required preflight audit is absent or the ZIP hash changes.

## One-T4 automatic GPU queue

```python
%cd "$CODE_ROOT"
!python active_learning_studies/simulator_augmented_classifier2_preliminary/run_resumable_one_t4_simulator_pretraining_queue.py run_automatic_queue --synthetic-archive "$SYNTHETIC_ARCHIVE" --drive-output "$DRIVE_OUTPUT" --device cuda
```

Expected total T4 time is 2.5–5 hours. The 15 independent cells are designed for approximately 5–25 minutes each. The extraction is ephemeral and occurs once per Colab runtime; no image archive or model collection is copied to Drive. Output JSON files are under `per_seed_results/`; active checkpoints are under `resumable_checkpoints/` and are deleted automatically per completed cell.

## Aggregate completed results

After all 15 result JSON files exist:

```python
%cd "$CODE_ROOT"
!python active_learning_studies/simulator_augmented_classifier2_preliminary/aggregate_paired_simulator_preliminary_results.py --drive-output "$DRIVE_OUTPUT"
```

Expected CPU time: under 2 minutes. Results are written to:

```text
/content/drive/MyDrive/rheed_simulator_classifier2_preliminary/account_1/aggregated_results/
```

## Push results without affecting implementation files

```python
%cd "$REPOSITORY_ROOT"
!git pull --rebase origin main
!mkdir -p code/active_learning_studies/simulator_augmented_classifier2_preliminary/results/account_1
!cp -R "$DRIVE_OUTPUT/preflight_audits" code/active_learning_studies/simulator_augmented_classifier2_preliminary/results/account_1/
!cp -R "$DRIVE_OUTPUT/per_seed_results" code/active_learning_studies/simulator_augmented_classifier2_preliminary/results/account_1/
!cp -R "$DRIVE_OUTPUT/aggregated_results" code/active_learning_studies/simulator_augmented_classifier2_preliminary/results/account_1/
!git add code/active_learning_studies/simulator_augmented_classifier2_preliminary/results/account_1
!git config user.name "Purpleline-z"
!git config user.email "purpleline@uchicago.edu"
!git commit -m "Add simulator preliminary account 1 results"
```

```python
import getpass
GITHUB_PAT = getpass.getpass('GitHub PAT: ')
!git remote set-url origin "https://{GITHUB_PAT}@github.com/Purpleline-z/rl_quantum_2026summer.git"
!git push origin main
```

Do not commit the ZIP, extracted images, reciprocal targets, masks, or checkpoints.
