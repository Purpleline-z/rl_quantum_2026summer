# Colab commands for the static peak-aware RHEED study

Every account mounts its own Google Drive and writes to its own `MyDrive` path. GitHub is cloned into Colab's temporary disk; Drive holds the durable JSON records, checkpoints, weights, and results.

## Initial cells on every runtime (about 3--8 minutes)

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
%%bash
set -euo pipefail
REPOSITORY_DIRECTORY="/content/rl_quantum_2026summer"
if [ ! -d "${REPOSITORY_DIRECTORY}/.git" ]; then
  git clone https://github.com/Purpleline-z/rl_quantum_2026summer.git "${REPOSITORY_DIRECTORY}"
fi
cd "${REPOSITORY_DIRECTORY}"
git pull --ff-only origin main
python -m pip install --quiet --upgrade pip
python -m pip install --quiet pandas pillow scikit-learn tqdm
```

All following cells use `RHEED_DATA_ROOT=/content/rl_quantum_2026summer/data` and persist results in `/content/drive/MyDrive/rl_quantum_2026summer_results/paper_replicate`. Re-run an interrupted cell unchanged: it reads its completed JSON or the last epoch checkpoint.

## CPU task 1: create the sealed split (5 minutes)

```bash
python -m paper_replicate.run_resumable_paper_replicate_task_queue --task-name verify_input_data_and_create_image_disjoint_split --data-root /content/rl_quantum_2026summer/data --drive-results-root /content/drive/MyDrive/rl_quantum_2026summer_results/paper_replicate --device cpu --resume
```

## GPU account A: architecture comparison (50--90 minutes total)

```bash
python -m paper_replicate.run_resumable_paper_replicate_task_queue --task-name compare_image_encoder_with_peak_aware_encoder --data-root /content/rl_quantum_2026summer/data --drive-results-root /content/drive/MyDrive/rl_quantum_2026summer_results/paper_replicate --seeds 42,79 --model-variants image_encoder_only,image_encoder_plus_peak_features --device cuda --checkpoint-heartbeat-minutes 30 --resume
```

## GPU account B: independent repeat (30--60 minutes)

```bash
python -m paper_replicate.run_resumable_paper_replicate_task_queue --task-name independently_repeat_peak_aware_model_comparison --data-root /content/rl_quantum_2026summer/data --drive-results-root /content/drive/MyDrive/rl_quantum_2026summer_results/paper_replicate --seeds 123,202,303 --model-variants image_encoder_only,image_encoder_plus_peak_features --device cuda --checkpoint-heartbeat-minutes 30 --resume
```

## GPU account B: sealed test and active-learning export (10--20 minutes)

Run after account A has created its comparison completion JSON.

```bash
python -m paper_replicate.run_resumable_paper_replicate_task_queue --task-name train_selected_model_and_evaluate_sealed_image_disjoint_test_set --data-root /content/rl_quantum_2026summer/data --drive-results-root /content/drive/MyDrive/rl_quantum_2026summer_results/paper_replicate --device cuda --checkpoint-heartbeat-minutes 30 --resume
```

## CPU tasks 2--4 (5--20 minutes each)

```bash
python -m paper_replicate.run_resumable_paper_replicate_task_queue --task-name audit_pairwise_absolute_and_ideal_label_coverage --data-root /content/rl_quantum_2026summer/data --drive-results-root /content/drive/MyDrive/rl_quantum_2026summer_results/paper_replicate --device cpu --resume
python -m paper_replicate.run_resumable_paper_replicate_task_queue --task-name validate_peak_feature_extraction_on_training_and_validation_images --data-root /content/rl_quantum_2026summer/data --drive-results-root /content/drive/MyDrive/rl_quantum_2026summer_results/paper_replicate --device cpu --resume
python -m paper_replicate.run_resumable_paper_replicate_task_queue --task-name run_model_and_data_protocol_tests --data-root /content/rl_quantum_2026summer/data --drive-results-root /content/drive/MyDrive/rl_quantum_2026summer_results/paper_replicate --device cpu --resume
python -m paper_replicate.run_resumable_paper_replicate_task_queue --task-name summarize_completed_gpu_tasks_and_prepare_active_learning_features --data-root /content/rl_quantum_2026summer/data --drive-results-root /content/drive/MyDrive/rl_quantum_2026summer_results/paper_replicate --device cpu --resume
```

The training progress bar reports the current epoch. `training_progress.json` is updated after every epoch; a durable `completed_task_result.json` is written before the temporary checkpoint is removed. Final model weights and feature exports are retained in Drive.
