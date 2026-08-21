# Colab commands: SimCLR three-seed identity-safe Task 3

Run these cells in order. Set `DRIVE_OUTPUT` once; do not change it between resumes.
Completed cells retain their compact JSON results but delete their training
checkpoints automatically. A shared baseline checkpoint is kept only until all
strategies that depend on it have completed, so checkpoints consume Drive disk
space temporarily and never accumulate in Colab RAM.

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
%cd /content
!git clone https://github.com/Purpleline-z/rl_quantum_2026summer.git
%cd /content/rl_quantum_2026summer
!git checkout main && git pull --ff-only origin main
!pip install wandb
!wandb login
```

```bash
%cd /content/rl_quantum_2026summer
!DRIVE_OUTPUT='/content/drive/MyDrive/rl_quantum_task3_simclr_identity_safe' ; \
 python code/active_learning_studies/pair_disjoint_not_image_disjoint/run_simclr_identity_safe_task3_colab.py \
 write_protocol --drive-output "$DRIVE_OUTPUT"
```

```bash
# Warm-up prints the initial ETA for the long queue.
!DRIVE_OUTPUT='/content/drive/MyDrive/rl_quantum_task3_simclr_identity_safe' ; \
 python code/active_learning_studies/pair_disjoint_not_image_disjoint/run_simclr_identity_safe_task3_colab.py \
 warmup --drive-output "$DRIVE_OUTPUT" --device cuda
```

```bash
# Task 3a: validation only. Re-run this exact cell after interruption.
!DRIVE_OUTPUT='/content/drive/MyDrive/rl_quantum_task3_simclr_identity_safe' ; \
 python code/active_learning_studies/pair_disjoint_not_image_disjoint/run_simclr_identity_safe_task3_colab.py \
 run_task3a --drive-output "$DRIVE_OUTPUT" --device cuda --wandb-project rl-quantum-task3-private
```

```bash
!DRIVE_OUTPUT='/content/drive/MyDrive/rl_quantum_task3_simclr_identity_safe' ; \
 python code/active_learning_studies/pair_disjoint_not_image_disjoint/run_simclr_identity_safe_task3_colab.py \
 aggregate_task3b --drive-output "$DRIVE_OUTPUT"
```

```bash
# Selector screen: validation only. Re-run unchanged to resume.
!DRIVE_OUTPUT='/content/drive/MyDrive/rl_quantum_task3_simclr_identity_safe' ; \
 python code/active_learning_studies/pair_disjoint_not_image_disjoint/run_simclr_identity_safe_task3_colab.py \
 run_selector_screen --drive-output "$DRIVE_OUTPUT" --device cuda --wandb-project rl-quantum-task3-private
!DRIVE_OUTPUT='/content/drive/MyDrive/rl_quantum_task3_simclr_identity_safe' ; \
 python code/active_learning_studies/pair_disjoint_not_image_disjoint/run_simclr_identity_safe_task3_colab.py \
 aggregate_selector_screen --drive-output "$DRIVE_OUTPUT"
```

```bash
# Final 8-strategy queue. Re-run this exact cell after interruption.
!DRIVE_OUTPUT='/content/drive/MyDrive/rl_quantum_task3_simclr_identity_safe' ; \
 python code/active_learning_studies/pair_disjoint_not_image_disjoint/run_simclr_identity_safe_task3_colab.py \
 run_task3c --drive-output "$DRIVE_OUTPUT" --device cuda --wandb-project rl-quantum-task3-private
```

```bash
# Publication-ready PNG/PDF/SVG figures and a W&B figure artifact.
!DRIVE_OUTPUT='/content/drive/MyDrive/rl_quantum_task3_simclr_identity_safe' ; \
 python code/active_learning_studies/pair_disjoint_not_image_disjoint/generate_simclr_identity_safe_task3_figures.py \
 --drive-output "$DRIVE_OUTPUT" --wandb-project rl-quantum-task3-private
```
