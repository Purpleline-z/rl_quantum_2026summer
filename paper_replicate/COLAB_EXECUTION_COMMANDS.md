# One-T4 five-fold unseen-image RHEED classifier

The study uses the laboratory's frozen, self-supervised RHEED-SimCLR encoder. The checkpoint has no pairwise preference-label training; it may have seen images without labels during representation pretraining. Every outer-fold test image is excluded from pairwise reward-head fine-tuning, validation, calibration, and architecture selection.

## One-time Google Drive authorization

Google Drive requires the runtime owner to authorize the first mount. No code can bypass that authorization.

```python
from google.colab import drive
drive.mount("/content/drive")
```

## The only experiment cell

Add a repository-write token as Colab Secret `GITHUB_TOKEN` before running. The cell reads it only for `git push`, does not write it to Drive/notebook/git config/remote URL, and deletes the temporary askpass helper. Do not reuse any token previously pasted into chat; revoke it and generate a replacement in GitHub.

```python
from pathlib import Path
import os, shutil, subprocess, tempfile
from google.colab import userdata

drive_root = Path("/content/drive/MyDrive")
if not drive_root.is_dir():
    raise RuntimeError("Google Drive is not mounted. Run the mount cell first; nothing was started.")
github_token = userdata.get("GITHUB_TOKEN")
if not github_token:
    raise RuntimeError("Colab Secret GITHUB_TOKEN is missing. Nothing was started.")

run_name = "five_fold_unseen_image_reward_classifier_t4"
drive_results_root = drive_root / "rl_quantum_2026summer_results" / "paper_replicate"
temporary_parent = Path(tempfile.mkdtemp(prefix="rheed_five_fold_t4_"))
repository_root = temporary_parent / "rl_quantum_2026summer"
environment = os.environ | {"PYTHONDONTWRITEBYTECODE": "1", "GIT_TERMINAL_PROMPT": "0"}

def run(command, cwd=None, env=environment):
    print("+", " ".join(map(str, command)))
    subprocess.run([str(item) for item in command], cwd=cwd, env=env, check=True)

try:
    run(["git", "clone", "--depth", "1", "--branch", "main", "https://github.com/Purpleline-z/rl_quantum_2026summer.git", repository_root])
    run(["python", "-m", "pip", "install", "--quiet", "pandas", "pillow", "scikit-learn", "tqdm"])
    run(["python", "-m", "paper_replicate.run_five_fold_unseen_image_reward_classifier_study",
         "--data-root", repository_root / "data", "--drive-results-root", drive_results_root,
         "--run-name", run_name, "--device", "cuda", "--seeds", "42,79,123", "--epochs", "12",
         "--repository-root", repository_root, "--resume"], cwd=repository_root)
    result_json = drive_results_root / run_name / "completed_task_result.json"
    run(["python", "-m", "paper_replicate.update_readme_from_five_fold_unseen_image_result",
         "--result-json", result_json, "--readme", repository_root / "paper_replicate" / "README.md"], cwd=repository_root)
    run(["python", "-m", "paper_replicate.publish_drive_results_to_github",
         "--drive-results-root", drive_results_root / run_name, "--repository-root", repository_root,
         "--run-name", run_name], cwd=repository_root)
    run(["git", "config", "user.name", "Purpleline-z"], cwd=repository_root)
    run(["git", "config", "user.email", "purpleline@uchicago.edu"], cwd=repository_root)
    run(["git", "add", "paper_replicate/README.md", f"paper_replicate/results/{run_name}"], cwd=repository_root)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repository_root).returncode:
        run(["git", "commit", "-m", "Add five-fold unseen-image RHEED classifier results"], cwd=repository_root)
        askpass = temporary_parent / "git_askpass.sh"
        askpass.write_text("#!/bin/sh\ncase \"$1\" in *Username*) echo Purpleline-z ;; *Password*) echo \"$GITHUB_TOKEN\" ;; esac\n")
        askpass.chmod(0o700)
        push_environment = environment | {"GITHUB_TOKEN": github_token, "GIT_ASKPASS": str(askpass)}
        try:
            run(["git", "pull", "--rebase", "origin", "main"], cwd=repository_root, env=push_environment)
            run(["git", "push", "origin", "main"], cwd=repository_root, env=push_environment)
        finally:
            askpass.unlink(missing_ok=True)
finally:
    github_token = None
    shutil.rmtree(temporary_parent, ignore_errors=True)
```

## Outputs and ETA

Results persist under:

`/content/drive/MyDrive/rl_quantum_2026summer_results/paper_replicate/five_fold_unseen_image_reward_classifier_t4/`

The T4 runs 5 outer folds × 3 seeds × 2 architectures, then one final all-label deployment training: approximately 45–75 minutes. Every epoch writes `training_progress.json` with elapsed time and ETA. A re-run of the unchanged cell resumes only incomplete jobs. After a durable `completed_task_result.json` is written, its temporary checkpoint is removed. Drive retains final weights and active-learning feature outputs.

Only small JSON/CSV/Markdown/PNG/PDF files and a SHA-256 manifest are committed under `paper_replicate/results/five_fold_unseen_image_reward_classifier_t4/`. Images, weights, checkpoints, caches, and credentials remain out of Git.
