# One-T4 Colab execution

This is the only experiment command. It uses one T4 runtime for the session-held-out audit, training, sealed evaluation, result interpretation, filtered publication, and push to `main`. It never reuses a `/content` clone, so prior Colab cache files, local `.gitignore` files, staged results, and old commits cannot affect it.

## Required one-time Drive authorization

Google requires an interactive authorization the first time a runtime mounts Drive. Run this once after opening the Colab runtime; no result directory is created before it succeeds.

```python
from google.colab import drive
drive.mount("/content/drive")
```

## Single all-in-one cell

Before running it, set `GITHUB_TOKEN` as a Colab Secret with repository write permission and notebook access enabled. The cell reads it only into the push subprocess and never writes it to Drive, a notebook, git configuration, or the remote URL. A token pasted into chat is exposed and must be revoked after this run; create a replacement token in GitHub rather than reusing it.

```python
from pathlib import Path
import os
import shutil
import subprocess
import tempfile

drive_root = Path("/content/drive/MyDrive")
if not drive_root.is_dir():
    raise RuntimeError("Google Drive is not mounted. Run the one-time mount cell first; no experiment was started.")

from google.colab import userdata
github_token = userdata.get("GITHUB_TOKEN")
if not github_token:
    raise RuntimeError("Colab Secret GITHUB_TOKEN is missing. No experiment was started.")

repository_url = "https://github.com/Purpleline-z/rl_quantum_2026summer.git"
run_name = "session_held_out_static_rheed_classifier_t4"
drive_results_root = drive_root / "rl_quantum_2026summer_results" / "paper_replicate"
drive_results_root.mkdir(parents=True, exist_ok=True)
temporary_parent = Path(tempfile.mkdtemp(prefix="rheed_single_t4_"))
repository_root = temporary_parent / "rl_quantum_2026summer"
environment = os.environ | {"PYTHONDONTWRITEBYTECODE": "1", "GIT_TERMINAL_PROMPT": "0"}

def run(command, *, cwd=None, env=environment):
    print("+", " ".join(map(str, command)))
    subprocess.run(list(map(str, command)), cwd=cwd, env=env, check=True)

try:
    run(["git", "clone", "--depth", "1", "--branch", "main", repository_url, repository_root])
    run(["python", "-m", "pip", "install", "--quiet", "pandas", "pillow", "scikit-learn", "tqdm"])
    run(["python", "-m", "paper_replicate.run_single_t4_session_held_out_study",
         "--data-root", repository_root / "data",
         "--drive-results-root", drive_results_root,
         "--run-name", run_name, "--device", "cuda",
         "--seeds", "42,79,123,202,303", "--epochs", "12",
         "--repository-root", repository_root, "--resume"], cwd=repository_root)
    result_json = drive_results_root / run_name / "completed_task_result.json"
    run(["python", "-m", "paper_replicate.update_readme_from_session_held_out_result",
         "--result-json", result_json, "--readme", repository_root / "paper_replicate" / "README.md"], cwd=repository_root)
    run(["python", "-m", "paper_replicate.publish_drive_results_to_github",
         "--drive-results-root", drive_results_root / run_name,
         "--repository-root", repository_root, "--run-name", run_name], cwd=repository_root)
    run(["git", "config", "user.name", "Purpleline-z"], cwd=repository_root)
    run(["git", "config", "user.email", "purpleline@uchicago.edu"], cwd=repository_root)
    run(["git", "add", "paper_replicate/README.md", f"paper_replicate/results/{run_name}"], cwd=repository_root)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repository_root).returncode == 0:
        print("No new result files to commit; the existing published run already matches Drive.")
    else:
        run(["git", "commit", "-m", "Add session-held-out static RHEED study results"], cwd=repository_root)
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

## Time and output

The initial audit takes about 2–5 minutes. If any held-out session has fewer than 50 decisive comparisons, it writes `session_held_out_data_readiness_audit.json` and `completed_task_result.json` under:

`/content/drive/MyDrive/rl_quantum_2026summer_results/paper_replicate/session_held_out_static_rheed_classifier_t4/`

It then stops without training or manufacturing an accuracy result. The audit records the additional pair count needed in each session.

Once all three sessions meet the threshold, one T4 needs about 50–90 minutes for 3 outer session folds × 5 seeds × 2 variants. Every epoch writes `training_progress.json` with elapsed time and remaining-time estimate. A repeat of this exact cell resumes Drive jobs with a completed JSON or a recent checkpoint. Checkpoints are deleted only after durable model results are written; images, weights, and Drive feature outputs are retained.

Only compact JSON/CSV/Markdown/PNG/PDF plus a SHA-256 manifest are published under `paper_replicate/results/session_held_out_static_rheed_classifier_t4/`. No raw RHEED image, model weight, checkpoint, cache, or token is published.
