"""
Google Colab GPU Experiment Runner
====================================
Multi-Head Bradley-Terry Active Learning — RHEED Image Quality Assessment

HOW TO USE (copy cells into a Colab notebook)
----------------------------------------------

## CELL 1 — Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

## CELL 2 — Clone / update the repository
```python
import os
GITHUB_USER  = 'Purpleline-z'
GITHUB_TOKEN = 'ghp_YOUR_TOKEN_HERE'   # Settings → Developer settings → Tokens (classic)
BRANCH       = 'claude/determined-tesla-wawr1z'
REPO_DIR     = '/content/rl_quantum_2026summer'

if not os.path.exists(REPO_DIR):
    os.system(
        f"git clone --branch {BRANCH} "
        f"https://{GITHUB_USER}:{GITHUB_TOKEN}@github.com/"
        f"{GITHUB_USER}/rl_quantum_2026summer.git {REPO_DIR}"
    )
else:
    os.system(f"cd {REPO_DIR} && git fetch origin && git checkout {BRANCH} && git pull")

print("Repo ready.")
```

## CELL 3 — Install dependencies (first run only)
```python
import subprocess
subprocess.run(["pip", "install", "-q", "tqdm", "scipy", "scikit-learn",
                "matplotlib", "torch", "torchvision"], check=True)
print("Dependencies installed.")
```

## CELL 4 — Set your data path
```python
# Adjust to wherever your RHEED data lives on Drive
DATA_ROOT = '/content/drive/MyDrive/RHEED_data'
REPO_DIR  = '/content/rl_quantum_2026summer'

import subprocess, os
os.makedirs(f"{REPO_DIR}/results", exist_ok=True)
```

## CELL 5 — Run synthetic benchmark (CPU, ~5 min for 20 seeds, has progress bar)
```python
import subprocess, sys, os
result = subprocess.run(
    [sys.executable,
     f"{REPO_DIR}/code/active_learning_program/synthetic_bt_benchmark.py"],
    cwd=f"{REPO_DIR}/code/active_learning_program",
    capture_output=False,
)
print("Return code:", result.returncode)
```

## CELL 6 — Run AL experiments (GPU, ~30–60 min total per strategy-seed combo)
# Each strategy × seed is one pipeline run. Run them in a loop:
```python
import subprocess, sys, os, json
from tqdm.auto import tqdm

STRATEGIES = [
    "random",
    "uncertainty",
    "cluster_quota_uncertainty",
    "cluster_margin",
    "mc_dropout_mutual_information",
]
SEEDS    = [42, 43, 44]
BUDGETS  = [20, 40, 80, 160, 320]
CODE_DIR = f"{REPO_DIR}/code/active_learning_program"

all_results = []
pbar = tqdm([(s, seed) for s in STRATEGIES for seed in SEEDS],
            desc="Strategy × Seed", unit="run")

for strategy, seed in pbar:
    pbar.set_postfix(strategy=strategy, seed=seed)
    budget_results = []
    for budget in BUDGETS:
        cmd = [
            sys.executable, "pairwise_active_learning_pipeline.py",
            "--strategies",     strategy,
            "--seed",           str(seed),
            "--budget",         str(budget),
            "--data-root",      DATA_ROOT,
            "--initial-pairs",  "20",
            "--candidate-pairs","300",
            "--epochs",         "15",
            "--acquisition-mode", "single-shot",
            "--device",         "auto",
        ]
        proc = subprocess.run(cmd, cwd=CODE_DIR, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"\\nFAIL: {strategy} seed={seed} budget={budget}")
            print(proc.stderr[-1000:])
            continue
        # Parse last accuracy line from stdout
        for line in reversed(proc.stdout.splitlines()):
            if "accuracy" in line.lower() or "acc" in line.lower():
                print(f"  {strategy} s={seed} b={budget}: {line.strip()}")
                break
        budget_results.append({"strategy": strategy, "seed": seed,
                                "budget": budget, "stdout": proc.stdout[-400:]})
    all_results.extend(budget_results)

results_path = f"{REPO_DIR}/results/rheed_al_results.json"
with open(results_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"Saved to {results_path}")
```

## CELL 7 — Parse pipeline output and aggregate by strategy/budget
```python
import json, numpy as np
from pathlib import Path

RESULT_ROOT = Path(f"{REPO_DIR}/results")
# The pipeline writes per-seed JSON files; aggregate them
import glob, re

records = []
for jf in sorted(glob.glob(f"{REPO_DIR}/results/active_learning_*_seed*/budget_curve.json")):
    with open(jf) as f:
        data = json.load(f)
    # Extract seed from directory name
    seed = int(re.search(r'seed(\d+)', jf).group(1))
    for row in data:
        row["seed"] = seed
        records.append(row)

if not records:
    print("No pipeline output found — check that experiments ran successfully.")
else:
    # Group by strategy and budget
    from collections import defaultdict
    grouped = defaultdict(lambda: defaultdict(list))
    for r in records:
        grouped[r["strategy"]][r["budget"]].append(r.get("holdout_accuracy", 0.0))

    summary = []
    for strategy, bdict in sorted(grouped.items()):
        for budget, accs in sorted(bdict.items()):
            summary.append({
                "strategy": strategy,
                "budget": budget,
                "mean_holdout_accuracy": np.mean(accs),
                "std_holdout_accuracy":  np.std(accs),
                "n_seeds": len(accs),
            })

    with open(f"{REPO_DIR}/results/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Summary saved. Rows:", len(summary))
    for row in summary:
        print(f"  {row['strategy']:35s}  budget={row['budget']:4d}  "
              f"acc={row['mean_holdall_accuracy']:.3f}±{row['std_holdout_accuracy']:.3f}")
```

## CELL 8 — Generate figures
```python
import subprocess, sys
subprocess.run([
    sys.executable,
    f"{REPO_DIR}/code/active_learning_program/plot_results.py",
    f"{REPO_DIR}/results/summary.json",
    "--outdir", f"{REPO_DIR}/results/figures",
    "--metric", "holdout_accuracy",
    "--title", "RHEED Active Learning: Accuracy vs. Labeled Budget",
], check=True)
print("Figures generated.")
```

## CELL 9 — Push results and figures to GitHub
```python
import os
GITHUB_USER  = 'Purpleline-z'
GITHUB_TOKEN = 'ghp_YOUR_TOKEN_HERE'
BRANCH       = 'claude/determined-tesla-wawr1z'
REPO_DIR     = '/content/rl_quantum_2026summer'

os.chdir(REPO_DIR)
os.system(f'git config user.email "purpleline@uchicago.edu"')
os.system(f'git config user.name "Purpleline-z"')
os.system(f'git add results/')
os.system(f'git commit -m "Add experiment results from Colab GPU run"')
os.system(
    f"git push https://{GITHUB_USER}:{GITHUB_TOKEN}@github.com/"
    f"{GITHUB_USER}/rl_quantum_2026summer.git {BRANCH}"
)
print("Pushed to GitHub.")
```

---
NOTE: The `pairwise_active_learning_pipeline.py --acquisition-mode single-shot`
flag runs all acquisition queries in one batch per budget level (vs. sequential
active learning across multiple rounds). For accurate AL comparisons, use
`--acquisition-mode sequential` and set `--budget` to the TOTAL budget you want
at the end of the run (the pipeline internally increments in `--batch-size` steps).

For a quick smoke test to verify the environment (< 2 min):
```bash
cd /content/rl_quantum_2026summer/code/active_learning_program
python pairwise_active_learning_pipeline.py \\
    --smoke-test \\
    --data-root /content/drive/MyDrive/RHEED_data
```
"""

# This file is documentation only — the actual implementation lives in:
#   code/active_learning_program/pairwise_active_learning_pipeline.py  (RHEED experiments)
#   code/active_learning_program/synthetic_bt_benchmark.py              (synthetic benchmark)
#   code/active_learning_program/plot_results.py                        (visualization)
