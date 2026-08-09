# RHEED Active Learning

This repository studies whether a limited labeling budget can be spent on informative **pairwise RHEED comparisons** to improve a downstream classifier of ideal reconstruction types. The review-ready evidence is in [`results/published`](results/published); full historical execution artifacts remain locally in `results/local_runs` and are ignored by Git.

## Layout

```text
project_root/
├── data/                 # immutable images, pairwise labels, metadata template
└── code/                 # this repository
    ├── src/              # runners and tests
    ├── classifier2/      # encoder checkpoint and original downstream scripts
    └── results/
        ├── published/    # compact, tracked scientific evidence
        └── local_runs/   # local job states, caches, checkpoints, historical runs
```

All scripts resolve data through `src/project_paths.py`. The default is the sibling `../data`; override it with `RHEED_DATA_ROOT` or the shared `--data-root` argument. All generated artifacts go under `results/local_runs`, never `data`.

## Colab + Google Drive

Upload the stable `data` directory to Google Drive once. For updates, zip and upload only `code`:

```powershell
Compress-Archive -Path .\code -DestinationPath .\code.zip -Force
```

In Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
!unzip -q /content/drive/MyDrive/rheed_project/code.zip -d /content
%env RHEED_DATA_ROOT=/content/drive/MyDrive/rheed_project/data
!python /content/code/src/active_learning_pipeline.py --smoke-test
```

## Reproduction commands

Run from `code/src`:

```bash
python active_learning_pipeline.py --smoke-test
python run_fixed_protocol_validation.py --stage final_comparison --generate-manifest
python run_strategy_followup.py generate-manifest
python run_budget_curve_study.py generate-manifest
python -m unittest discover -s tests
```

Large resumable `.pth` checkpoints are enabled only when `--max-runtime-minutes` is greater than 120. The default 50-minute jobs keep lightweight manifests, results, plots, logs, and job state but do not save model/optimizer checkpoints. `--resume` requires a job configured above 120 minutes.

Read [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md) for the protocol definitions, reconciled results, and scientific limitations.

