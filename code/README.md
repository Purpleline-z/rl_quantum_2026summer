# RHEED Active Learning

This repository studies whether a limited labeling budget can be spent on informative **pairwise RHEED comparisons** to improve a downstream classifier of ideal reconstruction types. The review-ready evidence is in [`results/completed_experiments`](results/completed_experiments); full historical execution artifacts remain locally in `results/local_runs` and are ignored by Git.

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
python run_selection_benchmark.py generate-stage1-manifest
python -m unittest discover -s tests
```

Large resumable `.pth` checkpoints are enabled only when `--max-runtime-minutes` is greater than 120. The default 50-minute jobs keep lightweight manifests, results, plots, logs, and job state but do not save model/optimizer checkpoints. `--resume` requires a job configured above 120 minutes.

## Next controlled studies

Run the symmetry ablation first. The existing `none` budget-100 results are reused; the manifest creates the 20 required `left_half_mirror` and `symmetric_average` jobs as well as their already-complete baseline entries.

```bash
python run_budget_curve_study.py generate-manifest --symmetry --data-root /content/drive/MyDrive/rheed_project/data
# Run each incomplete job ID in results/local_runs/budget_curve_study/symmetry_manifest.json
python run_budget_curve_study.py run-job --manifest results/local_runs/budget_curve_study/symmetry_manifest.json --job-id <job-id> --data-root /content/drive/MyDrive/rheed_project/data
python run_budget_curve_study.py aggregate
```

The focused low-budget extension schedules only 10 new seeds at budget 10, for the two primary selectors. Do not use the exploratory manifest for this study.

```bash
python run_budget_curve_study.py generate-manifest --budgets 10 --seeds 404,505,606,707,808,909,1010,1111,1212,1313 --strategies random,uncertainty --manifest-name low_budget_extension_manifest --data-root /content/drive/MyDrive/rheed_project/data
# Run every job in results/local_runs/budget_curve_study/low_budget_extension_manifest.json
python run_budget_curve_study.py aggregate-low-budget-extension
```

Before any temporal loss or decoding rule is considered, create the descriptive order audit:

```bash
python run_temporal_constraint_audit.py --data-root /content/drive/MyDrive/rheed_project/data
```

It does not train on `HTR comes last`, `starts at 1x1`, or `1x1 -> bad -> other`. Those remain tentative physics hypotheses pending confirmation and a saved, validated downstream classifier for any prediction-based audit.

Read [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md) for the protocol definitions, reconciled results, and scientific limitations.

The result-oriented five-selector benchmark is separate from older exploratory runners. Run it from `src`:

```bash
python run_selection_benchmark.py generate-stage1-manifest
python run_selection_benchmark.py prepare-reused-evidence --manifest ../results/local_runs/selection_benchmark/stage1_selector_curves_none/study_manifest.json
# Run each job in that manifest, then aggregate Stage 1.
python run_selection_benchmark.py generate-stage2-manifest
```

Stage 1 completes the five-selector budget curve without symmetry. Stage 2 evaluates `left_half_mirror` and `symmetric_average` for every selector and budget. Both manifests are independent, immutable, and aggregate only their own listed jobs.

Two additional selector studies run from the same `src` directory:

```bash
# GPU session A: 30 MC-dropout selector-screening jobs
python run_selection_benchmark.py generate-mc-screen-manifest
python run_selection_benchmark.py run-manifest --manifest ../results/local_runs/selection_benchmark/mc_dropout_screen_low_budget/study_manifest.json

# GPU session B: 25 Cluster-Margin curve jobs
python run_selection_benchmark.py generate-cluster-margin-manifest
python run_selection_benchmark.py run-manifest --manifest ../results/local_runs/selection_benchmark/cluster_margin_curve_none/study_manifest.json
```

Each command skips completed job results after a Colab interruption. Aggregate only after a manifest finishes.
