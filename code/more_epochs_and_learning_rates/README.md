# Strict Image-Disjoint Active-Learning Reproduction

This directory is a new, standalone reproduction protocol. It does not modify, replace, or pool results from the existing pair-disjoint benchmark.

## Design

The primary analysis partitions image identifiers before grouping pairs. A pair is eligible for initial labelling or acquisition only when both image identifiers are in the training partition. Validation and outer-test pair rows require both images to be in their respective partitions. The run aborts if an image or exact pair crosses partitions.

The protocol compares the supplied SimCLR checkpoint and ImageNet ResNet-18 initialization. Hyperparameters are chosen only from validation results over the declared learning-rate and epoch grids. The outer test is not read during calibration. The selected protocol is then held fixed for all strategies and budgets.

The main analysis is strict image-disjoint. A separately labelled `pair_disjoint_sensitivity` run may be enabled only after the primary analysis; it is never combined with primary summaries.

## Run

Run from the repository `code` directory after the source data are available through the existing project path resolver:

```powershell
py -3.13 .\more_epochs_and_learning_rates\run_reproduction.py audit
py -3.13 .\more_epochs_and_learning_rates\run_reproduction.py calibrate
py -3.13 .\more_epochs_and_learning_rates\run_reproduction.py experiment
py -3.13 .\more_epochs_and_learning_rates\run_reproduction.py sensitivity
py -3.13 .\more_epochs_and_learning_rates\run_reproduction.py aggregate
```

Use `--smoke-test` for a small structural check. Results are written to `more_epochs_and_learning_rates/results/` and are intentionally ignored by version control.

## Outputs

- `splits/seed-*/audit.json`: image and pair membership plus exhaustive overlap checks.
- `calibration/grid_results.csv` and `calibration/locked_protocol.json`: validation-only model selection.
- `raw_results.csv`: one row per seed, initialization, training-budget control, strategy, and budget.
- `summary.csv`, `paired_differences.csv`, `paired_tests.csv`: descriptive and paired statistical results.
- `figures/`: per-seed and mean-with-95%-bootstrap-CI curves.
- `TECHNICAL_REPORT.md`: a generated English report that separates observations, statistical support, and unresolved mechanisms.

The command refuses to claim an active-learning advantage unless a strict image-disjoint paired comparison against Random survives Holm-Bonferroni correction.
