#!/usr/bin/env python3
"""V1a hyper-parameter grid: lr in {1e-3, 3e-3, 1e-2} x steps in {100, 300, 1000} on seeds 42, 79, 123 (classifier2-style split B), label sets = the 30 initial judgments plus {0, 10, 20, 40, 60}
uniformly random pool judgments (fixed draws), criterion = VALIDATION ideal-image accuracy (the 28 'utility' images; the 28 test ideal images are never evaluated here).

  python3 tune.py run [SEED ...]     # one CSV per seed in results/tuning/ (idempotent)
  python3 tune.py table              # merges them into results/tuning_grid.csv and prints the mean validation accuracy per setting
"""
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import v1a_core as V

OUT = V.HERE / "results" / "tuning"; GRID_LR = (1e-3, 3e-3, 1e-2); GRID_STEPS = (100, 300, 1000); TUNE_SEEDS = (42, 79, 123); BUDGETS = (0, 10, 20, 40, 60)


def label_set(ctx, budget: int) -> list:
    if budget == 0: return list(ctx.initial)
    rng = np.random.default_rng(ctx.seed * 100003 + budget * 101); return list(ctx.initial) + [ctx.pool[i] for i in rng.choice(len(ctx.pool), size=budget, replace=False)]


def run(seeds) -> None:
    torch.set_num_threads(1); OUT.mkdir(parents=True, exist_ok=True); cache = V.load_cache()
    for seed in seeds:
        target = OUT / f"seed{seed}.csv"
        if target.exists(): continue
        rows = []; started = time.monotonic()
        with tempfile.TemporaryDirectory() as scratch:
            ctx = V.V1aContext(seed, scratch, cache, 1e-3, 100)
            for budget in BUDGETS:
                labelled = label_set(ctx, budget)
                for lr in GRID_LR:
                    for steps in GRID_STEPS:
                        model = ctx.fit(labelled, lr=lr, steps=steps); truth, pred = V.win_rate_predict(ctx.exp, ctx.features, model, "utility"); acc, f1 = V.accuracy_and_macro_f1(truth, pred)
                        rows.append({"seed": seed, "budget": budget, "lr": lr, "steps": steps, "val_ideal_accuracy": acc, "val_ideal_macro_f1": f1, "n_val_ideal": len(truth)})
                ctx._fits.clear()
        pd.DataFrame(rows).to_csv(target, index=False); print(f"seed {seed}: {len(rows)} fits in {time.monotonic() - started:.0f}s", flush=True)


def table() -> pd.DataFrame:
    frame = pd.concat([pd.read_csv(f) for f in sorted(OUT.glob("seed*.csv"))], ignore_index=True); frame.to_csv(V.HERE / "results" / "tuning_cells.csv", index=False)
    grid = frame.groupby(["lr", "steps"], as_index=False).agg(mean_val_accuracy=("val_ideal_accuracy", "mean"), mean_val_macro_f1=("val_ideal_macro_f1", "mean"), n_fits=("seed", "size"))
    grid = grid.sort_values(["mean_val_accuracy", "steps", "lr"], ascending=[False, True, True]).reset_index(drop=True); grid.to_csv(V.HERE / "results" / "tuning_grid.csv", index=False); return grid


if __name__ == "__main__":
    if sys.argv[1] == "run": run([int(s) for s in sys.argv[2:]] or TUNE_SEEDS)
    else: print(table().to_string())
