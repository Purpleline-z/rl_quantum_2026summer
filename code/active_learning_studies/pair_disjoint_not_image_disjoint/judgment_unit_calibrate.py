#!/usr/bin/env python3
"""Re-calibrate the reward-head schedule (learning rate, full-batch steps) for the JUDGMENT unit, per budget.

Mirrors ``run_pair_endpoint_study.py::calibrate`` (same seeds 42/79/123, same grid lr {1e-3,3e-3,1e-2} x steps {100,300,1000}, same selection metric: mean
validation decisive log-loss, i.e. ``endpoint.evaluate_preferences(...)['decisive_log_loss']`` on the 20 validation groups of Split A; accuracy is stored but not
used; tie-break: lower log-loss, then fewer steps, then lower lr) with these differences only:
  * the training rows are judgment rows: the judgments of the 10 initial groups plus ``budget`` judgments drawn at random from the pool judgments
    (``harness.reference_batch``: random.Random(seed*1_000_003 + budget)), instead of whole pair groups;
  * a schedule is selected PER BUDGET (initial rows only, 10, 20, 40, 60 judgments), not one schedule over all budgets;
  * the outer test groups are never touched (only ``ctx.validation`` is evaluated).
Split B has no validation set, so its schedule is the one selected here on Split A.

Outputs (next to the old schedule.json, which is not touched): schedule_judgment_unit.json, calibration_judgment_unit_cells.csv, calibration_judgment_unit_summary.csv.
Extra check, not used for selection (REPLICATES env, default 5): the same grid on ``REPLICATES`` independent random batches per (seed, budget), file
calibration_judgment_unit_replicates_summary.csv, to see whether the 1-batch choice is stable.
"""
from __future__ import annotations

import itertools
import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import torch

import frozen_encoder_reward_head as frozen
import judgment_unit_study as study
import pair_preference_endpoint as endpoint
import run_frozen_encoder_task3 as harness
import run_pair_endpoint_study as single

GRID = list(itertools.product((1e-3, 3e-3, 1e-2), (100, 300, 1000)))
BUDGETS = (0, 10, 20, 40, 60)  # 0 = the initial rows only
OUT = single.OUT


def batch(ctx, seed: int, budget: int, replicate: int = 0) -> list:
    """Initial judgments + a deterministic random batch of ``budget`` pool judgments (replicate 0 is exactly the mirrored ``reference_batch``)."""
    if budget == 0: return list(ctx.initial)
    if replicate == 0: return list(ctx.initial) + harness.reference_batch(ctx.pool, seed, budget)
    import random
    return list(ctx.initial) + random.Random(seed * 1_000_003 + budget + 7_000_001 * replicate).sample(ctx.pool, min(budget, len(ctx.pool)))


def evaluate_grid(ctx, ids) -> list[dict]:
    rows = study.rows_of(ctx.exp, ids); out = []
    for lr, steps in GRID:
        model = frozen.train_model(ctx.exp, ctx.features, [], lr, steps, rows=rows)
        out.append({"learning_rate": lr, "steps": steps, **endpoint.evaluate_preferences(ctx.exp, ctx.features, model, ctx.validation)})
    return out


def select(summary: pd.DataFrame) -> pd.Series:
    return summary.sort_values(["decisive_log_loss", "steps", "learning_rate"]).iloc[0]


def choose(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    summary = frame.groupby(["budget", "learning_rate", "steps"], as_index=False)[["decisive_log_loss", "decisive_accuracy"]].mean(); schedule = {}
    for budget, g in summary.groupby("budget"):
        best = select(g); schedule["initial" if budget == 0 else str(int(budget))] = {"learning_rate": float(best.learning_rate), "steps": int(best.steps)}
    return schedule, summary


def main() -> None:
    torch.set_num_threads(2); replicates = int(os.environ.get("REPLICATES", "5")); cache = frozen.load_feature_cache(single.CACHE, single.DATA); rows, extra = [], []
    with tempfile.TemporaryDirectory() as name:
        for seed in harness.CALIBRATION_SEEDS:
            ctx = study.Context(seed, Path(name), cache, "A", "groups", .01, 100)  # Split A, initial groups; lr/steps below are the grid, not these
            for budget in BUDGETS:
                for r in evaluate_grid(ctx, batch(ctx, seed, budget)): rows.append({"seed": seed, "budget": budget, **r})
                for rep in range(1, replicates + 1):
                    for r in evaluate_grid(ctx, batch(ctx, seed, budget, rep)): extra.append({"seed": seed, "budget": budget, "replicate": rep, **r})
            print("seed", seed, "done", flush=True)
    frame = pd.DataFrame(rows); frame.to_csv(OUT / "calibration_judgment_unit_cells.csv", index=False)
    schedule, summary = choose(frame); summary.to_csv(OUT / "calibration_judgment_unit_summary.csv", index=False)
    pooled = frame.groupby(["learning_rate", "steps"], as_index=False)[["decisive_log_loss", "decisive_accuracy"]].mean(); best = select(pooled)
    schedule["selection"] = ("per budget (initial rows only, 10, 20, 40, 60 judgments): lowest mean validation decisive log-loss over seeds 42/79/123 on Split A's 20 validation groups, "
                             "one deterministic random batch per seed and budget; tie-break fewer steps then lower lr; outer test groups unused; Split B reuses this schedule")
    schedule["pooled_over_budgets_for_reference"] = {"learning_rate": float(best.learning_rate), "steps": int(best.steps)}
    (OUT / "schedule_judgment_unit.json").write_text(json.dumps(schedule, indent=1)); print(json.dumps(schedule, indent=1))
    if extra:
        ef = pd.DataFrame(extra); es, esum = choose(ef); esum.to_csv(OUT / "calibration_judgment_unit_replicates_summary.csv", index=False)
        print("replicate-based choice (not used):", {k: v for k, v in es.items()})


if __name__ == "__main__":
    main()
