#!/usr/bin/env python3
"""Acquisition-strategy comparison on the held-out pair-preference endpoint, with frozen cached features (CPU).

Per seed: 10 initial groups, 20 validation + 40 test groups (image-disjoint), the rest is the candidate pool.  Hyper-parameters (learning rate,
steps) are chosen on validation log-loss with random batches on seeds 42/79/123; strategies are then compared on the test groups.
"""
from __future__ import annotations

import argparse
import itertools
import json
import tempfile
import time
from pathlib import Path

import pandas as pd
import torch

import frozen_encoder_reward_head as frozen
import pair_preference_endpoint as endpoint
import run_frozen_encoder_task3 as harness
from new_pair_strategies import NEW_STRATEGIES

HERE = Path(__file__).resolve().parent; DATA = HERE.parents[2] / "data"
OUT = HERE / "results" / "pair_endpoint_study"; CACHE = harness.OUT / "simclr_feature_cache.pt"
BUDGETS = (10, 20, 40, 60); ORIGINAL = harness.STRATEGIES; RANDOM_REPLICATES = 5
GRID = list(itertools.product((1e-3, 3e-3, 1e-2), (100, 300, 1000)))


def setup(seed, scratch, cache):
    exp = harness.make_experiment(seed, str(DATA), scratch); initial, candidates = exp.load_and_split(); features = frozen.FrozenFeatures(exp, cache); features.install()
    return (exp, features, *endpoint.split_heldout(exp, initial, candidates))


def calibrate(scratch, cache) -> dict:
    rows = []
    for seed in harness.CALIBRATION_SEEDS:
        exp, features, initial, pool, validation, _ = setup(seed, scratch, cache)
        for budget in BUDGETS:
            ids = initial + harness.reference_batch(pool, seed, budget)
            for lr, steps in GRID:
                result = endpoint.evaluate_preferences(exp, features, frozen.train_model(exp, features, ids, lr, steps), validation)
                rows.append({"seed": seed, "budget": budget, "learning_rate": lr, "steps": steps, **result})
    frame = pd.DataFrame(rows); OUT.mkdir(parents=True, exist_ok=True); frame.to_csv(OUT / "calibration_cells.csv", index=False)
    summary = frame.groupby(["learning_rate", "steps"], as_index=False)[["decisive_log_loss", "decisive_accuracy"]].mean(); summary.to_csv(OUT / "calibration_summary.csv", index=False)
    best = summary.sort_values(["decisive_log_loss", "steps", "learning_rate"]).iloc[0]
    schedule = {"learning_rate": float(best.learning_rate), "steps": int(best.steps), "selection": "mean validation log-loss, seeds 42/79/123, all budgets; test groups unused"}
    (OUT / "schedule.json").write_text(json.dumps(schedule, indent=1)); return schedule


def run(scratch, cache, schedule) -> None:
    cells = OUT / "cells"; cells.mkdir(parents=True, exist_ok=True); lr, steps = schedule["learning_rate"], schedule["steps"]
    for seed in harness.ALL_SEEDS:
        exp, features, initial, pool, validation, test = setup(seed, scratch, cache); started = time.monotonic()
        labeled = [{"pair_id": i, "img1": exp.groups[i].iloc[0].resolved_img1, "img2": exp.groups[i].iloc[0].resolved_img2} for i in initial]
        baseline = frozen.train_model(exp, features, initial, lr, steps); rows, cache_ = exp.candidates_with_clusters(pool, baseline)
        target = cells / f"seed{seed}_budget0_initial_only.json"
        if not target.exists():
            target.write_text(json.dumps({"seed": seed, "budget": 0, "strategy": "initial_only", "pool": len(pool),
                                          **{f"test_{k}": v for k, v in endpoint.evaluate_preferences(exp, features, baseline, test).items()},
                                          "type_accuracy": frozen.evaluate_model(exp, features, baseline)["test_accuracy"]}, indent=1))
        names = list(ORIGINAL) + list(NEW_STRATEGIES) + [f"random_r{r}" for r in range(1, RANDOM_REPLICATES)]
        for budget in BUDGETS:
            for name in names:
                target = cells / f"seed{seed}_budget{budget}_{name}.json"
                if target.exists(): continue
                if name.startswith("random_r"):
                    import random as pyrandom
                    ids = [r["pair_id"] for r in pyrandom.Random(seed * 7919 + budget * 31 + int(name[8:])).sample(rows, budget)]
                elif name in NEW_STRATEGIES: ids = [x["pair_id"] for x in NEW_STRATEGIES[name](rows, labeled, baseline, cache_, budget, seed)]
                else: ids = [x["pair_id"] for x in exp.select(name, rows, baseline, cache_, [], budget=budget, labeled_ids=initial)[0]]
                model = frozen.train_model(exp, features, initial + ids, lr, steps)
                target.write_text(json.dumps({"seed": seed, "budget": budget, "strategy": name, "pool": len(pool), "selected_pair_ids": sorted(ids),
                    **{f"test_{k}": v for k, v in endpoint.evaluate_preferences(exp, features, model, test).items()},
                    **{f"validation_{k}": v for k, v in endpoint.evaluate_preferences(exp, features, model, validation).items()},
                    "type_accuracy": frozen.evaluate_model(exp, features, model)["test_accuracy"]}, indent=1))
        print(f"finished seed {seed} in {time.monotonic() - started:.0f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=("calibrate", "run", "all")); args = parser.parse_args()
    torch.set_num_threads(2); cache = frozen.load_feature_cache(CACHE, DATA)
    with tempfile.TemporaryDirectory() as name:
        scratch = Path(name)
        schedule = calibrate(scratch, cache) if args.action in ("calibrate", "all") else json.loads((OUT / "schedule.json").read_text())
        print("schedule", schedule, flush=True)
        if args.action in ("run", "all"): run(scratch, cache, schedule)


if __name__ == "__main__":
    main()
