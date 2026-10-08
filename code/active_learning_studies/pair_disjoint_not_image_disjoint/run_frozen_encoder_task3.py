#!/usr/bin/env python3
"""Frozen-encoder Task 3 on CPU: validation-only calibration, then the eight-strategy comparison.

The SimCLR encoder is frozen and its features cached (frozen_encoder_reward_head.py).  Calibration
(seeds 42, 79, 123; outer test never opened) picks learning rate and number of full-batch steps per
budget.  The final comparison uses all five seeds with a 158-group candidate pool, so every budget
(10..100) is smaller than the pool and strategies can genuinely differ.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import tempfile
import time
from pathlib import Path

import pandas as pd
import torch

import frozen_encoder_reward_head as frozen
from pairwise_active_learning_pipeline import Config, Experiment

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "frozen_encoder_task3"
STRATEGIES = ("random", "uncertainty", "core_set", "cluster_quota_uncertainty", "uncertainty_diversity",
              "cluster_margin_pairwise", "mc_dropout_probability_variance", "mc_dropout_mutual_information")
BUDGETS = (10, 25, 50, 75, 100)
CALIBRATION_SEEDS, ALL_SEEDS = (42, 79, 123), (42, 79, 123, 202, 303)
LEARNING_RATES, STEPS = (1e-3, 3e-3, 1e-2), (100, 300, 1000)
POOL = 158  # 168 usable groups - 10 initial


def make_experiment(seed: int, data_root: str | None, scratch: Path) -> Experiment:
    cfg = Config(seed=seed, initial_pairs=10, candidate_pairs=POOL, device="cpu", data_root=data_root, dropout_p=.2, mc_samples=10,
                 clusters=20, diversity_lambda=.5, encoder_initialization="simclr", acquisition_mode="single-shot",
                 exclude_all_ideal_identities_from_pairwise=True, manifest_dir=str(scratch / f"manifests_seed{seed}"))
    return Experiment(cfg)


def reference_batch(candidates: list[str], seed: int, budget: int) -> list[str]:
    return random.Random(seed * 1_000_003 + budget).sample(candidates, min(budget, len(candidates)))


def calibrate(data_root, scratch, cache) -> dict:
    rows = []
    for seed in CALIBRATION_SEEDS:
        exp = make_experiment(seed, data_root, scratch); initial, candidates = exp.load_and_split()
        features = frozen.FrozenFeatures(exp, cache)
        for budget in BUDGETS:
            ids = initial + reference_batch(candidates, seed, budget)
            for lr, steps in itertools.product(LEARNING_RATES, STEPS):
                model = frozen.train_model(exp, features, ids, lr, steps)
                rows.append({"seed": seed, "budget": budget, "learning_rate": lr, "steps": steps,
                             "utility_validation_accuracy": frozen.evaluate_model(exp, features, model, "utility_validation")["test_accuracy"]})
        print(f"calibrated seed {seed}", flush=True)
    frame = pd.DataFrame(rows); OUT.mkdir(parents=True, exist_ok=True); frame.to_csv(OUT / "calibration_cells.csv", index=False)
    summary = frame.groupby(["budget", "learning_rate", "steps"], as_index=False).utility_validation_accuracy.mean()
    summary.to_csv(OUT / "calibration_summary.csv", index=False)
    schedule = {}
    for budget, group in summary.groupby("budget"):
        best = group.sort_values(["utility_validation_accuracy", "steps", "learning_rate"], ascending=[False, True, True]).iloc[0]
        schedule[str(budget)] = {"learning_rate": float(best.learning_rate), "steps": int(best.steps),
                                 "mean_utility_validation_accuracy": float(best.utility_validation_accuracy)}
    (OUT / "schedule.json").write_text(json.dumps({"selection": "mean utility-validation accuracy, seeds 42/79/123; ties: fewer steps then lower lr",
                                                   "outer_test_used": False, "protocol_by_budget": schedule}, indent=2))
    return schedule


def final(data_root, scratch, cache, schedule) -> None:
    cells = OUT / "cells"; cells.mkdir(parents=True, exist_ok=True)
    for seed in ALL_SEEDS:
        exp = make_experiment(seed, data_root, scratch); initial, candidates = exp.load_and_split()
        features = frozen.FrozenFeatures(exp, cache); features.install()
        audit = exp.protocol_audit(initial, candidates)
        for budget in BUDGETS:
            lr, steps = schedule[str(budget)]["learning_rate"], schedule[str(budget)]["steps"]
            baseline = frozen.train_model(exp, features, initial, lr, steps)
            rows, cache_ = exp.candidates_with_clusters(candidates, baseline)
            for strategy in STRATEGIES:
                target = cells / f"seed{seed}_budget{budget}_{strategy}.json"
                if target.exists(): continue
                started = time.monotonic()
                selected, _, _ = exp.select(strategy, rows, baseline, cache_, [], budget=budget, labeled_ids=initial)
                ids = [item["pair_id"] for item in selected]
                model = frozen.train_model(exp, features, initial + ids, lr, steps)
                validation, outer = frozen.evaluate_model(exp, features, model, "utility_validation"), frozen.evaluate_model(exp, features, model)
                target.write_text(json.dumps({"seed": seed, "budget": budget, "strategy": strategy, "learning_rate": lr, "steps": steps,
                    "selected_pair_ids": sorted(ids), "utility_validation_accuracy": validation["test_accuracy"],
                    "outer_test_accuracy": outer["test_accuracy"], "outer_test_by_class": outer["by_class"],
                    "identity_audit": audit, "elapsed_seconds": time.monotonic() - started}, indent=1))
        # reference: no acquisition (initial groups only), trained with the budget-10 schedule
        target = cells / f"seed{seed}_budget0_initial_only.json"
        if not target.exists():
            model = frozen.train_model(exp, features, initial, schedule["10"]["learning_rate"], schedule["10"]["steps"])
            target.write_text(json.dumps({"seed": seed, "budget": 0, "strategy": "initial_only",
                "utility_validation_accuracy": frozen.evaluate_model(exp, features, model, "utility_validation")["test_accuracy"],
                "outer_test_accuracy": frozen.evaluate_model(exp, features, model)["test_accuracy"]}, indent=1))
        print(f"finished seed {seed}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("action", choices=("calibrate", "final", "all"))
    parser.add_argument("--data-root", default=None); args = parser.parse_args()
    torch.set_num_threads(8); cache: dict = {}
    with tempfile.TemporaryDirectory() as scratch_name:
        scratch = Path(scratch_name)
        if args.action in ("calibrate", "all"): schedule = calibrate(args.data_root, scratch, cache)
        else: schedule = json.loads((OUT / "schedule.json").read_text())["protocol_by_budget"]
        if args.action in ("final", "all"): final(args.data_root, scratch, cache, schedule)


if __name__ == "__main__":
    main()
