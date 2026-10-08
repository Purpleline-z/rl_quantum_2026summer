#!/usr/bin/env python3
"""Aggregate the held-out pair-preference studies.

usage: aggregate_pair_endpoint.py [single|sequential] [study|extension|all]
For every metric: mean by strategy x budget, per-seed paired difference vs random (random = mean of its replicate draws), mean rank, and a seed-level
Wilcoxon signed-rank test against random with Holm correction over strategies.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

OUT = Path(__file__).resolve().parent / "results" / "pair_endpoint_study"
STUDY_SEEDS = {42, 79, 123, 202, 303}
# metric column -> (label, lower_is_better)
METRICS = {"test_decisive_log_loss": ("log-loss", True), "test_calibrated_log_loss": ("calibrated log-loss", True),
           "test_decisive_auc": ("AUC", False), "test_decisive_accuracy": ("accuracy", False)}


def load(mode: str = "single", seeds: str = "all") -> pd.DataFrame:
    """mode: single (one batch chosen with the 10-group model) or sequential (rounds of 10).  seeds: study (5), extension (400-429), all."""
    rows = []
    if mode == "single":
        rows = [json.loads(p.read_text()) for p in sorted((OUT / "cells").glob("*.json"))]
    else:
        for path in sorted((OUT / "sequential_cells").glob("*.json")):
            d = json.loads(path.read_text())
            for budget, values in d["checkpoints"].items(): rows.append({"seed": d["seed"], "strategy": d["strategy"], "budget": int(budget), **{k: v for k, v in values.items() if k != "selected_pair_ids"}})
    frame = pd.DataFrame(rows)
    if seeds == "study": frame = frame[frame.seed.isin(STUDY_SEEDS)]
    elif seeds == "extension": frame = frame[~frame.seed.isin(STUDY_SEEDS)]
    frame["family"] = frame.strategy.where(~frame.strategy.str.startswith("random_r"), "random")  # replicates fold into "random"
    return frame


def holm(pvalues: dict) -> dict:
    order = sorted(pvalues, key=pvalues.get); adjusted = {}; running = 0.0
    for rank, key in enumerate(order): running = max(running, min(1.0, (len(order) - rank) * pvalues[key])); adjusted[key] = running
    return adjusted


def analyse(cells: pd.DataFrame, metric: str, lower: bool, tag: str) -> None:
    if metric not in cells or cells[metric].isna().all(): print(f"\n[{metric}] not recorded in these cells"); return
    acquired = cells[(cells.budget > 0) & cells[metric].notna()]
    per_seed = acquired.groupby(["seed", "budget", "family"], as_index=False)[metric].mean()
    summary = per_seed.groupby(["family", "budget"], as_index=False).agg(mean=(metric, "mean"), sd=(metric, "std"), n=("seed", "nunique")); summary.to_csv(OUT / f"{tag}_{metric}_by_strategy_and_budget.csv", index=False)
    sign = -1.0 if lower else 1.0   # sign * (strategy - random) > 0 means better
    random = per_seed[per_seed.family == "random"][["seed", "budget", metric]].rename(columns={metric: "random"})
    paired = per_seed.merge(random, on=["seed", "budget"]); paired["gain"] = sign * (paired[metric] - paired.random)
    seed_level = paired[paired.family != "random"].groupby(["family", "seed"]).gain.mean().reset_index(); tests = {}
    for family, group in seed_level.groupby("family"):
        v = group.gain.to_numpy(); tests[family] = float(wilcoxon(v).pvalue) if len(v) >= 6 and np.any(v != 0) else float("nan")
    adjusted = holm({k: v for k, v in tests.items() if not np.isnan(v)})
    stats = seed_level.groupby("family").gain.agg(mean_gain="mean", sd="std", n="count", frac_better=lambda x: (x > 0).mean()); stats["wilcoxon_p"] = pd.Series(tests); stats["holm_p"] = pd.Series(adjusted)
    stats = stats.sort_values("mean_gain", ascending=False); stats.to_csv(OUT / f"{tag}_{metric}_vs_random.csv")
    per_seed["rank"] = per_seed.groupby(["seed", "budget"])[metric].rank(ascending=lower); ranks = per_seed.groupby("family")["rank"].mean().sort_values()
    print(f"\n=== {METRICS[metric][0]} ({'lower' if lower else 'higher'} is better); gain = improvement over random, per seed averaged over budgets ===")
    out = stats.join(ranks.rename("mean_rank")); print(out.round(4).to_string())
    print("by budget (mean):"); print(summary.pivot(index="family", columns="budget", values="mean").round(3).loc[out.index].to_string())


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "single"; seeds = sys.argv[2] if len(sys.argv) > 2 else "all"; tag = f"{mode}_{seeds}"
    cells = load(mode, seeds); print(f"mode={mode} seeds={seeds}: {cells.seed.nunique()} seeds, {cells.family.nunique()} strategy families, {len(cells)} cells")
    for metric, (_, lower) in METRICS.items(): analyse(cells, metric, lower, tag)
    initial = cells[cells.strategy == "initial_only"]
    if len(initial): print("\ninitial only (10 groups):", {m: round(initial[m].mean(), 3) for m in METRICS if m in initial and initial[m].notna().any()})


if __name__ == "__main__":
    main()
