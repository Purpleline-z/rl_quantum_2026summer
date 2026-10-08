#!/usr/bin/env python3
"""Aggregate the held-out pair-preference study: log-loss/accuracy tables, paired differences vs random, mean ranks, curves."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "results" / "pair_endpoint_study"


STUDY_SEEDS = {42, 79, 123, 202, 303}


def load(mode: str = "single", seeds: str = "all") -> pd.DataFrame:
    """mode: single (one batch from the 10-group model) or sequential (rounds of 10).  seeds: study (5), extension (400-429), all."""
    if mode == "single":
        rows = [json.loads(p.read_text()) for p in sorted((OUT / "cells").glob("*.json"))]
    else:
        rows = []
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
    for rank, key in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * pvalues[key])); adjusted[key] = running
    return adjusted


def main() -> None:
    import sys
    from scipy.stats import wilcoxon
    mode = sys.argv[1] if len(sys.argv) > 1 else "single"; seeds = sys.argv[2] if len(sys.argv) > 2 else "all"; tag = f"{mode}_{seeds}"
    cells = load(mode, seeds); acquired = cells[cells.budget > 0]; print(f"mode={mode} seeds={seeds}: {acquired.seed.nunique()} seeds, {acquired.family.nunique()} strategy families")
    per_seed = acquired.groupby(["seed", "budget", "family"], as_index=False)[["test_decisive_log_loss", "test_decisive_accuracy", "type_accuracy"]].mean()
    summary = per_seed.groupby(["family", "budget"], as_index=False).agg(log_loss=("test_decisive_log_loss", "mean"), log_loss_sd=("test_decisive_log_loss", "std"),
                                                                         accuracy=("test_decisive_accuracy", "mean"), accuracy_sd=("test_decisive_accuracy", "std"),
                                                                         type_accuracy=("type_accuracy", "mean"), n=("seed", "nunique"))
    summary.to_csv(OUT / f"{tag}_summary_by_strategy_and_budget.csv", index=False)
    base = per_seed[per_seed.family == "random"][["seed", "budget", "test_decisive_log_loss", "test_decisive_accuracy"]].rename(columns={"test_decisive_log_loss": "random_ll", "test_decisive_accuracy": "random_acc"})
    paired = per_seed.merge(base, on=["seed", "budget"]); paired["d_ll"] = paired.test_decisive_log_loss - paired.random_ll; paired["d_acc"] = paired.test_decisive_accuracy - paired.random_acc
    gain = paired[paired.family != "random"].groupby(["family", "budget"]).agg(d_log_loss=("d_ll", "mean"), sd_ll=("d_ll", "std"), d_accuracy=("d_acc", "mean"), n=("d_ll", "count"),
                                                                               seeds_better_ll=("d_ll", lambda x: (x < 0).mean())).reset_index()
    gain["t_ll"] = gain.d_log_loss / (gain.sd_ll / np.sqrt(gain.n)); gain.to_csv(OUT / f"{tag}_paired_difference_vs_random.csv", index=False)
    # rank of each family within every (seed, budget) by log-loss (1 = best), averaged
    per_seed["rank"] = per_seed.groupby(["seed", "budget"]).test_decisive_log_loss.rank()
    ranks = per_seed.groupby("family")["rank"].mean().sort_values(); ranks.to_csv(OUT / f"{tag}_mean_rank_by_log_loss.csv")
    overall = per_seed.groupby("family").agg(log_loss=("test_decisive_log_loss", "mean"), accuracy=("test_decisive_accuracy", "mean")).join(ranks).sort_values("rank")
    avg_gain = paired[paired.family != "random"].groupby("family").agg(mean_d_ll=("d_ll", "mean"), seeds_budgets_better=("d_ll", lambda x: (x < 0).mean()), n=("d_ll", "count"))
    tests = {}; per_family = paired[paired.family != "random"].groupby(["family", "seed"]).d_ll.mean().reset_index()  # one number per seed: mean over budgets
    for family, group in per_family.groupby("family"):
        values = group.d_ll.to_numpy(); tests[family] = float(wilcoxon(values).pvalue) if len(values) >= 6 and np.any(values != 0) else float("nan")
    adjusted = holm({k: v for k, v in tests.items() if not np.isnan(v)})
    stats = per_family.groupby("family").d_ll.agg(mean_diff="mean", sd="std", n="count", frac_better=lambda x: (x < 0).mean()); stats["wilcoxon_p"] = pd.Series(tests); stats["holm_p"] = pd.Series(adjusted)
    stats = stats.sort_values("mean_diff"); stats.to_csv(OUT / f"{tag}_seed_level_test_vs_random.csv")
    print("log-loss (test) mean over seeds, by budget"); print(summary.pivot(index="family", columns="budget", values="log_loss").round(3).to_string())
    print("\ndecisive accuracy (test)"); print(summary.pivot(index="family", columns="budget", values="accuracy").round(3).to_string())
    print("\noverall (averaged over budgets) with mean rank by log-loss"); print(overall.round(3).to_string())
    print("\npaired log-loss difference vs random, one value per seed (mean over budgets), Wilcoxon signed-rank with Holm correction"); print(stats.round(4).to_string())
    initial = cells[cells.strategy == "initial_only"]
    if len(initial): print("\ninitial only (10 groups): log-loss", round(initial.test_decisive_log_loss.mean(), 3), "accuracy", round(initial.test_decisive_accuracy.mean(), 3))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for family, group in summary.groupby("family"):
        for ax, column in zip(axes, ("log_loss", "accuracy")):
            ax.plot(group.budget, group[column], marker="o", lw=2.4 if family == "random" else 1, label=family, color="black" if family == "random" else None)
    axes[0].set(xlabel="Acquired pair groups", ylabel="Held-out decisive log-loss (lower is better)"); axes[1].set(xlabel="Acquired pair groups", ylabel="Held-out decisive accuracy")
    for ax in axes: ax.grid(alpha=.25)
    axes[0].legend(fontsize=5, ncol=2); fig.savefig(OUT / f"{tag}_pair_endpoint_curves.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
