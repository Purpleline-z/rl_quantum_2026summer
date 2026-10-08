#!/usr/bin/env python3
"""Aggregate frozen-encoder Task 3 cells: accuracy table, paired differences vs random, curves."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "frozen_encoder_task3"
BASELINE = HERE / "results" / "simclr_three_seed_identity_safe_task3" / "frozen_encoder_nearest_neighbour_baseline.csv"


def load() -> pd.DataFrame:
    rows = []
    for folder in ("cells", "cells_extra"):
        for path in sorted((OUT / folder).glob("*.json")):
            d = json.loads(path.read_text()); d["group"] = "extra" if folder == "cells_extra" else "original"; rows.append(d)
    frame = pd.DataFrame(rows)
    return frame.drop_duplicates(["seed", "budget", "strategy"])


def main() -> None:
    cells = load(); acquired = cells[cells.budget > 0]
    summary = acquired.groupby(["strategy", "budget"], as_index=False).agg(mean=("outer_test_accuracy", "mean"), sd=("outer_test_accuracy", "std"),
                                                                           validation=("utility_validation_accuracy", "mean"), n=("seed", "nunique"))
    summary.to_csv(OUT / "outer_test_summary_by_strategy_and_budget.csv", index=False)
    random = acquired[acquired.strategy == "random"][["seed", "budget", "outer_test_accuracy"]].rename(columns={"outer_test_accuracy": "random"})
    paired = acquired.merge(random, on=["seed", "budget"]); paired["difference"] = paired.outer_test_accuracy - paired.random
    gain = paired[paired.strategy != "random"].groupby(["strategy", "budget"]).difference.agg(["mean", "std", "count", lambda x: (x > 0).mean()]).reset_index()
    gain.columns = ["strategy", "budget", "mean_difference", "sd", "n_seeds", "fraction_seeds_above_random"]
    gain["paired_t"] = gain.mean_difference / (gain.sd / np.sqrt(gain.n_seeds)); gain.to_csv(OUT / "paired_difference_vs_random.csv", index=False)
    table = summary.assign(cell=lambda d: d["mean"].map("{:.3f}".format) + " ± " + d["sd"].map("{:.3f}".format)).pivot(index="strategy", columns="budget", values="cell")
    print(table.to_string()); print(); print(gain.round(3).to_string())
    initial = cells[cells.strategy == "initial_only"]
    if len(initial): print("\ninitial-only (10 groups, no acquisition):", round(initial.outer_test_accuracy.mean(), 3), "±", round(initial.outer_test_accuracy.std(), 3))
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for strategy, group in summary.groupby("strategy"):
        ax.errorbar(group.budget, group["mean"], yerr=group.sd / np.sqrt(group.n), marker="o", capsize=2, label=strategy, lw=2.2 if strategy == "random" else 1.1)
    if BASELINE.exists(): ax.axhline(pd.read_csv(BASELINE).outer_test_accuracy.mean(), color="black", ls="--", lw=1, label="frozen SimCLR 1-NN (no pair labels)")
    ax.set(xlabel="Acquired pair groups", ylabel="Outer-test accuracy (mean ± s.e. over seeds)", title="Frozen-encoder reward head, order-independent training"); ax.grid(alpha=.25)
    ax.legend(fontsize=6, ncol=2); fig.savefig(OUT / "frozen_encoder_accuracy_curves.png", dpi=200, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
