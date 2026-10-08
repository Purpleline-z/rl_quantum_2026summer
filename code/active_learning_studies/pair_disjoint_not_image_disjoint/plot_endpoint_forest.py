#!/usr/bin/env python3
"""Forest plot: gain of each strategy over random on the held-out preference endpoint, single-shot vs sequential, 35 seeds."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import aggregate_pair_endpoint as agg
from make_endpoint_report_tables import NAMES

HERE = Path(__file__).resolve().parent
OUT = HERE / "paper_assets" / "endpoint_gain_over_random.png"
INK, MUTED, MARK = "#1f2933", "#6b7785", "#2563a8"   # text, recessive grid/axes, one data hue
PANELS = (("test_decisive_log_loss", True, "Log-loss gain\n(lower loss than Random →)"), ("test_decisive_auc", False, "AUC gain\n(higher AUC than Random →)"))


def seed_gains(mode: str, metric: str, lower: bool) -> pd.DataFrame:
    cells = agg.load(mode, "all"); cells = cells[(cells.budget > 0) & cells[metric].notna()]
    per_seed = cells.groupby(["seed", "budget", "family"], as_index=False)[metric].mean()
    random = per_seed[per_seed.family == "random"][["seed", "budget", metric]].rename(columns={metric: "random"})
    paired = per_seed.merge(random, on=["seed", "budget"]); paired["gain"] = (-1.0 if lower else 1.0) * (paired[metric] - paired.random)
    return paired[paired.family != "random"].groupby(["family", "seed"]).gain.mean().reset_index()


def main() -> None:
    rng = np.random.default_rng(0); order = None; data = {}
    for mode in ("single", "sequential"):
        for metric, lower, _ in PANELS:
            gains = seed_gains(mode, metric, lower); table = pd.read_csv(agg.OUT / f"{mode}_all_{metric}_vs_random.csv", index_col=0)
            rows = {}
            for family, g in gains.groupby("family"):
                v = g.gain.to_numpy(); boots = rng.choice(v, size=(5000, len(v))).mean(1)
                rows[family] = (v.mean(), *np.percentile(boots, [2.5, 97.5]), table.loc[family, "holm_p"] < 0.05)
            data[(mode, metric)] = rows
            if order is None: order = sorted(rows, key=lambda f: rows[f][0])   # single-shot log-loss gain, worst to best (best at top)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 12.2), sharey=True)
    for r, mode in enumerate(("single", "sequential")):
        for c, (metric, lower, label) in enumerate(PANELS):
            ax = axes[r, c]; rows = data[(mode, metric)]
            for y, family in enumerate(order):
                mean, lo, hi, sig = rows[family]
                ax.plot([lo, hi], [y, y], color=MARK, lw=1.6, solid_capstyle="round", alpha=.85)
                ax.plot(mean, y, marker="o", ms=7, mfc=MARK if sig else "white", mec=MARK, mew=1.8, ls="none")
            ax.axvline(0, color=INK, lw=1.2); ax.set_yticks(range(len(order))); ax.set_yticklabels([NAMES.get(f, f) for f in order], fontsize=7.5, color=INK)
            ax.grid(axis="x", color=MUTED, alpha=.18, lw=.8); ax.set_axisbelow(True)
            for side in ("top", "right", "left"): ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(MUTED); ax.tick_params(axis="both", colors=MUTED, length=0); ax.tick_params(axis="y", labelcolor=INK)
            ax.set_xlabel(label, fontsize=9, color=INK); ax.set_title(("Single-shot batch" if mode == "single" else "Sequential rounds of 10") + (" · log-loss" if c == 0 else " · AUC"), fontsize=10.5, loc="left", color=INK, fontweight="bold")
    fig.text(0.01, 0.005, "Mean over 35 seeds of the per-seed gain over Random (averaged over budgets 10–60), 95% bootstrap interval over seeds. Filled marker: Wilcoxon signed-rank p < 0.05 after Holm correction across strategies; open marker: not significant.",
             fontsize=8, color=MUTED, ha="left", va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); OUT.parent.mkdir(exist_ok=True); fig.savefig(OUT, dpi=200, facecolor="white"); print("wrote", OUT)


if __name__ == "__main__":
    main()
