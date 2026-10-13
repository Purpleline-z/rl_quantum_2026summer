"""Figure 1: learning curves (AUC, accuracy, log-loss vs judgment budget, 95% bootstrap band over seeds) of the default head, aw8, aw8_all, aw8_all_st2lr0.5, lap3_k10 and lap3_k10_st4 (+ shuffled-graph control);
1b: paired difference to aw8_all (inside each seed block).  Data: tables/curves.csv, tables/curves_diff.csv (pp_curves.py).  Block N (3000-3034 + 5000-5034) holds default/aw8/aw8_all, block S (8300-8334 + 8350-8384)
holds the others; aw8_all appears in both (dotted in N) so the level shift between the two held-out sets is visible; the paired panels are the like-for-like comparison."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import pp_lib as L, pp_style as S

S.setup(); lv = pd.read_csv(L.TABLES / "curves.csv"); df = pd.read_csv(L.TABLES / "curves_diff.csv")
YL = {"auc": "AUC (held-out decisive judgments)", "acc": "accuracy", "ll": "log-loss (lower is better)"}
fig, axs = plt.subplots(3, 2, figsize=(11.5, 10.5), sharex=True, gridspec_kw=dict(wspace=0.5))
for i, m in enumerate(("auc", "acc", "ll")):
    for j, split in enumerate(("A", "B")):
        ax = axs[i, j]; labels = []
        for blk, learners in (("N", ["baseline", "aw8", "aw8_all"]), ("S", ["aw8_all", "aw8_all_st2lr0.5", "lap3_k10", "lap3_k10_st4"])):
            for l in learners:
                d = lv[(lv.block == blk) & (lv.learner == l) & (lv.split == split) & (lv.metric == m)].sort_values("budget"); dotted = (blk == "N" and l == "aw8_all")
                ax.plot(d.budget, d["mean"], ":" if dotted else "-", color=S.COL[l], lw=1.6 if dotted else 2.0, marker="o", ms=3.5); ax.fill_between(d.budget, d.lo, d.hi, color=S.COL[l], alpha=0.14, lw=0)
                if not dotted: last = d.iloc[-1]; labels.append((last.budget, last["mean"], S.LABEL[l].split(" (")[0] + (" [N]" if blk == "N" else ""), S.COL[l]))
        S.logx(ax); ax.set_ylabel(YL[m] if j == 0 else ""); ax.set_title(f"split {split}" if i == 0 else "")
        S.spread(ax, labels)
        if i == 2: ax.set_xlabel("budget: judgments added to the initial groups")
handles = [plt.Line2D([], [], color=S.COL[l], lw=2, label=S.LABEL[l]) for l in ("baseline", "aw8", "aw8_all", "aw8_all_st2lr0.5", "lap3_k10", "lap3_k10_st4")] + [plt.Line2D([], [], color=S.COL["aw8_all"], ls=":", label="aw8_all measured in block N (other held-out set)")]
fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.045))
shuf = lv[(lv.block == "S") & (lv.learner == "lap3_k10_st4_shuf") & (lv.metric == "auc")].groupby("split")["mean"].mean()
fig.suptitle("Learning curves, random selection (mean over seeds, band = 95% bootstrap over 70 seeds per block).\nBlock N: seeds 3000-3034 + 5000-5034 (default head, aw8, aw8_all); block S: seeds 8300-8334 + 8350-8384 (all others).", fontsize=8.5, y=1.0)
S.save(fig, "fig1a_learning_curves", f"Source: tables/curves.csv (pp_curves.py). Seed blocks have different held-out sets: compare lines inside one block, or use fig1b. The shuffled-graph control (AUC {shuf['A']:.2f} in A, {shuf['B']:.2f} in B, mean over budgets) is omitted from the panels (in the table).", note_y=-0.13)
fig, axs = plt.subplots(2, 2, figsize=(11.5, 6.8), sharex=True, gridspec_kw=dict(wspace=0.5))
for i, m in enumerate(("auc", "acc")):
    for j, split in enumerate(("A", "B")):
        ax = axs[i, j]; ax.axhline(0, color=S.INK2, lw=0.8); labs = []
        for blk, ls in (("N", ["baseline", "aw8"]), ("S", ["aw8_all_st2lr0.5", "lap3_k10", "lap3_k10_st4"])):
            for l in ls:
                d = df[(df.block == blk) & (df.learner == l) & (df.split == split) & (df.metric == m)].sort_values("budget")
                ax.plot(d.budget, d.gain, "-", color=S.COL[l], marker="o", ms=3.5); ax.fill_between(d.budget, d.lo, d.hi, color=S.COL[l], alpha=0.16, lw=0)
                last = d.iloc[-1]; labs.append((last.budget, last.gain, S.LABEL[l].split(" (")[0] + (" [N]" if blk == "N" else ""), S.COL[l]))
        S.logx(ax); S.spread(ax, labs); ax.set_ylabel(f"gain over aw8_all: Δ{m.upper() if m == 'auc' else 'accuracy'}" if j == 0 else ""); ax.set_title(f"split {split}" if i == 0 else "")
        if i == 1: ax.set_xlabel("budget: judgments added to the initial groups")
handles = [plt.Line2D([], [], color=S.COL[l], lw=2, label=S.LABEL[l] + (" [block N]" if l in ("baseline", "aw8") else " [block S]")) for l in ("baseline", "aw8", "aw8_all_st2lr0.5", "lap3_k10", "lap3_k10_st4")]
fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06)); fig.suptitle("Paired gain over aw8_all by budget (identical label sets inside each block; 95% bootstrap band over 70 seeds; shuffled-graph control omitted: about -0.1)", fontsize=8.5, y=0.995)
S.save(fig, "fig1b_gain_over_aw8_all", "Source: tables/curves_diff.csv (pp_curves.py).", note_y=-0.1)
