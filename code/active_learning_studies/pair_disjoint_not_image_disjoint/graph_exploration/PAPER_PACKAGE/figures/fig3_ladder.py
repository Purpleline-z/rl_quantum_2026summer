"""Figure 3: the learner ablation ladder - gain of every step with 95% bootstrap interval (paired over seeds), AUC and accuracy, splits A (circle) and B (square).  Rows: the four steps
(primary block first, replications below), the whole recipe, the sum of the steps (indicative: different seed blocks), and the controls.  Data: tables/ladder.csv (pp_ladder.py)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import pp_lib as L, pp_style as S

S.setup(); t = pd.read_csv(L.TABLES / "ladder.csv")
ROWS = [("L1", "ND1", "1. default head -> aw8 (anchor weight 8)  [ND 3000-3034, pre-reg.]"), ("L1", "ND4", "     replication, ND 5000-5034 (descriptive)"),
        ("L2", "ND4", "2. aw8 -> aw8_all (all ideal images)  [ND 5000-5034, descr.]"), ("L2", "ND1", "     ND 3000-3034 (descr.)"),
        ("L3", "GL1", "3. aw8_all -> lap3_k10 (Laplacian)  [GL 7100-7134, pre-reg.]"), ("L3", "GL2", "     replication GL 7200-7234 (pre-reg.)"), ("L3", "GLC", "     cold start GL 7300-7334 (pre-reg.)"),
        ("L4", "SH1", "4. lap3_k10 -> lap3_k10_st4 (4x steps)  [SH 8300-8334, pre-reg.]"), ("L4", "SH2", "     replication SH 8350-8384 (pre-reg.)"), ("L4", "SHC", "     cold start SH 8385-8399 (pre-reg., 15 seeds)"),
        ("SUM", "ND1+ND4+GL1+SH1", "sum of steps 1-4 (indicative; 4 seed blocks)"),
        ("T", "SH1", "whole recipe: aw8_all -> lap3_k10_st4 (direct)  [SH 8300-8334]"),
        ("C1", "SH1", "control: aw8_all -> aw8_all_st2lr0.5 (graph-free tuning)"), ("C2", "SH1", "PRIMARY: st4 vs tuned graph-free [SH 8300-8334, pre-reg.]"), ("C2", "SH2", "     replication SH 8350-8384 (pre-reg.)"), ("C2", "SHC", "     cold start SH 8385-8399 (pre-reg.)")]
fig, axs = plt.subplots(1, 2, figsize=(12.5, 7.4), sharey=True, gridspec_kw=dict(wspace=0.05)); n = len(ROWS)
for ax, m, name in zip(axs, ("auc", "acc"), ("ΔAUC", "Δaccuracy")):
    ax.axvline(0, color=S.INK2, lw=0.8)
    for k, (sid, blk, lab) in enumerate(ROWS):
        for off, split in ((-0.14, "A"), (0.14, "B")):
            r = t[(t.step == sid) & (t.block == blk) & (t.split == split) & (t.metric == m)].iloc[0]
            ax.errorbar(r.gain, n - 1 - k + off, xerr=[[r.gain - r.lo], [r.hi - r.gain]], fmt=S.SPLIT_MARK[split], color=S.SPLIT_COL[split], ms=4.5, lw=1.3, capsize=0, mfc=(S.SPLIT_COL[split] if "pre-reg" in lab or sid in ("SUM", "T", "C1") else "none"))
    ax.set_xlabel(name + " (95% bootstrap CI over seeds)"); ax.set_title({"auc": "AUC", "acc": "accuracy"}[m]); ax.grid(axis="y", visible=False)
    for k in (1, 3, 6, 9, 10, 11, 12):  # separators between groups
        ax.axhline(n - 1 - k - 0.5, color=S.GRID, lw=0.8)
axs[0].set_yticks(range(n)); axs[0].set_yticklabels([r[2] for r in ROWS][::-1], fontsize=7.6, ha="left"); axs[0].tick_params(axis="y", pad=250, length=0)
fig.legend(handles=[plt.Line2D([], [], marker="o", color=S.SPLIT_COL["A"], ls="", label="split A"), plt.Line2D([], [], marker="s", color=S.SPLIT_COL["B"], ls="", label="split B"), plt.Line2D([], [], marker="o", mfc="none", color=S.INK2, ls="", label="hollow = descriptive, not a frozen test")], loc="lower center", ncol=3, bbox_to_anchor=(0.55, -0.02))
fig.suptitle("Learner ablation ladder (gain of each step, paired over seeds; log-loss in tables/ladder.csv)", y=0.97, fontsize=10, fontweight="bold")
S.save(fig, "fig3_ladder", "Source: tables/ladder.csv (pp_ladder.py). Steps are measured on different seed blocks (each paired inside its own block). Red-team caveat: the gain is uneven across anchor-distance strata; the extra gain of step 4 (longer schedule) is anchor-near (fig6).", note_y=-0.05)
