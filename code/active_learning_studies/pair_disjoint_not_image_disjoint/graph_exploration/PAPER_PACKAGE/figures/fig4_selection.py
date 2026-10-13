"""Figure 4: selection rules vs random selection as the learner gets stronger (single-shot selection; random = mean of 3 draws; paired).  Rows: learner (default head, aw8, aw8_all [descriptive], lap3_k10, lap3_k10_st4)
x selection rule; filled markers = frozen pre-registered tests that were CONFIRMED (only vopt_u with the default head), hollow = not confirmed / descriptive.  Data: tables/selection.csv (pp_selection.py)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import matplotlib.pyplot as plt
import pp_lib as L, pp_style as S

S.setup(); t = pd.read_csv(L.TABLES / "selection.csv")
ROWS = [("ND1", "baseline", "vopt_u", "default head: vopt_u  [ND 3000-3034]", True), ("ND1", "aw8", "vopt_u", "aw8: vopt_u", False), ("ND1", "aw8_all", "vopt_u", "aw8_all: vopt_u (descriptive)", False)]
for blk, tag in (("SH1", "lap3_k10 [SH 8300-8334]"), ("SH2", "lap3_k10 [SH 8350-8384]"), ("SH1S4", "lap3_k10_st4 [SH 8300-8334]")):
    lr = "lap3_k10_st4" if blk == "SH1S4" else "lap3_k10"
    for sel, nm in (("vopt_u", "vopt_u"), ("gvopt_node3", "graph-aware gvopt_node3"), ("typed_decisive_bald", "typed_decisive_bald"), ("vopt_htr4", "HTR-directed vopt_htr4")): ROWS.append((blk, lr, sel, f"{tag}: {nm}", False))
n = len(ROWS); fig, axs = plt.subplots(1, 2, figsize=(12.5, 7.2), sharey=True, gridspec_kw=dict(wspace=0.05))
for ax, m, name in zip(axs, ("auc", "acc"), ("ΔAUC over random selection", "Δaccuracy over random selection")):
    ax.axvline(0, color=S.INK2, lw=0.8)
    for k, (blk, lr, sel, lab, conf) in enumerate(ROWS):
        for off, split in ((-0.14, "A"), (0.14, "B")):
            r = t[(t.block == blk) & (t.learner == lr) & (t.selector == sel) & (t.split == split) & (t.metric == m)].iloc[0]
            ax.errorbar(r.gain, n - 1 - k + off, xerr=[[r.gain - r.lo], [r.hi - r.gain]], fmt=S.SPLIT_MARK[split], color=S.SPLIT_COL[split], ms=4.5, lw=1.3, mfc=S.SPLIT_COL[split] if conf else "none")
    for k in (0, 2, 6, 10): ax.axhline(n - 1 - k - 0.5, color=S.GRID, lw=0.8)
    ax.set_xlabel(name + " (95% bootstrap CI)"); ax.set_title({"auc": "AUC", "acc": "accuracy"}[m]); ax.grid(axis="y", visible=False)
axs[0].set_yticks(range(n)); axs[0].set_yticklabels([r[3] for r in ROWS][::-1], fontsize=7.8, ha="left"); axs[0].tick_params(axis="y", pad=215, length=0)
fig.legend(handles=[plt.Line2D([], [], marker="o", color=S.SPLIT_COL["A"], ls="", label="split A"), plt.Line2D([], [], marker="s", color=S.SPLIT_COL["B"], ls="", label="split B"), plt.Line2D([], [], marker="o", mfc="none", color=S.INK2, ls="", label="hollow = pre-registered but NOT confirmed (or descriptive)")], loc="lower center", ncol=3, bbox_to_anchor=(0.55, -0.02))
fig.suptitle("Selection rules vs random selection: the gain vanishes as the learner improves", y=0.97, fontsize=10, fontweight="bold")
S.save(fig, "fig4_selection", "Source: tables/selection.csv (pp_selection.py); Holm-corrected pre-registered verdicts in tables/verdicts.csv. Same 168 labelled groups re-split in every seed.", note_y=-0.05)
