"""Figure 2: per-type panels.  2a accuracy by budget for each judgment type ((1x1), c(6x2), RT13, HTR; HTR last and boxed), block S learners; 2b paired accuracy gain over aw8_all with 95% band.
Per-type test samples are tiny (HTR: a handful of held-out judgments per seed and split): see tables/curves_type.csv for the seed counts; HTR accuracy is coarse.  Data: tables/curves_type.csv."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import matplotlib.pyplot as plt
import pp_lib as L, pp_style as S

S.setup(); t = pd.read_csv(L.TABLES / "curves_type.csv"); TY = [("t1x1_acc", "(1x1)"), ("tc6x2_acc", "c(6x2)"), ("t13_acc", "RT13"), ("htr_acc", "HTR")]
LS = ["aw8_all", "aw8_all_st2lr0.5", "lap3_k10", "lap3_k10_st4"]
for kind in ("level", "gain"):
    fig, axs = plt.subplots(2, 4, figsize=(13, 6.4), sharex=True, sharey="col")
    for j, (m, name) in enumerate(TY):
        for i, split in enumerate(("A", "B")):
            ax = axs[i, j]
            if kind == "gain": ax.axhline(0, color=S.INK2, lw=0.8)
            for l in LS:
                if kind == "level": d = t[(t.block == "S") & (t.learner == l) & (t.split == split) & (t.metric == m) & t.ref.isna()].sort_values("budget") if "ref" in t else t[(t.block == "S") & (t.learner == l) & (t.split == split) & (t.metric == m)].sort_values("budget"); y = "mean"
                else:
                    if l == "aw8_all": continue
                    d = t[(t.block == "S") & (t.learner == l) & (t.split == split) & (t.metric == m) & (t.ref == "aw8_all")].sort_values("budget"); y = "gain"
                if not len(d): continue
                ax.plot(d.budget, d[y], color=S.COL[l], marker="o", ms=3.2); ax.fill_between(d.budget, d.lo, d.hi, color=S.COL[l], alpha=0.14, lw=0)
            S.logx(ax, 8, 75); ax.set_title(f"{name}, split {split}" if True else "", color=S.INK)
            if name == "HTR":
                for sp in ax.spines.values(): sp.set_visible(True); sp.set_color(S.INK2)
            if i == 1: ax.set_xlabel("budget (judgments)")
            if j == 0: ax.set_ylabel("accuracy" if kind == "level" else "Δaccuracy vs aw8_all")
    handles = [plt.Line2D([], [], color=S.COL[l], lw=2, label=S.LABEL[l]) for l in (LS if kind == "level" else LS[1:])]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02)); fig.suptitle("Per-type held-out accuracy, block S (70 seeds; random selection; seeds without a test judgment of the type skipped)" + ("" if kind == "level" else ": paired gain over aw8_all"), fontsize=9, y=1.0)
    S.save(fig, "fig2a_per_type_accuracy" if kind == "level" else "fig2b_per_type_gain", "Source: tables/curves_type.csv (pp_curves.py). HTR panels boxed: the lab's priority type, but only a handful of held-out HTR judgments per seed and split.", note_y=-0.07)
