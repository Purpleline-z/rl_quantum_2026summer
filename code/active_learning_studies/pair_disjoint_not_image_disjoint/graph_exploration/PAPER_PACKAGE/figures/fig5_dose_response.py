"""Figure 5: dose-response of the graph (gain over aw8_all, AUC; rows = split A / B).
 (a) noise in the graph topology, sigma in {0, .5, 1, 2, 3}, lambda 3 and 10, graph-mechanism confirmations GM1 (filled) and GM2 (hollow); the shuffled-graph control at lambda 10 is far below the axis (value annotated)
 (b) random vs local rewiring of a fraction of the edges, lambda 10, development seeds (NOT evidence)
 (c) share of the trajectory frames in the graph, graph-endpoint confirmations GE1 (filled) and GE2 (hollow); series 'all' = ideal images + all pair-group images + that share; 'out' = frames outside every pair group
Data: tables/dose_noise.csv, dose_rewire.csv, dose_nodes.csv (pp_dose.py)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import pp_lib as L, pp_style as S

S.setup(); dn = pd.read_csv(L.TABLES / "dose_noise.csv"); dr = pd.read_csv(L.TABLES / "dose_rewire.csv"); dg = pd.read_csv(L.TABLES / "dose_nodes.csv")
C3, C10 = "#2a78d6", "#eb6834"
fig, axs = plt.subplots(2, 3, figsize=(13.5, 7.2))
def bar(ax, x, d, color, filled, dx=0.0, label=None):
    ax.errorbar(np.asarray(x) + dx, d.gain, yerr=[d.gain - d.lo, d.hi - d.gain], fmt="o-" if filled else "o--", color=color, mfc=color if filled else "none", ms=5, lw=1.5 if filled else 1.2, capsize=2, label=label)
for i, split in enumerate(("A", "B")):
    ax = axs[i, 0]; ax.axhline(0, color=S.INK2, lw=0.8)
    for lam, col in ((3, C3), (10, C10)):
        for blk, filled, dx in (("GM1", True, -0.04), ("GM2", False, 0.04)):
            d = dn[(dn.block == blk) & (dn.lam == lam) & (dn.split == split) & (dn.metric == "auc") & dn.learner.str.startswith("gm_lap") & ~dn.learner.str.contains("shuf")].sort_values("sigma")
            bar(ax, d.sigma, d, col, filled, dx, label=f"lambda {lam}, {blk}" if i == 0 else None)
    sh = dn[(dn.block == "GM1") & (dn.learner == "gm_lap10_n0_shuf") & (dn.split == split) & (dn.metric == "auc")].iloc[0]
    ax.set_xlabel("topology noise sigma (0 = clean kNN graph)"); ax.set_ylabel(f"split {split}: ΔAUC over aw8_all")
    ax.set_ylim(-0.04, 0.03); ax.text(0.98, 0.03, f"shuffled graph (lambda 10): {sh.gain:+.3f}\n[{sh.lo:+.3f}, {sh.hi:+.3f}] (off scale)", transform=ax.transAxes, fontsize=7, color=S.INK2, ha="right")
    ax = axs[i, 1]; ax.axhline(0, color=S.INK2, lw=0.8)
    for kind, col, nm in (("random", C10, "random rewiring"), ("local", C3, "local rewiring (one of the 50 nearest)")):
        d = dr[(dr.kind == kind) & (dr.split == split) & (dr.metric == "auc")].sort_values("frac"); c0 = dr[(dr.kind == "clean") & (dr.split == split) & (dr.metric == "auc")]; d = pd.concat([c0, d])
        bar(ax, d.frac, d, col, False, label=nm if i == 0 else None)
    ax.set_xlabel("share of edges rewired (%) [development]"); ax.set_ylim(-0.14, 0.03)
    ax = axs[i, 2]; ax.axhline(0, color=S.INK2, lw=0.8)
    for series, col, nm in (("all", C3, "ideal + pair-group images + share of frames"), ("out", C10, "frames outside every pair group")):
        for blk, filled, dx in (("GE1", True, -1.0), ("GE2", False, 1.0)):
            names = ["all_p0", "all_p10", "all_p25", "all_p50", "all_p100"] if series == "all" else ["out_p0", "out_p25", "out_p50", "all_p100"]; xs = [0, 10, 25, 50, 100] if series == "all" else [0, 25, 50, 100]
            d = pd.concat([dg[(dg.block == blk) & (dg.variant == v) & (dg.split == split) & (dg.metric == "auc")] for v in names]); bar(ax, xs, d, col, filled, dx, label=f"{nm}, {blk}" if i == 0 else None)
    ax.set_xlabel("share of trajectory frames in the graph (%)"); ax.set_ylim(-0.04, 0.03)
for ax in axs[:, 0]: ax.set_xticks([0, .5, 1, 2, 3])
axs[0, 0].set_title("(a) graph noise (pre-registered GM1, GM2)"); axs[0, 1].set_title("(b) edge rewiring (development)"); axs[0, 2].set_title("(c) unlabelled frames in the graph (pre-registered GE1, GE2)")
for ax in axs[0]: ax.legend(fontsize=6.8, loc="lower left")
for ax in axs[1:, :].ravel(): pass
fig.subplots_adjust(wspace=0.28, hspace=0.32)
fig.suptitle("Dose-response of the graph: gain over aw8_all (AUC, 95% bootstrap CI over 35 seeds per block; 20 seeds in (b))", y=0.995, fontsize=10, fontweight="bold")
S.save(fig, "fig5_dose_response", "Source: tables/dose_noise.csv, dose_rewire.csv, dose_nodes.csv (pp_dose.py). The shuffled-graph control sits far below the axis: 'beats the shuffled control' is a low bar (see claim G3).", note_y=-0.01)
