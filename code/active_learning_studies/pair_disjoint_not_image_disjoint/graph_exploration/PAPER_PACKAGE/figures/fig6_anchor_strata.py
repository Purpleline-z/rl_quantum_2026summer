"""Figure 6: where the gain lives - AUC gain by distance of the held-out judgment to the nearest typed anchor (ideal image): anchor-near (<= 0.001), 0.001-0.01, 0.01-0.03, anchor-far (> 0.03: both images > 0.03 from
every anchor), and all judgments.  Red-team development block (seeds 8800-8819, 20 seeds; budgets 10-60, 3 draws; a1c: budgets 20/40, 2 draws).  If the red-team confirmation block (8850-8884) exists its estimates are
overlaid (hollow diamonds).  The bars on the lower axis give the share of judgments in each stratum.  Data: tables/strata.csv, tables/anchor_fractions.csv (pp_strata.py)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import pp_lib as L, pp_style as S

S.setup(); t = pd.read_csv(L.TABLES / "strata.csv"); fr = pd.read_csv(L.TABLES / "anchor_fractions.csv")
ST = ["near<=0.001", "0.001-0.01", "0.01-0.03", "far>0.03", "all"]; XL = ["near\n<= 0.001", "0.001-\n0.01", "0.01-\n0.03", "far\n> 0.03", "all"]
PAIRS = [("aw8_all", "baseline", "step 1-2: aw8_all vs default head", "#2a78d6"), ("lap3_k10", "aw8_all", "step 3: lap3_k10 vs aw8_all", "#e87ba4"), ("lap3_k10_st4", "lap3_k10", "step 4: lap3_k10_st4 vs lap3_k10", "#008300"), ("lap3_k10_st4", "aw8_all_st2lr0.5", "PRIMARY: lap3_k10_st4 vs tuned graph-free", "#eda100")]
fig, axs = plt.subplots(2, 1, figsize=(9.5, 9), gridspec_kw=dict(height_ratios=[3.2, 1], hspace=0.3), sharex=False)
fig2, axs2 = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
for ax_i, split in enumerate(("A", "B")):
    ax = axs2[ax_i]; ax.axhline(0, color=S.INK2, lw=0.8)
    for k, (c, r, lab, col) in enumerate(PAIRS):
        for src, mk, fill in (("development", "o", True), ("confirmation", "D", False)):
            d = t[(t.source.str.startswith(src)) & (t.cand == c) & (t.ref == r) & (t.split == split) & (t.metric == "auc")].set_index("stratum").reindex(ST).dropna(subset=["gain"])
            if not len(d): continue
            xs = np.array([ST.index(s) for s in d.index]) + (k - 1.5) * 0.15
            ax.errorbar(xs, d.gain, yerr=[d.gain - d.lo, d.hi - d.gain], fmt=mk, color=col, mfc=col if fill else "none", ms=5, lw=1.3, capsize=2, label=lab if (src == "development") else None)
    ax.set_xticks(range(len(ST))); ax.set_xticklabels(XL); ax.set_xlabel("distance of the judgment's closer image to the nearest typed anchor (1 - cos)"); ax.set_title(f"split {split}")
    if ax_i == 0: ax.set_ylabel("ΔAUC (95% bootstrap CI over seeds)"); ax.legend(fontsize=7.5, loc="upper right")
fig2.suptitle("AUC gain by anchor proximity of the held-out judgment (red-team development block, 20 seeds; hollow diamonds = confirmation block if present)", fontsize=9.5, fontweight="bold", y=1.0)
S.save(fig2, "fig6_anchor_strata", "Source: tables/strata.csv (pp_strata.py; cross-checked against rt_tables.auc_by_stratum). Anchor-far judgments are ~22% of the test judgments (about 13 per seed-budget cell): wide intervals.", note_y=-0.04)
plt.close(fig)
fig, ax = plt.subplots(figsize=(7, 3.2)); sel = fr[fr.stratum.isin(ST[:4])]
for k, (name, col) in enumerate((("any typed anchor (references + utility + outer-test ideal images)", "#2a78d6"), ("reference images only", "#eb6834"))):
    d = sel[sel.anchor_set == name].set_index("stratum").reindex(ST[:4]); ax.bar(np.arange(4) + (k - 0.5) * 0.36, d.share_of_judgments * 100, width=0.34, color=col, label=name)
    for x, v in zip(np.arange(4) + (k - 0.5) * 0.36, d.share_of_judgments * 100): ax.text(x, v + 0.8, f"{v:.0f}%", ha="center", fontsize=7, color=S.INK2)
ax.set_xticks(range(4)); ax.set_xticklabels(XL[:4]); ax.set_ylabel("% of held-out judgments"); ax.legend(fontsize=7); ax.set_title("Share of held-out judgments by distance to the nearest typed anchor")
S.save(fig, "fig6b_anchor_fractions", "Source: tables/anchor_fractions.csv (pp_strata.py), 2359 judgments (20 seeds x splits A, B).", note_y=-0.12)
