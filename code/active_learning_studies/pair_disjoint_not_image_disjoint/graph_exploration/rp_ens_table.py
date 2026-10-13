"""Ensemble control table: rp_ens5 (five heads, draw 0 only) vs the baseline lap3_k10_st4 restricted to the same draw (paired on identical label sets).  usage: python3 rp_ens_table.py > rp_ens_table.md"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import gl_tables as gt
D = HERE.parents[1] / "new_directions" / "results" / "rp"
fa = gt.load([D / "devA"]); fe = gt.load([D / "devE"]); b = fa[(fa.learner == "lap3_k10_st4") & (fa.draw == 0)]; rows = []
for sp in "AB":
    for m, name in (("auc", "AUC"), ("acc", "accuracy"), ("ll", "log-loss (baseline - ens)"), ("htr_acc", "HTR accuracy"), ("htr_ll", "HTR log-loss (baseline - ens)")):
        s = -1 if m.endswith("ll") else 1
        x = fe[fe.split == sp].groupby(["seed", "budget"])[m].mean(); y = b[b.split == sp].groupby(["seed", "budget"])[m].mean(); d = (s * (x - y)).dropna().groupby(level="seed").mean()
        try: p = float(wilcoxon(d).pvalue)
        except ValueError: p = float("nan")
        rows.append({"split": sp, "metric": name, "ens5 minus lap3_k10_st4 (draw 0)": d.mean(), "share seeds better": float((d > 0).mean()), "p (uncorrected)": p, "n seeds": len(d)})
print(gt.to_md(pd.DataFrame(rows)))
