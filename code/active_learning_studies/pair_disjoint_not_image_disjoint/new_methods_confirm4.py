#!/usr/bin/env python3
"""Confirmatory analysis of pre-registration 4 (HTR-targeted acquisition).  usage: new_methods_confirm4.py <run>
12 one-sided Wilcoxon tests (2 methods x 3 HTR metrics x 2 baselines: Random, random_htr), Holm over the 12; secondary all-type metrics vs Random (uncorrected)."""
import sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import new_methods_analyze as an
run = sys.argv[1]; f = an.load(run); col = {name: (m, sign) for m, sign, name in an.METRICS}
per = f.groupby(["split", "cond", "seed", "family", "budget"])[[m for m, _, _ in an.METRICS]].mean().reset_index()
def pooled(method, base, name):
    m, sign = col[name]; t = per.pivot_table(index=["split", "cond", "seed", "budget"], columns="family", values=m)
    d = (t[method] - t[base]) * (1 if sign > 0 else -1); return d.groupby(["split", "cond", "seed"]).mean().groupby("seed").mean().dropna()
rows = []
for method in ("vopt_htr", "fisher_htr"):
    for base in ("random", "random_htr"):
        for name in ("htr_acc", "htr_auc", "htr_ll"):
            s = pooled(method, base, name); rows.append({"method": method, "vs": base, "metric": name, "n": len(s), "mean_gain": s.mean(), "sd": s.std(), "share_seeds_better": (s > 0).mean(), "p": wilcoxon(s, alternative="greater").pvalue})
t = pd.DataFrame(rows); order = np.argsort(t.p.values); m = len(t); holm = np.empty(m); run_max = 0
for rank, i in enumerate(order): run_max = max(run_max, min(1.0, (m - rank) * t.p.values[i])); holm[i] = run_max
t["holm_p"] = holm; pd.set_option("display.width", 200); pd.set_option("display.float_format", lambda x: f"{x:+.4f}"); print(t.to_string(index=False))
print("\nSecondary: all-type endpoints and other types vs Random (uncorrected)")
for method in ("vopt_htr", "fisher_htr", "vopt_u"):
    out = []
    for name in ("acc", "AUC", "logloss", "t13_acc", "tc6x2_acc", "t1x1_acc"):
        s = pooled(method, "random", name); out.append(f"{name} {s.mean():+.4f} (p {wilcoxon(s, alternative='greater').pvalue:.3f})")
    print(method, "|", " | ".join(out))
t.to_csv(an.ROOT / run / "confirmatory4_summary.csv", index=False)
