#!/usr/bin/env python3
"""Confirmatory analysis of pre-registration 6 (cross-world replication): per-seed gain over Random averaged over budgets, conditions and the two directions; one-sided
Wilcoxon, Holm over 12; per-direction results as secondary.  Writes results/new_methods/confirm6_summary.csv and CONFIRM6_TABLE.md."""
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import new_methods_analyze as an
fr = {d: an.load(f"confirm6_cross{d}") for d in (1, 2)}
col = {name: (m, s) for m, s, name in an.METRICS}
P = {d: f.groupby(["split", "cond", "seed", "family", "budget"])[[m for m, _, _ in an.METRICS]].mean().reset_index() for d, f in fr.items()}
def gains(method, name, d):
    m, sign = col[name]; t = P[d].pivot_table(index=["split", "cond", "seed", "budget"], columns="family", values=m); x = (t[method] - t["random"]) * (1 if sign > 0 else -1)
    return x.groupby(["cond", "seed"]).mean().groupby("seed").mean()
rows = []
for method in ("vopt_u", "vopt_u_inf1", "fisher_dopt", "bald_decisive"):
    for name in ("logloss", "AUC", "acc"):
        g1, g2 = gains(method, name, 1), gains(method, name, 2); s = ((g1 + g2) / 2).dropna()
        rows.append({"method": method, "metric": name, "n": len(s), "pooled": s.mean(), "share": (s > 0).mean(), "p": wilcoxon(s, alternative="greater").pvalue,
                     "dir1": g1.mean(), "p1": wilcoxon(g1.dropna(), alternative="greater").pvalue, "dir2": g2.mean(), "p2": wilcoxon(g2.dropna(), alternative="greater").pvalue})
t = pd.DataFrame(rows); order = np.argsort(t.p.values); m = len(t); h = np.empty(m); r = 0
for rank, i in enumerate(order): r = max(r, min(1, (m - rank) * t.p.values[i])); h[i] = r
t["holm12"] = h; t.to_csv(an.ROOT / "confirm6_summary.csv", index=False)
fmt = lambda x: "<0.0001" if x < 1e-4 else f"{x:.4f}"
out = ["### Pre-registration 6: cross-world replication, seeds 1200-1234 (Holm over 12 tests)", "", "| method | metric | seeds | pooled gain | seeds better | one-sided p | Holm p | direction 1 (H1 -> H2): gain (p) | direction 2 (H2 -> H1): gain (p) |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
for x in t.itertuples(): out.append(f"| {x.method} | {x.metric} | {x.n} | {x.pooled:+.4f} | {x.share:.0%} | {fmt(x.p)} | {fmt(x.holm12)} | {x.dir1:+.4f} ({fmt(x.p1)}) | {x.dir2:+.4f} ({fmt(x.p2)}) |")
(an.ROOT / "CONFIRM6_TABLE.md").write_text("\n".join(out) + "\n"); print("\n".join(out))
