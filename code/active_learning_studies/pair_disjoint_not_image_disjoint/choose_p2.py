#!/usr/bin/env python3
"""Dev-seed summary for the freeze decision (OVERNIGHT2_PLAN.md, Phase 2): mean over the four blocks of the per-seed gain over Random and over vopt_u, per metric, plus the number of blocks in which the gain over Random is positive on all three metrics."""
import numpy as np
import aggregate_overnight2 as a

rules = ["vopt_u", "gvopt_lap", "gvopt_lap03", "gvopt_lap3", "gvopt_lap10", "gvopt_type", "gvopt_prop", "gvopt_sigma", "gvopt_lapsigma"]
tab = {r: {m: [] for m in a.METRICS} for r in rules}; vs = {r: {m: [] for m in a.METRICS} for r in rules}
for split, cond in a.BLOCKS:
    res = {m: a.gains("graph_p2", split, cond, m) for m in a.METRICS}
    for r in rules:
        for m in a.METRICS:
            tab[r][m].append(res[m][r].mean()); vs[r][m].append((res[m][r] - res[m]["vopt_u"]).dropna().mean() if r != "vopt_u" else 0.0)
print(f"{'rule':16s} " + "  ".join(f"{m.replace('test_decisive_','')[:8]:>8s}" for m in a.METRICS) + "   | minus vopt_u: " + "  ".join(f"{m.replace('test_decisive_','')[:8]:>8s}" for m in a.METRICS) + "  blocks all-positive")
for r in rules:
    pos = sum(all(tab[r][m][i] > 0 for m in a.METRICS) for i in range(4))
    print(f"{r:16s} " + "  ".join(f"{np.mean(tab[r][m]):+8.4f}" for m in a.METRICS) + "   |                 " + "  ".join(f"{np.mean(vs[r][m]):+8.4f}" for m in a.METRICS) + f"  {pos}/4")
