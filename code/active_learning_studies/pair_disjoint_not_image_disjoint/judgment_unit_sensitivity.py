#!/usr/bin/env python3
"""Sensitivity to the initial labelled set: 10 random judgments (X_random) against the initial groups' judgments (X_groups), on the seeds both runs share.
usage: judgment_unit_sensitivity.py A | B"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
import aggregate_pair_endpoint as agg
import judgment_unit_aggregate as ja
import judgment_unit_strategies as ju

split = sys.argv[1]; rows = []
frames = {}
for init in ("groups", "random"):
    for cond in ("single", "sequential"):
        frames[(init, cond)] = ja.load(ja.ROOT / f"{split}_{init}", cond)
out = []
for cond in ("single", "sequential"):
    seeds = sorted(set(frames[("groups", cond)].seed) & set(frames[("random", cond)].seed)); gains = {}
    for init in ("groups", "random"):
        f = frames[(init, cond)]; f = f[f.seed.isin(seeds) & ~f.family.isin([ju.EXTRA_BASELINE])]
        for metric, lower, label in ja.BUDGET_METRICS[:2]:
            ps = f.groupby(["seed", "budget", "family"], as_index=False)[metric].mean(); r = ps[ps.family == "random"][["seed", "budget", metric]].rename(columns={metric: "random"})
            p = ps.merge(r, on=["seed", "budget"]); p["gain"] = (-1 if lower else 1) * (p[metric] - p.random); sl = p[p.family != "random"].groupby(["family", "seed"]).gain.mean().reset_index()
            tests = {k: float(wilcoxon(g.gain).pvalue) if np.any(g.gain != 0) else 1.0 for k, g in sl.groupby("family")}; adj = agg.holm(tests); mean = sl.groupby("family").gain.mean()
            gains[(init, label)] = (mean, adj, f[f.family == "random"].groupby("budget")[metric].mean())
    for _, _, label in ja.BUDGET_METRICS[:2]:
        a, b = gains[("groups", label)], gains[("random", label)]; rho = spearmanr(a[0], b[0].reindex(a[0].index))
        sig = lambda g: ", ".join(f"{k} {g[0][k]:+.3f}" for k in g[1] if g[1][k] < .05) or "none"
        out.append(f"{split} {cond} {label}: {len(seeds)} shared seeds; Spearman rank correlation of the 32 variants' mean gains, initial-groups vs initial-random: {rho.statistic:.2f}; "
                   f"Holm<0.05 with initial groups: {sig(a)}; with 10 random initial judgments: {sig(b)}; Random level by budget (initial groups): {', '.join(f'{v:.3f}' for v in a[2])}; (random initial): {', '.join(f'{v:.3f}' for v in b[2])}")
print("\n".join(out))
