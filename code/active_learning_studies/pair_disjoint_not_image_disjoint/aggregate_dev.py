#!/usr/bin/env python3
"""Compact development table: gain over Random (5-draw mean) per strategy for a seed range, four blocks (split x condition), log-loss and AUC.

usage: aggregate_dev.py LO-HI [strategy,strategy,...]      (dev seeds only: never use this to pick among rules on the confirmation seeds)
Prints mean gain, number of seeds better, and the raw (uncorrected) Wilcoxon p for log-loss; this is a development aid, not the reported test.
"""
import sys

import numpy as np
import pandas as pd

import aggregate_graph_vs_random as agg

lo, hi = (int(x) for x in sys.argv[1].split("-")); wanted = sys.argv[2].split(",") if len(sys.argv) > 2 else None
blocks = [("A", "single"), ("A", "sequential"), ("B", "single"), ("B", "sequential")]; table = {}
for split, cond in blocks:
    cells = agg.load(split, cond); cells = cells[(cells.seed >= lo) & (cells.seed <= hi)]
    for metric, (label, lower) in agg.METRICS.items():
        sign = -1.0 if lower else 1.0; per = cells.groupby(["seed", "budget", "family"], as_index=False)[metric].mean()
        random = per[per.family == "random"][["seed", "budget", metric]].rename(columns={metric: "random"}); paired = per.merge(random, on=["seed", "budget"]); paired["gain"] = sign * (paired[metric] - paired.random)
        seed_level = paired[paired.family != "random"].groupby(["family", "seed"]).gain.mean().reset_index()
        for family, g in seed_level.groupby("family"):
            v = g.gain.to_numpy(); key = (family, f"{split}-{cond[:3]}"); table.setdefault(key, {})[label] = (v.mean(), (v > 0).mean(), agg.wilcoxon_p(v), len(v))
families = sorted({k[0] for k in table}); families = [f for f in families if wanted is None or f in wanted]
header = "strategy".ljust(40) + "".join(f"{s+'-'+c[:3]:>16}" for s, c in blocks)
for label in ("log-loss", "AUC"):
    print(f"\n{label} gain over Random (seeds {lo}-{hi}); cell = mean gain (share of seeds better, raw p)"); print(header)
    for f in sorted(families, key=lambda f: -np.mean([table[(f, f'{s}-{c[:3]}')][label][0] for s, c in blocks if (f, f'{s}-{c[:3]}') in table])):
        row = f.ljust(40)
        for s, c in blocks:
            v = table.get((f, f"{s}-{c[:3]}"), {}).get(label)
            row += f"{v[0]:+.3f}({v[1]:.0%},{v[2]:.2f})".rjust(16) if v else "-".rjust(16)
        print(row)
