#!/usr/bin/env python3
"""Per-budget gains over Random pooled over the four cells.  usage: new_methods_by_budget.py <run> <method list>"""
import sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import new_methods_analyze as an
run, methods = sys.argv[1], sys.argv[2].split(","); f = an.load(run)
per = f.groupby(["split", "cond", "seed", "family", "budget"])[["test_decisive_log_loss", "test_decisive_auc", "test_decisive_accuracy"]].mean().reset_index()
r = per[per.family == "random"].set_index(["split", "cond", "seed", "budget"])
rows = []
for fam in methods:
    g = per[per.family == fam].set_index(["split", "cond", "seed", "budget"]); c = g.index.intersection(r.index)
    d = pd.DataFrame({"logloss": r.loc[c, "test_decisive_log_loss"] - g.loc[c, "test_decisive_log_loss"], "AUC": g.loc[c, "test_decisive_auc"] - r.loc[c, "test_decisive_auc"], "acc": g.loc[c, "test_decisive_accuracy"] - r.loc[c, "test_decisive_accuracy"]})
    s = d.groupby(["seed", "budget"]).mean()
    for b in (10, 20, 40, 60):
        x = s.xs(b, level="budget"); rows.append({"method": fam, "budget": b, **{f"{m}": f"{x[m].mean():+.4f}" for m in ("logloss", "AUC", "acc")}, "p_acc": f"{wilcoxon(x.acc, alternative='greater').pvalue:.3f}", "p_AUC": f"{wilcoxon(x.AUC, alternative='greater').pvalue:.3f}"})
print(pd.DataFrame(rows).to_string(index=False))
