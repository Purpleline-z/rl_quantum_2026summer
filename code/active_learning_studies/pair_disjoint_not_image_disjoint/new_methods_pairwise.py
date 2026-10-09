#!/usr/bin/env python3
"""Paired comparison of two strategies (no Random involved): per-seed difference of a metric averaged over budgets and the four split x condition cells; one-sided Wilcoxon (A better than B).
usage: new_methods_pairwise.py <run> <A> <B> <metric list: logloss,AUC,acc,htr_acc,htr_auc,htr_ll>"""
import sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import new_methods_analyze as an
run, a, b, metrics = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
f = an.load(run); col = {name: (m, sign) for m, sign, name in an.METRICS}
for name in metrics:
    m, sign = col[name]; per = f[f.family.isin([a, b])].groupby(["split", "cond", "seed", "family", "budget"])[m].mean().unstack("family")
    d = (per[a] - per[b]) * (1 if sign > 0 else -1); s = d.groupby(["split", "cond", "seed"]).mean().groupby("seed").mean()
    print(f"{name:8s} {a} - {b}: mean {s.mean():+.4f} (sd {s.std():.4f}), seeds better {np.mean(s > 0):.2f}, one-sided p {wilcoxon(s, alternative='greater').pvalue:.4f}, n={len(s)}")
