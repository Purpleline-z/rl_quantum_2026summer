#!/usr/bin/env python3
"""95% bootstrap confidence intervals (10,000 resamples over seeds) of the pooled gain over a baseline for the headline confirmatory results."""
import numpy as np, pandas as pd
import new_methods_final_tables as ft   # reuses seeds_gain (regenerates the tables as a side effect)
rng = np.random.default_rng(0)
def ci(s): b = rng.choice(s.values, size=(10000, len(s))).mean(1); return np.percentile(b, [2.5, 97.5])
items = [("confirm", "vopt_u", "random", ("logloss", "AUC", "acc"), "700-734 main"), ("confirm2", "vopt_u", "random", ("logloss", "AUC", "acc"), "800-834 main replication"), ("confirm2", "vopt_u_inf1", "random", ("logloss", "AUC", "acc"), "800-834 main replication"),
         ("confirm3_coldstart", "vopt_u", "random", ("logloss", "AUC", "acc"), "900-934 cold start"), ("confirm3_coldstart", "vopt_u_inf1", "random", ("logloss", "AUC", "acc"), "900-934 cold start"),
         ("confirm4_htr", "vopt_htr", "random", ("htr_acc", "htr_auc", "htr_ll"), "1000-1034 HTR-targeted vs Random"), ("confirm4_htr", "vopt_htr", "random_htr", ("htr_acc", "htr_auc", "htr_ll"), "1000-1034 HTR-targeted vs random_htr"),
         ("confirm5_priority", "vopt_hw4", "random", ("htr_acc", "htr_auc", "htr_ll", "acc", "AUC"), "1100-1134 priority weight")]
rows = ["| set | method | vs | metric | mean gain | 95% bootstrap CI |", "|---|---|---|---|---:|---|"]
for run, method, base, metrics, label in items:
    for m in metrics:
        s, _ = ft.seeds_gain(run, method, base, m); lo, hi = ci(s); rows.append(f"| {label} | {method} | {base} | {m} | {s.mean():+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
out = "\n".join(rows); print(out); (ft.an.ROOT / "BOOTSTRAP_CIS.md").write_text(out + "\n")
