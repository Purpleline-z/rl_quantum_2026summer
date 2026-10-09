#!/usr/bin/env python3
"""Confirmatory analysis of the pre-registered new methods (see NEW_METHODS_PREREGISTRATION.md).
usage: new_methods_confirm.py <run name> <comma list of pre-registered methods> [metrics comma list, default logloss,AUC,acc]
Primary tests: per method and metric, the per-seed gain over Random (mean of 5 draws) averaged over budgets 10/20/40/60 and pooled (mean) over the four
split x condition cells; one-sided Wilcoxon signed-rank (alternative: gain > 0); Holm correction over all (method, metric) tests.  Secondary: the same per cell (uncorrected)."""
import sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import new_methods_analyze as an

run, methods = sys.argv[1], sys.argv[2].split(","); metrics = sys.argv[3].split(",") if len(sys.argv) > 3 else ["logloss", "AUC", "acc"]
frame = an.load(run); g = an.gains(frame, set(methods))
rows = []
for m in methods:
    for metric in metrics:
        sub = g[(g.family == m) & (g.metric == metric)]
        seeds_all = sorted(set(sub.seed)); cells = {(s, c): sub[(sub.split == s) & (sub.cond == c)].set_index("seed").gain for s in "AB" for c in ("single", "sequential")}
        complete = [sd for sd in seeds_all if all(sd in cells[k].index for k in cells)]
        pooled = pd.Series({sd: np.mean([cells[k][sd] for k in cells]) for sd in complete})
        p = wilcoxon(pooled, alternative="greater").pvalue if len(pooled) > 5 and (pooled != 0).any() else np.nan
        rows.append({"method": m, "metric": metric, "n_seeds": len(pooled), "mean_gain": pooled.mean(), "sd": pooled.std(), "share_seeds_better": (pooled > 0).mean(), "p_one_sided": p,
                     **{f"{s}_{c[:3]}": cells[(s, c)].mean() for s in "AB" for c in ("single", "sequential")},
                     **{f"p_{s}_{c[:3]}": (wilcoxon(cells[(s, c)], alternative="greater").pvalue if len(cells[(s, c)]) > 5 else np.nan) for s in "AB" for c in ("single", "sequential")}})
t = pd.DataFrame(rows); order = np.argsort(t.p_one_sided.fillna(1).values); m = len(t); holm = np.empty(m); running = 0
for rank, i in enumerate(order): running = max(running, min(1.0, (m - rank) * t.p_one_sided.fillna(1).values[i])); holm[i] = running
t["holm_p"] = holm
pd.set_option("display.width", 220); pd.set_option("display.float_format", lambda x: f"{x:+.4f}")
print(t.drop(columns=[c for c in t.columns if c.startswith("p_")]).to_string(index=False))
t.to_csv(an.ROOT / run / "confirmatory_summary.csv", index=False)
