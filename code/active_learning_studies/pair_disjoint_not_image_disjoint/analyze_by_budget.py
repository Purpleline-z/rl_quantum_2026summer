#!/usr/bin/env python3
"""Per-budget comparison: gain over Random at each budget separately (the headline tables average over budgets within a seed)."""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import aggregate_pair_endpoint as agg

METRICS = (("test_decisive_log_loss", True, "log-loss"), ("test_decisive_auc", False, "AUC"))


def main() -> None:
    rows = []
    for mode in ("single", "sequential"):
        cells = agg.load(mode, "all"); cells = cells[cells.budget > 0]
        for metric, lower, label in METRICS:
            ps = cells.groupby(["seed", "budget", "family"], as_index=False)[metric].mean()
            rnd = ps[ps.family == "random"][["seed", "budget", metric]].rename(columns={metric: "random"})
            paired = ps.merge(rnd, on=["seed", "budget"]); paired["gain"] = (-1.0 if lower else 1.0) * (paired[metric] - paired.random)
            for budget, g in paired[paired.family != "random"].groupby("budget"):
                tests = {}
                for family, h in g.groupby("family"):
                    v = h.gain.to_numpy(); tests[family] = float(wilcoxon(v).pvalue) if np.any(v != 0) else 1.0
                adj = agg.holm(tests)
                for family, h in g.groupby("family"): rows.append({"mode": mode, "metric": label, "budget": budget, "family": family, "mean_gain": h.gain.mean(), "p": tests[family], "holm_p": adj[family], "n": len(h)})
    frame = pd.DataFrame(rows); frame.to_csv(agg.OUT / "by_budget_gain_vs_random.csv", index=False)
    for (mode, label), g in frame.groupby(["mode", "metric"]):
        print(f"\n=== {mode}, {label}: best strategy at each budget (mean gain over Random, Holm p across {g.family.nunique()} variants)")
        for budget, h in g.groupby("budget"):
            top = h.sort_values("mean_gain", ascending=False).head(3); sig = h[h.holm_p < .05].sort_values("mean_gain", ascending=False)
            print(f"  budget {budget:>2}: top3 " + "; ".join(f"{r.family} {r.mean_gain:+.3f} (p={r.holm_p:.2f})" for r in top.itertuples()) + (f" | significant: {', '.join(f'{r.family} {r.mean_gain:+.3f}' for r in sig.itertuples())}" if len(sig) else " | none significant"))


if __name__ == "__main__":
    main()
