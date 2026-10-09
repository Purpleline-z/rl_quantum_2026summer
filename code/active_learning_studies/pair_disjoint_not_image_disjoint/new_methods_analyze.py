#!/usr/bin/env python3
"""Compact analysis of the new-methods runs.  usage: new_methods_analyze.py <run name under results/new_methods, e.g. dev> [methods comma list]
Per-seed gain over Random (mean of the 5 draws) averaged over budgets 10/20/40/60, per split x condition cell and pooled over the four cells.
Metrics: log-loss (gain = Random - method), AUC, accuracy, calibrated log-loss (Split A only).  One-sided Wilcoxon (alternative: gain > 0); no multiplicity correction here
(development); the confirmatory analysis is new_methods_confirm.py."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent / "results" / "new_methods"
METRICS = (("test_decisive_log_loss", -1, "logloss"), ("test_decisive_auc", 1, "AUC"), ("test_decisive_accuracy", 1, "acc"), ("test_calibrated_log_loss", -1, "callogloss"))


def load(run):
    rows = []
    for split in "AB":
        for cond in ("single", "sequential"):
            for p in sorted((ROOT / run / split / cond).glob("seed*_*.json")):
                d = json.loads(p.read_text())
                if d["strategy"] == "initial_only": continue
                fam = d["strategy"].rsplit("_r", 1)[0] if d["strategy"].startswith("random") and d["strategy"][-1].isdigit() and "_r" in d["strategy"] else d["strategy"]
                for b, v in d["checkpoints"].items():
                    rows.append({"split": split, "cond": cond, "seed": d["seed"], "family": fam, "budget": int(b), **{m: v.get(m, np.nan) for m, _, _ in METRICS}})
    return pd.DataFrame(rows)


def gains(frame, only=None):
    """DataFrame: split, cond, seed, family, metric -> gain averaged over budgets (positive = better than Random)."""
    out = []
    per = frame.groupby(["split", "cond", "seed", "family", "budget"])[[m for m, _, _ in METRICS]].mean().reset_index()
    rand = per[per.family == "random"].set_index(["split", "cond", "seed", "budget"])
    for fam, g in per[per.family != "random"].groupby("family"):
        if only and fam not in only: continue
        g = g.set_index(["split", "cond", "seed", "budget"]); common = g.index.intersection(rand.index)
        for m, sign, name in METRICS:
            diff = sign * (g.loc[common, m] - rand.loc[common, m]) * -1 if False else sign * (rand.loc[common, m] - g.loc[common, m]) * -1
            # gain: log-loss lower is better -> Random - method; AUC/acc higher is better -> method - Random
            diff = (rand.loc[common, m] - g.loc[common, m]) if sign < 0 else (g.loc[common, m] - rand.loc[common, m])
            d = diff.groupby(level=["split", "cond", "seed"]).mean().reset_index(name="gain"); d["family"], d["metric"] = fam, name; out.append(d)
    return pd.concat(out, ignore_index=True)


def summarize(g):
    rows = []
    for (fam, metric), sub in g.groupby(["family", "metric"]):
        cells = {f"{s}_{c[:3]}": sub[(sub.split == s) & (sub.cond == c)].set_index("seed").gain for s in "AB" for c in ("single", "sequential")}
        pooled = sub.groupby("seed").gain.mean()   # mean over the four cells per seed
        def stat(x):
            x = x.dropna(); return (x.mean(), (x > 0).mean(), wilcoxon(x, alternative="greater").pvalue if len(x) > 5 and (x != 0).any() else np.nan)
        pm = stat(pooled); row = {"method": fam, "metric": metric, "n": len(pooled), "pooled": pm[0], "pooled_share": pm[1], "pooled_p": pm[2]}
        for k, v in cells.items(): row[k] = v.mean() if len(v) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    run = sys.argv[1]; only = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
    f = load(run); g = gains(f, only); s = summarize(g)
    pd.set_option("display.width", 200); pd.set_option("display.float_format", lambda x: f"{x:+.4f}" if abs(x) < 10 else f"{x}")
    for metric in ("logloss", "AUC", "acc", "callogloss"):
        t = s[s.metric == metric].drop(columns="metric").set_index("method"); t["pooled_p"] = t.pooled_p.map(lambda x: f"{x:.3f}"); t["pooled_share"] = t.pooled_share.map(lambda x: f"{x:.2f}")
        print(f"\n== {metric}: mean gain over Random (A/B x single/sequential cells, pooled over the 4 cells per seed; one-sided p, uncorrected)"); print(t.sort_values("pooled", ascending=False, key=lambda c: c.astype(float) if c.dtype != object else c).to_string())
    s.to_csv(ROOT / run / "summary.csv", index=False)
