"""Pre-registered confirmatory tests.  A test compares two cells (selector, learner) A and B on one split and one metric: per seed the gain is the mean over budgets and random draws of
sign * (metric_A - metric_B) (sign = +1 for AUC and accuracy, -1 for log-loss, so a positive gain means A is better); the test is a two-sided Wilcoxon signed-rank test over seeds; p-values
are Holm-corrected within the family given in the spec.  The spec is a JSON file committed before the confirmatory cells exist.

  python3 nd_confirm.py SPEC.json RESULTS_DIR [RESULTS_DIR ...]
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
class _Sign(dict):
    def __missing__(self, key): return -1 if key.endswith("ll") else 1
SIGN = _Sign({"auc": 1, "acc": 1, "ll": -1})


def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); out = np.empty_like(p); run = 0.0
    for rank, i in enumerate(order): run = max(run, (len(p) - rank) * p[i]); out[i] = min(1.0, run)
    return out


def load(dirs):
    return pd.DataFrame([r for d in dirs for f in sorted(Path(d).glob("seed*_*.json")) for r in json.loads(f.read_text())])


def gains(frame, split, metric, a, b):
    g = frame[frame.split == split]; mean = lambda cell: g[(g.selector == cell[0]) & (g.learner == cell[1])].groupby(["seed", "budget"])[metric].mean()
    diff = (SIGN[metric] * (mean(a) - mean(b))).dropna(); return diff.groupby(level="seed").mean()


def run(spec, frame):
    rows = []
    for t in spec["tests"]:
        per = gains(frame, t["split"], t["metric"], tuple(t["a"]), tuple(t["b"]))
        try: p = float(wilcoxon(per).pvalue) if np.any(per.to_numpy() != 0) else 1.0
        except ValueError: p = 1.0
        rng = np.random.default_rng(0); boots = rng.choice(per.to_numpy(), size=(4000, len(per))).mean(1)
        rows.append({**t, "n_seeds": len(per), "gain": per.mean(), "lo": np.percentile(boots, 2.5), "hi": np.percentile(boots, 97.5), "share_better": float((per > 0).mean()), "p": p})
    out = pd.DataFrame(rows)
    for fam in out.family.unique(): m = out.family == fam; out.loc[m, "holm"] = holm(out.loc[m, "p"])
    return out


if __name__ == "__main__":
    spec = json.loads(Path(sys.argv[1]).read_text()); frame = load(sys.argv[2:]); out = run(spec, frame)
    pd.set_option("display.width", 250); pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    out["cells"] = [f"{a[0]}:{a[1]} - {b[0]}:{b[1]}" for a, b in zip(out.a, out.b)]
    print(out[["family", "split", "metric", "cells", "n_seeds", "gain", "lo", "hi", "share_better", "p", "holm"]].to_string(index=False))
