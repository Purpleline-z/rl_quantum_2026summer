"""Gain of every (selector, learner) over the reference cell (random selection, baseline head), paired by seed, split and budget.

  python3 nd_analyze_select.py results/sel1 [--ref-learner baseline] [--holm]
Per seed the gain is averaged over budgets (and over the three random draws of the reference); intervals are paired bootstrap over seeds; p-values are two-sided Wilcoxon signed-rank
over seeds (nominal; the Holm-corrected value is over all rows of the table printed, per split and metric).
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

METRICS = {"auc": +1, "acc": +1, "ll": -1, "acc_amb": +1, "ll_amb": -1}

def load(dirs):
    return pd.DataFrame([r for d in dirs for f in sorted(Path(d).glob("seed*_*.json")) for r in json.loads(f.read_text())])

def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); out = np.empty_like(p); run = 0.0
    for rank, i in enumerate(order): run = max(run, (len(p) - rank) * p[i]); out[i] = min(1.0, run)
    return out

def table(frame, ref_learner="baseline", metrics=("auc", "acc", "ll")):
    ref = frame[(frame.selector == "random") & (frame.learner == ref_learner)].groupby(["seed", "split", "budget"])[list(METRICS)].mean().add_suffix("_ref")
    cells = frame[~((frame.selector == "random") & (frame.learner == ref_learner))].groupby(["seed", "split", "budget", "selector", "learner"])[list(METRICS)].mean().reset_index().join(ref, on=["seed", "split", "budget"])
    rows = []
    for (split, sel, lrn), g in cells.groupby(["split", "selector", "learner"]):
        row = {"split": split, "selector": sel, "learner": lrn, "n_seeds": g.seed.nunique()}
        for m in metrics:
            per = (METRICS[m] * (g[m] - g[f"{m}_ref"])).groupby(g.seed).mean().to_numpy(); rng = np.random.default_rng(0); boots = rng.choice(per, size=(2000, len(per))).mean(1)
            row[f"{m}"] = per.mean(); row[f"{m}_lo"], row[f"{m}_hi"] = np.percentile(boots, [2.5, 97.5]); row[f"{m}_share"] = float((per > 0).mean())
            try: row[f"{m}_p"] = wilcoxon(per).pvalue if np.any(per != 0) else 1.0
            except ValueError: row[f"{m}_p"] = 1.0
        rows.append(row)
    out = pd.DataFrame(rows)
    for m in metrics:
        for split in out.split.unique(): sel = out.split == split; out.loc[sel, f"{m}_holm"] = holm(out.loc[sel, f"{m}_p"])
    return out

if __name__ == "__main__":
    dirs = [a for a in sys.argv[1:] if not a.startswith("--")]; frame = load(dirs); pd.set_option("display.width", 250); pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    mets = ("auc", "acc", "ll") + (("acc_amb", "ll_amb") if "acc_amb" in frame else ())
    t = table(frame, metrics=mets)
    for split in sorted(t.split.unique()):
        print(f"\n=== split {split}: gain over random selection with the baseline head (log-loss gain = reference - cell; positive is better) ==="); s = t[t.split == split]
        cols = ["selector", "learner", "n_seeds"] + sum([[m, f"{m}_lo", f"{m}_hi", f"{m}_p", f"{m}_holm"] for m in ("auc", "acc", "ll")], []); print(s[cols].to_string(index=False))
        if "acc_amb" in t: print("-- ambiguous half of the held-out judgments --"); print(s[["selector", "learner", "acc_amb", "acc_amb_p", "ll_amb", "ll_amb_p"]].to_string(index=False))
