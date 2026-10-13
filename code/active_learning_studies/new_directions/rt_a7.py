"""Audit 7: sensitivity of the stored confirmations to the way gains are aggregated and tested (no training).
For every stored experiment (selection-htr c1_learn, c2_learn, cold_learn; graph-learner confirmations) and pair (candidate, reference): per-seed gain under several aggregations and the p-value under
(a) the pre-registered Wilcoxon signed-rank test, (b) a sign-flip permutation test (Monte-Carlo, 200k flips, on the per-seed mean gains), plus the share of seeds with a positive gain.
  python3 rt_a7.py > results/rt/a7.md"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

RES = Path(__file__).resolve().parent / "results"
SPECS = [("sh/c1_learn", "random", "lap3_k10_st4", "aw8_all_st2lr0.5"), ("sh/c1_learn", "random", "lap3_k10_st4", "lap3_k10"), ("sh/c2_learn", "random", "lap3_k10_st4", "aw8_all_st2lr0.5"),
         ("sh/c2_learn", "random", "lap3_k10_st4", "lap3_k10"), ("sh/cold_learn", "random", "lap3_k10_st4", "aw8_all_st2lr0.5"), ("sh/cold_learn", "random", "lap3_k10_st4", "lap3_k10"),
         ("gl/confirm1", "random", "lap3_k10", "aw8_all"), ("gl/confirm2", "random", "lap3_k10", "aw8_all")]
AGGS = {"mean over all budgets (pre-registered)": lambda g: g.groupby("seed").auc_diff.mean(), "budgets 20 and 40 only": lambda g: g[g.budget.isin([20, 40])].groupby("seed").auc_diff.mean(),
        "median over budgets": lambda g: g.groupby(["seed", "budget"]).auc_diff.mean().groupby("seed").median(), "budget 60 only": lambda g: g[g.budget == 60].groupby("seed").auc_diff.mean(),
        "budget 10 only": lambda g: g[g.budget == 10].groupby("seed").auc_diff.mean()}


def load(d, selector):
    rows = [r for f in sorted((RES / d).glob("seed*_*.json")) for r in json.loads(f.read_text())]; df = pd.DataFrame(rows)
    if "selector" not in df: df["selector"] = "random"
    return df[df.selector == selector]


def signflip(x, n=200000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x); obs = abs(x.mean()); flips = rng.choice([-1.0, 1.0], size=(n, len(x))); return float(((np.abs((flips * x).mean(1)) >= obs - 1e-15).sum() + 1) / (n + 1))


def md(df):
    cols = list(df.columns); out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows(): out.append("| " + " | ".join((f"{v:+.4f}" if isinstance(v, float) and abs(v) < 1 and not 0 <= v <= 1.0 or isinstance(v, float) and v < 0 else (f"{v:.4g}" if isinstance(v, float) else str(v))) for v in r) + " |")
    return "\n".join(out)


def main():
    rows = []
    for d, sel, cand, ref in SPECS:
        if not (RES / d).is_dir(): continue
        df = load(d, sel)
        for split in sorted(df.split.unique()):
            g = df[df.split == split]; a = g[g.learner == cand].set_index(["seed", "budget", "draw"]); b = g[g.learner == ref].set_index(["seed", "budget", "draw"]); common = a.index.intersection(b.index)
            for metric in ("auc", "acc"):
                gg = (a.loc[common, metric] - b.loc[common, metric]).rename("auc_diff").reset_index()
                for name, f in AGGS.items():
                    per = f(gg).dropna()
                    if len(per) < 5: continue
                    rows.append({"experiment": d, "split": split, "cand - ref": f"{cand} - {ref}", "metric": metric, "aggregation": name, "n_seeds": len(per), "gain": float(per.mean()), "share_positive": float((per > 0).mean()),
                                 "p_wilcoxon": float(wilcoxon(per).pvalue), "p_signflip": signflip(per.to_numpy())})
    df = pd.DataFrame(rows); print("## Aggregation and test sensitivity of the stored confirmations\n"); print(md(df))
    print("\n## Summary: smallest gain / largest p over the aggregations, per experiment x split x metric\n")
    s = df.groupby(["experiment", "split", "cand - ref", "metric"]).agg(min_gain=("gain", "min"), max_p_wilcoxon=("p_wilcoxon", "max"), max_p_signflip=("p_signflip", "max"), min_share=("share_positive", "min")).reset_index(); print(md(s))


if __name__ == "__main__": main()
