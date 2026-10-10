"""Summarise a screen: absolute means per learner and paired gains over the baseline."""
import json, sys
from pathlib import Path
import pandas as pd
import nd_core as core

def load(dirs):
    rows = [r for d in dirs for f in sorted(Path(d).glob("seed*_*.json")) for r in json.loads(f.read_text())]
    return pd.DataFrame(rows)

def main():
    dirs = [a for a in sys.argv[1:] if not a.startswith("--")]; frame = load(dirs); base = "baseline"
    pd.set_option("display.width", 200); pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("absolute means (all budgets, draws, seeds):"); print(frame.groupby(["split", "learner"])[["acc", "ll", "auc"]].mean())
    t = core.paired_table(frame, base); print("\npaired gain over baseline (log-loss gain = baseline - learner; positive is better), 95% bootstrap interval over seeds:")
    cols = ["split", "learner", "n_seeds", "acc_gain", "acc_lo", "acc_hi", "auc_gain", "auc_lo", "auc_hi", "ll_gain", "ll_lo", "ll_hi", "auc_share_better"]; print(t[cols].to_string(index=False))
    by_b = []
    for (split, learner, budget), g in frame.groupby(["split", "learner", "budget"]):
        if learner == base: continue
        b = frame[(frame.split == split) & (frame.learner == base) & (frame.budget == budget)].set_index(["seed", "draw"])
        g = g.set_index(["seed", "draw"]); j = g.join(b, rsuffix="_b", how="inner"); by_b.append({"split": split, "learner": learner, "budget": budget, "auc_gain": (j.auc - j.auc_b).mean(), "acc_gain": (j.acc - j.acc_b).mean(), "ll_gain": (j.ll_b - j.ll).mean()})
    print("\nper budget:"); print(pd.DataFrame(by_b).to_string(index=False))
main()
