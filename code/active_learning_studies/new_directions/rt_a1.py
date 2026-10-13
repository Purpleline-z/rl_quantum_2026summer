"""Audit 1 runner: per-judgment predictions of several learners on random label sets, with the novelty of every test judgment w.r.t. the training images and the graph nodes.

  python3 rt_a1.py --seeds 8800-8819 --splits A,B --out results/rt/a1 [--shard i/n] [--learners ...] [--budgets 10,20,40,60] [--draws 3]
One gzip-CSV per (seed, split); resumable.  Long format: one row per (budget, draw, learner, test judgment).
"""
import argparse, tempfile, time
from pathlib import Path
import numpy as np, pandas as pd, torch
import nd_core as core, nd_graph_learner as G, rt_core as rt

DEFAULT = "aw8_all,aw8_all_st2lr0.5,lap3_k10,lap3_k10_st4"


def seeds_of(text):
    if "-" in text: lo, hi = text.split("-"); return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in text.split(",")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="8800-8819"); ap.add_argument("--splits", default="A,B"); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--learners", default=DEFAULT); ap.add_argument("--budgets", default="10,20,40,60"); ap.add_argument("--draws", type=int, default=3); args = ap.parse_args(); torch.set_num_threads(1)
    names = args.learners.split(","); budgets = [int(b) for b in args.budgets.split(",")]; shard, n = (int(x) for x in args.shard.split("/")); args.out.mkdir(parents=True, exist_ok=True)
    cache = core.feature_cache(); sim = rt.Sim(cache)
    with tempfile.TemporaryDirectory() as scratch:
        for seed in [s for i, s in enumerate(seeds_of(args.seeds)) if i % n == shard]:
            for split in args.splits.split(","):
                target = args.out / f"seed{seed}_{split}.csv.gz"
                if target.exists(): continue
                t0 = time.monotonic(); ctx = core.make_ctx(seed, split, scratch, cache); test = core.TestSet(ctx); gd = G.graph_for(ctx, 10, False); nodes = list(gd.paths); rows = []
                for budget in budgets:
                    for draw in range(args.draws):
                        labelled = core.random_labelled(ctx, budget, draw); nov = rt.novelty_table(sim, ctx, labelled, test, nodes)
                        for name in names:
                            d = rt.predict(ctx, name, labelled, seed * 7 + draw, test)
                            df = nov.copy(); df["seed"], df["split"], df["budget"], df["draw"], df["learner"] = seed, split, budget, draw, name
                            df["j"] = np.arange(len(d)); df["d"] = d; df["sign"] = test.sign; df["weight"] = test.weight; df["k"] = test.k; rows.append(df)
                pd.concat(rows).to_csv(target, index=False); print(f"seed {seed} split {split}: {time.monotonic() - t0:.0f}s", flush=True)


if __name__ == "__main__": main()
