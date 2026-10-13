"""Stage-1 runner of the calibration study (see nd_calibration.py).  One gzip-pickle per (seed, split); resumable at cell level.

  python3 cal_run.py --seeds 9100-9119 --splits A,B --out results/cal/dev1 --draws 1 --folds 5 --ens 4 --ls 0.1,0.25 [--shard i/n]
"""
import argparse, tempfile, time
from pathlib import Path
import torch
import nd_core as core, nd_graph_learner as G, nd_calibration as C, rt_core as rt


def seeds_of(text):
    if "-" in text: lo, hi = text.split("-"); return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in text.split(",")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", required=True); ap.add_argument("--splits", default="A,B"); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--draws", type=int, default=1); ap.add_argument("--draw0", type=int, default=0); ap.add_argument("--budgets", default="10,20,40,60"); ap.add_argument("--folds", type=int, default=5); ap.add_argument("--ens", type=int, default=0)
    ap.add_argument("--ls", default=""); ap.add_argument("--learner", default="lap3_k10_st4"); ap.add_argument("--fold-graph", default="all"); args = ap.parse_args(); torch.set_num_threads(1)
    comps = {"ens": args.ens, "ls": tuple(float(x) for x in args.ls.split(",") if x)}; budgets = [int(b) for b in args.budgets.split(",")]; shard, n = (int(x) for x in args.shard.split("/")); args.out.mkdir(parents=True, exist_ok=True)
    cache = core.feature_cache(); sim = rt.Sim(cache)
    with tempfile.TemporaryDirectory() as scratch:
        for seed in [s for i, s in enumerate(seeds_of(args.seeds)) if i % n == shard]:
            for split in args.splits.split(","):
                target = args.out / f"seed{seed}_{split}.pkl.gz"
                if target.exists(): continue
                t0 = time.monotonic(); ctx = core.make_ctx(seed, split, scratch, cache); test = core.TestSet(ctx); gd = G.graph_for(ctx, C.K_GRAPH, False); anchors = C.anchor_paths(ctx); recs = {}
                for budget in budgets:
                    for draw in range(args.draw0, args.draw0 + args.draws):
                        labelled = core.random_labelled(ctx, budget, draw); recs[(budget, draw)] = C.label_set_record(ctx, gd, sim, cache, test, labelled, seed * 7 + draw, comps, args.folds, anchors, args.learner, args.fold_graph)
                C.save_cell(target, {"learner": args.learner, "seed": seed, "split": split, "comps": comps, "folds": args.folds, "fold_graph": args.fold_graph, "recs": recs}); print(f"seed {seed} split {split}: {time.monotonic() - t0:.0f}s", flush=True)


if __name__ == "__main__": main()
