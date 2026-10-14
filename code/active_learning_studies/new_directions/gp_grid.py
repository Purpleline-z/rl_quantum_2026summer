"""Fixed-hyperparameter grid of the graph GP on the development seeds (one pickle per (seed, split)): per label set and setting (kernel kind, length factor c, graph strength beta, amplitude) the predictive mean and
variance of every test judgment.  Used to see how the metrics depend on the hyperparameters per type and to compare hand-set with learned hyperparameters.
  python3 gp_grid.py --seeds 11000-11019 --splits A,B --out results/gp/grid1 --cs 0.1,0.15,0.25,0.4 --betas 0,0.1,0.3,1 --amps 1,3,10,30 [--kind rbf] [--shard i/n]"""
import argparse, tempfile, time
from pathlib import Path
import numpy as np, torch
import nd_core as core, nd_calibration as C, nd_gpg as P, rt_core as rt, nd_graph_learner as G
from gp_run import seeds_of


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", required=True); ap.add_argument("--splits", default="A,B"); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--cs", default="0.25"); ap.add_argument("--betas", default="0.3"); ap.add_argument("--amps", default="3"); ap.add_argument("--kind", default="rbf"); ap.add_argument("--budgets", default="10,20,40,60"); ap.add_argument("--aw", type=float, default=8.0)
    args = ap.parse_args(); torch.set_num_threads(1); shard, n = (int(x) for x in args.shard.split("/")); args.out.mkdir(parents=True, exist_ok=True)
    cs = [tuple(float(y) for y in x.split(":")) if ":" in x else float(x) for x in args.cs.split(",")]; betas = [float(x) for x in args.betas.split(",")]; amps = [tuple(float(y) for y in x.split(":")) if ":" in x else float(x) for x in args.amps.split(",")]; budgets = [int(b) for b in args.budgets.split(",")]
    cache = core.feature_cache(); sim = rt.Sim(cache)
    with tempfile.TemporaryDirectory() as scratch:
        for seed in [s for i, s in enumerate(seeds_of(args.seeds)) if i % n == shard]:
            for split in args.splits.split(","):
                target = args.out / f"seed{seed}_{split}.pkl.gz"
                if target.exists(): continue
                t0 = time.monotonic(); ctx = core.make_ctx(seed, split, scratch, cache); test = core.TestSet(ctx); gd = G.graph_for(ctx, 10, False); anchors = C.anchor_paths(ctx); geo = P.geometry_for(ctx); recs = {}
                for budget in budgets:
                    labelled = core.random_labelled(ctx, budget, 0); geom = C.geometry(sim, gd, cache, anchors, test.img1, test.img2, {x for j in labelled for x in (ctx.info[j]["img1"], ctx.info[j]["img2"])}); res = {}
                    for c in cs:
                        for b in betas:
                            for a in amps:
                                m = P.fit_gpg(ctx, labelled, seed, dict(select="fixed", kind=args.kind, c=c, beta=b, aw=args.aw, amp=(dict(zip(P.ACTIVE, a)) if isinstance(a, tuple) else {k: a for k in P.ACTIVE})), geo=geo).predict(test); res[(args.kind, c, b, a)] = (m["mu"].astype(np.float32), m["var"].astype(np.float32))
                    recs[(budget, 0)] = {"test": C.test_arrays(test, geom), "grid": res}
                C.save_cell(target, {"seed": seed, "split": split, "recs": recs}); print(f"seed {seed} split {split}: {time.monotonic() - t0:.0f}s", flush=True)


if __name__ == "__main__": main()
