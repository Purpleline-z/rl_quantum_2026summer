"""Stage-1 runner of the graph-GP study: per (seed, split, budget, draw) label set, fit the requested learners and store the per-judgment predictions (mean / variance of the preference for GP
learners, logit gap for MLP learners), the fit time and the geometry features of the test judgments (distance to the nearest typed anchor / labelled image, graph density).  One gzip-pickle per
(seed, split); resumable.

  python3 gp_run.py --seeds 11000-11019 --splits A,B --out results/gp/dev1 --learners lap3_k10_st4,aw8_all_st2lr0.5,gp_fixed,gp_evc,gp_cv [--budgets 10,20,40,60] [--draws 1 --draw0 0]
                    [--initial random] [--shard i/n]
"""
import argparse, tempfile, time, os
from pathlib import Path
import numpy as np, torch
import nd_core as core, nd_graph_learner as G, nd_calibration as C, nd_gpg as P, rt_core as rt

GP_SPECS = {}
for _sel in ("fixed", "ev", "evc", "cv"):
    _s = {"fixed": "fixed", "ev": "evidence", "evc": "evidence_cond", "cv": "cv"}[_sel]
    GP_SPECS[f"gp_{_sel}"] = dict(cfg=dict(select=_s), shuffle=False)
    GP_SPECS[f"gp_{_sel}_noG"] = dict(cfg=dict(select=_s, betas=(0.0,), beta=0.0), shuffle=False)          # kernel without the graph deformation
    GP_SPECS[f"gp_{_sel}_shuf"] = dict(cfg=dict(select=_s), shuffle=True)                                   # shuffled graph (node identities permuted)
    GP_SPECS[f"gp_{_sel}_mix"] = dict(cfg=dict(select=_s, kind="mix", kinds=("mix",), cs=((0.1, 0.4, 0.5), (0.15, 0.7, 0.5), (0.1, 0.4, 0.8), (0.25, 0.7, 0.5))), shuffle=False)   # multi-scale RBF mixture
    GP_SPECS[f"gp_{_sel}_arccos"] = dict(cfg=dict(select=_s, kind="arccos", kinds=("arccos",), cs=(1.0,)), shuffle=False)   # NNGP kernel of the head's own function class


GP_SPECS["gp_evt"] = dict(cfg=dict(select="evidence", per_type_config=True, cs=(0.1, 0.15, 0.25, 0.4, 0.7), betas=(0.0, 0.03, 0.1, 0.3, 1.0, 3.0), amps=(0.1, 0.3, 1.0, 3.0, 10.0, 30.0)), shuffle=False)   # per-type kernel shape by evidence
GP_SPECS["gp_evct"] = dict(cfg=dict(select="evidence_cond", per_type_config=True, cs=(0.1, 0.15, 0.25, 0.4, 0.7), betas=(0.0, 0.03, 0.1, 0.3, 1.0, 3.0), amps=(0.1, 0.3, 1.0, 3.0, 10.0, 30.0)), shuffle=False)


def seeds_of(text):
    if "-" in text: lo, hi = text.split("-"); return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in text.split(",")]


def run_learner(name, ctx, labelled, seed0, test, geos):
    t0 = time.monotonic()
    if name in GP_SPECS:
        spec = GP_SPECS[name]; geo = geos(spec["shuffle"]); model = P.fit_gpg(ctx, labelled, seed0, dict(spec["cfg"]), geo=geo); fit_s = time.monotonic() - t0; pr = model.predict(test)
        info = {k: v for k, v in model.info.items() if k != "seconds"}; info["fit_seconds"] = fit_s; info["amp"] = {k: v["amp"] for k, v in model.per_type.items()}
        return {"mu": pr["mu"].astype(np.float32), "var": pr["var"].astype(np.float32), "seconds": time.monotonic() - t0, "info": info}
    d = np.asarray(G.LEARNERS[name](ctx, labelled, seed0)(test), dtype=np.float32); return {"mu": d, "var": None, "seconds": time.monotonic() - t0, "info": {}}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", required=True); ap.add_argument("--splits", default="A,B"); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--learners", required=True); ap.add_argument("--budgets", default="10,20,40,60"); ap.add_argument("--draws", type=int, default=1); ap.add_argument("--draw0", type=int, default=0)
    ap.add_argument("--initial", default="groups"); args = ap.parse_args(); torch.set_num_threads(1); os.environ["ND_INITIAL"] = args.initial
    names = args.learners.split(","); budgets = [int(b) for b in args.budgets.split(",")]; shard, n = (int(x) for x in args.shard.split("/")); args.out.mkdir(parents=True, exist_ok=True)
    cache = core.feature_cache(); sim = rt.Sim(cache)
    with tempfile.TemporaryDirectory() as scratch:
        for seed in [s for i, s in enumerate(seeds_of(args.seeds)) if i % n == shard]:
            for split in args.splits.split(","):
                target = args.out / f"seed{seed}_{split}.pkl.gz"
                if target.exists(): continue
                t0 = time.monotonic(); ctx = core.make_ctx(seed, split, scratch, cache); test = core.TestSet(ctx); gd = G.graph_for(ctx, 10, False); anchors = C.anchor_paths(ctx); recs = {}
                geos = lambda shuf, ctx=ctx: P.geometry_for(ctx, shuf)
                for budget in budgets:
                    for draw in range(args.draw0, args.draw0 + args.draws):
                        labelled = core.random_labelled(ctx, budget, draw); seed0 = seed * 7 + draw
                        geom = C.geometry(sim, gd, cache, anchors, test.img1, test.img2, {x for j in labelled for x in (ctx.info[j]["img1"], ctx.info[j]["img2"])})
                        recs[(budget, draw)] = {"n_labelled": len(labelled), "test": C.test_arrays(test, geom), "learners": {nm: run_learner(nm, ctx, labelled, seed0, test, geos) for nm in names}}
                C.save_cell(target, {"seed": seed, "split": split, "initial": args.initial, "learners": names, "recs": recs}); print(f"seed {seed} split {split}: {time.monotonic() - t0:.0f}s", flush=True)


if __name__ == "__main__": main()
