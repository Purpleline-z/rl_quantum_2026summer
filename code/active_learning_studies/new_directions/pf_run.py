"""Runner of the physics-features study: per-judgment held-out predictions of any registered learner on random label sets (identical for every learner: seed, budget, draw), with the distance of every test judgment
to the nearest typed ideal-image anchor (1 - cosine of centred SimCLR features, both images -> min over the two, all anchors of aw8_all as in rt_a1b), so that the standard endpoint and the anchor-far stratum come from
the same file.  One gzip-CSV per (seed, split); resumable per learner (existing learners of a file are skipped).
  python3 pf_run.py --learners lap3_k10_st4,pf_pfa_hs --seeds 10600-10609 --splits A,B --out results/pf/devA [--shard i/n] [--ctx blocked] [--draws 3] [--budgets 10,20,40,60]"""
import argparse, tempfile, time
from pathlib import Path
import numpy as np, pandas as pd, torch
import nd_core as core, nd_graph_learner as G, nd_representation as R, nd_physics as NP, rt_core as rt, nd_calibration as _cal  # noqa: F401 (registers nothing here; keeps import order identical to other runners)

REG = {**rt.REGISTRY, **R.LEARNERS, **NP.LEARNERS}


def seeds_of(text):
    if "-" in text: lo, hi = text.split("-"); return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in text.split(",")]


def anchor_distance(sim, ctx, test):
    allanc = sorted({str(p) for t in (ctx.exp.references, ctx.exp.utility_images, ctx.exp.test_images) for ps in t.values() for p in ps})
    da, db = 1 - sim.max_sim(list(test.img1), allanc), 1 - sim.max_sim(list(test.img2), allanc); return np.minimum(da, db), np.maximum(da, db)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", required=True); ap.add_argument("--splits", default="A,B"); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--learners", required=True); ap.add_argument("--budgets", default="10,20,40,60"); ap.add_argument("--draws", type=int, default=3); ap.add_argument("--ctx", default="standard"); args = ap.parse_args(); torch.set_num_threads(1)
    names = args.learners.split(","); missing = [n for n in names if REG.get(n) is None]; assert not missing, f"unknown learners {missing}"
    budgets = [int(b) for b in args.budgets.split(",")]; shard, n = (int(x) for x in args.shard.split("/")); args.out.mkdir(parents=True, exist_ok=True); cache = core.feature_cache(); sim = rt.Sim(cache)
    splits = ["blocked"] if args.ctx == "blocked" else args.splits.split(",")
    with tempfile.TemporaryDirectory() as scratch:
        for seed in [s for i, s in enumerate(seeds_of(args.seeds)) if i % n == shard]:
            for split in splits:
                target = args.out / f"seed{seed}_{split}.csv.gz"; old = pd.read_csv(target) if target.exists() else None; todo = [m for m in names if old is None or m not in set(old.learner)]
                if not todo: continue
                t0 = time.monotonic()
                if split == "blocked":
                    import rt_blocked as rb; ctx = rb.make_blocked_ctx(seed, scratch, cache)
                else: ctx = core.make_ctx(seed, split, scratch, cache)
                test = core.TestSet(ctx); amin, amax = anchor_distance(sim, ctx, test); rows = []
                for budget in budgets:
                    for draw in range(args.draws):
                        labelled = core.random_labelled(ctx, budget, draw)
                        for name in todo:
                            d = np.asarray(REG[name](ctx, labelled, seed * 7 + draw)(test), dtype=float)
                            rows.append(pd.DataFrame({"seed": seed, "split": split, "budget": budget, "draw": draw, "learner": name, "j": np.arange(len(d)), "d": d, "sign": test.sign, "weight": test.weight, "k": test.k, "anc_min": amin, "anc_max": amax}))
                df = pd.concat(rows) if old is None else pd.concat([old] + rows); tmp = target.with_suffix(".tmp"); df.to_csv(tmp, index=False, compression="gzip"); tmp.rename(target); print(f"seed {seed} split {split} [{','.join(todo)}]: {time.monotonic() - t0:.0f}s", flush=True)


if __name__ == "__main__": main()
