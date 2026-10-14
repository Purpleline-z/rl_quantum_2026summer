#!/usr/bin/env python3
"""Runner of the synthetic-anchor study: learners (nd_synthetic + every earlier registry learner) on identical random label sets; standard endpoint, anchor-far / anchor-near strata (as tg_run.py) and the
red-team time-blocked split.
  python3 syn_run.py --learners lap3_k10_st4,sy_fix16 --seeds 11100-11119 --splits A,B --draws 1 --out results/syn/dev1 [--shard i/n]     (--splits blocked for the time-blocked contexts)
Resumable per (budget, draw, learner); one JSON list per (seed, split)."""
import argparse, json, tempfile, time
from pathlib import Path
import numpy as np, torch
import nd_core as core, nd_graph_learner as G, nd_learners as L, nd_synthetic as Y, rt_blocked as rb, rt_core as rt, tg_run


def registry():
    import syn_specs   # registers the study's learners (dev / frozen) into nd_synthetic
    return {**L.LEARNERS, **G.LEARNERS, **Y.LEARNERS}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--learners", required=True); ap.add_argument("--seeds", default="11100-11119"); ap.add_argument("--splits", default="A,B"); ap.add_argument("--draws", type=int, default=1)
    ap.add_argument("--budgets", default="10,20,40,60"); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--shard", default="0/1"); ap.add_argument("--block", type=int, default=100); ap.add_argument("--buffer", type=int, default=25)
    args = ap.parse_args(); torch.set_num_threads(1); names = args.learners.split(","); reg = registry(); fns = {n: reg[n] for n in names}; budgets = [int(b) for b in args.budgets.split(",")]; shard, nshards = (int(x) for x in args.shard.split("/"))
    args.out.mkdir(parents=True, exist_ok=True); cache = core.feature_cache(); sim = rt.Sim(cache)
    with tempfile.TemporaryDirectory() as scratch:
        for seed in [s for i, s in enumerate(tg_run.seeds_of(args.seeds)) if i % nshards == shard]:
            assert seed in core.SA_SEEDS_ALL, f"seed {seed} is not in this study's block"
            for split in args.splits.split(","):
                target = args.out / f"seed{seed}_{split}.json"; done = json.loads(target.read_text()) if target.exists() else []; have = {(r["budget"], r["draw"], r["learner"]) for r in done}
                if all((b, d, n) in have for b in budgets for d in range(args.draws) for n in names): continue
                started = time.monotonic(); ctx = rb.make_blocked_ctx(seed, scratch, cache, args.block, args.buffer) if split == "blocked" else core.make_ctx(seed, split, scratch, cache)
                test = core.TestSet(ctx); far = tg_run.anchor_distance(sim, ctx, test) > tg_run.FAR; inb = tg_run.in_block_mask(ctx, test, args.block) if split == "blocked" else None; extra = {"n_test": int(len(test.k)), "far_share": float(far.mean())}
                if split == "blocked": extra.update({k: v for k, v in ctx.blocked.items() if k != "test_blocks"}); extra["inb_share"] = float(inb.mean())
                for budget in budgets:
                    for draw in range(args.draws):
                        labelled = core.random_labelled(ctx, budget, draw)
                        for name in names:
                            if (budget, draw, name) in have: continue
                            d = np.asarray(fns[name](ctx, labelled, seed * 7 + draw)(test), dtype=float)
                            done.append({"seed": seed, "split": split, "budget": budget, "draw": draw, "learner": name, **core.metrics_from_d(d, test), **tg_run.stratum_metrics(d, test, far, inb), **extra})
                    target.write_text(json.dumps(done))
                print(f"seed {seed} split {split}: {len(names)} learners in {time.monotonic() - started:.0f}s", flush=True)


if __name__ == "__main__": main()
