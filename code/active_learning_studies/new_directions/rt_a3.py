"""Audit 3 runner: learners on time-blocked contexts (see rt_blocked.py).  One JSON list per seed; resumable.
  python3 rt_a3.py --seeds 8800-8819 --out results/rt/a3 [--shard i/n] [--learners ...] [--block 100 --buffer 25]
Records: seed, budget, draw, learner, metrics_from_d, test novelty summary (median / max of nov_train_max), blocked-split sizes."""
import argparse, json, tempfile, time
from pathlib import Path
import numpy as np, torch
import nd_core as core, nd_graph_learner as G, rt_core as rt, rt_blocked as rb

DEFAULT = "aw8_all,aw8_all_st2lr0.5,lap3_k10,lap3_k10_st4,lap3_k10_st4_shuf"


def seeds_of(text):
    if "-" in text: lo, hi = text.split("-"); return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in text.split(",")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="8800-8819"); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--shard", default="0/1"); ap.add_argument("--learners", default=DEFAULT)
    ap.add_argument("--budgets", default="10,20,40,60"); ap.add_argument("--draws", type=int, default=3); ap.add_argument("--block", type=int, default=100); ap.add_argument("--buffer", type=int, default=25); args = ap.parse_args(); torch.set_num_threads(1)
    names = args.learners.split(","); budgets = [int(b) for b in args.budgets.split(",")]; shard, n = (int(x) for x in args.shard.split("/")); args.out.mkdir(parents=True, exist_ok=True); cache = core.feature_cache(); sim = rt.Sim(cache)
    with tempfile.TemporaryDirectory() as scratch:
        for seed in [s for i, s in enumerate(seeds_of(args.seeds)) if i % n == shard]:
            target = args.out / f"seed{seed}_blocked.json"
            if target.exists(): continue
            t0 = time.monotonic(); ctx = rb.make_blocked_ctx(seed, scratch, cache, args.block, args.buffer); test = core.TestSet(ctx); nodes = list(G.graph_for(ctx, 10, False).paths); recs = []
            for budget in budgets:
                for draw in range(args.draws):
                    labelled = core.random_labelled(ctx, budget, draw); nov = rt.novelty_table(sim, ctx, labelled, test, nodes)
                    for name in names:
                        d = rt.predict(ctx, name, labelled, seed * 7 + draw, test)
                        recs.append({"seed": seed, "split": "blocked", "budget": budget, "draw": draw, "learner": name, **core.metrics_from_d(d, test), "nov_train_max_median": float(nov.nov_train_max.median()), "nov_train_max_q90": float(nov.nov_train_max.quantile(.9)),
                                     "nov_node_max_median": float(nov.nov_node_max.median()), **{k: v for k, v in ctx.blocked.items() if k != "test_blocks"}})
            target.write_text(json.dumps(recs)); print(f"seed {seed}: {time.monotonic() - t0:.0f}s", flush=True)


if __name__ == "__main__": main()
