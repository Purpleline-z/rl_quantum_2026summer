"""Audit 2 runner: remove from the GRAPH every node that lies within cosine distance delta (1 - cos of centred, normalised features) of any held-out image, keep all labelled judgments.

Why only the graph: strict de-duplication of the labelled data is impossible here (median distance of a training image to its nearest held-out image is 0.0007 (A) / 0.0013 (B);
at delta = 0.002 only 13-22 % of the candidate pool and 4 of ~30 initial judgments would survive), so this audit asks the narrower question whether the Laplacian gain depends on unlabelled
near-duplicates of the test images being graph nodes.
  python3 rt_a2.py --seeds 8800-8819 --splits A,B --out results/rt/a2 [--shard i/n] [--deltas 0,0.003,0.01]
"""
import argparse, json, tempfile, time
from pathlib import Path
import numpy as np, torch
import nd_core as core, nd_graph_learner as G, rt_core as rt, rt_blocked as rb

DEFAULT = "aw8_all_st2lr0.5,lap3_k10,lap3_k10_st4"


def seeds_of(text):
    if "-" in text: lo, hi = text.split("-"); return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in text.split(",")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="8800-8819"); ap.add_argument("--splits", default="A,B"); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--learners", default=DEFAULT); ap.add_argument("--budgets", default="20,40"); ap.add_argument("--draws", type=int, default=2); ap.add_argument("--deltas", default="0,0.003,0.01"); args = ap.parse_args(); torch.set_num_threads(1)
    names = args.learners.split(","); budgets = [int(b) for b in args.budgets.split(",")]; deltas = [float(x) for x in args.deltas.split(",")]; shard, n = (int(x) for x in args.shard.split("/")); args.out.mkdir(parents=True, exist_ok=True)
    cache = core.feature_cache(); sim = rt.Sim(cache); trajectory = [p for p in sorted(cache) if rt.run_of(p) is not None]
    with tempfile.TemporaryDirectory() as scratch:
        for seed in [s for i, s in enumerate(seeds_of(args.seeds)) if i % n == shard]:
            for split in args.splits.split(","):
                target = args.out / f"seed{seed}_{split}.json"
                if target.exists(): continue
                t0 = time.monotonic(); recs = []
                for delta in deltas:
                    ctx = core.make_ctx(seed, split, scratch, cache); test = core.TestSet(ctx); held = sorted(rt.heldout_images(ctx))
                    zone = [p for p, dm in zip(trajectory, 1 - sim.max_sim(trajectory, held)) if dm <= delta] if delta > 0 else []
                    if zone: rb.install_exclusion(ctx, zone)
                    nodes = list(G.graph_for(ctx, 10, False).paths)
                    for budget in budgets:
                        for draw in range(args.draws):
                            labelled = core.random_labelled(ctx, budget, draw)
                            for name in names:
                                d = rt.predict(ctx, name, labelled, seed * 7 + draw, test)
                                recs.append({"seed": seed, "split": split, "delta": delta, "budget": budget, "draw": draw, "learner": name, **core.metrics_from_d(d, test), "n_nodes": len(nodes), "n_zone": len(zone)})
                target.write_text(json.dumps(recs)); print(f"seed {seed} split {split}: {time.monotonic() - t0:.0f}s", flush=True)


if __name__ == "__main__": main()
