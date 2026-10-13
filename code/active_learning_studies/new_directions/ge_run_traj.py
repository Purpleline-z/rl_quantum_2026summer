"""EXPLORATORY (dev seeds): can the preference gain be kept while the classification loss is avoided if the Laplacian penalty acts only between TRAJECTORY frames (graph without ideal-image nodes)?
Classification (reference-only anchors, 60 judgments, inductive; aw8_ref vs lap3 on the full graph vs lap3 on the trajectory-only graph) and preference endpoint (aw8_all vs the same two graphs), identical random label sets."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

import nd_core as core
import ge_classify as gc
import ge_heads as gh

traj_only = lambda ps: [p for p in ps if "Trajectories" in p]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", required=True); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--shard", default="0/1"); a = ap.parse_args()
    lo, hi = a.seeds.split("-"); seeds = range(int(lo), int(hi) + 1); torch.set_num_threads(1); shard, n = (int(x) for x in a.shard.split("/")); a.out.mkdir(parents=True, exist_ok=True); cache = core.feature_cache()
    with tempfile.TemporaryDirectory() as scratch:
        for i, seed in enumerate(seeds):
            target = a.out / f"seed{seed}.json"
            if i % n != shard or target.exists(): continue
            rows = []; ctx = core.make_ctx(seed, "B", scratch, cache); ref, test, util = gc.split_arrays(ctx); y = test[1]; fn = core.identity_features(ctx)
            full = gh.make_graph(ctx, 10); tro = gh.make_graph(ctx, 10, keep=traj_only)
            for draw in range(3):
                lab = core.random_labelled(ctx, 60, draw); s = seed * 7 + draw
                for name, gd in (("aw8_ref", None), ("lap3_full", full), ("lap3_trajonly", tro)):
                    head = gh.fit_head(ctx, lab, s, gd, fn, None if gd is None else gd.x, 512, lap=0.0 if gd is None else 3.0, all_ideal=False); rows.append({"endpoint": "class", "split": "B", "seed": seed, "budget": 60, "draw": draw, "arm": name, "acc": float((gh.winrate_vectors(ctx, head, test[0]).argmax(1) == y).mean())})
            for split in ("A", "B"):
                c2 = core.make_ctx(seed, split, scratch, cache); tst = core.TestSet(c2); f2 = core.identity_features(c2); g_full = gh.make_graph(c2, 10); g_tro = gh.make_graph(c2, 10, keep=traj_only)
                for budget in (20, 60):
                    for draw in range(3):
                        lab = core.random_labelled(c2, budget, draw); s = seed * 7 + draw
                        for name, gd in (("aw8_all", None), ("lap3_full", g_full), ("lap3_trajonly", g_tro)):
                            head = gh.fit_head(c2, lab, s, gd, f2, None if gd is None else gd.x, 512, lap=0.0 if gd is None else 3.0, all_ideal=True); rows.append({"endpoint": "pref", "split": split, "seed": seed, "budget": budget, "draw": draw, "arm": name, **core.metrics_from_d(core.head_d(head, f2, tst), tst)})
            target.write_text(json.dumps(rows)); print("seed", seed, flush=True)


if __name__ == "__main__":
    main()
