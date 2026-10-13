"""Dose-response of the number of UNLABELLED images in the graph (preference endpoint, lap3_k10 vs aw8_all, identical random label sets).

Series 'all': nodes = ideal images (always) + a fixed per-seed random fraction p of the allowed trajectory images (nested across p).
Series 'out': nodes = ideal images + every allowed trajectory image that occurs in a pair group (labelled-group images and candidates) + fraction p of the trajectory images that occur in NO pair group
(the purely unlabelled frames).  p=1 of both series is the standard lap3_k10 graph.  Control: p=1 with the node identities permuted (shuffled)."""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

import nd_core as core
import ge_heads as gh

FRACTIONS = (0.0, 0.25, 0.5, 1.0)


def parse_seeds(text):
    if "-" in text: lo, hi = text.split("-"); return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in text.split(",")]


def pair_group_images(ctx) -> set[str]:
    return {str(x) for g in ctx.exp.groups for r in ctx.exp.groups[g].itertuples() for x in (r.resolved_img1, r.resolved_img2)}


def keep_fn(ctx, series: str, p: float):
    pair_imgs = pair_group_images(ctx)
    def keep(paths):
        traj = [x for x in paths if "Trajectories" in x]; other = [x for x in paths if "Trajectories" not in x]; rng = np.random.default_rng(ctx.seed * 31 + 7)
        if series == "all": pool = traj; fixed = []
        else: pool = [x for x in traj if x not in pair_imgs]; fixed = [x for x in traj if x in pair_imgs]
        order = [pool[i] for i in rng.permutation(len(pool))]; return other + fixed + order[: int(round(p * len(order)))]
    return keep


def variants(ctx):
    out = {"base": None}
    for p in FRACTIONS: out[f"all_p{int(p * 100)}"] = ("all", p, False)
    out["all_p10"] = ("all", 0.1, False)   # exploratory extra level
    for p in FRACTIONS[:-1]: out[f"out_p{int(p * 100)}"] = ("out", p, False)
    out["all_p100_shuf"] = ("all", 1.0, True); out["all_p0_shuf"] = ("all", 0.0, True); return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", required=True); ap.add_argument("--splits", default="A,B"); ap.add_argument("--budgets", default="20,60"); ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--shard", default="0/1"); ap.add_argument("--lam", type=float, default=3.0); a = ap.parse_args()
    torch.set_num_threads(1); shard, n = (int(x) for x in a.shard.split("/")); a.out.mkdir(parents=True, exist_ok=True); cache = core.feature_cache(); budgets = [int(b) for b in a.budgets.split(",")]
    with tempfile.TemporaryDirectory() as scratch:
        for i, seed in enumerate(parse_seeds(a.seeds)):
            if i % n != shard: continue
            for split in a.splits.split(","):
                target = a.out / f"seed{seed}_{split}.json"
                if target.exists(): continue
                t0 = time.monotonic(); ctx = core.make_ctx(seed, split, scratch, cache); test = core.TestSet(ctx); fn = core.identity_features(ctx); rows = []; graphs = {}
                for name, spec in variants(ctx).items():
                    if spec is not None: series, p, shuf = spec; graphs[name] = gh.make_graph(ctx, 10, shuffle=shuf, keep=keep_fn(ctx, series, p))
                for budget in budgets:
                    for draw in range(a.draws):
                        lab = core.random_labelled(ctx, budget, draw); s = seed * 7 + draw
                        for name in variants(ctx):
                            gd = graphs.get(name); head = gh.fit_head(ctx, lab, s, gd, fn, None if gd is None else gd.x, 512, lap=0.0 if gd is None else a.lam, all_ideal=True)
                            rows.append({"seed": seed, "split": split, "budget": budget, "draw": draw, "variant": name, "n_nodes": 0 if gd is None else len(gd.paths), "n_edges": 0 if gd is None else int(gd.n_edges), **core.metrics_from_d(core.head_d(head, fn, test), test)})
                target.write_text(json.dumps(rows)); print(f"seed {seed} split {split}: {time.monotonic() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
