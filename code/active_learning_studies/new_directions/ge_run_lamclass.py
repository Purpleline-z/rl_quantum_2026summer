"""EXPLORATORY (dev seeds): does the Laplacian term hurt type classification, and how fast with lambda?  Inductive graph, reference-only anchors, 60 random judgments, 3 draws."""
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

LAMS = (0.0, 0.3, 1.0, 3.0, 10.0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", required=True); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--shard", default="0/1"); a = ap.parse_args()
    lo, hi = a.seeds.split("-"); seeds = range(int(lo), int(hi) + 1); torch.set_num_threads(1); shard, n = (int(x) for x in a.shard.split("/")); a.out.mkdir(parents=True, exist_ok=True); cache = core.feature_cache()
    with tempfile.TemporaryDirectory() as scratch:
        for i, seed in enumerate(seeds):
            target = a.out / f"seed{seed}.json"
            if i % n != shard or target.exists(): continue
            ctx = core.make_ctx(seed, "B", scratch, cache); ref, test, util = gc.split_arrays(ctx); y = test[1]; fn = core.identity_features(ctx); gd = gc.build_graph(ctx, 10, False); rows = []
            for draw in range(3):
                lab = core.random_labelled(ctx, 60, draw); s = seed * 7 + draw
                for lam in LAMS:
                    head = gh.fit_head(ctx, lab, s, gd if lam else None, fn, gd.x if lam else None, 512, lap=lam, all_ideal=False); pred = gh.winrate_vectors(ctx, head, test[0]).argmax(1)
                    rows.append({"seed": seed, "lam": lam, "draw": draw, "acc": float((pred == y).mean())})
            target.write_text(json.dumps(rows)); print("seed", seed, flush=True)


if __name__ == "__main__":
    main()
