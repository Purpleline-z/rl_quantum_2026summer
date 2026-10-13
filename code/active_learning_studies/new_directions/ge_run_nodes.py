"""EXPLORATORY mechanism run (dev seeds only): which graph nodes carry the Laplacian-regularisation gain?  lap3_k10 vs aw8_all on the preference endpoint with node sets
  ideal          ideal images only
  train          ideal + the images of the labelled judgments (initial + selected)
  cand           ideal + the images of the candidate pool that are not in a labelled judgment
  groups         ideal + every pair-group image (labelled + candidate)   [= dose series 'out' at p = 0]
  all            standard graph (all trajectory frames)
Node sets that depend on the labelled draw are rebuilt per draw."""
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
import ge_run_dose as gd_


def parse_seeds(text):
    if "-" in text: lo, hi = text.split("-"); return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in text.split(",")]


def images_of(ctx, judgments) -> set[str]:
    return {str(ctx.info[j][k]) for j in judgments for k in ("img1", "img2")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", required=True); ap.add_argument("--splits", default="A,B"); ap.add_argument("--budgets", default="20,60"); ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--shard", default="0/1"); a = ap.parse_args()
    torch.set_num_threads(1); shard, n = (int(x) for x in a.shard.split("/")); a.out.mkdir(parents=True, exist_ok=True); cache = core.feature_cache(); budgets = [int(b) for b in a.budgets.split(",")]
    with tempfile.TemporaryDirectory() as scratch:
        for i, seed in enumerate(parse_seeds(a.seeds)):
            if i % n != shard: continue
            for split in a.splits.split(","):
                target = a.out / f"seed{seed}_{split}.json"
                if target.exists(): continue
                t0 = time.monotonic(); ctx = core.make_ctx(seed, split, scratch, cache); test = core.TestSet(ctx); fn = core.identity_features(ctx); rows = []; pair_imgs = gd_.pair_group_images(ctx)
                pool_imgs = images_of(ctx, ctx.pool); full = gh.make_graph(ctx, 10)
                for budget in budgets:
                    for draw in range(a.draws):
                        lab = core.random_labelled(ctx, budget, draw); s = seed * 7 + draw; lab_imgs = images_of(ctx, lab); cand_imgs = pool_imgs - lab_imgs
                        sets = {"ideal": lambda ps: [p for p in ps if "Trajectories" not in p], "train": lambda ps: [p for p in ps if "Trajectories" not in p or p in lab_imgs], "cand": lambda ps: [p for p in ps if "Trajectories" not in p or p in cand_imgs],
                                "groups": lambda ps: [p for p in ps if "Trajectories" not in p or p in pair_imgs]}
                        res = {"base": core.head_d(gh.fit_head(ctx, lab, s, None, fn, None, 512, 0.0, True), fn, test), "all": core.head_d(gh.fit_head(ctx, lab, s, full, fn, full.x, 512, 3.0, True), fn, test)}
                        nn_nodes = {"all": len(full.paths)}
                        for name, keep in sets.items():
                            g = gh.make_graph(ctx, 10, keep=keep); nn_nodes[name] = len(g.paths); res[name] = core.head_d(gh.fit_head(ctx, lab, s, g, fn, g.x, 512, 3.0, True), fn, test)
                        for name, d in res.items(): rows.append({"seed": seed, "split": split, "budget": budget, "draw": draw, "variant": name, "n_nodes": nn_nodes.get(name, 0), **core.metrics_from_d(d, test)})
                target.write_text(json.dumps(rows)); print(f"seed {seed} split {split}: {time.monotonic() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
