#!/usr/bin/env python3
"""Recipe-search runner: learners on identical random label sets (paired), standard endpoint + anchor-far stratum + per-type metrics + per-judgment scores.

  python3 rc_run.py --learners lap3_k10_st4,rc_fullg --seeds 9300-9309 --splits A,B --draws 2 --out results/rc/dev1 [--shard 0/4]

One JSON per (seed, split): {"meta": {anc_min, k, sign, weight of every test judgment}, "records": [one per (budget, draw, learner)]}.  Resumable (existing (budget, draw, learner) cells are skipped).
``far`` = judgments whose two images are both > FAR (cosine distance 1 - cos, centred/normalised SimCLR features) from every typed ideal-image anchor (references, utility, outer-test ideal images),
i.e. ``anc_all_min`` of the red-team ``rt_a1b.py``.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import nd_core as core
import nd_graph_learner as G
import nd_recipe as R
import rt_core as rt

FAR = 0.03


def registry() -> dict:
    out = dict(G.LEARNERS); out.update(R.LEARNERS); out["lap3_k10_st4"] = G.LEARNERS["lap3_k10_st4"]; return out


def parse_seeds(text: str):
    if "-" in text: lo, hi = text.split("-"); return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in text.split(",")]


def anchor_distance(ctx, test, sim) -> np.ndarray:
    """min over the two images of 1 - max cosine to any typed ideal image (same definition as rt_a1b.anc_all_min)."""
    allanc = sorted({str(p) for t in (ctx.exp.references, ctx.exp.utility_images, ctx.exp.test_images) for ps in t.values() for p in ps})
    da, db = 1 - sim.max_sim(list(test.img1), allanc), 1 - sim.max_sim(list(test.img2), allanc); return np.minimum(da, db)


def subset(test, mask) -> SimpleNamespace:
    return SimpleNamespace(sign=test.sign[mask], weight=test.weight[mask], k=test.k[mask])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--learners", required=True); ap.add_argument("--seeds", default="9300-9309"); ap.add_argument("--splits", default="A,B")
    ap.add_argument("--draws", type=int, default=3); ap.add_argument("--budgets", default="10,20,40,60"); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--shard", default="0/1")
    args = ap.parse_args(); torch.set_num_threads(1)
    names = args.learners.split(","); reg = registry(); fns = {n: reg[n] for n in names}; budgets = [int(b) for b in args.budgets.split(",")]
    shard, nshards = (int(x) for x in args.shard.split("/")); args.out.mkdir(parents=True, exist_ok=True); cache = core.feature_cache(); sim = rt.Sim(cache)
    seeds = [s for i, s in enumerate(parse_seeds(args.seeds)) if i % nshards == shard]
    with tempfile.TemporaryDirectory() as scratch:
        for seed in seeds:
            for split in args.splits.split(","):
                target = args.out / f"seed{seed}_{split}.json"; blob = json.loads(target.read_text()) if target.exists() else {"meta": None, "records": []}
                have = {(r["budget"], r["draw"], r["learner"]) for r in blob["records"]}
                if all((b, d, n) in have for b in budgets for d in range(args.draws) for n in names): continue
                started = time.monotonic(); ctx = core.make_ctx(seed, split, scratch, cache); test = core.TestSet(ctx); anc = anchor_distance(ctx, test, sim); far = anc > FAR
                blob["meta"] = {"anc_min": np.round(anc, 6).tolist(), "k": test.k.tolist(), "sign": test.sign.tolist(), "weight": np.round(test.weight, 6).tolist()}
                for budget in budgets:
                    for draw in range(args.draws):
                        labelled = core.random_labelled(ctx, budget, draw)
                        for name in names:
                            if (budget, draw, name) in have: continue
                            predictor = fns[name](ctx, labelled, seed * 7 + draw); d = np.asarray(predictor(test), dtype=float)
                            rec = {"seed": seed, "split": split, "budget": budget, "draw": draw, "learner": name, **core.metrics_from_d(d, test), "n_far": int(far.sum())}
                            if far.sum() >= 2:
                                m = core.metrics_from_d(d[far], subset(test, far)); rec.update({f"far_{k}": v for k, v in m.items() if k in ("acc", "ll", "auc")})
                            rec["d"] = np.round(d, 4).tolist(); blob["records"].append(rec)
                    target.write_text(json.dumps(blob))
                print(f"seed {seed} split {split}: {len(names)} learners in {time.monotonic() - started:.0f}s", flush=True)


if __name__ == "__main__": main()
