#!/usr/bin/env python3
"""Runner of the temporal-graph study: learners (nd_temporal_graph + every earlier registry learner) on identical random label sets, standard endpoint plus the red-team strata.

  python3 tg_run.py --learners lap3_k10_st4,tg_u2 --seeds 9200-9219 --splits A,B --draws 1 --out results/tg/dev1 [--shard i/n]
  python3 tg_run.py ... --splits blocked        # red-team time-blocked split (rt_blocked.make_blocked_ctx; one context per seed)

One JSON list per (seed, split); resumable per (budget, draw, learner).  Record = nd_core.metrics_from_d (standard endpoint) plus
  far_*  : the same metrics on the test judgments whose images are BOTH more than FAR (0.03) cosine distance from every typed ideal-image anchor ("anchor-far" endpoint, rt_a1b.py definition)
  near_* : the complement (at least one image within 0.03 of an anchor)
  n_knn / n_temporal : sizes of the edge sets of the learner in this context.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
import types
from pathlib import Path

import numpy as np
import torch

import nd_core as core
import nd_graph_learner as G
import nd_learners as L
import nd_temporal_graph as T
import rt_blocked as rb
import rt_core as rt

FAR = 0.03


def registry() -> dict:
    return {**L.LEARNERS, **G.LEARNERS, **T.LEARNERS}


def seeds_of(text):
    if "-" in text: lo, hi = text.split("-"); return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in text.split(",")]


def anchor_distance(sim: rt.Sim, ctx, test) -> np.ndarray:
    """min over the two images of a test judgment of (1 - cos) to the nearest typed ideal image (references, utility, outer-test ideal images), as ``rt_a1b.anc_all_min``."""
    anchors = sorted({str(p) for t in (ctx.exp.references, ctx.exp.utility_images, ctx.exp.test_images) for ps in t.values() for p in ps})
    return np.minimum(1 - sim.max_sim(list(test.img1), anchors), 1 - sim.max_sim(list(test.img2), anchors))


def subset(test, mask):
    return types.SimpleNamespace(sign=test.sign[mask], weight=test.weight[mask], k=test.k[mask])


def stratum_metrics(d, test, far_mask) -> dict:
    out = {}
    for tag, m in (("far", far_mask), ("near", ~far_mask)):
        if m.sum() >= 2:
            mm = core.metrics_from_d(d[m], subset(test, m)); out.update({f"{tag}_auc": mm["auc"], f"{tag}_acc": mm["acc"], f"{tag}_ll": mm["ll"]})
        else: out.update({f"{tag}_auc": float("nan"), f"{tag}_acc": float("nan"), f"{tag}_ll": float("nan")})
        out[f"{tag}_n"] = int(m.sum())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--learners", required=True); ap.add_argument("--seeds", default="9200-9219"); ap.add_argument("--splits", default="A,B")
    ap.add_argument("--draws", type=int, default=1); ap.add_argument("--budgets", default="10,20,40,60"); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--shard", default="0/1"); ap.add_argument("--block", type=int, default=100)
    ap.add_argument("--buffer", type=int, default=25); args = ap.parse_args(); torch.set_num_threads(1)
    names = args.learners.split(","); reg = registry(); fns = {n: reg[n] for n in names}; budgets = [int(b) for b in args.budgets.split(",")]; shard, nshards = (int(x) for x in args.shard.split("/"))
    args.out.mkdir(parents=True, exist_ok=True); cache = core.feature_cache(); sim = rt.Sim(cache)
    allowed = set(core.TG_DEV_SEEDS) | set(core.TG_CONFIRM) | set(range(9220, 9250)) | set(range(9285, 9300))
    with tempfile.TemporaryDirectory() as scratch:
        for seed in [s for i, s in enumerate(seeds_of(args.seeds)) if i % nshards == shard]:
            assert seed in allowed, f"seed {seed} is not in this study's block"
            for split in args.splits.split(","):
                target = args.out / f"seed{seed}_{split}.json"; done = json.loads(target.read_text()) if target.exists() else []; have = {(r["budget"], r["draw"], r["learner"]) for r in done}
                if all((b, d, n) in have for b in budgets for d in range(args.draws) for n in names): continue
                started = time.monotonic()
                ctx = rb.make_blocked_ctx(seed, scratch, cache, args.block, args.buffer) if split == "blocked" else core.make_ctx(seed, split, scratch, cache)
                test = core.TestSet(ctx); far = anchor_distance(sim, ctx, test) > FAR; extra = {"n_test": int(len(test.k)), "far_share": float(far.mean())}
                if split == "blocked": extra.update({k: v for k, v in ctx.blocked.items() if k != "test_blocks"})
                for budget in budgets:
                    for draw in range(args.draws):
                        labelled = core.random_labelled(ctx, budget, draw)
                        for name in names:
                            if (budget, draw, name) in have: continue
                            d = np.asarray(fns[name](ctx, labelled, seed * 7 + draw)(test), dtype=float)
                            rec = {"seed": seed, "split": split, "budget": budget, "draw": draw, "learner": name, **core.metrics_from_d(d, test), **stratum_metrics(d, test, far), **extra}
                            if name in T.SPEC_OF: rec.update(T.edge_counts(ctx, name))
                            done.append(rec)
                    target.write_text(json.dumps(done))
                print(f"seed {seed} split {split}: {len(names)} learners in {time.monotonic() - started:.0f}s", flush=True)


if __name__ == "__main__": main()
