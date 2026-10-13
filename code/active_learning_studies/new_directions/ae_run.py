"""Anchor-exchange runner (idempotent per seed x split file).  Modes
  dose    anchors {0, 10%, 25%, 50%, 100%} of the 116 non-test ideal images (3 random subsets for the partial levels) x pairwise budgets {0, 10, 20, 40, 60} (+ the original 144-anchor reference arm)
  active  36 anchors chosen by rand (3 subsets) / kcg / kcf / typi / amb / htrq and the shuffled-graph controls x budgets {10, 20, 40, 60}
  lam     lambda in {0, 3, 10} x anchors {0, 25%, 100%} x budgets {20, 60}
Same random pairwise label set for every arm of a budget (paired)."""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

import nd_core as core
import nd_graph_learner as gl
import ae_core as ae

LEVELS = (0, 12, 29, 58, 116)   # 0, 10%, 25%, 50%, 100% of 116
BUDGET_DOSE = (0, 10, 20, 40, 60); BUDGET_ACTIVE = (10, 20, 40, 60); BUDGET_LAM = (20, 60)
RULES = ("rand0", "rand1", "rand2", "kcg", "kcg_shuf", "kcf", "typi", "amb", "amb_shuf", "htrq", "htrq_shuf")


def parse_seeds(text):
    if "-" in text: lo, hi = text.split("-"); return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in text.split(",")]


def labelled_for(ctx, budget, draw=0): return list(ctx.initial) if budget == 0 else core.random_labelled(ctx, budget, draw)


def configs(ctx, mode, gd, gs, rules, n_active):
    uni = ae.universe(ctx, "no_test"); out = {}
    if mode == "dose":
        for n in LEVELS:
            for sid in ((0,) if n in (0, 116) else (0, 1, 2)):
                out[f"n{n}_s{sid}"] = (ae.random_subset(ctx, uni, n, sid) if n < 116 else uni, n)
        out["all144"] = (ae.universe(ctx, "all144"), 144)
    elif mode == "active":
        for r in rules:
            sid = int(r[-1]) if r.startswith("rand") else 0; name = "rand" if r.startswith("rand") else r
            out[r] = (ae.active_subset(ctx, uni, name, n_active, gd, gs, sid), n_active)
    elif mode == "lam":
        for n in (0, 29, 116): out[f"n{n}"] = (ae.random_subset(ctx, uni, n, 0) if n < 116 else uni, n)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", required=True); ap.add_argument("--seeds", required=True); ap.add_argument("--splits", default="A,B"); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", default="0/1"); ap.add_argument("--rules", default=",".join(RULES)); ap.add_argument("--n-active", type=int, default=36); ap.add_argument("--draws", type=int, default=1); a = ap.parse_args()
    torch.set_num_threads(1); shard, nsh = (int(x) for x in a.shard.split("/")); a.out.mkdir(parents=True, exist_ok=True); cache = core.feature_cache(); rules = a.rules.split(",")
    budgets = {"dose": BUDGET_DOSE, "active": BUDGET_ACTIVE, "lam": BUDGET_LAM}[a.mode]
    with tempfile.TemporaryDirectory() as scratch:
        for i, seed in enumerate(parse_seeds(a.seeds)):
            if i % nsh != shard: continue
            for split in a.splits.split(","):
                target = a.out / f"seed{seed}_{split}.json"
                if target.exists(): continue
                t0 = time.monotonic(); ctx = core.make_ctx(seed, split, scratch, cache); test = core.TestSet(ctx); gd = gl.graph_for(ctx, 10); gs = gl.graph_for(ctx, 10, True); rows = []
                cfg = configs(ctx, a.mode, gd, gs, rules, a.n_active)
                for budget in budgets:
                    for draw in range(a.draws):
                        lab = labelled_for(ctx, budget, draw); s = seed * 7 + draw
                        for name, (anchors, n) in cfg.items():
                            for lam in ((0.0, 3.0, 10.0) if a.mode == "lam" else (3.0,)):
                                head = ae.fit_head(ctx, lab, s, gd, anchors, lap=lam); rows.append({"seed": seed, "split": split, "mode": a.mode, "budget": budget, "draw": draw, "arm": name, "n_anchors": n, "lam": lam, "anchor_types": [len(anchors[c]) for c in ae.CLASSES if c in anchors], **ae.evaluate(ctx, head, test, anchors)})
                target.write_text(json.dumps(rows)); print(f"seed {seed} split {split}: {time.monotonic() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
