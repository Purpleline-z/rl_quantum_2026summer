#!/usr/bin/env python3
"""Compare learners on identical random label sets (paired).

  python3 nd_run.py --learners baseline,symmetric --seeds 2000-2019 --splits A,B --draws 3 --out results/screen1 [--shard 0/4]

Per (seed, split) one JSON in <out>/ with one record per (budget, draw, learner).  Existing files are reused, so reruns resume.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import torch

import nd_core as core
import nd_learners as learners


def parse_seeds(text: str):
    if "-" in text: lo, hi = text.split("-"); return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in text.split(",")]


def registry() -> dict:
    out = dict(learners.LEARNERS)
    try:
        import nd_bayes; out.update(nd_bayes.LEARNERS)
    except ImportError: pass
    try:
        import nd_gp; out.update(nd_gp.LEARNERS)
    except ImportError: pass
    try:
        import nd_temporal; out.update(nd_temporal.LEARNERS)
    except ImportError: pass
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learners", required=True); parser.add_argument("--seeds", default="2000-2019"); parser.add_argument("--splits", default="A,B")
    parser.add_argument("--draws", type=int, default=3); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--shard", default="0/1")
    args = parser.parse_args(); torch.set_num_threads(1)
    names = args.learners.split(","); reg = registry(); fns = {n: reg[n] for n in names}
    shard, nshards = (int(x) for x in args.shard.split("/")); args.out.mkdir(parents=True, exist_ok=True); cache = core.feature_cache()
    seeds = [s for i, s in enumerate(parse_seeds(args.seeds)) if i % nshards == shard]
    with tempfile.TemporaryDirectory() as scratch:
        for seed in seeds:
            for split in args.splits.split(","):
                target = args.out / f"seed{seed}_{split}.json"; done = json.loads(target.read_text()) if target.exists() else []
                have = {(r["budget"], r["draw"], r["learner"]) for r in done}
                if all((b, d, n) in have for b in core.BUDGETS for d in range(args.draws) for n in names): continue
                started = time.monotonic(); ctx = core.make_ctx(seed, split, scratch, cache); test = core.TestSet(ctx); val = core.TestSet(ctx, ctx.validation) if len(ctx.validation) else None
                for budget in core.BUDGETS:
                    for draw in range(args.draws):
                        labelled = core.random_labelled(ctx, budget, draw)
                        for name in names:
                            if (budget, draw, name) in have: continue
                            predictor = fns[name](ctx, labelled, seed * 7 + draw)
                            d_test = predictor(test); record = {"seed": seed, "split": split, "budget": budget, "draw": draw, "learner": name, **core.metrics_from_d(d_test, test)}
                            if val is not None: record["cal_ll"] = core.calibrated_log_loss(predictor(val), val, d_test, test)
                            done.append(record)
                    target.write_text(json.dumps(done))
                print(f"seed {seed} split {split}: {len(names)} learners in {time.monotonic() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
