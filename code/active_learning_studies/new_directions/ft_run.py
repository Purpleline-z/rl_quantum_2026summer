#!/usr/bin/env python3
"""Runner of the CPU fine-tuning study: learners on identical random label sets (paired), standard endpoint + anchor-far / near strata (+ in-block stratum on the time-blocked split) + per-judgment scores + fit time.

  python3 ft_run.py --learners lap3_k10_st4,ftp_frozen,ftp_base --seeds 10300-10309 --splits A,B,blocked --draws 1 --budgets 20,40 --out results/ft/dev1 [--shard i/n]

One JSON per (seed, split): {"meta": {...per-judgment anchor distance, sign, weight, type}, "records": [...]}, same layout as ``rc_run.py`` (so ``rc_tables.py`` reads it).  Resumable per (budget, draw, learner).
Learners: every registry learner of the earlier studies (``nd_graph_learner``, ``nd_recipe``), ``nd_finetune`` (``ftp_*``) and ``ft_shared`` (``fts_*``).  Split ``blocked`` = ``rt_blocked.make_blocked_ctx`` (100-frame blocks, 25-frame buffer).
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

import nd_core as core
import nd_graph_learner as G
import nd_recipe as R
import nd_finetune as FT
import rt_blocked as rb
import rt_core as rt
import rc_run
import tg_run

FAR = 0.03
ALLOWED = set(core.FT_SEEDS_ALL)


class _Registry(dict):
    def __missing__(self, name):
        if name.startswith("ftp:"): return FT.parse(name)
        raise KeyError(name)


def registry() -> dict:
    out = _Registry(G.LEARNERS); out.update(R.LEARNERS); out.update(FT.LEARNERS)
    try:
        import ft_shared; out.update(ft_shared.learners())
    except ImportError as e: print("ft_shared unavailable:", e)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--learners", required=True); ap.add_argument("--seeds", default="10300-10309"); ap.add_argument("--splits", default="A,B")
    ap.add_argument("--draws", type=int, default=1); ap.add_argument("--budgets", default="20,40"); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--shard", default="0/1")
    args = ap.parse_args(); torch.set_num_threads(1)
    names = args.learners.split(","); reg = registry(); fns = {n: reg[n] for n in names}; budgets = [int(b) for b in args.budgets.split(",")]; shard, nshards = (int(x) for x in args.shard.split("/"))
    args.out.mkdir(parents=True, exist_ok=True); cache = core.feature_cache(); sim = rt.Sim(cache)
    for seed in [s for i, s in enumerate(rc_run.parse_seeds(args.seeds)) if i % nshards == shard]:
        assert seed in ALLOWED, f"seed {seed} is not in this study's block"
        for split in args.splits.split(","):
            target = args.out / f"seed{seed}_{split}.json"; blob = json.loads(target.read_text()) if target.exists() else {"meta": None, "records": []}; have = {(r["budget"], r["draw"], r["learner"]) for r in blob["records"]}
            if all((b, d, n) in have for b in budgets for d in range(args.draws) for n in names): continue
            started = time.monotonic()
            with tempfile.TemporaryDirectory() as scratch:
                ctx = rb.make_blocked_ctx(seed, scratch, cache) if split == "blocked" else core.make_ctx(seed, split, scratch, cache)
                test = core.TestSet(ctx); anc = tg_run.anchor_distance(sim, ctx, test); far = anc > FAR; inb = tg_run.in_block_mask(ctx, test) if split == "blocked" else None
                blob["meta"] = {"anc_min": np.round(anc, 6).tolist(), "k": test.k.tolist(), "sign": test.sign.tolist(), "weight": np.round(test.weight, 6).tolist(), "n_test": int(len(test.k)), "far_share": float(far.mean())}
                for budget in budgets:
                    for draw in range(args.draws):
                        labelled = core.random_labelled(ctx, budget, draw)
                        for name in names:
                            if (budget, draw, name) in have: continue
                            t0 = time.monotonic(); predictor = fns[name](ctx, labelled, seed * 7 + draw); d = np.asarray(predictor(test), dtype=float); dt = time.monotonic() - t0
                            rec = {"seed": seed, "split": split, "budget": budget, "draw": draw, "learner": name, **core.metrics_from_d(d, test), "n_far": int(far.sum()), "seconds": round(dt, 2)}
                            rec.update(tg_run.stratum_metrics(d, test, far, inb)); info = getattr(predictor, "info", None)
                            if info: rec.update({f"info_{k}": round(float(v), 4) for k, v in info.items()})
                            rec["d"] = np.round(d, 4).tolist(); blob["records"].append(rec)
                    target.write_text(json.dumps(blob))
            print(f"seed {seed} split {split}: {len(names)} learners in {time.monotonic() - started:.0f}s", flush=True)


if __name__ == "__main__": main()
