#!/usr/bin/env python3
"""Runner of the transductive / forward-in-time study.

  python3 td_run.py --mode forward  --learners BASE,L:trans_fut --seeds 10500-10519 --out results/td/dev1 [--cuts 12,24,...] [--shard i/n]
  python3 td_run.py --mode standard --learners BASE,L:trans_fut --seeds 10500-10519 --splits A,B --out results/td/devs1       (budgets 10/20/40/60, draw 0, identical random label sets)
  python3 td_run.py --mode blocked  ...                                                                                     (red-team time-blocked split)

One JSON list per (seed, cut) / (seed, split) / (seed, blocked); resumable per learner.  Record per (seed, cut|budget, window, learner): nd_core.metrics_from_d on the decisive test judgments,
the anchor-far / anchor-near strata (tg_run definition, FAR = 0.03), and the logit gaps ``d`` of every test judgment (for group-level bootstraps; group ids and anchor distances are stored once per window in ``meta``).
Forward windows: FA = the next 40 calendar-ordered groups after the cut, FB = the following 34.
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
import rt_blocked as rb
import rt_core as rt
import td_core as td
import td_learners as TL
import tg_run as tg


def seeds_of(text):
    if "-" in text: lo, hi = text.split("-"); return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in text.split(",")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--mode", choices=["forward", "standard", "blocked"], required=True); ap.add_argument("--learners", required=True)
    ap.add_argument("--seeds", required=True); ap.add_argument("--splits", default="A,B"); ap.add_argument("--cuts", default=",".join(map(str, td.CUTS))); ap.add_argument("--budgets", default="10,20,40,60")
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--shard", default="0/1"); args = ap.parse_args(); torch.set_num_threads(1)
    names = args.learners.split(","); fns = {n: TL.get(n) for n in names}; shard, nshards = (int(x) for x in args.shard.split("/")); budgets = [int(b) for b in args.budgets.split(",")]
    args.out.mkdir(parents=True, exist_ok=True); cache = core.feature_cache(); sim = rt.Sim(cache)
    for seed in [s for i, s in enumerate(seeds_of(args.seeds)) if i % nshards == shard]:
        assert seed in core.TD_SEEDS_ALL, f"seed {seed} is not in this study's block"
        units = [("F", c) for c in args.cuts.split(",")] if args.mode == "forward" else [(s, None) for s in args.splits.split(",")] if args.mode == "standard" else [("blocked", None)]
        for unit, cut in units:
            tag = f"seed{seed}_cut{cut}" if args.mode == "forward" else f"seed{seed}_{unit}"; target = args.out / f"{tag}.json"
            done = json.loads(target.read_text()) if target.exists() else []; have = {(r["unit"], r["window"], r["learner"]) for r in done}
            wins = ["FA", "FB"] if args.mode == "forward" else ["T"]; steps = [None] if args.mode == "forward" else budgets
            if all((str(b), w, n) in have for b in steps for w in wins for n in names): continue
            started = time.monotonic()
            with tempfile.TemporaryDirectory() as scratch:
                if args.mode == "forward": ctx = td.make_forward_ctx(seed, int(cut), scratch, cache); tests = td.forward_tests(ctx)
                else:
                    ctx = core.make_ctx(seed, unit, scratch, cache) if args.mode == "standard" else rb.make_blocked_ctx(seed, scratch, cache); tests = {"T": core.TestSet(ctx)}
                    tests["T"].group = [g for g in ctx.test for w in ctx.exp.groups[g].Winner if w in ("1", "2")]; assert len(tests["T"].group) == len(tests["T"].k)
                far = {w: tg.anchor_distance(sim, ctx, t) for w, t in tests.items()}
                for budget in steps:
                    labelled = ctx.initial + ctx.pool if args.mode == "forward" else core.random_labelled(ctx, budget, 0); ctx._td_heads = {}
                    for name in names:
                        if (str(budget), wins[0], name) in have and all((str(budget), w, name) in have for w in wins): continue
                        pw = getattr(fns[name], "per_window", False) and args.mode == "forward"; pred = None
                        for w in wins:
                            if pred is None or pw: ctx._window = w; pred = fns[name](ctx, labelled, seed * 7)
                            t = tests[w]; d = np.asarray(pred(t), dtype=float); fm = far[w] > tg.FAR
                            rec = {"seed": seed, "unit": str(budget), "split": unit if args.mode != "forward" else "F", "cut": cut, "window": w, "learner": name, "budget": budget, **core.metrics_from_d(d, t), **tg.stratum_metrics(d, t, fm),
                                   "n_labelled": len(set(labelled)), "d": [round(float(x), 4) for x in d]}
                            if args.mode == "forward": rec.update({"n_train": ctx.forward["n_train"], "n_train_kept": ctx.forward["n_train_kept"]})
                            done.append(rec)
                    meta = {"seed": seed, "unit": "meta", "window": "meta", "learner": "meta", "windows": {w: {"group": list(getattr(tests[w], "group", [])), "anc_dist": [round(float(x), 4) for x in far[w]], "sign": tests[w].sign.tolist(), "k": tests[w].k.tolist(), "weight": [round(float(x), 4) for x in tests[w].weight]} for w in tests}}
                    if not any(r["unit"] == "meta" for r in done): done.append(meta)
                    target.write_text(json.dumps(done))
            print(f"{tag}: {len(names)} learners in {time.monotonic() - started:.0f}s", flush=True)


if __name__ == "__main__": main()
