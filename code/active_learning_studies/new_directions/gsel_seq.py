"""Sequential version of the graph-selection screen (PROTOCOL EXTENSION, marked as such in the pre-registration): rounds of 10 judgments, after every round the selection model
(fitted with the aw8_all loss on the judgments revealed so far; the default-head model is not used) and the P(decisive) predictor see all revealed outcomes.  Evaluation at 10/20/40/60
revealed judgments with the aw8_all learner, exactly as in gsel_select.py (same output format, so gsel_analyze.py works).  Random = the same random label sets as in the single-shot runs."""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import torch

import nd_core as core
import nd_learners as L
import nd_select as ns
import gsel_select as gs
import judgment_unit_study as ju_study
import new_methods_strategies as nm

ROUND = 10
CURRENT: dict = {}


def _install_model_override():
    """The '@aw' rules call ns._selection_model(ctx, 8.0, True) for the model fitted on the initial judgments; in the sequential run it must be the model fitted on the current labelled set."""
    ns._selection_model = lambda ctx, aw, all_ideal: CURRENT["model"]


def fit_selection_model(ctx, labelled):
    return ns._Model(core.fit_generic(ctx, labelled, core.identity_features(ctx), 512, *ctx.params_for(labelled), ctx.seed, anchor_weight=8.0, all_ideal=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selectors", required=True); parser.add_argument("--seeds", default="8000-8019"); parser.add_argument("--splits", default="A,B"); parser.add_argument("--draws", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True); parser.add_argument("--shard", default="0/1")
    args = parser.parse_args(); torch.set_num_threads(1); _install_model_override(); learner = L.LEARNERS["aw8_all"]
    sel_names = args.selectors.split(","); shard, nshards = (int(x) for x in args.shard.split("/")); args.out.mkdir(parents=True, exist_ok=True); cache = core.feature_cache()
    with tempfile.TemporaryDirectory() as scratch:
        for seed in [s for i, s in enumerate(ns.parse_seeds(args.seeds)) if i % nshards == shard]:
            for split in args.splits.split(","):
                target = args.out / f"seed{seed}_{split}.json"; done = json.loads(target.read_text()) if target.exists() else []; have = {(r["budget"], r["selector"], r["draw"]) for r in done}
                need = [(b, s, (range(args.draws) if s == "random" else [0])) for b in core.BUDGETS for s in sel_names]
                if all((b, s, d) in have for b, s, ds in need for d in ds): continue
                started = time.monotonic(); ctx = core.make_ctx(seed, split, scratch, cache); test = core.TestSet(ctx); ns.CTX_REF["ctx"] = ctx; nm.CTX["ctx"] = ctx
                by_id = {f"{j[0]}#{j[1]}": j for j in ctx.pool}; initial = set(ctx.initial); first = ctx.fit(ctx.initial)
                for sel in sel_names:
                    if sel == "random":
                        for budget in core.BUDGETS:
                            for draw in range(args.draws):
                                if (budget, sel, draw) in have: continue
                                labelled = core.random_labelled(ctx, budget, draw); cands, _, _ = ju_study.make_candidates(ctx, ctx.pool, ctx.initial, first, seed)
                                lookup = {(x["base_pair"], int(x["pos"])): x for x in cands}; picked = [lookup[j] for j in labelled if j not in initial]
                                d = learner(ctx, labelled, seed * 7 + draw)(test); done.append({"seed": seed, "split": split, "budget": budget, "selector": sel, "draw": draw, "learner": "aw8_all", **core.metrics_from_d(d, test), **gs.per_type_auc(d, test), **gs.composition(picked, by_id)})
                        target.write_text(json.dumps(done)); continue
                    if all((b, sel, 0) in have for b in core.BUDGETS): continue
                    labelled = list(ctx.initial); revealed_items: list = []
                    for rnd in range(max(core.BUDGETS) // ROUND):
                        remaining = [j for j in ctx.pool if j not in set(labelled)]; CURRENT["model"] = fit_selection_model(ctx, labelled)
                        model = ctx.fit(labelled); cands, labeled_items, ccache = ju_study.make_candidates(ctx, remaining, labelled, model, seed + rnd)
                        picks = gs.REGISTRY[sel](cands, labeled_items, CURRENT["model"] if sel.endswith("@aw") else model, ccache, ROUND, seed + rnd)
                        ids = {f"{j[0]}#{j[1]}": j for j in remaining}; chosen = [ids[x["pair_id"]] for x in picks]; assert len(set(chosen)) == ROUND
                        labelled = labelled + chosen; revealed_items += picks; budget = (rnd + 1) * ROUND
                        if budget in core.BUDGETS and (budget, sel, 0) not in have:
                            d = learner(ctx, labelled, seed * 7)(test); done.append({"seed": seed, "split": split, "budget": budget, "selector": sel, "draw": 0, "learner": "aw8_all", **core.metrics_from_d(d, test), **gs.per_type_auc(d, test), **gs.composition(revealed_items, by_id)})
                    target.write_text(json.dumps(done))
                print(f"seed {seed} split {split}: {time.monotonic() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
