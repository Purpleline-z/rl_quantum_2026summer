"""Classification endpoint runner (idempotent per seed): 28 outer-test ideal images, paper split, dev / confirmation seeds.
Rows: seed, method, protocol (none|inductive|transductive), budget (-1 for pure classifiers), draw, acc, recalls."""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

import nd_core as core
import ge_classify as gc
import ge_heads as gh

BUDGETS = (0, 60)


def parse_seeds(text):
    if "-" in text: lo, hi = text.split("-"); return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in text.split(",")]


def run_seed_ideal(seed: int, scratch, cache) -> list[dict]:
    """Variants on a graph over the ideal images only (dev screen 2 of the classification endpoint)."""
    ctx = core.make_ctx(seed, "B", scratch, cache); ref, test, util = gc.split_arrays(ctx); y = test[1]; rows = []
    add = lambda method, protocol, pred: rows.append({"seed": seed, "method": method, "protocol": protocol, "budget": -1, "draw": 0, **gc.accuracy(y, pred)})
    for k in (5, 10):
        for proto in ("inductive", "transductive"):
            t = proto == "transductive"; gd = gc.build_graph_ideal(ctx, k, t); add(f"ls_ideal_k{k}", proto, gc.predict_label_spreading(ctx, test, gd, t)); add(f"sgc_ideal_k{k}", proto, gc.predict_sgc(ctx, ref, test, gd, t, seed))
    gd = gc.build_graph_ideal(ctx, 10, False, shuffle=True); add("ls_ideal_k10_shuf", "inductive", gc.predict_label_spreading(ctx, test, gd, False)); return rows


def run_seed_confirm(seed: int, scratch, cache) -> list[dict]:
    """The frozen classification candidates of PREREGISTRATION_GE1: references nn, logreg; candidates laplr / sgc (inductive + transductive), lap3_ref (inductive, 60 judgments) vs its non-graph reference aw8_ref
    and its shuffled-graph control; shuffled-graph control of laplr."""
    ctx = core.make_ctx(seed, "B", scratch, cache); ref, test, util = gc.split_arrays(ctx); y = test[1]; rows = []
    add = lambda method, protocol, budget, draw, pred: rows.append({"seed": seed, "method": method, "protocol": protocol, "budget": budget, "draw": draw, **gc.accuracy(y, pred)})
    add("nn", "none", -1, 0, gc.predict_1nn(ctx, ref, test)); add("logreg", "none", -1, 0, gc.predict_logreg(ctx, ref, test, seed))
    graphs = {"inductive": gc.build_graph(ctx, 10, False), "transductive": gc.build_graph(ctx, 10, True)}; shuf = gc.build_graph(ctx, 10, False, shuffle=True)
    for proto, gd in graphs.items():
        t = proto == "transductive"; add("laplr", proto, -1, 0, gc.predict_laplr(ctx, ref, test, gd, t, seed)); add("sgc", proto, -1, 0, gc.predict_sgc(ctx, ref, test, gd, t, seed))
    add("laplr_shuf", "inductive", -1, 0, gc.predict_laplr(ctx, ref, test, shuf, False, seed))
    fn = core.identity_features(ctx); tp = test[0]
    for draw in range(3):
        lab = core.random_labelled(ctx, 60, draw); s = seed * 7 + draw
        add("aw8_ref", "none", 60, draw, gh.winrate_vectors(ctx, gh.fit_head(ctx, lab, s, None, fn, None, 512, 0.0, all_ideal=False), tp).argmax(1))
        gd = graphs["inductive"]; add("lap3_ref", "inductive", 60, draw, gh.winrate_vectors(ctx, gh.fit_head(ctx, lab, s, gd, fn, gd.x, 512, lap=3.0, all_ideal=False), tp).argmax(1))
        add("lap3_ref_shuf", "inductive", 60, draw, gh.winrate_vectors(ctx, gh.fit_head(ctx, lab, s, shuf, fn, shuf.x, 512, lap=3.0, all_ideal=False), tp).argmax(1))
    return rows


def run_seed_laplr(seed: int, scratch, cache) -> list[dict]:
    ctx = core.make_ctx(seed, "B", scratch, cache); ref, test, util = gc.split_arrays(ctx); y = test[1]; rows = []
    add = lambda method, protocol, pred: rows.append({"seed": seed, "method": method, "protocol": protocol, "budget": -1, "draw": 0, **gc.accuracy(y, pred)})
    for proto in ("inductive", "transductive"):
        t = proto == "transductive"; add("laplr", proto, gc.predict_laplr(ctx, ref, test, gc.build_graph(ctx, 10, t), t, seed))
    add("laplr_shuf", "inductive", gc.predict_laplr(ctx, ref, test, gc.build_graph(ctx, 10, False, shuffle=True), False, seed)); return rows


def run_seed(seed: int, scratch, cache, heads: bool = True) -> list[dict]:
    ctx = core.make_ctx(seed, "B", scratch, cache); ref, test, util = gc.split_arrays(ctx); y = test[1]; rows = []
    add = lambda method, protocol, budget, draw, pred: rows.append({"seed": seed, "method": method, "protocol": protocol, "budget": budget, "draw": draw, **gc.accuracy(y, pred)})
    add("nn", "none", -1, 0, gc.predict_1nn(ctx, ref, test)); add("logreg", "none", -1, 0, gc.predict_logreg(ctx, ref, test, seed))
    graphs = {"inductive": gc.build_graph(ctx, 10, False), "transductive": gc.build_graph(ctx, 10, True)}; shuf = gc.build_graph(ctx, 10, False, shuffle=True)
    for proto, gd in graphs.items():
        t = proto == "transductive"
        add("ls", proto, -1, 0, gc.predict_label_spreading(ctx, test, gd, t)); add("sgc", proto, -1, 0, gc.predict_sgc(ctx, ref, test, gd, t, seed)); add("gcn", proto, -1, 0, gc.predict_gcn(ctx, ref, test, gd, t, seed))
    add("ls_shuf", "inductive", -1, 0, gc.predict_label_spreading(ctx, test, shuf, False)); add("gcn_shuf", "inductive", -1, 0, gc.predict_gcn(ctx, ref, test, shuf, False, seed))
    if heads:
        fn = core.identity_features(ctx); tp = test[0]
        for budget in BUDGETS:
            for draw in range(3 if budget else 1):
                lab = list(ctx.initial) if budget == 0 else core.random_labelled(ctx, budget, draw); s = seed * 7 + draw
                h_aw = gh.fit_head(ctx, lab, s, None, fn, None, 512, 0.0, all_ideal=False); w_aw = gh.winrate_vectors(ctx, h_aw, tp); add("aw8_ref", "none", budget, draw, w_aw.argmax(1))
                for proto, gd in graphs.items():
                    h = gh.fit_head(ctx, lab, s, gd, fn, gd.x, 512, lap=3.0, all_ideal=False); w = gh.winrate_vectors(ctx, h, tp); add("lap3_ref", proto, budget, draw, w.argmax(1))
                    q = gc.class_scores_from_q(gd, gd.q(), tp, proto == "transductive", ctx.features.cache); fuse = w / w.sum(1, keepdims=True) + q / q.sum(1, keepdims=True); add("fusion", proto, budget, draw, fuse.argmax(1))
                h = gh.fit_head(ctx, lab, s, shuf, fn, shuf.x, 512, lap=3.0, all_ideal=False); add("lap3_ref_shuf", "inductive", budget, draw, gh.winrate_vectors(ctx, h, tp).argmax(1))
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", required=True); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--shard", default="0/1"); ap.add_argument("--no-heads", action="store_true"); ap.add_argument("--ideal", action="store_true"); ap.add_argument("--laplr-only", action="store_true"); ap.add_argument("--confirm", action="store_true"); a = ap.parse_args()
    torch.set_num_threads(1); shard, n = (int(x) for x in a.shard.split("/")); a.out.mkdir(parents=True, exist_ok=True); cache = core.feature_cache()
    with tempfile.TemporaryDirectory() as scratch:
        for i, seed in enumerate(parse_seeds(a.seeds)):
            if i % n != shard or (a.out / f"seed{seed}.json").exists(): continue
            t0 = time.monotonic(); rows = run_seed_confirm(seed, scratch, cache) if a.confirm else run_seed_laplr(seed, scratch, cache) if a.laplr_only else run_seed_ideal(seed, scratch, cache) if a.ideal else run_seed(seed, scratch, cache, not a.no_heads); (a.out / f"seed{seed}.json").write_text(json.dumps(rows)); print(f"seed {seed}: {time.monotonic() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
