"""Harsh-split runner: lap{1,3,10}_k10, aw8_all, shuffled and noisy-graph controls on held-out run / held-out temperature (inductive graph without the held-out frames), identical random label sets."""
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
import ge_harsh as hh

ARMS = {"lap3_k5": (3.0, False, 0.0), "lap3_k20": (3.0, False, 0.0), "lapF25": (3.0, False, 0.0), "lapF50": (3.0, False, 0.0), "lapF100": (3.0, False, 0.0), "lap3_T": (3.0, False, 0.0), "base": None, "lap1": (1.0, False, 0.0), "lap3": (3.0, False, 0.0), "lap10": (10.0, False, 0.0), "lap3_shuf": (3.0, True, 0.0), "lap3_n3": (3.0, False, 3.0)}


def parse_seeds(text):
    if "-" in text: lo, hi = text.split("-"); return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in text.split(",")]


def run_seed(seed, kind, cache, scratch, budgets, draws, arms, save_d=None):
    rows = []; fn = None; dump = {}
    per_fold = []
    probe = core.make_ctx(seed, "A", scratch, cache)
    for fold_index, (fold, held) in enumerate(hh.folds_of(probe, kind)):
        ctx = core.make_ctx(seed, "A", scratch, cache); test_groups, frames = hh.make_fold(ctx, held, fold_index); test = core.TestSet(ctx, test_groups); fn = core.identity_features(ctx)
        def dropped(name):
            if name == "lap3_T": return ()
            if name.startswith("lapF"):   # exploratory: a fraction of the held-out condition's UNLABELLED frames (not the test images) is allowed into the graph
                order = sorted(frames); rng = np.random.default_rng(seed * 17 + fold_index); order = [order[i] for i in rng.permutation(len(order))]; return set(order[int(round(int(name[4:]) / 100 * len(order))):])
            return frames
        graphs = {name: gh.make_graph(ctx, {"lap3_k5": 5, "lap3_k20": 20}.get(name, 10), shuffle=spec[1], noise=spec[2], drop_paths=dropped(name), keep_heldout_pair_images=(name == "lap3_T")) for name, spec in ARMS.items() if spec is not None and name in arms}
        per_fold.append({"ctx": ctx, "test": test, "graphs": graphs, "fold": fold})
    pooled = hh.Pooled([f["test"] for f in per_fold])
    if save_d is not None:
        groups = []
        for f in per_fold:
            r = f["ctx"].exp.rows_for(f["ctx"].test); r = r[r.Winner.isin(["1", "2"])]; groups += list(r.pair_id)
        dump.update({"meta_sign": pooled.sign, "meta_weight": pooled.weight, "meta_k": pooled.k, "meta_group": np.array(groups)})
    for budget in budgets:
        for draw in range(draws):
            ds = {name: [] for name in arms}
            for f in per_fold:
                ctx = f["ctx"]; lab = list(ctx.initial) + [ctx.pool[i] for i in np.random.default_rng(seed * 100003 + budget * 101 + draw).choice(len(ctx.pool), size=min(budget, len(ctx.pool)), replace=False)]; s = seed * 7 + draw
                for name in arms:
                    gd = f["graphs"].get(name); lam = 0.0 if gd is None else ARMS[name][0]
                    head = gh.fit_head(ctx, lab, s, gd, fn, None if gd is None else gd.x, 512, lap=lam, all_ideal=True); ds[name].append(core.head_d(head, fn, f["test"]))
            for name in arms: dump[f"d_{budget}_{draw}_{name}"] = np.concatenate(ds[name]).astype(np.float32)
            for name in arms: rows.append({"seed": seed, "kind": kind, "budget": budget, "draw": draw, "arm": name, "n_folds": len(per_fold), **core.metrics_from_d(np.concatenate(ds[name]), pooled)})
    if save_d is not None: np.savez_compressed(save_d, **dump)
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", required=True); ap.add_argument("--kinds", default="run,temp"); ap.add_argument("--budgets", default="20,60"); ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--arms", default=",".join(ARMS)); ap.add_argument("--save-d", action="store_true"); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--shard", default="0/1"); a = ap.parse_args()
    torch.set_num_threads(1); shard, n = (int(x) for x in a.shard.split("/")); a.out.mkdir(parents=True, exist_ok=True); cache = core.feature_cache()
    with tempfile.TemporaryDirectory() as scratch:
        for i, seed in enumerate(parse_seeds(a.seeds)):
            if i % n != shard: continue
            for kind in a.kinds.split(","):
                target = a.out / f"seed{seed}_{kind}.json"
                if target.exists(): continue
                t0 = time.monotonic(); rows = run_seed(seed, kind, cache, scratch, [int(b) for b in a.budgets.split(",")], a.draws, a.arms.split(","), save_d=(a.out / f"seed{seed}_{kind}.d.npz") if a.save_d else None); target.write_text(json.dumps(rows)); print(f"seed {seed} {kind}: {time.monotonic() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
