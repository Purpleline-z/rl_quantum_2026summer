#!/usr/bin/env python3
"""Time-proximity audit (exploratory, spare seeds 9285-9299 of this study's block): per-judgment predictions of several learners with, for every test judgment,
  tgap_lab  : max over the two test images of the smallest |camera-frame difference| (same growth run) to any image of a labelled judgment of the fit (9999 if the run has none),
  tgap_node : the same with the unlabelled graph nodes (all other frames not held out),
  anc_all_min : anchor distance (rt_a1b definition).
Question: is the gain of the incumbent (and of the temporal candidates) concentrated in test judgments that are temporally adjacent to training frames?

  python3 tg_audit_time.py --seeds 9285-9299 --splits A,B --out results/tg/audit_time [--learners lap3_k10_st4,aw8_all_st2lr0.5,tg_dec5] [--shard i/n] [--draws 1]
One gzip-CSV per (seed, split); long format (one row per budget, draw, learner, test judgment); resumable."""
import argparse, tempfile, time
from pathlib import Path
import numpy as np, pandas as pd, torch
import nd_core as core, nd_graph_learner as G, rt_core as rt, tg_run

DEFAULT = "lap3_k10_st4,aw8_all_st2lr0.5,tg_dec5"


def frame_gap(paths, refs) -> np.ndarray:
    """Smallest |frame number difference| of each path to any ref path of the same run (9999 if none)."""
    by_run: dict = {}
    for p in refs:
        r, f = rt.run_of(p), rt.frame_of(p)
        if r is not None: by_run.setdefault(r, []).append(f)
    by_run = {r: np.array(sorted(v)) for r, v in by_run.items()}; out = []
    for p in paths:
        r, f = rt.run_of(p), rt.frame_of(p)
        if r is None or r not in by_run: out.append(9999); continue
        v = by_run[r]; i = np.searchsorted(v, f); cand = [abs(int(v[j]) - f) for j in (i - 1, i) if 0 <= j < len(v)]; out.append(min(cand))
    return np.array(out, dtype=float)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="9285-9299"); ap.add_argument("--splits", default="A,B"); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--learners", default=DEFAULT); ap.add_argument("--budgets", default="10,20,40,60"); ap.add_argument("--draws", type=int, default=1); args = ap.parse_args(); torch.set_num_threads(1)
    names = args.learners.split(","); reg = tg_run.registry(); budgets = [int(b) for b in args.budgets.split(",")]; shard, n = (int(x) for x in args.shard.split("/")); args.out.mkdir(parents=True, exist_ok=True)
    cache = core.feature_cache(); sim = rt.Sim(cache)
    with tempfile.TemporaryDirectory() as scratch:
        for seed in [s for i, s in enumerate(tg_run.seeds_of(args.seeds)) if i % n == shard]:
            assert 9285 <= seed <= 9299, "the audit runs on the spare seeds of this study only"
            for split in args.splits.split(","):
                target = args.out / f"seed{seed}_{split}.csv.gz"
                if target.exists(): continue
                t0 = time.monotonic(); ctx = core.make_ctx(seed, split, scratch, cache); test = core.TestSet(ctx); nodes = list(G.graph_for(ctx, 10, False).paths); anc = tg_run.anchor_distance(sim, ctx, test); rows = []
                for budget in budgets:
                    for draw in range(args.draws):
                        labelled = core.random_labelled(ctx, budget, draw); lab = {str(x) for x in rt.labelled_images(ctx, labelled)}
                        gl = np.maximum(frame_gap(list(test.img1), lab), frame_gap(list(test.img2), lab)); gn = np.maximum(frame_gap(list(test.img1), nodes), frame_gap(list(test.img2), nodes))
                        for name in names:
                            d = np.asarray(reg[name](ctx, labelled, seed * 7 + draw)(test), dtype=float)
                            df = pd.DataFrame({"j": np.arange(len(d)), "d": d, "sign": test.sign, "weight": test.weight, "k": test.k, "tgap_lab": gl, "tgap_node": gn, "anc_all_min": anc})
                            df["seed"], df["split"], df["budget"], df["draw"], df["learner"] = seed, split, budget, draw, name; rows.append(df)
                pd.concat(rows).to_csv(target, index=False); print(f"seed {seed} split {split}: {time.monotonic() - t0:.0f}s", flush=True)


if __name__ == "__main__": main()
