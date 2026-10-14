"""Latent-type study runner (idempotent per seed x split file; one shard of n).  Writes per seed x split
  <out>/seed<S>_<split>.json     records (seed, split, budget, draw, learner, metrics_from_d)
  <out>/seed<S>_<split>.csv.gz   per-judgment predictions (long format of rt_a1: seed, split, budget, draw, learner, j, d, sign, weight, k) for the anchor-far stratification
Splits: A, B (standard) and 'blocked' (time-blocked contexts of rt_blocked.py).  Same random label set for every learner of a budget/draw.
  python3 lt_run.py --seeds 10700-10719 --splits A,B --learners lap3_k10_st4,... --out results/lt/dev1 [--shard i/n] [--draws 1] [--budgets 10,20,40,60]"""
import argparse, json, tempfile, time
from pathlib import Path
import numpy as np, pandas as pd, torch
import nd_core as core, nd_graph_learner as G, nd_learners as NL, lt_learner as L

REGISTRY = {**NL.LEARNERS, **G.LEARNERS, **L.LT_LEARNERS}


def seeds_of(t):
    if "-" in t: lo, hi = t.split("-"); return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in t.split(",")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", required=True); ap.add_argument("--splits", default="A,B"); ap.add_argument("--learners", required=True); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--budgets", default="10,20,40,60"); ap.add_argument("--draws", type=int, default=1); a = ap.parse_args(); torch.set_num_threads(1)
    names = a.learners.split(","); budgets = [int(b) for b in a.budgets.split(",")]; shard, n = (int(x) for x in a.shard.split("/")); a.out.mkdir(parents=True, exist_ok=True); cache = core.feature_cache()
    missing = [x for x in names if x not in REGISTRY]; assert not missing, missing
    with tempfile.TemporaryDirectory() as scratch:
        for seed in [s for i, s in enumerate(seeds_of(a.seeds)) if i % n == shard]:
            for split in a.splits.split(","):
                target = a.out / f"seed{seed}_{split}.json"
                if target.exists(): continue
                t0 = time.monotonic()
                if split == "blocked":
                    import rt_blocked as rb; ctx = rb.make_blocked_ctx(seed, scratch, cache)
                else: ctx = core.make_ctx(seed, split, scratch, cache)
                test = core.TestSet(ctx); recs, frames = [], []
                for budget in budgets:
                    for draw in range(a.draws):
                        lab = core.random_labelled(ctx, budget, draw)
                        for name in names:
                            d = np.asarray(REGISTRY[name](ctx, lab, seed * 7 + draw)(test), dtype=float)
                            recs.append({"seed": seed, "split": split, "budget": budget, "draw": draw, "learner": name, **core.metrics_from_d(d, test)})
                            frames.append(pd.DataFrame({"seed": seed, "split": split, "budget": budget, "draw": draw, "learner": name, "j": np.arange(len(d)), "d": np.round(d, 5), "sign": test.sign, "weight": test.weight, "k": test.k}))
                pd.concat(frames).to_csv(a.out / f"seed{seed}_{split}.csv.gz", index=False); target.write_text(json.dumps(recs)); print(f"seed {seed} {split}: {time.monotonic() - t0:.0f}s", flush=True)


if __name__ == "__main__": main()
