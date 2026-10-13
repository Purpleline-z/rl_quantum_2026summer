"""Audit 5: randomness decomposition of the paired gain (head initialisation + Laplacian edge sampling) on fixed label sets, and a thread-count / label-set-identity check.
  python3 rt_a5.py --seeds 8800-8809 --splits A,B --out results/rt/a5 [--shard i/n] ; analysis in rt_tables.py a5
One JSON per (seed, split): rows (budget, init_seed r, learner, metrics) for budget 40 and 20, draw 0, r = 0..4 (the learner is called with seed*7 + 1000*r)."""
import argparse, hashlib, json, tempfile, time
from pathlib import Path
import numpy as np, torch
import nd_core as core, rt_core as rt

LEARNERS = ("aw8_all_st2lr0.5", "lap3_k10", "lap3_k10_st4")


def seeds_of(text):
    lo, hi = text.split("-"); return list(range(int(lo), int(hi) + 1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="8800-8809"); ap.add_argument("--splits", default="A,B"); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--shard", default="0/1"); ap.add_argument("--inits", type=int, default=5)
    args = ap.parse_args(); torch.set_num_threads(1); shard, n = (int(x) for x in args.shard.split("/")); args.out.mkdir(parents=True, exist_ok=True); cache = core.feature_cache()
    with tempfile.TemporaryDirectory() as scratch:
        for seed in [s for i, s in enumerate(seeds_of(args.seeds)) if i % n == shard]:
            for split in args.splits.split(","):
                target = args.out / f"seed{seed}_{split}.json"
                if target.exists(): continue
                t0 = time.monotonic(); ctx = core.make_ctx(seed, split, scratch, cache); test = core.TestSet(ctx); recs = []
                for budget in (20, 40):
                    lab1, lab2 = core.random_labelled(ctx, budget, 0), core.random_labelled(ctx, budget, 0); assert lab1 == lab2, "label sets differ between calls"; digest = hashlib.sha1(repr(lab1).encode()).hexdigest()[:10]
                    for r in range(args.inits):
                        for name in LEARNERS:
                            d = rt.predict(ctx, name, lab1, seed * 7 + 1000 * r, test); recs.append({"seed": seed, "split": split, "budget": budget, "init": r, "learner": name, "labelset": digest, **core.metrics_from_d(d, test)})
                if seed == seeds_of(args.seeds)[0] and split == "A":   # thread-count check on one fit
                    lab = core.random_labelled(ctx, 20, 0); torch.set_num_threads(1); d1 = rt.predict(ctx, "lap3_k10", lab, 11, test); torch.set_num_threads(4); d4 = rt.predict(ctx, "lap3_k10", lab, 11, test); torch.set_num_threads(1)
                    recs.append({"seed": seed, "split": split, "thread_check_max_abs_diff": float(np.abs(d1 - d4).max())})
                target.write_text(json.dumps(recs)); print(f"seed {seed} split {split}: {time.monotonic() - t0:.0f}s", flush=True)


if __name__ == "__main__": main()
