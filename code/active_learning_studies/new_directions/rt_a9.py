"""Audit 9 (anchor-strict): remove from the typed anchors (references, utility and outer-test ideal images) every ideal image that is a (near-)copy of a held-out image (cosine distance <= delta),
for all learners alike, and re-evaluate on the standard splits.  Anchors that are copies of test frames are the main route by which typed labels reach held-out judgments (audit 1).
  python3 rt_a9.py --seeds 8800-8809 --splits A,B --out results/rt/a9 [--shard i/n] [--delta 0.003] [--budgets 20,40] [--draws 2]"""
import argparse, json, tempfile, time
from pathlib import Path
import numpy as np, torch
import nd_core as core, rt_core as rt

DEFAULT = "baseline,aw8_all,aw8_all_st2lr0.5,lap3_k10,lap3_k10_st4"


def seeds_of(text):
    lo, hi = (text.split("-") * 2)[:2]; return list(range(int(lo), int(hi) + 1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="8800-8809"); ap.add_argument("--splits", default="A,B"); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--learners", default=DEFAULT); ap.add_argument("--delta", type=float, default=0.003); ap.add_argument("--budgets", default="20,40"); ap.add_argument("--draws", type=int, default=2); args = ap.parse_args(); torch.set_num_threads(1)
    names = args.learners.split(","); budgets = [int(b) for b in args.budgets.split(",")]; shard, n = (int(x) for x in args.shard.split("/")); args.out.mkdir(parents=True, exist_ok=True); cache = core.feature_cache(); sim = rt.Sim(cache)
    with tempfile.TemporaryDirectory() as scratch:
        for seed in [s for i, s in enumerate(seeds_of(args.seeds)) if i % n == shard]:
            for split in args.splits.split(","):
                target = args.out / f"seed{seed}_{split}.json"
                if target.exists(): continue
                t0 = time.monotonic(); ctx = core.make_ctx(seed, split, scratch, cache); test = core.TestSet(ctx); held = sorted(rt.heldout_images(ctx)); removed = {}; kept = {}
                for tag in ("references", "utility_images", "test_images"):
                    table = getattr(ctx.exp, tag); removed[tag] = kept[tag] = 0
                    for cls in list(table):
                        ps = list(table[cls]); dist = 1 - sim.max_sim([str(p) for p in ps], held); keep = [p for p, dm in zip(ps, dist) if dm > args.delta]
                        removed[tag] += len(ps) - len(keep); kept[tag] += len(keep); table[cls] = keep
                recs = []
                for budget in budgets:
                    for draw in range(args.draws):
                        labelled = core.random_labelled(ctx, budget, draw)
                        for name in names:
                            d = rt.predict(ctx, name, labelled, seed * 7 + draw, test)
                            recs.append({"seed": seed, "split": split, "budget": budget, "draw": draw, "learner": name, **core.metrics_from_d(d, test), "delta": args.delta, "anchors_removed": sum(removed.values()), "anchors_kept": sum(kept.values()),
                                         "refs_kept": kept["references"], "min_class_refs": min(len(v) for v in ctx.exp.references.values())})
                target.write_text(json.dumps(recs)); print(f"seed {seed} split {split}: {time.monotonic() - t0:.0f}s removed {sum(removed.values())} kept {sum(kept.values())}", flush=True)


if __name__ == "__main__": main()
