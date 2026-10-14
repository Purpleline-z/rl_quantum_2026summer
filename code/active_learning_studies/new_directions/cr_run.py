"""Runner of the cross-run transfer study (CR).  One JSON per (seed, split kind, run): a "meta" record (fold counts, per-judgment test group / sign / weight / anchor-far flag) and one record per
(learner, b) with the standard metrics, the anchor-far / anchor-near metrics and the per-judgment score gaps.

  python3 cr_run.py --learners cr_src,cr_tgt,cr_pool --seeds 10900-10911 --kinds A,B --budgets 0,5,10,20,40 --out results/cr/dev1 [--shard i/n]
Resumable: finished (learner, b) records are skipped.  b is irrelevant for cr_src (stored once at b = 0 and reused by the tables for every b)."""
import argparse, json, tempfile, time
from pathlib import Path
import numpy as np, torch
import nd_core as core, nd_crossrun as C, rg_run, rt_core as rt


def seeds_of(text):
    if "-" in text: lo, hi = text.split("-"); return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in text.split(",")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--learners", required=True); ap.add_argument("--seeds", default="10900-10911"); ap.add_argument("--kinds", default="A,B"); ap.add_argument("--runs", default=",".join(C.RUNS))
    ap.add_argument("--budgets", default="0,5,10,20,40"); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--shard", default="0/1"); args = ap.parse_args()
    torch.set_num_threads(1); names = args.learners.split(","); budgets = [int(b) for b in args.budgets.split(",")]; shard, n = (int(x) for x in args.shard.split("/")); args.out.mkdir(parents=True, exist_ok=True)
    cache = core.feature_cache(); sim = rt.Sim(cache)
    with tempfile.TemporaryDirectory() as scratch:
        for seed in [s for i, s in enumerate(seeds_of(args.seeds)) if i % n == shard]:
            ctx = None
            for kind in args.kinds.split(","):
                for run in args.runs.split(","):
                    target = args.out / f"seed{seed}_{kind}_{run}.json"; done = json.loads(target.read_text()) if target.exists() else []; have = {(r["learner"], r["budget"]) for r in done if "learner" in r}
                    skip = lambda nm, b: ((nm.startswith(("cr_src", "cr_ind_src", "cr_nolap_src", "cr_anchor0", "cr_align_src"))) and b != 0) or (nm.startswith("cr_lapR") and not nm.endswith("_pool") and b != 0) or (nm.startswith("cr_lapR") and nm.endswith("_pool") and b == 0) or (nm.startswith(("cr_pool", "cr_iw", "cr_align_pool")) and b == 0)   # b = 0 of pooled / weighted learners is the source-only head
                    need = [(nm, b) for nm in names for b in budgets if not skip(nm, b)]
                    if all(x in have for x in need): continue
                    t0 = time.monotonic()
                    if ctx is None: ctx = core.make_ctx(seed, "A", scratch, cache)
                    fold = C.Fold(ctx, run, kind); runner = C.Runner(fold); far = rg_run.anchor_min_distance(sim, ctx, fold.test) > rg_run.FAR
                    if not any(r.get("meta") for r in done):
                        done.append({"meta": True, "seed": seed, "split": kind, "run": run, **fold.counts, "n_test": int(len(fold.test.k)), "n_far": int(far.sum()), "n_source_judgments": len(fold.source), "n_pool_judgments": len(fold.target_order),
                                     "n_frames": len(fold.frames), "test_group": [str(g) for g in fold.test_group_of], "sign": [float(x) for x in fold.test.sign], "weight": [round(float(x), 4) for x in fold.test.weight], "k": [int(x) for x in fold.test.k], "far": [bool(x) for x in far]})
                    for b in budgets:
                        for nm in names:
                            if (nm, b) in have or skip(nm, b): continue
                            d = C.d_of(ctx, runner.head(nm, b), fold.test, runner.align[0] if nm.startswith("cr_align_") else None)
                            done.append({"seed": seed, "split": kind, "run": run, "budget": b, "selector": "random", "draw": 0, "learner": nm, **core.metrics_from_d(d, fold.test), **rg_run.subset_metrics(d, fold.test, far, "far"),
                                         **rg_run.subset_metrics(d, fold.test, ~far, "near"), "d": [round(float(x), 4) for x in d]})
                        target.write_text(json.dumps(done))
                    print(f"seed {seed} split {kind} run {run}: {time.monotonic() - t0:.0f}s ({len(fold.test.k)} test judgments, {int(far.sum())} anchor-far)", flush=True)


if __name__ == "__main__":
    main()
