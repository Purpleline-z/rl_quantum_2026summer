"""Honest-inference study (branch claude/honest-inference-*): per-judgment held-out predictions of the confirmed ladder on many random splits (rt_a1.py style, with held-out pair-group ids).

  python3 hi_run.py --seeds 11300-11319 --splits A,B --out results/hi/dev --draws 3 [--shard i/n] [--learners ...] [--budgets 10,20,40,60]
  python3 hi_run.py --blocked --seeds 11300-11319 --out results/hi/dev_blocked --draws 1 [--shard i/n]          (time-blocked contexts of rt_blocked.py: one context per seed)
Per (seed, split) two files, both idempotent (skipped when they exist):
  <out>/meta/seed<S>_<split>.csv     one row per held-out decisive judgment j: group (pair id of the held-out pair group), img1, img2, k (type), sign, weight, distances (1 - cos) of its images to the nearest typed anchor
  <out>/pred/seed<S>_<split>.csv.gz  long format (budget, draw, learner, j, d) with d = logit gap of the learner for judgment j; labelled sets are identical for every learner (nd_core.random_labelled)
"""
import argparse, os, tempfile, time
from pathlib import Path
import numpy as np, pandas as pd, torch
import nd_core as core, nd_graph_learner as G, nd_graph_mechanism as GM, rt_core as rt

LEARNERS_DEFAULT = "baseline,aw8,aw8_all,aw8_all_st2lr0.5,lap3_k10,lap3_k10_st4,lap3_k10_st4_shuf,lap3_k10_st4_rt13off,gmc_t_no_rt13"
BLOCKED_DEFAULT = "baseline,aw8_all,aw8_all_st2lr0.5,lap3_k10,lap3_k10_st4,lap3_k10_st4_shuf"


def registry():
    reg = {**rt.REGISTRY, **GM.LEARNERS}
    reg["lap3_k10_st4_rt13off"] = G._mlp_learner(lap=3.0, steps_scale=4.0, lap_w={3: 0.0})   # as nd_calibration.py line 65 (Laplacian penalty switched off for the RT13 score)
    return reg


def seeds_of(text):
    if "-" in text: lo, hi = text.split("-"); return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in text.split(",")]


def rel(p):
    p = str(p); i = p.find("Trajectories"); return p[i:] if i >= 0 else p[p.find("data/") + 5:] if "data/" in p else p


def meta_table(ctx, test, sim) -> pd.DataFrame:
    rows = ctx.exp.rows_for(ctx.test); rows = rows[rows.Winner.isin(["1", "2"])].reset_index(drop=True)
    assert len(rows) == len(test.sign) and (rows.resolved_img1.tolist() == test.img1)
    out = {"j": np.arange(len(rows)), "group": [str(g) for g in rows.pair_id], "img1": [rel(p) for p in test.img1], "img2": [rel(p) for p in test.img2], "k": test.k, "sign": test.sign, "weight": test.weight}
    for tag, table in (("ref", ctx.exp.references), ("util", ctx.exp.utility_images), ("outer", ctx.exp.test_images)):
        anchors = sorted({str(p) for ps in table.values() for p in ps}); da, db = 1 - sim.max_sim(list(test.img1), anchors), 1 - sim.max_sim(list(test.img2), anchors)
        out[f"anc_{tag}_min"], out[f"anc_{tag}_max"] = np.minimum(da, db), np.maximum(da, db)
    allanc = sorted({str(p) for t in (ctx.exp.references, ctx.exp.utility_images, ctx.exp.test_images) for ps in t.values() for p in ps}); da, db = 1 - sim.max_sim(list(test.img1), allanc), 1 - sim.max_sim(list(test.img2), allanc)
    out["anc_all_min"], out["anc_all_max"] = np.minimum(da, db), np.maximum(da, db)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", required=True); ap.add_argument("--splits", default="A,B"); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--learners", default=None); ap.add_argument("--budgets", default="10,20,40,60"); ap.add_argument("--draws", type=int, default=2); ap.add_argument("--blocked", action="store_true"); args = ap.parse_args()
    torch.set_num_threads(1); names = (args.learners or (BLOCKED_DEFAULT if args.blocked else LEARNERS_DEFAULT)).split(","); budgets = [int(b) for b in args.budgets.split(",")]; shard, n = (int(x) for x in args.shard.split("/"))
    reg = registry(); cache = core.feature_cache(); sim = rt.Sim(cache); (args.out / "meta").mkdir(parents=True, exist_ok=True); (args.out / "pred").mkdir(parents=True, exist_ok=True)
    if args.blocked: import rt_blocked as rb
    with tempfile.TemporaryDirectory() as scratch:
        for seed in [s for i, s in enumerate(seeds_of(args.seeds)) if i % n == shard]:
            for split in (["blocked"] if args.blocked else args.splits.split(",")):
                target = args.out / "pred" / f"seed{seed}_{split}.csv.gz"
                if target.exists(): continue
                t0 = time.monotonic(); ctx = rb.make_blocked_ctx(seed, scratch, cache) if args.blocked else core.make_ctx(seed, split, scratch, cache); test = core.TestSet(ctx)
                meta = meta_table(ctx, test, sim); meta.to_csv(args.out / "meta" / f"seed{seed}_{split}.csv", index=False); rows = []
                for budget in budgets:
                    for draw in range(args.draws):
                        labelled = core.random_labelled(ctx, budget, draw)
                        for name in names:
                            d = np.asarray(reg[name](ctx, labelled, seed * 7 + draw)(test), dtype=float)
                            rows.append(pd.DataFrame({"budget": budget, "draw": draw, "learner": name, "j": np.arange(len(d)), "d": d}))
                tmp = target.with_suffix(".tmp.gz"); pd.concat(rows).to_csv(tmp, index=False); os.replace(tmp, target); print(f"seed {seed} split {split}: {time.monotonic() - t0:.0f}s ({len(test.sign)} judgments, {len(set(meta.group))} groups)", flush=True)


if __name__ == "__main__": main()
