"""Runner of the regulariser study (G2): learners on the shared random label sets (selector = "random" in the records so that ``nd_confirm.run`` reads them unchanged), standard held-out
endpoint plus the red-team "anchor-far" endpoint (both test images > 0.03 cosine distance from every typed ideal-image anchor; ``rt_a1b.py`` definition: min over the two images of 1 - max cosine
similarity to the anchors of the seed's split) and its complement ("anchor-near").  Per-judgment score gaps are stored (rounded) so that other strata can be computed later.

  python3 rg_run.py --learners a,b,c --seeds 9000-9019 --splits A,B --draws 1 --out results/rg/dev1 [--shard i/n]
Resumable: a (seed, split) file is complete when every (budget, draw, learner) record exists; finished records are skipped.
"""
import argparse, json, tempfile, time
from pathlib import Path
import numpy as np, torch
import nd_core as core, nd_graph_learner as G, nd_learners as L, nd_regularisers as R, rt_core as rt

FAR = 0.03


def registry():
    reg = L.LEARNERS.copy(); reg.update(G.LEARNERS); reg.update(R.LEARNERS); return reg


def seeds_of(text):
    if "-" in text: lo, hi = text.split("-"); return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in text.split(",")]


def anchor_min_distance(sim, ctx, test) -> np.ndarray:
    anchors = sorted({str(p) for t in (ctx.exp.references, ctx.exp.utility_images, ctx.exp.test_images) for ps in t.values() for p in ps})
    da, db = 1 - sim.max_sim(list(test.img1), anchors), 1 - sim.max_sim(list(test.img2), anchors); return np.minimum(da, db)


def subset_metrics(d, test, mask, tag) -> dict:
    out = {f"n_{tag}": int(mask.sum())}
    if mask.sum() == 0: return {**out, f"acc_{tag}": float("nan"), f"ll_{tag}": float("nan"), f"auc_{tag}": float("nan")}
    s, w, dd = test.sign[mask], test.weight[mask], np.asarray(d, float)[mask]
    out[f"acc_{tag}"] = float((w * ((dd * s) > 0)).sum() / w.sum()); out[f"ll_{tag}"] = float((w * np.logaddexp(0.0, -s * dd)).sum() / w.sum())
    from sklearn.metrics import roc_auc_score
    out[f"auc_{tag}"] = float(roc_auc_score(s > 0, dd)) if len(set(s)) == 2 else float("nan"); return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--learners", required=True); ap.add_argument("--seeds", default="9000-9019"); ap.add_argument("--splits", default="A,B"); ap.add_argument("--draws", type=int, default=1)
    ap.add_argument("--budgets", default="10,20,40,60"); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--shard", default="0/1"); ap.add_argument("--no-d", action="store_true"); args = ap.parse_args()
    torch.set_num_threads(1); reg = registry(); names = args.learners.split(","); budgets = [int(b) for b in args.budgets.split(",")]; shard, n = (int(x) for x in args.shard.split("/")); args.out.mkdir(parents=True, exist_ok=True)
    for nm in names: assert nm in reg, nm
    cache = core.feature_cache(); sim = rt.Sim(cache)
    with tempfile.TemporaryDirectory() as scratch:
        for seed in [s for i, s in enumerate(seeds_of(args.seeds)) if i % n == shard]:
            for split in args.splits.split(","):
                target = args.out / f"seed{seed}_{split}.json"; done = json.loads(target.read_text()) if target.exists() else []; have = {(r["budget"], r["draw"], r["learner"]) for r in done}
                need = [(b, d, nm) for b in budgets for d in range(args.draws) for nm in names]
                if all(x in have for x in need): continue
                t0 = time.monotonic(); ctx = core.make_ctx(seed, split, scratch, cache); test = core.TestSet(ctx); far = anchor_min_distance(sim, ctx, test) > FAR
                for budget in budgets:
                    for draw in range(args.draws):
                        labelled = core.random_labelled(ctx, budget, draw)
                        for nm in names:
                            if (budget, draw, nm) in have: continue
                            d = np.asarray(reg[nm](ctx, labelled, seed * 7 + draw)(test), dtype=float)
                            rec = {"seed": seed, "split": split, "budget": budget, "selector": "random", "draw": draw, "learner": nm, **core.metrics_from_d(d, test), **subset_metrics(d, test, far, "far"), **subset_metrics(d, test, ~far, "near")}
                            if not args.no_d: rec["d"] = [round(float(x), 4) for x in d]
                            done.append(rec)
                    target.write_text(json.dumps(done))
                print(f"seed {seed} split {split}: {time.monotonic() - t0:.0f}s ({len(far)} judgments, {int(far.sum())} anchor-far)", flush=True)


if __name__ == "__main__": main()
