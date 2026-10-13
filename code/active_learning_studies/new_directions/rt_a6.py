"""Audit 6: identity-leak audit by content hash and by near-duplicate distance, plus a distance-scale calibration (ideal images within / between reconstruction types).
  python3 rt_a6.py [--seeds 8800-8804] > results/rt/a6.md"""
import argparse, tempfile
import numpy as np, pandas as pd, torch
import nd_core as core, nd_graph_learner as G, rt_core as rt


def md(df):
    cols = list(df.columns); out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows(): out.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="8800-8804"); args = ap.parse_args(); lo, hi = args.seeds.split("-"); seeds = list(range(int(lo), int(hi) + 1)); torch.set_num_threads(1)
    cache = core.feature_cache(); sim = rt.Sim(cache); hashes = G.content_hashes(sorted(cache)); rows = []
    with tempfile.TemporaryDirectory() as scratch:
        for seed in seeds:
            for split in "AB":
                ctx = core.make_ctx(seed, split, scratch, cache); H = rt.heldout_images(ctx); Hh = {hashes[p] for p in H}; gd = G.graph_for(ctx, 10, False); nodes = set(gd.paths)
                outer = {str(p) for ps in ctx.exp.test_images.values() for p in ps}; util = {str(p) for ps in ctx.exp.utility_images.values() for p in ps}; refs = {str(p) for ps in ctx.exp.references.values() for p in ps}
                lab = {x for j in ctx.initial + ctx.pool for x in (ctx.info[j]["img1"], ctx.info[j]["img2"])}
                h = lambda S: {hashes[p] for p in S}; anchors_all = refs | util | outer
                near = lambda A: float((1 - sim.max_sim(sorted(A), sorted(H))).min()) if A else float("nan")
                rows.append({"seed": seed, "split": split, "held-out images": len(H), "graph nodes": len(nodes), "nodes∩held-out (path)": len(nodes & H), "nodes∩held-out (hash)": len(h(nodes) & Hh),
                             "nodes∩outer-test ideal (hash)": len(h(nodes) & h(outer)), "labelled/pool images∩held-out (path)": len(lab & H), "labelled/pool images∩held-out (hash)": len(h(lab) & Hh),
                             "labelled/pool∩outer-test ideal (hash)": len(h(lab) & h(outer)), "anchors (all_ideal)": len(anchors_all), "of which outer-test ideal": len(outer), "anchors∩held-out (hash)": len(h(anchors_all) & Hh),
                             "min dist anchors→held-out": round(near(anchors_all), 4), "min dist outer-test ideal→held-out": round(near(outer), 4)})
    df = pd.DataFrame(rows); print("## Identity audit (content hash and exact path)\n"); print(md(df)); ids = {c: [p for p in ps] for c, ps in core.make_ctx(seeds[0], "A", tempfile.mkdtemp(), cache).exp.references.items()}
    cls = {str(p): c for c, ps in ids.items() for p in ps}; paths = sorted(cls); X = sim.xn[[sim.index[p] for p in paths]]; S = X @ X.T; n = len(paths); same, diff = [], []
    for i in range(n):
        for j in range(i + 1, n): (same if cls[paths[i]] == cls[paths[j]] else diff).append(1 - float(S[i, j]))
    print("\n## Distance scale: 1 - cos between reference (ideal) images\n"); print(pd.DataFrame({"pair kind": ["same type", "different type"], "n pairs": [len(same), len(diff)], "median": [np.median(same), np.median(diff)], "5% quantile": [np.quantile(same, .05), np.quantile(diff, .05)], "min": [min(same), min(diff)]}).round(4).pipe(md))


if __name__ == "__main__": main()
