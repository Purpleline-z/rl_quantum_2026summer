"""Edge statistics of kNN graphs built from different representations (descriptive, development): overlap with the SimCLR graph, share of within-run edges, median frame gap of within-run edges, share of 'near-duplicate'
edges (SimCLR cosine distance < 0.01), share of edges touching an ideal image.   python3 pf_graph_stats.py [--seed 10600]"""
import argparse, tempfile
import numpy as np, torch
import nd_core as core, nd_graph_learner as G, nd_physics as NP, nd_representation as R, rt_core as rt


def stats(ctx, graph: str, base_gd, sim):
    cg = R._proxy(ctx, f"G:{graph}", R.ctx_rep(ctx, graph)) if graph != "s4" else ctx; gd = G.graph_for(cg, 10); paths = gd.paths; assert paths == base_gd.paths
    e = gd.edges.numpy(); eb = {tuple(x) for x in base_gd.edges.numpy()}; ov = np.mean([tuple(x) in eb for x in e]); runs = [rt.run_of(p) for p in paths]; frames = [rt.frame_of(p) for p in paths]
    same = [(runs[i] is not None and runs[i] == runs[j]) for i, j in e]; gap = [abs(frames[i] - frames[j]) for (i, j), s in zip(e, same) if s]
    xs = base_gd.xn; dist = 1 - (xs[e[:, 0]] * xs[e[:, 1]]).sum(1).numpy(); ideal = np.array([runs[i] is None or "ideal" in paths[i] for i in range(len(paths))]); touch = ideal[e[:, 0]] | ideal[e[:, 1]]
    return {"graph": graph, "edges": len(e), "overlap with SimCLR graph": ov, "within-run share": float(np.mean(same)), "median frame gap (within-run)": float(np.median(gap)), "near-duplicate share (<0.01)": float((dist < 0.01).mean()), "median SimCLR distance": float(np.median(dist)), "share touching ideals": float(touch.mean())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=10600); ap.add_argument("--out", default=None); args = ap.parse_args(); cache = core.feature_cache()
    with tempfile.TemporaryDirectory() as s:
        ctx = core.make_ctx(args.seed, "A", s, cache); base = G.graph_for(ctx, 10); import pandas as pd
        rows = [stats(ctx, g, base, None) for g in ("s4", "pfa2", "pfb2", "pfc2", "pfa2s", "pfc2s", "rpp")]; df = pd.DataFrame(rows)
        md = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)] + ["| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |" for r in df.itertuples(index=False)]; print("\n".join(md))
        if args.out: open(args.out, "w").write("\n".join(md) + "\n")
