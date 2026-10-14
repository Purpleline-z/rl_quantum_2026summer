"""Anchor-matching plug-in (development analysis; no retraining): d' = d + beta * [log q_k(img1) - log q_k(img2)] on stored per-judgment predictions of a base learner, q = class posterior of the image from its physics (or SimCLR /
random-projection) features estimated from the typed ideal images (``nd_physics.delta_logq``); also the zero-shot score d = delta alone ("zs:<src>") and the type-permuted control ("~shuf").
  python3 pf_plugin.py DIR.. [--base lap3_k10_st4] [--sources lr_pfa,...] [--betas 0.25,0.5,1]"""
import sys, glob, tempfile, argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
import nd_core as core, nd_physics as NP

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pair_disjoint_not_image_disjoint" / "graph_exploration")); import pf_tables as T  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("dirs", nargs="+"); ap.add_argument("--base", default="lap3_k10_st4"); ap.add_argument("--sources", default="lr_pfa,lr_pfb,lr_pfc,lr_s4,lr_rpp,proto_pfa,proto_s4"); ap.add_argument("--betas", default="0.25,0.5,1.0"); ap.add_argument("--shuf", default="lr_pfa")
    args = ap.parse_args(); torch.set_num_threads(1); cache = core.feature_cache(); sources = args.sources.split(","); betas = [float(b) for b in args.betas.split(",")]; frames = []
    with tempfile.TemporaryDirectory() as scratch:
        for d in args.dirs:
            for f in sorted(glob.glob(str(Path(d) / "seed*.csv.gz"))):
                df = pd.read_csv(f); df = df[df.learner == args.base]
                if df.empty: continue
                seed, split = int(df.seed.iloc[0]), df.split.iloc[0]; ctx = core.make_ctx(seed, split, scratch, cache); test = core.TestSet(ctx); rows = [df]
                for src in sources:
                    dl = NP.delta_logq(ctx, test, src); z = df.copy(); z["d"] = dl[z.j.to_numpy()]; z["learner"] = f"zs:{src}"; rows.append(z)
                    for beta in betas: e = df.copy(); e["d"] = e.d + beta * dl[e.j.to_numpy()]; e["learner"] = f"{args.base}+{src}@{beta:g}"; rows.append(e)
                for src in args.shuf.split(","):
                    for beta in betas:
                        e = df.copy(); dls = np.zeros(len(e))
                        for (b, dr), idx in e.groupby(["budget", "draw"]).groups.items(): dls[e.index.get_indexer(idx)] = NP.delta_logq(ctx, test, src, shuffle_seed=seed * 31 + dr * 7 + b)[e.loc[idx, "j"].to_numpy()]
                        e["d"] = e.d + beta * dls; e["learner"] = f"{args.base}+{src}@{beta:g}~shuf"; rows.append(e)
                allp = pd.concat(rows); frames.append(pd.DataFrame([{"seed": a[0], "split": a[1], "budget": a[2], "draw": a[3], "learner": a[4], **T._cell_metrics(gg)} for a, gg in allp.groupby(["seed", "split", "budget", "draw", "learner"])]))
    fr = pd.concat(frames); print(T.to_md(T.dev_table(fr, args.base)))
