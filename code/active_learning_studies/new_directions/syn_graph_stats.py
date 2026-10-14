"""How the real ideal images (typed anchors) connect to the kNN graph under the loader features, the 16-bit fix and the direct-beam mask (script-generated table; mean over the given seeds / split A).
  python3 syn_graph_stats.py [--seeds=11100,11101,11102]"""
import sys, tempfile
import numpy as np, pandas as pd, torch
import nd_core as core, nd_synthetic as Y
from tg_md import md


def main():
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--")); seeds = [int(s) for s in opts.get("seeds", "11100,11101,11102").split(",")]; cache = core.feature_cache(); torch.set_num_threads(1); rows = []
    with tempfile.TemporaryDirectory() as scratch:
        acc = {}
        for seed in seeds:
            ctx = core.make_ctx(seed, "A", scratch, cache)
            for var in (None, "fix16", "mask"):
                gd, node_x, edges = Y.graph_for(ctx, var, None); idx = {p: i for i, p in enumerate(gd.paths)}; nbr = gd.nbr; traj = torch.tensor(["Trajectories" in p for p in gd.paths])
                for typ in ("1x1", "c6x2", "RT13", "HTR"):
                    ids = [i for p, i in idx.items() if f"STO_ideal_{typ}/" in p]
                    if not ids: continue
                    ids_t = torch.tensor(ids); share_traj = float(traj[nbr[ids_t]].float().mean()); is16 = [i for i in ids if "I;16" in str(__import__("PIL.Image", fromlist=["Image"]).open(gd.paths[i]).mode)]
                    acc.setdefault((var or "loader (baseline)", typ), []).append((share_traj, len(ids)))
        for (var, typ), v in acc.items(): rows.append({"ideal features": var, "type": typ, "ideal nodes (mean)": np.mean([x[1] for x in v]), "share of the 10 nearest neighbours that are trajectory frames": np.mean([x[0] for x in v])})
    print(md(pd.DataFrame(rows), "{:.3f}"))


if __name__ == "__main__": main()
