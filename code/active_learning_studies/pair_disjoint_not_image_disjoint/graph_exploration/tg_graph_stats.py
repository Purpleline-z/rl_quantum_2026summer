"""Descriptive statistics of the temporal edge sets (script-generated table for the results document).

  python3 tg_graph_stats.py [--seeds=9200,9201,9202] [--split=A]       (run from anywhere; PYTHONPATH must contain code/active_learning_program)

Per edge set: number of edges (mean over contexts), share that already are edges of the k = 10 kNN graph, median / 90th percentile / share > 0.05 of the cosine distance (1 - cos of centred, normalised SimCLR features)
between the two endpoints, share of edges between frames whose temperature differs by more than 20 C, and (trajectory frames only) the median rank of the temporal neighbour among the node's feature neighbours.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "new_directions"))
import nd_core as core  # noqa: E402
import nd_graph_learner as G  # noqa: E402
import nd_temporal_graph as T  # noqa: E402
from tg_tables import md  # noqa: E402

SETS = {"kNN graph (k=10), all edges": None, "chain +-1": "tg_u1", "chain +-2": "tg_u2", "chain +-5": "tg_u5", "chain +-5, only edges NOT in the kNN graph": "tg_novel5", "temperature-bin partners": "tg_tbin",
        "chain +-2 shuffled within runs (control)": "tg_u2_shuf", "chain +-2 locally shuffled (control)": "tg_u2_lshuf", "random kNN edges, same number as chain +-2 (control)": "tg_u2_krand"}


def main():
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--")); seeds = [int(s) for s in opts.get("seeds", "9200,9201,9202").split(",")]; split = opts.get("split", "A"); torch.set_num_threads(1); rows = []
    cache = core.feature_cache()
    with tempfile.TemporaryDirectory() as scratch:
        acc: dict[str, list] = {k: [] for k in SETS}
        for seed in seeds:
            ctx = core.make_ctx(seed, split, scratch, cache); gd = G.graph_for(ctx, 10, False); nd = T.Nodes(gd); xn = gd.xn; knn = {(int(a), int(b)) for a, b in gd.edges.numpy()}
            sim = xn @ xn.T; rank = (-sim).argsort(1).argsort(1)   # rank[i, j] = position of j among i's feature neighbours (0 = itself)
            for label, name in SETS.items():
                if name is None: i, j = gd.edges[:, 0].numpy(), gd.edges[:, 1].numpy()
                else: i, j = (t.numpy() for t in T.edge_sets(ctx, gd, T.SPEC_OF[name])["temporal"][:2])
                dist = 1 - sim[i, j].numpy(); inknn = np.mean([(int(min(a, b)), int(max(a, b))) in knn for a, b in zip(i, j)]); both = nd.is_traj[i] & nd.is_traj[j]
                dT = np.abs(nd.temp[i] - nd.temp[j])[both]; rk = np.minimum(rank[i, j].numpy(), rank[j, i].numpy())
                acc[label].append({"edges": len(i), "share in kNN graph": inknn, "median cos distance": float(np.median(dist)), "p90 cos distance": float(np.percentile(dist, 90)), "share distance > 0.05": float((dist > 0.05).mean()),
                                   "share |dT| > 20C": float((dT > 20).mean()) if both.any() else float("nan"), "median feature-neighbour rank": float(np.median(rk))})
        for label, lst in acc.items(): rows.append({"edge set": label, **pd.DataFrame(lst).mean().to_dict()})
    t = pd.DataFrame(rows); t["edges"] = t["edges"].round(0).astype(int).astype(str); print(f"Mean over contexts: split {split}, seeds {seeds}.\n"); print(md(t, "{:.3f}"))


if __name__ == "__main__": main()
