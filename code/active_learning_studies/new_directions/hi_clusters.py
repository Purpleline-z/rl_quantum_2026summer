"""Independence units for the group-level inference (honest-inference study).  A pair group is the unit of the held-out design (all its judgments are on the same side), but the 284 images are frames of
three growth runs that are near-duplicates of each other, so groups are not independent either.  Single-linkage components chain into one giant component (tau = 0.002: 164 of 168 groups), so the units are:
  unit_g          the pair group itself (168 units)                                                   <- primary
  unit_c<tau>     complete-linkage clusters of the images (diameter <= tau, 1 - cosine of centred, L2-normalised frozen SimCLR features); a group is its UNORDERED pair of image clusters, groups with the
                  same pair of clusters form one unit (tau = 0.01, 0.03, 0.1)
  unit_b<K>       coarse blocks: average-linkage clustering of the groups (embedding = normalised mean of the two images' features), cut into K clusters (K = 84, 42, 21): a stress test of strong
                  dependence between groups that look alike
Output results/hi/clusters.csv (one row per group) -- needs torch (feature cache); everything downstream is numpy only.   python3 hi_clusters.py"""
import numpy as np, pandas as pd, torch
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from pathlib import Path
import nd_core as core, rt_core as rt

HERE = Path(__file__).resolve().parent; TAUS = (0.0, 0.002, 0.01, 0.03)


def main():
    pg = pd.read_csv(HERE.parent / "pair_disjoint_not_image_disjoint" / "results" / "pair_graph_summary.csv"); groups = list(pg.pair_id); cache = core.feature_cache(); sim = rt.Sim(cache)
    def key_of(rel): return next(p for p in sim.paths if p.endswith(rel))
    imgs = sorted({x.strip() for g in groups for x in g.split("||")}); idx = {r: i for i, r in enumerate(imgs)}; vec = torch.stack([sim.xn[sim.index[key_of(r)]] for r in imgs])
    D = (1 - vec @ vec.T).clamp_min(0).numpy(); np.fill_diagonal(D, 0); D = (D + D.T) / 2; gi = [(idx[g.split("||")[0].strip()], idx[g.split("||")[1].strip()]) for g in groups]
    out = pd.DataFrame({"group": groups, "unit_g": range(len(groups))}); Z = linkage(squareform(D, checks=False), method="complete")
    for tau in (0.01, 0.03, 0.1):
        c = fcluster(Z, tau, criterion="distance"); pair = [tuple(sorted((c[a], c[b]))) for a, b in gi]; ids = {p: n for n, p in enumerate(sorted(set(pair)))}; out[f"unit_c{tau:g}"] = [ids[p] for p in pair]
    emb = torch.stack([torch.nn.functional.normalize(vec[a] + vec[b], dim=0) for a, b in gi]); Dg = (1 - emb @ emb.T).clamp_min(0).numpy(); np.fill_diagonal(Dg, 0); Zg = linkage(squareform((Dg + Dg.T) / 2, checks=False), method="average")
    for K in (84, 42, 21): out[f"unit_b{K}"] = fcluster(Zg, K, criterion="maxclust") - 1
    out["run1"] = [g.split("||")[0].strip().split("/")[-1].split("_")[1] for g in groups]; out["run2"] = [g.split("||")[1].strip().split("/")[-1].split("_")[1] for g in groups]
    (HERE / "results" / "hi").mkdir(parents=True, exist_ok=True); out.to_csv(HERE / "results" / "hi" / "clusters.csv", index=False)
    for col in [c for c in out.columns if c.startswith("unit_")]:
        s = out[col].value_counts(); print(f"{col}: {len(s)} units over {len(out)} groups; sizes: max {s.max()}, median {s.median():.0f}, singletons {(s == 1).sum()}")


if __name__ == "__main__": main()
