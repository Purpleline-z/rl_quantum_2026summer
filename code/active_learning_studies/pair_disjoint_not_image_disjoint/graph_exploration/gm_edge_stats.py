"""Graph statistics of every lambda-10 development graph vs its development gain (descriptive; writes gm_edge_stats.csv):
quality = same-class share of ideal-ideal edges (circular for the LDA topologies), mean cosine of edge endpoints in the ORIGINAL final feature space (how close the linked images really are),
mean degree.  Uses development seeds 8000-8019, split A node sets.   usage: PYTHONPATH=../../../active_learning_program python3 gm_edge_stats.py"""
import sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd, torch
HERE = Path(__file__).resolve().parent; NEW = HERE.parents[1] / "new_directions"
sys.path.insert(0, str(NEW)); sys.path.insert(0, str(HERE))
import nd_core as core, nd_graph_learner as G, nd_graph_mechanism as M
torch.set_num_threads(1)
SPECS = {"gm_lap10_n0": {}, "gm_lap10_n0.5": {"noise": .5}, "gm_lap10_n1": {"noise": 1.}, "gm_lap10_n2": {"noise": 2.}, "gm_lap10_n3": {"noise": 3.},
         "gm10_mutual_k15": {"construct": "mutual", "k": 15}, "gm10_eps_d10": {"construct": "eps", "deg": 10}, "gm10_snn": {"construct": "snn", "k": 10, "snn_k": 20}, "gm10_gauss_k10": {"construct": "gauss"},
         "gm10_lscale_k10": {"construct": "lscale"}, "gm10_bmatch_b10": {"construct": "bmatch", "b": 10}, "gm10_pcaw32_k10": {"whiten": 32}, "gm10_l3_k10": {"feat": "l3"}, "gm10_l2_k10": {"feat": "l2"},
         "gm10_l3l4_k10": {"feat": "l3l4"}, "gm10_k5": {"k": 5}, "gm10_k20": {"k": 20}, "gm10_k30": {"k": 30}, "gm10_lda1": {"lda": 1.0}, "gm10_lda3": {"lda": 3.0},
         "gm10_place_unl": {"place": "unl"}, "gm10_place_it": {"place": "it"}, "gm10_ring3": {"construct": "ring", "k": 10, "r0": 3}, "gm10_ring6": {"construct": "ring", "k": 10, "r0": 6}, "gm10_drop20": {"construct": "drop", "k": 10, "frac": 0.2}, "gm10_drop40": {"construct": "drop", "k": 10, "frac": 0.4}, "gm10_imnet": {"feat": "imnet"}, "gm10_imnet_simclr": {"feat": "imnet_simclr"}}
rows = []
with tempfile.TemporaryDirectory() as scratch:
    cache = core.feature_cache()
    for seed in range(8000, 8020):
        ctx = core.make_ctx(seed, "A", scratch, cache); base = G.graph_for(ctx, 10); xn = base.xn
        for name, sp in SPECS.items():
            e, w = M.build_edges(base, sp); e, w = M.filter_edges(base, e, w, sp.get("place", "all")); q = M.graph_quality(base, e, w)
            cos = float((w * (xn[e[:, 0]] * xn[e[:, 1]]).sum(1)).sum() / w.sum()); n = len(base.paths); bk = set((base.edges[:, 0] * n + base.edges[:, 1]).tolist()); ek = (e[:, 0] * n + e[:, 1]).tolist()
            overlap = float(np.mean([k in bk for k in ek])) if len(ek) else float("nan"); rows.append({"seed": seed, "graph": name, **q, "edge_cosine": cos, "overlap_with_head_knn": overlap})
pd.DataFrame(rows).to_csv(HERE / "gm_edge_stats.csv", index=False); print("saved", len(rows))
