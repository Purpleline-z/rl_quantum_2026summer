"""Per-type graph quality: for ideal nodes of each class, the share of their ideal-image neighbours (kNN k=10 graph, seed node sets of the confirmation seeds, split A) that have the same class,
and the confusion among classes.  Descriptive: does the graph look more reliable for the types where the Laplacian helps (1x1, HTR) than for RT13?   usage: PYTHONPATH=../../../active_learning_program python3 gm_type_purity.py > gm_type_purity.md"""
import sys, tempfile
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parents[1] / "new_directions"))
import nd_core as core, nd_graph_learner as G
torch.set_num_threads(1); NAMES = {0: "(1x1)", 2: "c(6x2)", 3: "RT13", 4: "HTR"}; seeds = list(core.GM_CONFIRM1)[:12]
cnt = {a: {b: 0 for b in NAMES} for a in NAMES}; deg_ideal = {a: [] for a in NAMES}; n_nodes = {a: [] for a in NAMES}
with tempfile.TemporaryDirectory() as scratch:
    cache = core.feature_cache()
    for s in seeds:
        ctx = core.make_ctx(s, "A", scratch, cache); gd = G.graph_for(ctx, 10); src = gd.source_class
        for a in NAMES: n_nodes[a].append(sum(1 for c in src.values() if c == a))
        for i, j in gd.edges.tolist():
            if i in src and j in src: cnt[src[i]][src[j]] += 1; cnt[src[j]][src[i]] += 1
        for i, c in src.items(): deg_ideal[c].append(int(gd.adj[i].sum()))
print("| class | ideal nodes per seed | ideal-ideal edges (both ends) | same-class share | most frequent other class | mean degree of ideal nodes (all neighbours) |"); print("|---|---|---|---|---|---|")
for a in NAMES:
    tot = sum(cnt[a].values()); same = cnt[a][a] / tot; other = max((b for b in NAMES if b != a), key=lambda b: cnt[a][b]); print(f"| {NAMES[a]} | {np.mean(n_nodes[a]):.1f} | {tot} | {same:.3f} | {NAMES[other]} ({cnt[a][other] / tot:.3f}) | {np.mean(deg_ideal[a]):.1f} |")
print(f"\n(seeds {seeds[0]}..{seeds[-1]}, split A node sets; chance = class frequency)")
