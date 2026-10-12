"""Graph statistics for the write-up (generated, not typed): size, degree, connected components, same-class share of ideal-ideal edges vs. a permutation baseline, (label spreading itself is not scored here: a label source always keeps its own class mass).
  python3 gl_mechanism.py [SEED ...] > mechanism.md      (needs PYTHONPATH=code/active_learning_program)"""
import sys, tempfile
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "new_directions"))
import nd_core as core, nd_graph_learner as G
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def stats(gd, label):
    n = len(gd.paths); e = gd.edges.numpy(); a = csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)); ncomp, comp = connected_components(a, directed=False)
    src = gd.source_class; ii = [(i, j) for i, j in e if i in src and j in src]
    same = np.mean([src[i] == src[j] for i, j in ii]) if ii else float("nan")
    rng = np.random.default_rng(0); cls = np.array(list(src.values())); base = np.mean([cls[rng.integers(len(cls))] == cls[rng.integers(len(cls))] for _ in range(20000)])
    return {"graph": label, "nodes": n, "undirected edges": len(e), "mean degree": round(2 * len(e) / n, 1), "components": ncomp, "largest component": int(np.bincount(comp).max()),
            "ideal nodes (label sources)": len(src), "ideal-ideal edges": len(ii), "same-class share of ideal-ideal edges": round(float(same), 3), "same-class share at random": round(float(base), 3)}


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or [7000]; rows = []
    with tempfile.TemporaryDirectory() as scratch:
        cache = core.feature_cache()
        for s in seeds:
            ctx = core.make_ctx(s, "A", scratch, cache)
            for k in (5, 10, 20): rows.append({"seed": s, **stats(G.graph_for(ctx, k), f"kNN k={k}")})
            rows.append({"seed": s, **stats(G.graph_for(ctx, 10, noise=3.0), "k=10, noisy features (n3)")}); rows.append({"seed": s, **stats(G.graph_for(ctx, 10, shuffle=True), "k=10, shuffled")})
    cols = list(rows[0]); print("| " + " | ".join(cols) + " |"); print("|" + "---|" * len(cols))
    for r in rows: print("| " + " | ".join(str(r[c]) for c in cols) + " |")
