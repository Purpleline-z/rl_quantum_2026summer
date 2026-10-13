"""Mechanistic probe (descriptive, development seeds 8000-8019, split A, budget 20, draw 0): how does the Laplacian penalty change the score of each type on HELD-OUT ideal images (the seed's outer-test ideals, never nodes
or anchors)?  For each type t: one-vs-rest AUC of the type-t score for images of class t, and the pairwise separation (AUC) of the RT13 score between RT13 and HTR images (the two classes whose ideal images are each other's
nearest neighbours in the graph).   usage: PYTHONPATH=../../../active_learning_program python3 gm_ideal_probe.py > gm_ideal_probe.md"""
import sys, tempfile
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import roc_auc_score
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parents[1] / "new_directions"))
import nd_core as core, nd_graph_learner as G, nd_graph_mechanism as M
torch.set_num_threads(1)
CLS = {k: v for k, v in core.frozen.TYPE_TO_INDEX.items() if v in (0, 2, 3, 4)}; NAME = {0: "(1x1)", 2: "c(6x2)", 3: "RT13", 4: "HTR"}
SPECS = {"aw8_all": {"lap": 0.0}, "lap3_k10-like (lambda 3)": {"lap": 3.0}, "uniform lambda 10": {"lap": 10.0}, "RT13 off (C1)": {"lap": 10.0, "type_w": {3: 0.0}}, "shuffled graph (lambda 10)": {"lap": 10.0, "shuffle": True}}
rows = {k: {"onevsrest": {t: [] for t in NAME}, "rt13_vs_htr": []} for k in SPECS}
with tempfile.TemporaryDirectory() as scratch:
    cache = core.feature_cache()
    for seed in range(8000, 8020):
        ctx = core.make_ctx(seed, "A", scratch, cache); lab = core.random_labelled(ctx, 20, 0); base = G.graph_for(ctx, 10); feats = {}; classes = list(ctx.exp.test_images)
        imgs = [(str(p), CLS[c]) for c in classes for p in ctx.exp.test_images[c] if c in CLS]; X = ctx.features.get([p for p, _ in imgs]); y = np.array([c for _, c in imgs])
        for name, spec in SPECS.items():
            if spec.get("lap", 0) > 0: b, e, w = M.gm_graph(ctx, "probe_" + name, spec)
            else: b, e, w = base, base.edges, torch.ones(len(base.edges))
            head = M.fit_gm(ctx, lab, seed * 7, b, e, w, spec)
            with torch.no_grad(): s = head(X)[:, [0, 2, 3, 4]].numpy()
            for k, t in enumerate((0, 2, 3, 4)):
                if len(set(y == t)) == 2: rows[name]["onevsrest"][t].append(roc_auc_score(y == t, s[:, k]))
            m = np.isin(y, (3, 4))
            if len(set(y[m])) == 2: rows[name]["rt13_vs_htr"].append(roc_auc_score(y[m] == 3, s[m, 2]))
print("| learner | " + " | ".join(f"{NAME[t]} one-vs-rest AUC" for t in NAME) + " | RT13 score: RT13 vs HTR images AUC | seeds |"); print("|---|---|---|---|---|---|---|")
for name in SPECS:
    r = rows[name]; print(f"| {name} | " + " | ".join(f"{np.mean(r['onevsrest'][t]):.4f}" for t in NAME) + f" | {np.mean(r['rt13_vs_htr']):.4f} | {len(r['rt13_vs_htr'])} |")
print("\n(held-out outer-test ideal images only; budget 20, one random label set per seed; descriptive, development seeds)")
