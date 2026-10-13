"""Ideal-image type classification on the paper's exact per-seed split (28 outer-test ideal images), controls and graph methods (graph-endpoint study, Prompt GE).

Labelled data = the REFERENCE images only (the absolute labels of the paper's protocol; the same labels the classification-baseline branch uses).  Utility-validation ideal images are never labelled
here; the outer-test ideal images are scored once.  Controls re-implemented from ``classification_baselines/baseline_protocol.py`` (read-only reference branch): cosine 1-NN on L2-normalised
frozen SimCLR features, and L2-logistic regression with C chosen by 5-fold CV inside the references.

Graph protocols (PROTOCOL EXTENSION, stated in PREREGISTRATION_GE*.md):
  inductive      the outer-test ideal images are NOT graph nodes; each attaches to its k nearest nodes (as in the preference studies)
  transductive   the outer-test ideal images are unlabelled graph nodes (looser protocol; reported separately, never mixed with inductive)
In both, the graph has the trajectory images + ideal images as nodes (minus the images of the seed's validation/test PAIR groups, as in nd_graph_learner).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

import nd_core as core
import nd_graph_learner as gl

CLASSES = ("(1 x 1)", "c(6 x 2)", "(√13 x √13)", "HTR")          # order of the baseline branch
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
ALPHA = gl.ALPHA


def split_arrays(ctx, cache=None):
    """references / outer-test images as (paths, labels as class index in CLASSES)."""
    flat = lambda table: ([str(p) for c in CLASSES if c in table for p in table[c]], np.array([CLASSES.index(c) for c in CLASSES if c in table for _ in table[c]]))
    return flat(ctx.exp.references), flat(ctx.exp.test_images), flat(ctx.exp.utility_images)


def feats(ctx, paths) -> np.ndarray:
    return np.stack([ctx.features.cache[p].numpy() for p in paths]).astype(np.float64)


# ------------------------------------------------------------------------------------------ controls (fit on references only)
def predict_1nn(ctx, ref, test) -> np.ndarray:
    (rp, ry), (tp, _) = ref, test; r = feats(ctx, rp); t = feats(ctx, tp); r = r / np.linalg.norm(r, axis=1, keepdims=True); t = t / np.linalg.norm(t, axis=1, keepdims=True)
    return ry[np.argmax(t @ r.T, axis=1)]


def predict_logreg(ctx, ref, test, seed: int = 0) -> np.ndarray:
    (rp, ry), (tp, _) = ref, test; r, t = feats(ctx, rp), feats(ctx, tp); mu, sd = r.mean(0), r.std(0) + 1e-8; zr, zt = (r - mu) / sd, (t - mu) / sd
    folds = max(2, min(5, int(np.bincount(ry).min()))); cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    scores = {c: cross_val_score(LogisticRegression(C=c, max_iter=2000), zr, ry, cv=cv).mean() for c in C_GRID}; best = max(scores.values()); c = min(k for k, s in scores.items() if s == best)
    return LogisticRegression(C=c, max_iter=5000).fit(zr, ry).predict(zt)


# ------------------------------------------------------------------------------------------ graphs for the classification endpoint
def build_graph(ctx, k: int = 10, transductive: bool = False, shuffle: bool = False, noise: float = 0.0, label_sources: str = "references"):
    """GraphData with (a) outer-test ideal images as unlabelled nodes iff ``transductive``, (b) labels only from the REFERENCE images (the utility-validation ideal images stay unlabelled nodes)."""
    original = gl.excluded_identities
    def allowed(c):
        paths = {str(x) for g in list(c.validation) + list(c.test) for r in c.exp.groups[g].itertuples() for x in (r.resolved_img1, r.resolved_img2)}
        if not transductive: paths |= {str(p) for ps in c.exp.test_images.values() for p in ps}
        return set(gl.content_hashes(sorted(paths)).values())
    gl.excluded_identities = allowed
    try: gd = gl.GraphData(ctx, k, shuffle=ctx.seed if shuffle else None, noise=noise)
    finally: gl.excluded_identities = original
    if label_sources == "references":
        gd.source_class = {gd.index[str(p)]: core.frozen.TYPE_TO_INDEX[c] for c, ps in ctx.exp.references.items() for p in ps if str(p) in gd.index}; gd._q = None
    return gd


def class_scores_from_q(gd, q: torch.Tensor, test_paths, transductive: bool, cache) -> np.ndarray:
    """(n_test, 4) class scores in the order of gl.ACTIVE = (0, 2, 3, 4) = (1x1, c6x2, sqrt13, HTR)."""
    if transductive: ids = [gd.index[p] for p in test_paths]; return q[ids].numpy()
    nb = gd.neighbours_of(test_paths, cache); return q[nb].mean(1).numpy()


ACTIVE_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3}   # column j of q <-> CLASSES[j] (ACTIVE order = CLASSES order by construction; checked in tests)


def predict_label_spreading(ctx, test, gd, transductive: bool) -> np.ndarray:
    return class_scores_from_q(gd, gd.q(), test[0], transductive, ctx.features.cache).argmax(1)


# SGC: logistic regression on one-step propagated features of the reference nodes
def predict_sgc(ctx, ref, test, gd, transductive: bool, seed: int = 0) -> np.ndarray:
    (rp, ry), (tp, _) = ref, test; cache = ctx.features.cache; px = gd.propagated_nodes().numpy().astype(np.float64)
    zr = px[[gd.index[p] for p in rp]]
    zt = px[[gd.index[p] for p in tp]] if transductive else gd.propagate_features(tp, cache).numpy().astype(np.float64)
    mu, sd = zr.mean(0), zr.std(0) + 1e-8; zr, zt = (zr - mu) / sd, (zt - mu) / sd
    folds = max(2, min(5, int(np.bincount(ry).min()))); cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    scores = {c: cross_val_score(LogisticRegression(C=c, max_iter=2000), zr, ry, cv=cv).mean() for c in C_GRID}; best = max(scores.values()); c = min(k_ for k_, s in scores.items() if s == best)
    return LogisticRegression(C=c, max_iter=5000).fit(zr, ry).predict(zt)


# 2-layer GCN node classifier, cross-entropy on the reference nodes (fixed hyper-parameters, no tuning on test or utility images)
def predict_gcn(ctx, ref, test, gd, transductive: bool, seed: int = 0, epochs: int = 200, hidden: int = 128, lr: float = 0.01, wd: float = 5e-4, dropout: float = 0.5) -> np.ndarray:
    (rp, ry), (tp, _) = ref, test; cache = ctx.features.cache; torch.manual_seed(seed)
    n = len(gd.paths); mu = gd.x.mean(0); sd = gd.x.std(0) + 1e-6; X = (gd.x - mu) / sd
    l1, l2 = nn.Linear(X.shape[1], hidden), nn.Linear(hidden, 4); params = list(l1.parameters()) + list(l2.parameters()); opt = torch.optim.Adam(params, lr=lr, weight_decay=wd)
    ridx = torch.as_tensor([gd.index[p] for p in rp]); y = torch.as_tensor(ry)
    def forward(train: bool):
        h = F.relu(torch.sparse.mm(gd.ahat, l1(F.dropout(X, dropout, train)))); return torch.sparse.mm(gd.ahat, l2(F.dropout(h, dropout, train)))
    for _ in range(epochs):
        opt.zero_grad(); loss = F.cross_entropy(forward(True)[ridx], y); loss.backward(); opt.step()
    with torch.no_grad():
        out = forward(False)
        if transductive: return out[[gd.index[p] for p in tp]].argmax(1).numpy()
        # inductive: a test image reads the average of the (propagated) outputs of its k nearest nodes together with its own un-propagated output
        nb = gd.neighbours_of(tp, cache); own = torch.stack([cache[p] for p in tp]).float(); own = (own - mu) / sd; so = l2(F.relu(l1(own)))
        return ((so + out[nb].sum(1)) / (gd.k + 1)).argmax(1).numpy()


def accuracy(truth: np.ndarray, pred: np.ndarray) -> dict:
    out = {"acc": float((truth == pred).mean()), "n_test": int(len(truth))}
    for j, c in enumerate(CLASSES): out[f"recall_{j}"] = float((pred[truth == j] == j).mean()) if (truth == j).any() else float("nan")
    out["htr_recall"] = out["recall_3"]; return out
