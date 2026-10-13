"""Graph-aware selectors for the selection x learner x HTR study (branch claude/selection-htr-*).

All of them see only what the repository's selectors may see (candidate judgments, revealed judgments, the model trained on the initial judgments, cached features, the typed reference images)
plus the *unlabelled* image nodes of the graph used by the Laplacian learner (``nd_graph_learner.GraphData``: all trajectory images and ideal images, validation/test pair-group images and
outer-test ideal images removed by content hash).  No label of any candidate is used.  Every graph rule has a ``_shuf`` control: node identities of the adjacency are permuted
(degree sequence kept) so that the graph quantities used by the rule are noise with the same marginal distribution.

  gdens_unc[_shuf]      information-density-weighted uncertainty (Settles & Craven 2008): own-head uncertainty x (0.25 + graph density of the two images); density of an image = mean cosine
                        similarity to its graph neighbours, percentile-ranked.
  gcover[_shuf]         per-type greedy maximum coverage of the graph nodes: a judgment of type t covers the graph neighbourhoods of its two images for head t; gain = newly covered nodes.
  gcover_dec[_shuf]     the same times P(decisive) of the earlier type-aware rules (graph_typed).
  gvopt_node{lam}[_shuf]  pool-wide I-optimal design (vopt_u) under the Laplacian prior of the *node* graph, i.e. the design counterpart of the Laplacian learner: prior precision
                        ridge*I + lam*M, M = H^T L H over the hidden features of all nodes, rescaled to mean diagonal 1 (He et al. 2010; Cai & He 2012; as graph_vopt.gvopt_lap but with the full node set).
  vopt_htr{w}           vopt_u with the gain of the HTR head multiplied by w (budget moves to HTR judgments).
  vopt_bridge{g}[_shuf] vopt_u with the gain of HTR / RT13 judgments multiplied by 1 + g * bridge, bridge of an image = 2 min(q_HTR, q_RT13) of the label-spreading posterior of the typed
                        reference images (images between the two types); mean of the two images, mean-normalised over the HTR/RT13 candidates.
"""
from __future__ import annotations

import numpy as np
import torch

import graph_typed as gt
import judgment_unit_strategies as ju
import new_methods_strategies as nm
import nd_graph_learner as G
from new_pair_strategies import ACTIVE_HEADS, _rank_one

HTR, RT13 = 4, 3          # type indices (core.TYPE_NAMES)
ACTIVE = G.ACTIVE          # (0, 2, 3, 4)


def _gd(shuffle: bool):
    return G.graph_for(nm.CTX["ctx"], 10, shuffle)


def _near(gd, path, cache, shuffle: bool):
    """Node indices standing in for an image that is not a node (its content also occurs in a validation/test group): its k nearest nodes (random nodes in the shuffled control)."""
    key = str(path)
    if key not in gd.__dict__.setdefault("_ext", {}):
        nb = gd.neighbours_of([path], cache)[0].numpy()
        gd._ext[key] = np.random.default_rng(abs(hash(key)) % (2 ** 31)).integers(0, len(gd.paths), len(nb)) if shuffle else nb
    return gd._ext[key]


def _scalar(gd, vec, paths, cache, shuffle):
    """Per-image value of a node vector: the node's own value, or the mean over the stand-in nodes."""
    return np.array([vec[gd.index[str(p)]] if str(p) in gd.index else vec[_near(gd, p, cache, shuffle)].mean() for p in paths])


def _reach(gd, path, cache, shuffle):
    """Boolean node vector of the graph neighbourhood (including itself) of an image."""
    out = np.zeros(len(gd.paths), bool)
    if str(path) in gd.index: i = gd.index[str(path)]; out |= gd.adj[i].numpy(); out[i] = True
    else: out[_near(gd, path, cache, shuffle)] = True
    return out


def _density(gd) -> np.ndarray:
    """Percentile-ranked mean similarity to graph neighbours (computed through the adjacency, so a shuffled graph gives noise)."""
    if not hasattr(gd, "_dens"):
        a = gd.adj.numpy().astype(np.float64); xn = gd.xn.numpy().astype(np.float64); sim = a * (xn @ xn.T); d = np.maximum(a.sum(1), 1.0); m = sim.sum(1) / d
        gd._dens = (np.argsort(np.argsort(m)) + 0.5) / len(m)
    return gd._dens


def gdens_unc(cands, labeled, model, cache, budget, seed=0, shuffle=False):
    gd = _gd(shuffle); r = _density(gd); dens = (_scalar(gd, r, [x["img1"] for x in cands], cache, shuffle) + _scalar(gd, r, [x["img2"] for x in cands], cache, shuffle)) / 2
    score = ju._own_uncertainty(model, cands, cache) * (0.25 + dens)
    return [cands[i] for i in np.argsort(-score, kind="stable")[:budget]]


def _coverage(cands, labeled, cache, budget, seed, shuffle, decisive: bool):
    gd = _gd(shuffle); R = lambda p: _reach(gd, p, cache, shuffle)
    reach = np.stack([R(x["img1"]) | R(x["img2"]) for x in cands])                              # (cands, nodes): nodes covered by each candidate
    covered = {t: np.zeros(len(gd.paths), bool) for t in ACTIVE}
    for x in labeled:
        t = int(x["type_idx"])
        if t in covered: covered[t] |= R(x["img1"]) | R(x["img2"])
    p = np.ones(len(cands))
    if decisive:
        g = gt.ImageGraph(cands, labeled, cache, gt.reference_types(nm.CTX["ctx"].exp), seed, shuffle); p = gt.decisive_probability(g, cands, labeled)
    kind = np.array([int(x["type_idx"]) for x in cands]); chosen = []; alive = np.ones(len(cands), bool)
    for _ in range(min(budget, len(cands))):
        gain = np.array([(reach[i] & ~covered[kind[i]]).sum() if alive[i] else -1.0 for i in range(len(cands))]) * p
        if gain.max() <= 0: gain = np.where(alive, p + 1e-9, -1.0)                                 # everything covered: fall back to the highest weight
        i = int(np.argmax(gain)); chosen.append(i); alive[i] = False; covered[kind[i]] |= reach[i]
    return [cands[i] for i in chosen]


def gcover(cands, labeled, model, cache, budget, seed=0, shuffle=False): return _coverage(cands, labeled, cache, budget, seed, shuffle, False)
def gcover_dec(cands, labeled, model, cache, budget, seed=0, shuffle=False): return _coverage(cands, labeled, cache, budget, seed, shuffle, True)


def _node_hidden(model, gd) -> np.ndarray:
    head = model.reward_head.eval()
    with torch.no_grad(): return head[1](head[0](gd.x)).numpy().astype(np.float64)


def gvopt_node(cands, labeled, model, cache, budget, seed=0, lam: float = 3.0, shuffle: bool = False, ridge: float = 1.0):
    gd = _gd(shuffle); items = cands + labeled; h, p, k = ju._own(model, items, cache); n = len(cands); w = p * (1 - p); d = h.shape[1]
    a = gd.adj.numpy().astype(np.float64); L = np.diag(a.sum(1)) - a; H = _node_hidden(model, gd); M = H.T @ L @ H; M = M * d / max(np.trace(M), 1e-12)
    base = np.linalg.inv(ridge * np.eye(d) + lam * M); inverse = {kk: base.copy() for kk in ACTIVE_HEADS}
    for i in range(n, len(h)): inverse[k[i]] = _rank_one(inverse[k[i]], h[i], w[i])
    members = {kk: np.flatnonzero(k[:n] == kk) for kk in ACTIVE_HEADS}
    def gains(kk):
        idx = members[kk]
        if len(idx) == 0: return idx, np.zeros(0)
        P = h[idx]; G_ = P @ inverse[kk] @ P.T; own = np.diag(G_); return idx, w[idx] / (1 + w[idx] * own) * (G_ ** 2).sum(0)
    score = np.full(n, -np.inf)
    for kk in ACTIVE_HEADS: idx, g = gains(kk); score[idx] = g
    alive = np.ones(n, bool); chosen = []
    for _ in range(min(budget, n)):
        pick = int(np.argmax(np.where(alive, score, -np.inf))); chosen.append(pick); alive[pick] = False; kk = k[pick]
        inverse[kk] = _rank_one(inverse[kk], h[pick], w[pick]); idx, g = gains(kk); score[idx] = np.where(alive[idx], g, -np.inf)
    return [cands[i] for i in chosen]


def vopt_htr(cands, labeled, model, cache, budget, seed=0, weight: float = 2.0):
    return nm.vopt(cands, labeled, model, cache, budget, seed, sensitivity=False, head_weight={HTR: weight})


def vopt_bridge(cands, labeled, model, cache, budget, seed=0, gamma: float = 2.0, shuffle: bool = False):
    gd = _gd(shuffle); q = gd.q().numpy(); bridge_node = 2 * np.minimum(q[:, ACTIVE.index(HTR)], q[:, ACTIVE.index(RT13)])
    def factor(c, l, ca):
        b = (_scalar(gd, bridge_node, [x["img1"] for x in c], ca, shuffle) + _scalar(gd, bridge_node, [x["img2"] for x in c], ca, shuffle)) / 2; t = np.array([int(x["type_idx"]) for x in c]); on = np.isin(t, (HTR, RT13))
        out = np.ones(len(c))
        if on.any(): out[on] = 1 + gamma * b[on] / max(float(b[on].mean()), 1e-9)
        return out
    return nm.vopt(cands, labeled, model, cache, budget, seed, sensitivity=False, informative=factor)


SELECTORS = {}
for _s in (False, True):
    _sfx = "_shuf" if _s else ""
    SELECTORS["gdens_unc" + _sfx] = (lambda sh: lambda *a, **kw: gdens_unc(*a, shuffle=sh, **kw))(_s)
    SELECTORS["gcover" + _sfx] = (lambda sh: lambda *a, **kw: gcover(*a, shuffle=sh, **kw))(_s)
    SELECTORS["gcover_dec" + _sfx] = (lambda sh: lambda *a, **kw: gcover_dec(*a, shuffle=sh, **kw))(_s)
    for _lam in (1.0, 3.0, 10.0): SELECTORS[f"gvopt_node{_lam:g}" + _sfx] = (lambda sh, lm: lambda *a, **kw: gvopt_node(*a, lam=lm, shuffle=sh, **kw))(_s, _lam)
    for _g in (1.0, 3.0): SELECTORS[f"vopt_bridge{_g:g}" + _sfx] = (lambda sh, gm: lambda *a, **kw: vopt_bridge(*a, gamma=gm, shuffle=sh, **kw))(_s, _g)
for _w in (2.0, 4.0): SELECTORS[f"vopt_htr{_w:g}"] = (lambda wt: lambda *a, **kw: vopt_htr(*a, weight=wt, **kw))(_w)
