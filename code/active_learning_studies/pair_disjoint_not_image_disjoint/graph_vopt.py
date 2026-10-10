"""Graph-regularised pool-wide variance-reduction design (overnight run 2; plan: graph_exploration/OVERNIGHT2_PLAN.md).

Base rule: ``new_methods_strategies.vopt_u`` (I-optimal design on the last layer of the own-type head under the Laplace posterior; gain of candidate i
= w_i / (1 + w_i G_ii) * sum_j c_j G_ij^2, G = Phi Sigma Phi^T over the candidate pool).  The score of a judgment is theta_k . (h(a) - h(b)), i.e. f(a) - f(b)
with f(x) = theta_k . h(x) and h the hidden layer of the head, so a smoothness prior on f over the image graph is a prior on theta_k.

Graph variants (graph = cosine 10-NN graph over the images of the candidates, the revealed judgments and the typed reference images, as in graph_typed.ImageGraph):
  gvopt_lap    Laplacian-regularised prior (Laplacian-regularised optimal design, He et al. 2010; Cai & He 2012): prior precision = ridge*I + lam * M, with
               M = H^T L H rescaled to mean diagonal 1, H = hidden features of the graph's images, L = graph Laplacian, lam = 1.  Directions of theta along which f varies
               across graph edges get more prior precision, so the design concentrates on directions that are smooth over the graph.
  gvopt_type   the pool judgments' weights c_j in the variance sum are their type-posterior relevance (q_a(t) + q_b(t)) / 2 (label spreading of the typed references), mean-normalised.
  gvopt_prop   the design features are graph-propagated hidden features (one SGC step), the head's own predictions (w) stay unchanged.
  gvopt_sigma / gvopt_lapsigma  the Sigma-optimal criterion (variance of the pool-mean prediction; favours cluster centres where V-optimality favours outliers, Ma et al. 2013) without / with the Laplacian prior.
Controls (``*_shuffled``): the graph is destroyed by permuting node identities (adjacency, posterior), keeping the degree sequence and the feature distribution.
"""
from __future__ import annotations

import numpy as np
import torch

import graph_strategies as gs
import graph_typed as gt
import judgment_unit_strategies as ju
import new_methods_strategies as nm
from new_pair_strategies import ACTIVE_HEADS, _rank_one

LAM = 1.0


def _hidden_nodes(model, graph, cache) -> np.ndarray:
    head = model.reward_head.eval()
    with torch.no_grad(): return head[1](head[0](torch.stack([cache[p] for p in graph.paths]))).numpy().astype(np.float64)


def _shuffled(graph: gt.ImageGraph, seed: int):
    perm = np.random.default_rng(seed * 1_000_003 + 31).permutation(len(graph.paths))
    return graph.adjacency[np.ix_(perm, perm)], graph.q[perm]


def _build(cands, labeled, cache, seed, shuffle):
    ctx = nm.CTX["ctx"]; graph = gt.ImageGraph(cands, labeled, cache, gt.reference_types(ctx.exp), seed)
    adjacency, q = (_shuffled(graph, seed) if shuffle else (graph.adjacency, graph.q)); return graph, adjacency, q


def _pagerank_weight(cands, graph, adjacency):
    rank = gs._percentile(gs.pagerank(adjacency)); return np.array([(rank[graph.index[x["img1"]]] + rank[graph.index[x["img2"]]]) / 2 for x in cands])


def design(cands, labeled, model, cache, budget, seed, mode: str, shuffle: bool, ridge: float = 1.0, lam: float = LAM, crit: str = "v"):
    """Greedy pool-wide variance-reduction design with the graph modification ``mode`` in {'lap', 'type', 'prop', 'none'}; ``crit`` 'v' = V-optimal (sum of squared
    pool covariances, as vopt_u), 'sigma' = Sigma-optimal (squared sum of pool covariances: variance of the pool-mean prediction; Ma, Garnett and Schneider 2013)."""
    graph, adjacency, q = _build(cands, labeled, cache, seed, shuffle); n = len(cands)
    items = cands + labeled; h, p, k = ju._own(model, items, cache); w = p * (1 - p)
    c = np.ones(n)
    if mode == "prop":   # design features: propagated hidden features of the two images
        smooth = gs.propagate(adjacency, _hidden_nodes(model, graph, cache), 1); idx1 = np.array([graph.index[x["img1"]] for x in items]); idx2 = np.array([graph.index[x["img2"]] for x in items])
        h = smooth[idx1] - smooth[idx2]
    if mode == "type":
        t = np.array([gt.TYPES.index(int(x["type_idx"])) for x in cands]); qa = np.array([q[graph.index[x["img1"]], tt] for x, tt in zip(cands, t)]); qb = np.array([q[graph.index[x["img2"]], tt] for x, tt in zip(cands, t)])
        c = (qa + qb) / 2 + 1e-3; c = c / c.mean()
    d = h.shape[1]
    if mode == "lap":
        H = _hidden_nodes(model, graph, cache); L = np.diag(adjacency.sum(1)) - adjacency; M = H.T @ L @ H; M = M * d / max(np.trace(M), 1e-12)
        base = np.linalg.inv(ridge * np.eye(d) + lam * M); inverse = {kk: base.copy() for kk in ACTIVE_HEADS}
        for i in range(n, len(h)): inverse[k[i]] = _rank_one(inverse[k[i]], h[i], w[i])
    else:
        inverse = ju._inverses(h, w, k, n, ridge)
    members = {kk: np.flatnonzero(k[:n] == kk) for kk in ACTIVE_HEADS}
    def gains(kk):
        idx = members[kk]
        if len(idx) == 0: return idx, np.zeros(0)
        P = h[idx]; G = P @ inverse[kk] @ P.T; own = np.diag(G)
        if crit == "sigma": return idx, w[idx] / (1 + w[idx] * own) * ((c[idx][:, None] * G).sum(0)) ** 2
        return idx, w[idx] / (1 + w[idx] * own) * ((c[idx][:, None] * G ** 2).sum(0))
    score = np.full(n, -np.inf)
    for kk in ACTIVE_HEADS: idx, g = gains(kk); score[idx] = g
    alive = np.ones(n, bool); chosen = []
    for _ in range(min(budget, n)):
        pick = int(np.argmax(np.where(alive, score, -np.inf))); chosen.append(pick); alive[pick] = False; kk = k[pick]
        inverse[kk] = _rank_one(inverse[kk], h[pick], w[pick]); idx, g = gains(kk); score[idx] = np.where(alive[idx], g, -np.inf)
    return [cands[i] for i in chosen]


def _rule(mode, shuffle, lam=LAM, crit="v"):
    def select(cands, labeled, model, cache, budget, seed=0): return design(cands, labeled, model, cache, budget, seed, mode, shuffle, lam=lam, crit=crit)
    return select


NEW = {"gvopt_lap": _rule("lap", False), "gvopt_lap_shuffled": _rule("lap", True), "gvopt_type": _rule("type", False), "gvopt_type_shuffled": _rule("type", True),
       "gvopt_prop": _rule("prop", False), "gvopt_prop_shuffled": _rule("prop", True), "gvopt_lap3": _rule("lap", False, 3.0), "gvopt_lap3_shuffled": _rule("lap", True, 3.0),
       "gvopt_lap03": _rule("lap", False, 0.3), "gvopt_lap10": _rule("lap", False, 10.0),
       "gvopt_sigma": _rule("none", False, crit="sigma"), "gvopt_lapsigma": _rule("lap", False, crit="sigma"), "gvopt_lapsigma_shuffled": _rule("lap", True, crit="sigma")}
