"""Graph-aware acquisition strategies for the judgment-unit study (branch ``claude/graph-active-selection``; plan: ``graph_exploration/GRAPH_PLAN.md``).

The graph is over IMAGES and is label-free: nodes are the images of the candidates and of the labelled judgments (nothing else is visible to a selector, so no
validation/test image can enter), edges are the k nearest neighbours in the cached encoder space, symmetrised.  Comparison outcomes are not used to build it
(the comparison graph is nearly a matching, see GRAPH_PLAN.md).  Selectors have the usual signature ``select(cands, labeled, model, cache, budget, seed)``.

Row-level rules (score of a judgment = own-head uncertainty x a graph factor in (0, 1]; the graph factor is a percentile rank, so only the order matters):
  graph_centrality_uncertainty   PageRank of the pair's two images (typical, well-connected images)
  graph_bridge_uncertainty       share of an image's neighbours that fall in another k-means cluster (images on a boundary between regions)
Pair-level rule (one judgment of each chosen pair, type at random):
  graph_core_set                 farthest-first k-centre on [(a+b)/2, |a-b|, a*b] of graph-propagated image features (SGC: A_hat^K X)
Controls, identical except that the graph information is destroyed:
  *_shuffled                     the per-image graph scores (or the propagated features) are permuted over the nodes, seeded
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import KMeans

import judgment_unit_strategies as ju
import new_pair_strategies as nps

NEIGHBOURS = 10; PROPAGATION_STEPS = 2; BRIDGE_CLUSTERS = 8


# ---------------------------------------------------------------------------------------------------- graph
def image_paths(*item_lists) -> list[str]:
    """Sorted unique images of the given candidate/labelled dicts (sorted so that results do not depend on the order of the candidates)."""
    return sorted({p for items in item_lists for x in items for p in (x["img1"], x["img2"])})


def knn_graph(features: np.ndarray, k: int = NEIGHBOURS) -> np.ndarray:
    """Symmetric 0/1 adjacency of the cosine k-nearest-neighbour graph (no self loops)."""
    k = min(k, len(features) - 1); z = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    sim = z @ z.T; np.fill_diagonal(sim, -np.inf); nearest = np.argsort(-sim, axis=1, kind="stable")[:, :k]
    adjacency = np.zeros((len(features), len(features))); adjacency[np.repeat(np.arange(len(features)), k), nearest.ravel()] = 1.0
    return np.maximum(adjacency, adjacency.T)


def pagerank(adjacency: np.ndarray, damping: float = .85, iterations: int = 100) -> np.ndarray:
    degree = adjacency.sum(1); transition = adjacency / np.maximum(degree, 1)[:, None]; n = len(adjacency); rank = np.full(n, 1 / n)
    for _ in range(iterations):
        rank = (1 - damping) / n + damping * (rank @ transition + rank[degree == 0].sum() / n)
    return rank


def bridge_score(adjacency: np.ndarray, features: np.ndarray, seed: int) -> np.ndarray:
    """Fraction of each node's neighbours that belong to a different k-means cluster than the node (0 = interior, 1 = pure boundary)."""
    clusters = KMeans(n_clusters=min(BRIDGE_CLUSTERS, len(features)), n_init=5, random_state=seed).fit_predict(features)
    differs = (clusters[:, None] != clusters[None]) & (adjacency > 0)
    return differs.sum(1) / np.maximum(adjacency.sum(1), 1)


def propagate(adjacency: np.ndarray, features: np.ndarray, steps: int = PROPAGATION_STEPS) -> np.ndarray:
    """SGC propagation H = A_hat^steps X with A_hat = D^-1/2 (A + I) D^-1/2 (label-free)."""
    a = adjacency + np.eye(len(adjacency)); d = a.sum(1) ** -.5; a_hat = a * d[:, None] * d[None]; h = features.copy()
    for _ in range(steps): h = a_hat @ h
    return h


def _percentile(values: np.ndarray) -> np.ndarray:
    """Rank in (0, 1]: ties get the mean rank, so equal scores stay equal."""
    order = np.argsort(np.argsort(values, kind="stable"), kind="stable").astype(float); out = np.empty(len(values))
    for v in np.unique(values): out[values == v] = order[values == v].mean()
    return (out + 1) / len(values)


def _image_features(cands, labeled, cache):
    paths = image_paths(cands, labeled); x = np.stack([nps._feature(cache, p) for p in paths]).astype(np.float64)
    return paths, x


def node_scores(kind: str, cands, labeled, cache, seed: int, shuffled: bool) -> dict[str, float]:
    """Percentile-ranked graph score of every image in ``cands + labeled``."""
    paths, x = _image_features(cands, labeled, cache); adjacency = knn_graph(x)
    raw = pagerank(adjacency) if kind == "centrality" else bridge_score(adjacency, x, seed)
    if shuffled: raw = raw[np.random.default_rng(seed * 1_000_003 + 17).permutation(len(raw))]
    score = _percentile(raw); return dict(zip(paths, score))


# ---------------------------------------------------------------------------------------------------- row-level rules
def _graph_uncertainty(kind: str, shuffled: bool):
    def select(cands, labeled, model, cache, budget, seed=0):
        unc = ju._own_uncertainty(model, cands, cache); score = node_scores(kind, cands, labeled, cache, seed, shuffled)
        factor = np.array([(score[c["img1"]] + score[c["img2"]]) / 2 for c in cands])
        return [cands[i] for i in np.argsort(-(unc * factor), kind="stable")[:budget]]
    return select


# ---------------------------------------------------------------------------------------------------- pair-level rule
def _graph_core_set(shuffled: bool):
    def rule(pairs, labeled, u, cache, need, seed, model):
        paths, x = _image_features(pairs, labeled, cache); h = propagate(knn_graph(x), x)
        if shuffled: h = h[np.random.default_rng(seed * 1_000_003 + 19).permutation(len(h))]
        where = {p: i for i, p in enumerate(paths)}
        def vectors(items):
            a = np.stack([h[where[t["img1"]]] for t in items]); b = np.stack([h[where[t["img2"]]] for t in items]); return np.concatenate([(a + b) / 2, np.abs(a - b), a * b], axis=1)
        allp = np.concatenate([vectors(pairs), vectors(labeled)]) if len(labeled) else vectors(pairs)
        allp = (allp - allp.mean(0)) / (allp.std(0) + 1e-6); allp = allp / np.sqrt(allp.shape[1]); cand, lab = allp[:len(pairs)], allp[len(pairs):]
        covered = lab.copy(); chosen = []; remaining = list(range(len(pairs)))
        for _ in range(min(need, len(pairs))):
            dist = np.sqrt(((cand[remaining][:, None] - covered[None]) ** 2).sum(2)).min(1) if len(covered) else np.linalg.norm(cand[remaining] - cand.mean(0), axis=1)
            pick = remaining[int(np.argmax(dist))]; chosen.append(pick); remaining.remove(pick); covered = np.vstack([covered, cand[pick]])
        return chosen
    return rule


ROW_RULES = {"graph_centrality_uncertainty": _graph_uncertainty("centrality", False), "graph_centrality_uncertainty_shuffled": _graph_uncertainty("centrality", True),
             "graph_bridge_uncertainty": _graph_uncertainty("bridge", False), "graph_bridge_uncertainty_shuffled": _graph_uncertainty("bridge", True)}
PAIR_RULES = {"graph_core_set": _graph_core_set(False), "graph_core_set_shuffled": _graph_core_set(True)}
GRAPH_NAMES = tuple(ROW_RULES) + tuple(PAIR_RULES)


def make_selector(name: str):
    if name in ROW_RULES: return ROW_RULES[name]
    if name in PAIR_RULES: return ju.pair_level(PAIR_RULES[name], False)
    raise ValueError(name)
