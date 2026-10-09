"""Type-aware graph acquisition rules (judgment-unit study; autonomous run, see graph_exploration/AUTONOMOUS_RUN_LOG.md).

Graph: nodes = images of the candidate and revealed judgments plus the typed REFERENCE (ideal) images, which every strategy already uses as training anchors; edges = symmetrised cosine
kNN (k = 10) on mean-centred cached SimCLR features.  No validation/test image and no label of any candidate is used.
Type posterior: label propagation (Zhou et al. 2004, F = (I - a S)^-1 Y, a = .9) of the one-hot type of the reference images over the graph, normalised per node: q_i(t), t in the four active types.
Judgment features (a pair a,b judged for type t): q_a(t), q_b(t), |q_a(t) - q_b(t)|, q_a(t) + q_b(t), min, max, ||q_a - q_b||_1, q_a . q_b  (8 numbers; a data-level diagnostic found that these
predict whether a judgment is decisive, tie or not_apply better than the 1024-d relation features, with AUC 0.83 / 0.88 / 0.90 for decisive / tie / not_apply on all 521 rows).
Rules (row level = a candidate is one (pair, type) judgment):
  typed_decisive_uncertainty   P(decisive) x own-head uncertainty
  typed_decisive_bald          BALD under the last-layer Laplace posterior x P(decisive)   (graph-informed `bald_decisive`)
  typed_decisive_coverage      greedy farthest-first in graph-propagated pair space, distance x P(decisive)
  coregcn, uncertaingcn        Sequential GCN (Caramalau et al., CVPR 2021) on the image graph, applied to pairs (pair level; one judgment per chosen pair)
P(decisive) is a ridge logistic regression on the revealed judgments' outcomes (decisive vs tie/not_apply) with the features above; 0.5 before both outcomes are seen.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

import graph_strategies as gs
import judgment_unit_strategies as ju
import new_pair_strategies as nps
from pairwise_active_learning_pipeline import TYPE_TO_INDEX

TYPES = (0, 2, 3, 4)
K_NEIGHBOURS = 10; ALPHA = .9


def reference_types(exp) -> dict[str, int]:
    """path -> active type index of the typed reference images (empty if there is no experiment object, as in the synthetic unit tests)."""
    if exp is None: return {}
    return {str(p): TYPE_TO_INDEX[name] for name, paths in exp.references.items() if TYPE_TO_INDEX.get(name) in TYPES for p in paths}


class ImageGraph:
    """kNN graph over candidate, revealed and reference images, with the propagated type posterior."""
    def __init__(self, cands, labeled, cache, refs: dict[str, int], seed: int = 0, shuffle: bool = False):
        refs = {p: t for p, t in refs.items() if p in cache}
        self.paths = sorted({p for items in (cands, labeled) for x in items for p in (x["img1"], x["img2"])} | set(refs)); self.index = {p: i for i, p in enumerate(self.paths)}
        x = np.stack([nps._feature(cache, p) for p in self.paths]).astype(np.float64); x = x - x.mean(0); self.features = x
        self.adjacency = gs.knn_graph(x, K_NEIGHBOURS)
        self.q = self._type_posterior(refs)
        if shuffle:  # control: the posterior of every image is replaced by that of a random other image
            self.q = self.q[np.random.default_rng(seed * 1_000_003 + 23).permutation(len(self.q))]

    def _type_posterior(self, refs) -> np.ndarray:
        n = len(self.paths); y = np.zeros((n, len(TYPES)))
        for p, t in refs.items(): y[self.index[p], TYPES.index(t)] = 1.0
        if not refs: return np.full((n, len(TYPES)), 1 / len(TYPES))
        d = np.maximum(self.adjacency.sum(1), 1); s = self.adjacency / np.sqrt(np.outer(d, d))
        f = np.maximum(np.linalg.solve(np.eye(n) - ALPHA * s, y), 0); return f / np.maximum(f.sum(1, keepdims=True), 1e-9)

    def judgment_features(self, items) -> np.ndarray:
        out = []
        for x in items:
            qa, qb = self.q[self.index[x["img1"]]], self.q[self.index[x["img2"]]]; k = TYPES.index(int(x["type_idx"]))
            out.append([qa[k], qb[k], abs(qa[k] - qb[k]), qa[k] + qb[k], min(qa[k], qb[k]), max(qa[k], qb[k]), np.abs(qa - qb).sum(), float(qa @ qb)])
        return np.array(out)


def decisive_probability(graph: ImageGraph, cands, labeled, c: float = 1.0) -> np.ndarray:
    """P(decisive) of every candidate; ridge logistic on the revealed judgments (type one-hot + 8 graph features), 0.5 until both outcomes occurred."""
    if not labeled: return np.full(len(cands), .5)
    y = np.array([int(x["outcomes"][0][1]) for x in labeled])
    if len(set(y)) < 2: return np.full(len(cands), .5)
    def design(items, mu=None, sd=None):
        f = graph.judgment_features(items)
        if mu is None: mu, sd = f.mean(0), f.std(0) + 1e-6
        return np.column_stack([np.eye(5)[[int(x["type_idx"]) for x in items]][:, list(TYPES)], (f - mu) / sd]), mu, sd
    xl, mu, sd = design(labeled); xc, _, _ = design(cands, mu, sd)
    return LogisticRegression(C=c, max_iter=2000).fit(xl, y).predict_proba(xc)[:, 1]


# ---------------------------------------------------------------------------------------------------- row-level rules
def _graph(cands, labeled, cache, refs, seed, shuffle): return ImageGraph(cands, labeled, cache, refs, seed, shuffle)


def typed_decisive_uncertainty(refs, shuffle=False):
    def select(cands, labeled, model, cache, budget, seed=0):
        g = _graph(cands, labeled, cache, refs, seed, shuffle); p = decisive_probability(g, cands, labeled); u = ju._own_uncertainty(model, cands, cache)
        return [cands[i] for i in np.argsort(-(p * u), kind="stable")[:budget]]
    return select


def typed_decisive_bald(refs, shuffle=False):
    def select(cands, labeled, model, cache, budget, seed=0):
        g = _graph(cands, labeled, cache, refs, seed, shuffle); p = decisive_probability(g, cands, labeled)
        return ju._bald_greedy(cands, labeled, model, cache, budget, seed, 1.0, 64, p)
    return select


def _pair_space(graph: ImageGraph, items, steps: int = gs.PROPAGATION_STEPS):
    h = gs.propagate(graph.adjacency, graph.features, steps); a = np.stack([h[graph.index[x["img1"]]] for x in items]); b = np.stack([h[graph.index[x["img2"]]] for x in items])
    return np.concatenate([(a + b) / 2, np.abs(a - b), a * b], axis=1)


def typed_decisive_coverage(refs, shuffle=False, gamma: float = 1.0, uncertainty_power: float = 0.0, q_weight: float = 0.0):
    def select(cands, labeled, model, cache, budget, seed=0):
        g = _graph(cands, labeled, cache, refs, seed, shuffle); p = decisive_probability(g, cands, labeled) ** gamma
        if uncertainty_power: p = p * (0.25 + ju._own_uncertainty(model, cands, cache)) ** uncertainty_power
        allp = _pair_space(g, cands + labeled); allp = (allp - allp.mean(0)) / (allp.std(0) + 1e-6); allp = allp / np.sqrt(allp.shape[1])
        onehot = np.eye(5)[[int(x["type_idx"]) for x in cands + labeled]][:, list(TYPES)]; allp = np.concatenate([allp, onehot], axis=1)
        if q_weight:  # coordinates of the pair in type-posterior space: mean and absolute difference of the two images' posteriors
            qa = np.stack([g.q[g.index[x["img1"]]] for x in cands + labeled]); qb = np.stack([g.q[g.index[x["img2"]]] for x in cands + labeled])
            allp = np.concatenate([allp, q_weight * (qa + qb) / 2, q_weight * np.abs(qa - qb)], axis=1)
        cand, lab = allp[:len(cands)], allp[len(cands):]; covered = lab.copy(); chosen = []; remaining = list(range(len(cands)))
        for _ in range(min(budget, len(cands))):
            dist = np.sqrt(((cand[remaining][:, None] - covered[None]) ** 2).sum(2)).min(1) if len(covered) else np.linalg.norm(cand[remaining] - cand.mean(0), axis=1)
            pick = remaining[int(np.argmax(dist * p[remaining]))]; chosen.append(pick); remaining.remove(pick); covered = np.vstack([covered, cand[pick]])
        return [cands[i] for i in chosen]
    return select


def typed_decisive_probcover(refs, shuffle=False, delta_q: float = 0.05, uncertainty_power: float = 0.0):
    """ProbCover-style (Yehuda et al. 2022) greedy weighted maximum cover in the graph-propagated pair space.
    A candidate covers the candidates within radius delta (delta = the delta_q quantile of all pairwise distances); the gain of a pick is the summed P(decisive) (times the head's
    uncertainty factor if uncertainty_power > 0) of the not-yet-covered candidates it covers.  Revealed judgments start as covered centres.  Unlike farthest-first it prefers dense regions, not outliers.
    When everything is covered the remaining picks fall back to the highest weight."""
    def select(cands, labeled, model, cache, budget, seed=0):
        g = _graph(cands, labeled, cache, refs, seed, shuffle); w = decisive_probability(g, cands, labeled)
        if uncertainty_power: w = w * (0.25 + ju._own_uncertainty(model, cands, cache)) ** uncertainty_power
        allp = _pair_space(g, cands + labeled); allp = (allp - allp.mean(0)) / (allp.std(0) + 1e-6); allp = allp / np.sqrt(allp.shape[1])
        onehot = np.eye(5)[[int(x["type_idx"]) for x in cands + labeled]][:, list(TYPES)]; allp = np.concatenate([allp, onehot], axis=1)
        cand, lab = allp[:len(cands)], allp[len(cands):]; n = len(cands)
        d = np.sqrt(((cand[:, None] - cand[None]) ** 2).sum(2)); delta = np.quantile(d[np.triu_indices(n, 1)], delta_q); near = d <= delta
        covered = np.zeros(n, bool)
        if len(lab): covered |= (np.sqrt(((cand[:, None] - lab[None]) ** 2).sum(2)) <= delta).any(1)
        chosen = []
        for _ in range(min(budget, n)):
            gain = (near & ~covered[None]).astype(float) @ w; gain[chosen] = -1
            pick = int(np.argmax(gain)) if gain.max() > 0 else int(np.argmax(np.where(np.isin(np.arange(n), chosen), -1, w)))
            chosen.append(pick); covered |= near[pick]
        return [cands[i] for i in chosen]
    return select


def typed_decisive_sampling(refs, shuffle=False, gamma: float = 2.0):
    """Random-like selection that favours judgments likely to be decisive: Gumbel-top-k sampling without replacement with weights P(decisive)^gamma (keeps Random's representativeness)."""
    def select(cands, labeled, model, cache, budget, seed=0):
        g = _graph(cands, labeled, cache, refs, seed, shuffle); p = decisive_probability(g, cands, labeled)
        key = gamma * np.log(np.clip(p, 1e-6, 1)) + np.random.default_rng(seed * 1_000_003 + 29).gumbel(size=len(cands))
        return [cands[i] for i in np.argsort(-key, kind="stable")[:budget]]
    return select


# ---------------------------------------------------------------------------------------------------- Sequential GCN (Caramalau et al. 2021)
class _GCN(nn.Module):
    def __init__(self, nfeat, nhid=128, dropout=.3):
        super().__init__(); self.w1 = nn.Linear(nfeat, nhid); self.w2 = nn.Linear(nhid, 1); self.dropout = dropout

    def forward(self, x, adj):
        hidden = torch.relu(adj @ self.w1(x)); feat = torch.nn.functional.dropout(hidden, self.dropout, self.training)
        return torch.sigmoid(adj @ self.w2(feat)).squeeze(-1), feat


def _gcn_scores(features: np.ndarray, labelled: np.ndarray, seed: int, steps: int = 200, lr: float = 1e-3, weight_decay: float = 5e-4, lam: float = 1.0):
    """Train the labelled-vs-pool GCN (loss -mean log s(labelled) - lam * mean log(1 - s(pool))); return s and the first-layer embedding."""
    torch.manual_seed(seed); x = torch.as_tensor(features / np.linalg.norm(features, axis=1, keepdims=True), dtype=torch.float32)
    sim = x @ x.T; sim = sim - torch.diag(torch.diag(sim)); adj = sim / sim.sum(1, keepdim=True).clamp(min=1e-9) + torch.eye(len(x))   # A = D^-1 (S - I) + I
    net = _GCN(x.shape[1]); optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay); lab = torch.as_tensor(labelled, dtype=torch.bool)
    for _ in range(steps):
        net.train(); s, _ = net(x, adj); s = s.clamp(1e-6, 1 - 1e-6)
        loss = -torch.log(s[lab]).mean() - lam * torch.log(1 - s[~lab]).mean() if lab.any() and (~lab).any() else s.sum() * 0
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    net.eval()
    with torch.no_grad(): s, feat = net(x, adj)
    return s.numpy(), feat.numpy()


def _gcn_pair_rule(kind: str, lr: float = 1e-3):
    def rule(pairs, labeled, u, cache, need, seed, model):
        paths, x = gs._image_features(pairs, labeled, cache); x = x - x.mean(0); where = {p: i for i, p in enumerate(paths)}
        is_labelled = np.zeros(len(paths), bool)
        for item in labeled: is_labelled[where[item["img1"]]] = is_labelled[where[item["img2"]]] = True
        s, feat = _gcn_scores(x, is_labelled, seed, lr=lr)
        if kind == "uncertain":   # unlabelled-looking images first: lowest labelled-score of the pair's two images
            score = np.array([1 - (s[where[p["img1"]]] + s[where[p["img2"]]]) / 2 for p in pairs]); return list(np.argsort(-score, kind="stable")[:need])
        def vectors(items):
            a = np.stack([feat[where[t["img1"]]] for t in items]); b = np.stack([feat[where[t["img2"]]] for t in items]); return np.concatenate([(a + b) / 2, np.abs(a - b)], axis=1)
        cand = vectors(pairs); covered = vectors(labeled) if labeled else np.empty((0, cand.shape[1])); chosen = []; remaining = list(range(len(pairs)))
        for _ in range(min(need, len(pairs))):
            dist = np.sqrt(((cand[remaining][:, None] - covered[None]) ** 2).sum(2)).min(1) if len(covered) else np.linalg.norm(cand[remaining] - cand.mean(0), axis=1)
            pick = remaining[int(np.argmax(dist))]; chosen.append(pick); remaining.remove(pick); covered = np.vstack([covered, cand[pick]])
        return chosen
    return rule


FACTORIES = {
             "typed_decisive_probcover_q02": lambda refs: typed_decisive_probcover(refs, delta_q=0.02), "typed_decisive_probcover_q05": lambda refs: typed_decisive_probcover(refs, delta_q=0.05),
             "typed_decisive_probcover_q10": lambda refs: typed_decisive_probcover(refs, delta_q=0.10), "typed_decisive_probcover_unc_q05": lambda refs: typed_decisive_probcover(refs, delta_q=0.05, uncertainty_power=1.0),
             "typed_decisive_probcover_q05_typeonly": lambda refs: typed_decisive_probcover({}, delta_q=0.05), "typed_decisive_probcover_q05_shuffled": lambda refs: typed_decisive_probcover(refs, True, delta_q=0.05),
             "typed_decisive_probcover_q10_typeonly": lambda refs: typed_decisive_probcover({}, delta_q=0.10), "typed_decisive_probcover_q10_shuffled": lambda refs: typed_decisive_probcover(refs, True, delta_q=0.10),
             "typed_decisive_probcover_q02_typeonly": lambda refs: typed_decisive_probcover({}, delta_q=0.02), "typed_decisive_probcover_q02_shuffled": lambda refs: typed_decisive_probcover(refs, True, delta_q=0.02),
             "typed_decisive_uncertainty": lambda refs: typed_decisive_uncertainty(refs), "typed_decisive_bald": lambda refs: typed_decisive_bald(refs),
             "typed_decisive_coverage": lambda refs: typed_decisive_coverage(refs), "typed_decisive_sampling": lambda refs: typed_decisive_sampling(refs),
             "typed_decisive_sampling_g1": lambda refs: typed_decisive_sampling(refs, gamma=1.0), "typed_decisive_sampling_g4": lambda refs: typed_decisive_sampling(refs, gamma=4.0),
             "typed_decisive_sampling_shuffled": lambda refs: typed_decisive_sampling(refs, True),
             # controls with the decisive predictor reduced to the type one-hot (no reference/graph features): isolates what the graph adds to the decisive predictor
             "typed_decisive_coverage_typeonly": lambda refs: typed_decisive_coverage({}), "typed_decisive_sampling_typeonly": lambda refs: typed_decisive_sampling({}),
             "typed_decisive_coverage_g2": lambda refs: typed_decisive_coverage(refs, gamma=2.0), "typed_decisive_coverage_unc": lambda refs: typed_decisive_coverage(refs, uncertainty_power=1.0),
             "typed_decisive_coverage_unc_g2": lambda refs: typed_decisive_coverage(refs, gamma=2.0, uncertainty_power=1.0), "typed_decisive_coverage_unc_u2": lambda refs: typed_decisive_coverage(refs, uncertainty_power=2.0),
             "typed_decisive_coverage_unc_q": lambda refs: typed_decisive_coverage(refs, uncertainty_power=1.0, q_weight=1.0),
             "typed_decisive_coverage_unc_typeonly": lambda refs: typed_decisive_coverage({}, uncertainty_power=1.0), "typed_decisive_coverage_unc_shuffled": lambda refs: typed_decisive_coverage(refs, True, uncertainty_power=1.0),
             "typed_decisive_uncertainty_shuffled": lambda refs: typed_decisive_uncertainty(refs, True), "typed_decisive_bald_shuffled": lambda refs: typed_decisive_bald(refs, True),
             "typed_decisive_coverage_shuffled": lambda refs: typed_decisive_coverage(refs, True)}
PAIR_FACTORIES = {"coregcn": lambda refs: ju.pair_level(_gcn_pair_rule("core"), False), "uncertaingcn": lambda refs: ju.pair_level(_gcn_pair_rule("uncertain"), False)}
TYPED_NAMES = tuple(FACTORIES) + tuple(PAIR_FACTORIES)


def make_selector(name: str, exp):
    refs = reference_types(exp)
    if name in FACTORIES: return FACTORIES[name](refs)
    if name in PAIR_FACTORIES: return PAIR_FACTORIES[name](refs)
    raise ValueError(name)
