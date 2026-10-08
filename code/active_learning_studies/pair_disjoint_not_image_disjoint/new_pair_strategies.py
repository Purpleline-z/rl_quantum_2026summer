"""Additional acquisition strategies that operate on cached encoder features and a frozen-encoder reward head.

Every selector has the signature ``select(candidates, labeled, model, cache, budget, seed)`` and returns the chosen
candidate dicts.  ``candidates``/``labeled`` are dicts with ``pair_id, img1, img2, type_idx``; ``cache`` maps an image path
to its 512-d encoder feature; ``model`` is a BTModel whose ``reward_head`` is Linear-ReLU-Dropout-Linear.

Pair representations (all symmetric in the two images, as a preference pair is unordered):
  mean      (a+b)/2                     the representation used by the original core-set rule
  relation  [(a+b)/2, |a-b|, a*b]       adds how the two images relate to each other
Both are z-scored over candidates+labeled and concatenated with a scaled one-hot reconstruction type.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import KMeans


def _feature(cache, path) -> np.ndarray:
    return cache[path].numpy() if isinstance(cache[path], torch.Tensor) else np.asarray(cache[path])


def pair_features(items, cache, kind="mean", type_weight=1.0, scaler=None):
    a = np.stack([_feature(cache, x["img1"]) for x in items]); b = np.stack([_feature(cache, x["img2"]) for x in items])
    blocks = [(a + b) / 2]
    if kind == "relation": blocks += [np.abs(a - b), a * b]
    elif kind != "mean": raise ValueError(kind)
    z = np.concatenate(blocks, axis=1)
    if scaler is None: scaler = (z.mean(0), z.std(0) + 1e-6)
    z = (z - scaler[0]) / scaler[1]; z = z / np.sqrt(z.shape[1])  # unit-ish scale per row regardless of kind
    onehot = np.eye(5)[[int(x["type_idx"]) for x in items]] * type_weight
    return np.concatenate([z, onehot], axis=1).astype(np.float64), scaler


def _hidden_and_logit(model, items, cache):
    head = model.reward_head.eval()
    with torch.no_grad():
        a = torch.stack([cache[x["img1"]] for x in items]); b = torch.stack([cache[x["img2"]] for x in items])
        ha, hb = head[1](head[0](a)), head[1](head[0](b))
        t = torch.tensor([int(x["type_idx"]) for x in items]); idx = torch.arange(len(items))
        d = (head[3](ha) - head[3](hb))[idx, t]
    return (ha - hb).numpy().astype(np.float64), torch.sigmoid(d).numpy().astype(np.float64), t.numpy()


def _uncertainty(p):  # Bernoulli entropy
    p = np.clip(p, 1e-7, 1 - 1e-7); return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def _kmeanspp(vectors, budget, rng):
    """k-means++ seeding (BADGE): first point = largest norm, then D^2 sampling."""
    chosen = [int(np.argmax((vectors ** 2).sum(1)))]
    d2 = ((vectors - vectors[chosen[0]]) ** 2).sum(1)
    while len(chosen) < min(budget, len(vectors)):
        d2[chosen] = 0.0; total = d2.sum()
        nxt = int(rng.choice(len(vectors), p=d2 / total)) if total > 0 else int(rng.choice([i for i in range(len(vectors)) if i not in chosen]))
        chosen.append(nxt); d2 = np.minimum(d2, ((vectors - vectors[nxt]) ** 2).sum(1))
    return chosen


def core_set_relation(candidates, labeled, model, cache, budget, seed=0, kind="relation"):
    """Farthest-first k-center in a pair space that includes the relation between the two images."""
    allp, scaler = pair_features(candidates + labeled, cache, kind); cand, lab = allp[:len(candidates)], allp[len(candidates):]
    covered = lab.copy(); chosen = []; remaining = list(range(len(candidates)))
    for _ in range(min(budget, len(candidates))):
        dist = np.sqrt(((cand[remaining][:, None] - covered[None]) ** 2).sum(2)).min(1) if len(covered) else np.linalg.norm(cand[remaining] - cand.mean(0), axis=1)
        pick = remaining[int(np.argmax(dist))]; chosen.append(pick); remaining.remove(pick); covered = np.vstack([covered, cand[pick]])
    return [candidates[i] for i in chosen]


def typiclust_pairs(candidates, labeled, model, cache, budget, seed=0, kind="mean", knn=10):
    """TypiClust (Hacohen et al., 2022) on pairs: cluster, then take the densest pair of the largest uncovered cluster."""
    allp, _ = pair_features(candidates + labeled, cache, kind); cand, lab = allp[:len(candidates)], allp[len(candidates):]
    chosen = []; labeled_vecs = [v for v in lab]
    for step in range(min(budget, len(candidates))):
        k = min(len(lab) + step + 1, len(cand)); pool = np.vstack([cand] + ([lab] if len(lab) else []))
        labels = KMeans(n_clusters=k, n_init=3, random_state=seed + step).fit_predict(pool)
        cand_labels, lab_labels = labels[:len(cand)], labels[len(cand):]
        covered = set(lab_labels.tolist()) | {cand_labels[i] for i in chosen}
        sizes = {c: int((labels == c).sum()) for c in set(labels.tolist())}
        options = [c for c in sizes if c not in covered and any(cand_labels[i] == c and i not in chosen for i in range(len(cand)))] or \
                  [c for c in sizes if any(cand_labels[i] == c and i not in chosen for i in range(len(cand)))]
        cluster = max(options, key=lambda c: sizes[c]); members = [i for i in range(len(cand)) if cand_labels[i] == cluster and i not in chosen]
        member_vecs = cand[members]; kk = min(knn, len(member_vecs) - 1) if len(member_vecs) > 1 else 0
        if kk > 0:
            dist = np.sqrt(((member_vecs[:, None] - member_vecs[None]) ** 2).sum(2)); typical = 1 / (np.sort(dist, 1)[:, 1:kk + 1].mean(1) + 1e-9)
            chosen.append(members[int(np.argmax(typical))])
        else: chosen.append(members[0])
    return [candidates[i] for i in chosen]


def badge_pairs(candidates, labeled, model, cache, budget, seed=0):
    """BADGE (Ash et al., 2020) for a Bradley--Terry head: gradient embedding of the loss under the model's own preference."""
    h, p, t = _hidden_and_logit(model, candidates, cache); y = (p > .5).astype(float)
    grad = np.zeros((len(candidates), 5 * h.shape[1]))
    for i in range(len(candidates)): grad[i, t[i] * h.shape[1]:(t[i] + 1) * h.shape[1]] = (p[i] - y[i]) * h[i]
    return [candidates[i] for i in _kmeanspp(grad, budget, np.random.default_rng(seed))]


def fisher_dopt(candidates, labeled, model, cache, budget, seed=0, ridge=1.0):
    """Greedy D-optimal design: maximise the log-determinant of the Bradley--Terry Fisher information of the last layer.

    Pair i contributes w_i phi_i phi_i^T with phi_i = h(a)-h(b) in its type's head block and w_i = p_i(1-p_i); labelled pairs
    are already in the information matrix.  The gain of adding i is log(1 + w_i phi_i^T M^-1 phi_i) (matrix-determinant lemma).
    """
    h, p, t = _hidden_and_logit(model, candidates + labeled, cache); w = p * (1 - p); n = len(candidates); d = h.shape[1]
    inverse = {k: np.eye(d) / ridge for k in range(5)}
    for i in range(n, len(h)):  # labelled pairs
        k = t[i]; v = inverse[k] @ h[i]; inverse[k] = inverse[k] - np.outer(v, v) * w[i] / (1 + w[i] * h[i] @ v)
    chosen = []; remaining = list(range(n))
    for _ in range(min(budget, n)):
        gains = [np.log1p(w[i] * h[i] @ inverse[t[i]] @ h[i]) for i in remaining]; pick = remaining[int(np.argmax(gains))]
        k = t[pick]; v = inverse[k] @ h[pick]; inverse[k] = inverse[k] - np.outer(v, v) * w[pick] / (1 + w[pick] * h[pick] @ v)
        chosen.append(pick); remaining.remove(pick)
    return [candidates[i] for i in chosen]


def image_coverage_uncertainty(candidates, labeled, model, cache, budget, seed=0):
    """Graph view A: images are nodes, labelled pairs are edges.  Prefer uncertain pairs that touch images with no labelled edge."""
    _, p, _ = _hidden_and_logit(model, candidates, cache); unc = _uncertainty(p) / np.log(2)
    seen = {x["img1"] for x in labeled} | {x["img2"] for x in labeled}; chosen = []; remaining = list(range(len(candidates)))
    for _ in range(min(budget, len(candidates))):
        score = [unc[i] * (1 + (candidates[i]["img1"] not in seen) + (candidates[i]["img2"] not in seen)) for i in remaining]
        pick = remaining[int(np.argmax(score))]; chosen.append(pick); remaining.remove(pick)
        seen |= {candidates[pick]["img1"], candidates[pick]["img2"]}
    return [candidates[i] for i in chosen]


def graph_facility_location(candidates, labeled, model, cache, budget, seed=0, kind="relation", neighbours=10, alpha=1.0):
    """Graph view B: kNN graph over pairs; greedy uncertainty-weighted facility location (submodular coverage).

    Maximise sum_j u_j^alpha * max_{i in S+L} sim(i, j): pairs are rewarded for covering many *uncertain* neighbours that no
    selected or labelled pair already covers.  sim = exp(-d^2 / median d^2), restricted to the kNN graph.
    """
    allp, _ = pair_features(candidates + labeled, cache, kind); n = len(candidates); _, p, _ = _hidden_and_logit(model, candidates, cache)
    u = (_uncertainty(p) / np.log(2)) ** alpha + 1e-3
    d2 = ((allp[:, None] - allp[None]) ** 2).sum(2); sim = np.exp(-d2 / np.median(d2[d2 > 0]))
    for row in range(len(allp)):  # keep only the k nearest (graph edges)
        keep = np.argsort(-sim[row])[:neighbours + 1]; mask = np.zeros(len(allp), bool); mask[keep] = True; sim[row, ~mask] = 0
    sim = np.maximum(sim, sim.T)
    covered = np.zeros(n); covered = np.maximum(covered, sim[n:, :n].max(0)) if len(labeled) else covered
    chosen = []
    for _ in range(min(budget, n)):
        gains = [(u * np.maximum(sim[i, :n] - covered, 0)).sum() if i not in chosen else -1 for i in range(n)]
        pick = int(np.argmax(gains)); chosen.append(pick); covered = np.maximum(covered, sim[pick, :n])
    return [candidates[i] for i in chosen]


def dpp_pairs(candidates, labeled, model, cache, budget, seed=0, kind="relation"):
    """Greedy MAP of a quality-diversity DPP (Bıyık et al., 2019): L_ij = q_i q_j S_ij, q = Bernoulli entropy, S = cosine kernel."""
    allp, _ = pair_features(candidates + labeled, cache, kind); n = len(candidates); _, p, _ = _hidden_and_logit(model, candidates, cache)
    q = _uncertainty(p) / np.log(2) + 1e-3; unit = allp / np.linalg.norm(allp, axis=1, keepdims=True)
    S = (unit @ unit.T + 1) / 2; Q = np.concatenate([q, np.full(len(labeled), 1.0)]); L = Q[:, None] * S * Q[None, :]
    chosen = list(range(n, len(allp))); picked = []  # labelled pairs are conditioned on (always in the set)
    for _ in range(min(budget, n)):
        best, best_gain = None, -np.inf
        for i in range(n):
            if i in picked: continue
            idx = chosen + picked; sub = L[np.ix_(idx, idx)] + 1e-9 * np.eye(len(idx))
            cross = L[i, idx]; gain = L[i, i] - cross @ np.linalg.solve(sub, cross) if len(idx) else L[i, i]  # Schur complement = marginal gain of det
            if gain > best_gain: best, best_gain = i, gain
        picked.append(best)
    return [candidates[i] for i in picked]


def probcover_pairs(candidates, labeled, model, cache, budget, seed=0, kind="mean", purity=.9):
    """ProbCover (Yehuda et al., 2022) on pairs: delta-balls cover the pair space.  delta is the largest radius at which a ball is
    at least ``purity`` pure in the (observable) reconstruction type; each pick covers the most still-uncovered pairs."""
    allp, _ = pair_features(candidates + labeled, cache, kind); n = len(candidates); types = np.array([int(x["type_idx"]) for x in candidates + labeled])
    d = np.sqrt(((allp[:, None] - allp[None]) ** 2).sum(2)); delta = np.median(d[d > 0])
    for radius in np.quantile(d[d > 0], np.linspace(.02, .6, 30)):
        inside = (d <= radius); pure = np.mean([(types[inside[i]] == types[i]).mean() for i in range(len(allp))])
        if pure < purity: break
        delta = radius
    inside = d <= delta; covered = inside[n:].any(0) if len(labeled) else np.zeros(len(allp), bool); chosen = []
    for _ in range(min(budget, n)):
        gain = np.array([(inside[i] & ~covered).sum() if i not in chosen else -1 for i in range(n)]); pick = int(np.argmax(gain))
        chosen.append(pick); covered |= inside[pick]
    return [candidates[i] for i in chosen]


def maxherding_pairs(candidates, labeled, model, cache, budget, seed=0, kind="relation"):
    """MaxHerding (Bae et al., 2024): smooth kernel coverage = uncertainty-free facility location on the pair graph."""
    return graph_facility_location(candidates, labeled, model, cache, budget, seed, kind=kind, neighbours=len(candidates), alpha=0.0)


def laplace_bald(candidates, labeled, model, cache, budget, seed=0, ridge=1.0, samples=64):
    """BALD (Houlsby et al., 2011) for the Bradley--Terry head under a Laplace posterior on the last layer.

    Per type head the weights are N(w_MAP, M^-1) with M = ridge*I + sum w_i phi_i phi_i^T over labelled pairs.  For a candidate
    with latent d = w.phi ~ N(mu, s^2), s^2 = phi^T M^-1 phi, the mutual information between the (unseen) label and the weights is
    H[E sigma(d)] - E H[sigma(d)].  Batches are built greedily and M is updated with each selected pair (fantasy update).
    """
    h, p, t = _hidden_and_logit(model, candidates + labeled, cache); w = p * (1 - p); n = len(candidates); dim = h.shape[1]
    logit = np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7))); rng = np.random.default_rng(seed); z = rng.standard_normal(samples)
    inverse = {k: np.eye(dim) / ridge for k in range(5)}
    def update(i):
        k = t[i]; v = inverse[k] @ h[i]; inverse[k] = inverse[k] - np.outer(v, v) * w[i] / (1 + w[i] * h[i] @ v)
    for i in range(n, len(h)): update(i)
    def bald(i):
        s = np.sqrt(max(h[i] @ inverse[t[i]] @ h[i], 1e-12)); draws = 1 / (1 + np.exp(-(logit[i] + s * z)))
        ent = lambda x: -(x * np.log(np.clip(x, 1e-9, 1)) + (1 - x) * np.log(np.clip(1 - x, 1e-9, 1)))
        return ent(draws.mean()) - ent(draws).mean()
    chosen = []; remaining = list(range(n))
    for _ in range(min(budget, n)):
        pick = remaining[int(np.argmax([bald(i) for i in remaining]))]; chosen.append(pick); remaining.remove(pick); update(pick)
    return [candidates[i] for i in chosen]


NEW_STRATEGIES = {
    "core_set_relation": core_set_relation, "typiclust_pairs": typiclust_pairs, "badge_pairs": badge_pairs, "fisher_dopt": fisher_dopt,
    "image_coverage_uncertainty": image_coverage_uncertainty, "graph_facility_location": graph_facility_location,
    "dpp_pairs": dpp_pairs, "probcover_pairs": probcover_pairs, "maxherding_pairs": maxherding_pairs, "laplace_bald": laplace_bald,
}
