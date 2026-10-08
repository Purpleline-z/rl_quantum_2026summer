"""Additional acquisition strategies that operate on cached encoder features and a frozen-encoder reward head.

Every selector has the signature ``select(candidates, labeled, model, cache, budget, seed)`` and returns the chosen
candidate dicts.  ``candidates``/``labeled`` are dicts with ``pair_id, img1, img2``; ``cache`` maps an image path
to its 512-d encoder feature; ``model`` is a BTModel whose ``reward_head`` is Linear-ReLU-Dropout-Linear.

Label-free typing: a pair group usually carries judgments for several reconstruction types (103 of 168 groups have all four),
and which types are judged is part of the label.  The selectors here therefore do not use a candidate's ``type_idx``; they
aggregate information over the four active reward heads (``ACTIVE_HEADS``).  ``type_weight`` stays available for experiments.

Pair representations (all symmetric in the two images, as a preference pair is unordered):
  mean      (a+b)/2                     the representation used by the original core-set rule
  relation  [(a+b)/2, |a-b|, a*b]       adds how the two images relate to each other
Both are z-scored over candidates+labeled and concatenated with a scaled one-hot reconstruction type.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import KMeans

ACTIVE_HEADS = (0, 2, 3, 4)  # (1x1), c(6x2), (sqrt13 x sqrt13), HTR; index 1 is the excluded Twinned(2x1)


def _feature(cache, path) -> np.ndarray:
    return cache[path].numpy() if isinstance(cache[path], torch.Tensor) else np.asarray(cache[path])


def pair_features(items, cache, kind="mean", type_weight=0.0, scaler=None):
    a = np.stack([_feature(cache, x["img1"]) for x in items]); b = np.stack([_feature(cache, x["img2"]) for x in items])
    blocks = [(a + b) / 2]
    if kind == "relation": blocks += [np.abs(a - b), a * b]
    elif kind != "mean": raise ValueError(kind)
    z = np.concatenate(blocks, axis=1)
    if scaler is None: scaler = (z.mean(0), z.std(0) + 1e-6)
    z = (z - scaler[0]) / scaler[1]; z = z / np.sqrt(z.shape[1])  # unit-ish scale per row regardless of kind
    if type_weight:
        onehot = np.eye(5)[[int(x["type_idx"]) for x in items]] * type_weight; z = np.concatenate([z, onehot], axis=1)
    return z.astype(np.float64), scaler


def _hidden_and_logit(model, items, cache):
    """Hidden-layer difference phi = h(a)-h(b) [n, d] and the preference probability under every head [n, 5]."""
    head = model.reward_head.eval()
    with torch.no_grad():
        a = torch.stack([cache[x["img1"]] for x in items]); b = torch.stack([cache[x["img2"]] for x in items])
        ha, hb = head[1](head[0](a)), head[1](head[0](b)); d = head[3](ha) - head[3](hb)
    return (ha - hb).numpy().astype(np.float64), torch.sigmoid(d).numpy().astype(np.float64)


def _mean_uncertainty(p):
    return np.mean([_uncertainty(p[:, k]) for k in ACTIVE_HEADS], axis=0) / np.log(2)


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
    """BADGE (Ash et al., 2020) for a Bradley--Terry head: gradient embedding of the loss under the model's own preference,
    one block per active head (the gradient of -log sigma(y*(r_a-r_b)) w.r.t. that head's last layer is (p-y)*phi)."""
    h, p = _hidden_and_logit(model, candidates, cache); d = h.shape[1]; grad = np.zeros((len(candidates), 5 * d))
    for k in ACTIVE_HEADS: grad[:, k * d:(k + 1) * d] = (p[:, [k]] - (p[:, [k]] > .5)) * h
    return [candidates[i] for i in _kmeanspp(grad, budget, np.random.default_rng(seed))]


def _rank_one(inverse, vector, weight):
    v = inverse @ vector; return inverse - np.outer(v, v) * weight / (1 + weight * vector @ v)


def fisher_dopt(candidates, labeled, model, cache, budget, seed=0, ridge=1.0):
    """Greedy D-optimal design on the Bradley--Terry Fisher information of the last layer, summed over the active heads.

    A pair contributes w_k phi phi^T to head k's information with w_k = p_k(1-p_k); labelled pairs are already included.  The gain of
    adding a pair is sum_k log(1 + w_k phi^T M_k^-1 phi) (matrix-determinant lemma).
    """
    h, p = _hidden_and_logit(model, candidates + labeled, cache); w = p * (1 - p); n = len(candidates); d = h.shape[1]
    inverse = {k: np.eye(d) / ridge for k in ACTIVE_HEADS}
    for i in range(n, len(h)):
        for k in ACTIVE_HEADS: inverse[k] = _rank_one(inverse[k], h[i], w[i, k])
    chosen = []; remaining = list(range(n))
    for _ in range(min(budget, n)):
        gains = [sum(np.log1p(w[i, k] * h[i] @ inverse[k] @ h[i]) for k in ACTIVE_HEADS) for i in remaining]; pick = remaining[int(np.argmax(gains))]
        for k in ACTIVE_HEADS: inverse[k] = _rank_one(inverse[k], h[pick], w[pick, k])
        chosen.append(pick); remaining.remove(pick)
    return [candidates[i] for i in chosen]


def image_coverage_uncertainty(candidates, labeled, model, cache, budget, seed=0):
    """Graph view A: images are nodes, labelled pairs are edges.  Prefer uncertain pairs that touch images with no labelled edge."""
    _, p = _hidden_and_logit(model, candidates, cache); unc = _mean_uncertainty(p)
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
    allp, _ = pair_features(candidates + labeled, cache, kind); n = len(candidates); _, p = _hidden_and_logit(model, candidates, cache)
    u = _mean_uncertainty(p) ** alpha + 1e-3
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
    allp, _ = pair_features(candidates + labeled, cache, kind); n = len(candidates); _, p = _hidden_and_logit(model, candidates, cache)
    q = _mean_uncertainty(p) + 1e-3; unit = allp / np.linalg.norm(allp, axis=1, keepdims=True)
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


def probcover_pairs(candidates, labeled, model, cache, budget, seed=0, kind="mean", target_coverage=.9):
    """ProbCover (Yehuda et al., 2022) on pairs: delta-balls cover the pair space and each pick covers the most still-uncovered pairs.

    The original picks delta from class purity; pair groups have no single observable class, so delta is the smallest radius at which
    ``budget`` greedy balls cover ``target_coverage`` of the pool (label-free).
    """
    allp, _ = pair_features(candidates + labeled, cache, kind); n = len(candidates)
    d = np.sqrt(((allp[:, None] - allp[None]) ** 2).sum(2)); base = np.zeros(len(allp), bool)
    if len(labeled): base = (d[n:] <= 0).any(0)
    def greedy(radius, steps):
        inside = d <= radius; covered = inside[n:].any(0) if len(labeled) else np.zeros(len(allp), bool); picks = []
        for _ in range(min(steps, n)):
            gain = np.array([(inside[i] & ~covered).sum() if i not in picks else -1 for i in range(n)]); pick = int(np.argmax(gain)); picks.append(pick); covered = covered | inside[pick]
        return picks, covered[:n].mean()
    radii = np.quantile(d[d > 0], np.linspace(.01, .9, 40)); delta = radii[-1]
    for radius in radii:
        if greedy(radius, budget)[1] >= target_coverage: delta = radius; break
    return [candidates[i] for i in greedy(delta, budget)[0]]


def maxherding_pairs(candidates, labeled, model, cache, budget, seed=0, kind="relation"):
    """MaxHerding (Bae et al., 2024): smooth kernel coverage = uncertainty-free facility location on the pair graph."""
    return graph_facility_location(candidates, labeled, model, cache, budget, seed, kind=kind, neighbours=len(candidates), alpha=0.0)


def laplace_bald(candidates, labeled, model, cache, budget, seed=0, ridge=1.0, samples=64):
    """BALD (Houlsby et al., 2011) for the Bradley--Terry head under a Laplace posterior on the last layer, summed over active heads.

    Per head the weights are N(w_MAP, M^-1) with M = ridge*I + sum w_i phi_i phi_i^T over labelled pairs.  For a candidate with latent
    d = w.phi ~ N(mu, s^2), s^2 = phi^T M^-1 phi, the mutual information between the (unseen) label and the weights is
    H[E sigma(d)] - E H[sigma(d)].  Batches are built greedily and M is updated with each selected pair (fantasy update).
    """
    h, p = _hidden_and_logit(model, candidates + labeled, cache); w = p * (1 - p); n = len(candidates); dim = h.shape[1]
    pc = np.clip(p, 1e-7, 1 - 1e-7); logit = np.log(pc / (1 - pc)); z = np.random.default_rng(seed).standard_normal(samples)
    inverse = {k: np.eye(dim) / ridge for k in ACTIVE_HEADS}
    for i in range(n, len(h)):
        for k in ACTIVE_HEADS: inverse[k] = _rank_one(inverse[k], h[i], w[i, k])
    ent = lambda x: -(x * np.log(np.clip(x, 1e-9, 1)) + (1 - x) * np.log(np.clip(1 - x, 1e-9, 1)))
    def bald(i):
        total = 0.0
        for k in ACTIVE_HEADS:
            s = np.sqrt(max(h[i] @ inverse[k] @ h[i], 1e-12)); draws = 1 / (1 + np.exp(-(logit[i, k] + s * z))); total += ent(draws.mean()) - ent(draws).mean()
        return total
    chosen = []; remaining = list(range(n))
    for _ in range(min(budget, n)):
        pick = remaining[int(np.argmax([bald(i) for i in remaining]))]; chosen.append(pick); remaining.remove(pick)
        for k in ACTIVE_HEADS: inverse[k] = _rank_one(inverse[k], h[pick], w[pick, k])
    return [candidates[i] for i in chosen]


def dropquery_pairs(candidates, labeled, model, cache, budget, seed=0, kind="mean", centroid_fraction=.5):
    """DropQuery-style (Mittal et al., 2024): representative pairs first, then uncertainty.

    The first ``centroid_fraction`` of the budget takes the pair nearest each k-means centre (cold-start coverage); the rest takes the most
    uncertain remaining pairs (entropy averaged over the active heads).
    """
    allp, _ = pair_features(candidates + labeled, cache, kind); n = len(candidates); cand = allp[:n]; _, p = _hidden_and_logit(model, candidates, cache)
    unc = _mean_uncertainty(p); k = max(1, int(round(budget * centroid_fraction))); chosen = []
    centres = KMeans(n_clusters=min(k, n), n_init=5, random_state=seed).fit(cand).cluster_centers_
    for centre in centres:
        order = np.argsort(((cand - centre) ** 2).sum(1))
        for i in order:
            if int(i) not in chosen: chosen.append(int(i)); break
    for i in np.argsort(-unc):
        if len(chosen) >= min(budget, n): break
        if int(i) not in chosen: chosen.append(int(i))
    return [candidates[i] for i in chosen[:budget]]


def uncertainty_all_heads(candidates, labeled, model, cache, budget, seed=0):
    """Plain uncertainty sampling, but averaging the Bernoulli entropy over the four active heads instead of the first label row's head."""
    _, p = _hidden_and_logit(model, candidates, cache); order = np.argsort(-_mean_uncertainty(p), kind="stable")[:budget]
    return [candidates[i] for i in order]


def delta_gap(candidates, labeled, model, cache, budget, seed=0):
    """Largest predicted quality gap (ActiveUltraFeedback, 2026: large-gap pairs give lower-noise preference labels than near-ties).

    Score = |r_a - r_b| averaged over the active heads.  Near-equal pairs are the ones annotators call ties or 'not applicable'
    (14% and 35% of our rows), so the opposite of uncertainty sampling is a legitimate hypothesis here.
    """
    _, p = _hidden_and_logit(model, candidates, cache); pc = np.clip(p, 1e-6, 1 - 1e-6); gap = np.abs(np.log(pc / (1 - pc)))[:, list(ACTIVE_HEADS)].mean(1)
    return [candidates[i] for i in np.argsort(-gap, kind="stable")[:budget]]


def delta_ucb(candidates, labeled, model, cache, budget, seed=0, ridge=1.0, beta=1.0):
    """Large gap that the model is still unsure about: |mu| + beta*s per head, with s the Laplace standard deviation of the last-layer logit
    (batch-aware: the information matrix is updated after each pick).  Interpolates between delta_gap (beta=0) and BALD-like exploration."""
    h, p = _hidden_and_logit(model, candidates + labeled, cache); w = p * (1 - p); n = len(candidates); d = h.shape[1]
    pc = np.clip(p, 1e-6, 1 - 1e-6); mu = np.abs(np.log(pc / (1 - pc))); inverse = {k: np.eye(d) / ridge for k in ACTIVE_HEADS}
    for i in range(n, len(h)):
        for k in ACTIVE_HEADS: inverse[k] = _rank_one(inverse[k], h[i], w[i, k])
    chosen = []; remaining = list(range(n))
    for _ in range(min(budget, n)):
        score = [np.mean([mu[i, k] + beta * np.sqrt(max(h[i] @ inverse[k] @ h[i], 0)) for k in ACTIVE_HEADS]) for i in remaining]; pick = remaining[int(np.argmax(score))]
        for k in ACTIVE_HEADS: inverse[k] = _rank_one(inverse[k], h[pick], w[pick, k])
        chosen.append(pick); remaining.remove(pick)
    return [candidates[i] for i in chosen]


def bald_decisive(candidates, labeled, model, cache, budget, seed=0, ridge=1.0, samples=64):
    """BALD x P(decisive): information about the preference, discounted by the chance the annotator gives a decisive answer.

    49% of judgments are tie/not_apply, which carry little preference information.  P(decisive) per (pair, head) comes from a strongly
    regularised logistic regression fitted on the *labelled* judgments only (``labeled[i]["outcomes"]`` = list of (type_idx, decisive)),
    using the head's one-hot, the feature distance and the cosine similarity of the two images.  Falls back to 0.5 without labelled outcomes.
    """
    from sklearn.linear_model import LogisticRegression
    h, p = _hidden_and_logit(model, candidates + labeled, cache); w = p * (1 - p); n = len(candidates); dim = h.shape[1]
    pc = np.clip(p, 1e-7, 1 - 1e-7); logit = np.log(pc / (1 - pc)); z = np.random.default_rng(seed).standard_normal(samples)
    def pair_stats(items):
        a = np.stack([_feature(cache, x["img1"]) for x in items]); b = np.stack([_feature(cache, x["img2"]) for x in items])
        return np.column_stack([np.linalg.norm(a - b, axis=1), (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))])
    rows_x, rows_y = [], []; stats_lab = pair_stats(labeled) if labeled else np.zeros((0, 2))
    for item, stat in zip(labeled, stats_lab):
        for type_idx, decisive in item.get("outcomes", []): rows_x.append(np.concatenate([np.eye(5)[type_idx], stat])); rows_y.append(int(decisive))
    stats_cand = pair_stats(candidates); decisive = np.full((n, 5), .5)
    if len(set(rows_y)) == 2:
        X = np.array(rows_x); mu, sd = X[:, 5:].mean(0), X[:, 5:].std(0) + 1e-6; X[:, 5:] = (X[:, 5:] - mu) / sd
        classifier = LogisticRegression(C=0.3, max_iter=1000).fit(X, rows_y)
        for k in ACTIVE_HEADS:
            Xc = np.column_stack([np.tile(np.eye(5)[k], (n, 1)), (stats_cand - mu) / sd]); decisive[:, k] = classifier.predict_proba(Xc)[:, 1]
    inverse = {k: np.eye(dim) / ridge for k in ACTIVE_HEADS}
    for i in range(n, len(h)):
        for k in ACTIVE_HEADS: inverse[k] = _rank_one(inverse[k], h[i], w[i, k])
    ent = lambda x: -(x * np.log(np.clip(x, 1e-9, 1)) + (1 - x) * np.log(np.clip(1 - x, 1e-9, 1)))
    def score(i):
        total = 0.0
        for k in ACTIVE_HEADS:
            s = np.sqrt(max(h[i] @ inverse[k] @ h[i], 1e-12)); draws = 1 / (1 + np.exp(-(logit[i, k] + s * z))); total += decisive[i, k] * (ent(draws.mean()) - ent(draws).mean())
        return total
    chosen = []; remaining = list(range(n))
    for _ in range(min(budget, n)):
        pick = remaining[int(np.argmax([score(i) for i in remaining]))]; chosen.append(pick); remaining.remove(pick)
        for k in ACTIVE_HEADS: inverse[k] = _rank_one(inverse[k], h[pick], w[pick, k])
    return [candidates[i] for i in chosen]


def _similarity(allp):
    d2 = ((allp[:, None] - allp[None]) ** 2).sum(2); return np.exp(-d2 / np.median(d2[d2 > 0]))


def fass_pairs(candidates, labeled, model, cache, budget, seed=0, kind="relation", filter_factor=3):
    """FASS (Wei et al., ICML 2015): keep the ``filter_factor * budget`` most uncertain candidates, then pick a submodular
    facility-location cover of that filtered set (greedy, 1-1/e guarantee).  Informativeness first, representativeness second."""
    allp, _ = pair_features(candidates + labeled, cache, kind); n = len(candidates); _, p = _hidden_and_logit(model, candidates, cache)
    keep = np.argsort(-_mean_uncertainty(p), kind="stable")[:min(n, filter_factor * budget)]; sim = _similarity(allp)
    covered = sim[n:, :][:, keep].max(0) if len(labeled) else np.zeros(len(keep)); chosen = []
    for _ in range(min(budget, len(keep))):
        gains = [np.maximum(sim[keep[j]][keep] - covered, 0).sum() if j not in chosen else -1 for j in range(len(keep))]; pick = int(np.argmax(gains))
        chosen.append(pick); covered = np.maximum(covered, sim[keep[pick]][keep])
    return [candidates[keep[j]] for j in chosen]


def graphcut_pairs(candidates, labeled, model, cache, budget, seed=0, kind="relation", trade_off=0.5):
    """Graph cut (Iyer et al.): maximise sum_{i in V} sum_{j in S} s_ij - trade_off * sum_{i,j in S} s_ij, greedily; representative yet non-redundant."""
    allp, _ = pair_features(candidates + labeled, cache, kind); n = len(candidates); sim = _similarity(allp); chosen = []; selected = list(range(n, len(allp)))
    for _ in range(min(budget, n)):
        gains = [sim[i, :n].sum() - trade_off * (2 * sim[i, selected].sum() + sim[i, i]) if i not in chosen else -np.inf for i in range(n)]
        pick = int(np.argmax(gains)); chosen.append(pick); selected.append(pick)
    return [candidates[i] for i in chosen]


NEW_STRATEGIES = {
    "core_set_relation": core_set_relation, "typiclust_pairs": typiclust_pairs, "badge_pairs": badge_pairs, "fisher_dopt": fisher_dopt,
    "image_coverage_uncertainty": image_coverage_uncertainty, "graph_facility_location": graph_facility_location,
    "uncertainty_all_heads": uncertainty_all_heads, "bald_decisive": bald_decisive, "delta_gap": delta_gap, "delta_ucb": delta_ucb, "fass_pairs": fass_pairs, "graphcut_pairs": graphcut_pairs, "dpp_pairs": dpp_pairs, "dropquery_pairs": dropquery_pairs, "probcover_pairs": probcover_pairs, "maxherding_pairs": maxherding_pairs, "laplace_bald": laplace_bald,
}
