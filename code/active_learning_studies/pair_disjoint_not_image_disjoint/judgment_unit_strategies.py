"""Acquisition strategies for the *judgment-unit* study: a candidate is ONE (pair, reconstruction type) judgment.

Every selector has the signature ``select(cands, labeled, model, cache, budget, seed)`` and returns ``budget`` distinct candidate dicts.
``cands``: one dict per still-unrevealed judgment with ``pair_id`` (unique judgment id), ``base_pair`` (the image pair), ``img1/img2`` (the
order shown for that judgment), ``type_idx`` (its reconstruction type, observable), ``cluster1`` (k-means cluster of the pair).
``labeled``: the judgments revealed so far (same keys plus ``outcomes`` = [(type_idx, decisive)]).  The outcome of a candidate is never visible.

Three families (see JUDGMENT_UNIT_RESULTS.md for the table):
  R  row level, own head   the score of a judgment is computed with the head of its own type; the rule may pick any type of any pair.
  P  pair level            the rule works on the unique image pairs of the candidates (coverage / diversity structure lives in pair space) and
                           then reveals one judgment of each chosen pair: the type is chosen at random (rules without a quality score) or as the
                           most uncertain remaining type (rules whose quality score is uncertainty).
  L  all-head pair level   the original all-head rules (no type information); one random type of each chosen pair is revealed.
Candidates are shuffled with the seed before every call, so ties (equal scores, duplicate pair vectors) are broken at random, not by file order.
"""
from __future__ import annotations

import random

import numpy as np
import torch

import new_pair_strategies as nps
import frozen_encoder_reward_head as frozen
from new_pair_strategies import ACTIVE_HEADS, _hidden_and_logit, _kmeanspp, _rank_one

LN2 = np.log(2)
_ent = lambda x: -(x * np.log(np.clip(x, 1e-9, 1)) + (1 - x) * np.log(np.clip(1 - x, 1e-9, 1)))


# ---------------------------------------------------------------------------------------------------- helpers
def _own(model, items, cache):
    """Hidden-difference phi [n,d], own-head preference probability [n] and head index [n] of every judgment (own type, own image order)."""
    h, p = _hidden_and_logit(model, items, cache); k = np.array([int(x["type_idx"]) for x in items], dtype=int)
    return h, p[np.arange(len(items)), k], k


def _own_uncertainty(model, items, cache):
    """Bernoulli entropy of the judgment's own head, scaled to [0, 1]."""
    _, p, _ = _own(model, items, cache); return nps._uncertainty(p) / LN2


def _sq(a, b=None):
    b = a if b is None else b
    return np.maximum((a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None] - 2 * a @ b.T, 0.0)


def _inverses(h, w, k, n, ridge):
    inverse = {kk: np.eye(h.shape[1]) / ridge for kk in ACTIVE_HEADS}
    for i in range(n, len(h)): inverse[k[i]] = _rank_one(inverse[k[i]], h[i], w[i])  # a revealed judgment informs ITS head only
    return inverse


def _leverage(h, k, inverse, rows=None):
    rows = range(len(h)) if rows is None else rows
    return np.array([max(h[i] @ inverse[k[i]] @ h[i], 0.0) for i in rows])


# ---------------------------------------------------------------------------------------------------- R: row level, own head
def delta_gap(cands, labeled, model, cache, budget, seed=0):
    _, p, _ = _own(model, cands, cache); pc = np.clip(p, 1e-6, 1 - 1e-6); gap = np.abs(np.log(pc / (1 - pc)))
    return [cands[i] for i in np.argsort(-gap, kind="stable")[:budget]]


def delta_ucb(cands, labeled, model, cache, budget, seed=0, ridge=1.0, beta=1.0):
    items = cands + labeled; h, p, k = _own(model, items, cache); n = len(cands); w = p * (1 - p); pc = np.clip(p, 1e-6, 1 - 1e-6); mu = np.abs(np.log(pc / (1 - pc)))
    inverse = _inverses(h, w, k, n, ridge); chosen = []; remaining = list(range(n))
    for _ in range(min(budget, n)):
        score = [mu[i] + beta * np.sqrt(max(h[i] @ inverse[k[i]] @ h[i], 0)) for i in remaining]; pick = remaining[int(np.argmax(score))]
        inverse[k[pick]] = _rank_one(inverse[k[pick]], h[pick], w[pick]); chosen.append(pick); remaining.remove(pick)
    return [cands[i] for i in chosen]


def fisher_dopt(cands, labeled, model, cache, budget, seed=0, ridge=1.0):
    items = cands + labeled; h, p, k = _own(model, items, cache); n = len(cands); w = p * (1 - p); inverse = _inverses(h, w, k, n, ridge)
    chosen = []; remaining = list(range(n))
    for _ in range(min(budget, n)):
        gains = [np.log1p(w[i] * h[i] @ inverse[k[i]] @ h[i]) for i in remaining]; pick = remaining[int(np.argmax(gains))]
        inverse[k[pick]] = _rank_one(inverse[k[pick]], h[pick], w[pick]); chosen.append(pick); remaining.remove(pick)
    return [cands[i] for i in chosen]


def _bald_greedy(cands, labeled, model, cache, budget, seed, ridge, samples, weight=None):
    items = cands + labeled; h, p, k = _own(model, items, cache); n = len(cands); w = p * (1 - p)
    pc = np.clip(p, 1e-7, 1 - 1e-7); logit = np.log(pc / (1 - pc)); z = np.random.default_rng(seed).standard_normal(samples)
    inverse = _inverses(h, w, k, n, ridge); weight = np.ones(n) if weight is None else weight
    def score(i):
        s = np.sqrt(max(h[i] @ inverse[k[i]] @ h[i], 1e-12)); draws = 1 / (1 + np.exp(-(logit[i] + s * z))); return weight[i] * (_ent(draws.mean()) - _ent(draws).mean())
    scores = np.array([score(i) for i in range(n)]); alive = np.ones(n, bool); chosen = []
    for _ in range(min(budget, n)):
        pick = int(np.argmax(np.where(alive, scores, -np.inf))); chosen.append(pick); alive[pick] = False
        inverse[k[pick]] = _rank_one(inverse[k[pick]], h[pick], w[pick])
        for i in np.flatnonzero(alive & (k[:n] == k[pick])): scores[i] = score(i)  # only judgments of the same head change
    return [cands[i] for i in chosen]


def laplace_bald(cands, labeled, model, cache, budget, seed=0, ridge=1.0, samples=64):
    """BALD for the own head under the last-layer Laplace posterior; a revealed judgment shrinks the posterior of its own head only."""
    return _bald_greedy(cands, labeled, model, cache, budget, seed, ridge, samples)


def bald_decisive(cands, labeled, model, cache, budget, seed=0, ridge=1.0, samples=64):
    decisive = nps._decisive_probability(cands, labeled, cache); weight = decisive[np.arange(len(cands)), [int(x["type_idx"]) for x in cands]]
    return _bald_greedy(cands, labeled, model, cache, budget, seed, ridge, samples, weight)


def badge_pairs(cands, labeled, model, cache, budget, seed=0):
    """BADGE: gradient embedding of the own head's loss under its own predicted preference, k-means++ seeding."""
    h, p, k = _own(model, cands, cache); d = h.shape[1]; grad = np.zeros((len(cands), 5 * d))
    for i in range(len(cands)): grad[i, k[i] * d:(k[i] + 1) * d] = (p[i] - float(p[i] > .5)) * h[i]
    return [cands[i] for i in _kmeanspp(grad, budget, np.random.default_rng(seed))]


def image_coverage_uncertainty(cands, labeled, model, cache, budget, seed=0):
    unc = _own_uncertainty(model, cands, cache); seen = {x["img1"] for x in labeled} | {x["img2"] for x in labeled}; chosen = []; remaining = list(range(len(cands)))
    for _ in range(min(budget, len(cands))):
        score = [unc[i] * (1 + (cands[i]["img1"] not in seen) + (cands[i]["img2"] not in seen)) for i in remaining]
        pick = remaining[int(np.argmax(score))]; chosen.append(pick); remaining.remove(pick); seen |= {cands[pick]["img1"], cands[pick]["img2"]}
    return [cands[i] for i in chosen]


def ensemble_bald(cands, labeled, models, cache, budget, seed=0, decisive=False):
    ps = np.stack([_own(m, cands, cache)[1] for m in models]); mi = _ent(ps.mean(0)) - _ent(ps).mean(0)
    weight = nps._decisive_probability(cands, labeled, cache)[np.arange(len(cands)), [int(x["type_idx"]) for x in cands]] if decisive else 1.0
    return [cands[i] for i in np.argsort(-(weight * mi), kind="stable")[:budget]]


ENSEMBLE = {"ensemble_bald": lambda *a, **k: ensemble_bald(*a, **k, decisive=False), "ensemble_bald_decisive": lambda *a, **k: ensemble_bald(*a, **k, decisive=True)}


def cluster_quota_uncertainty(cands, labeled, model, cache, budget, seed=0):
    """The original rule (quota of the most uncertain per k-means cluster, then fill) on judgments; uncertainty = own head.  The original de-duplicated image
    pairs; here the unit is the judgment, so only an identical judgment is excluded (it cannot occur: ids are unique)."""
    from collections import defaultdict
    unc = _own_uncertainty(model, cands, cache); grouped = defaultdict(list)
    for item, u in zip(cands, unc): grouped[int(item["cluster1"])].append((u, item))
    total = sum(map(len, grouped.values())); selected, used = [], set()
    for _, members in sorted(grouped.items()):
        quota = max(1, round(budget * len(members) / max(1, total)))
        for u, item in sorted(members, key=lambda t: -t[0])[:quota]:
            if item["pair_id"] not in used and len(selected) < budget: selected.append(item); used.add(item["pair_id"])
    for u, item in sorted((t for m in grouped.values() for t in m), key=lambda t: -t[0]):
        if len(selected) >= budget: break
        if item["pair_id"] not in used: selected.append(item); used.add(item["pair_id"])
    return selected


# ---------------------------------------------------------------------------------------------------- P: pair level with an uncertainty quality
def graph_fl(pairs, labeled, u, cache, budget, kind="relation", neighbours=10, alpha=1.0):
    allp, _ = nps.pair_features(pairs + labeled, cache, kind); n = len(pairs); u = u ** alpha + 1e-3
    d2 = _sq(allp); sim = np.exp(-d2 / np.median(d2[d2 > 0]))
    for row in range(len(allp)):
        keep = np.argsort(-sim[row])[:neighbours + 1]; mask = np.zeros(len(allp), bool); mask[keep] = True; sim[row, ~mask] = 0
    sim = np.maximum(sim, sim.T); covered = np.zeros(n); covered = np.maximum(covered, sim[n:, :n].max(0)) if len(labeled) else covered; chosen = []
    for _ in range(min(budget, n)):
        gains = [(u * np.maximum(sim[i, :n] - covered, 0)).sum() if i not in chosen else -1 for i in range(n)]
        pick = int(np.argmax(gains)); chosen.append(pick); covered = np.maximum(covered, sim[pick, :n])
    return chosen


def dpp(pairs, labeled, u, cache, budget, kind="relation"):
    allp, _ = nps.pair_features(pairs + labeled, cache, kind); n = len(pairs); q = u + 1e-3; unit = allp / np.linalg.norm(allp, axis=1, keepdims=True)
    S = (unit @ unit.T + 1) / 2; Q = np.concatenate([q, np.full(len(labeled), 1.0)]); L = Q[:, None] * S * Q[None, :]
    base = list(range(n, len(allp))); picked = []
    for _ in range(min(budget, n)):
        best, best_gain = None, -np.inf
        for i in range(n):
            if i in picked: continue
            idx = base + picked; sub = L[np.ix_(idx, idx)] + 1e-9 * np.eye(len(idx)); cross = L[i, idx]
            gain = L[i, i] - cross @ np.linalg.solve(sub, cross) if len(idx) else L[i, i]
            if gain > best_gain: best, best_gain = i, gain
        picked.append(best)
    return picked


def fass(pairs, labeled, u, cache, budget, kind="relation", filter_factor=3):
    allp, _ = nps.pair_features(pairs + labeled, cache, kind); n = len(pairs); keep = np.argsort(-u, kind="stable")[:min(n, filter_factor * budget)]
    d2 = _sq(allp); sim = np.exp(-d2 / np.median(d2[d2 > 0])); covered = sim[n:, :][:, keep].max(0) if len(labeled) else np.zeros(len(keep)); chosen = []
    for _ in range(min(budget, len(keep))):
        gains = [np.maximum(sim[keep[j]][keep] - covered, 0).sum() if j not in chosen else -1 for j in range(len(keep))]; pick = int(np.argmax(gains))
        chosen.append(pick); covered = np.maximum(covered, sim[keep[pick]][keep])
    return [int(keep[j]) for j in chosen]


def _from_pairs(function):
    """Adapt a rule from new_pair_strategies (returns pair dicts) to the (pairs, labeled, u, cache, need, seed) -> indices interface."""
    def run(pairs, labeled, u, cache, need, seed, model):
        picked = function(pairs, labeled, model, cache, need, seed); ids = {x["pair_id"]: i for i, x in enumerate(pairs)}; return [ids[x["pair_id"]] for x in picked]
    return run


# ---------------------------------------------------------------------------------------------------- pair-level wrapper
def pair_level(rule, quality: bool, exp=None):
    """Run ``rule`` on the unique pairs of the candidates and reveal one judgment of every chosen pair.

    ``quality``: the rule receives u = the largest own-head uncertainty among the pair's remaining judgments, and the revealed judgment is that most
    uncertain one; otherwise the revealed judgment is a uniformly random remaining judgment of the pair.  If the budget exceeds the number of pairs the
    rule is run again on what is left (each pair can be asked about more than once)."""
    def select(cands, labeled, model, cache, budget, seed=0):
        rng = random.Random(seed * 1_000_003 + 11); unc = _own_uncertainty(model, cands, cache) if quality else None
        remaining = list(range(len(cands))); done = []; known = list(labeled)
        while len(done) < min(budget, len(cands)):
            by_pair: dict = {}
            for i in remaining: by_pair.setdefault(cands[i]["base_pair"], []).append(i)
            pairs = [{k: cands[rows[0]][k] for k in ("img1", "img2", "cluster1")} | {"pair_id": b} for b, rows in by_pair.items()]
            lab = list({x["base_pair"]: {"pair_id": x["base_pair"], "img1": x["img1"], "img2": x["img2"]} for x in known}.values())
            best = {b: max(rows, key=lambda i: unc[i]) for b, rows in by_pair.items()} if quality else None
            u = np.array([unc[best[p["pair_id"]]] for p in pairs]) if quality else None
            need = min(budget - len(done), len(pairs)); chosen = rule(pairs, lab, u, cache, need, seed + len(done), model)
            for c in chosen:
                b = pairs[c]["pair_id"]; row = best[b] if quality else rng.choice(by_pair[b]); done.append(row); remaining.remove(row); known.append(cands[row])
        return [cands[i] for i in done]
    return select


def _quality_rule(function):
    return lambda pairs, labeled, u, cache, need, seed, model: function(pairs, labeled, u, cache, need)


def _fixed_rule(name):
    return _from_pairs(getattr(nps, name))


def dropquery(cands, labeled, model, cache, budget, seed=0, centroid_fraction=.5):
    """Half of the budget: the pair nearest each k-means centre of the candidate pairs (a random remaining type of it); the rest: the most uncertain own-head judgments."""
    from sklearn.cluster import KMeans
    rng = random.Random(seed * 1_000_003 + 13); by_pair: dict = {}
    for i, c in enumerate(cands): by_pair.setdefault(c["base_pair"], []).append(i)
    pairs = [{"img1": cands[r[0]]["img1"], "img2": cands[r[0]]["img2"]} for r in by_pair.values()]; names = list(by_pair)
    lab = list({x["base_pair"]: {"img1": x["img1"], "img2": x["img2"]} for x in labeled}.values())
    allp, _ = nps.pair_features(pairs + lab, cache, "mean"); cand = allp[:len(pairs)]; k = max(1, int(round(budget * centroid_fraction))); chosen = []
    for centre in KMeans(n_clusters=min(k, len(pairs)), n_init=5, random_state=seed).fit(cand).cluster_centers_:
        for j in np.argsort(((cand - centre) ** 2).sum(1)):
            row = rng.choice(by_pair[names[int(j)]])
            if names[int(j)] not in {cands[c]["base_pair"] for c in chosen}: chosen.append(row); break
    unc = _own_uncertainty(model, cands, cache)
    for i in np.argsort(-unc, kind="stable"):
        if len(chosen) >= min(budget, len(cands)): break
        if int(i) not in chosen: chosen.append(int(i))
    return [cands[i] for i in chosen[:budget]]


# ---------------------------------------------------------------------------------------------------- registry
PAIR_RULES = {  # name -> (rule, uses uncertainty quality)
    "core_set_relation": (_fixed_rule("core_set_relation"), False), "typiclust_pairs": (_fixed_rule("typiclust_pairs"), False),
    "probcover_pairs": (_fixed_rule("probcover_pairs"), False), "maxherding_pairs": (_fixed_rule("maxherding_pairs"), False),
    "graphcut_pairs": (_fixed_rule("graphcut_pairs"), False), "uncertainty_all_heads": (_fixed_rule("uncertainty_all_heads"), False),
    "graph_facility_location": (_quality_rule(lambda p, l, u, c, n: graph_fl(p, l, u, c, n)), True),
    "dpp_pairs": (_quality_rule(lambda p, l, u, c, n: dpp(p, l, u, c, n)), True),
    "fass_pairs": (_quality_rule(lambda p, l, u, c, n: fass(p, l, u, c, n)), True)}
ROW_RULES = {"delta_gap": delta_gap, "delta_ucb": delta_ucb, "fisher_dopt": fisher_dopt, "laplace_bald": laplace_bald, "bald_decisive": bald_decisive,
             "badge_pairs": badge_pairs, "image_coverage_uncertainty": image_coverage_uncertainty, "dropquery_pairs": dropquery,
             "cluster_quota_uncertainty": cluster_quota_uncertainty}
ORIGINAL_ROW = ("uncertainty", "uncertainty_diversity", "cluster_margin_pairwise", "mc_dropout_probability_variance", "mc_dropout_mutual_information")  # run through Experiment.select
ALL_HEAD = ("uncertainty_lf", "cluster_quota_uncertainty_lf", "uncertainty_diversity_lf", "cluster_margin_pairwise_lf", "mc_dropout_probability_variance_lf", "mc_dropout_mutual_information_lf")
NEW_NAMES = ("core_set_relation", "typiclust_pairs", "badge_pairs", "fisher_dopt", "image_coverage_uncertainty", "graph_facility_location", "uncertainty_all_heads", "bald_decisive", "delta_gap",
             "delta_ucb", "fass_pairs", "graphcut_pairs", "dpp_pairs", "dropquery_pairs", "probcover_pairs", "maxherding_pairs", "laplace_bald")
ORIGINAL_NAMES = ("random", "uncertainty", "core_set", "cluster_quota_uncertainty", "uncertainty_diversity", "cluster_margin_pairwise", "mc_dropout_probability_variance", "mc_dropout_mutual_information")
ENSEMBLE_NAMES = tuple(ENSEMBLE)
STRATEGY_FAMILY = ORIGINAL_NAMES + NEW_NAMES + ALL_HEAD + ENSEMBLE_NAMES   # 33 variants including random
RANDOM_REPLICATES = 5
EXTRA_BASELINE = "random_pair_type"   # reported separately: uniform pair, then uniform type of that pair


def make_selector(name: str, exp):
    """Return select(cands, labeled, model, cache, budget, seed) for a non-random, non-ensemble strategy."""
    if name in ROW_RULES: return ROW_RULES[name]
    if name in PAIR_RULES: rule, quality = PAIR_RULES[name]; return pair_level(rule, quality)
    def via_experiment(label_free: bool, base: str, pair_unit: bool):
        inner = lambda pairs, labeled, u, cache, need, seed, model: _experiment_rule(exp, base, label_free, pairs, labeled, cache, need, seed, model)
        if pair_unit: return pair_level(inner, False)
        def select(cands, labeled, model, cache, budget, seed=0): return _experiment_select(exp, base, cands, labeled, model, cache, budget, seed)
        return select
    if name == "core_set": return via_experiment(False, "core_set", True)
    if name.endswith("_lf"): return via_experiment(True, name[:-3], True)
    if name in ORIGINAL_ROW: return via_experiment(False, name, False)
    raise ValueError(name)


def _experiment_select(exp, base, cands, labeled, model, cache, budget, seed):
    exp.cfg.seed, previous = seed, exp.cfg.seed
    try:
        labeled_pairs = list(dict.fromkeys(x["base_pair"] for x in labeled))
        return exp.select(base, cands, model, cache, [], budget=budget, labeled_ids=labeled_pairs)[0]
    finally: exp.cfg.seed = previous


def _experiment_rule(exp, base, label_free, pairs, labeled, cache, need, seed, model):
    exp.cfg.seed, previous = seed, exp.cfg.seed
    try:
        if label_free: inputs, model = frozen.label_free_inputs(pairs, model)
        else: inputs = pairs
        picked = exp.select(base, inputs, model, cache, [], budget=need, labeled_ids=[x["pair_id"] for x in labeled])[0]
        ids = {x["pair_id"]: i for i, x in enumerate(pairs)}; return [ids[x["pair_id"]] for x in picked]
    finally: exp.cfg.seed = previous
