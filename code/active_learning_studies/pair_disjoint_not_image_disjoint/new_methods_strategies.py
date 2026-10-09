"""New acquisition rules for the judgment-unit protocol (candidate = one (pair, type) judgment; own-type head).  Same selector signature as
``judgment_unit_strategies``: select(cands, labeled, model, cache, budget, seed) -> budget distinct candidates.

vopt*            pool-wide predictive-variance reduction (I/V-optimal design) on the last layer of the own-type head under the Laplace posterior:
                 greedy gain of a candidate i = P_i * w_i/(1 + w_i G_ii) * sum_j c_j G_ij^2, G = Phi Sigma Phi^T over the candidate pool,
                 c_j = (p_j (1-p_j))^2 [* P(decisive_j)] the sensitivity of the predicted preference probability of pool judgment j.
                 Variants differ in the outcome-informativeness factor P_i (1, or the predicted probability of a decisive answer).
"""
from __future__ import annotations

import numpy as np
import torch

import judgment_unit_strategies as ju
import new_pair_strategies as nps
from new_pair_strategies import ACTIVE_HEADS, _rank_one

CTX = {}   # set by new_methods_study.py: {"ctx": current judgment_unit_study.Context}


# ------------------------------------------------------------------------------------------ P(decisive) predictors
def _stats(items, cache):
    a = np.stack([nps._feature(cache, x["img1"]) for x in items]); b = np.stack([nps._feature(cache, x["img2"]) for x in items])
    return np.column_stack([np.linalg.norm(a - b, axis=1), (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))])


def _anchor_scores(items, cache):
    """Own-type scores s_k(image) of both images under the ANCHOR-ONLY head (reference and bad-image anchors, no pair labels): identical for labelled and
    candidate judgments, so there is no in-sample bias.  Returns [n, 3] = max, min, |difference|."""
    ctx = CTX["ctx"]
    if "anchor_head" not in CTX or CTX.get("anchor_seed") != ctx.seed:
        CTX["anchor_head"] = ctx.fit([]); CTX["anchor_seed"] = ctx.seed
    head = CTX["anchor_head"].reward_head.eval(); k = np.array([x["type_idx"] for x in items])
    def s(key):
        with torch.no_grad(): v = head[3](head[1](head[0](torch.stack([cache[x[key]] for x in items])))).numpy().astype(float)
        return v[np.arange(len(items)), k]
    a, b = s("img1"), s("img2"); return np.column_stack([np.maximum(a, b), np.minimum(a, b), np.abs(a - b)])


def decisive_v1(cands, labeled, cache):
    """The existing predictor (type + distance/cosine), own type only."""
    d = nps._decisive_probability(cands, labeled, cache); return d[np.arange(len(cands)), [int(x["type_idx"]) for x in cands]]


def decisive_v2(cands, labeled, cache, C=0.3):
    """Type + distance/cosine + anchor-only head scores of both images; regularised logistic regression on the outcomes revealed so far (0.5 if one class)."""
    from sklearn.linear_model import LogisticRegression
    ys = np.array([int(ok) for x in labeled for _, ok in x.get("outcomes", [])]); n = len(cands)
    if len(set(ys.tolist())) < 2: return np.full(n, .5)
    def design(items):
        k = np.array([x["type_idx"] for x in items]); return np.column_stack([np.eye(5)[k], _stats(items, cache), _anchor_scores(items, cache)])
    X = design(labeled); mu, sd = X[:, 5:].mean(0), X[:, 5:].std(0) + 1e-6; X[:, 5:] = (X[:, 5:] - mu) / sd
    Z = design(cands); Z[:, 5:] = (Z[:, 5:] - mu) / sd
    return LogisticRegression(C=C, max_iter=2000).fit(X, ys).predict_proba(Z)[:, 1]


# ------------------------------------------------------------------------------------------ pool-wide variance reduction
def vopt(cands, labeled, model, cache, budget, seed=0, ridge=1.0, informative=None, target_decisive=None):
    items = cands + labeled; h, p, k = ju._own(model, items, cache); n = len(cands); w = p * (1 - p)
    inverse = ju._inverses(h, w, k, n, ridge)
    info = np.ones(n) if informative is None else informative(cands, labeled, cache)
    c = (w[:n]) ** 2 * (np.ones(n) if target_decisive is None else target_decisive(cands, labeled, cache))
    members = {kk: np.flatnonzero(k[:n] == kk) for kk in ACTIVE_HEADS}
    def gains(kk):
        idx = members[kk]
        if len(idx) == 0: return idx, np.zeros(0)
        P = h[idx]; G = P @ inverse[kk] @ P.T; own = np.diag(G); cj = c[idx]
        return idx, info[idx] * w[idx] / (1 + w[idx] * own) * ((cj[:, None] * G ** 2).sum(0))
    score = np.full(n, -np.inf); table = {}
    for kk in ACTIVE_HEADS: idx, g = gains(kk); score[idx] = g
    alive = np.ones(n, bool); chosen = []
    for _ in range(min(budget, n)):
        pick = int(np.argmax(np.where(alive, score, -np.inf))); chosen.append(pick); alive[pick] = False; kk = k[pick]
        inverse[kk] = _rank_one(inverse[kk], h[pick], w[pick]); idx, g = gains(kk); score[idx] = np.where(alive[idx], g, -np.inf)
    return [cands[i] for i in chosen]


def vopt_plain(*a, **kw): return vopt(*a, **kw)
def vopt_dec(*a, **kw): return vopt(*a, informative=decisive_v1, target_decisive=decisive_v1, **kw)
def vopt_dec2(*a, **kw): return vopt(*a, informative=decisive_v2, target_decisive=decisive_v2, **kw)
def vopt_inf2(*a, **kw): return vopt(*a, informative=decisive_v2, **kw)


# ------------------------------------------------------------------------------------------ BALD / Fisher with the improved outcome model
def bald_dec2(cands, labeled, model, cache, budget, seed=0, ridge=1.0, samples=64):
    """BALD of the own head under the Laplace posterior x P(decisive) from decisive_v2 (type + pair statistics + anchor-only head scores)."""
    return ju._bald_greedy(cands, labeled, model, cache, budget, seed, ridge, samples, decisive_v2(cands, labeled, cache))


def fisher_dec(cands, labeled, model, cache, budget, seed=0, ridge=1.0, decisive=decisive_v1):
    """Greedy D-optimal Fisher design of the own head with the gain multiplied by the predicted probability of a decisive answer."""
    items = cands + labeled; h, p, k = ju._own(model, items, cache); n = len(cands); w = p * (1 - p); inverse = ju._inverses(h, w, k, n, ridge)
    d = decisive(cands, labeled, cache); chosen = []; remaining = list(range(n))
    for _ in range(min(budget, n)):
        gains = [d[i] * np.log1p(w[i] * h[i] @ inverse[k[i]] @ h[i]) for i in remaining]; pick = remaining[int(np.argmax(gains))]
        inverse[k[pick]] = _rank_one(inverse[k[pick]], h[pick], w[pick]); chosen.append(pick); remaining.remove(pick)
    return [cands[i] for i in chosen]


def fisher_dec2(*a, **kw): return fisher_dec(*a, decisive=decisive_v2, **kw)


NEW = {"vopt": vopt_plain, "vopt_dec": vopt_dec, "vopt_dec2": vopt_dec2, "vopt_inf2": vopt_inf2, "bald_dec2": bald_dec2, "fisher_dec": fisher_dec, "fisher_dec2": fisher_dec2}
