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
def vopt(cands, labeled, model, cache, budget, seed=0, ridge=1.0, informative=None, target_decisive=None, sensitivity=True, quota=False, flip=False):
    items = cands + labeled; h, p, k = ju._own(model, items, cache); n = len(cands); w = p * (1 - p)
    inverse = ju._inverses(h, w, k, n, ridge)
    info = np.ones(n) if informative is None else informative(cands, labeled, cache)
    c = (w[:n] ** 2 if sensitivity else 1.0) * (np.ones(n) if target_decisive is None else target_decisive(cands, labeled, cache))
    members = {kk: np.flatnonzero(k[:n] == kk) for kk in ACTIVE_HEADS}
    def gains(kk):
        idx = members[kk]
        if len(idx) == 0: return idx, np.zeros(0)
        P = h[idx]; G = P @ inverse[kk] @ P.T; own = np.diag(G); cj = c[idx]
        return idx, info[idx] * w[idx] / (1 + w[idx] * own) * ((cj[:, None] * G ** 2).sum(0))
    score = np.full(n, -np.inf); table = {}
    for kk in ACTIVE_HEADS: idx, g = gains(kk); score[idx] = g
    alive = np.ones(n, bool); chosen = []
    if quota:   # at most ceil(pool share of the type * budget) picks per head, so the types are allocated like the pool (stratified design)
        cap = {kk: int(np.ceil(len(members[kk]) / n * budget)) for kk in ACTIVE_HEADS}; used = {kk: 0 for kk in ACTIVE_HEADS}
    for _ in range(min(budget, n)):
        masked = np.where(alive, score, -np.inf)
        if quota:
            for kk in ACTIVE_HEADS:
                if used[kk] >= cap[kk]: masked[members[kk]] = -np.inf
            if not np.isfinite(masked).any(): masked = np.where(alive, score, -np.inf)
        pick = int(np.argmax(masked)); chosen.append(pick); alive[pick] = False; kk = k[pick]
        if quota: used[kk] += 1
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


def _with(fn, **fixed):
    def wrapped(*a, **kw): return fn(*a, **fixed, **kw)
    return wrapped


def vopt_uniform(cands, labeled, model, cache, budget, seed=0, ridge=1.0):
    """vopt with unit target weights (every pool judgment counts the same, instead of the sensitivity weight w_j^2)."""
    items = cands + labeled; h, p, k = ju._own(model, items, cache); n = len(cands)
    # reuse vopt by temporarily replacing the sensitivity weights: emulate with target_decisive = 1/w^2
    w = (p * (1 - p))[:n]
    return vopt(cands, labeled, model, cache, budget, seed, ridge, target_decisive=lambda c, l, ca: 1.0 / np.maximum(w, 1e-6) ** 2)


def aopt(cands, labeled, model, cache, budget, seed=0, ridge=1.0):
    """Greedy A-optimal design (reduction of trace Sigma) of the own-type last layer: gain = w ||Sigma phi||^2 / (1 + w phi' Sigma phi)."""
    items = cands + labeled; h, p, k = ju._own(model, items, cache); n = len(cands); w = p * (1 - p); inverse = ju._inverses(h, w, k, n, ridge)
    def gain(i): s = inverse[k[i]] @ h[i]; return w[i] * (s @ s) / (1 + w[i] * (h[i] @ s))
    score = np.array([gain(i) for i in range(n)]); alive = np.ones(n, bool); chosen = []
    for _ in range(min(budget, n)):
        pick = int(np.argmax(np.where(alive, score, -np.inf))); chosen.append(pick); alive[pick] = False; kk = k[pick]
        inverse[kk] = _rank_one(inverse[kk], h[pick], w[pick])
        for i in np.flatnonzero(alive & (k[:n] == kk)): score[i] = gain(i)
    return [cands[i] for i in chosen]



# ------------------------------------------------------------------------------------------ NTK I-optimal design (all head parameters)
def _ntk(model, items, cache):
    """Neural tangent kernel of the preference logit d = r_k(a) - r_k(b) (own type k) over ALL head parameters (W1, b1, W2_k; biases b2 cancel).
    Last-layer part: only between judgments of the same type head; first-layer part (W1, b1 are shared by all heads): between every pair of judgments."""
    head = model.reward_head.eval(); W1 = head[0].weight.detach().numpy().astype(np.float64); b1 = head[0].bias.detach().numpy().astype(np.float64); W2 = head[3].weight.detach().numpy().astype(np.float64)
    xa = np.stack([cache[x["img1"]].numpy() for x in items]).astype(np.float64); xb = np.stack([cache[x["img2"]].numpy() for x in items]).astype(np.float64)
    za, zb = xa @ W1.T + b1, xb @ W1.T + b1; ma, mb = (za > 0).astype(np.float64), (zb > 0).astype(np.float64); k = np.array([int(x["type_idx"]) for x in items]); w = W2[k]
    phi = np.maximum(za, 0) - np.maximum(zb, 0); same = (k[:, None] == k[None, :]).astype(np.float64); K = same * (phi @ phi.T); n = len(items)
    for h in range(W1.shape[0]):
        U = np.concatenate([(ma[:, h:h + 1] * xa - mb[:, h:h + 1] * xb) * w[:, h:h + 1], ((ma[:, h] - mb[:, h]) * w[:, h])[:, None]], axis=1); K += U @ U.T
    return K, k


def ntk_vopt(cands, labeled, model, cache, budget, seed=0, tau=1.0):
    """Greedy variance reduction of the preference logits over the candidate pool under the linearised (tangent-kernel) posterior of the whole head:
    prior covariance K / tau, Gaussian observations with precision w_i = p_i (1 - p_i); gain of candidate i = sum_j C_ji^2 / (C_ii + 1/w_i)."""
    items = cands + labeled; n = len(cands); K, k = _ntk(model, items, cache); _, p, _ = ju._own(model, items, cache); w = np.clip(p * (1 - p), 1e-4, None)
    C = K / tau; L = np.arange(n, len(items))
    if len(L):
        A = C[np.ix_(L, L)] + np.diag(1 / w[L]); C = C - C[:, L] @ np.linalg.solve(A, C[L, :])
    C = C[:n, :n].copy(); alive = np.ones(n, bool); chosen = []
    for _ in range(min(budget, n)):
        gain = (C ** 2).sum(0) / (np.diag(C) + 1 / w[:n]); pick = int(np.argmax(np.where(alive, gain, -np.inf))); chosen.append(pick); alive[pick] = False
        c = C[:, pick].copy(); C -= np.outer(c, c) / (C[pick, pick] + 1 / w[pick])
    return [cands[i] for i in chosen]


def ntk_vopt_t1(*a, **kw): return ntk_vopt(*a, tau=1.0, **kw)
def ntk_vopt_t300(*a, **kw): return ntk_vopt(*a, tau=300.0, **kw)


# ------------------------------------------------------------------------------------------ type-targeted rules (HTR = head 4)
def _restrict(rule, type_idx, fill=None):
    """Apply ``rule`` to the candidates of one reconstruction type only; if the budget exceeds them, fill the rest with ``fill`` (default: ``rule``) on the others."""
    def select(cands, labeled, model, cache, budget, seed=0, **kw):
        own = [x for x in cands if int(x["type_idx"]) == type_idx]; picked = rule(own, labeled, model, cache, min(budget, len(own)), seed) if own else []
        if len(picked) < budget:
            ids = {x["pair_id"] for x in picked}; rest = [x for x in cands if x["pair_id"] not in ids]
            picked = picked + (fill or rule)(rest, labeled + picked, model, cache, budget - len(picked), seed)
        return picked
    return select


def random_htr(cands, labeled, model, cache, budget, seed=0):
    rng = np.random.default_rng(seed * 7919 + len(labeled)); own = [x for x in cands if int(x["type_idx"]) == 4]; rest = [x for x in cands if int(x["type_idx"]) != 4]
    take = list(rng.permutation(len(own))[:budget]); picked = [own[i] for i in take]
    if len(picked) < budget: picked += [rest[i] for i in rng.permutation(len(rest))[: budget - len(picked)]]
    return picked


def vopt_u(*a, **kw): return vopt(*a, sensitivity=False, **kw)
def vopt_u_quota(*a, **kw): return vopt(*a, sensitivity=False, quota=True, **kw)
def vopt_u_inf1_quota(*a, **kw): return vopt(*a, sensitivity=False, informative=decisive_v1, quota=True, **kw)
def vopt_u_inf2(*a, **kw): return vopt(*a, sensitivity=False, informative=decisive_v2, **kw)
def vopt_u_dec2(*a, **kw): return vopt(*a, sensitivity=False, informative=decisive_v2, target_decisive=decisive_v2, **kw)
def vopt_u_inf1(*a, **kw): return vopt(*a, sensitivity=False, informative=decisive_v1, **kw)


NEW = {"random_htr": random_htr, "vopt_htr": None, "fisher_htr": None, "ntk_vopt_t1": ntk_vopt_t1, "ntk_vopt_t300": ntk_vopt_t300, "vopt_u_quota": vopt_u_quota, "vopt_u_inf1_quota": vopt_u_inf1_quota, "aopt": aopt, "vopt_u": vopt_u, "vopt_u_inf2": vopt_u_inf2, "vopt_u_dec2": vopt_u_dec2, "vopt_u_inf1": vopt_u_inf1,
       "vopt_r03": _with(vopt, ridge=0.3), "vopt_r3": _with(vopt, ridge=3.0), "vopt_r10": _with(vopt, ridge=10.0), "vopt_uniform": vopt_uniform,
       "vopt": vopt_plain, "vopt_dec": vopt_dec, "vopt_dec2": vopt_dec2, "vopt_inf2": vopt_inf2, "bald_dec2": bald_dec2, "fisher_dec": fisher_dec, "fisher_dec2": fisher_dec2}


NEW["vopt_htr"] = _restrict(vopt_u, 4, fill=vopt_u)
NEW["fisher_htr"] = _restrict(ju.fisher_dopt, 4, fill=vopt_u)
