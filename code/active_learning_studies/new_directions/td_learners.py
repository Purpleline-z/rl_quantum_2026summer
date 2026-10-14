"""Learners of the transductive / forward-in-time study.  Names are parsed (``get(name)``), nothing is registered by hand:

  BASE                       lap3_k10_st4 on the inductive node set (forward: ``ind`` = past only; standard / blocked: ``std`` = the repository's graph) -- the baseline of the pre-registration
  STD                        lap3_k10_st4 on the repository's graph (forward: every frame except the test images, i.e. the future frames are nodes) -- decomposition only
  G0                         aw8_all_st2lr0.5 (the tuned graph-free control)
  L:<mode>[:shuf]            lap3_k10_st4 on the node set <mode> in {ind, std, trans_test, trans_fut}; ``:shuf`` the shuffled-graph control (node identities of the adjacency permuted)
  SP:<fit>:<prop>:a<alpha>[:shuf]   test-time score propagation: head of L:<fit>, scores of every node of the <prop> graph smoothed S* = (1-a) sum_t a^t P^t S0, test judgments scored from S*
  TW:<mode>:t<tau>[:shuf]    lap3_k10_st4 on <mode> with recency weights exp(-age / tau) (age in calendar-ordered groups, mean weight normalised to 1; forward cuts only)
  AQ:<mode>:r<w>[:shuf]      lap3_k10_st4 on <mode> + weight-w type-posterior ranking loss; the posterior q = label spreading of the ideal-image classes over the <mode> graph (reaches the test frames when they are nodes)
  CT:<mode>:n<M>:m<margin>   co-training: head of L:<mode>, M confident pseudo-labelled pairs per type among unlabelled nodes, refit (2 rounds)
All of them see features of the node set only; labels are never used for any node outside the labelled judgments.
"""
from __future__ import annotations

import numpy as np
import torch

import nd_core as core
import nd_graph_learner as G
import td_core as td

ALPHA_STEPS = 20


def _key(ctx, labelled, seed, *parts):
    return (tuple(sorted(labelled)), seed) + parts


def _mode(ctx, mode):
    return "std" if (mode in ("ind",) and getattr(ctx, "forward", None) is None) else mode


def fitted(ctx, labelled, seed, mode, shuf=False, lam=3.0, **kw):
    """Head of the Laplacian learner on the node set ``mode`` (cached per labelled set within one run of the runner)."""
    store = ctx.__dict__.setdefault("_td_heads", {}); key = _key(ctx, labelled, seed, mode, shuf, lam, tuple(sorted((k, str(v)[:40]) for k, v in kw.items())))
    if key not in store:
        gd = td.td_graph(ctx, _mode(ctx, mode), 10, shuf); store[key] = td.fit_td(ctx, labelled, seed, gd, lam=lam, **kw)
    return store[key]


def recency(ctx, labelled, tau: float) -> dict:
    """age (in calendar-ordered groups) of every labelled judgment, weight exp(-age/tau), normalised to mean 1 over the judgments."""
    rank = ctx.forward["order_rank"]; newest = max(rank[j[0]] for j in labelled); raw = {j: float(np.exp(-(newest - rank[j[0]]) / tau)) for j in set(labelled)}; mean = float(np.mean(list(raw.values())))
    return {j: v / mean for j, v in raw.items()}


def pseudo_rows(ctx, head, gd, labelled, seed, n_per_type: int, margin: float):
    """Confident pseudo-labelled pairs among unlabelled nodes (not an image of any labelled judgment, not an ideal image): for every active type the ``n_per_type`` random pairs with the largest
    |s_k(i) - s_k(j)| among 20 x n random candidates whose gap exceeds ``margin``; winner = the image with the larger score; weight 1."""
    used = {str(x) for x in td.rt.labelled_images(ctx, labelled)}; unl = [i for i, p in enumerate(gd.paths) if p not in used and td.img_time(p) is not None]
    rng = np.random.default_rng(seed * 6007 + 3)
    with torch.no_grad(): s = head(gd.x)
    xa, xb, typ, win = [], [], [], []
    for k in td.ACTIVE:
        i = rng.choice(unl, size=20 * n_per_type); j = rng.choice(unl, size=20 * n_per_type); gap = (s[i, k] - s[j, k]).numpy(); ok = np.where((np.abs(gap) >= margin) & (i != j))[0]
        ok = ok[np.argsort(-np.abs(gap[ok]))][:n_per_type]
        for o in ok: xa.append(gd.x[i[o]]); xb.append(gd.x[j[o]]); typ.append(k); win.append("1" if gap[o] > 0 else "2")
    if not xa: return None
    return (torch.stack(xa), torch.stack(xb), torch.as_tensor(typ), torch.ones(len(typ)), np.array(win, dtype=object))


def get(name: str):
    parts = name.split(":"); kind = parts[0]; shuf = parts[-1] == "shuf"; parts = [p for p in parts if p != "shuf"]
    featfn = lambda ctx: core.identity_features(ctx)
    if name == "G0": return lambda ctx, labelled, seed: G.LEARNERS["aw8_all_st2lr0.5"](ctx, labelled, seed)
    if kind in ("BASE", "STD"):
        mode = "ind" if kind == "BASE" else "std"
        def learner(ctx, labelled, seed):
            h = fitted(ctx, labelled, seed, mode); return lambda test: core.head_d(h, featfn(ctx), test)
        return learner
    if kind == "L":
        mode = parts[1]
        def learner(ctx, labelled, seed):
            h = fitted(ctx, labelled, seed, mode, shuf); return lambda test: core.head_d(h, featfn(ctx), test)
        return learner
    if kind == "SP":
        fit_mode, prop_mode, alpha = parts[1], parts[2], float(parts[3][1:])
        def learner(ctx, labelled, seed):
            h = fitted(ctx, labelled, seed, fit_mode); gd = td.td_graph(ctx, _mode(ctx, prop_mode), 10, False)
            s = td.propagate_scores(gd, h, alpha, ALPHA_STEPS, shuffle_seed=ctx.seed if shuf else None)
            return lambda test: td.d_from_node_scores(gd, s, h, test, cache=ctx.features.cache)
        return learner
    if kind == "TW":
        mode, tau = parts[1], float(parts[2][1:])
        def learner(ctx, labelled, seed):
            h = fitted(ctx, labelled, seed, mode, shuf, rowmult=recency(ctx, labelled, tau)); return lambda test: core.head_d(h, featfn(ctx), test)
        return learner
    if kind == "AQ":
        mode, w = parts[1], float(parts[2][1:])
        def learner(ctx, labelled, seed):
            gd = td.td_graph(ctx, _mode(ctx, mode), 10, shuf); h = fitted(ctx, labelled, seed, mode, shuf, rankq=w, q=gd.q()); return lambda test: core.head_d(h, featfn(ctx), test)
        return learner
    if kind == "CT":
        mode, n, margin = parts[1], int(parts[2][1:]), float(parts[3][1:])
        def learner(ctx, labelled, seed):
            gd = td.td_graph(ctx, _mode(ctx, mode), 10, shuf); h = fitted(ctx, labelled, seed, mode, shuf)
            for _ in range(2):
                extra = pseudo_rows(ctx, h, gd, labelled, seed, n, margin)
                h = td.fit_td(ctx, labelled, seed, gd, extra=extra)
            return lambda test: core.head_d(h, featfn(ctx), test)
        return learner
    raise KeyError(name)
