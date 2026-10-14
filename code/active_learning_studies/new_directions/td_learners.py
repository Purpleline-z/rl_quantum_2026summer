"""Learners of the transductive / forward-in-time study.  Names are parsed (``get(name)``), nothing is registered by hand:

  BASE                       lap3_k10_st4 on the inductive node set (forward: ``ind`` = past only; standard / blocked: ``std`` = the repository's graph) -- the baseline of the pre-registration
  STD                        lap3_k10_st4 on the repository's graph (forward: every frame except the test images, i.e. the future frames are nodes) -- decomposition only
  G0                         aw8_all_st2lr0.5 (the tuned graph-free control)
  L:<mode>[:l<lam>][:f<frac>][:shuf]   (f = share of the sampled edges that touch the images about to be scored)  lap3_k10_st4 on the node set <mode> in {ind, std, trans_test, trans_fut}; ``:shuf`` the shuffled-graph control (node identities of the adjacency permuted)
  SP:<fit>:<prop>:a<alpha>[:shuf]   test-time score propagation: head of L:<fit>, scores of every node of the <prop> graph smoothed S* = (1-a) sum_t a^t P^t S0, test judgments scored from S*
  TW:<mode>:t<tau>[:shuf]    lap3_k10_st4 on <mode> with recency weights exp(-age / tau) (age in calendar-ordered groups, mean weight normalised to 1; forward cuts only)
  AQ:<mode>:r<w>[:shuf]      lap3_k10_st4 on <mode> + weight-w type-posterior ranking loss; the posterior q = label spreading of the ideal-image classes over the <mode> graph (reaches the test frames when they are nodes)
  C:<mode>:c<w>              graph-free control: dropout-consistency (weight w) of the scores of random nodes of <mode>, no graph;  LC:<mode>:c<w> = Laplacian + consistency
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
    if getattr(ctx, "forward", None) is None: return "std" if mode == "ind" else mode
    return f"trans_win_{ctx._window}" if mode == "trans_win" else mode


def _digest(v):
    import hashlib
    if isinstance(v, dict): v = sorted(v.items())
    if hasattr(v, "numpy"): v = v.numpy().tobytes()
    return hashlib.md5(repr(v).encode() if not isinstance(v, bytes) else v).hexdigest()


def fitted(ctx, labelled, seed, mode, shuf=False, lam=3.0, **kw):
    """Head of the Laplacian learner on the node set ``mode`` (cached per labelled set within one run of the runner)."""
    store = ctx.__dict__.setdefault("_td_heads", {}); key = _key(ctx, labelled, seed, mode, shuf, lam, tuple(sorted((k, _digest(v)) for k, v in kw.items())), getattr(ctx, '_window', None) if mode == 'trans_win' else None)
    if key not in store:
        gd = td.td_graph(ctx, _mode(ctx, mode), 10, shuf); store[key] = td.fit_td(ctx, labelled, seed, gd, lam=lam, **kw)
    return store[key]


def focus_of(ctx, gd):
    """Boolean mask over the nodes of ``gd``: images of the test groups about to be scored (forward: the current window; standard / blocked: the held-out groups)."""
    if getattr(ctx, "forward", None) is not None: groups = ctx.forward["near"] if ctx._window == "FA" else ctx.forward["far"]; paths = {p for g in groups for p in td.group_images(ctx, g)}
    else: paths = td.heldout_paths(ctx)
    return torch.as_tensor([p in paths for p in gd.paths])


def tok(parts, letter, default):
    for p in parts[2:]:
        if p[:1] == letter and p[1:].replace(".", "", 1).isdigit(): return float(p[1:])
    return default


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
        mode = parts[1]; lam = tok(parts, "l", 3.0); foc = tok(parts, "f", 0.0)
        def learner(ctx, labelled, seed):
            gd = td.td_graph(ctx, _mode(ctx, mode), 10, shuf); kw = dict(focus_mask=focus_of(ctx, gd), focus_frac=foc) if foc > 0 else {}
            h = fitted(ctx, labelled, seed, mode, shuf, lam=lam, **kw); return lambda test: core.head_d(h, featfn(ctx), test)
        learner.per_window = mode == "trans_win"; return learner
    if kind == "C":   # graph-free control on the same node set: dropout-consistency of the scores of random nodes (weight c), no edges (lam = 0); "LC" adds the Laplacian
        mode, c = parts[1], float(parts[2][1:])
        def learner(ctx, labelled, seed):
            h = fitted(ctx, labelled, seed, mode, False, lam=0.0, cons=c); return lambda test: core.head_d(h, featfn(ctx), test)
        learner.per_window = mode == "trans_win"; return learner
    if kind == "LC":
        mode, c = parts[1], float(parts[2][1:])
        def learner(ctx, labelled, seed):
            h = fitted(ctx, labelled, seed, mode, shuf, cons=c); return lambda test: core.head_d(h, featfn(ctx), test)
        learner.per_window = mode == "trans_win"; return learner
    if kind == "SP":
        fit_mode, prop_mode, alpha = parts[1], parts[2], float(parts[3][1:])
        def learner(ctx, labelled, seed):
            h = fitted(ctx, labelled, seed, fit_mode); gd = td.td_graph(ctx, _mode(ctx, prop_mode), 10, False)
            s = td.propagate_scores(gd, h, alpha, ALPHA_STEPS, shuffle_seed=ctx.seed if shuf else None)
            return lambda test: td.d_from_node_scores(gd, s, h, test, cache=ctx.features.cache, shuffle_seed=ctx.seed if shuf else None)
        learner.per_window = 'trans_win' in (fit_mode, prop_mode); return learner
    if kind == "TW":
        mode, tau = parts[1], float(parts[2][1:])
        def learner(ctx, labelled, seed):
            h = fitted(ctx, labelled, seed, mode, shuf, rowmult=recency(ctx, labelled, tau)); return lambda test: core.head_d(h, featfn(ctx), test)
        learner.per_window = mode == 'trans_win'; return learner
    if kind == "AQ":
        mode, w = parts[1], tok(parts, "r", 1.0); foc = tok(parts, "f", 0.0); lam = tok(parts, "l", 3.0)
        def learner(ctx, labelled, seed):
            gd = td.td_graph(ctx, _mode(ctx, mode), 10, shuf); kw = dict(focus_mask=focus_of(ctx, gd), focus_frac=foc) if foc > 0 else {}
            h = fitted(ctx, labelled, seed, mode, shuf, lam=lam, rankq=w, q=gd.q(), **kw); return lambda test: core.head_d(h, featfn(ctx), test)
        learner.per_window = mode == 'trans_win'; return learner
    if kind == "CT":
        mode, n, margin = parts[1], int(parts[2][1:]), float(parts[3][1:])
        def learner(ctx, labelled, seed):
            gd = td.td_graph(ctx, _mode(ctx, mode), 10, shuf); h = fitted(ctx, labelled, seed, mode, shuf)
            for _ in range(2):
                extra = pseudo_rows(ctx, h, gd, labelled, seed, n, margin)
                h = td.fit_td(ctx, labelled, seed, gd, extra=extra)
            return lambda test: core.head_d(h, featfn(ctx), test)
        learner.per_window = mode == 'trans_win'; return learner
    raise KeyError(name)
