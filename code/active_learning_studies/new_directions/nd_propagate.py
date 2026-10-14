"""Non-parametric score propagation on the kNN graph of unlabelled images (study ``claude/score-propagation-*``).

FREE SCORES.  Every graph node i gets a free score vector f(i) in R^4 (one score per active type 0/2/3/4).  The scores minimise a convex objective

    L(F) =  1/n * sum_rows BT-terms(f_t(a) - f_t(b))                      Bradley-Terry likelihood of the labelled judgments (the repository's terms: winner, tie |d| smoothed, not_apply relu smoothed)
          + aw * mean_(p,o) mean_(i in anchors(p), j in anchors(o)) -log sigma(f_p(i) - f_p(j))   the reference-anchor ranking terms (all ideal images, as in aw8_all)
          + bad_w * mean relu(f(bad) + 1)
          + lam * mean_edges ||f(i) - f(j)||^2                              graph Laplacian penalty over ALL nodes
          + rho * mean_nodes ||f(i)||^2                                     ridge (gauge / shrinkage)

solved by damped Newton with the exact dense Hessian (separable per type column, convex; tie and relu terms smoothed: relu = softplus(10 x)/10).  No parametric head.  Images that are not nodes (held-out test images) receive the kernel-weighted average of the scores of their
``kt`` nearest nodes (cosine on mean-centred, normalised SimCLR features; weights softmax(sim / tau)).  Anchors / labelled-row images / bad images that are not graph nodes (the 28 outer-test ideal images, which the graph excludes
by design, or content twins of held-out images) are appended as extra nodes with kNN edges into the graph (``extra=True``); held-out test images are never nodes.

HYBRID (a): S = H + R with H the detached head scores (``lap3_k10_st4``) and R free residual scores; the likelihood / anchor terms act on S, the penalties (Laplacian ``lamr`` and ridge ``rhor``) act on R only (``pen='R'``)
or the Laplacian acts on S (``pen='S'``).  Test score = H(x) + kernel average of R.

Baselines: ``ls`` (pure label spreading of the anchors' class one-hots over the graph, no labelled judgment used), ``kn`` (free scores without Laplacian, test score = kernel average over the SCORED nodes only: plain
nearest-neighbour in score space), ``fs_lam0`` (free scores without penalty = memorisation).  Controls: ``sh1`` = shuffled graph (node identities of the adjacency permuted, extras attached to random nodes).

Variant names (parsed by ``parse_name``): ``fs_lam30_aw8_rho0.01_kt5_tau0.05[_sh1][_ex0][_k10][_rs1]``, ``hy_lamr10_rhor0.1_kt5_tau0.05[_penS]``, ``ls``, ``kn_kt5``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

import nd_core as core
import judgment_unit_study as ju_study
import nd_graph_learner as G

ACTIVE = G.ACTIVE
COL = {t: c for c, t in enumerate(ACTIVE)}


def _norm(x: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    xc = x - mu; return xc / xc.norm(dim=1, keepdim=True).clamp_min(1e-9)


class PropData:
    """Per-(context, graph) structure: node set + extras, edges, anchor index sets, test kernel similarities."""
    def __init__(self, ctx, gd, extra: bool = True, shuffle: bool = False):
        cache = ctx.features.cache; self.ctx, self.gd = ctx, gd; N = len(gd.paths); e = ctx.exp
        anchor_paths = {c: [str(p) for p in list(ps) + list(e.utility_images.get(c, [])) + list(e.test_images.get(c, []))] for c, ps in e.references.items()}
        bad_paths = [str(p) for p in e.bad_paths]
        row_imgs = set()
        for j in list(ctx.initial) + list(ctx.pool): row_imgs |= {str(ctx.info[j]["img1"]), str(ctx.info[j]["img2"])}
        held = {str(x) for g in list(ctx.validation) + list(ctx.test) for r in e.groups[g].itertuples() for x in (r.resolved_img1, r.resolved_img2)}
        held_h = set(G.content_hashes(sorted(held)).values()) if held else set()
        cand = sorted({p for ps in anchor_paths.values() for p in ps} | set(bad_paths)); hc = G.content_hashes(cand) if cand else {}
        cand = [p for p in cand if p in row_imgs or hc[p] not in held_h]     # a typed/bad image whose content equals a held-out pair-group image never becomes a node (labelled content twins may: they are labelled data)
        wanted = sorted(set(cand) | row_imgs)
        self.extra_paths = [p for p in wanted if p not in gd.index] if extra else [p for p in wanted if p not in gd.index and p in row_imgs]   # row images must exist as nodes
        self.index = dict(gd.index); self.index.update({p: N + i for i, p in enumerate(self.extra_paths)}); self.N = N; self.M = N + len(self.extra_paths)
        self.feat = torch.cat([gd.x] + ([torch.stack([cache[p] for p in self.extra_paths]).float()] if self.extra_paths else []))        # raw features of all nodes (head input)
        edges = [gd.edges]
        if self.extra_paths:
            xe = _norm(self.feat[N:], gd.mu); nb = (xe @ gd.xn.T).topk(gd.k, dim=1).indices
            if shuffle: nb = torch.as_tensor(np.random.default_rng(ctx.seed * 31 + len(self.extra_paths)).integers(0, N, nb.shape))
            ee = torch.stack([torch.arange(N, self.M)[:, None].expand_as(nb).reshape(-1), nb.reshape(-1)], 1); edges.append(ee)
        self.edges = torch.cat(edges).long()
        self.anchor_idx = {COL[core.frozen.TYPE_TO_INDEX[c]]: torch.as_tensor([self.index[p] for p in ps if p in self.index]) for c, ps in anchor_paths.items()}
        self.bad_idx = torch.as_tensor([self.index[p] for p in bad_paths if p in self.index], dtype=torch.long)
        self.xn_nodes = _norm(self.feat, gd.mu); self._test_cache = {}

    def test_sims(self, test) -> tuple[torch.Tensor, torch.Tensor]:
        key = id(test)
        if key not in self._test_cache:
            cache = self.ctx.features.cache; f = lambda paths: _norm(torch.stack([cache[str(p)] for p in paths]).float(), self.gd.mu) @ self.xn_nodes[:self.N].T
            self._test_cache[key] = (f(test.img1), f(test.img2))
        return self._test_cache[key]


def prop_for(ctx, k: int = 10, shuffle: bool = False, extra: bool = True) -> PropData:
    cached = ctx.__dict__.setdefault("_prop", {}); key = (k, shuffle, extra)
    if key not in cached: cached[key] = PropData(ctx, G.graph_for(ctx, k, shuffle), extra, shuffle)
    return cached[key]


@dataclass
class Rows:
    ia: torch.Tensor; ib: torch.Tensor; col: torch.Tensor; win1: torch.Tensor; win2: torch.Tensor; tie: torch.Tensor; na: torch.Tensor; w: torch.Tensor; n: int


def rows_of(pd_: PropData, labelled) -> Rows:
    rows = ju_study.rows_of(pd_.ctx.exp, labelled)
    if not len(rows): z = torch.zeros(0); l = torch.zeros(0, dtype=torch.long); return Rows(l, l, l, z, z, z, z, z, 0)
    idx = lambda s: torch.as_tensor([pd_.index[str(p)] for p in s]); win = rows.Winner.to_numpy(); m = lambda v: torch.as_tensor((win == v).astype(np.float64))
    return Rows(idx(rows.resolved_img1), idx(rows.resolved_img2), torch.as_tensor([COL[int(t)] for t in rows.type_idx.to_numpy()]), m("1"), m("2"), m("tie"), m("not_apply"),
                torch.as_tensor(rows.confidence_weight.to_numpy(), dtype=torch.float64), len(rows))


BETA = 10.0


def _sp(x: torch.Tensor) -> torch.Tensor:
    """smoothed relu (softplus with sharpness BETA): keeps the problem twice differentiable so that Newton converges quadratically"""
    return F.softplus(BETA * x) / BETA


def objective(S: torch.Tensor, R: torch.Tensor | None, pd_: PropData, rows: Rows, aw: float, bad_w: float, lam: float, rho: float, pen_total: bool = False) -> torch.Tensor:
    """S = total scores (M x 4); penalties act on R if given (hybrid) else on S."""
    P = S if R is None else R
    loss = torch.zeros((), dtype=S.dtype)
    if rows.n:
        a, b = S[rows.ia, rows.col], S[rows.ib, rows.col]; d = a - b
        terms = -F.logsigmoid(d) * rows.win1 - F.logsigmoid(-d) * rows.win2 + torch.sqrt(d * d + 1e-4) * rows.tie + (_sp(a) + _sp(b)) * rows.na
        loss = loss + (terms * rows.w).sum() / rows.n
    if aw > 0:
        vals = []
        for p, ip in pd_.anchor_idx.items():
            for o, io in pd_.anchor_idx.items():
                if o != p: vals.append(F.softplus(-(S[ip, p][:, None] - S[io, p][None, :])).mean())
        loss = loss + aw * torch.stack(vals).mean()
    if bad_w > 0 and len(pd_.bad_idx): loss = loss + bad_w * _sp(S[pd_.bad_idx] + 1.0).mean()
    if lam > 0:
        Q = S if pen_total else P; e = pd_.edges; loss = loss + lam * ((Q[e[:, 0]] - Q[e[:, 1]]) ** 2).sum(1).mean()
    if rho > 0: loss = loss + rho * (P ** 2).sum(1).mean()
    return loss


def solve_lbfgs(pd_: PropData, rows: Rows, aw: float = 8.0, bad_w: float = 0.1, lam: float = 30.0, rho: float = 0.01, H: torch.Tensor | None = None, pen_total: bool = False,
                max_iter: int = 300) -> torch.Tensor:
    """Reference solver (slow: the Laplacian makes the problem ill-conditioned, 300 L-BFGS iterations stop ~0.01 above the optimum); kept for tests."""
    X = torch.zeros(pd_.M, len(ACTIVE), dtype=torch.float64, requires_grad=True)
    Hd = None if H is None else H.double()
    opt = torch.optim.LBFGS([X], lr=1.0, max_iter=max_iter, tolerance_grad=1e-8, tolerance_change=1e-12, history_size=30, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); S = X if Hd is None else Hd + X; loss = objective(S, None if Hd is None else X, pd_, rows, aw, bad_w, lam, rho, pen_total); loss.backward(); return loss
    opt.step(closure); return X.detach()


def _dense_laplacian(pd_: PropData) -> torch.Tensor:
    if getattr(pd_, "_L", None) is None:
        e = pd_.edges; L = torch.zeros(pd_.M, pd_.M, dtype=torch.float64); one = torch.ones(len(e), dtype=torch.float64)
        L.index_put_((e[:, 0], e[:, 0]), one, accumulate=True); L.index_put_((e[:, 1], e[:, 1]), one, accumulate=True); L.index_put_((e[:, 0], e[:, 1]), -one, accumulate=True); L.index_put_((e[:, 1], e[:, 0]), -one, accumulate=True)
        pd_._L = L
    return pd_._L


def _add_pairs(Hm: torch.Tensor, u: torch.Tensor, v: torch.Tensor, k: torch.Tensor) -> None:
    Hm.index_put_((u, u), k, accumulate=True); Hm.index_put_((v, v), k, accumulate=True); Hm.index_put_((u, v), -k, accumulate=True); Hm.index_put_((v, u), -k, accumulate=True)


def solve(pd_: PropData, rows: Rows, aw: float = 8.0, bad_w: float = 0.1, lam: float = 30.0, rho: float = 0.01, H: torch.Tensor | None = None, pen_total: bool = False,
          max_iter: int = 40, tol: float = 1e-7) -> torch.Tensor:
    """Free scores F (M x 4) (residuals R when H, the detached head scores of all M nodes, is given), by damped Newton with the exact (dense, Cholesky) Hessian.  The objective is separable over the four type columns
    (rows, anchors, bad images, Laplacian and ridge all act column-wise), so each column is a convex problem of size M; the not_apply / bad hinge terms have zero curvature (the Laplacian + ridge keep the Hessian PD)."""
    M, E = pd_.M, len(pd_.edges); n_cols = len(ACTIVE); X = torch.zeros(M, n_cols, dtype=torch.float64); Hd = None if H is None else H.double()
    L = _dense_laplacian(pd_); H0 = (2.0 * lam / E) * L + (2.0 * rho / M) * torch.eye(M, dtype=torch.float64)
    n_pairs = sum(1 for p in pd_.anchor_idx for o in pd_.anchor_idx if o != p)
    def loss_and_grad(Xv):
        Z = Xv.clone().requires_grad_(True); S = Z if Hd is None else Hd + Z; l = objective(S, None if Hd is None else Z, pd_, rows, aw, bad_w, lam, rho, pen_total); l.backward(); return float(l), Z.grad.detach()
    loss, g = loss_and_grad(X)
    for _ in range(max_iter):
        if float(g.abs().max()) < tol: break
        S = X if Hd is None else Hd + X; step = torch.zeros_like(X)
        for c in range(n_cols):
            Hm = H0.clone(); f = S[:, c]
            if rows.n:
                m = rows.col == c
                if m.any():
                    u, v = rows.ia[m], rows.ib[m]; d = f[u] - f[v]; sg = torch.sigmoid(d) * torch.sigmoid(-d)
                    k = rows.w[m] / rows.n * (sg * (rows.win1[m] + rows.win2[m]) + 1e-4 / (d * d + 1e-4) ** 1.5 * rows.tie[m]); _add_pairs(Hm, u, v, k)
            if rows.n:   # smoothed not_apply hinge on both images
                m = rows.col == c
                if m.any() and bool(rows.na[m].any()):
                    for side in (rows.ia[m], rows.ib[m]):
                        x = f[side]; k = rows.w[m] / rows.n * rows.na[m] * BETA * torch.sigmoid(BETA * x) * torch.sigmoid(-BETA * x); Hm.index_put_((side, side), k, accumulate=True)
            if bad_w > 0 and len(pd_.bad_idx):
                x = f[pd_.bad_idx] + 1.0; k = bad_w / (len(pd_.bad_idx) * n_cols) * BETA * torch.sigmoid(BETA * x) * torch.sigmoid(-BETA * x); Hm.index_put_((pd_.bad_idx, pd_.bad_idx), k, accumulate=True)
            if aw > 0 and c in pd_.anchor_idx:
                ic = pd_.anchor_idx[c]
                for o, io in pd_.anchor_idx.items():
                    if o == c: continue
                    u = ic[:, None].expand(len(ic), len(io)).reshape(-1); v = io[None, :].expand(len(ic), len(io)).reshape(-1); d = f[u] - f[v]
                    _add_pairs(Hm, u, v, aw / n_pairs / (len(ic) * len(io)) * torch.sigmoid(d) * torch.sigmoid(-d))
            step[:, c] = -torch.cholesky_solve(g[:, c:c + 1], torch.linalg.cholesky(Hm))[:, 0]
        t, gs = 1.0, float((g * step).sum())
        while True:
            new_loss, new_g = loss_and_grad(X + t * step)
            if new_loss <= loss + 1e-4 * t * gs or t < 1e-4: break
            t *= 0.5
        if loss - new_loss < 1e-12 and t < 1e-4: break
        X = X + t * step; loss, g = new_loss, new_g
    return X


# ------------------------------------------------------------------------------------------ prediction at non-node images
def kernel_weights(sims: torch.Tensor, kt: int, tau: float, scored: torch.Tensor | None = None) -> torch.Tensor:
    s = sims.clone()
    if scored is not None: s[:, ~scored] = -9.0
    top = s.topk(min(kt, s.shape[1]), dim=1); w = torch.softmax(top.values / tau, dim=1) if tau > 0 else torch.full_like(top.values, 1.0 / top.values.shape[1])
    out = torch.zeros_like(s); out.scatter_(1, top.indices, w.to(out.dtype)); return out


def predict_d(pd_: PropData, test, Fn: torch.Tensor, kt: int, tau: float, scored: torch.Tensor | None = None, offset: tuple | None = None) -> np.ndarray:
    """d = score(img1) - score(img2) of the own type; ``Fn`` = node scores (M x 4, only the first N rows are used for interpolation); ``offset`` = (H1, H2) head scores of the test images (hybrid)."""
    s1, s2 = pd_.test_sims(test); N = pd_.N
    w1, w2 = kernel_weights(s1, kt, tau, None if scored is None else scored[:N]), kernel_weights(s2, kt, tau, None if scored is None else scored[:N])
    a, b = (w1 @ Fn[:N].float()), (w2 @ Fn[:N].float())
    if offset is not None: a, b = a + offset[0], b + offset[1]
    cols = torch.as_tensor([COL[int(t)] for t in test.k]); r = torch.arange(len(cols)); return (a - b)[r, cols].numpy()


def scored_mask(pd_: PropData, rows: Rows) -> torch.Tensor:
    m = torch.zeros(pd_.M, dtype=torch.bool); m[rows.ia] = True; m[rows.ib] = True
    for ip in pd_.anchor_idx.values(): m[ip] = True
    return m


# ------------------------------------------------------------------------------------------ variants
NAME = re.compile(r"([A-Za-z]+)([0-9.]+(?:e-?[0-9]+)?)?$")


def parse_name(name: str) -> dict:
    """``fs_lam30_aw8_rho0.01_kt5_tau0.05_sh1`` -> {'kind': 'fs', 'lam': 30.0, ...}; flags without a number are True (``penS`` -> pen_total)."""
    toks = name.split("_"); out = {"kind": toks[0]}
    for t in toks[1:]:
        m = NAME.match(t)
        if not m: raise ValueError(name)
        key, val = m.group(1), m.group(2); out[key] = float(val) if val is not None else True
    return out


DEFAULTS = {"lam": 30.0, "aw": 8.0, "rho": 0.01, "kt": 5, "tau": 0.05, "bad": 0.1, "k": 10, "sh": 0, "ex": 1, "rs": 0, "lamr": 10.0, "rhor": 0.1, "penS": False}


def fit_variant(ctx, labelled, name: str, H_cache: dict | None = None):
    """Returns a predictor ``test -> d`` for a non-head variant (``fs``, ``kn``, ``ls``).  Hybrids are built in ``sp_run`` where the head is fitted once per cell."""
    p = {**DEFAULTS, **parse_name(name)}; kind = p["kind"]; pd_ = prop_for(ctx, int(p["k"]), bool(p["sh"]), bool(p["ex"]))
    if kind == "ls":
        q = G.graph_for(ctx, int(p["k"]), bool(p["sh"])).q(); Fn = torch.zeros(pd_.M, len(ACTIVE)); Fn[:pd_.N] = q
        if p.get("log"): Fn = torch.log(Fn + 1e-3)
        return lambda test: predict_d(pd_, test, Fn, int(p["kt"]), p["tau"])
    rows = rows_of(pd_, labelled)
    if kind == "kn": Fn = solve(pd_, rows, p["aw"], p["bad"], 0.0, p["rho"]); return lambda test: predict_d(pd_, test, Fn, int(p["kt"]), p["tau"], scored_mask(pd_, rows))
    if kind == "fs":
        Fn = solve(pd_, rows, p["aw"], p["bad"], p["lam"], p["rho"]); scored = scored_mask(pd_, rows) if p["rs"] else None
        return lambda test: predict_d(pd_, test, Fn, int(p["kt"]), p["tau"], scored)
    raise ValueError(name)
