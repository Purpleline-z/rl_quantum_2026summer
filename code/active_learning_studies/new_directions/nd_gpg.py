"""Graph-kernel Gaussian-process preference learner (branch claude/graph-gp-*): the Bayesian counterpart of the Laplacian learner ``lap3_k10_st4``.

Model.  One latent utility function f_k per reconstruction type k on the frozen SimCLR features (cosine geometry of the mean-centred, L2-normalised features, as in the graph).  Prior
f_k ~ GP(0, a_k * k~), with the *semi-supervised deformed kernel* of Sindhwani, Niyogi & Belkin (2005) on the kNN graph of unlabelled images (identity-safe nodes of ``nd_graph_learner.GraphData``):
    k~(x, y) = k(x, y) - k_x^T (I + beta L K)^-1 beta L k_y ,       K = k on the nodes, L = graph Laplacian,
i.e. the prior precision on the nodes becomes K^-1 + beta L: exactly the Gaussian-process version of "RKHS penalty + beta f^T L f" (the MAP of this model is manifold regularisation; the
Laplacian learner of the repository is the same penalty on an MLP).  The deformed kernel is defined for out-of-sample points (test images, outer ideal images, which are never graph nodes) through k.
Likelihood (as the repository's loss, per type): decisive pair -> Bradley-Terry, weight = confidence weight; tie -> sqrt((f_a - f_b)^2 + eps); not_apply -> softplus(f) on both images; typed ideal
images (references + utility + outer-test ideal images, the ``aw8_all`` anchors) as ranking constraints softplus(-(f_p - f_o)) with the repository's weights (aw / 12 per class pair, mean over the
image pairs, times the number of labelled rows so that the ratio to the labelled terms equals the head's); bad images softplus(f + 1).  Inference: Laplace (Newton in the dual).
Hyperparameters (kernel shape, graph strength beta, per-type amplitude a_k): Laplace log-evidence, or grouped cross-validated decisive log-loss on the in-budget judgments, or fixed.
Prediction: mean and (Laplace) variance of f(a) - f(b); probabilities sigmoid(mu / sqrt(1 + pi v / 8)) (MacKay's probit approximation).
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

import judgment_unit_study as ju_study
import nd_core as core
import nd_graph_learner as G

ACTIVE = (0, 2, 3, 4)
EPS_TIE = 1e-4
JITTER = 1e-6
DT = torch.float64


# ------------------------------------------------------------------------------------------ per-context geometry (cached)
class Geometry:
    """Nodes, features, Laplacian and kernel caches of one (context, graph) pair."""
    def __init__(self, ctx, k: int = 10, shuffle: bool = False):
        self.ctx = ctx; self.gd = G.graph_for(ctx, k, shuffle); gd = self.gd; self.n = len(gd.paths); self.cache = ctx.features.cache
        self.xn = gd.xn.to(DT); a = gd.adj.to(DT); self.deg = a.sum(1); self.L = torch.diag(self.deg) - a
        d2 = (2 - 2 * self.xn @ self.xn.T).clamp_min(0); self.median = float(d2[d2 > 1e-12].sqrt().median()); self._k = {}; self._b = {}
        self.shuffled = shuffle

    def phi(self, paths) -> torch.Tensor:
        """Centred, L2-normalised features of arbitrary images (identical to the node rows for nodes)."""
        x = torch.stack([self.cache[p] for p in paths]).to(DT) - self.gd.mu.to(DT); return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)

    def kern(self, xa, xb, kind: str, c: float) -> torch.Tensor:
        cos = (xa @ xb.T).clamp(-1, 1)
        if kind == "rbf": return torch.exp(-(2 - 2 * cos).clamp_min(0) / (2 * (c * self.median) ** 2))
        if kind == "arccos":      # NNGP kernel of an infinitely wide one-hidden-layer ReLU net on unit-norm inputs (the head's own function class)
            th = torch.arccos(cos); return (torch.sin(th) + (math.pi - th) * cos) / math.pi
        if kind == "cos": return cos + 1.0
        raise ValueError(kind)

    def knodes(self, kind, c) -> torch.Tensor:
        key = (kind, c)
        if key not in self._k: self._k[key] = self.kern(self.xn, self.xn, kind, c)
        return self._k[key]

    def deform(self, kind, c, beta) -> torch.Tensor:
        """B = (I + beta L K)^-1 beta L  (n x n, symmetric); k~(x, y) = k(x, y) - k_x^T B k_y.  beta = 0 -> plain kernel."""
        key = (kind, c, beta)
        if key not in self._b:
            if beta == 0: self._b[key] = None
            else:
                K = self.knodes(kind, c); M = beta * self.L; A = torch.eye(self.n, dtype=DT) + M @ K; B = torch.linalg.solve(A, M); self._b[key] = 0.5 * (B + B.T)
        return self._b[key]

    def ktilde(self, pa, pb, kind, c, beta, xa=None, xb=None):
        """Deformed kernel matrix between two lists of images."""
        xa = self.phi(pa) if xa is None else xa; xb = self.phi(pb) if xb is None else xb; K = self.kern(xa, xb, kind, c); B = self.deform(kind, c, beta)
        if B is None: return K
        return K - self.kern(xa, self.xn, kind, c) @ B @ self.kern(self.xn, xb, kind, c)


# ------------------------------------------------------------------------------------------ observation structure of one labelled set
def _sp(x): return torch.nn.functional.softplus(x)


class Terms:
    """All likelihood terms of one type on an index set S.  Labelled terms: (i, j, kind, weight) with u = f_i - f_j (j = -1: single image).  Anchor and bad terms are kept as matrices."""
    def __init__(self, i, j, code, w, anchors, bad_idx, bad_w, m):
        self.i, self.j, self.code, self.w, self.m = i, j, code, w, m; self.anchors = anchors; self.bad_idx, self.bad_w = bad_idx, bad_w   # anchors: list of (P idx, O idx, weight per pair)
        n = len(i); self.C = torch.zeros(n, m, dtype=DT)
        if n: r = torch.arange(n); self.C[r, i] += 1.0; hasj = j >= 0; self.C[r[hasj], j[hasj]] -= 1.0

    def value_grad_hess(self, f, want_hess=True):
        """J_data(f), its gradient and Hessian in f."""
        m = self.m; val = torch.zeros((), dtype=DT); g = torch.zeros(m, dtype=DT); H = torch.zeros(m, m, dtype=DT) if want_hess else None
        if len(self.i):
            u = self.C @ f; code = self.code; s = torch.where(code == 0, -1.0, 1.0).to(DT); c0 = torch.where(code == 4, 1.0, 0.0).to(DT)           # codes: 0 sp(-u); 1 sp(u); 2 tie; 3 sp(u) single; 4 sp(u + 1) bad-like (unused here)
            z = s * u + c0; sg = torch.sigmoid(z); tie = code == 2
            psi = torch.where(tie, torch.sqrt(u ** 2 + EPS_TIE), _sp(z)); d1 = torch.where(tie, u / torch.sqrt(u ** 2 + EPS_TIE), s * sg); d2 = torch.where(tie, EPS_TIE / (u ** 2 + EPS_TIE) ** 1.5, sg * (1 - sg))
            val = val + (self.w * psi).sum(); g = g + self.C.T @ (self.w * d1)
            if want_hess: H = H + (self.C * (self.w * d2)[:, None]).T @ self.C
        for P, O, wpair in self.anchors:
            if len(P) == 0 or len(O) == 0: continue
            u = f[P][:, None] - f[O][None, :]; sg = torch.sigmoid(-u); psi = _sp(-u); val = val + wpair * psi.sum(); gu = -wpair * sg                                   # d/du of softplus(-u) = -sigmoid(-u)
            g.index_add_(0, P, gu.sum(1)); g.index_add_(0, O, -gu.sum(0))
            if want_hess:
                W = wpair * sg * (1 - sg); H[P[:, None], P[None, :]] += torch.diag(W.sum(1)); H[O[:, None], O[None, :]] += torch.diag(W.sum(0)); H[P[:, None], O[None, :]] -= W; H[O[:, None], P[None, :]] -= W.T
        if len(self.bad_idx):
            z = f[self.bad_idx] + 1.0; sg = torch.sigmoid(z); val = val + self.bad_w * _sp(z).sum(); g.index_add_(0, self.bad_idx, self.bad_w * sg)
            if want_hess: H[self.bad_idx, self.bad_idx] += self.bad_w * sg * (1 - sg)
        return val, g, H


def solve_laplace(A: torch.Tensor, terms: Terms, iters: int = 25, tol: float = 1e-7, alpha0=None):
    """MAP in the dual: f = A alpha, J(alpha) = 1/2 alpha^T A alpha + J_data(A alpha); Newton steps delta = -(I + H A)^-1 (alpha + g) with backtracking.  Returns (alpha, f, H, J, log_evidence)."""
    m = A.shape[0]; alpha = torch.zeros(m, dtype=DT) if alpha0 is None else alpha0.clone(); I = torch.eye(m, dtype=DT)
    f = A @ alpha; val, g, H = terms.value_grad_hess(f); J = 0.5 * alpha @ f + val
    for _ in range(iters):
        delta = -torch.linalg.solve(I + H @ A, alpha + g); step = 1.0
        for _ in range(12):
            a2 = alpha + step * delta; f2 = A @ a2; v2, g2, H2 = terms.value_grad_hess(f2); J2 = 0.5 * a2 @ f2 + v2
            if J2 <= J + 1e-12: break
            step *= 0.5
        else: break
        done = abs(J - J2) < tol * (1 + abs(J)); alpha, f, val, g, H, J = a2, f2, v2, g2, H2, J2
        if done: break
    sign, logdet = torch.linalg.slogdet(I + H @ A); return alpha, f, H, float(J), float(-J - 0.5 * logdet)


# ------------------------------------------------------------------------------------------ the learner
DEFAULT = dict(kind="rbf", c=0.25, beta=0.3, aw=8.0, amp={0: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}, select="fixed", betas=(0.03, 0.1, 0.3, 1.0, 3.0), cs=(0.15, 0.25, 0.4), amps=(0.3, 1.0, 3.0, 10.0, 30.0), kinds=None, a0=3.0)


class Observations:
    """Index structure shared by all hyperparameter settings of one labelled set."""
    def __init__(self, geo: Geometry, ctx, labelled, aw: float, all_ideal: bool = True):
        self.geo = geo; rows = ju_study.rows_of(ctx.exp, labelled); self.rows = rows; self.n_lab = max(len(rows), 1)
        paths = []; seen = {}
        def add(p):
            p = str(p)
            if p not in seen: seen[p] = len(paths); paths.append(p)
            return seen[p]
        self.ia = np.array([add(p) for p in rows.resolved_img1], dtype=int) if len(rows) else np.empty(0, int); self.ib = np.array([add(p) for p in rows.resolved_img2], dtype=int) if len(rows) else np.empty(0, int)
        self.k = rows.type_idx.to_numpy().astype(int) if len(rows) else np.empty(0, int); self.w = rows.confidence_weight.to_numpy().astype(float) if len(rows) else np.empty(0); self.win = rows.Winner.to_numpy() if len(rows) else np.empty(0, object)
        self.cls = {}
        for c, ps in ctx.exp.references.items():
            allp = list(ps) + (list(ctx.exp.utility_images.get(c, [])) + list(ctx.exp.test_images.get(c, [])) if all_ideal else []); self.cls[c] = np.array([add(p) for p in allp], dtype=int)
        self.bad = np.array([add(p) for p in ctx.exp.bad_paths], dtype=int) if ctx.exp.bad_paths else np.empty(0, int); self.paths = paths; self.m = len(paths); self.aw = aw
        self.col = {c: core.frozen.TYPE_TO_INDEX[c] for c in self.cls}; self.x = geo.phi(paths)

    def terms(self, kk: int, mask=None, bad_w: float = 0.10, n_lab=None, labelled: bool = True) -> Terms:
        sel = (self.k == kk) if mask is None else ((self.k == kk) & mask); n_lab = (self.n_lab if mask is None else max(int(mask.sum()), 1)) if n_lab is None else n_lab
        if not labelled: sel = np.zeros(len(self.k), bool)
        i, j, code, w = [], [], [], []
        for t in np.flatnonzero(sel):
            ww = float(self.w[t]); win = self.win[t]
            if win == "1": i.append(self.ia[t]); j.append(self.ib[t]); code.append(0); w.append(ww)
            elif win == "2": i.append(self.ia[t]); j.append(self.ib[t]); code.append(1); w.append(ww)
            elif win == "tie": i.append(self.ia[t]); j.append(self.ib[t]); code.append(2); w.append(ww)
            else: i += [self.ia[t], self.ib[t]]; j += [-1, -1]; code += [3, 3]; w += [ww, ww]
        mk = lambda a, dt: torch.as_tensor(np.array(a, dtype=dt))
        anchors = []; pref = [c for c in self.cls if self.col[c] == kk]
        if pref:
            P = torch.as_tensor(self.cls[pref[0]])
            for c, idx in self.cls.items():
                if c != pref[0]: anchors.append((P, torch.as_tensor(idx), self.aw * n_lab / 12.0 / (len(P) * len(idx))))
        bad_idx = torch.as_tensor(self.bad); bad_wt = bad_w * n_lab / (max(len(self.bad), 1) * 5.0)
        return Terms(mk(i, int) if i else torch.empty(0, dtype=torch.long), mk(j, int) if j else torch.empty(0, dtype=torch.long), mk(code, int) if code else torch.empty(0, dtype=torch.long), mk(w, float) if w else torch.empty(0, dtype=DT), anchors, bad_idx, bad_wt, self.m)

    def decisive_ll(self, F: dict, mask) -> float:
        sel = np.flatnonzero(mask & np.isin(self.win, ["1", "2"]))
        if len(sel) == 0: return float("nan")
        d = np.array([float(F[self.k[t]][self.ia[t]] - F[self.k[t]][self.ib[t]]) if self.k[t] in F else 0.0 for t in sel]); s = np.where(self.win[sel] == "1", 1.0, -1.0); w = self.w[sel]
        return float((w * np.logaddexp(0.0, -s * d)).sum() / w.sum())


from nd_bayes import grouped_folds  # noqa: E402


class GPModel:
    """Fitted model: per-type amplitude, posterior mode, Hessian, kernel config; ``predict`` gives mean / variance of the preference of test judgments."""
    def __init__(self, geo, obs, cfg, per_type, info):
        self.geo, self.obs, self.cfg, self.per_type, self.info = geo, obs, cfg, per_type, info

    def predict(self, test) -> dict:
        geo, obs = self.geo, self.obs; paths = sorted({str(p) for p in list(test.img1) + list(test.img2)}); index = {p: i for i, p in enumerate(paths)}; xt = geo.phi(paths)
        ia = np.array([index[str(p)] for p in test.img1]); ib = np.array([index[str(p)] for p in test.img2]); mu = np.zeros(len(test.k)); var = np.zeros(len(test.k)); cache = {}; I = torch.eye(obs.m, dtype=DT)
        for kk in ACTIVE:
            m = np.flatnonzero(test.k == kk)
            if len(m) == 0 or kk not in self.per_type: continue
            st = self.per_type[kk]; key = (st["kind"], st["c"], st["beta"])
            if key not in cache: cache[key] = (geo.ktilde(None, None, *key, xa=xt, xb=obs.x), geo.ktilde(None, None, *key, xa=xt, xb=xt))
            Kts, Ktt = cache[key]; a = st["amp"]; Ats = a * Kts; fmean = Ats @ st["alpha"]; As = a * st["Ktilde"]
            post = a * Ktt - Ats @ torch.linalg.solve(I + st["H"] @ As, st["H"] @ Ats.T)                         # posterior covariance of f at the test images
            ra, rb = torch.as_tensor(ia[m]), torch.as_tensor(ib[m]); mu[m] = (fmean[ra] - fmean[rb]).numpy(); var[m] = (post[ra, ra] + post[rb, rb] - 2 * post[ra, rb]).clamp_min(0).numpy()
        return {"mu": mu, "var": var}


def prob_score(mu, var, mode: str = "probit"):
    """Logit of the predictive probability: probit approximation of E[sigmoid(d)] with d ~ N(mu, var)."""
    return mu / np.sqrt(1.0 + math.pi * var / 8.0) if mode == "probit" else mu


# ------------------------------------------------------------------------------------------ fitting with hyperparameter selection
def _fit_type(geo, obs, kk, kind, c, beta, amp, mask=None, Ktilde=None, alpha0=None, labelled=True):
    Kt = geo.ktilde(None, None, kind, c, beta, xa=obs.x, xb=obs.x) if Ktilde is None else Ktilde; Kt = Kt + JITTER * torch.eye(obs.m, dtype=DT); A = amp * Kt
    alpha, f, H, J, logev = solve_laplace(A, obs.terms(kk, mask, labelled=labelled), alpha0=alpha0); return dict(kind=kind, c=c, beta=beta, amp=amp, alpha=alpha, f=f, H=H, J=J, logev=logev, Ktilde=Kt)


def fit_gpg(ctx, labelled, seed: int, cfg: dict | None = None, geo: Geometry | None = None, shuffle: bool = False):
    """Fit the graph-kernel GP.  ``cfg`` keys: kind, c, beta, aw, amp (per type dict) for select = 'fixed'; for the learned versions select in {'evidence', 'evidence_cond', 'cv'} with the grids
    betas, cs (kernel shape; stage A at amplitude ``a0``) and amps (stage B, amplitude per type for the evidence criteria, one shared amplitude for CV).
    'evidence' = Laplace log-evidence of all terms; 'evidence_cond' = log-evidence of all terms minus that of the anchor / bad-image terms alone (the conditional marginal likelihood of the judgments
    given the anchors; the anchors' own evidence is the same for every label set and would otherwise dominate the choice); 'cv' = grouped 3-fold cross-validated decisive log-loss."""
    cfg = {**DEFAULT, **(cfg or {})}; geo = geo or geometry_for(ctx, shuffle); obs = Observations(geo, ctx, labelled, cfg["aw"]); t0 = time.monotonic(); per_type = {}; info = {"select": cfg["select"]}
    kinds = cfg["kinds"] or (cfg["kind"],); a0 = cfg.get("a0", 3.0); sel = cfg["select"]
    if sel == "fixed":
        for kk in ACTIVE: per_type[kk] = _fit_type(geo, obs, kk, cfg["kind"], cfg["c"], cfg["beta"], cfg["amp"][kk])
    else:
        grid = [(kd, c, b) for kd in kinds for c in cfg["cs"] for b in cfg["betas"]]; scores = {}; kt = {}
        if sel in ("evidence", "evidence_cond"):
            for key in grid:
                kt[key] = geo.ktilde(None, None, *key, xa=obs.x, xb=obs.x); tot = 0.0
                for kk in ACTIVE:
                    r = _fit_type(geo, obs, kk, *key, a0, Ktilde=kt[key]); tot += r["logev"] - (_fit_type(geo, obs, kk, *key, a0, Ktilde=kt[key], labelled=False)["logev"] if sel == "evidence_cond" else 0.0)
                scores[key] = tot
            key = max(scores, key=scores.get); info["config"] = key; info["crit"] = scores[key]
            for kk in ACTIVE:
                cand = []
                for a in cfg["amps"]:
                    r = _fit_type(geo, obs, kk, *key, a, Ktilde=kt[key]); cand.append((r["logev"] - (_fit_type(geo, obs, kk, *key, a, Ktilde=kt[key], labelled=False)["logev"] if sel == "evidence_cond" else 0.0), r))
                per_type[kk] = max(cand, key=lambda t: t[0])[1]
        elif sel == "cv":
            folds = grouped_folds(obs.rows, 3, seed)
            def cv_loss(key, a):
                Kt = kt.setdefault(key, geo.ktilde(None, None, *key, xa=obs.x, xb=obs.x)); losses = []
                for fo in range(3):
                    train = folds != fo
                    if train.sum() < 5 or (~train).sum() == 0: continue
                    F = {kk: _fit_type(geo, obs, kk, *key, a, mask=train, Ktilde=Kt)["f"] for kk in ACTIVE}; v = obs.decisive_ll(F, ~train)
                    if not np.isnan(v): losses.append(v)
                return float(np.mean(losses)) if losses else np.inf
            scores = {key: cv_loss(key, a0) for key in grid}; key = min(scores, key=scores.get); info["config"] = key
            ampscore = {a: cv_loss(key, a) for a in cfg["amps"]}; a = min(ampscore, key=ampscore.get); info["amp"] = a; info["crit"] = ampscore[a]
            per_type = {kk: _fit_type(geo, obs, kk, *key, a, Ktilde=kt[key]) for kk in ACTIVE}
        else: raise ValueError(sel)
    info["seconds"] = time.monotonic() - t0; info["m"] = obs.m
    return GPModel(geo, obs, cfg, per_type, info)


def geometry_for(ctx, shuffle: bool = False, k: int = 10) -> Geometry:
    cached = ctx.__dict__.setdefault("_gp_geo", {}); key = (shuffle, k)
    if key not in cached: cached[key] = Geometry(ctx, k, shuffle)
    return cached[key]
