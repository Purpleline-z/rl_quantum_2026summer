"""Gaussian-process preference learner on the frozen SimCLR features (Chu & Ghahramani 2005 style).

One latent utility function f_k per reconstruction type over ALL cached images (the held-out images are latent nodes without observations, so their utility follows from the
kernel only; no held-out label is ever used).  Kernel: RBF on L2-normalised SimCLR features, length scale = c x median pairwise distance.  Observations of type k:
decisive pair -> logistic Bradley-Terry likelihood of f(a) - f(b); tie -> smoothed |f(a) - f(b)|; not_apply -> softplus(f) on both images; reference anchors (class k beats the other
classes) and bad images exactly as in the repository's loss.  MAP in the dual (f = K alpha); the prior strength lam (and then the length-scale factor c) is chosen by grouped
cross-validation of the held-out decisive log-loss on the labelled rows.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

import judgment_unit_study as ju_study
import nd_core as core
from nd_bayes import grouped_folds

ACTIVE = (0, 2, 3, 4)
LAMBDAS = (0.001, 0.003, 0.01, 0.03, 0.1)
LENGTH_FACTORS = (0.25, 0.5, 1.0, 2.0)
START_FACTOR = 0.5
_STATE: dict = {}


def _nodes(ctx):
    if "paths" not in _STATE:
        paths = sorted(ctx.features.cache); X = torch.stack([ctx.features.cache[p] for p in paths]).double(); X = X / X.norm(dim=1, keepdim=True)
        D2 = (2 - 2 * X @ X.T).clamp_min(0); _STATE.update(paths=paths, index={p: i for i, p in enumerate(paths)}, D2=D2, median=float(D2[D2 > 0].median().sqrt()))
    return _STATE


def kernel(ctx, factor: float) -> torch.Tensor:
    s = _nodes(ctx)
    if factor not in s.setdefault("K", {}):
        ell = factor * s["median"]; s["K"][factor] = torch.exp(-s["D2"] / (2 * ell ** 2)) + 1e-4 * torch.eye(len(s["paths"]), dtype=torch.float64)
    return s["K"][factor]


class GP:
    def __init__(self, ctx, rows, anchor_weight=.25, bad_weight=.10):
        s = _nodes(ctx); idx = s["index"]; self.ctx = ctx; self.n = len(s["paths"])
        self.ia = np.array([idx[str(p)] for p in rows.resolved_img1], dtype=int) if len(rows) else np.empty(0, int)
        self.ib = np.array([idx[str(p)] for p in rows.resolved_img2], dtype=int) if len(rows) else np.empty(0, int)
        self.k = rows.type_idx.to_numpy().astype(int) if len(rows) else np.empty(0, int); self.w = rows.confidence_weight.to_numpy().astype(float) if len(rows) else np.empty(0)
        self.win = rows.Winner.to_numpy() if len(rows) else np.empty(0, object)
        self.refs = {c: np.array([idx[str(p)] for p in ps]) for c, ps in ctx.exp.references.items()}; self.bad = np.array([idx[str(p)] for p in ctx.exp.bad_paths], dtype=int) if ctx.exp.bad_paths else np.empty(0, int)
        self.anchor_weight, self.bad_weight = anchor_weight, bad_weight; self.col = {c: core.frozen.TYPE_TO_INDEX[c] for c in self.refs}

    def solve(self, K, lam: float, kk: int, mask=None, iters: int = 60):
        sel = (self.k == kk) if mask is None else ((self.k == kk) & mask); n_all = max(int(((self.k == kk) if mask is None else mask).sum()), 1)
        ia, ib, w, win = self.ia[sel], self.ib[sel], torch.as_tensor(self.w[sel]), self.win[sel]
        m1, m2, mt, mn = (torch.as_tensor(win == v) for v in ("1", "2", "tie", "not_apply"))
        preferred = [c for c in self.refs if self.col[c] == kk]; P = self.refs[preferred[0]] if preferred else np.empty(0, int)
        O = np.concatenate([v for c, v in self.refs.items() if self.col[c] != kk]) if len(self.refs) > 1 else np.empty(0, int)
        alpha = torch.zeros(self.n, dtype=torch.float64, requires_grad=True); opt = torch.optim.LBFGS([alpha], lr=1.0, max_iter=iters, line_search_fn="strong_wolfe", tolerance_grad=1e-7, tolerance_change=1e-10)
        def closure():
            opt.zero_grad(); f = K @ alpha; loss = (.5 * lam) * (alpha @ f)
            if len(ia):
                d = f[ia] - f[ib]; terms = F.softplus(-d) * m1 + F.softplus(d) * m2 + torch.sqrt(d ** 2 + 1e-4) * mt + (F.softplus(f[ia]) + F.softplus(f[ib])) * mn
                loss = loss + (terms * w).sum() / n_all
            if len(P) and len(O): loss = loss + self.anchor_weight * F.softplus(-(f[P][:, None] - f[O][None, :])).mean()
            if len(self.bad): loss = loss + self.bad_weight * F.softplus(f[self.bad] + 1.0).mean()
            loss.backward(); return loss
        opt.step(closure)
        with torch.no_grad(): return (K @ alpha).detach()

    def decisive_log_loss(self, fs: dict, mask) -> float:
        sel = np.flatnonzero(mask & np.isin(self.win, ["1", "2"]))
        if len(sel) == 0: return float("nan")
        d = np.array([float(fs[self.k[i]][self.ia[i]] - fs[self.k[i]][self.ib[i]]) if self.k[i] in fs else 0.0 for i in sel]); sign = np.where(self.win[sel] == "1", 1.0, -1.0); w = self.w[sel]
        return float((w * np.logaddexp(0.0, -sign * d)).sum() / w.sum())

    def solve_all(self, K, lam, mask=None): return {kk: self.solve(K, lam, kk, mask) for kk in ACTIVE}


def fit_gp(ctx, labelled, seed: int, tune_length: bool = True):
    rows = ju_study.rows_of(ctx.exp, labelled); gp = GP(ctx, rows); folds = grouped_folds(rows, 3, seed)
    def cv(factor, lam):
        K = kernel(ctx, factor); losses = []
        for f in range(3):
            train = folds != f
            if train.sum() < 5 or (~train).sum() == 0: continue
            v = gp.decisive_log_loss(gp.solve_all(K, lam, train), ~train)
            if not np.isnan(v): losses.append(v)
        return float(np.mean(losses)) if losses else np.inf
    factor = START_FACTOR; scores = {lam: cv(factor, lam) for lam in LAMBDAS}; lam = min(scores, key=lambda v: (scores[v], v)); best = scores[lam]
    if tune_length:
        for c in LENGTH_FACTORS:
            if c == START_FACTOR: continue
            sc = cv(c, lam)
            if sc < best - 1e-9: best, factor = sc, c
    fs = gp.solve_all(kernel(ctx, factor), lam); idx = _nodes(ctx)["index"]
    def predictor(test):
        ia = np.array([idx[str(p)] for p in test.img1]); ib = np.array([idx[str(p)] for p in test.img2]); d = np.zeros(len(test.k))
        for kk in ACTIVE:
            m = test.k == kk; d[m] = (fs[kk][ia[m]] - fs[kk][ib[m]]).numpy()
        return d
    return predictor, (factor, lam)


def gp_preference(ctx, labelled, seed): return fit_gp(ctx, labelled, seed, tune_length=False)[0]
def gp_tuned_length(ctx, labelled, seed): return fit_gp(ctx, labelled, seed, tune_length=True)[0]


def ensemble_mlp_gp(ctx, labelled, seed):
    """Average of the MLP head's and the GP's logit gaps, each rescaled to unit RMS on the labelled judgments' own images so neither dominates."""
    import nd_learners
    a = nd_learners.baseline(ctx, labelled, seed); b = fit_gp(ctx, labelled, seed, tune_length=False)[0]
    def predictor(test):
        da, db = np.asarray(a(test)), np.asarray(b(test)); return da / (np.sqrt((da ** 2).mean()) + 1e-9) + db / (np.sqrt((db ** 2).mean()) + 1e-9)
    return predictor


LEARNERS = {"ensemble_mlp_gp": ensemble_mlp_gp, "gp_preference": gp_preference, "gp_tuned_length": gp_tuned_length}
