"""Prior-centred (and optionally type-coupled) last-layer preference head.

Hidden layer phi(x) = ReLU(W1 x + b1) comes from the head trained on the reference and bad-image anchors only (no pair labels), so cross-validation on the pair
rows is honest.  The last layer of type k is W_k = W0_k + q + V_k, where W0 is the anchor-only solution (prior mean: L2-SP style shrinkage towards the anchor model),
V_k a type-specific deviation and q a deviation shared by all types (the "quality" direction every attribute judgment may share).  Penalty
(lam_v/2) sum_k ||V_k||^2 + (lam_q/2) ||q||^2; lam_q = inf gives independent types.  The data term is the repository's loss (Bradley-Terry for decisive, |a-b| for ties,
push-down for not_apply, reference-ranking and bad-image anchors), which is convex in the last layer, so the MAP is unique and independent of a learning-rate schedule.
The penalty strengths are chosen by grouped 5-fold cross-validation of the held-out decisive log-loss on the labelled rows (never on the test groups).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

import judgment_unit_study as ju_study
import nd_core as core

ACTIVE = (0, 2, 3, 4)
GRID_V = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
COUPLING = (None, 1.0, 0.1)   # None: independent types; else lam_q = lam_v * c (smaller c = stronger sharing)


class Problem:
    def __init__(self, ctx, rows, phi_fn, W0, b0, anchor_weight=.25, bad_weight=.10):
        self.W0, self.b0 = W0.double(), b0.double(); self.anchor_weight, self.bad_weight = anchor_weight, bad_weight
        refs, bad = core.anchors(ctx, phi_fn); self.refs = {c: v.double() for c, v in refs.items()}; self.bad = None if bad is None else bad.double()
        self.set_rows(rows, phi_fn)
        self.names = list(self.refs); self.col = {c: core.frozen.TYPE_TO_INDEX[c] for c in self.names}

    def set_rows(self, rows, phi_fn):
        self.n = len(rows)
        if self.n == 0: self.pa = self.pb = torch.empty(0, self.W0.shape[1], dtype=torch.float64); self.k = torch.empty(0, dtype=torch.long); self.w = torch.empty(0, dtype=torch.float64); self.win = np.array([], dtype=object); return
        self.pa, self.pb = phi_fn(list(rows.resolved_img1)).double(), phi_fn(list(rows.resolved_img2)).double(); self.k = torch.as_tensor(rows.type_idx.to_numpy())
        self.w = torch.as_tensor(rows.confidence_weight.to_numpy(), dtype=torch.float64); self.win = rows.Winner.to_numpy()

    def loss(self, W, b, mask=None):
        """Data term of the repository's loss on the rows (optionally a subset ``mask``), anchors always included."""
        total = torch.zeros((), dtype=torch.float64)
        if self.n:
            idx = torch.arange(self.n) if mask is None else torch.as_tensor(np.flatnonzero(mask))
            if len(idx):
                k = self.k[idx]; a = (self.pa[idx] * W[k]).sum(1) + b[k]; c = (self.pb[idx] * W[k]).sum(1) + b[k]; win = self.win[idx.numpy()]
                m1, m2 = torch.as_tensor(win == "1"), torch.as_tensor(win == "2"); mt, mn = torch.as_tensor(win == "tie"), torch.as_tensor(win == "not_apply")
                terms = -F.logsigmoid(a - c) * m1 - F.logsigmoid(c - a) * m2 + (a - c).abs() * mt + (F.relu(a) + F.relu(c)) * mn
                total = total + (terms * self.w[idx]).sum() / len(idx)
        if len(self.refs) > 1:
            scores = {c: f @ W.T + b for c, f in self.refs.items()}; values = []
            for p in self.names:
                for o in self.names:
                    if o != p: values.append(-F.logsigmoid(scores[p][:, self.col[p]][:, None] - scores[o][:, self.col[p]][None, :]).mean())
            total = total + self.anchor_weight * torch.stack(values).mean()
        if self.bad is not None and len(self.bad): total = total + self.bad_weight * F.relu(self.bad @ W.T + b + 1.0).mean()
        return total

    def decisive_log_loss(self, W, b, mask) -> float:
        idx = np.flatnonzero(mask & np.isin(self.win, ["1", "2"]))
        if len(idx) == 0: return float("nan")
        idx_t = torch.as_tensor(idx); k = self.k[idx_t]; d = (self.pa[idx_t] * W[k]).sum(1) - (self.pb[idx_t] * W[k]).sum(1)
        sign = torch.as_tensor(np.where(self.win[idx] == "1", 1.0, -1.0)); w = self.w[idx_t]
        return float(((F.softplus(-d * sign)) * w).sum() / w.sum())

    def solve(self, lam_v: float, lam_q, mask=None, iters: int = 60):
        q = torch.zeros(self.W0.shape[1], dtype=torch.float64, requires_grad=True); V = torch.zeros_like(self.W0, requires_grad=True); b = self.b0.clone().requires_grad_(True)
        params = [V, b] + ([q] if lam_q is not None else [])
        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=iters, line_search_fn="strong_wolfe", tolerance_grad=1e-7, tolerance_change=1e-10)
        def closure():
            opt.zero_grad(); W = self.W0 + V + (q if lam_q is not None else 0.0)
            penalty = .5 * lam_v * (V[list(ACTIVE)] ** 2).sum() + (.5 * lam_q * (q ** 2).sum() if lam_q is not None else 0.0)
            loss = self.loss(W, b, mask) + penalty; loss.backward(); return loss
        opt.step(closure)
        with torch.no_grad(): W = self.W0 + V + (q if lam_q is not None else 0.0)
        return W.detach(), b.detach()


def grouped_folds(rows, n_folds: int, seed: int):
    groups = rows.pair_id.to_numpy(); uniq = np.unique(groups); rng = np.random.default_rng(seed); rng.shuffle(uniq); fold_of = {g: i % n_folds for i, g in enumerate(uniq)}
    return np.array([fold_of[g] for g in groups])


def fit_bayes(ctx, labelled, seed: int, coupled: bool, prior_centre: bool = True, fixed=None, raw: bool = False):
    """Return (predictor, chosen (lam_v, lam_q)).  ``raw``: linear Bradley-Terry head directly on the 512-d SimCLR features (zero prior mean) instead of on the anchor-trained hidden layer."""
    fn = core.identity_features(ctx)
    if raw:
        phi = lambda paths: fn(paths).detach(); W0 = torch.zeros(5, 512); b0 = torch.zeros(5)
    else:
        lr, steps = ctx.params_for([])  # anchor-only head: the 'initial' schedule, no pair labels
        anchor_head = core.fit_generic(ctx, [], fn, 512, lr, steps, seed)
        L1 = torch.nn.Sequential(anchor_head[0], anchor_head[1]); phi = lambda paths: L1(fn(paths)).detach()
        W0, b0 = anchor_head[3].weight.detach(), anchor_head[3].bias.detach()
        if not prior_centre: W0 = torch.zeros_like(W0); b0 = torch.zeros_like(b0)
    rows = ju_study.rows_of(ctx.exp, labelled); problem = Problem(ctx, rows, phi, W0, b0)
    if fixed is not None: lam_v, lam_q = fixed
    else:
        folds = grouped_folds(rows, 5, seed)
        def cv_score(lam_v, lam_q):
            losses = []
            for f in range(5):
                train = folds != f
                if train.sum() < 5 or (~train).sum() == 0: continue
                W, b = problem.solve(lam_v, lam_q, mask=train); v = problem.decisive_log_loss(W, b, ~train)
                if not np.isnan(v): losses.append(v)
            return float(np.mean(losses)) if losses else np.inf
        scores = {lam_v: cv_score(lam_v, None) for lam_v in GRID_V}; lam_v = min(scores, key=lambda v: (scores[v], v)); lam_q = None; best = scores[lam_v]
        if coupled:   # stage 2: with the type-specific strength fixed, try sharing a deviation across types
            for c in COUPLING[1:]:
                sc = cv_score(lam_v, lam_v * c)
                if sc < best - 1e-9: best, lam_q = sc, lam_v * c
    W, b = problem.solve(lam_v, lam_q)
    def predictor(test):
        with torch.no_grad():
            pa, pb = phi(test.img1).double(), phi(test.img2).double(); k = torch.as_tensor(test.k)
            return ((pa - pb) * W[k]).sum(1).numpy()
    return predictor, (lam_v, lam_q)


def bayes_independent(ctx, labelled, seed): return fit_bayes(ctx, labelled, seed, coupled=False)[0]
def bayes_coupled(ctx, labelled, seed): return fit_bayes(ctx, labelled, seed, coupled=True)[0]
def bayes_zero_prior(ctx, labelled, seed): return fit_bayes(ctx, labelled, seed, coupled=False, prior_centre=False)[0]
def ridge_bt(ctx, labelled, seed): return fit_bayes(ctx, labelled, seed, coupled=False, raw=True)[0]
def ridge_bt_coupled(ctx, labelled, seed): return fit_bayes(ctx, labelled, seed, coupled=True, raw=True)[0]


LEARNERS = {"bayes_independent": bayes_independent, "bayes_coupled": bayes_coupled, "bayes_zero_prior": bayes_zero_prior, "ridge_bt": ridge_bt, "ridge_bt_coupled": ridge_bt_coupled}
