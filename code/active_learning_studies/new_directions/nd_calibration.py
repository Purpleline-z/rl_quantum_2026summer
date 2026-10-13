"""Calibration of the Laplacian learner ``lap3_k10_st4`` (branch claude/calibration-*).

Stage 1 (``label_set_record``, needs torch): for one (seed, split, budget, draw) label set it trains
  * the candidate head itself (``lap3_k10_st4``: aw8_all + Laplacian lambda 3, k = 10 graph, training steps x 4) -> test logits ``d`` (identical to ``nd_graph_learner.LEARNERS['lap3_k10_st4']``);
  * ``K`` cross-fitting heads on K-1 folds of the LABELLED judgments (folds = pair groups, so no group is in two folds), same lr / steps as the full head -> out-of-fold (OOF) logits of every
    labelled judgment (and the folds' test logits, used by the cross-fit ensemble);
  * optionally 4 extra heads on the full labelled set (other initialisation seeds) -> logit ensemble;
  * optionally heads trained with a label-smoothed Bradley-Terry loss.
Everything a calibrator may use about the data is the labelled judgments' OOF logits and image-geometry features; the test judgments' labels are only used to SCORE (``metrics_variants``).

Stage 2 (``calibrators``, numpy only): every calibrator is ``fit(oof) -> params`` and ``apply(params, test, d) -> d'``; ``oof`` holds only labelled judgments.  ``score`` computes log-loss, accuracy, AUC,
reliability-bin sums (for ECE), per-type log-loss and the anchor-far / anchor-near strata from the transformed logits.
"""
from __future__ import annotations

import contextlib
import gzip
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score

import judgment_unit_study as ju_study
import nd_core as core
import nd_graph_learner as G

ACTIVE = (0, 2, 3, 4)
TYPE_NAME = core.TYPE_NAMES
LAM, K_GRAPH, STEPS = 3.0, 10, 4.0          # lap3_k10_st4
N_BINS = 10
FAR_THRESHOLD = 0.03                         # anchor-far: both images > 0.03 (1 - cos) from every typed anchor (red-team stratum)
OUTCOME = {"1": 1, "2": -1, "tie": 0, "not_apply": 9}


# ------------------------------------------------------------------------------------------ stage 1: heads
class FixedSchedule:
    """View of a context whose ``params_for`` returns a fixed (lr, steps): the cross-fitting heads use the schedule of the FULL labelled set, not the one their smaller fold set would select."""
    def __init__(self, ctx, lr_steps): self._ctx, self._ls = ctx, lr_steps
    def __getattr__(self, name): return getattr(self._ctx, name)
    def params_for(self, judgments): return self._ls


def smoothed_bt_terms(eps: float):
    def terms_fn(a, b, m, weight):
        t = ((-(1 - eps) * F.logsigmoid(a - b) - eps * F.logsigmoid(b - a)) * m["1"] + (-(1 - eps) * F.logsigmoid(b - a) - eps * F.logsigmoid(a - b)) * m["2"]
             + (a - b).abs() * m["tie"] + (F.relu(a) + F.relu(b)) * m["not_apply"])
        return (t * weight).sum()
    return terms_fn


@contextlib.contextmanager
def label_smoothing(eps: float):
    """Swap the Bradley-Terry term of ``nd_graph_learner`` for its label-smoothed version (target 1 - eps / eps instead of 1 / 0) while a head is trained."""
    old = G._bt_terms
    if eps > 0: G._bt_terms = smoothed_bt_terms(eps)
    try: yield
    finally: G._bt_terms = old


def fit_head(ctx, gd, labelled, seed: int, lr_steps=None, eps: float = 0.0):
    """The ``lap3_k10_st4`` head (with ``eps`` = 0 exactly the registry learner)."""
    view = ctx if lr_steps is None else FixedSchedule(ctx, lr_steps)
    with label_smoothing(eps):
        return G.fit_mlp_graph(view, labelled, seed, gd, core.identity_features(ctx), gd.x, gd.x.shape[1], lap=LAM, steps_scale=STEPS)


class Items:
    """Minimal stand-in for ``core.TestSet`` (img1, img2, own type) for labelled judgments."""
    def __init__(self, ctx, judgments):
        self.img1 = [ctx.info[j]["img1"] for j in judgments]; self.img2 = [ctx.info[j]["img2"] for j in judgments]; self.k = np.array([ctx.info[j]["type_idx"] for j in judgments], dtype=int)


def logits(head, ctx, items) -> np.ndarray:
    return core.head_d(head, core.identity_features(ctx), items)


def folds_of(labelled, k: int, seed: int) -> list[list]:
    """K folds of the labelled judgments, assigned by pair group (a group is never split)."""
    groups = sorted({j[0] for j in labelled}); order = np.random.default_rng(seed).permutation(len(groups)); fold_of = {groups[i]: n % k for n, i in enumerate(order)}
    return [[j for j in sorted(set(labelled)) if fold_of[j[0]] == f] for f in range(k)]


# ------------------------------------------------------------------------------------------ image geometry features
def density(gd, cache, paths) -> np.ndarray:
    """Mean (1 - cos) to the 10 nearest graph nodes (excluding the image itself): large = sparse region of the unlabelled-image graph."""
    x = torch.stack([cache[p] for p in paths]).float() - gd.mu; xn = x / x.norm(dim=1, keepdim=True).clamp_min(1e-9); s = xn @ gd.xn.T
    for i, p in enumerate(paths):
        if p in gd.index: s[i, gd.index[p]] = -2.0
    return (1 - s.topk(10, dim=1).values).mean(1).numpy()


def anchor_paths(ctx) -> list:
    return sorted({str(p) for t in (ctx.exp.references, ctx.exp.utility_images, ctx.exp.test_images) for ps in t.values() for p in ps})


def geometry(sim, gd, cache, anchors, img1, img2, train_images) -> dict:
    """Per judgment (both images): distance to the nearest typed anchor / nearest training image / density; and their min over the two images (the pair is 'far' if even the nearer image is far)."""
    out = {}
    for tag, refs in (("anc", anchors), ("lab", sorted(train_images))):
        da, db = 1 - sim.max_sim(list(img1), refs), 1 - sim.max_sim(list(img2), refs); out[f"{tag}_min"] = np.minimum(da, db); out[f"{tag}_max"] = np.maximum(da, db)
    da, db = density(gd, cache, list(img1)), density(gd, cache, list(img2)); out["dens_max"] = np.maximum(da, db); out["dens_min"] = np.minimum(da, db)
    return out


def test_arrays(test, geom: dict) -> dict:
    return {"k": test.k.copy(), "sign": test.sign.copy(), "weight": test.weight.copy(), **{f"g_{k}": v.astype(np.float32) for k, v in geom.items()}}


def label_set_record(ctx, gd, sim, cache, test, labelled, seed0: int, comps: dict, k_folds: int, anchors) -> dict:
    """Everything stage 2 needs for one label set.  ``comps``: ens (extra full-data heads), ls (tuple of label-smoothing epsilons)."""
    rec = {"n_labelled": len(labelled)}
    lr_steps = ctx.params_for(labelled); head = fit_head(ctx, gd, labelled, seed0); rec["d"] = logits(head, ctx, test)
    base_geom = geometry(sim, gd, cache, anchors, test.img1, test.img2, {x for j in labelled for x in (ctx.info[j]["img1"], ctx.info[j]["img2"])}); rec["test"] = test_arrays(test, base_geom)
    # out-of-fold logits of the labelled judgments
    rows = ju_study.rows_of(ctx.exp, labelled); lab_sorted = sorted(set(labelled)); idx = {j: i for i, j in enumerate(lab_sorted)}
    oof = {"d": np.full(len(lab_sorted), np.nan), "outcome": np.array([OUTCOME[ctx.info[j]["winner"]] for j in lab_sorted]), "weight": rows.confidence_weight.to_numpy().astype(float),
           "k": np.array([ctx.info[j]["type_idx"] for j in lab_sorted]), "fold": np.zeros(len(lab_sorted), dtype=int)}
    geoms = {}; rec["d_fold_test"] = []
    for f, held in enumerate(folds_of(labelled, k_folds, seed0)):
        train = [j for j in lab_sorted if j not in set(held)]; fh = fit_head(ctx, gd, train, seed0 + 17 * (f + 1), lr_steps); items = Items(ctx, held); d = logits(fh, ctx, items)
        for j, v in zip(held, d): oof["d"][idx[j]] = v; oof["fold"][idx[j]] = f
        g = geometry(sim, gd, cache, anchors, items.img1, items.img2, {x for j in train for x in (ctx.info[j]["img1"], ctx.info[j]["img2"])})
        for key, v in g.items(): geoms.setdefault(key, np.full(len(lab_sorted), np.nan, dtype=np.float32))[[idx[j] for j in held]] = v
        rec["d_fold_test"].append(logits(fh, ctx, test))
    oof.update({f"g_{k}": v for k, v in geoms.items()}); rec["oof"] = oof
    if comps.get("ens"): rec["d_ens"] = [logits(fit_head(ctx, gd, labelled, seed0 + 1000 * (e + 1)), ctx, test) for e in range(comps["ens"])]
    for eps in comps.get("ls", ()): rec[f"d_ls{eps:g}"] = logits(fit_head(ctx, gd, labelled, seed0, eps=eps), ctx, test)
    return rec


def save_cell(path: Path, records: dict) -> None:
    tmp = Path(str(path) + ".tmp")
    with gzip.open(tmp, "wb") as fh: pickle.dump(records, fh, protocol=4)
    tmp.replace(path)


def load_cell(path: Path) -> dict:
    with gzip.open(path, "rb") as fh: return pickle.load(fh)


# ------------------------------------------------------------------------------------------ stage 2: calibrators (numpy)
LOG_T_GRID = np.linspace(np.log(0.2), np.log(5.0), 161)


def _dec(oof):
    m = np.isin(oof["outcome"], (1, -1)); return m, oof["d"][m], oof["outcome"][m].astype(float), oof["weight"][m]


def _wll(d, s, w, t=1.0):
    return float((w * np.logaddexp(0.0, -s * d / t)).sum() / max(w.sum(), 1e-12))


def fit_global_t(oof, prior=None) -> float:
    """Temperature minimising the confidence-weighted decisive log-loss of the OOF logits (grid in log T; optional Gaussian prior on log T: (mu, sd) -> penalty (log T - mu)^2 / (2 sd^2) per... total weight)."""
    m, d, s, w = _dec(oof)
    if len(d) < 3: return 1.0
    pen = (lambda lt: 0.0) if prior is None else (lambda lt: (lt - prior[0]) ** 2 / (2 * prior[1] ** 2) / max(w.sum(), 1e-9) * 1.0)
    vals = [_wll(d, s, w, np.exp(lt)) + pen(lt) for lt in LOG_T_GRID]; return float(np.exp(LOG_T_GRID[int(np.argmin(vals))]))


def fit_type_t(oof, tau: float = 8.0) -> dict:
    """Per-type temperatures shrunk to the global one: minimise sum_i w_i loss_i(T_k) / sum w + tau * (log T_k - log T_global)^2 / n_k-free scale (penalty weight tau per unit of total weight / n)."""
    m, d, s, w = _dec(oof); tg = fit_global_t(oof); out = {k: tg for k in core.TYPE_NAMES}
    k = oof["k"][m]
    for kk in core.TYPE_NAMES:
        sel = k == kk
        if sel.sum() < 2: continue
        n = float(w[sel].sum()); lt0 = np.log(tg)
        vals = [(w[sel] * np.logaddexp(0.0, -s[sel] * d[sel] / np.exp(lt))).sum() + tau * (lt - lt0) ** 2 for lt in LOG_T_GRID]; out[kk] = float(np.exp(LOG_T_GRID[int(np.argmin(vals))]))
    return out


def apply_type_t(params: dict, test, d):
    t = np.array([params[int(k)] for k in test["k"]]); return d / t


def feature_u(geom_key: str, vals, scale: float = 0.01):
    """Standardised 'farness' feature: log(1 + (1 - cos) / scale) -- 0 for near-duplicates, grows with distance."""
    return np.log1p(np.asarray(vals, dtype=float) / scale)


def fit_conf_t(oof, key: str = "lab_min", gamma_pen: float = 4.0):
    """T_i = T * exp(gamma * u_i), u = farness of the judgment's images; (log T, gamma) minimise the OOF log-loss + gamma_pen * gamma^2 (ridge towards 'no dependence')."""
    m, d, s, w = _dec(oof)
    if len(d) < 4: return (1.0, 0.0)
    u = feature_u(key, oof[f"g_{key}"][m]); best = (np.inf, 0.0, 0.0)
    for g in np.linspace(-0.6, 1.2, 37):
        for lt in LOG_T_GRID[::2]:
            v = _wll(d, s, w, np.exp(lt + g * u)) if False else float((w * np.logaddexp(0.0, -s * d / np.exp(lt + g * u))).sum() / w.sum()) + gamma_pen * g * g / max(w.sum(), 1.0)
            if v < best[0]: best = (v, lt, g)
    return (float(np.exp(best[1])), float(best[2]))


def apply_conf_t(params, test, d, key: str = "lab_min"):
    t, g = params; return d / (t * np.exp(g * feature_u(key, test[f"g_{key}"])))


def fit_shrink_far(oof, key: str = "anc_min", delta: float = FAR_THRESHOLD):
    """Global temperature (OOF) times a far-from-anchor shrink: d' = d / T / (1 + gamma * relu(m - delta) / delta) -- gamma by OOF, ridge towards 0; (the OOF judgments rarely are far, so gamma is
    only identified when they are)."""
    m, d, s, w = _dec(oof); t = fit_global_t(oof); x = np.maximum(oof[f"g_{key}"][m] - delta, 0.0) / delta
    if (x > 0).sum() < 3: return (t, 0.0)
    best = (np.inf, 0.0)
    for g in np.linspace(0.0, 6.0, 49):
        v = _wll(d / (1 + g * x), s, w, t) + 0.02 * g * g / max(w.sum(), 1.0) * 10
        if v < best[0]: best = (v, g)
    return (t, float(best[1]))


def apply_shrink_far(params, test, d, key: str = "anc_min", delta: float = FAR_THRESHOLD):
    t, g = params; return d / t / (1 + g * np.maximum(test[f"g_{key}"] - delta, 0.0) / delta)


def davidson_nll(theta_log, tlog, d, outcome, w):
    """Davidson (1970) ties model with logits d / T: P(1) = e^{x/2}/Z, P(2) = e^{-x/2}/Z, P(tie) = nu/Z, Z = e^{x/2} + e^{-x/2} + nu; nu = exp(theta_log)."""
    x = d / np.exp(tlog); nu = np.exp(theta_log); z = np.logaddexp(np.logaddexp(x / 2, -x / 2), theta_log)
    ll = np.where(outcome == 1, x / 2 - z, np.where(outcome == -1, -x / 2 - z, theta_log - z)); return float(-(w * ll).sum() / max(w.sum(), 1e-12))


def fit_davidson(oof):
    """Platt-style post-hoc model of ALL preference outcomes: (T, nu) of Davidson's ties model fitted by OOF maximum likelihood over decisive and tie judgments (not_apply carries no preference and is dropped).
    The decisive-conditional probability that image 1 wins is P1 / (P1 + P2) = sigmoid(x / T'), with T' the effective temperature -- so what changes the decisive log-loss is T, and nu only enters through
    the likelihood of the tie rows that constrain T (the shared logit scale)."""
    mask = np.isin(oof["outcome"], (1, -1, 0)); d, o, w = oof["d"][mask], oof["outcome"][mask], oof["weight"][mask]
    if (np.abs(o) == 1).sum() < 3: return (1.0, 0.0)
    best = (np.inf, 0.0, 0.0)
    for tl in LOG_T_GRID[::2]:
        for th in np.linspace(-3, 2, 21):
            v = davidson_nll(th, tl, d, o, w)
            if v < best[0]: best = (v, tl, th)
    return (float(np.exp(best[1])), float(best[2]))


def apply_davidson(params, test, d): return d / params[0]


# ------------------------------------------------------------------------------------------ scoring
def scores(d, test) -> dict:
    """Confidence-weighted log-loss / accuracy, AUC, per-type log-loss, reliability-bin sums, and the anchor-far / anchor-near strata of the log-loss and accuracy."""
    s, w, k = test["sign"], test["weight"], test["k"]; z = s * d; ll = np.logaddexp(0.0, -z); acc = (z > 0).astype(float)
    out = {"ll": float((w * ll).sum() / w.sum()), "acc": float((w * acc).sum() / w.sum()), "auc": float(roc_auc_score(s > 0, d)) if len(set(s)) == 2 else float("nan"), "n": int(len(d))}
    for kk, name in TYPE_NAME.items():
        m = k == kk; out[f"{name}_ll"] = float(ll[m].mean()) if m.any() else float("nan"); out[f"{name}_acc"] = float(acc[m].mean()) if m.any() else float("nan")
    far = test["g_anc_min"] > FAR_THRESHOLD
    for tag, m in (("far", far), ("near", ~far)):
        out[f"{tag}_n"] = int(m.sum()); out[f"{tag}_w"] = float(w[m].sum()); out[f"{tag}_llw"] = float((w[m] * ll[m]).sum()); out[f"{tag}_accw"] = float((w[m] * acc[m]).sum())
    p = 1 / (1 + np.exp(-d)); conf = np.maximum(p, 1 - p); correct = ((p > .5) == (s > 0)).astype(float); b = np.minimum((conf - .5) / .5 * N_BINS, N_BINS - 1e-9).astype(int)
    out["bin_w"] = np.bincount(b, weights=w, minlength=N_BINS); out["bin_conf"] = np.bincount(b, weights=w * conf, minlength=N_BINS); out["bin_acc"] = np.bincount(b, weights=w * correct, minlength=N_BINS)
    return out


def ece_from_bins(bin_w, bin_conf, bin_acc) -> float:
    w = np.asarray(bin_w); tot = w.sum()
    return float(np.sum(np.abs(np.asarray(bin_acc) - np.asarray(bin_conf))) / tot) if tot > 0 else float("nan")


CALIBRATORS = {}   # name -> (needs: str, callable(rec) -> logits)


def variants(rec: dict, priors: dict | None = None) -> dict:
    """All logit variants of one label set (name -> test logits).  ``rec`` is a stage-1 record; calibrators only read ``rec['oof']`` and the test features (never the test labels)."""
    oof, test, d = rec["oof"], rec["test"], rec["d"]; out = {"raw": d}
    tg = fit_global_t(oof); out["T_glob"] = d / tg
    for tau in (2.0, 8.0, 32.0): out[f"T_type_tau{tau:g}"] = apply_type_t(fit_type_t(oof, tau), test, d)
    for key in ("lab_min", "anc_min", "dens_max"): out[f"T_conf_{key}"] = apply_conf_t(fit_conf_t(oof, key), test, d, key)
    out["shrink_far"] = apply_shrink_far(fit_shrink_far(oof), test, d)
    out["Davidson"] = apply_davidson(fit_davidson(oof), test, d)
    if "d_ens" in rec:
        e = np.mean([d] + list(rec["d_ens"]), axis=0); out["ENS"] = e; out["ENS_T"] = e / fit_global_t(oof)   # (the OOF logits come from single heads: ENS_T is a slightly conservative temperature)
    if "d_fold_test" in rec:
        e = np.mean([d] + list(rec["d_fold_test"]), axis=0); out["CF"] = e; out["CF_T"] = e / tg
    for key in rec:
        if key.startswith("d_ls"): out[key[2:]] = rec[key]; out[key[2:] + "_T"] = rec[key] / tg
    if priors:   # constants fitted on the development seeds' OOF logits (labelled judgments only): applied unchanged
        if "T_budget" in priors: out["T_prior"] = d / priors["T_budget"][rec["n_labelled_budget"]]
    return out
