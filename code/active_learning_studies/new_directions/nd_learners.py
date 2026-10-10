"""Learner variants compared on identical random label sets.  A learner is ``fn(ctx, labelled, seed) -> predictor`` where ``predictor(test: TestSet) -> d``
(the own-type logit gap r_k(a) - r_k(b) for every held-out decisive judgment).  ``baseline`` is the repository's head (re-tuned per-budget schedule) built with the same
generic fit, so every comparison is paired and differs only in the named change."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageOps

import judgment_unit_study as ju_study  # noqa: F401  (path set up by nd_core)
import nd_core as core

DATA = core.single_study.DATA
FEATURE_DIM = 512


# ------------------------------------------------------------------------------------------ feature views
_MIRROR: dict[str, torch.Tensor] = {}
_MIRROR_FILE = core.HERE / "results" / "feature_cache_mirror.pt"


def mirror_features(ctx, paths) -> torch.Tensor:
    """SimCLR features of the horizontally mirrored image (RHEED patterns are mirror symmetric about the vertical axis); cached on disk."""
    if not _MIRROR and _MIRROR_FILE.exists(): _MIRROR.update(core.frozen.load_feature_cache(_MIRROR_FILE, DATA))
    keys = [str(p) for p in paths]; missing = [k for k in dict.fromkeys(keys) if k not in _MIRROR]
    if missing:
        feats = ctx.features
        with torch.no_grad():
            for i in range(0, len(missing), 16):
                chunk = missing[i:i + 16]; batch = torch.stack([feats.tf(ImageOps.mirror(Image.open(k).convert("L"))) for k in chunk])
                for k, v in zip(chunk, feats.base_model.encoder(batch)): _MIRROR[k] = v
        _MIRROR_FILE.parent.mkdir(parents=True, exist_ok=True); core.frozen.save_feature_cache(_MIRROR, _MIRROR_FILE, DATA)
    return torch.stack([_MIRROR[k] for k in keys]) if keys else torch.empty(0, FEATURE_DIM)


_NAME = re.compile(r"^(\d+)_(RR\d+[A-Z])_(\d+)C_(\d{4})")
_RUN_MAX: dict[str, int] = {}


def _run_lengths() -> dict[str, int]:
    if not _RUN_MAX:
        for p in (DATA / "original data" / "Trajectories").glob("*/*.bmp"):
            m = _NAME.match(p.name)
            if m: _RUN_MAX[m.group(2)] = max(_RUN_MAX.get(m.group(2), 0), int(m.group(1)))
    return _RUN_MAX


def meta_vector(path) -> np.ndarray:
    """[temperature/1000 C, frame index / frames in the run, time of day / 2400, present] parsed from the trajectory file name; zeros for ideal images."""
    m = _NAME.match(Path(str(path)).name)
    if not m: return np.zeros(4)
    idx, run, temp, clock = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
    return np.array([temp / 1000.0, idx / _run_lengths()[run], clock / 2400.0, 1.0])


def metadata_features(ctx, scale: float):
    rms = float(torch.stack(list(ctx.features.cache.values())).pow(2).mean().sqrt())
    def fn(paths):
        base = ctx.features.get(paths); meta = torch.tensor(np.stack([meta_vector(p) for p in paths]), dtype=torch.float32) * scale * rms if len(paths) else torch.empty(0, 4)
        return torch.cat([base, meta], dim=1)
    return fn


class SymmetricHead(nn.Module):
    """Average of the head over the original and the mirrored view: exactly invariant to the mirror symmetry, and test-time augmentation for free."""
    def __init__(self, inner: nn.Module): super().__init__(); self.inner = inner
    def forward(self, x): return .5 * (self.inner(x[..., :FEATURE_DIM]) + self.inner(x[..., FEATURE_DIM:]))


# ------------------------------------------------------------------------------------------ learners
def _lr_steps(ctx, labelled): return ctx.params_for(labelled)


def baseline(ctx, labelled, seed):
    lr, steps = _lr_steps(ctx, labelled); fn = core.identity_features(ctx)
    head = core.fit_generic(ctx, labelled, fn, FEATURE_DIM, lr, steps, seed)
    return lambda test: core.head_d(head, fn, test)


def decisive_only(ctx, labelled, seed):
    """Drop ties and not_apply rows from training (the anchors stay)."""
    lr, steps = _lr_steps(ctx, labelled); fn = core.identity_features(ctx); rows = ju_study.rows_of(ctx.exp, labelled)
    rows = rows[rows.Winner.isin(["1", "2"])] if len(rows) else rows
    head = core.fit_generic(ctx, labelled, fn, FEATURE_DIM, lr, steps, seed, rows=rows)
    return lambda test: core.head_d(head, fn, test)


def no_not_apply(ctx, labelled, seed):
    lr, steps = _lr_steps(ctx, labelled); fn = core.identity_features(ctx); rows = ju_study.rows_of(ctx.exp, labelled)
    rows = rows[rows.Winner != "not_apply"] if len(rows) else rows
    head = core.fit_generic(ctx, labelled, fn, FEATURE_DIM, lr, steps, seed, rows=rows)
    return lambda test: core.head_d(head, fn, test)


def no_tie(ctx, labelled, seed):
    lr, steps = _lr_steps(ctx, labelled); fn = core.identity_features(ctx); rows = ju_study.rows_of(ctx.exp, labelled)
    rows = rows[rows.Winner != "tie"] if len(rows) else rows
    head = core.fit_generic(ctx, labelled, fn, FEATURE_DIM, lr, steps, seed, rows=rows)
    return lambda test: core.head_d(head, fn, test)


def symmetric(ctx, labelled, seed):
    lr, steps = _lr_steps(ctx, labelled)
    def fn(paths): return torch.cat([ctx.features.get(paths), mirror_features(ctx, paths)], dim=1)
    torch.manual_seed(seed); head = SymmetricHead(core.new_head(FEATURE_DIM, seed))
    rows = ju_study.rows_of(ctx.exp, labelled); refs, bad = core.anchors(ctx, fn)
    if len(rows):
        xa, xb = fn(list(rows.resolved_img1)), fn(list(rows.resolved_img2)); typ = torch.as_tensor(rows.type_idx.to_numpy())
        weight, winner = torch.as_tensor(rows.confidence_weight.to_numpy(), dtype=torch.float32), rows.Winner.to_numpy()
    else: xa = xb = torch.empty(0, 2 * FEATURE_DIM); typ = torch.empty(0, dtype=torch.long); weight = torch.empty(0); winner = np.array([], dtype=object)
    core.frozen.fit_head(head, xa, xb, typ, weight, winner, refs, bad, lr, steps, ctx.exp.cfg.weight_decay, bad_weight=ctx.exp.cfg.bad_anchor_weight)
    head.eval(); return lambda test: core.head_d(head, fn, test)


def metadata(ctx, labelled, seed, scale: float = 1.0):
    lr, steps = _lr_steps(ctx, labelled); fn = metadata_features(ctx, scale)
    head = core.fit_generic(ctx, labelled, fn, FEATURE_DIM + 4, lr, steps, seed)
    return lambda test: core.head_d(head, fn, test)


def pseudo_label(ctx, labelled, seed, threshold: float = .9, weight: float = .5):
    """Self-training: label the unlabelled pool judgments (their outcomes stay hidden) with the current head where it is confident, retrain with them at a reduced weight."""
    lr, steps = _lr_steps(ctx, labelled); fn = core.identity_features(ctx)
    head = core.fit_generic(ctx, labelled, fn, FEATURE_DIM, lr, steps, seed)
    rest = [j for j in ctx.pool if j not in set(labelled)]
    if not rest: return lambda test: core.head_d(head, fn, test)
    img1 = [ctx.info[j]["img1"] for j in rest]; img2 = [ctx.info[j]["img2"] for j in rest]; k = np.array([ctx.info[j]["type_idx"] for j in rest])
    with torch.no_grad(): a, b = head(fn(img1)), head(fn(img2))
    p = torch.sigmoid((a - b)[torch.arange(len(rest)), torch.as_tensor(k)]).numpy(); keep = (p >= threshold) | (p <= 1 - threshold)
    if not keep.any(): return lambda test: core.head_d(head, fn, test)
    idx = np.flatnonzero(keep); winner = np.where(p[idx] >= .5, "1", "2").astype(object)
    extra = ([img1[i] for i in idx], [img2[i] for i in idx], k[idx], np.full(len(idx), weight), winner)
    head2 = core.fit_generic(ctx, labelled, fn, FEATURE_DIM, lr, steps, seed, extra=extra)
    return lambda test: core.head_d(head2, fn, test)


CV_GRID = ((3e-4, 100), (1e-3, 50), (1e-3, 100), (1e-3, 300), (3e-3, 50), (3e-3, 100), (1e-2, 50), (1e-2, 100))


def cv_schedule(ctx, labelled, seed):
    """Schedule-free head: (learning rate, steps) chosen by grouped 4-fold cross-validation of the decisive log-loss on the labelled rows, instead of the fixed per-budget table."""
    import numpy as np
    import nd_bayes
    fn = core.identity_features(ctx); rows = ju_study.rows_of(ctx.exp, labelled); folds = nd_bayes.grouped_folds(rows, 4, seed); best = (np.inf, None)
    for lr, steps in CV_GRID:
        losses = []
        for f in range(4):
            tr, te = rows[folds != f], rows[folds == f]; te = te[te.Winner.isin(["1", "2"])]
            if len(tr) < 5 or len(te) == 0: continue
            head = core.fit_generic(ctx, [], fn, FEATURE_DIM, lr, steps, seed, rows=tr)
            with torch.no_grad(): d = (head(fn(list(te.resolved_img1))) - head(fn(list(te.resolved_img2))))[torch.arange(len(te)), torch.as_tensor(te.type_idx.to_numpy())].numpy()
            sign = np.where((te.Winner == "1").to_numpy(), 1.0, -1.0); w = te.confidence_weight.to_numpy(); losses.append(float((w * np.logaddexp(0.0, -sign * d)).sum() / w.sum()))
        score = float(np.mean(losses)) if losses else np.inf
        if score < best[0] - 1e-9: best = (score, (lr, steps))
    lr, steps = best[1] or ctx.params_for(labelled); head = core.fit_generic(ctx, labelled, fn, FEATURE_DIM, lr, steps, seed)
    return lambda test: core.head_d(head, fn, test)



# ------------------------------------------------------------------------------------------ Rao-Kupper tie model (principled handling of ties)
def fit_head_rk(head, nu_raw, xa, xb, typ, weight, winner, refs, bad, lr, steps, weight_decay=1e-4, anchor_weight=.25, bad_weight=.10):
    """Same optimiser and anchor terms as ``frozen.fit_head``; decisive and tie rows use the Rao-Kupper likelihood with a learnable threshold nu_k >= 0 per type:
    P(a wins) = sigmoid(d - nu), P(b wins) = sigmoid(-d - nu), P(tie) = 1 - both; not_apply keeps the repository's push-down term."""
    import torch.nn.functional as F
    head.eval(); params = list(head.parameters()) + [nu_raw]; optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    index = torch.arange(len(typ)); masks = {l: torch.as_tensor(winner == l) for l in core.frozen.LABELS}
    for _ in range(steps):
        a, b = head(xa)[index, typ], head(xb)[index, typ]; d = a - b; nu = F.softplus(nu_raw)[typ] + 1e-3
        pa, pb = torch.sigmoid(d - nu), torch.sigmoid(-d - nu); ptie = (1 - pa - pb).clamp_min(1e-6)
        terms = -torch.log(pa.clamp_min(1e-6)) * masks["1"] - torch.log(pb.clamp_min(1e-6)) * masks["2"] - torch.log(ptie) * masks["tie"] + (F.relu(a) + F.relu(b)) * masks["not_apply"]
        loss = (terms * weight).sum() / max(len(typ), 1)
        if refs and len(refs) > 1:
            scores = {n: head(f) for n, f in refs.items()}; values = []
            for p_, sp in scores.items():
                col = core.frozen.TYPE_TO_INDEX[p_]
                for o_, so in scores.items():
                    if o_ != p_: values.append(-F.logsigmoid(sp[:, col][:, None] - so[:, col][None, :]).mean())
            loss = loss + anchor_weight * torch.stack(values).mean()
        if bad is not None and len(bad): loss = loss + bad_weight * F.relu(head(bad) + 1.0).mean()
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); optimizer.step()
    return head.eval()


def rao_kupper(ctx, labelled, seed):
    lr, steps = _lr_steps(ctx, labelled); fn = core.identity_features(ctx); rows = ju_study.rows_of(ctx.exp, labelled); head = core.new_head(FEATURE_DIM, seed); refs, bad = core.anchors(ctx, fn)
    nu_raw = torch.full((5,), float(np.log(np.expm1(.5))), requires_grad=True)
    if len(rows):
        xa, xb = fn(list(rows.resolved_img1)), fn(list(rows.resolved_img2)); typ = torch.as_tensor(rows.type_idx.to_numpy())
        weight, winner = torch.as_tensor(rows.confidence_weight.to_numpy(), dtype=torch.float32), rows.Winner.to_numpy()
    else: xa = xb = torch.empty(0, FEATURE_DIM); typ = torch.empty(0, dtype=torch.long); weight = torch.empty(0); winner = np.array([], dtype=object)
    fit_head_rk(head, nu_raw, xa, xb, typ, weight, winner, refs, bad, lr, steps, ctx.exp.cfg.weight_decay, bad_weight=ctx.exp.cfg.bad_anchor_weight)
    return lambda test: core.head_d(head, fn, test)


LEARNERS = {"rao_kupper": rao_kupper, "cv_schedule": cv_schedule, "baseline": baseline, "decisive_only": decisive_only, "no_not_apply": no_not_apply, "no_tie": no_tie, "symmetric": symmetric,
            "metadata": metadata, "pseudo_label": pseudo_label}


def _anchor_variant(weight):
    def learner(ctx, labelled, seed):
        lr, steps = _lr_steps(ctx, labelled); fn = core.identity_features(ctx)
        head = core.fit_generic(ctx, labelled, fn, FEATURE_DIM, lr, steps, seed, anchor_weight=weight)
        return lambda test: core.head_d(head, fn, test)
    return learner


LEARNERS.update({"anchor_w0": _anchor_variant(0.0), "anchor_w1": _anchor_variant(1.0), "anchor_w4": _anchor_variant(4.0)})
