"""HTR failure analysis, Stage 2: targeted learner variants on top of the incumbent ``lap3_k10_st4`` (``nd_graph_learner``).

Diagnosis (see ``HTR_FAILURES_RESULTS.md``): the decisive HTR labels are almost perfectly consistent with a graded 'HTR-ness' = similarity to the typed HTR ideal images (leave-one-out label agreement of near-identical
comparisons 44/45; the label-free rule 'the frame nearer to the ideal HTR images wins' gets 44/45), but the incumbent head, trained on a handful of pairwise labels plus a binary HTR-vs-rest anchor loss, orders frames that
are NOT near an HTR ideal image (1x1-like vs RT13-like frames) by a poorly determined direction.  The variants therefore inject the anchor-similarity score  r(x) = mean of the top-k cosine similarities
(centred, normalised SimCLR features; the graph's centring) between x and the HTR anchors  into the HTR output only, so no other type can change:

  hfb_g<G>     post-hoc blend      d' = d + G * (r(a) - r(b))  for HTR rows (k = 4); top-1 similarity          (G in 5, 10, 20)
  hfb3_g10     same with the mean of the top-3 similarities (smoother, less dependent on a single near-copy anchor)
  hfs          anchor-similarity features: the head sees [x, 10 * s(x)] with s(x) = top-1 cosine similarity to each class's anchors (self excluded for anchors), incl. the graph nodes of the Laplacian term
  hfc1         cross-type coupling d_HTR - d_RT13 (31/31 pairs with both judgments have opposite winners); hfcb10 = coupling + blend G=10
  hfq1         teacher loss: lambda * softplus(-sign(r_i - r_j) * (s_HTR(i) - s_HTR(j))) on random node pairs (HTR output only; the repository's ``rankq`` path with a q that has a constant column for the other types)

Anchors are read from ``ctx.exp`` (references + utility + outer-test ideal images of the seed, exactly the anchors of ``aw8_all``), so an anchor-strict context (near-copy anchors removed) changes the variants' anchors too.
The identical image is excluded when it is itself an anchor.
"""
from __future__ import annotations

import numpy as np
import torch

import nd_core as core
import nd_graph_learner as G
import rt_core as rt

HTR = "HTR"; ACTIVE_CLASSES = ("(1 x 1)", "c(6 x 2)", "(√13 x √13)", "HTR")      # ctx.exp table keys, in the column order of ``G.ACTIVE`` = (0, 2, 3, 4)


def anchor_paths(ctx, cls: str) -> list[str]:
    ex = ctx.exp; return sorted({str(p) for p in list(ex.references.get(cls, [])) + list(ex.utility_images.get(cls, [])) + list(ex.test_images.get(cls, []))})


def similarity(ctx, paths, cls: str, topk: int = 1) -> torch.Tensor:
    """Mean of the top-k cosine similarities (graph centring) of every path to the class's anchors, the identical image excluded."""
    gd = G.graph_for(ctx, 10, False); key = ("_hf_anchor", cls); anchors = anchor_paths(ctx, cls)
    if key not in ctx.__dict__ or ctx.__dict__[key][0] != anchors:
        x = ctx.features.get(anchors).float() - gd.mu; ctx.__dict__[key] = (anchors, x / x.norm(dim=1, keepdim=True).clamp_min(1e-9))
    anchors, xa = ctx.__dict__[key]; q = ctx.features.get([str(p) for p in paths]).float() - gd.mu; q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-9); s = q @ xa.T
    index = {p: i for i, p in enumerate(anchors)}
    for row, p in enumerate(paths):
        if str(p) in index: s[row, index[str(p)]] = -2.0
    return s.topk(min(topk, s.shape[1]), dim=1).values.mean(1)


def blend_learner(gamma: float, topk: int = 1, base: str = "lap3_k10_st4"):
    def learner(ctx, labelled, seed):
        fn = _base_fn(ctx, labelled, seed, base)   # the variants of one (label set, seed) share one fit of the base learner (deterministic, identical to refitting)
        def predict(test):
            d = np.asarray(fn(test), dtype=float); htr = test.k == 4
            gap = (similarity(ctx, test.img1, HTR, topk) - similarity(ctx, test.img2, HTR, topk)).numpy()
            return d + gamma * gap * htr
        return predict
    return learner


def _base_fn(ctx, labelled, seed, base):
    key = (tuple(labelled), seed, base); last = ctx.__dict__.get("_hf_base")
    if last is None or last[0] != key: ctx.__dict__["_hf_base"] = last = (key, G.LEARNERS[base](ctx, labelled, seed))
    return last[1]


def coupled_learner(lam: float = 1.0, gamma: float = 0.0, base: str = "lap3_k10_st4"):
    """Cross-type coupling: in the 31 pairs with a decisive HTR and a decisive RT13 judgment the winners are opposite in 31 of 31 (HTR-ness and RT13-ness are one axis there), so the HTR logit gap is replaced by
    d_HTR - lam * d_RT13 (the RT13 output of the same head on the same two images); optional additional anchor-similarity blend ``gamma`` as in ``hfb``."""
    import copy
    def learner(ctx, labelled, seed):
        fn = _base_fn(ctx, labelled, seed, base)
        def predict(test):
            d = np.asarray(fn(test), dtype=float); t3 = copy.copy(test); t3.k = np.full_like(test.k, 3); d3 = np.asarray(fn(t3), dtype=float); out = np.where(test.k == 4, d - lam * d3, d)
            if gamma: out = out + gamma * (similarity(ctx, test.img1, HTR, 1) - similarity(ctx, test.img2, HTR, 1)).numpy() * (test.k == 4)
            return out
        return predict
    return learner


def _class_sims(ctx, paths) -> torch.Tensor:
    return torch.stack([similarity(ctx, paths, c, 1) for c in ACTIVE_CLASSES], 1)


def aug_learner(scale: float = 10.0, lap: float = 3.0, steps_scale: float = 4.0, k: int = 10):
    def learner(ctx, labelled, seed):
        gd = G.graph_for(ctx, k, False); base = core.identity_features(ctx)
        featfn = lambda paths: torch.cat([base(paths).float(), scale * _class_sims(ctx, [str(p) for p in paths])], 1)
        node_x = torch.cat([gd.x, scale * _class_sims(ctx, gd.paths)], 1)
        head = G.fit_mlp_graph(ctx, labelled, seed, gd, featfn, node_x, node_x.shape[1], lap=lap, steps_scale=steps_scale)
        return lambda test: core.head_d(head, featfn, test)
    return learner


def teacher_learner(lam: float = 1.0, lap: float = 3.0, steps_scale: float = 4.0, k: int = 10):
    def learner(ctx, labelled, seed):
        gd = G.graph_for(ctx, k, False); featfn = core.identity_features(ctx)
        r = similarity(ctx, gd.paths, HTR, 1); q = torch.zeros(len(gd.paths), len(G.ACTIVE)); q[:, G.ACTIVE.index(4)] = r      # constant (zero) columns for the other types -> no gradient there
        head = G.fit_mlp_graph(ctx, labelled, seed, gd, featfn, gd.x, gd.x.shape[1], lap=lap, rankq=lam, steps_scale=steps_scale, q=q)
        return lambda test: core.head_d(head, featfn, test)
    return learner


LEARNERS_HF = {"hfb_g5": blend_learner(5.0), "hfb_g10": blend_learner(10.0), "hfb_g20": blend_learner(20.0), "hfb3_g10": blend_learner(10.0, topk=3), "hfs": aug_learner(), "hfq1": teacher_learner(1.0), "hfc1": coupled_learner(1.0), "hfcb10": coupled_learner(1.0, 10.0)}
rt.REGISTRY.update(LEARNERS_HF)
