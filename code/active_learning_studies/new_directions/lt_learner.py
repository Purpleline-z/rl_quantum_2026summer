"""Latent-type learners: ``nd_graph_learner.fit_mlp_graph`` (``lap3_k10_st4``) plus posteriors of the latent-type model (``lt_model``) used in three ways:

  pa<w>     soft pseudo-anchors: every image of the labelled judgments enters the anchor ranking loss with soft class memberships (posterior over the four types) against the hard typed anchors, weight w * aw
  lik       answerability-aware loss weighting: a not_apply row pushes down the score of an image with weight P(image not of the row's type) (instead of pushing both down), a decisive row is weighted by
            P(at least one image of the row's type); tie rows unchanged
  aux<w>    auxiliary target: cross-entropy of the softmax over the four type columns towards the posterior of the judgment images (hard one-hots for the typed anchors), weight w
Posterior sources (``post``): 'em' latent-class EM (prior = label spreading on the real graph), 'prior' = graph prior only, no outcome likelihood (the earlier self-training in disguise), 'random' = EM
posteriors permuted among the images, 'shufprior' = EM with the prior built on the shuffled graph.  With every term off the fit is exactly ``lap3_k10_st4`` (unit-tested).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

import judgment_unit_study as ju_study
import nd_core as core
import nd_graph_learner as G
import lt_model as M
from nd_graph_learner import _bt_terms, _anchor_loss, ACTIVE, EDGE_BATCH


def posteriors(ctx, labelled, seed, post: str = "em", rho: float = 0.15, scale: float = 1.0):
    """(Judgments, belief (n, 5), pair beliefs, table) for the labelled judgments of one fit."""
    key = ("lt_post", post, rho, scale, tuple(labelled), seed if post == "random" else 0); cached = ctx.__dict__.setdefault("_lt_cache", {})
    if key in cached: return cached[key]
    rows = ju_study.rows_of(ctx.exp, labelled); jud = M.Judgments(rows); shuf = post == "shufprior"; gd = G.graph_for(ctx, 10, shuf); cache = ctx.features.cache
    prior = M.prior_from_q(M.node_prior(gd, jud.images, cache, gd.q().numpy()), rho=rho)
    mode = {"em": "em", "shufprior": "em", "prior": "prior", "random": "random"}[post]
    belief, table, pair = M.posterior_of_images(jud, prior, mode=mode, prior_scale=scale, rng=np.random.default_rng(seed * 13 + 1)); cached[key] = (jud, belief, pair, table); return cached[key]


def fit_lt(ctx, labelled, seed, gd, featfn, node_x, dim, lap=3.0, steps_scale=4.0, aw=8.0, pa=0.0, lik=False, aux=0.0, post="em", rho=0.15, scale=1.0, pa_thr=0.0, pa_far=0.0):
    lr, steps = ctx.params_for(labelled); steps = int(round(steps * steps_scale)); rows = ju_study.rows_of(ctx.exp, labelled); head = core.new_head(dim, seed); refs, bad = core.anchors(ctx, featfn, True)
    if len(rows):
        xa, xb = featfn(list(rows.resolved_img1)), featfn(list(rows.resolved_img2)); typ = torch.as_tensor(rows.type_idx.to_numpy())
        weight, winner = torch.as_tensor(rows.confidence_weight.to_numpy(), dtype=torch.float32), rows.Winner.to_numpy()
    else: xa = xb = torch.empty(0, dim); typ = torch.empty(0, dtype=torch.long); weight = torch.empty(0); winner = np.array([], dtype=object)
    use_post = (pa > 0 or lik or aux > 0) and len(rows) > 0
    if use_post:
        jud, belief, pair, table = posteriors(ctx, labelled, seed, post, rho, scale); rp = torch.as_tensor(M.row_probabilities(jud, pair, belief), dtype=torch.float32)
        px = featfn(jud.images); pb = torch.as_tensor(belief, dtype=torch.float32)[:, :4]                                   # soft type memberships of the judgment images (mass on "none" dropped)
        conf = pb.sum(1)                                                                                                    # probability that the image is of one of the four types
        keep = torch.ones(len(pb))
        if pa_thr > 0: keep = keep * ((pb / pb.sum(1, keepdim=True).clamp_min(1e-9)).max(1).values >= pa_thr).float()      # pseudo-anchors only where the posterior is confident
        if pa_far > 0:                                                                                                      # ... and only for images farther than pa_far (cosine distance) from every typed ideal image
            anc = gd.xn[sorted(gd.source_class)]; ix = [gd.index[p] for p in jud.images if p in gd.index]; dist = torch.ones(len(pb))
            if ix: dist[[i for i, p in enumerate(jud.images) if p in gd.index]] = 1 - (gd.xn[ix] @ anc.T).max(1).values
            keep = keep * (dist > pa_far).float()
        pb = pb * keep[:, None]; conf = conf * keep
    head.eval(); opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=ctx.exp.cfg.weight_decay); index = torch.arange(len(typ)); masks = {l: torch.as_tensor(winner == l) for l in core.frozen.LABELS}
    cols = {c: core.frozen.TYPE_TO_INDEX[c] for c in refs}; gen = torch.Generator().manual_seed(seed * 1000003 + 17); active = list(ACTIVE); names = list(refs)
    if use_post and lik:
        rm = {l: torch.as_tensor(winner == l) for l in core.frozen.LABELS}; pna, pnb, pboth0 = rp[:, 0], rp[:, 1], rp[:, 2]; dec = (rm["1"] | rm["2"]).float()
    for _ in range(steps):
        a, b = head(xa)[index, typ], head(xb)[index, typ]
        if use_post and lik:
            terms = (-F.logsigmoid(a - b) * masks["1"] - F.logsigmoid(b - a) * masks["2"]) * (1 - pboth0) + (a - b).abs() * masks["tie"] + (F.relu(a) * pna + F.relu(b) * pnb) * masks["not_apply"]
            loss = (terms * weight).sum() / max(len(typ), 1)
        else: loss = _bt_terms(a, b, masks, weight) / max(len(typ), 1)
        if len(refs) > 1:
            sc = {n: head(f) for n, f in refs.items()}; loss = loss + aw * _anchor_loss(sc, cols, None)
            if use_post and pa > 0:
                sp = head(px); vals = []
                for ci, name in enumerate(names):
                    col = cols[name]; u = pb[:, ci]; v = keep * (1 - u)
                    neg = torch.cat([sc[o][:, col] for o in names if o != name]); pos = sc[name][:, col]
                    if u.sum() > 1e-6: vals.append((u[:, None] * -F.logsigmoid(sp[:, col][:, None] - neg[None, :])).sum() / (u.sum() * len(neg)))
                    vals.append((v[:, None] * -F.logsigmoid(pos[None, :] - sp[:, col][:, None])).sum() / (v.sum().clamp_min(1e-6) * len(pos)))
                loss = loss + aw * pa * torch.stack(vals).mean()
        if bad is not None and len(bad): loss = loss + ctx.exp.cfg.bad_anchor_weight * F.relu(head(bad) + 1.0).mean()
        if lap > 0 and gd is not None:
            e = gd.edges[torch.randint(gd.n_edges, (EDGE_BATCH,), generator=gen)]; si, sj = head(node_x[e[:, 0]])[:, active], head(node_x[e[:, 1]])[:, active]
            loss = loss + lap * (((si - sj) ** 2)).sum(1).mean()
        if use_post and aux > 0:
            lp = F.log_softmax(head(px)[:, active], 1); tgt = pb / pb.sum(1, keepdim=True).clamp_min(1e-9); ce = -(tgt * lp).sum(1); loss = loss + aux * (conf * ce).sum() / conf.sum().clamp_min(1e-6)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
    return head.eval()


def _lt_learner(**kw):
    def learner(ctx, labelled, seed):
        gd = G.graph_for(ctx, 10); featfn = core.identity_features(ctx); head = fit_lt(ctx, labelled, seed, gd, featfn, gd.x, gd.x.shape[1], **kw)
        return lambda test: core.head_d(head, featfn, test)
    return learner


LT_LEARNERS: dict = {"lt_off": _lt_learner()}
for _w in (0.25, 1.0):
    for _p in ("em", "prior", "random", "shufprior"): LT_LEARNERS[f"lt_pa{_w:g}_{_p}"] = _lt_learner(pa=_w, post=_p)
for _p in ("em", "prior", "random", "shufprior"): LT_LEARNERS[f"lt_lik_{_p}"] = _lt_learner(lik=True, post=_p)
for _w in (0.1, 0.5):
    for _p in ("em", "prior", "random"): LT_LEARNERS[f"lt_aux{_w:g}_{_p}"] = _lt_learner(aux=_w, post=_p)
for _p in ("em", "prior"):
    LT_LEARNERS[f"lt_pa0.5thr0.8_{_p}"] = _lt_learner(pa=0.5, pa_thr=0.8, post=_p); LT_LEARNERS[f"lt_pa0.5far_{_p}"] = _lt_learner(pa=0.5, pa_far=0.01, post=_p); LT_LEARNERS[f"lt_pa0.5far_thr0.8_{_p}"] = _lt_learner(pa=0.5, pa_far=0.01, pa_thr=0.8, post=_p)
LT_LEARNERS["lt_pa1_lik_em"] = _lt_learner(pa=1.0, lik=True, post="em")
for _p in ("em", "prior"): LT_LEARNERS[f"lt_aux1_{_p}"] = _lt_learner(aux=1.0, post=_p); LT_LEARNERS[f"lt_aux0.5_lik_{_p}"] = _lt_learner(aux=0.5, lik=True, post=_p)
LT_LEARNERS["lt_aux0.1_em"] = LT_LEARNERS["lt_aux0.1_em"]


# ------------------------------------------------------------------------------------------ inference-time use of the latent-type model ("extra input" realised without retraining)
def _memo_head(ctx, labelled, seed):
    cached = ctx.__dict__.setdefault("_lt_heads", {}); key = (tuple(labelled), seed)
    if key not in cached:
        gd = G.graph_for(ctx, 10); featfn = core.identity_features(ctx); cached[key] = fit_lt(ctx, labelled, seed, gd, featfn, gd.x, gd.x.shape[1])
    return cached[key]


def _inf_learner(lam: float, post: str = "em", rho: float = 0.15):
    """d' = d + lam * log P(a is the decisive winner) / P(b is) under the confusion table T (EM-learned on the labelled judgments) and the graph-prior memberships of the two test images (attached to their k nearest
    graph nodes).  post: em (table from EM), prior (table = the verbal pseudo-count table, no learning), shufprior (prior from the shuffled graph), random (test priors permuted among the test images)."""
    def learner(ctx, labelled, seed):
        head = _memo_head(ctx, labelled, seed); featfn = core.identity_features(ctx)
        _, _, _, table = posteriors(ctx, labelled, seed, "em" if post in ("em", "random") else ("shufprior" if post == "shufprior" else "prior"), rho)
        if post == "prior": table = M.init_table()
        def predict(test):
            d = core.head_d(head, featfn, test); gd = G.graph_for(ctx, 10, post == "shufprior"); cache = ctx.features.cache
            pa = M.prior_from_q(M.node_prior(gd, [str(p) for p in test.img1], cache, gd.q().numpy()), rho=rho); pb = M.prior_from_q(M.node_prior(gd, [str(p) for p in test.img2], cache, gd.q().numpy()), rho=rho)
            if post == "random":
                rng = np.random.default_rng(seed * 17 + 3); pa, pb = pa[rng.permutation(len(pa))], pb[rng.permutation(len(pb))]
            sa = pa[np.arange(len(pa)), test.k.copy() * 0 + np.array([M.COL[int(k)] for k in test.k])]; sb = pb[np.arange(len(pb)), np.array([M.COL[int(k)] for k in test.k])]
            wa = sum(x * y * table[i, j, 0] for i, x in ((1, sa), (0, 1 - sa)) for j, y in ((1, sb), (0, 1 - sb))); wb = sum(x * y * table[i, j, 1] for i, x in ((1, sa), (0, 1 - sa)) for j, y in ((1, sb), (0, 1 - sb)))
            return d + lam * np.log(np.clip(wa, 1e-9, None) / np.clip(wb, 1e-9, None))
        return predict
    return learner


for _l in (0.5, 1.0, 2.0):
    for _p in ("em", "prior", "shufprior", "random"): LT_LEARNERS[f"lt_inf{_l:g}_{_p}"] = _inf_learner(_l, _p)
