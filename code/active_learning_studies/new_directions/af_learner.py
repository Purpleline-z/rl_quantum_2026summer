"""Anchor-far study (branch claude/anchor-far-*): learners that create typed supervision AWAY from the existing typed anchors, on top of lap3_k10_st4.

Base: `fit_af(..., spec={})` is `nd_graph_learner.fit_mlp_graph(lap=3, steps_scale=4)` (tested to be bit-identical); every extra term draws its random numbers from a SEPARATE generator, so the
base random stream is unchanged.  A spec key switches one mechanism on:

  mix_aa    (lam)           anchor-anchor mixup: for anchors a (type c1), b (type c2): score_c1 must increase with the share of a along the path a -> b; score_c2 with the share of b
  ict_an    (lam)           interpolation consistency between a typed anchor and a random unlabelled node: s(l a + (1-l) n) = l s(a) + (1-l) s(n) (targets detached)
  ict_nn    (lam)           same with two unlabelled nodes (control: removes the typed anchor from ict_an)
  ict_near  (lam, m)        ict_an with n among the m nearest nodes of the anchor
  pseudo    (lam, hops, tau, thr)  graph-propagated typed posteriors of the references/utility images (label spreading, as nd_graph_learner.GraphData.q) as soft pseudo-anchors: nodes within
                            `hops` graph hops of a typed anchor, class = argmax q, weight = max q * exp(-hop / tau) (>= thr); trained with the anchor ranking loss against the real anchors of the
                            other types and the pseudo-anchors of the other types.  Targets come only from the references via the graph (not from the model's own predictions).
  nbr_a     (lam, m, tau)   anchor-neighbourhood smoothing: each real anchor node is tied to its m nearest nodes with weights exp(-rank / tau)
  dlam      (beta, sign)    distance-dependent lambda: edge weight 1 + beta * min(d_mid / 0.03, 3) (sign=+1: stronger far from anchors; -1: 1 / (1 + ...), stronger near), normalised to mean 1
  far_edges (lam, k, hops)  multi-scale graph: extra kNN-k edges among nodes at least `hops` graph hops from every anchor
  runcentre ()              per-run feature centring: x - mean_run(x) + mean_all(x) for trajectory frames (transductive, features only)
Controls: shuffle=True permutes node identities of the graph (and with it hops, q and every graph-based edge set); `perm_anchor_classes=True` permutes the anchor classes inside mix_aa.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

import nd_core as core
import nd_graph_learner as G
import rt_core as rt
from nd_graph_learner import ACTIVE, EDGE_BATCH, ju_study, _bt_terms, _anchor_loss

_SIM: dict = {}


def sim() -> rt.Sim:
    if "s" not in _SIM: _SIM["s"] = rt.Sim(core.feature_cache())
    return _SIM["s"]


class Geo:
    """Distances of the graph nodes to the typed anchors (feature distance and graph hops), cached per (ctx, graph)."""
    def __init__(self, ctx, gd):
        s = sim(); anchors = sorted({str(p) for t in (ctx.exp.references, ctx.exp.utility_images, ctx.exp.test_images) for ps in t.values() for p in ps})
        node = torch.as_tensor([s.index[p] for p in gd.paths]); anc = torch.as_tensor([s.index[p] for p in anchors]); self.d_anchor = (1 - (s.xn[node] @ s.xn[anc].T).max(1).values).numpy()
        n = len(gd.paths); self.hop = np.full(n, 1e9); src = sorted(gd.source_class); self.hop[src] = 0; adj = gd.adj.numpy(); frontier = np.zeros(n, bool); frontier[src] = True; h = 0
        while frontier.any():
            h += 1; nxt = adj[frontier].any(0) & (self.hop > 1e8); self.hop[nxt] = h; frontier = nxt
        self.src = np.array(src, dtype=int); self.n = n


def geo_for(ctx, gd) -> Geo:
    cache = ctx.__dict__.setdefault("_af_geo", {})
    if id(gd) not in cache: cache[id(gd)] = Geo(ctx, gd)
    return cache[id(gd)]


def _edge_sets(ctx, gd, geo: Geo, spec: dict):
    """Extra weighted edge sets (list of (edges[long, 2], weights[float], lam))."""
    sets = []; rng = np.random.default_rng(ctx.seed * 131 + 7)
    if "nbr_a" in spec:
        lam, m, tau = spec["nbr_a"]; sim_nodes = gd.xn @ gd.xn.T; sim_nodes.fill_diagonal_(-2.0); nb = sim_nodes.topk(m, dim=1).indices; rows, cols, w = [], [], []
        for a in geo.src:
            for r in range(m): rows.append(int(a)); cols.append(int(nb[a, r])); w.append(float(np.exp(-r / tau)))
        e = torch.as_tensor(np.stack([rows, cols], 1)); cols_t = e[:, 1]
        if gd.shuffled: e = torch.stack([e[:, 0], torch.as_tensor(rng.integers(0, geo.n, len(e)))], 1)    # control: random partners (same anchors, same weights)
        sets.append((e, torch.as_tensor(w, dtype=torch.float32), lam))
    if "far_edges" in spec:
        lam, k, hops = spec["far_edges"]; sim_nodes = gd.xn @ gd.xn.T; sim_nodes.fill_diagonal_(-2.0); nb = sim_nodes.topk(k, dim=1).indices; far = geo.hop >= hops; rows, cols = [], []
        for i in np.flatnonzero(far):
            for j in nb[i].tolist():
                if far[j] and i < j: rows.append(int(i)); cols.append(int(j))
        if rows:
            e = torch.as_tensor(np.stack([rows, cols], 1))
            if gd.shuffled: e = torch.as_tensor(np.stack([rng.permutation(e[:, 0].numpy()), rng.permutation(e[:, 1].numpy())], 1))
            sets.append((e, torch.ones(len(e)), lam))
    return sets


def _pseudo(gd, geo: Geo, spec: dict):
    lam, hops, tau, thr = spec["pseudo"]; q = gd.q().numpy(); cls = q.argmax(1); conf = q.max(1) * np.exp(-geo.hop / tau); ok = (geo.hop >= 1) & (geo.hop <= hops) & (conf >= thr)
    idx = np.flatnonzero(ok); return lam, idx, cls[idx], conf[idx].astype(np.float32)


def _run_mean_shift(ctx, gd, cache):
    """mean_all - mean_run for every run code (frames only); images without a run code are not shifted."""
    runs = {}
    for i, p in enumerate(gd.paths):
        r = rt.run_of(p)
        if r is not None: runs.setdefault(r, []).append(i)
    allm = gd.x[[i for ids in runs.values() for i in ids]].mean(0)
    return {r: (allm - gd.x[ids].mean(0)) for r, ids in runs.items()}


def fit_af(ctx, labelled, seed, gd, featfn, node_x, spec: dict, lap: float = 3.0, steps_scale: float = 4.0, aw: float = 8.0):
    lr, steps = ctx.params_for(labelled); steps = int(round(steps * steps_scale)); rows = ju_study.rows_of(ctx.exp, labelled); dim = node_x.shape[1]; head = core.new_head(dim, seed); refs, bad = core.anchors(ctx, featfn, True)
    if len(rows):
        xa, xb = featfn(list(rows.resolved_img1)), featfn(list(rows.resolved_img2)); typ = torch.as_tensor(rows.type_idx.to_numpy())
        weight, winner = torch.as_tensor(rows.confidence_weight.to_numpy(), dtype=torch.float32), rows.Winner.to_numpy()
    else: xa = xb = torch.empty(0, dim); typ = torch.empty(0, dtype=torch.long); weight = torch.empty(0); winner = np.array([], dtype=object)
    head.eval(); opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=ctx.exp.cfg.weight_decay); index = torch.arange(len(typ)); masks = {l: torch.as_tensor(winner == l) for l in core.frozen.LABELS}
    cols = {c: core.frozen.TYPE_TO_INDEX[c] for c in refs}; gen = torch.Generator().manual_seed(seed * 1000003 + 17); gen2 = torch.Generator().manual_seed(seed * 7919 + 29); active = list(ACTIVE)
    geo = geo_for(ctx, gd) if any(k in spec for k in ("pseudo", "nbr_a", "far_edges", "dlam", "ict_near")) else None
    edge_w = None
    if "dlam" in spec and geo is not None:
        beta, sign = spec["dlam"]; dmid = torch.as_tensor((geo.d_anchor[gd.edges[:, 0].numpy()] + geo.d_anchor[gd.edges[:, 1].numpy()]) / 2, dtype=torch.float32); z = torch.clamp(dmid / 0.03, max=3.0)
        edge_w = (1 + beta * z) if sign > 0 else 1 / (1 + beta * z); edge_w = edge_w / edge_w.mean()
    extra = _edge_sets(ctx, gd, geo, spec) if geo is not None else []
    pseudo = _pseudo(gd, geo, spec) if "pseudo" in spec else None
    anc_x = {c: f for c, f in refs.items()}; anc_all = torch.cat([f for f in refs.values()]); anc_cls = torch.cat([torch.full((len(f),), cols[c]) for c, f in refs.items()])
    anc_cls_mix = anc_cls[torch.as_tensor(np.random.default_rng(seed + 3).permutation(len(anc_cls)))] if spec.get("perm_anchor_classes") else anc_cls
    near_nb = None
    if "ict_near" in spec:
        m = spec["ict_near"][1]; sn = gd.xn @ gd.xn.T; sn.fill_diagonal_(-2.0); near_nb = sn.topk(m, dim=1).indices
    for _ in range(steps):
        loss = _bt_terms(head(xa)[index, typ], head(xb)[index, typ], masks, weight) / max(len(typ), 1)
        if len(refs) > 1: loss = loss + aw * _anchor_loss({n: head(f) for n, f in refs.items()}, cols, None)
        if bad is not None and len(bad): loss = loss + ctx.exp.cfg.bad_anchor_weight * F.relu(head(bad) + 1.0).mean()
        if lap > 0:
            sel = torch.randint(gd.n_edges, (EDGE_BATCH,), generator=gen); e = gd.edges[sel]; si, sj = head(node_x[e[:, 0]])[:, active], head(node_x[e[:, 1]])[:, active]; d2 = ((si - sj) ** 2).sum(1)
            loss = loss + lap * ((d2 * edge_w[sel]).mean() if edge_w is not None else d2.mean())
        for e_set, w_set, lam_e in extra:
            sel = torch.randint(len(e_set), (128,), generator=gen2); e = e_set[sel]; d2 = ((head(node_x[e[:, 0]])[:, active] - head(node_x[e[:, 1]])[:, active]) ** 2).sum(1); loss = loss + lam_e * (d2 * w_set[sel]).sum() / w_set[sel].sum()
        if "mix_aa" in spec:
            B = 64; i = torch.randint(len(anc_all), (B,), generator=gen2); j = torch.randint(len(anc_all), (B,), generator=gen2); ok = anc_cls_mix[i] != anc_cls_mix[j]
            if ok.any():
                a, b = anc_all[i[ok]], anc_all[j[ok]]; c1, c2 = anc_cls_mix[i[ok]], anc_cls_mix[j[ok]]; hi, lo = 0.8 * a + 0.2 * b, 0.3 * a + 0.7 * b; sh, sl = head(hi), head(lo); r = torch.arange(len(c1))
                loss = loss + spec["mix_aa"] * (-F.logsigmoid(sh[r, c1] - sl[r, c1]) - F.logsigmoid(sl[r, c2] - sh[r, c2])).mean()
        for key in ("ict_an", "ict_nn", "ict_near"):
            if key in spec:
                B = 64; lam_i = spec[key] if key != "ict_near" else spec[key][0]; li = 0.5 + 0.5 * torch.rand(B, 1, generator=gen2)
                if key == "ict_nn": a_f = node_x[torch.randint(len(node_x), (B,), generator=gen2)]
                elif key == "ict_an": a_f = anc_all[torch.randint(len(anc_all), (B,), generator=gen2)]
                else:
                    ai = torch.as_tensor(rng_choice(geo.src, B, gen2)); a_f = node_x[ai]
                if key == "ict_near": n_f = node_x[near_nb[ai, torch.randint(near_nb.shape[1], (B,), generator=gen2)]]
                else: n_f = node_x[torch.randint(len(node_x), (B,), generator=gen2)]
                z = li * a_f + (1 - li) * n_f
                with torch.no_grad(): t = li * head(a_f)[:, active] + (1 - li) * head(n_f)[:, active]
                loss = loss + lam_i * ((head(z)[:, active] - t) ** 2).sum(1).mean()
        if pseudo is not None:
            lam_p, idx, pc, pw = pseudo
            if len(idx):
                sp = head(node_x[torch.as_tensor(idx)]); wp = torch.as_tensor(pw); pcl = torch.as_tensor([cols_idx(c) for c in pc]); terms = []
                for c in active:
                    mine = (pcl == c)
                    if not mine.any(): continue
                    real_o = torch.cat([head(f)[:, c] for n_, f in refs.items() if cols[n_] != c]); other = pcl != c; sc = sp[mine][:, c]; wm = wp[mine]
                    l_real = -F.logsigmoid(sc[:, None] - real_o[None, :]).mean(1); terms.append((wm * l_real).sum() / wm.sum())
                    if other.any(): l_ps = -F.logsigmoid(sc[:, None] - sp[other][:, c][None, :]); w2 = wm[:, None] * wp[other][None, :]; terms.append((w2 * l_ps).sum() / w2.sum())
                if terms: loss = loss + lam_p * torch.stack(terms).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
    return head.eval()


def cols_idx(class_position: int) -> int:
    """q columns follow ACTIVE = (0, 2, 3, 4): map the argmax position to the head column."""
    return ACTIVE[int(class_position)]


def rng_choice(arr, n, gen):
    return arr[torch.randint(len(arr), (n,), generator=gen).numpy()]


def af_learner(spec: dict, k: int = 10, shuffle: bool = False, lap: float = 3.0, steps_scale: float = 4.0):
    def learner(ctx, labelled, seed):
        gd = G.graph_for(ctx, k, shuffle); cache = ctx.features.cache; base_feat = core.identity_features(ctx); node_x = gd.x; featfn = base_feat
        if "runcentre" in spec:
            shift = _run_mean_shift(ctx, gd, cache)
            def featfn(paths):
                x = base_feat(paths).clone()
                for i, p in enumerate(paths):
                    r = rt.run_of(p)
                    if r in shift: x[i] = x[i] + shift[r]
                return x
            node_x = featfn(gd.paths)
        head = fit_af(ctx, labelled, seed, gd, featfn, node_x, spec, lap=lap, steps_scale=steps_scale)
        return lambda test: core.head_d(head, featfn, test)
    return learner


LEARNERS: dict = {}


def _reg(name, spec, **kw):
    LEARNERS[name] = af_learner(spec, **kw)
    if kw.get("shuffle") is None: LEARNERS[name + "_shuf"] = af_learner(spec, shuffle=True)


LEARNERS["af_base"] = af_learner({})
# V1 mixup / interpolation consistency
for _l in (0.3, 1.0): LEARNERS[f"af_mixaa{_l:g}"] = af_learner({"mix_aa": _l}); LEARNERS[f"af_mixaa{_l:g}_perm"] = af_learner({"mix_aa": _l, "perm_anchor_classes": True})
for _l in (0.03, 0.1, 0.3, 1.0, 3.0): LEARNERS[f"af_ictan{_l:g}"] = af_learner({"ict_an": _l}); LEARNERS[f"af_ictnn{_l:g}"] = af_learner({"ict_nn": _l})
for _l in (0.1, 1.0): LEARNERS[f"af_ictnear{_l:g}"] = af_learner({"ict_near": (_l, 30)})
# V2 pseudo anchors
for _l in (0.3, 1.0, 3.0):
    for _h in (2, 4):
        LEARNERS[f"af_pseudo{_l:g}h{_h}"] = af_learner({"pseudo": (_l, _h, 2.0, 0.4)}); LEARNERS[f"af_pseudo{_l:g}h{_h}_shuf"] = af_learner({"pseudo": (_l, _h, 2.0, 0.4)}, shuffle=True)
# V3 anchor neighbourhood
for _l in (1.0, 3.0, 10.0): LEARNERS[f"af_nbra{_l:g}"] = af_learner({"nbr_a": (_l, 30, 10.0)}); LEARNERS[f"af_nbra{_l:g}_shuf"] = af_learner({"nbr_a": (_l, 30, 10.0)}, shuffle=True)
# V4 distance-dependent lambda
for _b in (2.0, 5.0): LEARNERS[f"af_dlamfar{_b:g}"] = af_learner({"dlam": (_b, +1)}); LEARNERS[f"af_dlamfar{_b:g}_shuf"] = af_learner({"dlam": (_b, +1)}, shuffle=True); LEARNERS[f"af_dlamnear{_b:g}"] = af_learner({"dlam": (_b, -1)})
# V5 per-run centring
LEARNERS["af_runcentre"] = af_learner({"runcentre": 1})
# V6 multi-scale
for _l in (1.0, 3.0): LEARNERS[f"af_far{_l:g}"] = af_learner({"far_edges": (_l, 30, 3)}); LEARNERS[f"af_far{_l:g}_shuf"] = af_learner({"far_edges": (_l, 30, 3)}, shuffle=True)
