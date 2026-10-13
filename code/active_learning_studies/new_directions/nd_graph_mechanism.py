"""Graph-mechanism learners: is the gain of the Laplacian-regularised head (``lap3_k10`` of nd_graph_learner.py) due to the QUALITY of the graph or to ANY smoothing?

All learners equal ``aw8_all`` (frozen SimCLR features, the repository's loss with anchor weight 8 and all ideal images as anchors, per-budget head schedule) plus one regulariser on the
scores of unlabelled nodes.  Graph regularisers vary the graph (construction, features, noise dose, edge placement, per-type weight, normalisation, annealing); the non-graph controls
smooth without any graph (Gaussian input noise matched to the kNN distance, mixup consistency on random node pairs, weight-decay sweep).

Identity safety is inherited from ``nd_graph_learner.GraphData`` (nodes = trajectory + ideal images minus the seed's validation/test pair-group images and outer-test ideal images, by SHA-256).
Topology features other than the final 512-d feature (layers 2/3 of the same frozen encoder; ``nd_gm_layers.py build``) are label-free and use the same node set.

Every graph also records its quality = (weighted) share of same-class edges among ideal-ideal edges (both ideal images have known class labels; chance ~0.25), written to
``results/gm/quality/seed<S>_<split>.json`` (graph name -> {quality, n_ii_edges, n_edges, mean_degree}), so dose-response claims can use it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import nd_core as core
import nd_graph_learner as G
import nd_gm_layers

ACTIVE = G.ACTIVE; QUALITY_DIR = core.HERE / "results" / "gm" / "quality"
_LAYERS: dict = {}


def _is_ideal(path: str) -> bool:
    return "STO_ideal" in path


# ------------------------------------------------------------------------------------------ topology features
def topology_features(base: G.GraphData, feat: str, whiten: int, noise: float) -> torch.Tensor:
    """Unit-norm, mean-centred features used ONLY to decide which nodes are neighbours."""
    if feat == "final": x = base.x
    else:
        if not _LAYERS: _LAYERS.update(nd_gm_layers.load())
        if feat in ("l2", "l3", "l4"): x = torch.stack([_LAYERS[feat][p] for p in base.paths]).float()
        elif feat == "l3l4": x = torch.cat([F.normalize(torch.stack([_LAYERS[l][p] for p in base.paths]).float(), dim=1) for l in ("l3", "l4")], dim=1)
        else: raise ValueError(feat)
    xc = x - x.mean(0)
    if whiten:   # PCA whitening on the node set (label-free): top `whiten` components scaled to unit variance
        u, s, v = torch.pca_lowrank(xc, q=whiten, center=False, niter=4); xc = (xc @ v[:, :whiten]) / (s[:whiten] / (len(xc) ** 0.5)).clamp_min(1e-6)
    xn = xc / xc.norm(dim=1, keepdim=True).clamp_min(1e-9)
    if noise > 0:
        gen = torch.Generator().manual_seed(12345); xn = xn + noise * torch.randn(xn.shape, generator=gen) / xn.shape[1] ** 0.5
    return xn


def _upper(rows: torch.Tensor, cols: torch.Tensor, w: torch.Tensor | None):
    """Undirected edge list (i < j), duplicates removed."""
    a, b = torch.minimum(rows, cols), torch.maximum(rows, cols); keep = a != b; a, b = a[keep], b[keep]
    n = int(max(a.max(), b.max())) + 1; key = a * n + b; uniq, inv = torch.unique(key, return_inverse=True)
    ia, ib = uniq // n, uniq % n
    if w is None: return ia, ib, torch.ones(len(uniq))
    ww = torch.zeros(len(uniq)).scatter_reduce_(0, inv, w[keep], reduce="amax", include_self=False); return ia, ib, ww


def build_edges(base: G.GraphData, spec: dict):
    """Returns (edges (E,2) long, weights (E,)) of the requested construction (union-symmetrised unless stated)."""
    xs = topology_features(base, spec.get("feat", "final"), spec.get("whiten", 0), spec.get("noise", 0.0)); n = len(xs); kind = spec.get("construct", "knn"); k = spec.get("k", 10)
    sim = xs @ xs.T; sim.fill_diagonal_(-2.0); nn_idx = sim.topk(max(k, 1) if kind != "bmatch" else 3 * spec.get("b", 10), dim=1).indices
    ar = torch.arange(n)[:, None]
    if kind == "knn":
        if spec.get("feat", "final") == "final" and not spec.get("whiten", 0) and not spec.get("noise", 0.0) and k == base.k: return base.edges.clone(), torch.ones(len(base.edges))   # identical to GraphData (RNG-identical to lap<L>_k10)
        ia, ib, w = _upper(ar.expand_as(nn_idx).reshape(-1), nn_idx.reshape(-1), None)
    elif kind == "mutual":
        a = torch.zeros(n, n, dtype=torch.bool); a[ar.expand_as(nn_idx), nn_idx] = True; m = a & a.T; r, c = torch.nonzero(m, as_tuple=True); keep = r < c; ia, ib, w = r[keep], c[keep], torch.ones(int(keep.sum()))
    elif kind == "eps":   # global similarity threshold giving the requested mean degree (no hubs by construction: hubness does not occur in epsilon graphs)
        deg = spec.get("deg", 10); iu = torch.triu_indices(n, n, 1); vals = sim[iu[0], iu[1]]; thr = vals.topk(min(len(vals), n * deg // 2)).values[-1]; keep = vals >= thr; ia, ib, w = iu[0][keep], iu[1][keep], torch.ones(int(keep.sum()))
    elif kind in ("gauss", "lscale"):
        ia, ib, _ = _upper(ar.expand_as(nn_idx).reshape(-1), nn_idx.reshape(-1), None); d2 = (2 - 2 * (xs[ia] * xs[ib]).sum(1)).clamp_min(0)
        if kind == "gauss": w = torch.exp(-d2 / d2.median().clamp_min(1e-9)).clamp_min(1e-6)
        else:
            kk = min(7, sim.shape[1] - 1); sig = (2 - 2 * sim.topk(kk, dim=1).values[:, -1]).clamp_min(1e-9).sqrt(); w = torch.exp(-d2 / (sig[ia] * sig[ib])).clamp_min(1e-6)
    elif kind == "snn":   # shared nearest neighbours: score = |N_k(i) ∩ N_k(j)| among the 3k nearest candidates, each node keeps its `keep` best partners
        kk = spec.get("snn_k", 20); top = sim.topk(kk, dim=1).indices; a = torch.zeros(n, n); a[ar.expand_as(top), top] = 1.0; shared = a @ a.T
        cand = sim.topk(min(3 * k, n - 1), dim=1).indices; score = shared[ar.expand_as(cand), cand]; best = score.topk(k, dim=1).indices; sel = cand.gather(1, best)
        ia, ib, _ = _upper(ar.expand_as(sel).reshape(-1), sel.reshape(-1), None); w = (shared[ia, ib] / kk).clamp_min(1e-3)
    elif kind == "bmatch":   # greedy b-matching: nodes have at most b edges; edges taken in decreasing similarity (approximation of Jebara et al. 2009)
        b = spec.get("b", 10); ia0, ib0, _ = _upper(ar.expand_as(nn_idx).reshape(-1), nn_idx.reshape(-1), None); order = torch.argsort(-sim[ia0, ib0]); deg = torch.zeros(n, dtype=torch.long); keep = []
        for e in order.tolist():
            i, j = int(ia0[e]), int(ib0[e])
            if deg[i] < b and deg[j] < b: keep.append(e); deg[i] += 1; deg[j] += 1
        keep = torch.as_tensor(keep, dtype=torch.long); ia, ib, w = ia0[keep], ib0[keep], torch.ones(len(keep))
    else: raise ValueError(kind)
    return torch.stack([ia, ib], 1), w


def filter_edges(base: G.GraphData, edges: torch.Tensor, w: torch.Tensor, place: str):
    if place == "all": return edges, w
    ideal = torch.as_tensor([_is_ideal(p) for p in base.paths]); n_ideal = ideal[edges[:, 0]].long() + ideal[edges[:, 1]].long()
    keep = {"unl": n_ideal == 0, "it": n_ideal == 1, "ii": n_ideal == 2, "ideal_any": n_ideal >= 1}[place]
    return edges[keep], w[keep]


def graph_quality(base: G.GraphData, edges: torch.Tensor, w: torch.Tensor) -> dict:
    """Weighted same-class share among ideal-ideal edges (both ideal images with known class), chance ~0.25."""
    src = base.source_class; cls = torch.full((len(base.paths),), -1, dtype=torch.long)
    for i, c in src.items(): cls[i] = c
    ci, cj = cls[edges[:, 0]], cls[edges[:, 1]]; m = (ci >= 0) & (cj >= 0); n_ii = int(m.sum())
    q = float((w[m] * (ci[m] == cj[m]).float()).sum() / w[m].sum()) if n_ii else float("nan")
    deg = torch.zeros(len(base.paths)).index_add_(0, edges[:, 0], w).index_add_(0, edges[:, 1], w)
    return {"quality": q, "n_ii_edges": n_ii, "n_edges": int(len(edges)), "mean_degree": float(2 * len(edges) / len(base.paths)), "isolated_nodes": int((deg == 0).sum())}


def _record_quality(ctx, name: str, stats: dict) -> None:
    QUALITY_DIR.mkdir(parents=True, exist_ok=True); f = QUALITY_DIR / f"seed{ctx.seed}_{'B' if os.environ.get('PAIR_STUDY_SPLIT') == 'classifier2' else 'A'}.json"
    cur = json.loads(f.read_text()) if f.exists() else {}
    if name not in cur: cur[name] = stats; f.write_text(json.dumps(cur))


_KEYS = ("construct", "k", "feat", "whiten", "noise", "deg", "b", "snn_k", "place")


def gm_graph(ctx, name: str, spec: dict):
    cache = ctx.__dict__.setdefault("_gm", {}); key = tuple(spec.get(x) for x in _KEYS)
    if key not in cache:
        base = G.graph_for(ctx, 10); edges, w = build_edges(base, spec); edges, w = filter_edges(base, edges, w, spec.get("place", "all")); cache[key] = (base, edges, w)
    base, edges, w = cache[key]; _record_quality(ctx, spec.get("quality_name", name), graph_quality(base, edges, w)); return base, edges, w


# ------------------------------------------------------------------------------------------ training
def fit_gm(ctx, labelled, seed: int, base: G.GraphData, edges, w, spec: dict):
    """aw8_all + (graph Laplacian | non-graph smoothing) terms.  With spec = {lap: L} on the default kNN graph this reproduces ``lap<L>_k10`` of nd_graph_learner exactly (same RNG stream)."""
    lam = float(spec.get("lap", 0.0)); lr, steps = ctx.params_for(labelled); rows = G.ju_study.rows_of(ctx.exp, labelled); head = core.new_head(base.x.shape[1], seed)
    featfn = core.identity_features(ctx); refs, bad = core.anchors(ctx, featfn, True); node_x = base.x
    if len(rows):
        xa, xb = featfn(list(rows.resolved_img1)), featfn(list(rows.resolved_img2)); typ = torch.as_tensor(rows.type_idx.to_numpy())
        weight, winner = torch.as_tensor(rows.confidence_weight.to_numpy(), dtype=torch.float32), rows.Winner.to_numpy()
    else: xa = xb = torch.empty(0, base.x.shape[1]); typ = torch.empty(0, dtype=torch.long); weight = torch.empty(0); winner = np.array([], dtype=object)
    wd = float(spec.get("wd", ctx.exp.cfg.weight_decay)); head.eval(); opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd); index = torch.arange(len(typ))
    masks = {l: torch.as_tensor(winner == l) for l in core.frozen.LABELS}; cols = {c: core.frozen.TYPE_TO_INDEX[c] for c in refs}; gen = torch.Generator().manual_seed(seed * 1000003 + 17); active = list(ACTIVE)
    tw = torch.as_tensor([float(spec.get("type_w", {}).get(t, 1.0)) for t in ACTIVE]); uniform = bool((w == 1).all()); n_e = len(edges)
    norm = spec.get("lap_type", "comb") == "norm"
    if norm:
        deg = torch.zeros(len(base.paths)).index_add_(0, edges[:, 0], w).index_add_(0, edges[:, 1], w); dn = (deg / deg[deg > 0].mean()).clamp_min(1e-3).sqrt()
    # matched perturbation scale for the noise control: median raw-feature distance between kNN neighbours
    if spec.get("noise_cons", 0) > 0 or spec.get("mix_cons", 0) > 0:
        e0 = base.edges[:2000]; scale = float((base.x[e0[:, 0]] - base.x[e0[:, 1]]).norm(dim=1).median())
    for t in range(steps):
        loss = G._bt_terms(head(xa)[index, typ], head(xb)[index, typ], masks, weight) / max(len(typ), 1)
        if len(refs) > 1: loss = loss + 8.0 * G._anchor_loss({n: head(f) for n, f in refs.items()}, cols)
        if bad is not None and len(bad): loss = loss + ctx.exp.cfg.bad_anchor_weight * F.relu(head(bad) + 1.0).mean()
        sched = {"const": 1.0, "decay": 1.0 - t / steps, "ramp": (t + 1) / steps}[spec.get("anneal", "const")]
        if lam > 0 and n_e:
            pick = torch.randint(n_e, (G.EDGE_BATCH,), generator=gen) if uniform else torch.multinomial(w, G.EDGE_BATCH, replacement=True, generator=gen)
            e = edges[pick]; si, sj = head(node_x[e[:, 0]])[:, active], head(node_x[e[:, 1]])[:, active]
            if norm: si, sj = si / dn[e[:, 0]][:, None], sj / dn[e[:, 1]][:, None]
            loss = loss + sched * lam * (tw * (si - sj) ** 2).sum(1).mean()
        if spec.get("noise_cons", 0) > 0:   # no graph: scores of random nodes must be stable under Gaussian input noise of the size of a kNN step
            u = torch.randint(len(node_x), (G.EDGE_BATCH,), generator=gen); xu = node_x[u]; eps = torch.randn(xu.shape, generator=gen); eps = eps / eps.norm(dim=1, keepdim=True) * scale
            loss = loss + sched * spec["noise_cons"] * (tw * (head(xu + eps)[:, active] - head(xu)[:, active]) ** 2).sum(1).mean()
        if spec.get("mix_cons", 0) > 0:     # no graph: interpolation consistency (ICT-style) on random node pairs
            i, j = torch.randint(len(node_x), (G.EDGE_BATCH,), generator=gen), torch.randint(len(node_x), (G.EDGE_BATCH,), generator=gen); a = torch.rand(G.EDGE_BATCH, 1, generator=gen)
            xm = a * node_x[i] + (1 - a) * node_x[j]; target = (a * head(node_x[i])[:, active] + (1 - a) * head(node_x[j])[:, active]).detach()
            loss = loss + sched * spec["mix_cons"] * (tw * (head(xm)[:, active] - target) ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
    return head.eval()


def gm_learner(name: str, **spec):
    def learner(ctx, labelled, seed):
        base, edges, w = gm_graph(ctx, name, spec) if spec.get("lap", 0) > 0 or spec.get("record_graph") else (G.graph_for(ctx, 10), None, None)
        if edges is None: edges, w = base.edges, torch.ones(len(base.edges))
        head = fit_gm(ctx, labelled, seed, base, edges, w, spec); featfn = core.identity_features(ctx)
        return lambda test: core.head_d(head, featfn, test)
    return learner


LEARNERS: dict = {}
def _reg(name: str, **spec): LEARNERS[name] = gm_learner(name, **spec)

# ---- round 1: dose-response (noise added to the topology features; sigma = norm of the noise relative to unit-norm features)
for _s in (0.5, 1, 2, 3):
    for _l in (3, 10): _reg(f"gm_lap{_l}_n{_s:g}", lap=_l, noise=float(_s))
_reg("gm_lap10_n0", lap=10)                       # clean graph at lambda 10 (reference for the lambda-10 dose-response)
_reg("gm_lap3_n0", lap=3, record_graph=True)      # clean graph at lambda 3 (== lap3_k10; recorded for quality)
# ---- round 2: other graph constructions (all lambda 3)
_reg("gm_mutual_k15", lap=3, construct="mutual", k=15); _reg("gm_eps_d10", lap=3, construct="eps", deg=10); _reg("gm_snn", lap=3, construct="snn", k=10, snn_k=20)
_reg("gm_gauss_k10", lap=3, construct="gauss", k=10); _reg("gm_lscale_k10", lap=3, construct="lscale", k=10); _reg("gm_bmatch_b10", lap=3, construct="bmatch", b=10)
_reg("gm_pcaw32_k10", lap=3, whiten=32); _reg("gm_l3_k10", lap=3, feat="l3"); _reg("gm_l2_k10", lap=3, feat="l2"); _reg("gm_l3l4_k10", lap=3, feat="l3l4"); _reg("gm_norm_k10", lap=3, lap_type="norm")
# ---- round 3: placement / per-type / non-graph controls / annealing
_reg("gm_place_unl", lap=3, place="unl"); _reg("gm_place_it", lap=3, place="it"); _reg("gm_htr10", lap=3, type_w={4: 3.0})   # HTR weight 3x => lambda 9 on HTR, 3 elsewhere
_reg("gm_noise3", noise_cons=3); _reg("gm_noise10", noise_cons=10); _reg("gm_mix3", mix_cons=3); _reg("gm_mix10", mix_cons=10)
_reg("gm_wd1e3", wd=1e-3); _reg("gm_wd1e2", wd=1e-2); _reg("gm_decay3", lap=3, anneal="decay"); _reg("gm_ramp3", lap=3, anneal="ramp")

# ---- the same constructions at lambda = 10 (round 2b: the dose-response of round 1 shows graph quality matters most at larger lambda)
for _name, _spec in (("mutual_k15", dict(construct="mutual", k=15)), ("eps_d10", dict(construct="eps", deg=10)), ("snn", dict(construct="snn", k=10, snn_k=20)), ("gauss_k10", dict(construct="gauss", k=10)),
                     ("lscale_k10", dict(construct="lscale", k=10)), ("bmatch_b10", dict(construct="bmatch", b=10)), ("pcaw32_k10", dict(whiten=32)), ("l3_k10", dict(feat="l3")), ("l2_k10", dict(feat="l2")),
                     ("l3l4_k10", dict(feat="l3l4")), ("norm_k10", dict(lap_type="norm"))):
    _reg(f"gm10_{_name}", lap=10, **_spec)

# ---- round 3 (lambda 10): where the regulariser acts, which types carry it, annealing; non-graph smoothing controls are in the lambda-free block above (gm_noise*, gm_mix*, gm_wd*)
_reg("gm10_place_unl", lap=10, place="unl"); _reg("gm10_place_it", lap=10, place="it")
_reg("gm10_type_htr", lap=10, type_w={0: 0.0, 2: 0.0, 3: 0.0, 4: 1.0}); _reg("gm10_type_no_htr", lap=10, type_w={4: 0.0})
_reg("gm10_decay", lap=10, anneal="decay"); _reg("gm10_ramp", lap=10, anneal="ramp")

# ---- round 4 (refinements): lambda sweep on the plain kNN graph, k sweep at lambda 10, larger lambda on the two constructions with the highest ideal-ideal quality
for _l in (5, 20, 40): _reg(f"gm_lap{_l}_k10", lap=_l)
for _kk in (5, 20, 30): _reg(f"gm10_k{_kk}", lap=10, k=_kk)
_reg("gm20_lscale_k10", lap=20, construct="lscale", k=10); _reg("gm20_mutual_k15", lap=20, construct="mutual", k=15); _reg("gm20_snn", lap=20, construct="snn", k=10, snn_k=20)
