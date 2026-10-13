"""Graph-augmented learners: use the kNN graph over ALL unlabelled images (trajectory frames + ideal images) to improve the reward head itself.

Protocol extension (stated in PREREGISTRATION_G*.md): the learner may use the (unlabelled) SimCLR features of every trajectory image and every ideal image to build a cosine kNN graph, EXCEPT
the images of the seed's validation / test pair groups and the seed's outer-test ideal images (identity-safe by SHA-256 content hash).  Everything else equals ``nd_learners.aw8_all``: frozen SimCLR
features, the repository's loss with reference-anchor weight 8 and ALL ideal images as anchors, the per-budget head schedule, held-out preference endpoint.

Learners (name -> what is added to the aw8_all loss / head):
  lap<L>_k<K>      manifold regularisation: L * mean over sampled kNN edges of sum_k (s_k(i) - s_k(j))^2, on every node (labelled or not)
  rq<L>_k<K>       type-posterior prior: label propagation of the ideal images' class one-hots over the kNN graph gives q(x); L * weighted ranking loss that pulls s_k(i) - s_k(j) towards sign(q_k(i) - q_k(j))
  sgc_k<K>         the head sees Ahat X (one propagation step; unseen images attach to their k nearest graph nodes)
  gcn2_k<K>        two-layer graph-convolution encoder Ahat relu(Ahat X W1) W2 -> per-type scores (anchor loss unchanged)
  combinations     lap+rq, lap+sgc, ...
Controls: shuf (node identities of the adjacency permuted, degree sequence kept; or random neighbours for the encoders), cons (no graph: consistency of the scores of unlabelled nodes under feature
dropout, same lambda grid), noedge (GCN with identity adjacency).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import judgment_unit_study as ju_study
import nd_core as core

DATA = core.single_study.DATA
HASH_FILE = core.HERE / "results" / "content_hashes.json"
ACTIVE = (0, 2, 3, 4)
ALPHA = 0.9               # label-spreading coefficient (as in graph_typed.py)
EDGE_BATCH = 256          # sampled edges / node pairs per optimisation step
_HASHES: dict[str, str] = {}


# ------------------------------------------------------------------------------------------ identity safety
def content_hashes(paths) -> dict[str, str]:
    """SHA-256 of the bytes of every image (cached on disk, keyed by path relative to the data root)."""
    if not _HASHES and HASH_FILE.exists(): _HASHES.update(json.loads(HASH_FILE.read_text()))
    root = str(Path(DATA).resolve()) + "/"; changed = False
    for p in paths:
        key = str(p)[len(root):] if str(p).startswith(root) else str(p)
        if key not in _HASHES: _HASHES[key] = hashlib.sha256(Path(p).read_bytes()).hexdigest(); changed = True
    if changed: HASH_FILE.parent.mkdir(parents=True, exist_ok=True); HASH_FILE.write_text(json.dumps(_HASHES))
    return {str(p): _HASHES[str(p)[len(root):] if str(p).startswith(root) else str(p)] for p in paths}


def excluded_identities(ctx) -> set[str]:
    """Content hashes of every image of the validation / test pair groups and of the outer-test ideal images."""
    paths = {str(x) for g in list(ctx.validation) + list(ctx.test) for r in ctx.exp.groups[g].itertuples() for x in (r.resolved_img1, r.resolved_img2)}
    paths |= {str(p) for ps in ctx.exp.test_images.values() for p in ps}
    return set(content_hashes(sorted(paths)).values())


# ------------------------------------------------------------------------------------------ the graph
class GraphData:
    """kNN graph over the allowed nodes of one (seed, split) context."""
    def __init__(self, ctx, k: int = 10, shuffle: int | None = None, noise: float = 0.0):
        cache = ctx.features.cache; all_paths = sorted(cache); hashes = content_hashes(all_paths); bad = excluded_identities(ctx)
        self.paths = [p for p in all_paths if hashes[p] not in bad]; self.index = {p: i for i, p in enumerate(self.paths)}; n = len(self.paths); self.k = k
        self.x = torch.stack([cache[p] for p in self.paths]).float(); mu = self.x.mean(0); xc = self.x - mu
        self.mu = mu; self.xn = xc / xc.norm(dim=1, keepdim=True).clamp_min(1e-9)
        xs = self.xn
        if noise > 0:   # dose-response control: the topology is built from features with added isotropic noise of norm `noise` (unit-norm features), everything else stays clean
            g = torch.Generator().manual_seed(12345); xs = self.xn + noise * torch.randn(self.xn.shape, generator=g) / self.xn.shape[1] ** 0.5
        sim = xs @ xs.T; sim.fill_diagonal_(-2.0); self.nbr = sim.topk(k, dim=1).indices            # (n, k) nearest nodes
        a = torch.zeros(n, n, dtype=torch.bool); a[torch.arange(n)[:, None].expand_as(self.nbr), self.nbr] = True; a = a | a.T      # symmetric kNN adjacency
        self.shuffled = shuffle is not None
        if shuffle is not None:   # control: permute node identities (topology and degree sequence kept, correspondence with the features destroyed)
            perm = torch.as_tensor(np.random.default_rng(shuffle * 7919 + 11).permutation(n)); a = a[perm][:, perm]
        self.adj = a; deg = a.sum(1).float() + 1.0                                                                    # self loop
        rows, cols = torch.nonzero(a, as_tuple=True); self.edges = torch.stack([rows, cols], 1)[rows < cols]            # undirected edge list
        ar = torch.arange(n); idx = torch.cat([torch.stack([rows, cols]), torch.stack([ar, ar])], 1); val = torch.cat([1.0 / deg[rows], 1.0 / deg])
        self.ahat = torch.sparse_coo_tensor(idx, val, (n, n)).coalesce()                                              # row-normalised (A + I)
        self.n_edges = len(self.edges)
        # label sources: ideal images whose class label is known to the learner and which are graph nodes (references + utility-validation ideals; outer-test ideals are not nodes)
        self.source_class: dict[int, int] = {}
        for table in (ctx.exp.references, ctx.exp.utility_images):
            for cls, ps in table.items():
                for p in ps:
                    if str(p) in self.index: self.source_class[self.index[str(p)]] = core.frozen.TYPE_TO_INDEX[cls]
        self._q = None; self._n = n; self._deg = deg

    # label spreading of the class one-hots over the graph: q(x) = normalised (I - alpha S)^-1 Y, S = D^-1/2 A D^-1/2
    def q(self) -> torch.Tensor:
        if self._q is None:
            n = self._n; y = torch.zeros(n, 5, dtype=torch.float64)
            for i, c in self.source_class.items(): y[i, c] = 1.0
            a = self.adj.double(); d = a.sum(1).clamp_min(1.0); s = a / torch.sqrt(d[:, None] * d[None, :])
            f = torch.linalg.solve(torch.eye(n, dtype=torch.float64) - ALPHA * s, y).clamp_min(0)[:, list(ACTIVE)]
            mass = f.sum(1, keepdim=True); self._q = torch.where(mass > 1e-9, f / mass.clamp_min(1e-9), torch.full_like(f, 1.0 / f.shape[1])).float()   # components without any labelled source: uniform
        return self._q

    # features for images that are not nodes (validation / test images, outer-test ideal images): attach to the k nearest nodes, never the other way round
    def neighbours_of(self, paths, cache) -> torch.Tensor:
        x = torch.stack([cache[p] for p in paths]).float(); xn = (x - self.mu) / (x - self.mu).norm(dim=1, keepdim=True).clamp_min(1e-9)
        return (xn @ self.xn.T).topk(self.k, dim=1).indices

    def propagate_features(self, paths, cache, shuffle_ext: bool = False) -> torch.Tensor:
        """Ahat X for nodes, mean of own feature and the k nearest nodes' features for non-nodes (random nodes in the shuffled control)."""
        out = torch.empty(len(paths), self.x.shape[1]); ext = [i for i, p in enumerate(paths) if p not in self.index]; nod = [i for i, p in enumerate(paths) if p in self.index]
        if nod: out[nod] = torch.sparse.mm(self.ahat, self.x)[[self.index[paths[i]] for i in nod]]
        if ext:
            nb = self.neighbours_of([paths[i] for i in ext], cache)
            if shuffle_ext: nb = torch.as_tensor(np.random.default_rng(len(ext) * 31 + 5).integers(0, len(self.paths), nb.shape))
            own = torch.stack([cache[paths[i]] for i in ext]).float(); out[ext] = (own + self.x[nb].sum(1)) / (self.k + 1)
        return out

    def propagated_nodes(self) -> torch.Tensor:
        if not hasattr(self, "_px"): self._px = torch.sparse.mm(self.ahat, self.x)
        return self._px


def graph_for(ctx, k: int, shuffle: bool = False, noise: float = 0.0) -> GraphData:
    key = (id(ctx), k, shuffle, noise)
    cached = ctx.__dict__.setdefault("_graphs", {})
    if key not in cached: cached[key] = GraphData(ctx, k, shuffle=ctx.seed if shuffle else None, noise=noise)
    return cached[key]


# ------------------------------------------------------------------------------------------ training loops
def _bt_terms(a, b, winner_masks, weight):
    terms = (-F.logsigmoid(a - b) * winner_masks["1"] - F.logsigmoid(b - a) * winner_masks["2"] + (a - b).abs() * winner_masks["tie"] + (F.relu(a) + F.relu(b)) * winner_masks["not_apply"])
    return (terms * weight).sum()


def _anchor_loss(scores: dict[str, torch.Tensor], cols: dict[str, int], type_w: dict | None = None) -> torch.Tensor:
    values = []
    for p, sp in scores.items():
        for o, so in scores.items():
            if o != p: values.append((1.0 if not type_w else float(type_w.get(cols[p], 1.0))) * -F.logsigmoid(sp[:, cols[p]][:, None] - so[:, cols[p]][None, :]).mean())
    return torch.stack(values).mean()


def fit_mlp_graph(ctx, labelled, seed, gd: GraphData | None, featfn, node_x: torch.Tensor | None, dim: int, lap: float = 0.0, rankq: float = 0.0, cons: float = 0.0, ridge: float = 0.0,
                  lap_w: dict | None = None, anchor_w: dict | None = None, lr_scale: float = 1.0, steps_scale: float = 1.0, aw: float = 8.0, cons_dropout: float = 0.3, q: torch.Tensor | None = None):
    """MLP head (the repository's head on whatever ``featfn`` returns) with optional graph terms evaluated on ``node_x`` (the features the head consumes for every node).
    With lap = rankq = cons = 0 this is exactly ``nd_learners.aw8_all``."""
    lr, steps = ctx.params_for(labelled); lr, steps = lr * lr_scale, int(round(steps * steps_scale)); rows = ju_study.rows_of(ctx.exp, labelled); head = core.new_head(dim, seed); refs, bad = core.anchors(ctx, featfn, True)
    tw = torch.ones(len(ACTIVE)) if not lap_w else torch.tensor([float(lap_w.get(t, 1.0)) for t in ACTIVE])   # per-type multiplier of the Laplacian weight
    if len(rows):
        xa, xb = featfn(list(rows.resolved_img1)), featfn(list(rows.resolved_img2)); typ = torch.as_tensor(rows.type_idx.to_numpy())
        weight, winner = torch.as_tensor(rows.confidence_weight.to_numpy(), dtype=torch.float32), rows.Winner.to_numpy()
    else: xa = xb = torch.empty(0, dim); typ = torch.empty(0, dtype=torch.long); weight = torch.empty(0); winner = np.array([], dtype=object)
    head.eval(); opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=ctx.exp.cfg.weight_decay); index = torch.arange(len(typ)); masks = {l: torch.as_tensor(winner == l) for l in core.frozen.LABELS}
    cols = {c: core.frozen.TYPE_TO_INDEX[c] for c in refs}; gen = torch.Generator().manual_seed(seed * 1000003 + 17); active = list(ACTIVE)
    for _ in range(steps):
        loss = _bt_terms(head(xa)[index, typ], head(xb)[index, typ], masks, weight) / max(len(typ), 1)
        if len(refs) > 1: loss = loss + aw * _anchor_loss({n: head(f) for n, f in refs.items()}, cols, anchor_w)
        if bad is not None and len(bad): loss = loss + ctx.exp.cfg.bad_anchor_weight * F.relu(head(bad) + 1.0).mean()
        if lap > 0 and gd is not None:
            e = gd.edges[torch.randint(gd.n_edges, (EDGE_BATCH,), generator=gen)]; si, sj = head(node_x[e[:, 0]])[:, active], head(node_x[e[:, 1]])[:, active]
            loss = loss + lap * (((si - sj) ** 2) * tw).sum(1).mean()
        if rankq > 0 and q is not None:
            i, j = torch.randint(len(node_x), (EDGE_BATCH,), generator=gen), torch.randint(len(node_x), (EDGE_BATCH,), generator=gen)
            si, sj = head(node_x[i])[:, active], head(node_x[j])[:, active]; dq = q[i] - q[j]; w = dq.abs(); w = w / (w.mean() + 1e-9)
            loss = loss + rankq * (w * F.softplus(-torch.sign(dq) * (si - sj))).mean()
        if cons > 0:   # no-graph control: unlabelled nodes' scores must be stable under feature dropout
            u = torch.randint(len(node_x), (EDGE_BATCH,), generator=gen); xu = node_x[u]; keep = (torch.rand(xu.shape, generator=gen) > cons_dropout).float() / (1 - cons_dropout)
            loss = loss + cons * ((head(xu)[:, active] - head(xu * keep)[:, active]) ** 2).sum(1).mean()
        if ridge > 0:   # no-graph control: shrink the scores of random (mostly unlabelled) nodes towards 0, same kind of strength knob as lap
            u = torch.randint(len(node_x), (EDGE_BATCH,), generator=gen); loss = loss + ridge * (head(node_x[u])[:, active] ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
    return head.eval()


class GCN2(nn.Module):
    """Two graph-convolution layers with a per-type output: S = Ahat relu(Ahat X W1) W2."""
    def __init__(self, dim: int, seed: int, hidden: int = 256):
        super().__init__(); torch.manual_seed(seed); self.l1 = nn.Linear(dim, hidden); self.l2 = nn.Linear(hidden, 5)


def _gcn_scores(group, model, gd, z2, h2):
    """Scores of a group of images: graph nodes read their propagated output, other images attach to their k nearest nodes (own term + neighbours' pre-propagation outputs)."""
    ids, nb, own = group; node = ids >= 0; out = z2[ids.clamp_min(0)]
    if (~node).any():
        xm = (own + gd.x[nb].sum(1)) / (gd.k + 1); su = model.l2(F.relu(model.l1(xm))); ext = (su + h2[nb].sum(1)) / (gd.k + 1); out = torch.where(node[:, None], out, ext)
    return out


def _gcn_forward(model, px, ahat):
    h2 = model.l2(F.relu(model.l1(px))); return h2, torch.sparse.mm(ahat, h2)


def fit_gcn2(ctx, labelled, seed, gd: GraphData, cache, aw: float = 8.0, noedge: bool = False, random_ext: bool = False, lap: float = 0.0):
    """Full-graph GCN encoder trained with the repository's loss.  Ideal / bad images that are not graph nodes (outer-test ideals) attach to their k nearest nodes."""
    lr, steps = ctx.params_for(labelled); rows = ju_study.rows_of(ctx.exp, labelled); model = GCN2(gd.x.shape[1], seed); n = len(gd.paths)
    ahat = gd.ahat if not noedge else torch.sparse_coo_tensor(torch.stack([torch.arange(n)] * 2), torch.ones(n), (n, n)).coalesce()
    px = torch.sparse.mm(ahat, gd.x)
    refs_paths = {c: [str(p) for p in list(ps) + list(ctx.exp.utility_images.get(c, [])) + list(ctx.exp.test_images.get(c, []))] for c, ps in ctx.exp.references.items()}
    bad_paths = [str(p) for p in ctx.exp.bad_paths]
    def prep(paths):   # (node ids or -1, neighbour ids for non-nodes, own raw features)
        ids = torch.as_tensor([gd.index.get(p, -1) for p in paths]); ext = ids < 0; nb = torch.zeros(len(paths), gd.k, dtype=torch.long)
        if ext.any():
            sel = [p for p, e in zip(paths, ext.tolist()) if e]; nbr = gd.neighbours_of(sel, cache)
            if random_ext: nbr = torch.as_tensor(np.random.default_rng(len(sel) * 17 + 3).integers(0, n, nbr.shape))
            nb[ext] = nbr
        return ids, nb, torch.stack([cache[p] for p in paths]).float()
    groups = {"a": prep(list(map(str, rows.resolved_img1))) if len(rows) else None, "b": prep(list(map(str, rows.resolved_img2))) if len(rows) else None,
              "bad": prep(bad_paths) if bad_paths else None, **{f"ref:{c}": prep(ps) for c, ps in refs_paths.items()}}
    cols = {c: core.frozen.TYPE_TO_INDEX[c] for c in refs_paths}; typ = torch.as_tensor(rows.type_idx.to_numpy()) if len(rows) else torch.empty(0, dtype=torch.long)
    weight = torch.as_tensor(rows.confidence_weight.to_numpy(), dtype=torch.float32) if len(rows) else torch.empty(0); winner = rows.Winner.to_numpy() if len(rows) else np.array([], dtype=object)
    masks = {l: torch.as_tensor(winner == l) for l in core.frozen.LABELS}; index = torch.arange(len(typ)); opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=ctx.exp.cfg.weight_decay)
    gen = torch.Generator().manual_seed(seed * 1000003 + 17); active = list(ACTIVE)
    for _ in range(steps):
        h2, z2 = _gcn_forward(model, px, ahat); sc = lambda g: _gcn_scores(g, model, gd, z2, h2); loss = torch.zeros(())
        if len(rows): loss = loss + _bt_terms(sc(groups["a"])[index, typ], sc(groups["b"])[index, typ], masks, weight) / len(typ)
        ref_scores = {c: sc(groups[f"ref:{c}"]) for c in refs_paths}
        if len(ref_scores) > 1: loss = loss + aw * _anchor_loss(ref_scores, cols)
        if groups["bad"] is not None: loss = loss + ctx.exp.cfg.bad_anchor_weight * F.relu(sc(groups["bad"]) + 1.0).mean()
        if lap > 0:
            e = gd.edges[torch.randint(gd.n_edges, (EDGE_BATCH,), generator=gen)]; loss = loss + lap * ((z2[e[:, 0]][:, active] - z2[e[:, 1]][:, active]) ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    model.eval()
    with torch.no_grad(): h2, z2 = _gcn_forward(model, px, ahat)
    def predict(paths):
        with torch.no_grad(): return _gcn_scores(prep(list(map(str, paths))), model, gd, z2, h2)
    return predict


def d_from_scores(score_fn, test) -> np.ndarray:
    with torch.no_grad(): a, b = score_fn(test.img1), score_fn(test.img2)
    return (a - b)[torch.arange(len(test.k)), torch.as_tensor(test.k)].numpy()


# ------------------------------------------------------------------------------------------ learner registry
def _mlp_learner(k=10, lap=0.0, rankq=0.0, cons=0.0, ridge=0.0, sgc=False, shuffle=False, q_kind="graph", noise=0.0, lap_w=None, anchor_w=None, lr_scale=1.0, steps_scale=1.0):
    def learner(ctx, labelled, seed):
        gd = graph_for(ctx, k, shuffle, noise); cache = ctx.features.cache
        if sgc: featfn = lambda paths: gd.propagate_features([str(p) for p in paths], cache, shuffle_ext=shuffle); node_x = gd.propagated_nodes(); dim = gd.x.shape[1]
        else: featfn = core.identity_features(ctx); node_x = gd.x; dim = gd.x.shape[1]
        q = None
        if rankq > 0:
            if q_kind == "graph": q = gd.q()
            else:   # no-graph control: class posterior from the cosine similarity to the class means of the labelled ideal images
                cls = torch.stack([gd.xn[[i for i, c in gd.source_class.items() if c == t]].mean(0) for t in ACTIVE]); sim = gd.xn @ F.normalize(cls, dim=1).T; q = torch.softmax(sim * 20.0, 1)
        head = fit_mlp_graph(ctx, labelled, seed, gd, featfn, node_x, dim, lap=lap, rankq=rankq, cons=cons, ridge=ridge, lap_w=lap_w, anchor_w=anchor_w, lr_scale=lr_scale, steps_scale=steps_scale, q=q)
        return lambda test: core.head_d(head, featfn, test)
    return learner


def _gcn_learner(k=10, noedge=False, shuffle=False, lap=0.0):
    def learner(ctx, labelled, seed):
        gd = graph_for(ctx, k, shuffle); fn = fit_gcn2(ctx, labelled, seed, gd, ctx.features.cache, noedge=noedge, random_ext=shuffle, lap=lap)
        return lambda test: d_from_scores(fn, test)
    return learner


LEARNERS: dict = {}
for _k in (5, 10, 20):
    for _lam in (0.1, 0.3, 1.0, 3.0):
        LEARNERS[f"lap{_lam:g}_k{_k}"] = _mlp_learner(k=_k, lap=_lam)
        LEARNERS[f"lap{_lam:g}_k{_k}_shuf"] = _mlp_learner(k=_k, lap=_lam, shuffle=True)
    LEARNERS[f"sgc_k{_k}"] = _mlp_learner(k=_k, sgc=True); LEARNERS[f"sgc_k{_k}_shuf"] = _mlp_learner(k=_k, sgc=True, shuffle=True)
    LEARNERS[f"gcn2_k{_k}"] = _gcn_learner(k=_k); LEARNERS[f"gcn2_k{_k}_shuf"] = _gcn_learner(k=_k, shuffle=True)
LEARNERS["gcn2_noedge"] = _gcn_learner(noedge=True)
for _lam in (0.1, 0.3, 1.0, 3.0):
    LEARNERS[f"rq{_lam:g}_k10"] = _mlp_learner(rankq=_lam); LEARNERS[f"rq{_lam:g}_k10_shuf"] = _mlp_learner(rankq=_lam, shuffle=True); LEARNERS[f"rq{_lam:g}_k10_noprop"] = _mlp_learner(rankq=_lam, q_kind="proto")
    LEARNERS[f"cons{_lam:g}"] = _mlp_learner(cons=_lam)

# combinations
LEARNERS["lap0.3_rq0.3_k10"] = _mlp_learner(lap=0.3, rankq=0.3)
LEARNERS["lap0.3_sgc_k10"] = _mlp_learner(lap=0.3, sgc=True)
LEARNERS["rq0.3_sgc_k10"] = _mlp_learner(rankq=0.3, sgc=True)
LEARNERS["gcn2_lap0.3_k10"] = _gcn_learner(lap=0.3)
for _lam in (0.1, 0.3, 1.0): LEARNERS[f"lap{_lam:g}_sgc_k10"] = _mlp_learner(lap=_lam, sgc=True)
for _k in (10, 20):
    for _lam in (10.0, 30.0):
        LEARNERS[f"lap{_lam:g}_k{_k}"] = _mlp_learner(k=_k, lap=_lam); LEARNERS[f"lap{_lam:g}_k{_k}_shuf"] = _mlp_learner(k=_k, lap=_lam, shuffle=True)
for _lam in (0.1, 0.3, 1.0, 3.0, 10.0): LEARNERS[f"ridge{_lam:g}"] = _mlp_learner(ridge=_lam)
for _lam in (1.0, 3.0, 10.0):
    for _sg in (1.0, 3.0): LEARNERS[f"lap{_lam:g}_k10_n{_sg:g}"] = _mlp_learner(lap=_lam, noise=_sg)
for _l, _r in ((3.0, 1.0), (3.0, 3.0), (10.0, 1.0)):
    LEARNERS[f"lap{_l:g}_rq{_r:g}_k10"] = _mlp_learner(lap=_l, rankq=_r); LEARNERS[f"lap{_l:g}_rq{_r:g}_k10_shuf"] = _mlp_learner(lap=_l, rankq=_r, shuffle=True)
# selection-x-HTR study: per-type Laplacian weight (HTR = type index 4) and schedule variations of the incumbent lap3_k10
for _w in (3.0, 10.0): LEARNERS[f"lap3h{_w:g}_k10"] = _mlp_learner(lap=3.0, lap_w={4: _w / 3.0}); LEARNERS[f"lap3h{_w:g}_k10_shuf"] = _mlp_learner(lap=3.0, lap_w={4: _w / 3.0}, shuffle=True)
for _st in (0.5, 2.0): LEARNERS[f"lap3_k10_st{_st:g}"] = _mlp_learner(lap=3.0, steps_scale=_st)
for _lr in (0.5, 2.0): LEARNERS[f"lap3_k10_lr{_lr:g}"] = _mlp_learner(lap=3.0, lr_scale=_lr)
for _a in (2.0, 4.0): LEARNERS[f"lap3_ha{_a:g}_k10"] = _mlp_learner(lap=3.0, anchor_w={4: _a})   # HTR reference-anchor weight x _a (HTR has 26 of 144 anchors)
LEARNERS["lap3_ha2h10_k10"] = _mlp_learner(lap=3.0, anchor_w={4: 2.0}, lap_w={4: 10.0 / 3.0})
# schedule re-tuning of the Laplacian learner and of its Laplacian-free counterpart (fit_mlp_graph with lap = 0 is exactly aw8_all, tested): same schedule knobs for both
for _st in (3.0, 4.0): LEARNERS[f"lap3_k10_st{_st:g}"] = _mlp_learner(lap=3.0, steps_scale=_st)
LEARNERS["lap3_k10_st2lr2"] = _mlp_learner(lap=3.0, steps_scale=2.0, lr_scale=2.0)
for _st in (2.0, 3.0, 4.0): LEARNERS[f"aw8_all_st{_st:g}"] = _mlp_learner(lap=0.0, steps_scale=_st)
for _st in (6.0, 8.0): LEARNERS[f"lap3_k10_st{_st:g}"] = _mlp_learner(lap=3.0, steps_scale=_st)
for _lam in (1.0, 10.0): LEARNERS[f"lap{_lam:g}_k10_st4"] = _mlp_learner(lap=_lam, steps_scale=4.0)
LEARNERS["lap3_k10_st4lr0.5"] = _mlp_learner(lap=3.0, steps_scale=4.0, lr_scale=0.5)
LEARNERS["lap3_k10_st4_shuf"] = _mlp_learner(lap=3.0, steps_scale=4.0, shuffle=True); LEARNERS["lap3_k10_st4_n3"] = _mlp_learner(lap=3.0, steps_scale=4.0, noise=3.0)
LEARNERS["aw8_all_lr2"] = _mlp_learner(lap=0.0, lr_scale=2.0); LEARNERS["aw8_all_st2lr0.5"] = _mlp_learner(lap=0.0, steps_scale=2.0, lr_scale=0.5)
# graph-free regulariser at the long schedule (robustness check of the Laplacian-free baseline): dropout-consistency of unlabelled nodes' scores (no graph) with steps x4
for _lam in (0.3, 1.0): LEARNERS[f"cons{_lam:g}_st4"] = _mlp_learner(cons=_lam, steps_scale=4.0)
LEARNERS["sgc_k10_noprop"] = None  # placeholder removed below
del LEARNERS["sgc_k10_noprop"]
