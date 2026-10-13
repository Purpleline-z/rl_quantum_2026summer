"""Regularisers on unlabelled images, graph or not, at the long head schedule (study G2, branch claude/regularisers-*).

Everything equals ``nd_graph_learner.fit_mlp_graph`` (= ``aw8_all``: frozen SimCLR features, the repository's loss with reference-anchor weight 8 on ALL ideal images, per-budget head schedule,
the head is trained in eval mode so its Dropout is inert) plus ONE extra term / switch on the unlabelled nodes (all trajectory + ideal images except the seed's validation/test images,
identity-safe by content hash).  With every option off it is exactly ``aw8_all`` (unit-tested), and with only ``cons`` / ``lap`` on it reproduces ``fit_mlp_graph`` to the last digit (unit-tested;
the random-number draws of the shared terms come first, in the same order).

Options (name -> what is added):
  cons       dropout-consistency of the unlabelled nodes' scores under input-feature dropout p=cons_p            (graph-free; existing control ``cons<L>_st<x>``)
  noise      same under additive Gaussian input noise of std ``noise_s`` x (per-feature std of the nodes)       (graph-free)
  mirror     same between an image and its horizontally mirrored image (mirror SimCLR features)                (graph-free; needs ``feature_cache_mirror.pt``)
  mix        mixup consistency: s(l x_i + (1-l) x_j) -> l s(x_i) + (1-l) s(x_j) (detached target), random node pairs, l ~ Beta(a, a)  (graph-free)
  rdrop      R-Drop style: two forward passes of the head in train mode (hidden dropout p=hdrop) on unlabelled nodes, squared score difference  (graph-free)
  hdrop      the head is trained in train mode (hidden dropout p) instead of eval mode                         (graph-free)
  wd         AdamW weight decay override                                                                       (graph-free)
  ema / swa  evaluate the exponential moving average of the head weights (decay) / the uniform average of the last ``swa`` fraction of the steps   (graph-free)
  inner      early stopping on an inner split of the labelled judgments: train on (1-inner) of the rows, keep the best-inner-loss step, retrain on all rows for that many steps   (graph-free)
  lap / lap_hidden   graph terms: Laplacian on the scores (k-NN, sampled edges; k = 1 = hard neighbour: the nearest node only), or on the hidden activations
Manifold mixup in the hidden layer is not implemented separately: with the repository's one-hidden-layer head the output layer is linear in the hidden activations, so mixing the post-ReLU
hidden state gives a consistency target that is met identically (zero loss, tested), and mixing the pre-activation is input mixup (``mix``).
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

import judgment_unit_study as ju_study
import nd_core as core
import nd_graph_learner as G
import nd_learners as L

ACTIVE = G.ACTIVE
EDGE_BATCH = G.EDGE_BATCH


def _flat(head) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in head.parameters()])


def _load_flat(head, vec: torch.Tensor):
    i = 0
    for p in head.parameters(): n = p.numel(); p.data.copy_(vec[i:i + n].view_as(p)); i += n


def _train(ctx, rows, seed, gd, featfn, node_x, dim, steps, lr, *, lap=0.0, lap_hidden=0.0, cons=0.0, cons_p=0.3, noise=0.0, noise_s=0.5, mirror=0.0, mirror_x=None, mix=0.0,
           mix_a=0.4, rdrop=0.0, hdrop=0.0, wd=None, ema=0.0, swa=0.0, aw=8.0, trace_rows=None, trace_every=0, tw=None):
    """One head fit.  ``rows`` = labelled judgments (data frame as returned by ``ju_study.rows_of``).  Returns the head (EMA / SWA weights if requested) and, if ``trace_rows`` is given, the
    list of (step, inner loss) used for early stopping."""
    head = core.new_head(dim, seed); refs, bad = core.anchors(ctx, featfn, True)
    if len(rows):
        xa, xb = featfn(list(rows.resolved_img1)), featfn(list(rows.resolved_img2)); typ = torch.as_tensor(rows.type_idx.to_numpy())
        weight, winner = torch.as_tensor(rows.confidence_weight.to_numpy(), dtype=torch.float32), rows.Winner.to_numpy()
    else: xa = xb = torch.empty(0, dim); typ = torch.empty(0, dtype=torch.long); weight = torch.empty(0); winner = np.array([], dtype=object)
    if hdrop > 0: head[2].p = hdrop
    head.train() if (hdrop > 0 or rdrop > 0) else head.eval()
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=ctx.exp.cfg.weight_decay if wd is None else wd)
    index = torch.arange(len(typ)); masks = {l: torch.as_tensor(winner == l) for l in core.frozen.LABELS}; cols = {c: core.frozen.TYPE_TO_INDEX[c] for c in refs}
    gen = torch.Generator().manual_seed(seed * 1000003 + 17); active = list(ACTIVE); tw = torch.ones(len(ACTIVE)) if tw is None else tw
    std = node_x.std(0, keepdim=True) if noise > 0 else None; hidden = lambda x: F.relu(head[0](x))
    avg = _flat(head).clone() if ema > 0 else None; swa_sum, swa_n = None, 0; trace = []
    if trace_rows is not None:
        ta, tb = featfn(list(trace_rows.resolved_img1)), featfn(list(trace_rows.resolved_img2)); tt = torch.as_tensor(trace_rows.type_idx.to_numpy()); tidx = torch.arange(len(tt))
        tw8 = torch.as_tensor(trace_rows.confidence_weight.to_numpy(), dtype=torch.float32); tm = {l: torch.as_tensor(trace_rows.Winner.to_numpy() == l) for l in core.frozen.LABELS}
    for step in range(steps):
        loss = G._bt_terms(head(xa)[index, typ], head(xb)[index, typ], masks, weight) / max(len(typ), 1)
        if len(refs) > 1: loss = loss + aw * G._anchor_loss({n: head(f) for n, f in refs.items()}, cols, None)
        if bad is not None and len(bad): loss = loss + ctx.exp.cfg.bad_anchor_weight * F.relu(head(bad) + 1.0).mean()
        if lap > 0 or lap_hidden > 0:
            e = gd.edges[torch.randint(gd.n_edges, (EDGE_BATCH,), generator=gen)]
            if lap > 0: si, sj = head(node_x[e[:, 0]])[:, active], head(node_x[e[:, 1]])[:, active]; loss = loss + lap * (((si - sj) ** 2) * tw).sum(1).mean()
            if lap_hidden > 0: hi, hj = hidden(node_x[e[:, 0]]), hidden(node_x[e[:, 1]]); loss = loss + lap_hidden * (((hi - hj) ** 2).mean(1) * 4.0).mean()
        if cons > 0:
            u = torch.randint(len(node_x), (EDGE_BATCH,), generator=gen); xu = node_x[u]; keep = (torch.rand(xu.shape, generator=gen) > cons_p).float() / (1 - cons_p)
            loss = loss + cons * ((head(xu)[:, active] - head(xu * keep)[:, active]) ** 2).sum(1).mean()
        if noise > 0:
            u = torch.randint(len(node_x), (EDGE_BATCH,), generator=gen); xu = node_x[u]; xn = xu + noise_s * std * torch.randn(xu.shape, generator=gen)
            loss = loss + noise * ((head(xu)[:, active] - head(xn)[:, active]) ** 2).sum(1).mean()
        if mirror > 0:
            u = torch.randint(len(node_x), (EDGE_BATCH,), generator=gen); loss = loss + mirror * ((head(node_x[u])[:, active] - head(mirror_x[u])[:, active]) ** 2).sum(1).mean()
        if mix > 0:
            i, j = torch.randint(len(node_x), (EDGE_BATCH,), generator=gen), torch.randint(len(node_x), (EDGE_BATCH,), generator=gen)
            lam = torch.as_tensor(np.random.default_rng(seed * 7919 + step).beta(mix_a, mix_a, (EDGE_BATCH, 1)), dtype=torch.float32)
            with torch.no_grad(): target = lam * head(node_x[i])[:, active] + (1 - lam) * head(node_x[j])[:, active]
            loss = loss + mix * ((head(lam * node_x[i] + (1 - lam) * node_x[j])[:, active] - target) ** 2).sum(1).mean()
        if rdrop > 0:
            u = torch.randint(len(node_x), (EDGE_BATCH,), generator=gen); xu = node_x[u]; loss = loss + rdrop * ((head(xu)[:, active] - head(xu)[:, active]) ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
        if ema > 0: avg = ema * avg + (1 - ema) * _flat(head)
        if swa > 0 and step >= steps * (1 - swa): v = _flat(head); swa_sum = v.clone() if swa_sum is None else swa_sum + v; swa_n += 1
        if trace_rows is not None and (step + 1) % trace_every == 0:
            head.eval()
            with torch.no_grad(): trace.append((step + 1, float(G._bt_terms(head(ta)[tidx, tt], head(tb)[tidx, tt], tm, tw8) / max(len(tt), 1))))
            head.train() if (hdrop > 0 or rdrop > 0) else head.eval()
    if ema > 0: _load_flat(head, avg)
    if swa > 0 and swa_n: _load_flat(head, swa_sum / swa_n)
    head.eval()
    return (head, trace) if trace_rows is not None else head


def fit_reg(ctx, labelled, seed, gd, featfn, node_x, dim, steps_scale=1.0, lr_scale=1.0, inner=0.0, inner_every=20, **opts):
    lr, steps = ctx.params_for(labelled); lr, steps = lr * lr_scale, int(round(steps * steps_scale)); rows = ju_study.rows_of(ctx.exp, labelled)
    if inner > 0 and len(rows) >= 8:
        rng = np.random.default_rng(seed * 31 + 7); n_val = max(2, int(round(inner * len(rows)))); perm = rng.permutation(len(rows)); val_i, tr_i = np.sort(perm[:n_val]), np.sort(perm[n_val:])
        _, trace = _train(ctx, rows.iloc[tr_i], seed, gd, featfn, node_x, dim, steps, lr, trace_rows=rows.iloc[val_i], trace_every=inner_every, **opts)
        best = min(trace, key=lambda t: (t[1], t[0]))[0]; steps = best
    return _train(ctx, rows, seed, gd, featfn, node_x, dim, steps, lr, **opts)


def _learner(k=10, shuffle=False, steps_scale=1.0, lr_scale=1.0, inner=0.0, **opts):
    def learner(ctx, labelled, seed):
        gd = G.graph_for(ctx, k, shuffle); featfn = core.identity_features(ctx); node_x = gd.x
        o = dict(opts)
        if o.get("mirror", 0) > 0: o["mirror_x"] = L.mirror_features(ctx, gd.paths)
        head = fit_reg(ctx, labelled, seed, gd, featfn, node_x, gd.x.shape[1], steps_scale=steps_scale, lr_scale=lr_scale, inner=inner, **o)
        return lambda test: core.head_d(head, featfn, test)
    return learner


LEARNERS: dict = {}


def _name(prefix, w, st, extra=""): return f"{prefix}{w:g}{extra}_st{st:g}"


for _st in (1.0, 2.0, 4.0, 6.0):
    for _w in (0.1, 0.3, 1.0, 3.0): LEARNERS[_name("rgcons", _w, _st)] = _learner(steps_scale=_st, cons=_w)                       # dropout consistency (p = 0.3)
    for _w in (0.3, 1.0, 3.0):
        LEARNERS[_name("rgnoise", _w, _st)] = _learner(steps_scale=_st, noise=_w)                                              # Gaussian-noise consistency (s = 0.5)
        LEARNERS[_name("rgmix", _w, _st)] = _learner(steps_scale=_st, mix=_w)                                                 # mixup consistency
        LEARNERS[_name("rgmirror", _w, _st)] = _learner(steps_scale=_st, mirror=_w)                                           # mirror consistency
        LEARNERS[_name("rgrdrop", _w, _st)] = _learner(steps_scale=_st, rdrop=_w, hdrop=0.2)                                 # R-Drop (hidden dropout 0.2)
    for _p in (0.2, 0.5): LEARNERS[f"rghdrop{_p:g}_st{_st:g}"] = _learner(steps_scale=_st, hdrop=_p)
    for _wd in (0.1, 1.0): LEARNERS[f"rgwd{_wd:g}_st{_st:g}"] = _learner(steps_scale=_st, wd=_wd)
    for _e in (0.9, 0.98): LEARNERS[f"rgema{_e:g}_st{_st:g}"] = _learner(steps_scale=_st, ema=_e)
    for _f in (0.5,): LEARNERS[f"rgswa{_f:g}_st{_st:g}"] = _learner(steps_scale=_st, swa=_f)
    LEARNERS[f"rginner_st{_st:g}"] = _learner(steps_scale=_st, inner=0.25)
    for _w in (0.3, 1.0, 3.0, 10.0):
        LEARNERS[_name("rglaph", _w, _st, "_k10")] = _learner(steps_scale=_st, lap_hidden=_w)                                  # Laplacian on the hidden layer
        LEARNERS[_name("rglaph", _w, _st, "_k10_shuf")] = _learner(steps_scale=_st, shuffle=True, lap_hidden=_w)
        LEARNERS[_name("rglap", _w, _st, "_k1")] = _learner(k=1, steps_scale=_st, lap=_w)                                       # hard neighbour: Laplacian on the 1-NN graph (nearest node only)
        LEARNERS[_name("rglap", _w, _st, "_k1_shuf")] = _learner(k=1, steps_scale=_st, shuffle=True, lap=_w)
    for _w in (0.3, 1.0):   # combinations with the Laplacian
        LEARNERS[f"rglap3_cons{_w:g}_k10_st{_st:g}"] = _learner(steps_scale=_st, lap=3.0, cons=_w)
        LEARNERS[f"rglap3_noise{_w:g}_k10_st{_st:g}"] = _learner(steps_scale=_st, lap=3.0, noise=_w)

# graph-free combinations (R2 of the development screens)
for _st in (2.0, 4.0, 6.0):
    LEARNERS[f"rgcn_c0.3_n1_st{_st:g}"] = _learner(steps_scale=_st, cons=0.3, noise=1.0)
    LEARNERS[f"rgcn_c0.3_n1_rd1_st{_st:g}"] = _learner(steps_scale=_st, cons=0.3, noise=1.0, rdrop=1.0, hdrop=0.2)
for _st in (2.0, 4.0, 6.0):
    LEARNERS[f"rglap3_cons0.3_k10_shuf_st{_st:g}"] = _learner(steps_scale=_st, shuffle=True, lap=3.0, cons=0.3)
