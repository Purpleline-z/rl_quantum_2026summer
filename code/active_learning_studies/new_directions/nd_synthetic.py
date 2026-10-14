"""Synthetic / corrected typed anchors for the Laplacian learner (synthetic-anchor study).

Base learner = ``lap3_k10_st4`` (``nd_graph_learner.fit_mlp_graph``: aw8_all anchors, Laplacian lambda 3 on the k = 10 cosine kNN graph of unlabelled images, 4x head steps).  ``fit_syn`` reproduces it
exactly when the spec adds nothing (unit-tested) and adds, independently:

  ideal   None | 'fix16' | 'mask'     replace the features of the REAL ideal images (anchors AND graph nodes) by the preprocessing-corrected ones of ``syn_ideal_fix.py``
  syn     dict(bank=<tag>, types=(..), share=<0..1>, nodes=<n per type or 0>, perm=<bool>, per_step=<n per class>)
            synthetic typed anchors from a bank of simulator images (features pre-computed by ``syn_bank.py``): every step, ``per_step`` random bank images per class are added to the anchor ranking loss; their
            total weight is ``share`` of the anchor weight of that class (real anchors keep weight 1 each).  ``nodes`` > 0 also adds that many bank images per type as unlabelled graph nodes (kNN graph rebuilt).
            ``perm``: control - the type labels of the synthetic images are randomly permuted (same images, same weights).
The synthetic images never see a held-out real image; no real held-out information enters the bank (the simulator is calibrated on ideal images / trajectory-frame statistics only: see syn_calibrate.py).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

import judgment_unit_study as ju_study
import nd_core as core
import nd_graph_learner as G
from syn_bank import OUT

ACTIVE = list(G.ACTIVE); EDGE_BATCH = G.EDGE_BATCH
CLASS_OF = {"1x1": "(1 x 1)", "c6x2": "c(6 x 2)", "RT13": "(√13 x √13)", "HTR": "HTR"}


@lru_cache(maxsize=8)
def load_bank(tag: str) -> dict:
    return torch.load(OUT / f"bank_{tag}.pt", weights_only=False)


@lru_cache(maxsize=1)
def ideal_fixed() -> dict:
    import syn_ideal_fix
    d = syn_ideal_fix.load(); return {"paths": d["paths"], "fix16": d["fix16"], "mask": d["mask"], "mask_keep16": d["mask_keep16"], "index": {p: i for i, p in enumerate(d["paths"])}}


@lru_cache(maxsize=1)
def is16_paths() -> frozenset:
    from PIL import Image
    return frozenset(p for p in ideal_fixed()["paths"] if Image.open(p).mode.startswith("I;16"))


def patched_cache(ctx, variant: str) -> dict:
    """Copy of the feature cache with the ideal-image features replaced (the shared cache is never mutated)."""
    key = ("_syn_cache", variant); store = ctx.__dict__.setdefault("_syn", {})
    if key not in store:
        fx = ideal_fixed(); c = dict(ctx.features.cache)
        for p, i in fx["index"].items():
            if p in c: c[p] = fx[variant][i]
        store[key] = c
    return store[key]


def graph_for(ctx, variant, syn_nodes, shuf=False):
    """GraphData of the context (features of the ideal images possibly replaced) + optional synthetic nodes; returns (gd, node_x, edges)."""
    store = ctx.__dict__.setdefault("_syn", {}); key = ("graph", variant, shuf, None if syn_nodes is None else (syn_nodes["bank"], tuple(syn_nodes["types"]), syn_nodes["nodes"]))
    if key in store: return store[key]
    if variant is None: gd = G.graph_for(ctx, 10, shuf)
    else:
        old = ctx.features.cache; ctx.features.cache = patched_cache(ctx, variant)
        try: gd = G.GraphData(ctx, 10, shuffle=ctx.seed if shuf else None)
        finally: ctx.features.cache = old
    node_x, edges = gd.x, gd.edges
    if syn_nodes is not None:
        bank = load_bank(syn_nodes["bank"]); sel = []
        for t in syn_nodes["types"]:
            idx = [i for i, ty in enumerate(bank["typ"]) if ty == t][: syn_nodes["nodes"]]; sel += idx
        xs = bank["feat"][sel].float(); node_x = torch.cat([gd.x, xs]); xn = node_x - gd.mu; xn = xn / xn.norm(dim=1, keepdim=True).clamp_min(1e-9)
        sim = xn @ xn.T; sim.fill_diagonal_(-2.0); nbr = sim.topk(gd.k, dim=1).indices; n = len(node_x); a = torch.zeros(n, n, dtype=torch.bool); a[torch.arange(n)[:, None].expand_as(nbr), nbr] = True; a = a | a.T
        rows, cols = torch.nonzero(a, as_tuple=True); edges = torch.stack([rows, cols], 1)[rows < cols]
    store[key] = (gd, node_x, edges); return store[key]


def _anchor_loss_w(scores, cols, weights):
    values = []
    for p, sp in scores.items():
        for o, so in scores.items():
            if o == p: continue
            m = -F.logsigmoid(sp[:, cols[p]][:, None] - so[:, cols[p]][None, :]); w = weights[p][:, None] * weights[o][None, :]; values.append((m * w).sum() / w.sum())
    return torch.stack(values).mean()


def make_spec(name, **kw):
    s = dict(name=name, ideal=None, ideal_graph="same", shuf_graph=False, drop16=False, extra=None, syn=None, lam=3.0, steps=4.0); s.update(kw); return s


def fit_syn(ctx, labelled, seed, spec):
    """Deterministic: identical (spec name, seed, labelled set) fits are shared inside one context (the two-head learners and the baseline reuse each other's heads)."""
    cache = ctx.__dict__.setdefault("_syn_heads", {}); key = (spec["name"], seed, tuple(map(str, labelled)))
    if key not in cache: cache[key] = _fit_syn(ctx, labelled, seed, spec)
    return cache[key]


def _fit_syn(ctx, labelled, seed, spec):
    syn = spec["syn"]; variant = spec["ideal"]; syn_nodes = {"bank": syn["bank"], "types": syn["types"], "nodes": syn["nodes"]} if syn and syn.get("nodes", 0) else None
    gvar = variant if spec["ideal_graph"] == "same" else spec["ideal_graph"]
    gd, node_x, edges = graph_for(ctx, gvar, syn_nodes, spec.get("shuf_graph", False)); dim = node_x.shape[1]; aw = 8.0
    old = ctx.features.cache
    if variant is not None: ctx.features.cache = patched_cache(ctx, variant)
    try:
        featfn = core.identity_features(ctx); lr, steps = ctx.params_for(labelled); steps = int(round(steps * spec["steps"])); rows = ju_study.rows_of(ctx.exp, labelled)
        head = core.new_head(dim, seed); refs, bad = core.anchors(ctx, featfn, True)
        if spec.get("drop16"):   # remove the 16-bit PNG ideal images (saturated by the loader) from the anchor set
            fx = ideal_fixed(); keep16 = {}
            for c in refs:
                ps = [str(p) for p in list(ctx.exp.references.get(c, [])) + list(ctx.exp.utility_images.get(c, [])) + list(ctx.exp.test_images.get(c, []))]; keep16[c] = torch.tensor([p not in is16_paths() for p in ps])
            refs = {c: f[keep16[c]] for c, f in refs.items()}
        if len(rows):
            xa, xb = featfn(list(rows.resolved_img1)), featfn(list(rows.resolved_img2)); typ = torch.as_tensor(rows.type_idx.to_numpy())
            weight, winner = torch.as_tensor(rows.confidence_weight.to_numpy(), dtype=torch.float32), rows.Winner.to_numpy()
        else: xa = xb = torch.empty(0, dim); typ = torch.empty(0, dtype=torch.long); weight = torch.empty(0); winner = np.array([], dtype=object)
    finally: ctx.features.cache = old
    cols = {c: core.frozen.TYPE_TO_INDEX[c] for c in refs}; head.eval(); opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=ctx.exp.cfg.weight_decay); index = torch.arange(len(typ)); masks = {l: torch.as_tensor(winner == l) for l in core.frozen.LABELS}
    gen = torch.Generator().manual_seed(seed * 1000003 + 17); gens = torch.Generator().manual_seed(seed * 1000003 + 41)
    pools = {}; wsyn = {}; extra_fixed = {}
    if spec["extra"]:   # additional anchors = the SAME real ideal images with other preprocessing (e.g. direct beam masked), all used every step with total weight share/(1-share) of the real anchors of the class
        fx = ideal_fixed(); var = spec["extra"]["variant"]; ex_share = spec["extra"]["share"]
        for c, ps in [(c, list(ctx.exp.references.get(c, [])) + list(ctx.exp.utility_images.get(c, [])) + list(ctx.exp.test_images.get(c, []))) for c in refs]:
            extra_fixed[c] = (torch.stack([fx[var][fx["index"][str(p)]] for p in ps]).float(), (ex_share / (1 - ex_share)) * 1.0)
    if syn:
        bank = load_bank(syn["bank"]); typs = np.array(bank["typ"]); sty = np.array(bank["style"]); labels = typs.copy()
        if syn.get("style"): keep = sty == syn["style"]
        else: keep = np.ones(len(typs), bool)
        if syn.get("perm"):   # control: permute the type labels of the synthetic images (same images)
            perm = np.random.default_rng(ctx.seed * 6007 + 3).permutation(len(labels)); labels = labels[perm]
        for t in syn["types"]:
            ids = np.where((labels == t) & keep)[0]
            if len(ids): pools[CLASS_OF[t]] = bank["feat"][ids].float()
        per = syn.get("per_step", 16); s = syn["share"]
        if s <= 0: pools = {}                                         # nodes-only variant: no synthetic anchors
        for c in pools: n_real = len(refs[c]); wsyn[c] = (s / (1 - s)) * n_real / per
    for _ in range(steps):
        loss = G._bt_terms(head(xa)[index, typ], head(xb)[index, typ], masks, weight) / max(len(typ), 1)
        if len(refs) > 1:
            if syn and syn.get("replace"):   # diagnostic: the REAL anchors are replaced by synthetic ones (no real anchor image is used)
                scores = {n: head(pools[n][torch.randint(len(pools[n]), (syn.get("per_step", 16),), generator=gens)]) for n in refs if n in pools}
                if len(scores) > 1: loss = loss + aw * G._anchor_loss(scores, cols, None)
            elif not pools and not extra_fixed: loss = loss + aw * G._anchor_loss({n: head(f) for n, f in refs.items()}, cols, None)
            else:
                scores, ws = {}, {}
                for n, f in refs.items():
                    w = torch.ones(len(f))
                    if n in extra_fixed: f = torch.cat([f, extra_fixed[n][0]]); w = torch.cat([w, torch.full((len(extra_fixed[n][0]),), extra_fixed[n][1])])
                    if n in pools:
                        pick = torch.randint(len(pools[n]), (per,), generator=gens); f = torch.cat([f, pools[n][pick]]); w = torch.cat([w, torch.full((per,), wsyn[n])])
                    scores[n], ws[n] = head(f), w
                loss = loss + aw * _anchor_loss_w(scores, cols, ws)
        if bad is not None and len(bad): loss = loss + ctx.exp.cfg.bad_anchor_weight * F.relu(head(bad) + 1.0).mean()
        if spec["lam"] > 0:
            e = edges[torch.randint(len(edges), (EDGE_BATCH,), generator=gen)]; si, sj = head(node_x[e[:, 0]])[:, ACTIVE], head(node_x[e[:, 1]])[:, ACTIVE]; loss = loss + spec["lam"] * ((si - sj) ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
    return head.eval()


def _learner(spec):
    def learner(ctx, labelled, seed):
        head = fit_syn(ctx, labelled, seed, spec); featfn = core.identity_features(ctx); return lambda test: core.head_d(head, featfn, test)
    return learner


SPEC_OF: dict = {}; LEARNERS: dict = {}


def register(name, **kw):
    SPEC_OF[name] = make_spec(name, **kw); LEARNERS[name] = _learner(SPEC_OF[name]); return name


register("sy_base_check")                       # == lap3_k10_st4
register("sy_fix16", ideal="fix16"); register("sy_mask", ideal="mask")
register("sy_mask_shuf", ideal="mask", shuf_graph=True)                      # control: same masked features, graph node identities permuted
register("sy_mask16", ideal="mask_keep16")                                   # direct beam masked in the 8-bit ideal images, the 16-bit PNGs keep their loader features
register("sy_maskanc", ideal="mask", ideal_graph=None)                       # masked ideal images as ANCHORS only (graph nodes keep the loader features)
register("sy_both50", extra=dict(variant="mask", share=0.5))                 # original anchors + masked copies of the same ideal images (weight share 0.5)
register("sy_both33", extra=dict(variant="mask", share=1 / 3))


# ------------------------------------------------------------------------------------------ two-head learners (inference-time combination)
def anchor_distance_fn(ctx):
    """Cosine distance (centred, normalised loader features, rt_a1b convention) of an image to the nearest typed ideal image (references, utility, outer-test ideal images) - uses only the image itself."""
    store = ctx.__dict__.setdefault("_syn", {})
    if "anc" not in store:
        c = ctx.features.cache; paths = sorted(c); mu = torch.stack([c[p].float() for p in paths]).mean(0)
        anc = sorted({str(p) for t in (ctx.exp.references, ctx.exp.utility_images, ctx.exp.test_images) for ps in t.values() for p in ps}); A = torch.stack([c[p].float() for p in anc]) - mu; A = A / A.norm(dim=1, keepdim=True).clamp_min(1e-9)
        def dist(images):
            x = torch.stack([c[str(p)].float() for p in images]) - mu; x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-9); return (1 - (x @ A.T).max(1).values).numpy()
        store["anc"] = dist
    return store["anc"]


def _two_head_learner(spec_a, spec_b, mode, thr=0.03, seed_b_offset=0):
    """mode: 'mean' (average of the two logit gaps), 'gate' (second head where BOTH test images are farther than ``thr`` from every typed anchor, first head elsewhere)."""
    def learner(ctx, labelled, seed):
        ha = fit_syn(ctx, labelled, seed, spec_a); hb = fit_syn(ctx, labelled, seed + seed_b_offset, spec_b); fa, fb = core.identity_features(ctx), core.identity_features(ctx)
        def predict(test):
            da = core.head_d(ha, fa, test); db = core.head_d(hb, fb, test)
            if mode == "mean": return 0.5 * (da + db)
            dist = anchor_distance_fn(ctx); far = np.minimum(dist(test.img1), dist(test.img2)) > thr; return np.where(far, db, da)
        return predict
    return learner


def register_two(name, spec_a, spec_b, mode, **kw):
    LEARNERS[name] = _two_head_learner(spec_a, spec_b, mode, **kw); SPEC_OF[name] = dict(name=name, a=spec_a["name"], b=spec_b["name"], mode=mode, **kw); return name
