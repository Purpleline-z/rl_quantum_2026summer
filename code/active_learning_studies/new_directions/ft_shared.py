"""Family S of the CPU fine-tuning study: ONE label-free (plus typed-anchor) adaptation of layer 4, shared by every seed and split.

Identity safety by construction: the adaptation sees only images that occur in no labelled pair group of the study (the content hash of every group image is excluded) plus the typed ideal images
(which every learner of the protocol already uses as anchors, ``aw8_all``).  No pair judgment is used.  The adapted layer 4 re-encodes all 1278 images (``results/ft/shared_<name>.pt``); the incumbent recipe
``lap3_k10_st4`` then runs on the adapted features (``fts_<name>``), optionally with the kNN graph of the *original* features (``fts_<name>_g0``) and with the shuffled-graph control (``_shuf``).
Terms (all on mini-batches, feature-map augmentation; bf16 autocast): typed-anchor ranking (aw), Laplacian over the kNN graph of the adaptation images (lap; graph from the frozen features, or shuffled for the control),
augmentation consistency of the head scores (cons), InfoNCE between two augmented views (nce), L2-SP towards the pretrained layer 4 (sp) and feature drift to the frozen feature (drift).  Usage:
  python3 ft_shared.py train NAME [NAME ...]      # trains the named variants (idempotent: skips existing files)
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import nd_core as core
import nd_finetune as FT
import nd_graph_learner as G
import nd_representation as RP

OUT = core.HERE / "results" / "ft"


@dataclass(frozen=True)
class SSpec:
    steps: int = 300; enc_lr: float = 3e-5; head_lr: float = 1e-3; head_pre: int = 400; last_block_only: bool = False
    aw: float = 8.0; lap: float = 3.0; cons: float = 0.0; nce: float = 0.0; tau: float = 0.2; sp: float = 1.0; drift: float = 0.0
    graph: str = "frozen"        # 'frozen' | 'current' | 'shuf'
    rebuild: int = 50; ab: int = 12; eb: int = 48; nb: int = 64; aug: bool = True; seed: int = 0


def adaptation_images():
    """(trajectory frames outside every labelled pair group and content twin of one, all typed ideal images by class)."""
    import tempfile
    cache = core.feature_cache()
    with tempfile.TemporaryDirectory() as sc: ctx = core.make_ctx(core.FT_DEV_SEEDS[0], "A", sc, cache)
    group_imgs = {str(x) for g in ctx.exp.groups.values() for r in g.itertuples() for x in (r.resolved_img1, r.resolved_img2)}
    ideal = {c: [str(p) for p in list(ps) + list(ctx.exp.utility_images.get(c, [])) + list(ctx.exp.test_images.get(c, []))] for c, ps in ctx.exp.references.items()}
    ideal_all = {p for ps in ideal.values() for p in ps}; hashes = G.content_hashes(sorted(cache)); banned = {hashes[p] for p in group_imgs}
    frames = [p for p in sorted(cache) if "Trajectories" in p and p not in group_imgs and hashes[p] not in banned and p not in ideal_all]
    return frames, ideal, ctx.exp.bad_paths


def train_shared(spec: SSpec, verbose: bool = True):
    t0 = time.monotonic(); frames, ideal, bad = adaptation_images(); ideal_all = sorted({p for ps in ideal.values() for p in ps}); nodes = frames + ideal_all
    assert not (set(frames) & set(ideal_all))
    cache = core.feature_cache(); x0 = torch.stack([cache[p].float() for p in nodes]); torch.manual_seed(spec.seed); gen = torch.Generator().manual_seed(spec.seed + 101)
    edges, _ = FT.knn_edges(x0, 10); n = len(nodes)
    if spec.graph == "shuf":
        perm = torch.as_tensor(np.random.default_rng(spec.seed * 7919 + 11).permutation(n)); edges = perm[edges]
    node3 = FT.l3_stack(nodes); ref3 = {c: FT.l3_stack(ps) for c, ps in ideal.items()}; cols = {c: core.frozen.TYPE_TO_INDEX[c] for c in ref3}
    bad3 = FT.l3_stack([str(p) for p in bad]) if bad else None; active = list(G.ACTIVE)
    layer4 = FT.fresh_layer4(); theta0 = [p.detach().clone() for p in layer4.parameters()]; params = FT._train_params(layer4, spec.last_block_only)
    head = core.new_head(512, spec.seed); head.eval(); proj = nn.Linear(512, 128, bias=False) if spec.nce > 0 else None
    # head pre-training on the frozen features (anchors + Laplacian), so the encoder is adapted against a sensible head
    opt_h = torch.optim.AdamW(head.parameters(), lr=spec.head_lr, weight_decay=1e-4); ref0 = {c: torch.stack([cache[p].float() for p in ps]) for c, ps in ideal.items()}
    for _ in range(spec.head_pre):
        loss = spec.aw * G._anchor_loss({c: head(v) for c, v in ref0.items()}, cols, None)
        if spec.lap > 0:
            e = edges[torch.randint(len(edges), (256,), generator=gen)]; loss = loss + spec.lap * ((head(x0[e[:, 0]])[:, active] - head(x0[e[:, 1]])[:, active]) ** 2).sum(1).mean()
        opt_h.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt_h.step()
    groups = [{"params": list(head.parameters()), "lr": spec.head_lr, "weight_decay": 1e-4}, {"params": params, "lr": spec.enc_lr, "weight_decay": 0.0}]
    if proj is not None: groups.append({"params": list(proj.parameters()), "lr": 1e-3, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups); cur = edges
    enc = lambda m: FT.encode(layer4, FT.augment(m, gen) if spec.aug else m)
    for step in range(spec.steps):
        if spec.graph == "current" and step % spec.rebuild == 0:
            with torch.no_grad(): cur = FT.knn_edges(FT.encode(layer4, node3))[0]
        scores = {}
        for c, m in ref3.items(): s = torch.randperm(len(m), generator=gen)[:spec.ab]; scores[c] = head(enc(m[s]))
        loss = spec.aw * G._anchor_loss(scores, cols, None)
        if bad3 is not None and len(bad3): loss = loss + 0.10 * F.relu(head(enc(bad3[torch.randperm(len(bad3), generator=gen)[:8]])) + 1.0).mean()
        if spec.lap > 0:
            e = cur[torch.randint(len(cur), (spec.eb,), generator=gen)]; loss = loss + spec.lap * ((head(enc(node3[e[:, 0]]))[:, active] - head(enc(node3[e[:, 1]]))[:, active]) ** 2).sum(1).mean()
        if spec.cons > 0 or spec.nce > 0 or spec.drift > 0:
            u = torch.randperm(n, generator=gen)[:spec.nb]; m = node3[u]; v1, v2 = enc(m), enc(m)
            if spec.cons > 0: loss = loss + spec.cons * ((head(v1)[:, active] - head(v2)[:, active]) ** 2).sum(1).mean()
            if spec.nce > 0:
                z1, z2 = F.normalize(proj(v1), dim=1), F.normalize(proj(v2), dim=1); lg = z1 @ z2.T / spec.tau; tgt = torch.arange(len(u)); loss = loss + spec.nce * 0.5 * (F.cross_entropy(lg, tgt) + F.cross_entropy(lg.T, tgt))
            if spec.drift > 0: loss = loss + spec.drift * ((v1 - x0[u]) ** 2).sum(1).mean()
        if spec.sp > 0: loss = loss + spec.sp * sum(((p - p0) ** 2).sum() for p, p0 in zip(layer4.parameters(), theta0) if p.requires_grad)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
        if verbose and step % 50 == 0: print(f"  step {step} loss {float(loss):.4f} ({time.monotonic() - t0:.0f}s)", flush=True)
    layer4.eval(); paths = sorted(cache); feats = FT.encode_all(layer4, paths)
    drift = float(sum(((p - p0) ** 2).sum() for p, p0 in zip(layer4.parameters(), theta0)) ** .5); cos = float(F.cosine_similarity(feats, torch.stack([cache[p].float() for p in paths]), dim=1).mean())
    return {"paths": paths, "feats": feats, "info": {"weight_drift": drift, "cos_to_frozen": cos, "seconds": time.monotonic() - t0, "n_adaptation_images": n, "n_frames": len(frames)}}


BASE = SSpec()
VARIANTS: dict[str, SSpec] = {
    "A": BASE,                                                   # anchors + frozen-graph Laplacian
    "A_lr1e-4": replace(BASE, enc_lr=1e-4), "A_s1000": replace(BASE, steps=1000),
    "A_sp0": replace(BASE, sp=0.0), "A_b1": replace(BASE, last_block_only=True),
    "A_shuf": replace(BASE, graph="shuf"),                       # control: the same adaptation with the shuffled graph
    "Aonly": replace(BASE, lap=0.0),                             # anchors only
    "AC": replace(BASE, cons=1.0), "AN": replace(BASE, nce=0.3), "ACN": replace(BASE, cons=1.0, nce=0.3),
    "Acur": replace(BASE, graph="current"),
    "AD": replace(BASE, drift=1.0),
}


def feature_file(name: str):
    return OUT / f"shared_{name}.pt"


def table_of(name: str) -> dict:
    blob = torch.load(feature_file(name), weights_only=False); return {p: v for p, v in zip(blob["paths"], blob["feats"])}


def learners() -> dict:
    """``fts_<name>`` (adapted features for head, anchors and graph), ``fts_<name>_g0`` (graph from the original features), ``_shuf`` (shuffled-graph control) for every trained variant on disk."""
    out = {}
    for f in sorted(OUT.glob("shared_*.pt")):
        name = f.stem[len("shared_"):]; key = f"ft_{name}"
        def raw(name=name, key=key):
            if "s4" not in RP._RAW: RP._RAW["s4"] = {str(k): v.float() for k, v in core.feature_cache().items()}
            if key not in RP._RAW:
                t = table_of(name); s4 = RP._RAW["s4"]; k = float(torch.stack([s4[p] for p in t]).norm(dim=1).mean() / torch.stack(list(t.values())).norm(dim=1).mean()); RP._RAW[key] = {p: v.float() * k for p, v in t.items()}   # norm matching as in the representation study
            return RP._RAW[key]
        def make(head_key, graph_key, shuffle, raw=raw, key=key):
            def learner(ctx, labelled, seed):
                raw(); return RP.rep_learner(head_key, graph=graph_key, shuffle=shuffle)(ctx, labelled, seed)
            return learner
        out[f"fts_{name}"] = make(key, key, False); out[f"fts_{name}_shuf"] = make(key, key, True); out[f"fts_{name}_g0"] = make(key, "s4", False); out[f"fts_{name}_g0_shuf"] = make(key, "s4", True)
    return out


if __name__ == "__main__":
    torch.set_num_threads(1); OUT.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 2 and sys.argv[1] == "train":
        for name in sys.argv[2:]:
            if feature_file(name).exists(): print("exists", name); continue
            print("training", name, VARIANTS[name], flush=True); r = train_shared(VARIANTS[name]); torch.save(r, feature_file(name)); print(name, r["info"], flush=True)
