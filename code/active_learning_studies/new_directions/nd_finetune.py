"""CPU partial fine-tuning of the SimCLR encoder (study ``claude/finetune-cpu-20261014``).

Idea: cache the layer-3 spatial maps (256 x 14 x 14) of all 1278 images once (``python3 nd_finetune.py build``; ~257 MB, not committed) and train only layer 4 (+ the reward head) on top of them
(bf16 autocast, AMX; ~7 ms per image forward+backward on one core).  Everything else is the protocol of ``nd_graph_learner`` (judgment-unit endpoint, per-budget head schedule, all ideal images as anchors
with weight 8, Laplacian term on the kNN graph of the unlabelled images, identity safety through ``GraphData``).

Two families:

* **P (per-fit refinement)** ``ftp_*``: stage 1 trains the head exactly as the incumbent ``lap3_k10_st4`` (frozen encoder); stage 2 continues head + layer 4 on mini-batches of the full loss
  (pair + anchor aw=8 + Laplacian lam=3 [+ consistency]) with L2-SP towards the pretrained layer 4 and feature-map augmentation.  The matched control ``ftp_frozen_*`` runs the *same* stage 2
  (same sampling, same augmentation, same number of steps) with the layer-4 learning rate 0.
* **S (shared label-free encoder)** ``fts_*``: layer 4 adapted once on images that occur in NO labelled pair group (840 trajectory frames + all ideal images; anchors + Laplacian + consistency + drift
  penalty, no pair labels) -> identity-safe for every seed and split by construction; the incumbent recipe then runs on the adapted features (``ft_shared.py``).
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import nd_core as core
import nd_graph_learner as G
import judgment_unit_study as ju_study
import nd_representation as RP

L3_FILE = core.HERE / "results" / "ft" / "l3_cache.pt"
ACTIVE = G.ACTIVE
_L3: dict = {}


# ------------------------------------------------------------------------------------------ encoder pieces and the layer-3 cache
def load_encoder() -> nn.Sequential:
    """The SimCLR ResNet-18 encoder of the study (``BTModel.encoder``: conv1, bn1, relu, maxpool, layer1..4, avgpool), eval mode."""
    model = core.frozen.pipeline.BTModel(core.HERE.parents[1] / "classifier2" / "simclr_encoder_checkpoint" / "simclr_resnet18_encoder.pth")
    return model.encoder.eval()


_L4: list = []


def fresh_layer4() -> nn.Module:
    """A copy of the pretrained layer 4 (the checkpoint is read once per process)."""
    if not _L4: _L4.append(load_encoder()[7])
    return copy.deepcopy(_L4[0]).eval()


@torch.no_grad()
def build_l3_cache() -> None:
    enc = load_encoder(); trunk, layer4 = nn.Sequential(*list(enc.children())[:7]), enc[7]; tf = core.frozen.pipeline.transform(); cache = core.feature_cache(); paths = sorted(cache); out = {}
    for i in range(0, len(paths), 16):
        chunk = paths[i:i + 16]; x = torch.stack([tf(Image.open(p).convert("L")) for p in chunk]); m = trunk(x)
        f = layer4(m).mean((2, 3)); err = float((f - torch.stack([cache[p] for p in chunk]).float()).abs().max()); assert err < 2e-3, f"layer4(l3) differs from the cached feature: {err}"
        for p, v in zip(chunk, m): out[p] = v.clone()
    L3_FILE.parent.mkdir(parents=True, exist_ok=True); torch.save(out, L3_FILE); print("saved", len(out), "layer-3 maps ->", L3_FILE)


def l3_table() -> dict:
    if not _L3: _L3.update(torch.load(L3_FILE, weights_only=False))
    return _L3


def l3_stack(paths) -> torch.Tensor:
    t = l3_table(); return torch.stack([t[str(p)] for p in paths])


# ------------------------------------------------------------------------------------------ feature-map augmentation
def augment(x: torch.Tensor, gen: torch.Generator, flip: bool = True, shift: int = 1, jitter: float = 0.1) -> torch.Tensor:
    """Per-sample augmentation of layer-3 maps: horizontal flip, random shift by up to ``shift`` cells (replicate padding), global intensity jitter and per-channel gain jitter."""
    b, c, h, w = x.shape; out = x
    if flip:
        m = torch.rand(b, generator=gen) < 0.5; out = torch.where(m[:, None, None, None], out.flip(3), out)
    if shift > 0:
        p = F.pad(out, (shift,) * 4, mode="replicate"); dy = torch.randint(0, 2 * shift + 1, (b,), generator=gen); dx = torch.randint(0, 2 * shift + 1, (b,), generator=gen)
        out = torch.stack([p[i, :, dy[i]:dy[i] + h, dx[i]:dx[i] + w] for i in range(b)])
    if jitter > 0:
        s = 1 + jitter * (2 * torch.rand(b, 1, 1, 1, generator=gen) - 1); g = 1 + 0.5 * jitter * torch.randn(b, c, 1, 1, generator=gen); out = out * s * g
    return out


def encode(layer4: nn.Module, maps: torch.Tensor, amp: bool = True, chunk: int = 128) -> torch.Tensor:
    outs = []
    for i in range(0, len(maps), chunk):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp): f = layer4(maps[i:i + chunk]).mean((2, 3))
        outs.append(f.float())
    return torch.cat(outs) if outs else torch.empty(0, 512)


@torch.no_grad()
def encode_all(layer4: nn.Module, paths) -> torch.Tensor:
    layer4.eval(); return encode(layer4, l3_stack(paths), amp=False, chunk=64)


def knn_edges(feats: torch.Tensor, k: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric cosine kNN edge list of mean-centred features (as ``GraphData``)."""
    xc = feats - feats.mean(0); xn = xc / xc.norm(dim=1, keepdim=True).clamp_min(1e-9); sim = xn @ xn.T; sim.fill_diagonal_(-2.0); nbr = sim.topk(k, dim=1).indices; n = len(feats)
    a = torch.zeros(n, n, dtype=torch.bool); a[torch.arange(n)[:, None].expand_as(nbr), nbr] = True; a = a | a.T; r, c = torch.nonzero(a, as_tuple=True); return torch.stack([r, c], 1)[r < c], a


# ------------------------------------------------------------------------------------------ the per-fit refinement (family P)
@dataclass(frozen=True)
class Spec:
    steps: int = 40                 # stage-2 mini-batch steps
    enc_lr: float = 3e-5            # Adam lr of the trained layer-4 parameters (0 = frozen-encoder control)
    head_lr_mult: float = 0.2       # stage-2 head lr relative to the per-budget schedule lr
    last_block_only: bool = False   # train only layer4[1] (block 2 of layer 4)
    sp: float = 1.0                 # L2-SP coefficient (sum of squared parameter drift)
    aug: bool = False               # feature-map augmentation of every image of the step
    graph: str = "frozen"           # 'frozen' (kNN graph of the cached features) | 'current' (rebuilt from the current embedding every `rebuild` steps) | 'shuf' (shuffled frozen graph control)
    rebuild: int = 20
    lap: float = 3.0; aw: float = 8.0; pair_w: float = 1.0; cons: float = 0.0   # loss weights (pair_w = 0: anchor-only fine-tuning; cons = consistency across two augmentations of the graph-batch nodes)
    pb: int = 32; ab: int = 12; eb: int = 24; bb: int = 8                      # batch sizes: judgments, anchors per class, edges, bad images
    amp: bool = True
    wise: float = 1.0               # WiSE-FT weight interpolation after training: theta = theta0 + wise * (theta - theta0)  (1 = no interpolation)


def _train_params(layer4: nn.Module, last_only: bool):
    for p in layer4.parameters(): p.requires_grad_(False)
    mod = layer4[1] if last_only else layer4; ps = list(mod.parameters())
    for p in ps: p.requires_grad_(True)
    return ps


def _anchor_loss(scores: dict, cols: dict) -> torch.Tensor:
    return G._anchor_loss(scores, cols, None)


def fit_refine(ctx, labelled, seed: int, spec: Spec, shuffle_graph: bool = False):
    """Stage 1 = incumbent head (frozen encoder); stage 2 = joint refinement.  Returns (layer4, head, info)."""
    t0 = time.monotonic(); gd = G.graph_for(ctx, 10, shuffle_graph or spec.graph == "shuf"); cache = ctx.features
    featfn = core.identity_features(ctx); node_x = gd.x; dim = node_x.shape[1]
    head = G.fit_mlp_graph(ctx, labelled, seed, gd, featfn, node_x, dim, lap=spec.lap, steps_scale=4.0); t1 = time.monotonic()
    layer4 = fresh_layer4(); theta0 = [p.detach().clone() for p in layer4.parameters()]
    if spec.steps <= 0: return layer4, head, {"t_stage1": t1 - t0, "t_stage2": 0.0}
    lr, _ = ctx.params_for(labelled); rows = ju_study.rows_of(ctx.exp, labelled); gen = torch.Generator().manual_seed(seed * 1000003 + 29)
    enc_params = _train_params(layer4, spec.last_block_only) if spec.enc_lr > 0 else []
    if spec.enc_lr <= 0:
        for p in layer4.parameters(): p.requires_grad_(False)
    groups = [{"params": list(head.parameters()), "lr": lr * spec.head_lr_mult, "weight_decay": ctx.exp.cfg.weight_decay}]
    if enc_params: groups.append({"params": enc_params, "lr": spec.enc_lr, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups)
    # fixed tensors: the maps of the judgments, the anchors (per class) and the bad images, the graph nodes
    n = len(rows); typ = torch.as_tensor(rows.type_idx.to_numpy()) if n else torch.empty(0, dtype=torch.long)
    a3 = l3_stack(rows.resolved_img1) if n else None; b3 = l3_stack(rows.resolved_img2) if n else None
    weight = torch.as_tensor(rows.confidence_weight.to_numpy(), dtype=torch.float32) if n else torch.empty(0); winner = rows.Winner.to_numpy() if n else np.array([], dtype=object)
    masks = {l: torch.as_tensor(winner == l) for l in core.frozen.LABELS}
    ref_paths = {c: [str(p) for p in list(ps) + list(ctx.exp.utility_images.get(c, [])) + list(ctx.exp.test_images.get(c, []))] for c, ps in ctx.exp.references.items()}
    ref3 = {c: l3_stack(ps) for c, ps in ref_paths.items()}; cols = {c: core.frozen.TYPE_TO_INDEX[c] for c in ref3}
    bad3 = l3_stack([str(p) for p in ctx.exp.bad_paths]) if ctx.exp.bad_paths else None
    node3 = l3_stack(gd.paths); edges = gd.edges; active = list(ACTIVE); bad_w = ctx.exp.cfg.bad_anchor_weight
    cur_edges = edges
    def enc(maps):
        if spec.aug: maps = augment(maps, gen)
        return encode(layer4, maps, amp=spec.amp)
    sched_t = 0.0
    for step in range(spec.steps):
        if spec.graph == "current" and step % spec.rebuild == 0:
            with torch.no_grad(): cur_edges, _ = knn_edges(encode(layer4, node3, amp=spec.amp, chunk=128))
        layer4.eval(); head.eval(); terms = []
        loss = torch.zeros(())
        if spec.pair_w > 0 and n:
            sel = torch.randperm(n, generator=gen)[:spec.pb] if n > spec.pb else torch.arange(n); idx = torch.arange(len(sel))
            fa, fb = enc(a3[sel]), enc(b3[sel]); ra, rb = head(fa)[idx, typ[sel]], head(fb)[idx, typ[sel]]; ms = {l: m[sel] for l, m in masks.items()}
            t = (-F.logsigmoid(ra - rb) * ms["1"] - F.logsigmoid(rb - ra) * ms["2"] + (ra - rb).abs() * ms["tie"] + (F.relu(ra) + F.relu(rb)) * ms["not_apply"])
            loss = loss + spec.pair_w * (t * weight[sel]).sum() / len(sel)
        if spec.aw > 0 and len(ref3) > 1:
            scores = {}
            for c, m in ref3.items():
                s = torch.randperm(len(m), generator=gen)[:spec.ab]; scores[c] = head(enc(m[s]))
            loss = loss + spec.aw * _anchor_loss(scores, cols)
        if bad3 is not None and len(bad3):
            s = torch.randperm(len(bad3), generator=gen)[:spec.bb]; loss = loss + bad_w * F.relu(head(enc(bad3[s])) + 1.0).mean()
        if spec.lap > 0:
            e = cur_edges[torch.randint(len(cur_edges), (spec.eb,), generator=gen)] if spec.graph != "shuf" else edges[torch.randint(len(edges), (spec.eb,), generator=gen)]
            fi, fj = enc(node3[e[:, 0]]), enc(node3[e[:, 1]]); si, sj = head(fi)[:, active], head(fj)[:, active]; loss = loss + spec.lap * ((si - sj) ** 2).sum(1).mean()
        if spec.cons > 0:
            u = torch.randint(len(node3), (spec.eb,), generator=gen); x = node3[u]; v1, v2 = enc(x), enc(x)
            loss = loss + spec.cons * ((head(v1)[:, active] - head(v2)[:, active]) ** 2).sum(1).mean()
        if spec.sp > 0 and enc_params:
            tp = {id(p) for p in enc_params}; loss = loss + spec.sp * sum(((p - p0) ** 2).sum() for p, p0 in zip(layer4.parameters(), theta0) if id(p) in tp)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
    if spec.wise < 1.0 and enc_params:
        with torch.no_grad():
            for p, p0 in zip(layer4.parameters(), theta0): p.copy_(p0 + spec.wise * (p - p0))
    layer4.eval(); head.eval()
    return layer4, head, {"t_stage1": t1 - t0, "t_stage2": time.monotonic() - t1,
                          "drift": float(sum(((p.detach() - p0) ** 2).sum() for p, p0 in zip(layer4.parameters(), theta0)) ** .5) if theta0 else 0.0}


def refine_learner(spec: Spec, shuffle_graph: bool = False):
    def learner(ctx, labelled, seed):
        layer4, head, info = fit_refine(ctx, labelled, seed, spec, shuffle_graph)
        def predict(test):
            with torch.no_grad(): fa, fb = encode_all(layer4, test.img1), encode_all(layer4, test.img2); d = (head(fa) - head(fb))[torch.arange(len(test.k)), torch.as_tensor(test.k)].numpy()
            predict.info = info; return d
        predict.info = info; return predict
    return learner


BASE = Spec()
VARIANTS: dict[str, Spec] = {
    "ftp_base": BASE,                                                     # layer 4 + head, enc lr 3e-5, 40 steps, L2-SP 1, augmentation, frozen graph
    "ftp_frozen": replace(BASE, enc_lr=0.0),                              # matched control: identical stage 2, encoder frozen
    "ftp_lr1e-5": replace(BASE, enc_lr=1e-5), "ftp_lr1e-4": replace(BASE, enc_lr=1e-4),
    "ftp_sp0": replace(BASE, sp=0.0), "ftp_sp10": replace(BASE, sp=10.0),
    "ftp_b1": replace(BASE, last_block_only=True),
    "ftp_s100": replace(BASE, steps=100), "ftp_s100_frozen": replace(BASE, steps=100, enc_lr=0.0),
    "ftp_cur": replace(BASE, graph="current"),
    "ftp_shuf": replace(BASE, graph="shuf"),                              # shuffled-graph control (Laplacian edges random but degree-preserving)
    "ftp_anchor": replace(BASE, pair_w=0.0),                              # anchor + Laplacian only (no pair labels in stage 2)
    "ftp_cons": replace(BASE, cons=1.0),
    "ftp_noaug": replace(BASE, aug=False),
}
LEARNERS = {k: refine_learner(v) for k, v in VARIANTS.items()}
LEARNERS["ftp_shuf"] = refine_learner(VARIANTS["ftp_shuf"], shuffle_graph=True)


def parse(name: str):
    """``ftp:key=value;key=value`` -> learner with those Spec fields changed on top of ``ftp_base`` (``shuf=1`` additionally shuffles the graph; ``graph=...`` as in Spec); values are parsed as bool / int / float / str."""
    assert name.startswith("ftp:"); kv = {}; shuf = False
    for item in name[4:].split(";"):
        if not item: continue
        k, v = item.split("="); 
        if k == "shuf": shuf = v in ("1", "true", "True"); continue
        cur = getattr(BASE, k); kv[k] = (v in ("1", "true", "True")) if isinstance(cur, bool) else type(cur)(v)
    return refine_learner(replace(BASE, **kv), shuffle_graph=shuf or kv.get("graph") == "shuf")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "build": torch.set_num_threads(4); build_l3_cache()
