"""Representation study: does a different frozen representation lift the tuned Laplacian learner `lap3_k10_st4`?

Everything is label-free and frozen (protocol extension, see PREREGISTRATION_R1.md): other layers of the SimCLR encoder, the cached ImageNet-ResNet-18 features of
image_representation_analysis (its weights are not available offline), deterministic views (mirror, centre / corner crops of 90 % side) averaged at feature level, centring + whitening + PCA estimated on the
graph's allowed nodes (validation/test pair-group images and outer-test ideal images are excluded by SHA-256, exactly as `nd_graph_learner.GraphData`).

A learner = (head representation H, graph representation G).  The head, the reference anchors, the Laplacian penalty's node inputs and the evaluation images all use H; the kNN graph (k = 10, cosine on centred
features) is built from G.  Loss, per-budget schedule (steps x4 for `st4`), anchors (weight 8, all ideal images) are those of `nd_graph_learner.fit_mlp_graph` (unit test: representation `s4` reproduces `lap3_k10_st4`).

Scale: blocks are put on the scale of the baseline feature so that the repository's per-budget learning rate stays meaningful: a single block is multiplied by (mean norm of SimCLR final) / (mean norm of the block);
concatenated blocks are L2-normalised per image and multiplied by (mean norm of SimCLR final) / sqrt(#blocks).
Controls: `_shuf` shuffled graph (node identities permuted, degrees kept); `_cap` capacity control = SimCLR-final block concatenated with a fixed random 512-d Gaussian projection of it (same concatenation recipe),
`ens5` = five heads (independent initialisations) of the baseline, scores averaged (free variance reduction).
"""
from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import nd_core as core
import nd_graph_learner as G
import nd_rep_build as B

REP_ROOT = core.HERE.parent / "image_representation_analysis" / "results" / "representation_exploration"
_STORE: dict = {}


def _blocks() -> dict:
    """Raw per-image feature tables (path string -> tensor) of every available frozen block."""
    if _STORE: return _STORE
    s4 = core.feature_cache(); _STORE["s4"] = {str(k): v.float() for k, v in s4.items()}
    layers = torch.load(core.HERE / "results" / "feature_layers_all.pt", weights_only=False); _STORE["s2"], _STORE["s3"] = layers["l2"], layers["l3"]
    root = str(core.single_study.DATA.resolve()) + "/"; man = pd.read_csv(REP_ROOT / "manifest.csv"); arr = torch.as_tensor(np.load(REP_ROOT / "cache" / "imagenet_resnet18.npy")).float()
    _STORE["i4"] = {root + "original data/" + p.split("original data/")[-1]: v for p, v in zip(man.path, arr)}
    views = torch.load(B.REP_DIR / "simclr_views.pt", weights_only=False); _STORE["mir"] = {p: (v[0] + v[1]) / 2 for p, v in views.items()}; _STORE["mc"] = {p: v.mean(0) for p, v in views.items()}
    assert set(_STORE["i4"]) >= set(_STORE["s4"]) and set(_STORE["s3"]) >= set(_STORE["s4"]) and set(views) >= set(_STORE["s4"])
    g = torch.Generator().manual_seed(2026); _STORE["proj"] = torch.randn(512, 512, generator=g) / 512 ** .5   # fixed random projection used by the capacity control
    return _STORE


def _stack(table: dict, paths) -> torch.Tensor:
    return torch.stack([table[p] for p in paths]).float()


def _mean_norm(table: dict, paths) -> float:
    return float(_stack(table, paths).norm(dim=1).mean())


def _scaled_single(table: dict, ref_paths) -> dict:
    s = _mean_norm(_blocks()["s4"], ref_paths) / _mean_norm(table, ref_paths); return {p: v.float() * s for p, v in table.items()}


def _scaled_concat(tables: list, ref_paths) -> dict:
    s = _mean_norm(_blocks()["s4"], ref_paths) / len(tables) ** .5
    return {p: torch.cat([F.normalize(t[p].float(), dim=0) * s for t in tables]) for p in tables[0]}


_RAW: dict = {}


def raw_rep(name: str) -> dict:
    """Context-independent representation tables (path -> tensor)."""
    if name in _RAW: return _RAW[name]
    b = _blocks(); ref = sorted(b["s4"])
    if name == "s4": out = b["s4"]
    elif name in ("s3", "i4", "mir", "mc"): out = _scaled_single(b[name], ref)
    elif name == "s34": out = _scaled_concat([b["s3"], b["s4"]], ref)
    elif name == "is": out = _scaled_concat([b["i4"], b["s4"]], ref)
    elif name == "s3is": out = _scaled_concat([b["s3"], b["i4"], b["s4"]], ref)
    elif name == "cap": out = _scaled_concat([b["s4"], {p: v @ b["proj"].T for p, v in b["s4"].items()}], ref)          # capacity control for concatenations: S4 + random projection of S4
    elif name == "capis": out = _scaled_concat([b["s4"], {p: v @ b["proj"].T for p, v in b["s4"].items()}], ref)
    else: raise KeyError(name)
    _RAW[name] = out; return out


def whiten_rep(table: dict, node_paths: list, dim: int, ref_norm: float) -> dict:
    """Centre, PCA-whiten (top `dim` components) with statistics of the allowed graph nodes only (label-free, held-out images excluded), rescaled to the baseline norm."""
    x = _stack(table, node_paths); mu = x.mean(0); u, s, v = torch.pca_lowrank(x - mu, q=dim, center=False, niter=4); comp = v[:, :dim]; sd = (s[:dim] / (len(x) ** .5)).clamp_min(1e-6)
    out = {p: ((t.float() - mu) @ comp) / sd for p, t in table.items()}; k = ref_norm / float(torch.stack([out[p] for p in node_paths]).norm(dim=1).mean()); return {p: z * k for p, z in out.items()}


def _proxy(ctx, name: str, table: dict):
    """A shallow copy of the context whose feature cache is the representation `name` (stable object per (ctx, name), so graph caches work)."""
    store = ctx.__dict__.setdefault("_rp", {})
    if name not in store:
        c = copy.copy(ctx); c.__dict__["_graphs"] = {}; c.features = SimpleNamespace(cache=table, get=lambda paths, t=table: _stack(t, [str(p) for p in paths])); store[name] = c
    return store[name]


def ctx_rep(ctx, name: str) -> dict:
    """Representation table for this context (whitening variants depend on the allowed node set)."""
    if name.startswith("w"):
        key = ("w", name); store = ctx.__dict__.setdefault("_rp_tab", {})
        if key not in store:
            node_paths = G.graph_for(ctx, 10).paths; ref = _mean_norm(_blocks()["s4"], node_paths); store[key] = whiten_rep(_blocks()["s4"], node_paths, int(name[1:]), ref)
        return store[key]
    return raw_rep(name)


def rep_learner(head: str = "s4", graph: str | None = None, lap: float = 3.0, steps_scale: float = 4.0, shuffle: bool = False, ensemble: int = 1, lr_scale: float = 1.0):
    graph = graph or head

    def learner(ctx, labelled, seed):
        ht = ctx_rep(ctx, head); gt = ctx_rep(ctx, graph); cg = _proxy(ctx, f"G:{graph}", gt) if graph != "s4" else ctx
        gd = G.graph_for(cg, 10, shuffle)
        featfn = (lambda paths, t=ht: _stack(t, [str(p) for p in paths])); node_x = _stack(ht, gd.paths); dim = node_x.shape[1]; heads = []
        for m in range(ensemble): heads.append(G.fit_mlp_graph(ctx, labelled, seed + 1000 * m, gd, featfn, node_x, dim, lap=lap, steps_scale=steps_scale, lr_scale=lr_scale))
        def predict(test):
            ds = [core.head_d(h, featfn, test) for h in heads]; return np.mean(ds, axis=0)
        return predict
    return learner


LEARNERS: dict = {}
LEARNERS["rp_base"] = rep_learner("s4")                                    # == lap3_k10_st4 (unit test)
LEARNERS["rp_base_shuf"] = rep_learner("s4", shuffle=True)
LEARNERS["rp_ens5"] = rep_learner("s4", ensemble=5)
for _name in ("s3", "s34", "i4", "is", "s3is", "mir", "mc", "w64", "w128"):
    LEARNERS[f"rp_{_name}"] = rep_learner(_name); LEARNERS[f"rp_{_name}_shuf"] = rep_learner(_name, shuffle=True)
LEARNERS["rp_cap"] = rep_learner("cap"); LEARNERS["rp_cap_shuf"] = rep_learner("cap", shuffle=True)
for _g, _h in (("is", "s4"), ("s4", "is"), ("s34", "s4"), ("s4", "s34"), ("mc", "s4"), ("w64", "s4")):
    LEARNERS[f"rp_g{_g}_h{_h}"] = rep_learner(_h, graph=_g); LEARNERS[f"rp_g{_g}_h{_h}_shuf"] = rep_learner(_h, graph=_g, shuffle=True)
