"""Physics-informed representations and learners (branch claude/physics-features-*).

Representation tables (path -> tensor) built from the pixel descriptors of ``pf_extract`` (no learning) and used exactly like ``nd_representation``: head input H and graph metric G are independent,
loss / anchors (weight 8, all ideal images) / schedule (steps x4) / kNN graph (k = 10, lambda = 3) are those of ``lap3_k10_st4`` (``nd_graph_learner.fit_mlp_graph``).
Label-free standardisation: z-scores use only the context's allowed graph nodes (validation / test pair-group images and the seed's outer-test ideals are excluded by SHA-256), clipped at +-5.
Scale rule (as in the representation study): a single block is multiplied by (mean norm of the SimCLR feature) / (mean norm of the block); concatenations are per-image L2-normalised blocks times mean-norm / sqrt(#blocks).

Names:  pfa / pfb / pfc = physics block A (pixel profiles + background + width, 136-d), B (peak table + Fourier + background + width + spacing, 125-d), C (all blocks, 366-d)
        f16 = SimCLR feature with the 16-bit ideal images decoded correctly;  <x>s = concat(<x>, s4);  prd = control: concat(fixed random Gaussian projection of s4 with the dimension of pfa, s4)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

import nd_core as core
import nd_graph_learner as G
import nd_representation as R

PF_DIR = core.HERE / "results" / "pf"
_S: dict = {}


def _load():
    if _S: return _S
    b = torch.load(PF_DIR / "blocks_v1.pt", weights_only=False); paths = b["paths"]; bl = b["blocks"]; _S["paths"] = paths; _S["index"] = {p: i for i, p in enumerate(paths)}
    _S["raw"] = {"pfa": np.hstack([bl["pxprof"], bl["bg"], bl["width"]]), "pfb": np.hstack([bl["peaks"], bl["fourier"], bl["bg"], bl["width"], bl["spacing"]]), "pfc": np.hstack([bl[k] for k in ("pxprof", "uprof", "peaks", "fourier", "bg", "width", "spacing")])}
    s4 = core.feature_cache(); f16 = torch.load(PF_DIR / "f16.pt", weights_only=False); _S["f16"] = {str(k): (f16[str(k)].float() if str(k) in f16 else v.float()) for k, v in s4.items()}
    g = torch.Generator().manual_seed(4242); dim = _S["raw"]["pfa"].shape[1]; _S["proj"] = torch.randn(512, dim, generator=g) / 512 ** .5
    return _S


def physics_table(ctx, name: str) -> dict:
    """z-scored (allowed-node statistics) physics block, clipped, scaled to the baseline norm."""
    s = _load(); nodes = G.graph_for(ctx, 10).paths; idx = [s["index"][p] for p in nodes]; X = s["raw"][name]; mu, sd = X[idx].mean(0), X[idx].std(0) + 1e-6
    Z = torch.as_tensor(np.clip((X - mu) / sd, -5, 5), dtype=torch.float32); scale = R._mean_norm(R._blocks()["s4"], nodes) / float(Z[idx].norm(dim=1).mean()); return {p: Z[i] * scale for p, i in s["index"].items()}


def _concat(tables: list, ref_paths) -> dict:
    sc = R._mean_norm(R._blocks()["s4"], ref_paths) / len(tables) ** .5
    return {p: torch.cat([F.normalize(t[p].float(), dim=0) * sc for t in tables]) for p in tables[0]}


def ctx_rep_pf(ctx, name: str) -> dict:
    store = ctx.__dict__.setdefault("_pf_tab", {})
    if name in store: return store[name]
    s = _load(); nodes = G.graph_for(ctx, 10).paths; s4 = R._blocks()["s4"]
    if name in ("pfa", "pfb", "pfc"): out = physics_table(ctx, name)
    elif name in ("pfas", "pfbs", "pfcs"): out = _concat([ctx_rep_pf(ctx, name[:-1]), s4], nodes)
    elif name == "f16": out = R._scaled_single(s["f16"], sorted(s["f16"])) if False else s["f16"]
    elif name == "f16pfa": out = _concat([ctx_rep_pf(ctx, "pfa"), s["f16"]], nodes)
    elif name == "prd": out = _concat([{p: v @ s["proj"] for p, v in s4.items()}, s4], nodes)
    else: raise KeyError(name)
    store[name] = out; return out


PF_NAMES = {"pfa", "pfb", "pfc", "pfas", "pfbs", "pfcs", "f16", "f16pfa", "prd"}
_orig_ctx_rep = R.ctx_rep


def _ctx_rep(ctx, name):
    return ctx_rep_pf(ctx, name) if name in PF_NAMES else _orig_ctx_rep(ctx, name)


R.ctx_rep = _ctx_rep          # rep_learner looks the function up in the module globals at call time


def q_physics(ctx, name: str, gd, temp: float = 20.0) -> torch.Tensor:
    """Anchor-matching signal: soft class posterior of every graph node from the cosine similarity of its (centred) features ``name`` to the class means of the labelled ideal nodes (references + utility images)."""
    t = ctx_rep_pf(ctx, name) if name in PF_NAMES else R.raw_rep(name); x = torch.stack([t[p].float() for p in gd.paths]); x = x - x.mean(0); xn = F.normalize(x, dim=1)
    cls = torch.stack([xn[[i for i, c in gd.source_class.items() if c == k]].mean(0) for k in G.ACTIVE]); return torch.softmax(xn @ F.normalize(cls, dim=1).T * temp, 1)


def pq_learner(rep: str, rankq: float = 1.0, lap: float = 3.0, steps_scale: float = 4.0, shuffle: bool = False, qrep: str | None = None):
    """lap3_k10_st4 plus a ranking term that pulls s_k(i) - s_k(j) of random node pairs towards sign(q_k(i) - q_k(j)), q from the physics (or other) prototype posterior; head and graph on SimCLR."""
    def learner(ctx, labelled, seed):
        gd = G.graph_for(ctx, 10, shuffle); cache = ctx.features.cache; featfn = core.identity_features(ctx); node_x = gd.x
        q = q_physics(ctx, qrep or rep, gd); head = G.fit_mlp_graph(ctx, labelled, seed, gd, featfn, node_x, node_x.shape[1], lap=lap, rankq=rankq, steps_scale=steps_scale, q=q)
        return lambda test: core.head_d(head, featfn, test)
    return learner


LEARNERS: dict = {}
for _f in ("pfa", "pfb", "pfc"):
    LEARNERS[f"pf_{_f}_h"] = R.rep_learner(_f, graph="s4")            # head = physics alone, graph = SimCLR
    LEARNERS[f"pf_{_f}_hs"] = R.rep_learner(f"{_f}s", graph="s4")     # head = physics (+) SimCLR, graph = SimCLR
    LEARNERS[f"pf_{_f}_g"] = R.rep_learner("s4", graph=_f)            # head = SimCLR, graph from physics alone
    LEARNERS[f"pf_{_f}_gs"] = R.rep_learner("s4", graph=f"{_f}s")     # head = SimCLR, graph from the fused similarity
    LEARNERS[f"pf_{_f}_hgs"] = R.rep_learner(f"{_f}s", graph=f"{_f}s")
    LEARNERS[f"pf_{_f}_hg"] = R.rep_learner(_f, graph=_f)
    for _k in ("h", "hs", "g", "gs", "hgs", "hg"): LEARNERS[f"pf_{_f}_{_k}_shuf"] = None
LEARNERS["pf_prd_hs"] = R.rep_learner("prd", graph="s4")
LEARNERS["pf_f16"] = R.rep_learner("f16", graph="f16")
LEARNERS["pf_f16_h"] = R.rep_learner("f16", graph="s4")
LEARNERS["pf_f16pfa_hg"] = R.rep_learner("f16pfa", graph="f16pfa")
LEARNERS["pf_q_pfa"] = pq_learner("pfa"); LEARNERS["pf_q_s4"] = pq_learner("s4"); LEARNERS["pf_q_pfa_shuf"] = pq_learner("pfa", shuffle=True)
for _f in ("pfa", "pfb", "pfc"):
    for _k, (_h, _g) in {"h": (_f, "s4"), "hs": (f"{_f}s", "s4"), "g": ("s4", _f), "gs": ("s4", f"{_f}s"), "hgs": (f"{_f}s", f"{_f}s"), "hg": (_f, _f)}.items(): LEARNERS[f"pf_{_f}_{_k}_shuf"] = R.rep_learner(_h, graph=_g, shuffle=True)
LEARNERS["pf_prd_hs_shuf"] = R.rep_learner("prd", graph="s4", shuffle=True)
