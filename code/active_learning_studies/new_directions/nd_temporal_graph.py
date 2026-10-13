"""Temporal edges for the Laplacian learner (temporal-graph study, branch claude/temporal-graph-*).

The incumbent ``lap3_k10_st4`` = ``aw8_all`` + Laplacian regulariser (lambda 3) on the k = 10 cosine kNN graph of the unlabelled images + 4x head-training steps.  The kNN graph is built from SimCLR
features only; adjacent frames of one growth run (file name <index>_<run>_<temperature>C_<frame>.bmp) are a label-free source of structure the graph ignores.  This module adds *temporal* terms:

  chain<W>        first-difference penalty mu * (s(i) - s(j))^2 on frames of the same run with |index gap| <= W (added to the kNN term)
  decay           the same with edge weights exp(-gap / tau_f) * exp(-|dT| / tau_T)
  tbin            edges between random frames (any run) in the same temperature bin (bin width 20 C, 10 partners per frame)
  curv            second-difference (curvature) penalty mu * (s(i-1) - 2 s(i) + s(i+1))^2 along each run instead of first differences
  knn modes       full (baseline) | none (temporal-only graph) | inter (trajectory-trajectory kNN edges kept only when same run and |index gap| <= 25; edges touching ideal images kept: intersection of the
                  feature kNN graph and a temporal window) | anchor (only kNN edges that touch an ideal image)
  unl             temporal edges only between frames that are not part of any labelled judgment of the current fit
  strict          temporal edges dropped for frames within 25 frame numbers (same run) of any held-out (validation / test) image (red-team time-buffer rule)
  shuf            control: node identities of the temporal edges are permuted within each run (degree and per-run structure kept, correspondence with time destroyed)

Identity safety: every edge connects two graph nodes; graph nodes never include a held-out image (SHA-256 content hash, ``nd_graph_learner.excluded_identities``), so no temporal edge reaches a held-out image.
Temporal adjacency uses the *index* of the file name (consecutive within a run, no gaps); the strict buffer uses the *frame number* of the file name (as the red-team blocked split does).  Temperature is NOT assumed
monotone in time (it is not: run RR220204A is a cooling run); it is only used for edge weights / bins and its role is a hypothesis to be tested.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

import judgment_unit_study as ju_study
import nd_core as core
import nd_graph_learner as G
import rt_core as rt

ACTIVE = list(G.ACTIVE)
EDGE_BATCH = G.EDGE_BATCH
BUFFER = 25            # frame numbers (same run) around every held-out image: strict rule
INTER_WINDOW = 25      # index gap of the "intersection" graph
TBIN_WIDTH, TBIN_PARTNERS = 20, 10
TAU_F, TAU_T = 2.0, 30.0


# ------------------------------------------------------------------------------------------ node metadata
class Nodes:
    """Run / index / temperature / frame number of every node of a ``GraphData`` (-1 for images that are not trajectory frames, e.g. ideal images)."""
    def __init__(self, gd):
        ms = [rt.NAME.search(str(p)) for p in gd.paths]; n = len(ms)
        self.run = np.array([m.group(2) if m else "" for m in ms]); self.idx = np.array([int(m.group(1)) if m else -1 for m in ms])
        self.temp = np.array([int(m.group(3)) if m else -1 for m in ms]); self.frame = np.array([int(m.group(4)) if m else -1 for m in ms]); self.n = n
        self.is_traj = self.idx >= 0; self.runs = sorted({r for r in self.run if r})
        self.by_run = {r: np.where(self.run == r)[0] for r in self.runs}


def chain_edges(nd: Nodes, window: int, decay: bool = False):
    """Pairs of frames of one run with 0 < index gap <= window (holes left by held-out frames are not bridged at gap 1 but are at larger windows).  Returns i, j, weight."""
    I, J, W = [], [], []
    for r in nd.runs:
        ids = nd.by_run[r]; pos = {int(nd.idx[i]): int(i) for i in ids}
        for idx, i in pos.items():
            for o in range(1, window + 1):
                j = pos.get(idx + o)
                if j is None: continue
                I.append(i); J.append(j); W.append(float(np.exp(-o / TAU_F) * np.exp(-abs(int(nd.temp[i]) - int(nd.temp[j])) / TAU_T)) if decay else 1.0)
    return np.array(I, dtype=np.int64), np.array(J, dtype=np.int64), np.array(W, dtype=np.float32)


def tbin_edges(nd: Nodes, seed: int = 0):
    """For every trajectory frame ``TBIN_PARTNERS`` random frames (any run, itself excluded) of the same 20 C temperature bin."""
    rng = np.random.default_rng(seed * 104729 + 3); tj = np.where(nd.is_traj)[0]; bins = nd.temp[tj] // TBIN_WIDTH; pairs = set()
    for b in np.unique(bins):
        members = tj[bins == b]
        for i in members:
            others = members[members != i]
            if not len(others): continue
            for j in rng.choice(others, size=min(TBIN_PARTNERS, len(others)), replace=False): pairs.add((int(min(i, j)), int(max(i, j))))
    arr = np.array(sorted(pairs), dtype=np.int64); return arr[:, 0], arr[:, 1], np.ones(len(arr), dtype=np.float32)


def curvature_triples(nd: Nodes, step: int = 1):
    """(a, b, c) = frames with index idx - step, idx, idx + step of one run (all three present)."""
    T = []
    for r in nd.runs:
        pos = {int(nd.idx[i]): int(i) for i in nd.by_run[r]}
        for idx, b in pos.items():
            a, c = pos.get(idx - step), pos.get(idx + step)
            if a is not None and c is not None: T.append((a, b, c))
    return np.array(T, dtype=np.int64).reshape(-1, 3)


def within_run_permutation(nd: Nodes, seed: int, local: bool = False) -> np.ndarray:
    """Node-id map: trajectory frames are permuted among the frames of their own run, other nodes are fixed.  ``local``: a permutation that only moves a frame by about +-INTER_WINDOW positions
    (positions in index order, jittered by U(-25, 25) and re-ranked): a harder control, the permuted pairs are still moderately similar frames."""
    perm = np.arange(nd.n)
    for k, r in enumerate(nd.runs):
        ids = nd.by_run[r]; rng = np.random.default_rng(seed * 7919 + 101 * k + 11)
        if not local: perm[ids] = rng.permutation(ids)
        else:
            order = ids[np.argsort(nd.idx[ids])]; newpos = np.argsort(np.arange(len(order)) + rng.uniform(-INTER_WINDOW, INTER_WINDOW, len(order))); perm[order] = order[newpos]
    return perm


def near_heldout(nd: Nodes, heldout_paths, gd, buffer: int = BUFFER) -> np.ndarray:
    """Boolean per node: same run and |frame number difference| <= buffer to some held-out image (the held-out images themselves are not nodes)."""
    frames: dict[str, list[int]] = {}
    for p in heldout_paths:
        m = rt.NAME.search(str(p))
        if m: frames.setdefault(m.group(2), []).append(int(m.group(4)))
    out = np.zeros(nd.n, dtype=bool)
    for r, fs in frames.items():
        fs = np.array(sorted(fs)); ids = nd.by_run.get(r, np.array([], dtype=int))
        if len(ids): out[ids] = np.abs(nd.frame[ids][:, None] - fs[None, :]).min(1) <= buffer
    return out


# ------------------------------------------------------------------------------------------ specification and edge sets
def make_spec(name: str, **kw) -> dict:
    spec = dict(name=name, lam=3.0, steps=4.0, knn="full", kind=None, window=0, decay=False, unl=False, strict=False, shuf=False, novel=False, mu=3.0, step=1, tbin_seed=0); spec.update(kw); return spec


def edge_sets(ctx, gd, spec: dict) -> dict:
    """Cached per (ctx, spec): the kNN edge array used for the Laplacian term and the temporal edge set (i, j, w) or triples; strict filtering and shuffling already applied."""
    cache = ctx.__dict__.setdefault("_tg", {}); key = (id(ctx), spec["knn"], spec["kind"], spec["window"], spec["decay"], spec["strict"], spec["shuf"], spec["novel"], spec["step"], spec["tbin_seed"])
    if key in cache: return cache[key]
    nd = cache.get("nodes") or Nodes(gd); cache["nodes"] = nd; knn = gd.edges.numpy()
    if spec["knn"] == "none": knn = knn[:0]
    elif spec["knn"] in ("inter", "anchor"):
        both = nd.is_traj[knn[:, 0]] & nd.is_traj[knn[:, 1]]
        if spec["knn"] == "anchor": keep = ~both
        else: keep = ~both | ((nd.run[knn[:, 0]] == nd.run[knn[:, 1]]) & (np.abs(nd.idx[knn[:, 0]] - nd.idx[knn[:, 1]]) <= INTER_WINDOW))
        if spec["shuf"] and spec["knn"] == "inter":   # control: random trajectory-trajectory kNN edges (same number), anchor edges kept
            rng = np.random.default_rng(ctx.seed * 6151 + 5); tt = np.where(both)[0]; kept_tt = int((keep & both).sum()); chosen = rng.choice(tt, size=kept_tt, replace=False)
            keep = ~both; keep[chosen] = True
        knn = knn[keep]
    out = {"knn": torch.as_tensor(knn, dtype=torch.long), "temporal": None, "triples": None, "nodes": nd}
    kind = spec["kind"]; heldout = None
    if kind in ("chain", "tbin", "curv"):
        if kind == "chain": i, j, w = chain_edges(nd, spec["window"], spec["decay"])
        elif kind == "tbin": i, j, w = tbin_edges(nd, spec["tbin_seed"])
        else: tr = curvature_triples(nd, spec["step"]); i, j, w = tr[:, 0], tr[:, 2], np.ones(len(tr), dtype=np.float32)
        if spec["strict"]:
            bad = near_heldout(nd, rt.heldout_images(ctx), gd); keepm = ~(bad[i] | bad[j] | (bad[tr[:, 1]] if kind == "curv" else False)); i, j, w = i[keepm], j[keepm], w[keepm]
            if kind == "curv": tr = tr[keepm]
        if spec["novel"] and kind != "curv":   # only temporal edges that are NOT edges of the feature kNN graph
            full = {(int(a), int(b)) for a, b in gd.edges.numpy()}; keepm = np.array([(int(min(a, b)), int(max(a, b))) not in full for a, b in zip(i, j)], dtype=bool); i, j, w = i[keepm], j[keepm], w[keepm]
        if spec["shuf"] == "knnrand" and kind != "curv":   # control: the same number of RANDOM kNN edges (generic feature similarity, no time information)
            rng = np.random.default_rng(ctx.seed * 3571 + 7); allk = gd.edges.numpy(); pick = rng.choice(len(allk), size=len(i), replace=False); i, j = allk[pick, 0], allk[pick, 1]
        elif spec["shuf"]:
            perm = within_run_permutation(nd, ctx.seed, local=(spec["shuf"] == "local")); i, j = perm[i], perm[j]
            if kind == "curv": tr = perm[tr]
        if kind == "curv": out["triples"] = torch.as_tensor(tr, dtype=torch.long)
        else: out["temporal"] = (torch.as_tensor(i), torch.as_tensor(j), torch.as_tensor(w))
    cache[key] = out; return out


# ------------------------------------------------------------------------------------------ the learner
def fit_tg(ctx, labelled, seed, gd, spec: dict):
    """``nd_graph_learner.fit_mlp_graph`` (lap term on the chosen kNN edges, same sampling stream as the incumbent) plus the temporal term of ``spec``.  With knn='full' and no temporal term the result is
    identical to ``lap<lam>_k10_st<steps>`` (tested)."""
    featfn = core.identity_features(ctx); node_x = gd.x; dim = gd.x.shape[1]; aw = 8.0
    lr, steps = ctx.params_for(labelled); steps = int(round(steps * spec["steps"])); rows = ju_study.rows_of(ctx.exp, labelled); head = core.new_head(dim, seed); refs, bad = core.anchors(ctx, featfn, True)
    es = edge_sets(ctx, gd, spec); knn = es["knn"]; temporal = es["temporal"]; triples = es["triples"]
    if spec["unl"] and temporal is not None:   # edges only between frames that appear in no labelled judgment of this fit
        used = {str(x) for x in rt.labelled_images(ctx, labelled)}; lab = torch.as_tensor([p in used for p in gd.paths]); keepm = ~(lab[temporal[0]] | lab[temporal[1]]); temporal = tuple(t[keepm] for t in temporal)
    if len(rows):
        xa, xb = featfn(list(rows.resolved_img1)), featfn(list(rows.resolved_img2)); typ = torch.as_tensor(rows.type_idx.to_numpy())
        weight, winner = torch.as_tensor(rows.confidence_weight.to_numpy(), dtype=torch.float32), rows.Winner.to_numpy()
    else: xa = xb = torch.empty(0, dim); typ = torch.empty(0, dtype=torch.long); weight = torch.empty(0); winner = np.array([], dtype=object)
    head.eval(); opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=ctx.exp.cfg.weight_decay); index = torch.arange(len(typ)); masks = {l: torch.as_tensor(winner == l) for l in core.frozen.LABELS}
    cols = {c: core.frozen.TYPE_TO_INDEX[c] for c in refs}; gen = torch.Generator().manual_seed(seed * 1000003 + 17); gent = torch.Generator().manual_seed(seed * 1000003 + 29); mu = spec["mu"]; lam = spec["lam"]
    for _ in range(steps):
        loss = G._bt_terms(head(xa)[index, typ], head(xb)[index, typ], masks, weight) / max(len(typ), 1)
        if len(refs) > 1: loss = loss + aw * G._anchor_loss({n: head(f) for n, f in refs.items()}, cols, None)
        if bad is not None and len(bad): loss = loss + ctx.exp.cfg.bad_anchor_weight * F.relu(head(bad) + 1.0).mean()
        if lam > 0 and len(knn):
            e = knn[torch.randint(len(knn), (EDGE_BATCH,), generator=gen)]; si, sj = head(node_x[e[:, 0]])[:, ACTIVE], head(node_x[e[:, 1]])[:, ACTIVE]
            loss = loss + lam * ((si - sj) ** 2).sum(1).mean()
        if temporal is not None and mu > 0 and len(temporal[0]):
            pick = torch.randint(len(temporal[0]), (EDGE_BATCH,), generator=gent); si, sj = head(node_x[temporal[0][pick]])[:, ACTIVE], head(node_x[temporal[1][pick]])[:, ACTIVE]
            loss = loss + mu * (temporal[2][pick] * ((si - sj) ** 2).sum(1)).mean()
        if triples is not None and mu > 0 and len(triples):
            t = triples[torch.randint(len(triples), (EDGE_BATCH,), generator=gent)]; sa, sb, sc = (head(node_x[t[:, c]])[:, ACTIVE] for c in range(3))
            loss = loss + mu * ((sa - 2 * sb + sc) ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
    return head.eval()


def _learner(spec: dict):
    def learner(ctx, labelled, seed):
        gd = G.graph_for(ctx, 10, False); head = fit_tg(ctx, labelled, seed, gd, spec); featfn = core.identity_features(ctx)
        return lambda test: core.head_d(head, featfn, test)
    return learner


# development variants (names tg_*) and their shuffled-time controls (_shuf); the baseline is ``lap3_k10_st4`` of nd_graph_learner
SPECS = {
    "tg_u1": make_spec("tg_u1", kind="chain", window=1), "tg_u2": make_spec("tg_u2", kind="chain", window=2), "tg_u5": make_spec("tg_u5", kind="chain", window=5),
    "tg_only2": make_spec("tg_only2", knn="none", kind="chain", window=2), "tg_inter": make_spec("tg_inter", knn="inter"),
    "tg_dec5": make_spec("tg_dec5", kind="chain", window=5, decay=True), "tg_tbin": make_spec("tg_tbin", kind="tbin"), "tg_curv": make_spec("tg_curv", kind="curv"),
    "tg_unl2": make_spec("tg_unl2", kind="chain", window=2, unl=True), "tg_anc2": make_spec("tg_anc2", knn="anchor", kind="chain", window=2),
    "tg_novel5": make_spec("tg_novel5", kind="chain", window=5, novel=True),
}
LEARNERS: dict = {}


SPEC_OF: dict = {}


def register(name: str, **kw):
    SPEC_OF[name] = make_spec(name, **kw); LEARNERS[name] = _learner(SPEC_OF[name]); return name


for _n, _s in list(SPECS.items()):
    kw = {k: v for k, v in _s.items() if k != "name"}; register(_n, **kw); register(_n + "_shuf", **{**kw, "shuf": True}); register(_n + "_lshuf", **{**kw, "shuf": "local"}); register(_n + "_krand", **{**kw, "shuf": "knnrand"}); register(_n + "_strict", **{**kw, "strict": True})
# strength / strict / baseline-equivalence helpers (further names are registered by the pre-registered specs, see tg_specs.json handling in tg_run.py)
for _mu in (1.0, 10.0):
    for _n in ("tg_u2", "tg_u5"):
        kw = {k: v for k, v in SPECS[_n].items() if k != "name"}; register(f"{_n}_mu{_mu:g}", **{**kw, "mu": _mu}); register(f"{_n}_mu{_mu:g}_shuf", **{**kw, "mu": _mu, "shuf": True})
# strength scan of the gentler temporal weights (development round 2)
for _n in ("tg_dec5", "tg_curv", "tg_u2", "tg_u5", "tg_novel5"):
    kw = {k: v for k, v in SPECS[_n].items() if k != "name"}
    for _mu in (0.1, 0.3, 1.0):
        if f"{_n}_mu{_mu:g}" not in LEARNERS: register(f"{_n}_mu{_mu:g}", **{**kw, "mu": _mu})
        for _tag, _val in (("shuf", True), ("lshuf", "local"), ("krand", "knnrand")):
            if f"{_n}_mu{_mu:g}_{_tag}" not in LEARNERS and not (_n == "tg_curv" and _tag == "krand"): register(f"{_n}_mu{_mu:g}_{_tag}", **{**kw, "mu": _mu, "shuf": _val})
register("tg_base_check", kind=None)   # = lap3_k10_st4 (identity test)


def edge_counts(ctx, name: str) -> dict:
    """Sizes of the edge sets a registered learner uses in this context (descriptive; 0 for the baseline)."""
    gd = G.graph_for(ctx, 10, False); spec = SPEC_OF[name]; es = edge_sets(ctx, gd, spec)
    return {"n_knn": int(len(es["knn"])), "n_temporal": int(len(es["temporal"][0])) if es["temporal"] is not None else int(len(es["triples"])) if es["triples"] is not None else 0}
