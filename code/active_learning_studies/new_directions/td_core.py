"""Transductive and forward-in-time protocol (branch claude/transductive-20261014).  PROTOCOL EXTENSION, stated in PREREGISTRATION_TD1.md.

In deployment the lab scores NEW frames of a growth run while the run is being grown; the images of those frames exist (unlabelled) when they are scored and the past of the run is labelled.
The repository's protocols differ on two points: held-out images never enter the graph (identity rule, "inductive") and the split is random.  This module adds

* graph node-set modes (which images the kNN graph / Laplacian regulariser / score propagation may see; features only, never labels):
    std         the repository's rule: every cached image except the images of the validation / test groups and the outer-test ideal images (+ the blocked-split buffer zone)
    trans       std + the images of the validation / test groups as unlabelled nodes (transductive)
    forward cuts only:
    ind         past only: std minus every frame later than the cut (deployment-inductive: the future does not exist yet)
    trans_test  ind + the images of the test groups (the minimal transductive set: only the frames being scored)
    trans_fut   = trans: std + test images (all frames of the dataset that exist after the cut are available unlabelled)
* the calendar time of images / groups and the forward-in-time contexts (``make_forward_ctx``): runs are ordered by date (RR220204A < RR220206A < RR220411A, folder dates 2022-02-04, 02-06, 04-11),
  an image's time is (run rank, frame number), a pair group's time is the time of its LATER image (the group exists once both images do); all 168 groups can be assigned a time (every image
  of every group is a trajectory frame), 104 of them are cross-run pairs.
* the learners (``fit_td``: the incumbent ``lap3_k10_st4`` loss with a free node set, recency weights, pseudo-labelled rows, type-posterior loss) and test-time score propagation.
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
RUN_RANK = {"RR220204A": 0, "RR220206A": 1, "RR220411A": 2}

_prev_excluded = G.excluded_identities


def _patched_excluded(ctx):
    override = getattr(ctx, "_td_excl", None)
    return override if override is not None else _prev_excluded(ctx)


G.excluded_identities = _patched_excluded     # GraphData looks the function up in nd_graph_learner's globals (as rt_blocked does)


# ------------------------------------------------------------------------------------------ time
def img_time(p):
    """(run rank, frame number) of a trajectory frame, None for ideal images."""
    r, f = rt.run_of(p), rt.frame_of(p)
    return None if r is None else (RUN_RANK[r], f)


def group_images(ctx, g):
    r = ctx.exp.groups[g].iloc[0]; return str(r.resolved_img1), str(r.resolved_img2)


def group_time(ctx, g):
    ts = [img_time(p) for p in group_images(ctx, g)]; assert all(t is not None for t in ts); return max(ts)


def calendar_order(ctx) -> list:
    """All pair groups sorted by group time (ties by id)."""
    return sorted(ctx.exp.groups, key=lambda g: (group_time(ctx, g), g))


# ------------------------------------------------------------------------------------------ exclusion sets and graphs
def _hashes(paths) -> set:
    paths = sorted({str(p) for p in paths}); h = G.content_hashes(paths); return {h[p] for p in paths}


def heldout_paths(ctx) -> set:
    return {x for g in list(ctx.validation) + list(ctx.test) for x in group_images(ctx, g)}


def exclusion_sets(ctx) -> dict:
    """mode -> set of content hashes the graph must not contain (computed once per ctx)."""
    cache = ctx.__dict__.get("_td_sets")
    if cache is not None: return cache
    saved = getattr(ctx, "_td_excl", None); ctx._td_excl = None
    std = set(_prev_excluded(ctx)); held = _hashes(heldout_paths(ctx)); outer = _hashes({str(p) for ps in ctx.exp.test_images.values() for p in ps})
    out = {"std": std, "trans": (std - held) | outer}
    # the blocked-split buffer zone (rt_blocked) is part of std and stays excluded in the transductive mode
    zone = getattr(ctx, "_extra_excluded_hashes", set()); out["trans"] = out["trans"] | zone
    fwd = getattr(ctx, "forward", None)
    if fwd is not None:
        allp = sorted(ctx.features.cache); future = [p for p in allp if img_time(p) is not None and img_time(p) > fwd["cut_time"]]; fh = _hashes(future)
        out["ind"] = std | fh; out["trans_test"] = ((std | fh) - held) | outer | zone
        out["trans_fut"] = out["trans"]
        for w, groups in (("FA", fwd["near"]), ("FB", fwd["far"])):   # causal transduction: only the frames that exist when the window is scored (time <= the time of the window's last group) are nodes
            end = group_time(ctx, groups[-1]); late = _hashes([p for p in allp if img_time(p) is not None and img_time(p) > end]); out[f"trans_win_{w}"] = late | outer | zone
    else: out["ind"] = std; out["trans_test"] = out["trans"]; out["trans_fut"] = out["trans"]; out["trans_win"] = out["trans"]
    ctx._td_excl = saved; ctx._td_sets = out; return out


def td_graph(ctx, mode: str, k: int = 10, shuffle: bool = False) -> "G.GraphData":
    """kNN graph of the node set of ``mode`` (cached).  ``shuffle``: the repository's control (node identities of the adjacency permuted; topology and degrees kept)."""
    cache = ctx.__dict__.setdefault("_td_graphs", {}); key = (mode, k, shuffle)
    if key not in cache:
        sets = exclusion_sets(ctx); ctx._td_excl = sets[mode]
        try: cache[key] = G.GraphData(ctx, k, shuffle=ctx.seed if shuffle else None)
        finally: ctx._td_excl = None
    return cache[key]


# ------------------------------------------------------------------------------------------ forward-in-time contexts
CUTS = (12, 24, 40, 54, 70, 90)       # numbers of calendar-ordered groups in the past (2 per run: 204A holds groups 1-29, 206A 30-64, 411A 65-168)
N_NEAR, N_FAR = 40, 34                # sizes of the two evaluation windows after the cut (the test sizes of Splits A and B)
N_INITIAL = 10


def make_forward_ctx(seed: int, cut: int, scratch, cache, jitter: int = 2, n_near: int = N_NEAR, n_far: int = N_FAR):
    """Context whose labelled set is the past of the calendar (groups ``cut`` +- seed jitter), whose test windows are the next ``n_near`` ("FA") and the following ``n_far`` ("FB") groups.

    Training groups that share an image with a test group are dropped (the repository's identity rule).  ``ctx.pool`` = the past judgments beyond the oldest ``N_INITIAL`` groups (so that the head schedule is
    the one of the number of revealed judgments); ``labelled = ctx.initial + ctx.pool`` (all past judgments)."""
    ctx = core.make_ctx(seed, "B", scratch, cache); exp = ctx.exp; order = calendar_order(ctx)
    rng = np.random.default_rng(seed * 7919 + cut); n_train = int(cut + rng.integers(-jitter, jitter + 1)); assert n_train + n_near + n_far <= len(order)
    past = order[:n_train]; near = order[n_train:n_train + n_near]; far = order[n_train + n_near:n_train + n_near + n_far]; test_images = {p for g in near + far for p in group_images(ctx, g)}
    train = [g for g in past if not (set(group_images(ctx, g)) & test_images)]; initial, pool = train[:N_INITIAL], train[N_INITIAL:]
    ctx.initial_groups, ctx.pool_groups, ctx.validation, ctx.test = initial, pool, [], near + far
    ctx.initial, ctx.pool = ju_study.judgments_of(exp, initial), ju_study.judgments_of(exp, pool); ctx.pool_index = {j: i for i, j in enumerate(ctx.pool)}; ctx.info = {}
    for g in set(initial) | set(pool):
        for pos, r in enumerate(exp.groups[g].itertuples()):
            ctx.info[(g, pos)] = {"img1": r.resolved_img1, "img2": r.resolved_img2, "type_idx": int(r.type_idx), "decisive": r.Winner in ("1", "2"), "winner": r.Winner}
    cut_time = group_time(ctx, order[n_train - 1]); rank = {g: i for i, g in enumerate(order)}
    ctx.forward = {"cut": cut, "n_train": n_train, "n_train_kept": len(train), "cut_time": cut_time, "near": near, "far": far, "order_rank": rank, "n_future_groups": len(order) - n_train - n_near - n_far}
    used = {x for j in ctx.initial + ctx.pool for x in (ctx.info[j]["img1"], ctx.info[j]["img2"])}; assert not (test_images & {str(u) for u in used}) and not (test_images & used), "forward split leaks"
    assert all(group_time(ctx, g) <= cut_time for g in train) and all(group_time(ctx, g) >= cut_time for g in near + far)
    return ctx


def forward_tests(ctx):
    """Decisive test judgments of the near ("FA") and far ("FB") windows plus the group id of every judgment (for group-level bootstraps)."""
    out = {}
    for tag, groups in (("FA", ctx.forward["near"]), ("FB", ctx.forward["far"])):
        t = core.TestSet(ctx, groups); t.group = [g for g in groups for w in ctx.exp.groups[g].Winner if w in ("1", "2")]; assert len(t.group) == len(t.k); out[tag] = t
    return out


# ------------------------------------------------------------------------------------------ the learner
def fit_td(ctx, labelled, seed, gd, lam: float = 3.0, steps_scale: float = 4.0, aw: float = 8.0, rowmult=None, extra=None, rankq: float = 0.0, q=None, lap_w=None, cons: float = 0.0, cons_dropout: float = 0.3, focus_mask=None, focus_frac: float = 0.0):
    """``nd_graph_learner.fit_mlp_graph`` (lap3_k10_st4 with lam = 3, steps_scale = 4; identical sampling stream, tested) with: ``rowmult`` = per-judgment multiplier of the confidence weight (dict
    judgment -> factor, e.g. recency), ``extra`` = pseudo-labelled rows (xa, xb, type_idx, weight, winner), ``rankq``/``q`` = type-posterior ranking loss on random node pairs."""
    featfn = core.identity_features(ctx); node_x = gd.x; dim = gd.x.shape[1]
    lr, steps = ctx.params_for(labelled); steps = int(round(steps * steps_scale)); judg = sorted(set(labelled)); rows = ju_study.rows_of(ctx.exp, labelled); head = core.new_head(dim, seed); refs, bad = core.anchors(ctx, featfn, True)
    tw = torch.ones(len(ACTIVE)) if not lap_w else torch.tensor([float(lap_w.get(t, 1.0)) for t in ACTIVE])
    if len(rows):
        xa, xb = featfn(list(rows.resolved_img1)), featfn(list(rows.resolved_img2)); typ = torch.as_tensor(rows.type_idx.to_numpy())
        weight, winner = torch.as_tensor(rows.confidence_weight.to_numpy(), dtype=torch.float32), rows.Winner.to_numpy()
        if rowmult is not None: weight = weight * torch.as_tensor([rowmult[j] for j in judg], dtype=torch.float32)
    else: xa = xb = torch.empty(0, dim); typ = torch.empty(0, dtype=torch.long); weight = torch.empty(0); winner = np.array([], dtype=object)
    n_real = len(typ)
    if extra is not None:
        xa, xb = torch.cat([xa, extra[0]]), torch.cat([xb, extra[1]]); typ = torch.cat([typ, extra[2]]); weight = torch.cat([weight, extra[3]]); winner = np.concatenate([winner, extra[4]])
    head.eval(); opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=ctx.exp.cfg.weight_decay); index = torch.arange(len(typ)); masks = {l: torch.as_tensor(winner == l) for l in core.frozen.LABELS}
    cols = {c: core.frozen.TYPE_TO_INDEX[c] for c in refs}; gen = torch.Generator().manual_seed(seed * 1000003 + 17)
    focus_edges = None; focus_nodes = None; n_f = 0
    if focus_mask is not None and focus_frac > 0 and gd is not None:   # transductive focus: a share of the sampled edges / node pairs touches the images that are about to be scored
        inc = focus_mask[gd.edges[:, 0]] | focus_mask[gd.edges[:, 1]]; focus_edges = gd.edges[inc]; focus_nodes = torch.where(focus_mask)[0]; n_f = int(round(EDGE_BATCH * focus_frac)) if len(focus_edges) else 0
    for _ in range(steps):
        loss = G._bt_terms(head(xa)[index, typ], head(xb)[index, typ], masks, weight) / max(n_real, 1)
        if len(refs) > 1: loss = loss + aw * G._anchor_loss({n: head(f) for n, f in refs.items()}, cols, None)
        if bad is not None and len(bad): loss = loss + ctx.exp.cfg.bad_anchor_weight * F.relu(head(bad) + 1.0).mean()
        if lam > 0 and gd is not None:
            e = gd.edges[torch.randint(gd.n_edges, (EDGE_BATCH - n_f,), generator=gen)]
            if n_f: e = torch.cat([e, focus_edges[torch.randint(len(focus_edges), (n_f,), generator=gen)]])
            si, sj = head(node_x[e[:, 0]])[:, ACTIVE], head(node_x[e[:, 1]])[:, ACTIVE]
            loss = loss + lam * (((si - sj) ** 2) * tw).sum(1).mean()
        if rankq > 0 and q is not None:
            i, j = torch.randint(len(node_x), (EDGE_BATCH,), generator=gen), torch.randint(len(node_x), (EDGE_BATCH,), generator=gen)
            if n_f and focus_nodes is not None and len(focus_nodes): i = torch.cat([i[: EDGE_BATCH - n_f], focus_nodes[torch.randint(len(focus_nodes), (n_f,), generator=gen)]])
            si, sj = head(node_x[i])[:, ACTIVE], head(node_x[j])[:, ACTIVE]; dq = q[i] - q[j]; w = dq.abs(); w = w / (w.mean() + 1e-9)
            loss = loss + rankq * (w * F.softplus(-torch.sign(dq) * (si - sj))).mean()
        if cons > 0:   # graph-free control (as nd_graph_learner.fit_mlp_graph): scores of random nodes must be stable under feature dropout
            u = torch.randint(len(node_x), (EDGE_BATCH,), generator=gen); xu = node_x[u]; keep = (torch.rand(xu.shape, generator=gen) > cons_dropout).float() / (1 - cons_dropout)
            loss = loss + cons * ((head(xu)[:, ACTIVE] - head(xu * keep)[:, ACTIVE]) ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
    return head.eval()


# ------------------------------------------------------------------------------------------ test-time score propagation
def row_normalised(gd) -> torch.Tensor:
    a = gd.adj.float(); return a / a.sum(1, keepdim=True).clamp_min(1.0)


def propagate_scores(gd, head, alpha: float, steps: int = 20, shuffle_seed=None) -> torch.Tensor:
    """S* = (1 - alpha) sum_t alpha^t P^t S0 (truncated at ``steps``), S0 = head scores of every node, P = row-normalised kNN adjacency.  ``shuffle_seed``: control (adjacency node identities permuted)."""
    with torch.no_grad(): s0 = head(gd.x)
    p = row_normalised(gd)
    if shuffle_seed is not None: perm = torch.as_tensor(np.random.default_rng(shuffle_seed * 7919 + 11).permutation(p.shape[0])); p = p[perm][:, perm]
    s = s0.clone()
    for _ in range(steps): s = (1 - alpha) * s0 + alpha * (p @ s)
    return s


def d_from_node_scores(gd, s, head, test, k_attach: int = 10, cache=None, shuffle_seed=None) -> np.ndarray:
    """Logit gap of the own-type output for the test judgments from (propagated) node scores; an image that is not a node is attached: mean of the propagated scores of its ``k_attach`` nearest nodes
    mixed 50/50 with its own head score (only used by graph modes that exclude test images)."""
    def score(paths):
        out = torch.empty(len(paths), s.shape[1]); ext = [i for i, p in enumerate(paths) if str(p) not in gd.index]
        for i, p in enumerate(paths):
            if str(p) in gd.index: out[i] = s[gd.index[str(p)]]
        if ext:
            nb = gd.neighbours_of([str(paths[i]) for i in ext], cache)
            if shuffle_seed is not None: nb = torch.as_tensor(np.random.default_rng(shuffle_seed * 31 + len(ext)).integers(0, len(gd.paths), nb.shape))   # control: random nodes instead of the nearest ones
            own = torch.stack([cache[str(paths[i])] for i in ext]).float()
            with torch.no_grad(): own_s = head(own)
            out[ext] = 0.5 * own_s + 0.5 * s[nb].mean(1)
        return out
    a, b = score(list(test.img1)), score(list(test.img2)); return (a - b)[torch.arange(len(test.k)), torch.as_tensor(test.k)].numpy()
