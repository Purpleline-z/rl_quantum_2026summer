"""Cross-run transfer (study CR, branch claude/cross-run-*): a NEW growth run R arrives, many old judgments exist, only b new judgments of R will be labelled.

Assignment rule (images are assigned to runs by file name; a pair group "touches" R if at least one of its two images belongs to R):
  * target groups   = groups touching R (within-run groups and mixed groups R-vs-other run);
  * source groups   = groups with NO image in R (source labels = all judgments of these groups);
  * test groups     = a subset of the target groups, split kind  A: a random 30 % of the target groups (stream seed, run);
                                                              B: forward in time inside R: target groups whose R-images all lie in the last third of the run (frame index / run length >= 2/3)
                       (the remaining target groups whose R-images all lie in the first two thirds form the target label pool; groups mixing early and late R-images are dropped);
  * target pool     (kind A) = target groups not in the test set;
  * image-disjointness (as in the repository protocol): every group (source or target) that shares an image with a test group is dropped;
  * graph           = all trajectory + ideal images (cached features) minus the images of the test groups and the outer-test ideal images (SHA-256) -- R's own frames are in the graph, unlabelled
                       (transductive); source judgments never involve an image of R.
The b target judgments are drawn as a nested prefix of a seeded permutation of the target pool's judgments (decisive or not, as in the repository protocol), so b = 0 < 5 < 10 < 20 < 40 are nested.

Learners (all with the recipe of ``lap3_k10_st4``: aw8_all head, Laplacian lambda = 3 on the k = 10 graph, 4x head steps; ``cr_*`` names):
  cr_src        source labels only (b-independent)
  cr_tgt        the b target labels only (b = 0: anchors + graph only)
  cr_pool       source + target labels pooled
  cr_poolw<w>   pooled, target rows weighted x w
  cr_ft         fine-tune the source head on the target labels with an L2 penalty to the source weights
  cr_prior      function-space prior: pooled-free; the source head's scores on the run's frames are the shrinkage target (squared distance, target labels free to deviate)
  cr_gprop      the prior target on R's frames is the source head's scores propagated over the kNN graph (label spreading, source frames clamped); ``_shuf`` = propagation over a shuffled graph
  cr_iw         source judgments importance-weighted by the feature-space proximity of their images to R's frames, pooled with the target labels
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import judgment_unit_study as ju_study
import nd_core as core
import nd_graph_learner as G
import nd_learners as L
import ge_harsh as H

RUNS = ("RR220204A", "RR220206A", "RR220411A")
BUDGETS = (0, 5, 10, 20, 40)
TEST_FRACTION = 0.30
ACTIVE = G.ACTIVE
SCHEDULE_LR_CAP = 0.001   # the repository schedule uses lr 0.003 only for its 60-judgment entry; source-containing label sets here are far larger and the dev check (dev_srcsched) showed lr 0.001 is +0.007 AUC / +0.012 accuracy better for the source-only head
EDGE_BATCH = G.EDGE_BATCH


# ------------------------------------------------------------------------------------------ the fold
class Fold:
    """One (seed, run, split kind): groups, judgments, held-out test set, target frames, nested target label draws; installs itself into ``ctx``."""
    def __init__(self, ctx, run: str, kind: str):
        gi = H.group_images(ctx); self.run, self.kind, self.ctx = run, kind, ctx; lengths = L._run_lengths()
        in_run = lambda x: (H.meta(x) or {}).get("run") == run
        late = lambda x: in_run(x) and H.meta(x)["frame"] / lengths[run] >= 2 / 3
        touching = sorted(g for g, (a, b) in gi.items() if in_run(a) or in_run(b)); source_all = sorted(g for g in gi if g not in touching)
        rng = random.Random(ctx.seed * 1009 + 17 * RUNS.index(run) + (0 if kind == "A" else 5))
        if kind == "A":
            test = sorted(rng.sample(touching, int(round(TEST_FRACTION * len(touching))))); pool_all = [g for g in touching if g not in test]
        elif kind == "B":
            test = [g for g in touching if all(late(x) for x in gi[g] if in_run(x))]; pool_all = [g for g in touching if all(not late(x) for x in gi[g] if in_run(x))]
        else: raise ValueError(kind)
        test_imgs = {x for g in test for x in gi[g]}; ok = lambda g: not (set(gi[g]) & test_imgs)
        self.test_groups = test; self.source_groups = [g for g in source_all if ok(g)]; self.pool_groups = [g for g in pool_all if ok(g)]
        self.counts = {"touching": len(touching), "test": len(test), "source": len(self.source_groups), "source_dropped": len(source_all) - len(self.source_groups), "pool": len(self.pool_groups), "pool_dropped": len(pool_all) - len(self.pool_groups)}
        ctx.initial_groups, ctx.pool_groups, ctx.validation, ctx.test = [], list(self.pool_groups), [], list(test)
        self.source = ju_study.judgments_of(ctx.exp, self.source_groups); pool = ju_study.judgments_of(ctx.exp, self.pool_groups)
        ctx.initial, ctx.pool = [], pool; ctx.pool_index = {j: i for i, j in enumerate(pool)}; ctx.info = {}
        for g in self.source_groups + self.pool_groups:
            for pos, r in enumerate(ctx.exp.groups[g].itertuples()): ctx.info[(g, pos)] = {"img1": r.resolved_img1, "img2": r.resolved_img2, "type_idx": int(r.type_idx), "decisive": r.Winner in ("1", "2"), "winner": r.Winner}
        ctx.__dict__.pop("_graphs", None)
        used = {str(x) for g in self.source_groups + self.pool_groups for x in gi[g]}; assert not (used & {str(x) for x in test_imgs}), "test image reachable from a labelled group"
        assert not any(in_run(x) for g in self.source_groups for x in gi[g]), "source group touches the held-out run"
        order = list(pool); random.Random(ctx.seed * 7919 + 31 * RUNS.index(run) + (0 if kind == "A" else 3)).shuffle(order); self.target_order = order
        self.frames = sorted(p for p in ctx.features.cache if in_run(p) and (p not in {str(x) for x in test_imgs}))   # R's frames available as unlabelled nodes
        self.test = core.TestSet(ctx); self.gi = gi
        self.test_group_of = self._test_groups()

    def _test_groups(self):
        """group id of every decisive test judgment (same order as TestSet)."""
        rows = self.ctx.exp.rows_for(self.test_groups); rows = rows[rows.Winner.isin(["1", "2"])].reset_index(drop=True)
        return list(rows["pair_id"]) if "pair_id" in rows else list(rows.index)

    def target(self, b: int) -> list:
        return self.target_order[:b]


# ------------------------------------------------------------------------------------------ training
def _sched(ctx, n_rows: int):
    return ju_study.schedule_for(ctx.schedule, n_rows)


def _flat(head):
    return torch.cat([p.reshape(-1) for p in head.parameters()])


def train_head(ctx, gd, seed: int, src: list, tgt: list, *, lap: float = 3.0, steps_scale: float = 4.0, tgt_weight: float = 1.0, src_weights=None, init=None, l2: float = 0.0, ref=None,
               prior_idx=None, prior_target=None, prior_w: float = 0.0, lr=None, steps=None, aw: float = 8.0, featfn=None, node_shift=None, lap_extra: float = 0.0, lap_extra_nodes=None):
    """The ``lap3_k10_st4`` head fit with transfer switches.  src / tgt: judgment lists; ``init``: a head to start from (copy); ``l2`` x sum (theta - theta_ref)^2 with ``ref`` the reference head;
    ``prior_idx`` / ``prior_target`` / ``prior_w``: squared distance of the scores of nodes ``prior_idx`` to ``prior_target`` (function-space shrinkage); ``src_weights`` per source judgment (sorted order)."""
    featfn = featfn or core.identity_features(ctx); node_x = gd.x if node_shift is None else gd.x + node_shift; dim = node_x.shape[1]; n = len(set(src)) + len(set(tgt))
    if lr is None or steps is None: lr0, steps0 = _sched(ctx, n)
    lr0 = min(lr0, SCHEDULE_LR_CAP); lr = lr0 if lr is None else lr; steps = int(round(steps0 * steps_scale)) if steps is None else steps
    rs = ju_study.rows_of(ctx.exp, src); rt_ = ju_study.rows_of(ctx.exp, tgt)
    if len(rs) and len(rt_): rows = __import__("pandas").concat([rs, rt_], ignore_index=True)
    else: rows = rs if len(rs) else rt_
    head = core.new_head(dim, seed)
    if init is not None: head.load_state_dict({k: v.clone() for k, v in init.state_dict().items()})
    refs, bad = core.anchors(ctx, featfn, True)
    if len(rows):
        xa, xb = featfn(list(rows.resolved_img1)), featfn(list(rows.resolved_img2)); typ = torch.as_tensor(rows.type_idx.to_numpy()); winner = rows.Winner.to_numpy()
        weight = torch.as_tensor(rows.confidence_weight.to_numpy(), dtype=torch.float32).clone()
        mult = torch.ones(len(rows)); mult[len(rs):] = tgt_weight
        if src_weights is not None: mult[:len(rs)] = torch.as_tensor(src_weights, dtype=torch.float32)
        weight = weight * mult
    else: xa = xb = torch.empty(0, dim); typ = torch.empty(0, dtype=torch.long); weight = torch.empty(0); winner = np.array([], dtype=object)
    head.eval(); opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=ctx.exp.cfg.weight_decay); index = torch.arange(len(typ)); masks = {l: torch.as_tensor(winner == l) for l in core.frozen.LABELS}
    cols = {c: core.frozen.TYPE_TO_INDEX[c] for c in refs}; gen = torch.Generator().manual_seed(seed * 1000003 + 17); active = list(ACTIVE)
    ref_vec = _flat(ref).detach() if (ref is not None and l2 > 0) else None
    for _ in range(steps):
        loss = G._bt_terms(head(xa)[index, typ], head(xb)[index, typ], masks, weight) / max(len(typ), 1)
        if len(refs) > 1: loss = loss + aw * G._anchor_loss({nm: head(f) for nm, f in refs.items()}, cols, None)
        if bad is not None and len(bad): loss = loss + ctx.exp.cfg.bad_anchor_weight * F.relu(head(bad) + 1.0).mean()
        if lap > 0 and gd is not None:
            e = gd.edges[torch.randint(gd.n_edges, (EDGE_BATCH,), generator=gen)]; si, sj = head(node_x[e[:, 0]])[:, active], head(node_x[e[:, 1]])[:, active]; sq = ((si - sj) ** 2).sum(1)
            if lap_extra > 0 and lap_extra_nodes is not None: sq = sq * (1.0 + lap_extra * (lap_extra_nodes[e[:, 0]] | lap_extra_nodes[e[:, 1]]).float())   # edges touching the held-out run's frames get the extra weight
            loss = loss + lap * sq.mean()
        if ref_vec is not None: loss = loss + l2 * ((_flat(head) - ref_vec) ** 2).sum()
        if prior_w > 0 and prior_idx is not None:
            u = torch.randint(len(prior_idx), (EDGE_BATCH,), generator=gen); sel = prior_idx[u]; loss = loss + prior_w * ((head(node_x[sel])[:, active] - prior_target[u]) ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
    return head.eval()


# ------------------------------------------------------------------------------------------ transfer helpers
def frame_index(gd, fold: Fold) -> torch.Tensor:
    """graph-node indices of R's frames."""
    return torch.as_tensor([gd.index[p] for p in fold.frames if p in gd.index], dtype=torch.long)


def propagate(gd, head, run_nodes: torch.Tensor, alpha: float = 0.9) -> torch.Tensor:
    """Label spreading of the source head's scores from the source nodes (every node outside R, clamped) to R's frames: F = (I - alpha S)^-1 Y_src, normalised by (I - alpha S)^-1 1_src.  Returns (len(run_nodes), 4)."""
    n = len(gd.paths); is_run = torch.zeros(n, dtype=torch.bool); is_run[run_nodes] = True; a = gd.adj.double(); d = a.sum(1).clamp_min(1.0); s = a / torch.sqrt(d[:, None] * d[None, :])
    with torch.no_grad(): sc = head(gd.x)[:, list(ACTIVE)].double()
    y = torch.zeros(n, len(ACTIVE), dtype=torch.float64); y[~is_run] = sc[~is_run]; m = (~is_run).double()[:, None]
    mat = torch.eye(n, dtype=torch.float64) - alpha * s; sol = torch.linalg.solve(mat, torch.cat([y, m], 1)); f, mass = sol[:, :len(ACTIVE)], sol[:, len(ACTIVE):]
    out = torch.where(mass > 1e-8, f / mass.clamp_min(1e-8), sc)[run_nodes].float(); return out


def proximity_weights(gd, fold: Fold, src: list, tau: float = 1.0, k: int = 5, clip=(0.2, 5.0)) -> np.ndarray:
    """Importance weights of the source judgments (sorted judgment order) from the feature-space proximity of their two images to R's frames: mean cosine to the k nearest R frames,
    standardised, w = exp(tau z), clipped and normalised to mean 1."""
    ctx = fold.ctx; run_nodes = frame_index(gd, fold); xr = gd.xn[run_nodes]; cache = ctx.features.cache
    def prox(path):
        x = cache[path].float() - gd.mu; x = x / x.norm().clamp_min(1e-9); return float((xr @ x).topk(k).values.mean())
    uniq = sorted(set(src)); info = ctx.info; paths = {p for j in uniq for p in (info[j]["img1"], info[j]["img2"])}; pr = {p: prox(p) for p in paths}
    s = np.array([0.5 * (pr[info[j]["img1"]] + pr[info[j]["img2"]]) for j in uniq]); z = (s - s.mean()) / (s.std() + 1e-9); w = np.clip(np.exp(tau * z), *clip); return w / w.mean()


def d_of(ctx, head, test, featfn=None) -> np.ndarray:
    return np.asarray(core.head_d(head, featfn or core.identity_features(ctx), test), dtype=float)


class Runner:
    """All learners of one fold; the source head and the graphs are computed once."""
    def __init__(self, fold: Fold):
        self.f, self.ctx = fold, fold.ctx; self.gd = G.graph_for(self.ctx, 10, False); self.gd_shuf = G.graph_for(self.ctx, 10, True); self.seed = fold.ctx.seed * 7 + RUNS.index(fold.run); self._src = None; self._prior = {}
        self.run_nodes = frame_index(self.gd, fold); self._gd_ind = None; self._align = None

    @property
    def src_head(self):
        if self._src is None: self._src = train_head(self.ctx, self.gd, self.seed, self.f.source, [])
        return self._src

    def _prior_target(self, kind):
        if kind not in self._prior:
            if kind == "own":
                with torch.no_grad(): self._prior[kind] = self.src_head(self.gd.x[self.run_nodes])[:, list(ACTIVE)]
            else:
                gd = self.gd_shuf if kind == "gprop_shuf" else self.gd; self._prior[kind] = propagate(gd, self.src_head, self.run_nodes)
        return self._prior[kind]

    def head(self, name: str, b: int):
        f, ctx, tgt = self.f, self.ctx, self.f.target(b); seed = self.seed + 101 * b
        if name == "cr_src": return self.src_head
        if name.startswith("cr_srcst"): return train_head(ctx, self.gd, seed, f.source, [], steps_scale=float(name[len("cr_srcst"):]))   # source-only step-count check (x of 100 steps x 4)
        if name.startswith("cr_srclr"): return train_head(ctx, self.gd, seed, f.source, [], lr=float(name[len("cr_srclr"):]), steps=400)   # source-only schedule check (lr; 400 steps)
        if name == "cr_tgt": return train_head(ctx, self.gd, seed, [], tgt)
        if name == "cr_pool": return train_head(ctx, self.gd, seed, f.source, tgt)
        if name.startswith("cr_poolw"): return train_head(ctx, self.gd, seed, f.source, tgt, tgt_weight=float(name[len("cr_poolw"):]))
        if name == "cr_iw": return train_head(ctx, self.gd, seed, f.source, tgt, src_weights=proximity_weights(self.gd, f, f.source))
        if name.startswith("cr_ft"):   # cr_ft<mu> : fine-tune the source head with an L2 penalty to its weights
            if b == 0: return self.src_head
            return train_head(ctx, self.gd, seed, [], tgt, init=self.src_head, l2=float(name[len("cr_ft"):] or 1.0), ref=self.src_head, steps_scale=2.0)
        if name.startswith("cr_prior") or name.startswith("cr_gprop"):
            if b == 0: return self.src_head
            kind = "gprop_shuf" if name.endswith("_shuf") else ("gprop" if name.startswith("cr_gprop") else "own"); base = name.replace("_shuf", ""); w = float(base[len("cr_gprop" if base.startswith("cr_gprop") else "cr_prior"):] or 1.0)
            return train_head(ctx, self.gd, seed, [], tgt, init=self.src_head, prior_idx=self.run_nodes, prior_target=self._prior_target(kind), prior_w=w, steps_scale=2.0)
        if name == "cr_ind_src": return train_head(ctx, self.gd_ind, self.seed, f.source, [])
        if name.startswith("cr_lapR"):   # cr_lapR<w> (source only) / cr_lapR<w>_pool: Laplacian weight (1 + w) on edges touching R's frames
            isr = torch.zeros(len(self.gd.paths), dtype=torch.bool); isr[self.run_nodes] = True; sub = name[len("cr_lapR"):]; pool = sub.endswith("_pool"); w = float(sub.replace("_pool", ""))
            if pool and b == 0: return train_head(ctx, self.gd, self.seed, f.source, [], lap_extra=w, lap_extra_nodes=isr)
            return train_head(ctx, self.gd, seed if pool else self.seed, f.source, tgt if pool else [], lap_extra=w, lap_extra_nodes=isr)
        if name == "cr_shuf_src": return train_head(ctx, self.gd_shuf, self.seed, f.source, [])   # shuffled-graph control of the Laplacian (source labels only)
        if name == "cr_nolap_src": return train_head(ctx, self.gd, self.seed, f.source, [], lap=0.0)
        if name == "cr_anchor0": return train_head(ctx, self.gd, self.seed, [], [], lap=0.0)
        if name.startswith("cr_align_"):
            ff, ns, _ = self.align; sub = name[len("cr_align_"):]
            if sub == "src" or b == 0: return train_head(ctx, self.gd, self.seed, f.source, [], featfn=ff, node_shift=ns)
            if sub == "pool": return train_head(ctx, self.gd, seed, f.source, tgt, featfn=ff, node_shift=ns)
            if sub == "tgt": return train_head(ctx, self.gd, seed, [], tgt, featfn=ff, node_shift=ns)
            raise ValueError(name)
        raise ValueError(name)


    # ---- decomposition / unsupervised-adaptation helpers
    @property
    def gd_ind(self):
        """graph WITHOUT the held-out run's frames (inductive: the new run's frames are not available as unlabelled nodes)."""
        if self._gd_ind is None:
            import ge_heads
            self._gd_ind = ge_heads.make_graph(self.ctx, 10, drop_paths=self.f.frames)
        return self._gd_ind

    @property
    def align(self):
        """(featfn, node_shift) with the run's features shifted by mean(source frames) - mean(run frames) (means over frames that are not test images)."""
        if self._align is None:
            ctx, f = self.ctx, self.f; cache = ctx.features.cache; test_imgs = {str(x) for g in f.test_groups for x in f.gi[g]}
            run_paths = [p for p in cache if (H.meta(p) or {}).get("run") == f.run]; other = [p for p in cache if H.meta(p) is not None and H.meta(p)["run"] != f.run and p not in test_imgs]
            mu_r = torch.stack([cache[p].float() for p in run_paths if p not in test_imgs]).mean(0); mu_s = torch.stack([cache[p].float() for p in other]).mean(0); shift = mu_s - mu_r; inrun = set(run_paths)
            base = core.identity_features(ctx)
            def featfn(paths):
                x = base(paths); add = torch.stack([shift if str(p) in inrun else torch.zeros_like(shift) for p in paths]) if len(paths) else x; return x + add
            ns = torch.zeros_like(self.gd.x); ns[self.run_nodes] = shift; self._align = (featfn, ns, float(shift.norm()))
        return self._align
