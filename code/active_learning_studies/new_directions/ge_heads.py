"""Reward-head learners with REFERENCE-ONLY anchors (valid for the classification endpoint: the outer-test ideal images are never used for anything, not even as unlabelled anchors) and a
generalised graph builder (node subsets for the dose-response, held-out conditions for the harsh splits).  ``fit_head`` is ``nd_graph_learner.fit_mlp_graph`` with ``all_ideal`` as a switch
(all_ideal=True reproduces it, unit-tested)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

import judgment_unit_study as ju_study
import nd_core as core
import nd_graph_learner as gl
from nd_graph_learner import _bt_terms, _anchor_loss, ACTIVE, EDGE_BATCH


def fit_head(ctx, labelled, seed, gd, featfn, node_x, dim, lap: float = 0.0, aw: float = 8.0, all_ideal: bool = True):
    lr, steps = ctx.params_for(labelled); rows = ju_study.rows_of(ctx.exp, labelled); head = core.new_head(dim, seed); refs, bad = core.anchors(ctx, featfn, all_ideal)
    if len(rows):
        xa, xb = featfn(list(rows.resolved_img1)), featfn(list(rows.resolved_img2)); typ = torch.as_tensor(rows.type_idx.to_numpy())
        weight, winner = torch.as_tensor(rows.confidence_weight.to_numpy(), dtype=torch.float32), rows.Winner.to_numpy()
    else: xa = xb = torch.empty(0, dim); typ = torch.empty(0, dtype=torch.long); weight = torch.empty(0); winner = np.array([], dtype=object)
    head.eval(); opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=ctx.exp.cfg.weight_decay); index = torch.arange(len(typ)); masks = {l: torch.as_tensor(winner == l) for l in core.frozen.LABELS}
    cols = {c: core.frozen.TYPE_TO_INDEX[c] for c in refs}; gen = torch.Generator().manual_seed(seed * 1000003 + 17); active = list(ACTIVE)
    for _ in range(steps):
        loss = _bt_terms(head(xa)[index, typ], head(xb)[index, typ], masks, weight) / max(len(typ), 1)
        if len(refs) > 1: loss = loss + aw * _anchor_loss({n: head(f) for n, f in refs.items()}, cols)
        if bad is not None and len(bad): loss = loss + ctx.exp.cfg.bad_anchor_weight * F.relu(head(bad) + 1.0).mean()
        if lap > 0 and gd is not None:
            e = gd.edges[torch.randint(gd.n_edges, (EDGE_BATCH,), generator=gen)]; si, sj = head(node_x[e[:, 0]])[:, active], head(node_x[e[:, 1]])[:, active]
            loss = loss + lap * ((si - sj) ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
    return head.eval()


def winrate_vectors(ctx, head, paths) -> np.ndarray:
    """(n, 4) win-rate of each image against the references of the other classes for each candidate class, in CLASSES order -- the rule of ``frozen.evaluate_model``."""
    from ge_classify import CLASSES
    fn = core.identity_features(ctx)
    with torch.no_grad(): s = head(fn(paths)).numpy(); refs = {c: head(fn([str(p) for p in ps])).numpy() for c, ps in ctx.exp.references.items()}
    out = np.zeros((len(paths), len(CLASSES)))
    for j, c in enumerate(CLASSES):
        idx = core.frozen.TYPE_TO_INDEX[c]; opp = np.concatenate([v for o, v in refs.items() if o != c])
        out[:, j] = (1 / (1 + np.exp(-np.clip(s[:, [idx]] - opp[:, idx][None, :], -50, 50)))).mean(1)
    return out


# ------------------------------------------------------------------------------------------ generalised graph
def make_graph(ctx, k: int = 10, shuffle: bool = False, noise: float = 0.0, keep=None, drop_paths=None, transductive_test_ideals: bool = False, label_sources: str = "all"):
    """GraphData over the allowed nodes of ``ctx`` with extra controls.
    keep(paths) -> subset of paths to keep as nodes (dose-response); drop_paths: image paths removed from the node set (held-out run/condition frames);
    transductive_test_ideals: keep the outer-test ideal images as unlabelled nodes; label_sources 'references' = labels only from reference images, 'all' = as nd_graph_learner."""
    original_ex, original_ch = gl.excluded_identities, gl.content_hashes
    drop = {str(p) for p in (drop_paths or ())}
    def excluded(c):
        paths = {str(x) for g in list(c.validation) + list(c.test) for r in c.exp.groups[g].itertuples() for x in (r.resolved_img1, r.resolved_img2)}
        if not transductive_test_ideals: paths |= {str(p) for ps in c.exp.test_images.values() for p in ps}
        paths |= drop
        return set(gl.content_hashes(sorted(paths)).values())
    gl.excluded_identities = excluded
    try:
        if keep is None: gd = gl.GraphData(ctx, k, shuffle=ctx.seed if shuffle else None, noise=noise)
        else:
            # node subset: build with the restricted feature cache (GraphData reads ctx.features.cache); restore afterwards
            full = ctx.features.cache; ctx.features.cache = {p: v for p, v in full.items() if p in set(keep(sorted(full)))}
            try: gd = gl.GraphData(ctx, k, shuffle=ctx.seed if shuffle else None, noise=noise)
            finally: ctx.features.cache = full
    finally: gl.excluded_identities = original_ex
    if label_sources == "references":
        gd.source_class = {gd.index[str(p)]: core.frozen.TYPE_TO_INDEX[c] for c, ps in ctx.exp.references.items() for p in ps if str(p) in gd.index}; gd._q = None
    return gd
