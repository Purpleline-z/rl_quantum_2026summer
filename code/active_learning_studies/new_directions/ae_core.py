"""Anchor-exchange study: how many typed absolute labels (anchors = labelled ideal images) is a pairwise judgment worth, and which anchors to label.

Learner = ``lap3_k10_st4`` of the earlier studies (aw8_all loss with anchor weight 8, Laplacian regulariser lambda 3 on the cosine kNN graph of unlabelled images, k = 10, head trained 4x longer), with ONE change:
the anchor term uses only a chosen SUBSET of the ideal images.  The graph's node set never changes with the anchor subset (every non-test ideal image stays an unlabelled node; only chosen ones carry labels).

Anchor universes (PROTOCOL EXTENSION, written into PREREGISTRATION_E*.md):
  'no_test'  references + utility-validation ideal images (116 per seed, labelled classes known): the universe of the whole study.  The 28 outer-test ideal images are never an anchor and never a graph node.
  'all144'   references + utility + outer-test (the original ``aw8_all`` anchors); only used for the exact-reproduction test and for one reference arm that quantifies the cost of dropping the 28 test anchors.
With every universe member selected, ``fit_head`` reproduces ``nd_graph_learner`` ``lap3_k10_st4`` exactly ('all144') -- unit-tested.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

import judgment_unit_study as ju_study
import nd_core as core
import nd_graph_learner as gl
from nd_graph_learner import _bt_terms, _anchor_loss, ACTIVE, EDGE_BATCH

CLASSES = ("(1 x 1)", "c(6 x 2)", "(√13 x √13)", "HTR")


# ------------------------------------------------------------------------------------------ universes and subsets
def universe(ctx, kind: str = "no_test") -> dict[str, list[str]]:
    """class -> list of ideal-image paths (original order: references, utility, [test])."""
    out = {}
    for c, ps in ctx.exp.references.items():
        out[c] = [str(p) for p in ps] + [str(p) for p in ctx.exp.utility_images.get(c, [])] + ([str(p) for p in ctx.exp.test_images.get(c, [])] if kind == "all144" else [])
    return out


def flat(uni: dict[str, list[str]]) -> list[tuple[str, str]]:
    return [(c, p) for c, ps in uni.items() for p in ps]


def subset_by_paths(uni, chosen: set[str]) -> dict[str, list[str]]:
    return {c: [p for p in ps if p in chosen] for c, ps in uni.items()}


def n_anchors(anchor_paths) -> int:
    return sum(len(v) for v in anchor_paths.values())


def random_subset(ctx, uni, n: int, subset_id: int = 0) -> dict[str, list[str]]:
    """n anchors drawn uniformly from the universe WITHOUT looking at the class (labels are not known before labelling); the classes that happen to be absent simply have no anchors."""
    items = flat(uni); rng = np.random.default_rng(ctx.seed * 7919 + 101 * subset_id + 13); order = [items[i][1] for i in rng.permutation(len(items))]
    return subset_by_paths(uni, set(order[:n]))


# ------------------------------------------------------------------------------------------ learner
def fit_head(ctx, labelled, seed, gd, anchor_paths, lap: float = 3.0, steps_scale: float = 4.0, aw: float = 8.0, lap_w: dict | None = None):
    """``nd_graph_learner.fit_mlp_graph`` with the anchor set given explicitly (classes without anchors are dropped; fewer than two classes -> no anchor loss)."""
    featfn = core.identity_features(ctx); node_x = gd.x; dim = gd.x.shape[1]
    lr, steps = ctx.params_for(labelled); steps = int(round(steps * steps_scale)); rows = ju_study.rows_of(ctx.exp, labelled); head = core.new_head(dim, seed)
    refs = {c: featfn(ps) for c, ps in anchor_paths.items() if len(ps)}; bad = featfn(ctx.exp.bad_paths) if ctx.exp.bad_paths else None
    tw = torch.ones(len(ACTIVE)) if not lap_w else torch.tensor([float(lap_w.get(t, 1.0)) for t in ACTIVE])
    if len(rows):
        xa, xb = featfn(list(rows.resolved_img1)), featfn(list(rows.resolved_img2)); typ = torch.as_tensor(rows.type_idx.to_numpy())
        weight, winner = torch.as_tensor(rows.confidence_weight.to_numpy(), dtype=torch.float32), rows.Winner.to_numpy()
    else: xa = xb = torch.empty(0, dim); typ = torch.empty(0, dtype=torch.long); weight = torch.empty(0); winner = np.array([], dtype=object)
    head.eval(); opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=ctx.exp.cfg.weight_decay); index = torch.arange(len(typ)); masks = {l: torch.as_tensor(winner == l) for l in core.frozen.LABELS}
    cols = {c: core.frozen.TYPE_TO_INDEX[c] for c in refs}; gen = torch.Generator().manual_seed(seed * 1000003 + 17); active = list(ACTIVE)
    for _ in range(steps):
        loss = _bt_terms(head(xa)[index, typ], head(xb)[index, typ], masks, weight) / max(len(typ), 1)
        if len(refs) > 1: loss = loss + aw * _anchor_loss({n: head(f) for n, f in refs.items()}, cols, None)
        if bad is not None and len(bad): loss = loss + ctx.exp.cfg.bad_anchor_weight * F.relu(head(bad) + 1.0).mean()
        if lap > 0 and gd is not None:
            e = gd.edges[torch.randint(gd.n_edges, (EDGE_BATCH,), generator=gen)]; si, sj = head(node_x[e[:, 0]])[:, active], head(node_x[e[:, 1]])[:, active]
            loss = loss + lap * (((si - sj) ** 2) * tw).sum(1).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
    return head.eval()


# ------------------------------------------------------------------------------------------ endpoints
def per_type_auc(d, test) -> dict:
    out = {}
    for k, name in core.TYPE_NAMES.items():
        m = test.k == k; s = test.sign[m]; out[f"{name}_auc"] = float(roc_auc_score(s > 0, np.asarray(d)[m])) if m.any() and len(set(s)) == 2 else float("nan")
    return out


def external_type_accuracy(ctx, head, anchor_paths) -> dict:
    """Paper endpoint: type of each of the 28 outer-test ideal images by the win-rate rule, with the SELECTED ANCHORS as the reference set (classes without anchors cannot be predicted)."""
    fn = core.identity_features(ctx); refs = {c: head(fn(ps)).detach().numpy() for c, ps in anchor_paths.items() if len(ps)}; correct = total = 0; by = {}
    if len(refs) < 2: return {"ext_acc": float("nan"), **{f"ext_acc_{j}": float("nan") for j in range(4)}}
    for truth, paths in ctx.exp.test_images.items():
        with torch.no_grad(): scores = head(fn([str(p) for p in paths])).numpy()
        ok = 0
        for score in scores:
            win = {}
            for cand in refs:
                opp = np.concatenate([v for o, v in refs.items() if o != cand]); idx = core.frozen.TYPE_TO_INDEX[cand]; win[cand] = float(np.mean(1 / (1 + np.exp(-np.clip(score[idx] - opp[:, idx], -50, 50)))))
            ok += max(win, key=win.get) == truth
        by[truth] = ok / len(paths); correct += ok; total += len(paths)
    return {"ext_acc": correct / total, **{f"ext_acc_{j}": by.get(c, float("nan")) for j, c in enumerate(CLASSES)}}


def evaluate(ctx, head, test, anchor_paths, label_ids=None) -> dict:
    d = core.head_d(head, core.identity_features(ctx), test)
    return {**core.metrics_from_d(d, test), **per_type_auc(d, test), **external_type_accuracy(ctx, head, anchor_paths)}


# ------------------------------------------------------------------------------------------ active anchor rules (labels of unselected ideal images are NEVER used)
def _feat(gd, paths): return gd.xn[[gd.index[p] for p in paths]].numpy().astype(np.float64)


def _hop_distances(gd, src_idx: list[int]) -> np.ndarray:
    n = len(gd.paths); a = gd.adj.numpy().astype(np.float32); return shortest_path(csr_matrix(a), method="D", unweighted=True, indices=src_idx)   # (len(src), n)


def kcenter_graph(ctx, uni, gd, n: int, first_pool: list[str] | None = None) -> list[str]:
    """Greedy k-center over the universe nodes with hop distance on the kNN graph (farthest-first; first = the node with the smallest mean hop distance, i.e. the graph medoid).  Ties: feature distance."""
    paths = [p for _, p in flat(uni)]; idx = [gd.index[p] for p in paths]; D = _hop_distances(gd, idx)[:, idx]; D = np.where(np.isfinite(D), D, D[np.isfinite(D)].max() + 1)
    X = _feat(gd, paths); tie = 1 - X @ X.T
    chosen = [int(np.argmin(D.mean(1)))]; mind = D[chosen[0]].copy() + 1e-3 * tie[chosen[0]]
    while len(chosen) < n:
        j = int(np.argmax(np.where(np.isin(np.arange(len(paths)), chosen), -1, mind))); chosen.append(j); mind = np.minimum(mind, D[j] + 1e-3 * tie[j])
    return [paths[i] for i in chosen]


def kcenter_feature(ctx, uni, gd, n: int) -> list[str]:
    paths = [p for _, p in flat(uni)]; X = _feat(gd, paths); sim = X @ X.T; chosen = [int(np.argmax(sim.mean(1)))]; mind = 1 - sim[chosen[0]]
    while len(chosen) < n:
        j = int(np.argmax(np.where(np.isin(np.arange(len(paths)), chosen), -1, mind))); chosen.append(j); mind = np.minimum(mind, 1 - sim[j])
    return [paths[i] for i in chosen]


def typiclust(ctx, uni, gd, n: int, knn: int = 20) -> list[str]:
    """TypiClust (Hacohen et al. 2022): k-means with n clusters on the L2-normalised features of the universe, then the most typical point (inverse mean distance to its knn nearest neighbours) of each cluster."""
    paths = [p for _, p in flat(uni)]; X = _feat(gd, paths); labels = KMeans(n_clusters=n, n_init=5, random_state=ctx.seed).fit_predict(X); dist = np.sqrt(np.maximum(2 - 2 * X @ X.T, 0)); chosen = []
    for c in range(n):
        members = np.flatnonzero(labels == c)
        if not len(members): continue
        typ = []
        for i in members:
            d = np.sort(dist[i])[1:knn + 1]; typ.append(1.0 / (d.mean() + 1e-9))
        chosen.append(int(members[int(np.argmax(typ))]))
    rest = [i for i in np.argsort(-(1.0 / (np.sort(dist, axis=1)[:, 1:knn + 1].mean(1) + 1e-9))) if i not in chosen]; chosen += rest[: n - len(chosen)]
    return [paths[i] for i in chosen[:n]]


def spreading_posterior(gd, labelled: dict[int, int], alpha: float = 0.9) -> np.ndarray:
    """Label spreading of the currently labelled anchors (node index -> class column) over the graph; rows normalised to sum 1 (uniform when no mass)."""
    n = len(gd.paths); y = np.zeros((n, 4)); 
    for i, c in labelled.items(): y[i, c] = 1.0
    a = gd.adj.numpy().astype(np.float64); d = np.maximum(a.sum(1), 1.0); s = a / np.sqrt(np.outer(d, d)); f = np.maximum(np.linalg.solve(np.eye(n) - alpha * s, y), 0); mass = f.sum(1, keepdims=True)
    return np.where(mass > 1e-9, f / np.maximum(mass, 1e-9), 0.25)


def sequential_rule(ctx, uni, gd, n: int, rule: str, rounds: int = 3, seed_subset: int = 0) -> list[str]:
    """Round 1: the first n/rounds anchors of the random order (same as ``random_subset``'s start).  Later rounds use the label-spreading posterior of the anchors labelled so far (classes learned from the
    LABELLED anchors only): 'amb' = the most ambiguous (entropy) unlabelled ideal images; 'htrq' = half of every round the most HTR-like unlabelled images (highest HTR posterior), half the most ambiguous."""
    items = flat(uni); cls_of = {p: CLASSES.index(c) for c, p in items}; rng = np.random.default_rng(ctx.seed * 7919 + 101 * seed_subset + 13); order = [items[i][1] for i in rng.permutation(len(items))]
    per = n // rounds; chosen = order[:per]
    while len(chosen) < n:
        q = spreading_posterior(gd, {gd.index[p]: cls_of[p] for p in chosen}); rest = [p for _, p in items if p not in set(chosen)]; qr = q[[gd.index[p] for p in rest]]
        ent = -(qr * np.log(np.clip(qr, 1e-9, 1))).sum(1); take = min(per, n - len(chosen))
        if rule == "amb": pick = list(np.argsort(-ent, kind="stable")[:take])
        else:
            h = take // 2; a = list(np.argsort(-qr[:, 3], kind="stable")[:h]); b = [i for i in np.argsort(-ent, kind="stable") if i not in a][: take - h]; pick = a + b
        chosen += [rest[i] for i in pick]
    return chosen[:n]


def stratified_random(ctx, uni, n: int, subset_id: int = 0) -> dict[str, list[str]]:
    """DIAGNOSTIC (uses the true classes of the universe): n/4 random anchors per class (class-balanced random)."""
    rng = np.random.default_rng(ctx.seed * 7919 + 101 * subset_id + 29); per = n // len(uni); out = {}
    for c, ps in uni.items(): out[c] = [ps[i] for i in rng.permutation(len(ps))[:per]]
    return out


def kcf_balanced(ctx, uni, gd, n: int, rounds: int = 3, seed_subset: int = 0) -> list[str]:
    """Sequential feature k-center with class balance: round 1 = the random start of ``sequential_rule``; later rounds pick farthest-first (feature distance to the chosen anchors) with the score multiplied by
    1 + deficit/target of the PREDICTED class (label-spreading posterior of the anchors labelled so far), so under-represented classes are sought.  Uses only the labels of anchors already chosen."""
    items = flat(uni); cls_of = {p: CLASSES.index(c) for c, p in items}; rng = np.random.default_rng(ctx.seed * 7919 + 101 * seed_subset + 13); order = [items[i][1] for i in rng.permutation(len(items))]
    per = n // rounds; chosen = order[:per]; paths = [p for _, p in items]; X = _feat(gd, paths); pos = {p: i for i, p in enumerate(paths)}; target = n / 4
    while len(chosen) < n:
        q = spreading_posterior(gd, {gd.index[p]: cls_of[p] for p in chosen}); pred = q[[gd.index[p] for p in paths]].argmax(1); counts = np.bincount([cls_of[p] for p in chosen], minlength=4)
        mind = 1 - (X @ X[[pos[p] for p in chosen]].T).max(1); take = min(per, n - len(chosen)); cur = set(chosen)
        for _ in range(take):
            deficit = np.maximum(0, target - counts)[pred] / target; score = mind * (1 + deficit); score[[pos[p] for p in cur]] = -1; j = int(np.argmax(score)); cur.add(paths[j]); chosen.append(paths[j]); counts[cls_of[paths[j]]] += 1
            mind = np.minimum(mind, 1 - X @ X[j])
    return chosen[:n]


def active_subset(ctx, uni, rule: str, n: int, gd, gd_shuf, subset_id: int = 0) -> dict[str, list[str]]:
    """rule in rand | kcg | kcg_shuf | kcf | typi | amb | amb_shuf | htrq | htrq_shuf."""
    if rule == "rand": return random_subset(ctx, uni, n, subset_id)
    if rule == "strat": return stratified_random(ctx, uni, n, subset_id)
    if rule == "kcf_bal": return subset_by_paths(uni, set(kcf_balanced(ctx, uni, gd, n, seed_subset=subset_id)))
    if rule == "kcg": return subset_by_paths(uni, set(kcenter_graph(ctx, uni, gd, n)))
    if rule == "kcg_shuf": return subset_by_paths(uni, set(kcenter_graph(ctx, uni, gd_shuf, n)))
    if rule == "kcf": return subset_by_paths(uni, set(kcenter_feature(ctx, uni, gd, n)))
    if rule == "typi": return subset_by_paths(uni, set(typiclust(ctx, uni, gd, n)))
    base, shuf = (rule[:-5], True) if rule.endswith("_shuf") else (rule, False)
    if base in ("amb", "htrq"): return subset_by_paths(uni, set(sequential_rule(ctx, uni, gd_shuf if shuf else gd, n, base, seed_subset=subset_id)))
    raise ValueError(rule)
