"""Graph selection rules under the tuned learner (Prompt 2, branch claude/graph-selection-htr-1012).

Single-shot selection screen like ``nd_select.py``: one batch of b judgments chosen from the model trained on the initial judgments, then the learner (default ``aw8_all``) is trained on
initial + selected and evaluated on the held-out decisive judgments.  Differences to nd_select: the registry contains the graph rules, every record also stores the per-type AUC and
the type composition of the batch (mechanism analysis), and the picks themselves.

Selector names: ``<rule>`` uses the repository's model fitted on the initial judgments (default learner); ``<rule>@aw`` uses the aw8_all-fitted model (learner-consistent selection) for the
rules that look at the model (vopt_u, gvopt_*); rules that do not look at the model are the same with and without the suffix.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import nd_core as core
import nd_learners as L
import nd_select as ns
import judgment_unit_strategies as ju
import judgment_unit_study as ju_study
import new_methods_strategies as nm
import graph_typed as gt
import graph_vopt as gv

TYPE_IDX = {"t1x1": 0, "tc6x2": 2, "t13": 3, "htr": 4}
HTR, RT13 = 4, 3


# ------------------------------------------------------------------------------------------ graph pieces shared by the HTR rules
def _graph(cands, labeled, cache, shuffle, seed):
    ctx = ns.CTX_REF["ctx"]; return gt.ImageGraph(cands, labeled, cache, gt.reference_types(ctx.exp), seed, shuffle)


def htr_rt13_ambiguity(graph: gt.ImageGraph) -> np.ndarray:
    """Per node: 1 - |2 r - 1| with r = q(HTR) / (q(HTR) + q(RT13)); 1 = the label-spreading posterior cannot tell HTR from RT13, 0 = clear.  Nodes with no mass on either: 0."""
    qh, qr = graph.q[:, gt.TYPES.index(HTR)], graph.q[:, gt.TYPES.index(RT13)]; tot = qh + qr; r = np.where(tot > 1e-9, qh / np.maximum(tot, 1e-9), .5)
    return np.where(tot > 1e-9, 1 - np.abs(2 * r - 1), 0.0)


def bridge_mass(graph: gt.ImageGraph) -> np.ndarray:
    """Per node: min(q(HTR), q(RT13)) -- large only for images close (on the kNN graph) to references of both types."""
    return np.minimum(graph.q[:, gt.TYPES.index(HTR)], graph.q[:, gt.TYPES.index(RT13)])


def _judgment_weight(graph, cands, node_score, types=(HTR, RT13), floor: float = 0.0):
    """Judgment-level weight: mean node score of its two images for judgments of type HTR/RT13 (others get ``floor``)."""
    w = np.array([.5 * (node_score[graph.index[x["img1"]]] + node_score[graph.index[x["img2"]]]) if int(x["type_idx"]) in types else floor for x in cands]); return w


def _coverage(graph, cands, labeled, weight, budget):
    """Farthest-first in graph-propagated pair space (as typed_decisive_coverage) with distance x ``weight``."""
    allp = gt._pair_space(graph, cands + labeled); allp = (allp - allp.mean(0)) / (allp.std(0) + 1e-6); allp = allp / np.sqrt(allp.shape[1])
    onehot = np.eye(5)[[int(x["type_idx"]) for x in cands + labeled]][:, list(gt.TYPES)]; allp = np.concatenate([allp, onehot], axis=1)
    cand, lab = allp[:len(cands)], allp[len(cands):]; covered = lab.copy(); chosen = []; remaining = list(range(len(cands)))
    for _ in range(min(budget, len(cands))):
        dist = np.sqrt(((cand[remaining][:, None] - covered[None]) ** 2).sum(2)).min(1) if len(covered) else np.linalg.norm(cand[remaining] - cand.mean(0), axis=1)
        pick = remaining[int(np.argmax(dist * weight[remaining]))]; chosen.append(pick); remaining.remove(pick); covered = np.vstack([covered, cand[pick]])
    return [cands[i] for i in chosen]


# ------------------------------------------------------------------------------------------ rules
def _tdc(name):
    def select(cands, labeled, model, cache, budget, seed=0): return gt.make_selector(name, ns.CTX_REF["ctx"].exp)(cands, labeled, model, cache, budget, seed)
    return select


def _gvopt(name, aw):
    def select(cands, labeled, model, cache, budget, seed=0):
        m = ns._selection_model(ns.CTX_REF["ctx"], 8.0, True) if aw else model; return gv.NEW[name](cands, labeled, m, cache, budget, seed)
    return select


def _vopt_u(aw):
    def select(cands, labeled, model, cache, budget, seed=0):
        m = ns._selection_model(ns.CTX_REF["ctx"], 8.0, True) if aw else model; return nm.NEW["vopt_u"](cands, labeled, m, cache, budget, seed)
    return select


def htr_rule(kind: str, shuffle: bool = False, floor: float = 0.0, aw: bool = False):
    """HTR-oriented graph rules.
    amb        top judgments (type HTR/RT13) by the mean HTR-vs-RT13 posterior ambiguity of their two images
    bridge     top judgments by the mean bridge mass min(q_HTR, q_RT13)
    cov_amb    typed coverage (farthest-first in graph-propagated pair space) with distance x ambiguity weight (floor for other types) x P(decisive)
    cov_bridge same with bridge mass
    vopt_amb   the repository's I-optimal rule with target weights = ambiguity weight (floor + ambiguity), i.e. variance reduction where HTR vs RT13 is unclear
    vopt_bridge the same with bridge mass"""
    def select(cands, labeled, model, cache, budget, seed=0):
        g = _graph(cands, labeled, cache, shuffle, seed); node = htr_rt13_ambiguity(g) if "amb" in kind else bridge_mass(g)
        node = node / max(float(node.max()), 1e-9); w = _judgment_weight(g, cands, node, floor=0.0)
        if kind in ("amb", "bridge"):
            order = np.argsort(-w, kind="stable")[:budget]; return [cands[i] for i in order]
        if kind in ("cov_amb", "cov_bridge"):
            p = gt.decisive_probability(g, cands, labeled); return _coverage(g, cands, labeled, (floor + w) * p, budget)
        if kind == "vopt_htr":   # class-priority control without any graph: weight 1 for HTR judgments, ``floor`` for the rest
            target = np.array([1.0 if int(x["type_idx"]) == HTR else floor for x in cands]); target = target / max(float(target.mean()), 1e-9); m = ns._selection_model(ns.CTX_REF["ctx"], 8.0, True) if aw else model
            return nm.vopt(cands, labeled, m, cache, budget, seed, sensitivity=False, target_decisive=lambda c, l, ca: target)
        if kind in ("vopt_amb", "vopt_bridge"):
            target = (floor + w) / max(float((floor + w).mean()), 1e-9); m = ns._selection_model(ns.CTX_REF["ctx"], 8.0, True) if aw else model
            return nm.vopt(cands, labeled, m, cache, budget, seed, sensitivity=False, target_decisive=lambda c, l, ca: target)
        raise ValueError(kind)
    return select


def vopt_dec(kind: str, aw: bool = True, power: float = 1.0):
    """vopt_u (target weights) with the graph-informed P(decisive) of each judgment as the target weight: ``kind`` 'graph' = label-spreading features, 'typeonly' = type one-hot only
    (no reference/graph features), 'shuffled' = graph destroyed.  The decisive-weight ingredient of typed coverage combined with the variance-reduction design."""
    def select(cands, labeled, model, cache, budget, seed=0):
        ctx = ns.CTX_REF["ctx"]; refs = {} if kind == "typeonly" else gt.reference_types(ctx.exp)
        g = gt.ImageGraph(cands, labeled, cache, refs, seed, kind == "shuffled"); p = gt.decisive_probability(g, cands, labeled) ** power; target = p / max(float(p.mean()), 1e-9)
        m = ns._selection_model(ctx, 8.0, True) if aw else model
        return nm.vopt(cands, labeled, m, cache, budget, seed, sensitivity=False, target_decisive=lambda c, l, ca: target)
    return select


def quota_coverage(shares: dict, shuffle: bool = False):
    """Typed coverage whose batch is split by type quota (largest-remainder rounding of ``shares``): the mechanism test for 'the gain of typed coverage is its type allocation'."""
    def select(cands, labeled, model, cache, budget, seed=0):
        g = _graph(cands, labeled, cache, shuffle, seed); p = gt.decisive_probability(g, cands, labeled); types = np.array([int(x["type_idx"]) for x in cands])
        raw = {k: shares[k] * budget for k in shares}; quota = {k: int(np.floor(v)) for k, v in raw.items()}
        for k in sorted(raw, key=lambda k: -(raw[k] - quota[k]))[: budget - sum(quota.values())]: quota[k] += 1
        chosen = []
        allp = gt._pair_space(g, cands + labeled); allp = (allp - allp.mean(0)) / (allp.std(0) + 1e-6); allp = allp / np.sqrt(allp.shape[1])
        onehot = np.eye(5)[[int(x["type_idx"]) for x in cands + labeled]][:, list(gt.TYPES)]; allp = np.concatenate([allp, onehot], axis=1)
        cand, lab = allp[:len(cands)], allp[len(cands):]; covered = lab.copy(); remaining = list(range(len(cands))); left = dict(quota)
        for _ in range(min(budget, len(cands))):
            ok = [i for i in remaining if left.get(int(types[i]), 0) > 0] or remaining
            dist = np.sqrt(((cand[ok][:, None] - covered[None]) ** 2).sum(2)).min(1); pick = ok[int(np.argmax(dist * p[ok]))]
            chosen.append(pick); remaining.remove(pick); covered = np.vstack([covered, cand[pick]]); left[int(types[pick])] = left.get(int(types[pick]), 0) - 1
        return [cands[i] for i in chosen]
    return select


REGISTRY = {
    "vopt_dec@aw": None, "vopt_dec_typeonly@aw": None, "vopt_dec_shuffled@aw": None,
    "vopt_u": _vopt_u(False), "vopt_u@aw": _vopt_u(True),
    "tdc": _tdc("typed_decisive_coverage"), "tdc_typeonly": _tdc("typed_decisive_coverage_typeonly"), "tdc_shuffled": _tdc("typed_decisive_coverage_shuffled"),
    "tcu": _tdc("typed_coverage_uniform"),
    "gvopt_lap3": _gvopt("gvopt_lap3", False), "gvopt_lap3@aw": _gvopt("gvopt_lap3", True), "gvopt_lap3_shuffled": _gvopt("gvopt_lap3_shuffled", False), "gvopt_lap3_shuffled@aw": _gvopt("gvopt_lap3_shuffled", True),
    "gvopt_type": _gvopt("gvopt_type", False), "gvopt_type@aw": _gvopt("gvopt_type", True), "gvopt_type_shuffled@aw": _gvopt("gvopt_type_shuffled", True),
    "htr_amb": htr_rule("amb"), "htr_amb_shuffled": htr_rule("amb", True), "htr_bridge": htr_rule("bridge"), "htr_bridge_shuffled": htr_rule("bridge", True),
    "htr_cov_amb": htr_rule("cov_amb", floor=.1), "htr_cov_amb_shuffled": htr_rule("cov_amb", True, floor=.1),
    "htr_cov_bridge": htr_rule("cov_bridge", floor=.1), "htr_cov_bridge_shuffled": htr_rule("cov_bridge", True, floor=.1),
    "htr_vopt_prio@aw": htr_rule("vopt_htr", floor=.3, aw=True),
    "htr_vopt_amb@aw": htr_rule("vopt_amb", floor=.3, aw=True), "htr_vopt_amb_shuffled@aw": htr_rule("vopt_amb", True, floor=.3, aw=True),
    "htr_vopt_bridge@aw": htr_rule("vopt_bridge", floor=.3, aw=True), "htr_vopt_bridge_shuffled@aw": htr_rule("vopt_bridge", True, floor=.3, aw=True),
}


REGISTRY.update({"vopt_dec@aw": vopt_dec("graph"), "vopt_dec_typeonly@aw": vopt_dec("typeonly"), "vopt_dec_shuffled@aw": vopt_dec("shuffled")})


def _vopt_ridge(ridge):
    """Non-graph control for the Laplacian prior: vopt_u with prior precision ``ridge`` * I (the graph prior adds about 3 to the mean diagonal)."""
    def select(cands, labeled, model, cache, budget, seed=0):
        return gv.design(cands, labeled, ns._selection_model(ns.CTX_REF["ctx"], 8.0, True), cache, budget, seed, "none", False, ridge=ridge)
    return select


REGISTRY.update({"vopt_ridge4@aw": _vopt_ridge(4.0), "vopt_ridge10@aw": _vopt_ridge(10.0)})


def register_extra(name: str, fn): REGISTRY[name] = fn


# ------------------------------------------------------------------------------------------ evaluation extras
def per_type_auc(d, test) -> dict:
    out = {}
    for k, name in core.TYPE_NAMES.items():
        m = test.k == k; s = test.sign[m]; out[f"{name}_auc"] = float(roc_auc_score(s > 0, np.asarray(d)[m])) if m.any() and len(set(s)) == 2 else float("nan")
    return out


def composition(picks, ids) -> dict:
    t = [int(x["type_idx"]) for x in picks]; imgs = {p for x in picks for p in (x["img1"], x["img2"])}
    return {**{f"pick_{n}": int(sum(1 for v in t if v == k)) for k, n in core.TYPE_NAMES.items()}, "pick_images": len(imgs), "pick_decisive": int(sum(1 for x in picks if x["outcomes"][0][1])), "pick_ids": [x["pair_id"] for x in picks]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selectors", required=True); parser.add_argument("--learners", default="aw8_all"); parser.add_argument("--seeds", default="8000-8019"); parser.add_argument("--splits", default="A,B")
    parser.add_argument("--draws", type=int, default=3); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--shard", default="0/1"); parser.add_argument("--budgets", default="10,20,40,60")
    args = parser.parse_args(); torch.set_num_threads(1)
    reg = L.LEARNERS.copy(); import nd_bayes, nd_gp, nd_temporal
    reg.update(nd_bayes.LEARNERS); reg.update(nd_gp.LEARNERS); reg.update(nd_temporal.LEARNERS)
    sel_names = args.selectors.split(","); learner_names = args.learners.split(","); budgets = [int(b) for b in args.budgets.split(",")]
    shard, nshards = (int(x) for x in args.shard.split("/")); args.out.mkdir(parents=True, exist_ok=True); cache = core.feature_cache()
    with tempfile.TemporaryDirectory() as scratch:
        for seed in [s for i, s in enumerate(ns.parse_seeds(args.seeds)) if i % nshards == shard]:
            for split in args.splits.split(","):
                target = args.out / f"seed{seed}_{split}.json"; done = json.loads(target.read_text()) if target.exists() else []; have = {(r["budget"], r["selector"], r["draw"], r["learner"]) for r in done}
                need = [(b, s, (range(args.draws) if s == "random" else [0]), n) for b in budgets for s in sel_names for n in learner_names]
                if all((b, s, d, n) in have for b, s, ds, n in need for d in ds): continue
                started = time.monotonic(); ctx = core.make_ctx(seed, split, scratch, cache); test = core.TestSet(ctx); ns.CTX_REF["ctx"] = ctx; nm.CTX["ctx"] = ctx
                model = ctx.fit(ctx.initial); cands, labeled_items, ccache = ju_study.make_candidates(ctx, ctx.pool, ctx.initial, model, seed); initial = set(ctx.initial)
                by_id = {f"{j[0]}#{j[1]}": j for j in ctx.pool}
                for budget in budgets:
                    for sel in sel_names:
                        comp = {}
                        if sel == "random":
                            label_sets = {d: core.random_labelled(ctx, budget, d) for d in range(args.draws)}
                        else:
                            picks = REGISTRY[sel](cands, labeled_items, model, ccache, budget, seed); label_sets = {0: list(ctx.initial) + [by_id[x["pair_id"]] for x in picks]}; comp = composition(picks, by_id)
                        for draw, labelled in label_sets.items():
                            if sel == "random":
                                picked = [by_id_item(cands, j) for j in labelled if j not in initial]; comp = composition([x for x in picked if x is not None], by_id)
                            for name in learner_names:
                                if (budget, sel, draw, name) in have: continue
                                d = reg[name](ctx, labelled, seed * 7 + draw)(test)
                                done.append({"seed": seed, "split": split, "budget": budget, "selector": sel, "draw": draw, "learner": name, **core.metrics_from_d(d, test), **per_type_auc(d, test), **comp})
                    target.write_text(json.dumps(done))
                print(f"seed {seed} split {split}: {time.monotonic() - started:.0f}s", flush=True)


_BY_KEY: dict = {}


def by_id_item(cands, j):
    """Candidate dict of the pool judgment ``j`` (used to read the type/images of random draws)."""
    key = id(cands)
    if key not in _BY_KEY: _BY_KEY.clear(); _BY_KEY[key] = {(x["base_pair"], int(x["pos"])): x for x in cands}
    return _BY_KEY[key].get((j[0], int(j[1])))


if __name__ == "__main__":
    main()
