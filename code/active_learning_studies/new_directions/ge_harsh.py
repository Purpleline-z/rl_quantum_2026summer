"""Harsh preference splits built from the authoritative file-name metadata (run id = growth sample/date, temperature, clock time).

  run   leave-one-run-out: test = pair groups whose BOTH images belong to the held-out run; train/pool = groups with NO image in that run (cross-run groups touching it are dropped); all frames of the
        held-out run are removed from the graph (inductive: held-out frames attach to their nearest nodes only).
  temp  held-out temperature tercile: test = groups with both images at >= the top-tercile temperature of the pair-group images; train/pool = groups with no such image; all trajectory frames at or above
        that temperature are removed from the graph.
The ordinary protocol is already strictly image-disjoint (no evaluation image occurs in a training or candidate group; asserted in ``judgment_unit_study.Context``); these splits additionally separate
growth runs / conditions, so neighbouring frames of one run can no longer sit on both sides.
Metrics of one seed pool the decisive test judgments of all folds of the split kind."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np

import nd_core as core
import nd_learners as L
import judgment_unit_study as ju_study


def meta(path: str):
    m = L._NAME.match(Path(path).name); return None if m is None else {"run": m.group(2), "temp": int(m.group(3)), "frame": int(m.group(1)), "clock": int(m.group(4))}


def group_images(ctx) -> dict:
    return {g: (str(ctx.exp.groups[g].iloc[0].resolved_img1), str(ctx.exp.groups[g].iloc[0].resolved_img2)) for g in ctx.exp.groups}


def folds_of(ctx, kind: str):
    """list of (fold name, held-out predicate on an image path) for the split kind."""
    gi = group_images(ctx); pair_imgs = sorted({x for v in gi.values() for x in v}); m = {x: meta(x) for x in pair_imgs}
    assert all(v is not None for v in m.values()), "file-name metadata must parse for every pair-group image"
    if kind == "run": return [(f"run:{r}", (lambda x, r=r: (meta(x) or {}).get("run") == r)) for r in sorted({v["run"] for v in m.values()})]
    if kind == "temp":
        cut = float(np.quantile([v["temp"] for v in m.values()], 2 / 3)); return [(f"temp>={cut:.0f}", lambda x, cut=cut: (meta(x) or {}).get("temp", -1) >= cut)]
    if kind == "temp3":   # exploratory: all three temperature terciles in turn (more test groups than the top tercile alone)
        q1, q2 = (float(v) for v in np.quantile([v["temp"] for v in m.values()], [1 / 3, 2 / 3]))
        bands = [("temp<%.0f" % q1, lambda x, q1=q1: (meta(x) or {}).get("temp", 1e9) < q1), ("temp mid", lambda x, q1=q1, q2=q2: q1 <= (meta(x) or {}).get("temp", -1) < q2), ("temp>=%.0f" % q2, lambda x, q2=q2: (meta(x) or {}).get("temp", -1) >= q2)]
        return bands
    raise ValueError(kind)


def make_fold(ctx, held_out, fold_seed: int, n_initial: int = 10):
    """Overwrite the split of ``ctx`` (initial / pool / validation / test and the derived judgment lists) with the held-out condition.  Returns (test groups, set of held-out frame paths)."""
    gi = group_images(ctx); test = [g for g, (a, b) in gi.items() if held_out(a) and held_out(b)]; train = [g for g, (a, b) in gi.items() if not held_out(a) and not held_out(b)]
    rng = random.Random(ctx.seed * 1000 + 53 + fold_seed); shuffled = train[:]; rng.shuffle(shuffled); chosen, covered = [], set()
    for g in shuffled:
        types = set(ctx.exp.groups[g].canonical_type)
        if types - covered: chosen.append(g); covered |= types
    for g in shuffled:
        if len(chosen) >= n_initial: break
        if g not in chosen: chosen.append(g)
    chosen = chosen[:n_initial]; pool = [g for g in shuffled if g not in chosen]
    ctx.initial_groups, ctx.pool_groups, ctx.validation, ctx.test = chosen, pool, [], test
    ctx.initial = ju_study.judgments_of(ctx.exp, chosen); ctx.pool = ju_study.judgments_of(ctx.exp, pool); ctx.pool_index = {j: i for i, j in enumerate(ctx.pool)}
    ctx.info = {}
    for g in chosen + pool:
        for pos, r in enumerate(ctx.exp.groups[g].itertuples()):
            ctx.info[(g, pos)] = {"img1": r.resolved_img1, "img2": r.resolved_img2, "type_idx": int(r.type_idx), "decisive": r.Winner in ("1", "2"), "winner": r.Winner}
    frames = {p for p in ctx.features.cache if meta(p) is not None and held_out(p)}
    used = {str(x) for g in chosen + pool for x in gi[g]}; test_imgs = {str(x) for g in test for x in gi[g]}
    assert not (used & test_imgs) and not (used & frames), "held-out frames reachable from training / pool groups"
    assert set(chosen).isdisjoint(pool) and set(test).isdisjoint(chosen + pool)
    return test, frames


class Pooled:
    """Decisive test judgments of several folds concatenated (the minimal TestSet interface used by metrics_from_d)."""
    def __init__(self, tests):
        self.img1 = [x for t in tests for x in t.img1]; self.img2 = [x for t in tests for x in t.img2]; self.k = np.concatenate([t.k for t in tests]); self.sign = np.concatenate([t.sign for t in tests]); self.weight = np.concatenate([t.weight for t in tests])
