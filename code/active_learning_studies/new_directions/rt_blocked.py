"""Time-blocked evaluation contexts (audit 3).

Images are frames of three growth runs (RR220204A, RR220206A, RR220411A).  A "blocked" context holds out whole time blocks of a run: every group with at least one image inside a test block
is a test group; every image within ``buffer`` frames of a test block (same run) — labelled or not — is removed from the training groups, the candidate pool and the graph nodes, so no training
or graph image is closer than ``buffer`` frames (same run) to a test image; groups that share any image with a test group are dropped from training as in the repository's splits.
"""
from __future__ import annotations

import random

import numpy as np

import judgment_unit_study as ju_study
import nd_core as core
import nd_graph_learner as G
import rt_core as rt

_orig_excluded = G.excluded_identities


def _patched(ctx):
    return _orig_excluded(ctx) | getattr(ctx, "_extra_excluded_hashes", set())


G.excluded_identities = _patched          # the graph builder looks the function up in this module's globals


def install_exclusion(ctx, paths) -> int:
    hashes = G.content_hashes(sorted(paths)); ctx._extra_excluded_hashes = {hashes[p] for p in paths}; return len(ctx._extra_excluded_hashes)


def make_blocked_ctx(seed: int, scratch, cache, block: int = 100, buffer: int = 25, min_test_groups: int = 34):
    ctx = core.make_ctx(seed, "B", scratch, cache); exp = ctx.exp
    groups = list(exp.groups); imgs = {g: (exp.groups[g].iloc[0].resolved_img1, exp.groups[g].iloc[0].resolved_img2) for g in groups}
    key = lambda p: (rt.run_of(p), rt.frame_of(p))
    blocks = sorted({(r, f // block) for g in groups for p in imgs[g] for r, f in [key(p)] if r is not None}); rng = random.Random(seed * 1000 + 77); rng.shuffle(blocks)
    test_blocks, test_groups = set(), []
    for b in blocks:
        if len(test_groups) >= min_test_groups: break
        test_blocks.add(b); test_groups = [g for g in groups if any(key(p)[0] is not None and (key(p)[0], key(p)[1] // block) in test_blocks for p in imgs[g])]
    def in_zone(p):
        r, f = key(p)
        return r is not None and any(r == tr and tb * block - buffer <= f < (tb + 1) * block + buffer for tr, tb in test_blocks)
    test_set = set(test_groups); test_images = {p for g in test_groups for p in imgs[g]}
    candidates = [g for g in groups if g not in test_set and not any(in_zone(p) for p in imgs[g]) and not (set(imgs[g]) & test_images)]
    covered, initial = set(), []; n_initial = len(ctx.initial_groups)
    for g in candidates:
        t = set(exp.groups[g].canonical_type)
        if t - covered: initial.append(g); covered |= t
    for g in candidates:
        if len(initial) >= n_initial: break
        if g not in initial: initial.append(g)
    initial = initial[:n_initial]; pool = [g for g in candidates if g not in initial]
    ctx.initial_groups, ctx.pool_groups, ctx.validation, ctx.test = initial, pool, [], test_groups
    ctx.initial, ctx.pool = ju_study.judgments_of(exp, initial), ju_study.judgments_of(exp, pool); ctx.pool_index = {j: i for i, j in enumerate(ctx.pool)}; ctx.info = {}
    for g in set(initial) | set(pool):
        for pos, r in enumerate(exp.groups[g].itertuples()):
            ctx.info[(g, pos)] = {"img1": r.resolved_img1, "img2": r.resolved_img2, "type_idx": int(r.type_idx), "decisive": r.Winner in ("1", "2"), "winner": r.Winner}
    zone_paths = [p for p in cache if in_zone(p)]; install_exclusion(ctx, zone_paths)
    used = {x for j in ctx.initial + ctx.pool for x in (ctx.info[j]["img1"], ctx.info[j]["img2"])}
    assert not (test_images & used) and not any(in_zone(p) for p in used), "blocked split leaks"
    ctx.blocked = {"test_blocks": sorted(test_blocks), "n_test_groups": len(test_groups), "n_initial_groups": len(initial), "n_pool_groups": len(pool), "n_zone_images": len(zone_paths), "block": block, "buffer": buffer}
    return ctx
