"""Integrity tests of the red-team audit tooling: seed block, time-blocked split leaks nothing, graph exclusion hook, anchor filtering, Holm helper."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

NEW = Path(__file__).resolve().parents[2] / "active_learning_studies" / "new_directions"; GE = Path(__file__).resolve().parents[2] / "active_learning_studies" / "pair_disjoint_not_image_disjoint" / "graph_exploration"
sys.path.insert(0, str(NEW)); sys.path.insert(0, str(GE)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nd_core as core  # noqa: E402

pytestmark = pytest.mark.skipif(not core.single_study.CACHE.exists(), reason="feature cache not built (build_feature_cache.py)")


def test_seed_block_is_new():
    assert set(core.RT_DEV_SEEDS) == set(range(8800, 8820)) and set(core.RT_CONFIRM) == set(range(8850, 8885)) and core.RT_SEEDS_ALL <= core.USED_SEEDS_BY_EARLIER_WORK
    assert not core.RT_SEEDS_ALL & (core.SH_SEEDS_ALL | core.GL_SEEDS_ALL)


def test_image_name_parsing():
    import rt_core as rt
    assert rt.run_of("/x/Trajectories/2022-02-04/115_RR220204A_441C_1738.bmp") == "RR220204A" and rt.frame_of("/x/Trajectories/2022-02-04/115_RR220204A_441C_1738.bmp") == 1738
    assert rt.frame_of("/x/Trajectories/2022-02-04/133_RR220204A_467C_1754-01.bmp") == 1754 and rt.session_of("/x/Trajectories/2022-02-04/a.bmp") == "2022-02-04"


def test_blocked_split_keeps_a_buffer_and_excludes_zone_images_from_the_graph():
    import nd_graph_learner as G, rt_blocked as rb, rt_core as rt
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as scratch:
        ctx = rb.make_blocked_ctx(8800, scratch, core.feature_cache(), 100, 25); test = core.TestSet(ctx)
        used = {x for j in ctx.initial + ctx.pool for x in (ctx.info[j]["img1"], ctx.info[j]["img2"])}; timg = set(test.img1) | set(test.img2)
        assert not used & timg
        blocks = set(map(tuple, ctx.blocked["test_blocks"]))
        for p in used:                       # no training image within 25 frames of any test block of its run
            r, f = rt.run_of(p), rt.frame_of(p)
            if r is None: continue
            assert not any(r == tr and tb * 100 - 25 <= f < (tb + 1) * 100 + 25 for tr, tb in blocks), p
        gd = G.graph_for(ctx, 10, False); hashes = G.content_hashes(gd.paths); assert not {hashes[p] for p in gd.paths} & ctx._extra_excluded_hashes
        zone_frames = [(rt.run_of(p), rt.frame_of(p)) for p in gd.paths if rt.run_of(p)]
        assert len(gd.paths) < 1180 and all(r is not None for r, _ in zone_frames)


def test_anchor_filter_and_holm():
    import rt_confirm as C
    p = C.holm(np.array([0.01, 0.04, 0.03])); assert np.allclose(p, [0.03, 0.06, 0.06]) and (p >= np.array([0.01, 0.04, 0.03])).all()
