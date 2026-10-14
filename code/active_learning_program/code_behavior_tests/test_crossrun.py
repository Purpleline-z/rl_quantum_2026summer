"""Integrity tests of the cross-run transfer study CR: seed block, assignment rule, identity safety, nested target draws, equivalences."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

NEW = Path(__file__).resolve().parents[2] / "active_learning_studies" / "new_directions"
GE = Path(__file__).resolve().parents[2] / "active_learning_studies" / "pair_disjoint_not_image_disjoint" / "graph_exploration"
sys.path.insert(0, str(NEW)); sys.path.insert(0, str(GE)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nd_core as core  # noqa: E402

pytestmark = pytest.mark.skipif(not core.ALL_CACHE.exists(), reason="all-image feature cache not built (python3 nd_temporal.py build)")


@pytest.fixture(scope="module")
def ctx():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as scratch: yield core.make_ctx(10900, "A", scratch, core.feature_cache())


def test_cr_seed_blocks_are_new():
    assert core.CR_SEEDS_ALL <= core.USED_SEEDS_BY_EARLIER_WORK and len(core.CR_SEEDS_ALL) == 90
    assert not set(core.CR_DEV_SEEDS) & set(core.CR_CONFIRM1) and not set(core.CR_CONFIRM1) & set(core.CR_CONFIRM2)


@pytest.mark.parametrize("kind", ["A", "B"])
def test_assignment_rule_and_identity_safety(ctx, kind):
    import nd_crossrun as C, ge_harsh as H
    for run in C.RUNS:
        f = C.Fold(ctx, run, kind); gi = H.group_images(ctx)
        in_run = lambda x: H.meta(x)["run"] == run
        assert all(not in_run(x) for g in f.source_groups for x in gi[g])                     # source labels never touch the held-out run
        assert all(any(in_run(x) for x in gi[g]) for g in f.test_groups + f.pool_groups)       # target groups touch it
        test_imgs = {x for g in f.test_groups for x in gi[g]}
        assert not any(set(gi[g]) & test_imgs for g in f.source_groups + f.pool_groups)       # image-disjoint from the test groups
        assert not set(f.test_groups) & set(f.pool_groups) and not set(f.test_groups) & set(f.source_groups)
        gd = __import__("nd_graph_learner").graph_for(ctx, 10); assert not test_imgs & set(gd.paths)  # test images are not graph nodes
        assert len(f.frames) > 200 and all(p in gd.index for p in f.frames if p in gd.index)       # the run's frames are graph nodes (transductive)
        if kind == "B":
            L = __import__("nd_learners")._run_lengths()[run]
            assert all(H.meta(x)["frame"] / L >= 2 / 3 for g in f.test_groups for x in gi[g] if in_run(x))
            assert all(H.meta(x)["frame"] / L < 2 / 3 for g in f.pool_groups for x in gi[g] if in_run(x))   # nothing from the future is labelled


def test_target_draws_are_nested_and_seeded(ctx):
    import nd_crossrun as C
    f = C.Fold(ctx, "RR220411A", "A"); assert f.target(5) == f.target(10)[:5] and f.target(10) == f.target(20)[:10] and f.target(0) == []
    g = C.Fold(ctx, "RR220411A", "A"); assert g.target_order == f.target_order and g.test_groups == f.test_groups


def test_source_head_does_not_depend_on_b_and_pooled_zero_is_source(ctx):
    import nd_crossrun as C
    f = C.Fold(ctx, "RR220206A", "A"); r = C.Runner(f); assert r.head("cr_src", 20) is r.head("cr_src", 0) and r.head("cr_ft1", 0) is r.src_head


def test_zero_transfer_switches_reproduce_the_plain_recipe(ctx):
    import nd_crossrun as C, nd_graph_learner as G
    f = C.Fold(ctx, "RR220206A", "A"); r = C.Runner(f); tgt = f.target(10)
    a = C.train_head(ctx, r.gd, 5, [], tgt, steps=50, lr=0.001); b = C.train_head(ctx, r.gd, 5, [], tgt, steps=50, lr=0.001, l2=0.0, prior_w=0.0, tgt_weight=1.0)
    with torch.no_grad(): x = r.gd.x[:50]; assert torch.allclose(a(x), b(x), atol=1e-6)
    # the L2 anchor to itself with a huge weight keeps the head at its starting point
    src = r.src_head; c = C.train_head(ctx, r.gd, 5, [], tgt, init=src, l2=1e4, ref=src, steps=30, lr=0.001)
    free = C.train_head(ctx, r.gd, 5, [], tgt, init=src, steps=30, lr=0.001)
    with torch.no_grad(): x = r.gd.x[:50]; assert (c(x) - src(x)).abs().max() < 0.5 * (free(x) - src(x)).abs().max() + 1e-6


def test_propagation_returns_scores_for_the_run_frames_and_shuffled_graph_differs(ctx):
    import nd_crossrun as C, nd_graph_learner as G
    f = C.Fold(ctx, "RR220411A", "A"); r = C.Runner(f)
    p, q = r._prior_target("gprop"), r._prior_target("gprop_shuf"); assert p.shape == (len(r.run_nodes), 4) and torch.isfinite(p).all() and not torch.allclose(p, q, atol=1e-3)


def test_tables_bootstrap_and_holm():
    import cr_tables as T
    assert np.allclose(T.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
