"""Integrity tests of the transductive / forward-in-time study: seed block, group times, forward contexts (no test image in training, in the inductive graph or after the cut for nodes), transductive
node sets contain the test images and no label, ``fit_td`` reproduces ``lap3_k10_st4`` exactly, score propagation limits and shuffled control."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

NEW = Path(__file__).resolve().parents[2] / "active_learning_studies" / "new_directions"
sys.path.insert(0, str(NEW)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nd_core as core  # noqa: E402

needs_cache = pytest.mark.skipif(not core.single_study.CACHE.exists() or not core.ALL_CACHE.exists(), reason="feature caches not built (build_feature_cache.py, nd_temporal.py build)")


def test_seed_block():
    assert core.TD_SEEDS_ALL <= core.USED_SEEDS_BY_EARLIER_WORK and len(core.TD_SEEDS_ALL) == 90
    assert (min(core.TD_DEV_SEEDS), max(core.TD_DEV_SEEDS), min(core.TD_CONFIRM1), max(core.TD_CONFIRM1), min(core.TD_CONFIRM2), max(core.TD_CONFIRM2)) == (10500, 10519, 10530, 10564, 10565, 10599)


@needs_cache
def test_group_times_forward_split_and_node_sets():
    import td_core as td
    torch.set_num_threads(1); cache = core.feature_cache()
    with tempfile.TemporaryDirectory() as s:
        ctx = td.make_forward_ctx(10500, 54, s, cache); order = td.calendar_order(ctx); assert len(order) == 168
        times = [td.group_time(ctx, g) for g in order]; assert times == sorted(times)                       # every group has a time; calendar order is monotone
        f = ctx.forward; cut = f["cut_time"]; assert all(t <= cut for t in times[:f["n_train"]]) and all(t >= cut for t in times[f["n_train"]:])
        test_imgs = {p for g in f["near"] + f["far"] for p in td.group_images(ctx, g)}; train_imgs = {str(x) for j in ctx.initial + ctx.pool for x in (ctx.info[j]["img1"], ctx.info[j]["img2"])}
        assert not test_imgs & train_imgs and not set(f["near"]) & set(f["far"])
        paths = lambda m: set(td.td_graph(ctx, m).paths)
        ind, tt, tf, std = paths("ind"), paths("trans_test"), paths("trans_fut"), paths("std")
        assert not ind & test_imgs and test_imgs <= tt and test_imgs <= tf and not std & test_imgs     # identity rule for ind/std, test images are unlabelled nodes in the transductive sets
        assert all(td.img_time(p) is None or td.img_time(p) <= cut for p in ind)                           # inductive graph: nothing later than the cut
        assert ind <= (tt | test_imgs) and tt <= tf and (std | test_imgs) <= tf
        outer = {str(p) for ps in ctx.exp.test_images.values() for p in ps}; assert not outer & (ind | tt | tf | std)       # outer-test ideal images are never nodes


@needs_cache
def test_fit_td_reproduces_baseline_and_propagation_limits():
    import nd_graph_learner as G, td_core as td
    torch.set_num_threads(1); cache = core.feature_cache()
    with tempfile.TemporaryDirectory() as s:
        ctx = core.make_ctx(10501, "A", s, cache); lab = core.random_labelled(ctx, 10, 0); test = core.TestSet(ctx); gd = G.graph_for(ctx, 10, False); f = core.identity_features(ctx)
        ref = G.LEARNERS["lap3_k10_st4"](ctx, lab, 3)(test); h = td.fit_td(ctx, lab, 3, gd); assert np.array_equal(ref, core.head_d(h, f, test))
        assert np.array_equal(td.td_graph(ctx, "std").paths, gd.paths)                                      # the std node set is the repository's graph
        s0 = td.propagate_scores(gd, h, 0.0); assert torch.allclose(s0, h(gd.x), atol=1e-6)                   # alpha = 0: head scores
        s5 = td.propagate_scores(gd, h, 0.5); s5s = td.propagate_scores(gd, h, 0.5, shuffle_seed=ctx.seed); assert not torch.allclose(s5, s5s) and s5.std() < s0.std() * 1.0001
        assert td.td_graph(ctx, "trans").n_edges > gd.n_edges and set(gd.paths) < set(td.td_graph(ctx, "trans").paths)


@needs_cache
def test_window_nodes_nesting_and_targeted_shuffle_keeps_degrees():
    import td_core as td
    torch.set_num_threads(1); cache = core.feature_cache()
    with tempfile.TemporaryDirectory() as s:
        ctx = td.make_forward_ctx(10502, 40, s, cache); f = ctx.forward; ctx._window = "FA"
        ind = set(td.td_graph(ctx, "ind").paths); wa = set(td.td_graph(ctx, "trans_win_FA").paths); wb = set(td.td_graph(ctx, "trans_win_FB").paths); tf = set(td.td_graph(ctx, "trans_fut").paths); tt = set(td.td_graph(ctx, "trans_test").paths)
        assert ind <= wa <= wb <= tf and tt <= tf and ind <= tt
        end = td.group_time(ctx, f["near"][-1]); assert all(td.img_time(p) is None or td.img_time(p) <= end for p in wa)   # causal: nothing after the last scored group of the window
        test_imgs = {p for g in f["near"] for p in td.group_images(ctx, g)}; assert test_imgs <= wa
        gd = td.td_graph(ctx, "trans_win_FA"); sh = td.td_graph(ctx, "trans_win_FA", 10, "test")
        idx = np.array([gd.index[p] for p in td.heldout_paths(ctx) if p in gd.index]); rest = np.setdiff1d(np.arange(len(gd.paths)), idx)
        assert np.array_equal(np.sort(gd.adj.sum(1).numpy()[idx]), np.sort(sh.adj.sum(1).numpy()[idx])) and np.array_equal(gd.adj.sum(1).numpy()[rest], sh.adj.sum(1).numpy()[rest])
        assert not torch.equal(gd.adj, sh.adj)
