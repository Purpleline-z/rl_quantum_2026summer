"""Integrity tests of the temporal-graph study: seed block, temporal edges only join graph nodes (never a held-out image), strict buffer, shuffled controls keep degrees and runs, the
baseline specification reproduces ``lap3_k10_st4`` exactly, chain / curvature edge semantics."""
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pytest
import torch

NEW = Path(__file__).resolve().parents[2] / "active_learning_studies" / "new_directions"
sys.path.insert(0, str(NEW)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nd_core as core  # noqa: E402

needs_cache = pytest.mark.skipif(not core.single_study.CACHE.exists() or not core.ALL_CACHE.exists(), reason="feature caches not built (build_feature_cache.py, nd_temporal.py build)")


def fake_nodes(spec):
    """spec: list of (run, index, temperature, frame) -> nd_temporal_graph.Nodes-like object."""
    import nd_temporal_graph as T
    nd = T.Nodes.__new__(T.Nodes); nd.n = len(spec); nd.run = np.array([s[0] for s in spec]); nd.idx = np.array([s[1] for s in spec]); nd.temp = np.array([s[2] for s in spec]); nd.frame = np.array([s[3] for s in spec])
    nd.is_traj = nd.idx >= 0; nd.runs = sorted({r for r in nd.run if r}); nd.by_run = {r: np.where(nd.run == r)[0] for r in nd.runs}; return nd


def test_seed_block_is_new_reserved_blocks_untouched_and_marked_used():
    assert core.TG_SEEDS_ALL <= core.USED_SEEDS_BY_EARLIER_WORK and len(core.TG_SEEDS_ALL) == 55
    assert not core.TG_SEEDS_ALL & core.TG_RESERVED_PARALLEL
    assert min(core.TG_DEV_SEEDS) == 9200 and max(core.TG_DEV_SEEDS) == 9219 and min(core.TG_CONFIRM) == 9250 and max(core.TG_CONFIRM) == 9284
    assert not core.TG_SEEDS_ALL & (set(range(7000, 7336)) | set(range(8200, 8400)) | set(range(8800, 8885)))


def test_chain_edges_use_index_gap_and_respect_runs():
    import nd_temporal_graph as T
    nd = fake_nodes([("A", 1, 100, 10), ("A", 2, 100, 12), ("A", 4, 101, 15), ("B", 3, 100, 10), ("B", 4, 100, 11), ("", -1, -1, -1)])
    i, j, w = T.chain_edges(nd, 1); assert sorted(zip(i.tolist(), j.tolist())) == [(0, 1), (3, 4)]
    i, j, w = T.chain_edges(nd, 2); assert sorted(zip(i.tolist(), j.tolist())) == [(0, 1), (1, 2), (3, 4)]            # index gap 2 bridges the hole at index 3 of run A; run B never joins run A
    i, j, w = T.chain_edges(nd, 3, decay=True); assert (w > 0).all() and (w <= 1).all() and w.min() < w.max()          # weights decay with the index gap


def test_curvature_triples_are_consecutive_frames_of_one_run():
    import nd_temporal_graph as T
    nd = fake_nodes([("A", 1, 1, 1), ("A", 2, 1, 2), ("A", 3, 1, 3), ("A", 5, 1, 5), ("B", 1, 1, 1), ("B", 2, 1, 2)])
    assert T.curvature_triples(nd, 1).tolist() == [[0, 1, 2]]


def test_within_run_permutation_keeps_runs_and_fixes_non_trajectory_nodes():
    import nd_temporal_graph as T
    nd = fake_nodes([("A", k, 1, k) for k in range(1, 40)] + [("B", k, 1, k) for k in range(1, 30)] + [("", -1, -1, -1)] * 3)
    for local in (False, True):
        perm = T.within_run_permutation(nd, 5, local=local); assert sorted(perm.tolist()) == list(range(nd.n)); assert (perm[nd.run == ""] == np.arange(nd.n)[nd.run == ""]).all()
        for r in nd.runs: assert set(perm[nd.by_run[r]]) == set(nd.by_run[r])
    local = T.within_run_permutation(nd, 5, local=True); moved = np.abs(nd.idx[local[nd.is_traj]] - nd.idx[nd.is_traj]); assert moved.max() <= 2 * T.INTER_WINDOW and (moved > 0).any()


def test_near_heldout_marks_only_same_run_within_buffer():
    import nd_temporal_graph as T
    nd = fake_nodes([("RR1", 1, 1, 100), ("RR1", 2, 1, 120), ("RR1", 3, 1, 140), ("RR2", 1, 1, 100)])
    held = ["/x/009_RR1_500C_110.bmp"]; assert T.near_heldout(nd, held, None, 25).tolist() == [True, True, False, False]


@pytest.fixture(scope="module")
def ctx_a():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as scratch:
        yield core.make_ctx(9200, "A", scratch, core.feature_cache())


@needs_cache
def test_temporal_edges_join_graph_nodes_only_never_heldout_images(ctx_a):
    import nd_graph_learner as G
    import nd_temporal_graph as T
    import rt_core as rt
    gd = G.graph_for(ctx_a, 10, False); held = {str(x) for x in rt.heldout_images(ctx_a)}; bad = G.excluded_identities(ctx_a); hashes = G.content_hashes(gd.paths)
    assert not set(gd.paths) & held and all(hashes[p] not in bad for p in gd.paths)
    for name in ("tg_u2", "tg_u5", "tg_dec5", "tg_tbin", "tg_novel5", "tg_u2_shuf", "tg_u2_lshuf", "tg_u2_krand"):
        es = T.edge_sets(ctx_a, gd, T.SPEC_OF[name]); i, j, w = es["temporal"]; assert len(i) > 0 and int(max(i.max(), j.max())) < len(gd.paths) and (i != j).all()
    es = T.edge_sets(ctx_a, gd, T.SPEC_OF["tg_curv"]); assert es["triples"].max() < len(gd.paths)


@needs_cache
def test_shuffled_control_keeps_degree_sequence_within_runs_and_destroys_time(ctx_a):
    import nd_graph_learner as G
    import nd_temporal_graph as T
    gd = G.graph_for(ctx_a, 10, False); real = T.edge_sets(ctx_a, gd, T.SPEC_OF["tg_u2"]); shuf = T.edge_sets(ctx_a, gd, T.SPEC_OF["tg_u2_shuf"]); nd = real["nodes"]
    assert len(real["temporal"][0]) == len(shuf["temporal"][0])
    deg = lambda es: np.bincount(np.concatenate([es["temporal"][0].numpy(), es["temporal"][1].numpy()]), minlength=nd.n)
    for r in nd.runs:   # the multiset of degrees per run is preserved
        ids = nd.by_run[r]; assert sorted(deg(real)[ids].tolist()) == sorted(deg(shuf)[ids].tolist())
    i, j, _ = shuf["temporal"]; assert (nd.run[i.numpy()] == nd.run[j.numpy()]).all(); real_gap = np.abs(nd.idx[real["temporal"][0].numpy()] - nd.idx[real["temporal"][1].numpy()]); shuf_gap = np.abs(nd.idx[i.numpy()] - nd.idx[j.numpy()])
    assert real_gap.max() <= 2 and shuf_gap.mean() > 10


@needs_cache
def test_strict_rule_leaves_no_edge_within_buffer_of_a_heldout_image(ctx_a):
    import nd_graph_learner as G
    import nd_temporal_graph as T
    import rt_core as rt
    gd = G.graph_for(ctx_a, 10, False); full = T.edge_sets(ctx_a, gd, T.SPEC_OF["tg_u2"]); strict = T.edge_sets(ctx_a, gd, T.SPEC_OF["tg_u2_strict"]); nd = full["nodes"]
    bad = T.near_heldout(nd, rt.heldout_images(ctx_a), gd, T.BUFFER); i, j, _ = strict["temporal"]
    assert len(i) < len(full["temporal"][0]) and not bad[i.numpy()].any() and not bad[j.numpy()].any()


@needs_cache
def test_blocked_context_has_no_node_within_the_buffer_of_an_in_block_test_image():
    """Property of the red-team blocked split that holds: no graph node within 25 frames of a test image that lies INSIDE a held-out block.  (Images outside the blocks that are the partners of cross-block test
    groups are held out too, but their neighbours stay in the graph: the study reports the 'both images in block' stratum for this reason.)"""
    import nd_graph_learner as G
    import nd_temporal_graph as T
    import rt_blocked as rb
    import rt_core as rt
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as scratch:
        ctx = rb.make_blocked_ctx(9200, scratch, core.feature_cache()); gd = G.graph_for(ctx, 10, False); es = T.edge_sets(ctx, gd, T.SPEC_OF["tg_u5"]); nd = es["nodes"]; blocks = {tuple(b) for b in ctx.blocked["test_blocks"]}
        inside = [p for p in rt.heldout_images(ctx) if rt.run_of(p) is not None and (rt.run_of(p), rt.frame_of(p) // 100) in blocks]; assert inside
        assert not T.near_heldout(nd, inside, gd, T.BUFFER).any(), "a graph node lies within 25 frames of an in-block test image"
        import tg_run
        test = core.TestSet(ctx); mask = tg_run.in_block_mask(ctx, test); assert 0 < mask.sum() < len(mask)


@needs_cache
def test_baseline_specification_reproduces_the_incumbent_learner_exactly(ctx_a):
    import nd_graph_learner as G
    import nd_temporal_graph as T
    test = core.TestSet(ctx_a); labelled = core.random_labelled(ctx_a, 10, 0)
    d0 = G.LEARNERS["lap3_k10"](ctx_a, labelled, 3)(test); spec = T.make_spec("check", steps=1.0, kind=None); gd = G.graph_for(ctx_a, 10, False)
    head = T.fit_tg(ctx_a, labelled, 3, gd, spec); d1 = core.head_d(head, core.identity_features(ctx_a), test); assert np.abs(d0 - d1).max() == 0.0
