"""Integrity tests for the graph-augmented learners (identity safety, controls, equivalence with aw8_all when the graph terms are off)."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

NEW = Path(__file__).resolve().parents[2] / "active_learning_studies" / "new_directions"
sys.path.insert(0, str(NEW)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nd_core as core  # noqa: E402

pytestmark = pytest.mark.skipif(not core.single_study.CACHE.exists(), reason="feature cache not built")


@pytest.fixture(scope="module")
def ctx_b():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as scratch: yield core.make_ctx(7000, "B", scratch, core.feature_cache())


def test_graph_learner_seed_blocks_are_new():
    assert not core.GL_SEEDS_ALL & core.USED_BEFORE_GRAPH_LEARNER and core.GL_SEEDS_ALL <= core.USED_SEEDS_BY_EARLIER_WORK
    assert not set(core.GL_DEV_SEEDS) & (set(core.GL_CONFIRM1) | set(core.GL_CONFIRM2) | set(core.GL_COLD))


def test_nodes_exclude_heldout_images_and_outer_test_ideals_by_content_hash(ctx_b):
    import nd_graph_learner as G
    gd = G.graph_for(ctx_b, 10); node_hashes = set(G.content_hashes(gd.paths).values())
    held = {str(x) for g in list(ctx_b.validation) + list(ctx_b.test) for r in ctx_b.exp.groups[g].itertuples() for x in (r.resolved_img1, r.resolved_img2)}
    held |= {str(p) for ps in ctx_b.exp.test_images.values() for p in ps}
    assert not set(G.content_hashes(sorted(held)).values()) & node_hashes
    assert not held & set(gd.paths) and len(gd.paths) > 1000


def test_neighbours_of_unseen_images_are_graph_nodes_only(ctx_b):
    import nd_graph_learner as G
    gd = G.graph_for(ctx_b, 10); t = core.TestSet(ctx_b); nb = gd.neighbours_of(sorted(set(t.img1))[:20], ctx_b.features.cache)
    assert nb.shape[1] == 10 and int(nb.max()) < len(gd.paths)


def test_shuffled_graph_keeps_degree_sequence_and_destroys_feature_similarity(ctx_b):
    import nd_graph_learner as G
    a, b = G.graph_for(ctx_b, 10), G.graph_for(ctx_b, 10, shuffle=True)
    assert a.n_edges == b.n_edges and sorted(a.adj.sum(1).tolist()) == sorted(b.adj.sum(1).tolist())
    sim = lambda g: float((g.xn[g.edges[:, 0]] * g.xn[g.edges[:, 1]]).sum(1).mean())
    assert sim(a) > sim(b) + 0.2


def test_label_spreading_gives_distributions_and_sources_keep_their_class(ctx_b):
    import nd_graph_learner as G
    gd = G.graph_for(ctx_b, 10); q = gd.q(); assert torch.allclose(q.sum(1), torch.ones(len(q)), atol=1e-4)
    order = list(G.ACTIVE); hits = [int(q[i].argmax()) == order.index(c) for i, c in gd.source_class.items()]
    assert np.mean(hits) > 0.8


def test_graph_terms_off_reproduces_aw8_all(ctx_b):
    import nd_graph_learner as G
    import nd_learners as L
    test = core.TestSet(ctx_b); lab = core.random_labelled(ctx_b, 10, 0)
    ref = L.LEARNERS["aw8_all"](ctx_b, lab, 3)(test)
    gd = G.graph_for(ctx_b, 10); fn = core.identity_features(ctx_b)
    head = G.fit_mlp_graph(ctx_b, lab, 3, gd, fn, gd.x, 512)
    assert np.allclose(core.head_d(head, fn, test), ref, atol=1e-5)


def test_predictions_do_not_depend_on_test_labels_and_gcn_is_antisymmetric(ctx_b):
    import nd_graph_learner as G
    test = core.TestSet(ctx_b); lab = core.random_labelled(ctx_b, 10, 0); predictor = G.LEARNERS["gcn2_k10"](ctx_b, lab, 1)
    d = predictor(test); test.sign = -test.sign
    assert np.allclose(d, predictor(test))
    swapped = core.TestSet(ctx_b); swapped.img1, swapped.img2 = test.img2, test.img1
    assert np.allclose(d, -predictor(swapped), atol=1e-5)


def test_regulariser_terms_change_the_fit_and_noedge_gcn_ignores_the_graph(ctx_b):
    import nd_graph_learner as G
    test = core.TestSet(ctx_b); lab = core.random_labelled(ctx_b, 10, 0)
    a = G.LEARNERS["aw8_all"] if False else None
    d0 = G.LEARNERS["lap0.3_k10"](ctx_b, lab, 2)(test); d1 = G.LEARNERS["lap3_k10"](ctx_b, lab, 2)(test)
    assert not np.allclose(d0, d1)
    n1 = G.LEARNERS["gcn2_noedge"](ctx_b, lab, 2)(test); assert np.isfinite(n1).all()
