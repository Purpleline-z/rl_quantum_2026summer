"""Graph-mechanism learners: own seed blocks, equivalence with lap<L>_k10, identity safety of every topology, quality measure, controls differ from the graph learner."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

NEW = Path(__file__).resolve().parents[2] / "active_learning_studies" / "new_directions"
sys.path.insert(0, str(NEW)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nd_core as core  # noqa: E402

pytestmark = pytest.mark.skipif(not core.single_study.CACHE.exists() or not (NEW / "results" / "feature_layers_all.pt").exists(), reason="feature caches not built")


@pytest.fixture(scope="module")
def ctx_a():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as scratch: yield core.make_ctx(8000, "A", scratch, core.feature_cache())


def test_seed_blocks_are_new_and_disjoint():
    assert core.GM_SEEDS_ALL <= core.USED_SEEDS_BY_EARLIER_WORK and not core.GM_SEEDS_ALL & core.GL_SEEDS_ALL
    assert not set(core.GM_DEV_SEEDS) & (set(core.GM_CONFIRM1) | set(core.GM_CONFIRM2)) and not set(core.GM_CONFIRM1) & set(core.GM_CONFIRM2)
    assert not core.GM_SEEDS_ALL & set(range(8200, 8600))


def test_default_gm_graph_is_the_graph_learner_graph_and_reproduces_lap3_k10(ctx_a):
    import nd_graph_learner as G
    import nd_graph_mechanism as M
    lab = core.random_labelled(ctx_a, 20, 0); test = core.TestSet(ctx_a)
    d1 = M.LEARNERS["gm_lap3_n0"](ctx_a, lab, 5)(test); d2 = G.LEARNERS["lap3_k10"](ctx_a, lab, 5)(test)
    assert np.allclose(d1, d2, atol=1e-6)
    d3 = M.LEARNERS["gm_lap3_n3"](ctx_a, lab, 5)(test); d4 = G.LEARNERS["lap3_k10_n3"](ctx_a, lab, 5)(test); assert np.allclose(d3, d4, atol=1e-6)


@pytest.mark.parametrize("spec", [{"construct": "mutual", "k": 15}, {"construct": "eps", "deg": 10}, {"construct": "snn", "k": 10, "snn_k": 20}, {"construct": "gauss"}, {"construct": "lscale"},
                                  {"construct": "bmatch", "b": 10}, {"whiten": 32}, {"feat": "l3"}, {"feat": "l2"}, {"noise": 2.0}])
def test_every_topology_uses_only_allowed_nodes_and_is_valid(ctx_a, spec):
    import nd_graph_learner as G
    import nd_graph_mechanism as M
    base = G.graph_for(ctx_a, 10); edges, w = M.build_edges(base, spec); n = len(base.paths)
    assert edges.shape[1] == 2 and int(edges.max()) < n and int(edges.min()) >= 0 and bool((edges[:, 0] < edges[:, 1]).all()) and bool((w > 0).all()) and len(torch.unique(edges[:, 0] * n + edges[:, 1])) == len(edges)
    held = {str(x) for g in list(ctx_a.validation) + list(ctx_a.test) for r in ctx_a.exp.groups[g].itertuples() for x in (r.resolved_img1, r.resolved_img2)} | {str(p) for ps in ctx_a.exp.test_images.values() for p in ps}
    assert not held & set(base.paths)          # nodes (hence every topology) exclude validation/test images and outer-test ideals
    q = M.graph_quality(base, edges, w); assert 0.0 <= q["quality"] <= 1.0


def test_bmatch_is_degree_bounded_and_noise_lowers_quality(ctx_a):
    import nd_graph_learner as G
    import nd_graph_mechanism as M
    base = G.graph_for(ctx_a, 10); e, w = M.build_edges(base, {"construct": "bmatch", "b": 10}); deg = torch.zeros(len(base.paths), dtype=torch.long).index_add_(0, e[:, 0], torch.ones(len(e), dtype=torch.long)).index_add_(0, e[:, 1], torch.ones(len(e), dtype=torch.long))
    assert int(deg.max()) <= 10
    q0 = M.graph_quality(base, *M.build_edges(base, {}))["quality"]; q3 = M.graph_quality(base, *M.build_edges(base, {"noise": 3.0}))["quality"]; assert q0 > q3 + 0.15


def test_placement_filters_partition_the_edges(ctx_a):
    import nd_graph_learner as G
    import nd_graph_mechanism as M
    base = G.graph_for(ctx_a, 10); e, w = M.build_edges(base, {}); sizes = [len(M.filter_edges(base, e, w, p)[0]) for p in ("unl", "it", "ii")]
    assert sum(sizes) == len(e)


def test_non_graph_controls_change_the_head_but_train(ctx_a):
    import nd_graph_mechanism as M
    lab = core.random_labelled(ctx_a, 20, 0); test = core.TestSet(ctx_a); base = M.LEARNERS["gm_lap3_n0"](ctx_a, lab, 5)(test)
    for name in ("gm_noise3", "gm_mix3", "gm_wd1e2", "gm_decay3"):
        d = M.LEARNERS[name](ctx_a, lab, 5)(test); assert np.isfinite(d).all() and not np.allclose(d, base)
