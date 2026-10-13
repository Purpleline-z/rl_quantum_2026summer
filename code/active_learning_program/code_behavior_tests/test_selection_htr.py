"""Integrity tests of the selection x graph-learner x HTR study: seed blocks, graph selectors use no hidden outcomes, return distinct pool members, shuffled controls differ,
per-type Laplacian weights reduce to the plain learner."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

NEW = Path(__file__).resolve().parents[2] / "active_learning_studies" / "new_directions"
sys.path.insert(0, str(NEW)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nd_core as core  # noqa: E402

pytestmark = pytest.mark.skipif(not core.single_study.CACHE.exists(), reason="feature cache not built (build_feature_cache.py)")
GRAPH_SELECTORS = ("gdens_unc", "gdens_unc_shuf", "gcover", "gcover_dec", "gcover_dec_shuf", "gvopt_node3", "gvopt_node3_shuf", "vopt_htr2", "vopt_bridge3", "vopt_bridge3_shuf")


@pytest.fixture(scope="module")
def ctx_a():
    import nd_select
    import new_methods_strategies as nm
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as scratch:
        ctx = core.make_ctx(8200, "A", scratch, core.feature_cache()); nd_select.CTX_REF["ctx"] = ctx; nm.CTX["ctx"] = ctx; yield ctx


@pytest.fixture(scope="module")
def pool(ctx_a):
    import judgment_unit_study as ju_study
    model = ctx_a.fit(ctx_a.initial); cands, labeled, cache = ju_study.make_candidates(ctx_a, ctx_a.pool, ctx_a.initial, model, 0); return model, cands, labeled, cache


def test_seed_block_is_new_and_marked_used():
    assert not set(core.SH_SEEDS_ALL) & (core.GL_SEEDS_ALL | set(range(7000, 7336)) | set(range(0, 7000)) - set(range(8200, 8220)))
    assert core.SH_SEEDS_ALL <= core.USED_SEEDS_BY_EARLIER_WORK and len(core.SH_SEEDS_ALL) == 105


def test_graph_selectors_ignore_hidden_outcomes_and_return_distinct_pool_members(ctx_a, pool):
    import nd_select
    model, cands, labeled, cache = pool; stripped = [{k: v for k, v in c.items() if k != "outcomes"} for c in cands]; ids = {c["pair_id"] for c in cands}
    for name in GRAPH_SELECTORS:
        a = [x["pair_id"] for x in nd_select.SELECTORS[name](cands, labeled, model, cache, 20, 0)]
        b = [x["pair_id"] for x in nd_select.SELECTORS[name](stripped, labeled, model, cache, 20, 0)]
        assert a == b and len(a) == 20 == len(set(a)) and set(a) <= ids, name


def test_graph_nodes_exclude_heldout_images_and_selectors_cover_stand_in_nodes(ctx_a, pool):
    import new_methods_strategies as nm
    import nd_graph_learner as G
    import nd_select_graph as sg
    model, cands, labeled, cache = pool; gd = G.graph_for(ctx_a, 10, False)
    hashes = G.content_hashes(gd.paths); bad = G.excluded_identities(ctx_a); assert all(hashes[p] not in bad for p in gd.paths)
    missing = [x["img1"] for x in cands if str(x["img1"]) not in gd.index] + [x["img2"] for x in cands if str(x["img2"]) not in gd.index]
    for p in missing[:3]: assert sg._near(gd, p, cache, False).max() < len(gd.paths)


def test_shuffled_controls_change_the_graph_quantities(ctx_a, pool):
    import nd_select
    model, cands, labeled, cache = pool
    for name in ("gdens_unc", "gcover_dec", "gvopt_node3", "vopt_bridge3"):
        a = [x["pair_id"] for x in nd_select.SELECTORS[name](cands, labeled, model, cache, 40, 0)]
        b = [x["pair_id"] for x in nd_select.SELECTORS[name + "_shuf"](cands, labeled, model, cache, 40, 0)]
        assert a != b, name


def test_htr_weight_moves_budget_to_htr(ctx_a, pool):
    import nd_select
    model, cands, labeled, cache = pool; kind = {c["pair_id"]: int(c["type_idx"]) for c in cands}
    base = [kind[x["pair_id"]] for x in nd_select.SELECTORS["vopt_u"](cands, labeled, model, cache, 30, 0)].count(4)
    heavy = [kind[x["pair_id"]] for x in nd_select.SELECTORS["vopt_htr4"](cands, labeled, model, cache, 30, 0)].count(4)
    assert heavy >= base


def test_per_type_weights_reduce_to_plain_laplacian_learner(ctx_a):
    import nd_graph_learner as G
    import nd_core as c
    test = c.TestSet(ctx_a); lab = c.random_labelled(ctx_a, 20, 0)
    d0 = np.asarray(G.LEARNERS["lap3_k10"](ctx_a, lab, 1)(test)); d1 = np.asarray(G._mlp_learner(lap=3.0, lap_w={4: 1.0})(ctx_a, lab, 1)(test)); d2 = np.asarray(G.LEARNERS["lap3h10_k10"](ctx_a, lab, 1)(test))
    assert np.allclose(d0, d1, atol=1e-5) and not np.allclose(d0, d2, atol=1e-4)
