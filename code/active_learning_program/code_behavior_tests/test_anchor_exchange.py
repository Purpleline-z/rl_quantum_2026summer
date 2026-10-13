"""Anchor-exchange study: seed registry, universes (outer-test ideals never anchors/nodes), exact reproduction of lap3_k10_st4, graph nodes independent of the anchor subset, rules do not peek at labels."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

NEW = Path(__file__).resolve().parents[2] / "active_learning_studies" / "new_directions"
sys.path.insert(0, str(NEW)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nd_core as core  # noqa: E402
import ae_analyze as aa  # noqa: E402

needs_cache = pytest.mark.skipif(not core.single_study.CACHE.exists() or not core.ALL_CACHE.exists(), reason="feature caches not built (nd_temporal.py build)")


def test_seed_block_registered_and_disjoint():
    assert set(core.AE_DEV_SEEDS) == set(range(8700, 8720)) and set(core.AE_CONFIRM1) == set(range(8750, 8785))
    assert not core.AE_SEEDS_ALL & (core.SH_SEEDS_ALL | core.GL_SEEDS_ALL | core.RESERVED_FOR_PARALLEL_STUDIES) and core.AE_SEEDS_ALL <= core.USED_SEEDS_BY_EARLIER_WORK


@pytest.fixture(scope="module")
def ctx():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as scratch: yield core.make_ctx(8701, "A", scratch, core.feature_cache())


@needs_cache
def test_universe_excludes_outer_test_images_and_144_universe_matches_original_anchors(ctx):
    import ae_core as ae
    u, u144 = ae.universe(ctx, "no_test"), ae.universe(ctx, "all144"); test = {str(p) for ps in ctx.exp.test_images.values() for p in ps}
    assert ae.n_anchors(u) == 116 and ae.n_anchors(u144) == 144 and not {p for ps in u.values() for p in ps} & test and test <= {p for ps in u144.values() for p in ps}
    refs, _ = core.anchors(ctx, core.identity_features(ctx), True); assert {c: len(v) for c, v in refs.items()} == {c: len(v) for c, v in u144.items()}


@needs_cache
def test_all_anchors_selected_reproduces_lap3_k10_st4_exactly(ctx):
    import ae_core as ae, nd_graph_learner as gl
    test = core.TestSet(ctx); lab = core.random_labelled(ctx, 10, 0); gd = gl.graph_for(ctx, 10)
    head = ae.fit_head(ctx, lab, 5, gd, ae.universe(ctx, "all144")); d = core.head_d(head, core.identity_features(ctx), test)
    assert np.abs(d - gl.LEARNERS["lap3_k10_st4"](ctx, lab, 5)(test)).max() == 0.0


@needs_cache
def test_graph_nodes_do_not_depend_on_anchor_subset_and_never_contain_test_ideals(ctx):
    import ae_core as ae, nd_graph_learner as gl
    gd = gl.graph_for(ctx, 10); u = ae.universe(ctx, "no_test"); test = {str(p) for ps in ctx.exp.test_images.values() for p in ps}
    assert not test & set(gd.paths) and all(p in gd.index for _, p in ae.flat(u))
    n_before = len(gd.paths); ae.random_subset(ctx, u, 12); assert len(gl.graph_for(ctx, 10).paths) == n_before


@needs_cache
def test_subsets_are_class_blind_reproducible_and_rules_return_the_requested_count(ctx):
    import ae_core as ae, nd_graph_learner as gl
    u = ae.universe(ctx, "no_test"); a, b = ae.random_subset(ctx, u, 29, 1), ae.random_subset(ctx, u, 29, 1); assert a == b and ae.n_anchors(a) == 29 and ae.random_subset(ctx, u, 29, 2) != a
    gd, gs = gl.graph_for(ctx, 10), gl.graph_for(ctx, 10, True)
    for rule in ("rand", "kcg", "kcg_shuf", "kcf", "typi", "amb", "amb_shuf", "htrq", "htrq_shuf"):
        s1, s2 = ae.active_subset(ctx, u, rule, 36, gd, gs), ae.active_subset(ctx, u, rule, 36, gd, gs); assert ae.n_anchors(s1) == 36 and s1 == s2
    assert ae.active_subset(ctx, u, "kcg", 36, gd, gs) != ae.active_subset(ctx, u, "kcg_shuf", 36, gd, gs)


@needs_cache
def test_spreading_posterior_depends_only_on_the_labelled_anchors(ctx):
    import ae_core as ae, nd_graph_learner as gl
    gd = gl.graph_for(ctx, 10); lab = {gd.index[p]: 0 for p in list(ae.universe(ctx)["(1 x 1)"])[:3]} | {gd.index[p]: 3 for p in list(ae.universe(ctx)["HTR"])[:3]}
    q1 = ae.spreading_posterior(gd, lab); q2 = ae.spreading_posterior(gd, dict(reversed(list(lab.items())))); assert np.allclose(q1, q2) and np.allclose(q1.sum(1), 1)


@needs_cache
def test_head_without_anchors_trains_and_external_accuracy_ignores_missing_classes(ctx):
    import ae_core as ae, nd_graph_learner as gl
    gd = gl.graph_for(ctx, 10); u = ae.universe(ctx, "no_test"); test = core.TestSet(ctx); lab = core.random_labelled(ctx, 10, 0)
    h0 = ae.fit_head(ctx, lab, 3, gd, {c: [] for c in u}); assert np.isnan(ae.external_type_accuracy(ctx, h0, {c: [] for c in u})["ext_acc"])
    two = {c: (ps if c in ("HTR", "(1 x 1)") else []) for c, ps in u.items()}; h2 = ae.fit_head(ctx, lab, 3, gd, two); r = ae.external_type_accuracy(ctx, h2, two); assert 0 <= r["ext_acc"] <= 1


def test_exchange_rate_recovers_a_synthetic_surface_and_holm_helper():
    ns, bs = [0, 12, 29, 58, 116], [0, 10, 20, 40, 60]; rng = np.random.default_rng(0)
    M = np.array([[0.70 + 0.04 * np.log(1 + n) / np.log(117) + 0.05 * np.log((30 + b) / 30) for b in bs] for n in ns]); Mbar = np.stack([M + rng.normal(0, 1e-4, M.shape) for _ in range(5)]).mean(0)
    est = aa.exchange_once(Mbar, ns, bs, 12, 58, 20); gain = 0.04 * (np.log(59) - np.log(13)) / np.log(117)
    assert est["extra_judgments"] == pytest.approx((50 * np.exp(gain / 0.05) - 30) - 20, rel=0.02)
    h = aa.holm({"a": .001, "b": .02, "c": .5}); assert h["a"] == pytest.approx(.003) and h["b"] == pytest.approx(.04)
