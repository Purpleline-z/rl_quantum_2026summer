"""Graph-endpoint study: seed registry, classification endpoint identity safety and control reproduction, aw8_all equivalence of the reference-anchor head, dose-response node sets, harsh splits, helpers."""
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
import ge_analyze as ga  # noqa: E402

needs_cache = pytest.mark.skipif(not core.single_study.CACHE.exists() or not core.ALL_CACHE.exists(), reason="feature caches not built (nd_temporal.py build)")


def test_seed_blocks_disjoint_and_registered():
    assert set(core.GE_DEV_SEEDS) == set(range(8400, 8420)) and set(core.GE_CONFIRM1) == set(range(8500, 8535)) and set(core.GE_CONFIRM2) == set(range(8550, 8585))
    assert not (set(core.GE_CONFIRM1) | set(core.GE_CONFIRM2)) & (core.USED_BY_SIBLING_SELECTION_SESSION | core.GL_SEEDS_ALL | core.ND_SEEDS_ALL)
    assert set(core.GE_CONFIRM1) <= core.USED_SEEDS_BY_EARLIER_WORK


def test_wilcoxon_holm_and_trend_helpers():
    h = ga.holm({"a": .001, "b": .01, "c": .5}); assert h["a"] == pytest.approx(.003) and h["b"] == pytest.approx(.02) and h["c"] == pytest.approx(.5)
    assert np.isnan(ga.wil([0, 0, 0, 0, 0, 0, 0])) and ga.wil(np.arange(1, 21)) < 1e-3
    rows = [{"split": "A", "seed": s, "variant": v, "auc": .001 * p * (1 + .1 * s), "acc": 0.0} for s in range(10) for v, p in (("all_p0", 0), ("all_p25", 1), ("all_p50", 2), ("all_p100", 4), ("out_p0", 0), ("out_p25", 1), ("out_p50", 2))]
    t = ga.dose_trend(pd.DataFrame(rows), "all", "auc"); assert t.slope.iloc[0] > 0 and t.p.iloc[0] < .01 and t.share_pos.iloc[0] == 1.0


@pytest.fixture(scope="module")
def ctx_b():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as scratch: yield core.make_ctx(8401, "B", scratch, core.feature_cache())


@needs_cache
def test_nn_control_reproduces_the_committed_paper_numbers_and_classification_data_are_identity_safe():
    import ge_classify as gc
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as scratch:
        cache = core.feature_cache()
        for seed, expected in ((42, 24 / 28), (79, 24 / 28), (123, 25 / 28)):
            ctx = core.make_ctx(seed, "B", scratch, cache); ref, test, util = gc.split_arrays(ctx)
            assert len(ref[0]) == 88 and len(test[0]) == 28 and abs(float((gc.predict_1nn(ctx, ref, test) == test[1]).mean()) - expected) < 1e-9
            assert not set(ref[0]) & set(test[0]) and not set(util[0]) & set(test[0]) and not set(ref[0]) & set(util[0])


@needs_cache
def test_graph_protocols_inductive_excludes_test_ideals_transductive_includes_them_unlabelled(ctx_b):
    import ge_classify as gc
    ref, test, util = gc.split_arrays(ctx_b); ind = gc.build_graph(ctx_b, 10, False); tra = gc.build_graph(ctx_b, 10, True)
    assert not set(test[0]) & set(ind.paths) and set(test[0]) <= set(tra.paths)
    refs = {str(p) for ps in ctx_b.exp.references.values() for p in ps}
    for gd in (ind, tra):
        labelled = {gd.paths[i] for i in gd.source_class}; assert labelled == refs & set(gd.paths) and not labelled & set(util[0]) and not labelled & set(test[0])
    held = {str(x) for g in list(ctx_b.validation) + list(ctx_b.test) for r in ctx_b.exp.groups[g].itertuples() for x in (r.resolved_img1, r.resolved_img2)}
    assert not held & set(ind.paths) and not held & set(tra.paths)


@needs_cache
def test_reference_only_head_with_all_ideal_flag_reproduces_aw8_all_and_lap(ctx_b):
    import ge_heads as gh, nd_learners as L, nd_graph_learner as gl
    test = core.TestSet(ctx_b); lab = core.random_labelled(ctx_b, 20, 0); fn = core.identity_features(ctx_b)
    h0 = gh.fit_head(ctx_b, lab, 5, None, fn, None, 512, 0.0, all_ideal=True); assert np.allclose(core.head_d(h0, fn, test), L.LEARNERS["aw8_all"](ctx_b, lab, 5)(test), atol=1e-6)
    gd = gh.make_graph(ctx_b, 10); h1 = gh.fit_head(ctx_b, lab, 5, gd, fn, gd.x, 512, 3.0, all_ideal=True); assert np.allclose(core.head_d(h1, fn, test), gl.LEARNERS["lap3_k10"](ctx_b, lab, 5)(test), atol=1e-6)
    h2 = gh.fit_head(ctx_b, lab, 5, None, fn, None, 512, 0.0, all_ideal=False); assert not np.allclose(core.head_d(h2, fn, test), core.head_d(h0, fn, test))


@needs_cache
def test_dose_response_node_sets_are_nested_contain_ideals_and_p1_is_the_standard_graph(ctx_b):
    import ge_run_dose as gd_, ge_heads as gh
    sizes = {p: len(gh.make_graph(ctx_b, 10, keep=gd_.keep_fn(ctx_b, "all", p)).paths) for p in (0.0, 0.25, 0.5, 1.0)}
    assert sizes[0.0] < sizes[0.25] < sizes[0.5] < sizes[1.0] and sizes[1.0] == len(gh.make_graph(ctx_b, 10).paths)
    g0 = gh.make_graph(ctx_b, 10, keep=gd_.keep_fn(ctx_b, "out", 0.0)); pair = gd_.pair_group_images(ctx_b); held = {str(x) for g in list(ctx_b.validation) + list(ctx_b.test) for r in ctx_b.exp.groups[g].itertuples() for x in (r.resolved_img1, r.resolved_img2)}
    assert (pair - held) <= set(g0.paths) | {p for p in pair if "Trajectories" not in p}


@needs_cache
@pytest.mark.parametrize("kind", ["run", "temp"])
def test_harsh_folds_are_disjoint_from_training_and_graph(ctx_b, kind):
    import ge_harsh as hh, ge_heads as gh
    for i, (name, held) in enumerate(hh.folds_of(ctx_b, kind)):
        with tempfile.TemporaryDirectory() as scratch: ctx = core.make_ctx(8401, "A", scratch, core.feature_cache())
        test_groups, frames = hh.make_fold(ctx, held, i); gi = hh.group_images(ctx)
        assert test_groups and all(held(gi[g][0]) and held(gi[g][1]) for g in test_groups) and len(ctx.initial_groups) == 10
        train_imgs = {x for g in ctx.initial_groups + ctx.pool_groups for x in gi[g]}; assert not train_imgs & frames
        graph = gh.make_graph(ctx, 10, drop_paths=frames); assert not set(graph.paths) & frames and not {x for g in test_groups for x in gi[g]} & set(graph.paths)
