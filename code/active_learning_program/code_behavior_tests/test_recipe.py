"""Tests for the recipe-search learners (nd_recipe.py): the default spec reproduces lap3_k10_st4, every variant runs and is finite, seed block is new, anchor-far definition."""
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
    with tempfile.TemporaryDirectory() as scratch: yield core.make_ctx(9300, "B", scratch, core.feature_cache())


def test_recipe_seed_blocks_are_new_and_disjoint_from_reserved_ranges():
    assert core.RC_SEEDS_ALL <= core.USED_SEEDS_BY_EARLIER_WORK and len(core.RC_SEEDS_ALL) == 55
    assert not core.RC_SEEDS_ALL & (core.RT_SEEDS_ALL | core.SH_SEEDS_ALL | core.GL_SEEDS_ALL)
    assert not set(core.RC_DEV_SEEDS) & set(core.RC_CONFIRM)


def test_default_spec_reproduces_lap3_k10_st4(ctx_b):
    import nd_graph_learner as G
    import nd_recipe as R
    labelled = core.random_labelled(ctx_b, 20, 0); test = core.TestSet(ctx_b)
    d0 = G.LEARNERS["lap3_k10_st4"](ctx_b, labelled, 5)(test); d1 = R.LEARNERS["rc_base"](ctx_b, labelled, 5)(test)
    assert np.allclose(d0, d1, atol=1e-6)


@pytest.mark.parametrize("name", ["rc_fullg", "rc_hinge", "rc_w512", "rc_d2", "rc_do0.3", "rc_rk", "rc_softtie", "rc_dn0.3", "rc_mix", "rc_ptlr", "rc_cos", "rc_noclip", "rc_b99", "rc_wd1", "rc_aw16"])
def test_every_variant_runs_and_differs_from_the_incumbent(ctx_b, name):
    import nd_recipe as R
    labelled = core.random_labelled(ctx_b, 10, 0); test = core.TestSet(ctx_b)
    # shortened schedule to keep the test fast
    spec = R.VARIANTS[name]; from dataclasses import replace
    fast = R.recipe_learner(replace(spec, steps_scale=0.5)); base = R.recipe_learner(replace(R.BASE, steps_scale=0.5))
    d, d0 = np.asarray(fast(ctx_b, labelled, 3)(test)), np.asarray(base(ctx_b, labelled, 3)(test))
    assert np.isfinite(d).all() and not np.allclose(d, d0)


def test_anchor_far_distance_matches_the_red_team_definition(ctx_b):
    import rc_run
    import rt_core as rt
    test = core.TestSet(ctx_b); sim = rt.Sim(core.feature_cache()); anc = rc_run.anchor_distance(ctx_b, test, sim)
    assert anc.shape == (len(test.k),) and (anc >= -1e-6).all() and (anc <= 2).all() and 0 < (anc > rc_run.FAR).mean() < 1


def test_confidence_weights_are_almost_all_one_so_the_power_knob_is_nearly_inert(ctx_b):
    import judgment_unit_study as ju
    rows = ju.rows_of(ctx_b.exp, list(ctx_b.initial) + list(ctx_b.pool)); w = rows.confidence_weight.round(3); assert set(w.unique()) <= {0.7, 1.0} and (w == 1.0).mean() > 0.95
