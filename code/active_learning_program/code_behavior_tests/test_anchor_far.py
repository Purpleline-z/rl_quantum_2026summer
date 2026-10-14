"""Integrity tests of the anchor-far learners: base equivalence with lap3_k10_st4, no use of test labels, shuffled controls differ, seed block."""
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


@pytest.fixture(scope="module")
def ctx_a():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as scratch:
        yield core.make_ctx(10800, "A", scratch, core.feature_cache())


def test_seed_block():
    assert set(core.AF_DEV_SEEDS) == set(range(10800, 10820)) and not (set(core.AF_CONFIRM1) & set(core.AF_CONFIRM2)) and core.AF_SEEDS_ALL <= core.USED_SEEDS_BY_EARLIER_WORK


def test_base_is_bit_identical_to_lap3_k10_st4(ctx_a):
    import rt_core as rt
    test = core.TestSet(ctx_a); lab = core.random_labelled(ctx_a, 20, 0)
    d0 = rt.predict(ctx_a, "lap3_k10_st4", lab, 5, test); d1 = rt.predict(ctx_a, "af_base", lab, 5, test)
    assert np.array_equal(d0, d1)


def test_variants_run_change_the_fit_and_ignore_test_labels(ctx_a):
    import rt_core as rt
    test = core.TestSet(ctx_a); lab = core.random_labelled(ctx_a, 20, 0); base = rt.predict(ctx_a, "af_base", lab, 5, test)
    for name in ("af_mixaa0.3", "af_pseudo1h2", "af_nbra3", "af_dlamfar2", "af_far1", "af_ictnn0.1"):
        d = rt.predict(ctx_a, name, lab, 5, test); assert np.isfinite(d).all() and not np.allclose(d, base, atol=1e-6), name
    flipped = core.TestSet(ctx_a); flipped.sign = -flipped.sign; flipped.weight = flipped.weight * 0 + 1
    assert np.array_equal(rt.predict(ctx_a, "af_pseudo1h2", lab, 5, test), rt.predict(ctx_a, "af_pseudo1h2", lab, 5, flipped))


def test_shuffled_controls_differ_and_hops_are_computed(ctx_a):
    import af_learner as af, nd_graph_learner as G, rt_core as rt
    test = core.TestSet(ctx_a); lab = core.random_labelled(ctx_a, 20, 0)
    assert not np.allclose(rt.predict(ctx_a, "af_pseudo1h2", lab, 5, test), rt.predict(ctx_a, "af_pseudo1h2_shuf", lab, 5, test), atol=1e-6)
    gd = G.graph_for(ctx_a, 10, False); geo = af.geo_for(ctx_a, gd)
    assert geo.hop[geo.src].max() == 0 and (geo.hop < 1e8).mean() > 0.5 and geo.d_anchor.min() >= -1e-6
