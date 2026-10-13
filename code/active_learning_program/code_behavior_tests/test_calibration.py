"""Tests of the calibration study: seed blocks, fold construction, calibrators (numpy only, synthetic data) and equivalence of the stage-1 head with the registry learner."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

NEW = Path(__file__).resolve().parents[2] / "active_learning_studies" / "new_directions"
sys.path.insert(0, str(NEW)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nd_core as core  # noqa: E402
import nd_calibration as C  # noqa: E402


def test_calibration_seed_blocks_are_new_and_registered():
    assert set(core.CAL_DEV_SEEDS) == set(range(9100, 9120)) and set(core.CAL_CONFIRM) == set(range(9150, 9185)) and core.CAL_SEEDS_ALL <= core.USED_SEEDS_BY_EARLIER_WORK
    assert not core.CAL_SEEDS_ALL & (core.SH_SEEDS_ALL | core.RT_SEEDS_ALL | core.GL_SEEDS_ALL | set(range(8000, 8200)) | set(range(8400, 8800)) | set(range(8885, 9000)))
    assert not set(core.CAL_DEV_SEEDS) & set(core.CAL_CONFIRM)


def synthetic_oof(n=4000, t0=2.0, seed=0):
    rng = np.random.default_rng(seed); d = rng.normal(0, 3, n); s = np.where(rng.random(n) < 1 / (1 + np.exp(-d / t0)), 1, -1)
    return {"d": d, "outcome": s, "weight": np.ones(n), "k": rng.choice([0, 2, 3, 4], n), "g_lab_min": rng.random(n) * 0.05, "g_anc_min": rng.random(n) * 0.05, "g_dens_max": rng.random(n) * 0.05}


def test_global_temperature_recovers_the_true_scale_and_does_not_touch_the_ranking():
    oof = synthetic_oof(t0=2.0); t = C.fit_global_t(oof); assert abs(t - 2.0) < 0.25
    rng = np.random.default_rng(1); d = rng.normal(0, 3, 4000); sign = np.where(rng.random(4000) < 1 / (1 + np.exp(-d / 2.0)), 1.0, -1.0)
    test = {"sign": sign, "weight": np.ones(4000), "k": np.zeros(4000, int), "g_anc_min": np.zeros(4000)}
    a, b = C.scores(d, test), C.scores(d / t, test); assert a["acc"] == b["acc"] and abs(a["auc"] - b["auc"]) < 1e-12 and b["ll"] < a["ll"]


def test_type_temperatures_shrink_towards_the_global_one_and_ties_do_not_enter_the_decisive_fit():
    oof = synthetic_oof(n=300); tg = C.fit_global_t(oof); strong = C.fit_type_t(oof, tau=1e6); assert all(abs(v - tg) < 1e-9 for v in strong.values())
    oof2 = dict(oof); oof2["outcome"] = oof["outcome"].copy(); oof2["outcome"][:50] = 0; m, d, s, w = C._dec(oof2); assert len(d) == len(oof["d"]) - 50 and set(np.unique(s)) <= {-1.0, 1.0}


def test_ece_of_a_calibrated_predictor_is_small_and_of_an_overconfident_one_is_large():
    rng = np.random.default_rng(3); n = 20000; d = rng.normal(0, 2, n); s = np.where(rng.random(n) < 1 / (1 + np.exp(-d)), 1.0, -1.0)
    test = {"sign": s, "weight": np.ones(n), "k": np.zeros(n, int), "g_anc_min": np.zeros(n)}
    good, bad = C.scores(d, test), C.scores(3 * d, test)
    assert C.ece_from_bins(good["bin_w"], good["bin_conf"], good["bin_acc"]) < 0.02 < C.ece_from_bins(bad["bin_w"], bad["bin_conf"], bad["bin_acc"])


def test_label_smoothing_zero_is_the_identity_and_the_swap_is_undone():
    import nd_graph_learner as G, torch.nn.functional as F
    a, b = torch.randn(7), torch.randn(7); m = {"1": torch.tensor([1, 0, 1, 0, 0, 0, 0.]), "2": torch.tensor([0, 1, 0, 0, 0, 0, 0.]), "tie": torch.tensor([0, 0, 0, 1, 0, 0, 0.]), "not_apply": torch.tensor([0, 0, 0, 0, 1, 1, 1.])}
    w = torch.ones(7); orig = G._bt_terms
    assert torch.allclose(C.smoothed_bt_terms(0.0)(a, b, m, w), orig(a, b, m, w))
    with C.label_smoothing(0.2): assert G._bt_terms is not orig and C.smoothed_bt_terms(0.2)(a, b, m, w) != orig(a, b, m, w)
    assert G._bt_terms is orig


pytestmark_cache = pytest.mark.skipif(not (core.single_study.CACHE.exists() and core.ALL_CACHE.exists()), reason="feature caches not built")


@pytest.fixture(scope="module")
def ctx_a():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as scratch: yield core.make_ctx(9100, "A", scratch, core.feature_cache())


@pytestmark_cache
def test_folds_never_split_a_pair_group_and_cover_every_labelled_judgment(ctx_a):
    lab = core.random_labelled(ctx_a, 40, 0); folds = C.folds_of(lab, 5, 1); assert sorted(j for f in folds for j in f) == sorted(set(lab))
    owner = {}
    for i, f in enumerate(folds):
        for j in f: assert owner.setdefault(j[0], i) == i


@pytestmark_cache
def test_stage1_head_equals_the_registry_learner_and_uses_no_test_label(ctx_a):
    import nd_graph_learner as G
    test = core.TestSet(ctx_a); lab = core.random_labelled(ctx_a, 10, 0); gd = G.graph_for(ctx_a, 10, False)
    d = C.logits(C.fit_head(ctx_a, gd, lab, 5), ctx_a, test); ref = G.LEARNERS["lap3_k10_st4"](ctx_a, lab, 5)(test); assert np.allclose(d, ref, atol=1e-6)
    held = {x for g in list(ctx_a.validation) + list(ctx_a.test) for r in ctx_a.exp.groups[g].itertuples() for x in (r.resolved_img1, r.resolved_img2)}
    assert not held & {x for j in lab for x in (ctx_a.info[j]["img1"], ctx_a.info[j]["img2"])}
