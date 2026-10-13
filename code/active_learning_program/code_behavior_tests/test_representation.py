"""Representation study: seed blocks, equivalence of the base representation with lap3_k10_st4, scale matching, identity safety of whitening / graphs built from other representations, capacity control, views."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

NEW = Path(__file__).resolve().parents[2] / "active_learning_studies" / "new_directions"
sys.path.insert(0, str(NEW)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nd_core as core  # noqa: E402

pytestmark = pytest.mark.skipif(not core.single_study.CACHE.exists() or not (NEW / "results" / "rep" / "simclr_views.pt").exists() or not (NEW / "results" / "feature_layers_all.pt").exists(), reason="representation caches not built")


@pytest.fixture(scope="module")
def ctx_a():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as scratch: yield core.make_ctx(8600, "A", scratch, core.feature_cache())


def test_seed_blocks_new_and_disjoint():
    assert core.RP_SEEDS_ALL <= core.USED_SEEDS_BY_EARLIER_WORK and not core.RP_SEEDS_ALL & set(range(7000, 7336)) and not core.RP_SEEDS_ALL & set(range(8000, 8600))
    assert min(core.RP_DEV_SEEDS) == 8600 and max(core.RP_DEV_SEEDS) == 8619 and min(core.RP_CONFIRM1) == 8650 and max(core.RP_CONFIRM1) == 8684


def test_base_representation_reproduces_lap3_k10_st4(ctx_a):
    import nd_graph_learner as G
    import nd_representation as R
    lab = core.random_labelled(ctx_a, 10, 0); test = core.TestSet(ctx_a)
    assert np.allclose(R.LEARNERS["rp_base"](ctx_a, lab, 3)(test), G.LEARNERS["lap3_k10_st4"](ctx_a, lab, 3)(test), atol=1e-6)


def test_blocks_are_scale_matched_and_have_expected_dimensions(ctx_a):
    import nd_representation as R
    s4 = R.raw_rep("s4"); paths = sorted(s4); ref = float(torch.stack([s4[p] for p in paths]).norm(dim=1).mean())
    for name, dim in (("s3", 256), ("s34", 768), ("i4", 512), ("is", 1024), ("mir", 512), ("mc", 512), ("cap", 1024)):
        t = R.raw_rep(name); x = torch.stack([t[p] for p in paths]); assert x.shape[1] == dim and abs(float(x.norm(dim=1).mean()) / ref - 1) < 0.1, name


def test_mirror_and_multi_view_are_means_of_views():
    import nd_rep_build as B
    import nd_representation as R
    views = torch.load(B.REP_DIR / "simclr_views.pt", weights_only=False); p = sorted(views)[3]; raw = R._blocks()
    assert torch.allclose(raw["mir"][p], (views[p][0] + views[p][1]) / 2) and torch.allclose(raw["mc"][p], views[p].mean(0)) and views[p].shape[0] == len(B.VIEWS)


def test_whitening_statistics_use_only_allowed_nodes_and_graph_nodes_exclude_heldout(ctx_a):
    import nd_graph_learner as G
    import nd_representation as R
    base = G.graph_for(ctx_a, 10); held = {str(x) for g in list(ctx_a.validation) + list(ctx_a.test) for r in ctx_a.exp.groups[g].itertuples() for x in (r.resolved_img1, r.resolved_img2)} | {str(p) for ps in ctx_a.exp.test_images.values() for p in ps}
    assert not held & set(base.paths)
    cg = R._proxy(ctx_a, "G:is", R.raw_rep("is")); gd = G.graph_for(cg, 10); assert gd.paths == base.paths and gd.x.shape[1] == 1024
    w = R.ctx_rep(ctx_a, "w64"); x = torch.stack([w[p] for p in base.paths]); assert x.shape[1] == 64
    assert float(x.mean(0).abs().max()) < 1e-3 * float(x.norm(dim=1).mean()) * 10   # centred on the allowed nodes


def test_capacity_control_is_s4_plus_a_fixed_random_projection():
    import nd_representation as R
    s4 = R.raw_rep("s4"); cap = R.raw_rep("cap"); p = sorted(s4)[0]; assert cap[p].shape[0] == 1024 and R.raw_rep("cap") is cap
    assert not torch.allclose(cap[p][:512], cap[p][512:]) and torch.allclose(torch.nn.functional.normalize(cap[p][:512], dim=0), torch.nn.functional.normalize(s4[p], dim=0), atol=1e-5)


def test_other_representations_train_and_ensemble_averages(ctx_a):
    import nd_representation as R
    lab = core.random_labelled(ctx_a, 10, 0); test = core.TestSet(ctx_a)
    for n in ("rp_s3", "rp_w64", "rp_gis_hs4", "rp_cap_shuf"): d = R.LEARNERS[n](ctx_a, lab, 3)(test); assert np.isfinite(d).all() and len(d) == len(test.k)
    d1 = R.LEARNERS["rp_base"](ctx_a, lab, 3)(test); d5 = R.rep_learner("s4", ensemble=2)(ctx_a, lab, 3)(test); assert not np.allclose(d1, d5)


def test_graph_free_representation_learner_equals_aw8_all_st2lr05(ctx_a):
    import nd_graph_learner as G
    import nd_representation as R
    lab = core.random_labelled(ctx_a, 10, 0); test = core.TestSet(ctx_a)
    assert np.allclose(R.LEARNERS["rp_gf_s4"](ctx_a, lab, 3)(test), G.LEARNERS["aw8_all_st2lr0.5"](ctx_a, lab, 3)(test), atol=1e-6)
    d = R.LEARNERS["rp_gs4_his_lr2"](ctx_a, lab, 3)(test); assert np.isfinite(d).all()
