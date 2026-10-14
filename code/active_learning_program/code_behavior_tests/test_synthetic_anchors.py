"""Tests of the synthetic-anchor study: simulator geometry (rod spacing, Laue circle through the specular spot, lattice areas, determinism), seed block, bank/anchor plumbing, exact baseline equivalence,
permuted-type control, masked-ideal preprocessing, calibration identity rule."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

NEW = Path(__file__).resolve().parents[2] / "active_learning_studies" / "new_directions"
sys.path.insert(0, str(NEW)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nd_core as core  # noqa: E402
import rheed_sim as S  # noqa: E402

needs_cache = pytest.mark.skipif(not core.single_study.CACHE.exists() or not core.ALL_CACHE.exists(), reason="feature caches not built")


def test_seed_block_is_new_and_marked_used():
    assert core.SA_SEEDS_ALL <= core.USED_SEEDS_BY_EARLIER_WORK and len(core.SA_SEEDS_ALL) == 90
    assert min(core.SA_DEV_SEEDS) == 11100 and max(core.SA_DEV_SEEDS) == 11119 and min(core.SA_CONFIRM1) == 11130 and max(core.SA_CONFIRM1) == 11164 and min(core.SA_CONFIRM2) == 11165 and max(core.SA_CONFIRM2) == 11199
    assert not set(core.SA_DEV_SEEDS) & set(core.SA_CONFIRM1) and not set(core.SA_CONFIRM1) & set(core.SA_CONFIRM2)


def test_detector_geometry_matches_the_calibration_constants():
    assert abs(S.SPACING_PX - 65.0) < 1e-9 and abs(S.L_PX * 2 * np.pi / S.A_LATTICE / S.K0 - 65.0) < 1e-6        # fundamental rod spacing on the screen = 65 px
    assert abs(S.LAM - 0.0994) < 1e-3 and abs(S.K0 - 63.2) < 0.3                                                  # 15 keV electrons


def test_superlattice_areas_and_rod_counts():
    for typ, area in (("c6x2", 6), ("RT13", 13)):
        for Sm in S.LATTICES[typ]: assert abs(abs(np.linalg.det(np.array(Sm, float))) - area) < 1e-9
    rng = np.random.default_rng(0)
    r1, r2 = S.rods("1x1", rng, 0.0), S.rods("RT13", rng, 0.0, super_weight=0.3)
    assert len(r2) > 5 * len(r1) and all(w > 0 for _, _, w in r1)
    gys = sorted(round(gy / (2 * np.pi / S.A_LATTICE), 3) for gx, gy, w in r1 if abs(gx) < 1e-6); assert gys[:3] == [-4.0, -3.0, -2.0] or abs(np.diff(gys).min() - 1.0) < 1e-6     # integer-order rods only for 1x1


def test_render_is_deterministic_and_zero_laue_circle_passes_through_the_specular_spot():
    a1 = S.generate("1x1", 1, 5)[0][0]; a2 = S.generate("1x1", 1, 5)[0][0]; assert a1.shape == (492, 656) and a1.dtype == np.uint8 and (a1 == a2).all()
    rng = np.random.default_rng(1); p = S.sample_params(rng, "1x1"); p.update(gain=1.0, azimuth=0.0, dx=0.0, dy=0.0, noise_read=0.0, shot=0.0, kikuchi=0.0, bg=0.0, blur=0.8, rod_mod=0.0, spec=0.0, x0=300.0, yh=135.0)
    img = S.render(p, rng).astype(float); zs = S.L_PX * np.tan(p["theta"]); row = int(round(135.0 + zs)); band = img[row - 6: row + 7]
    cols = band.mean(0); peak = int(np.argmax(cols)); assert abs(peak - 300) < 6                                    # specular (0,0) rod sits at the beam centre on the zeroth Laue circle
    side = [int(300 + 65 * s) for s in (-1, 1)]; assert all(img[:, c - 4: c + 5].max() > 0.2 * img.max() for c in side)    # +-1 order rods are 65 px away


def test_shadow_edge_is_dark_above_the_horizon_in_trajectory_style():
    img = S.generate("c6x2", 1, 2, "traj")[0][0]; assert img[:100].mean() < 3.0


@needs_cache
def test_ideal_fixed_variants_have_the_expected_shape_and_only_16bit_files_change_in_fix16():
    import syn_ideal_fix
    from PIL import Image
    d = syn_ideal_fix.load(); cache = core.feature_cache(); assert d["fix16"].shape == (len(d["paths"]), 512) and d["mask"].shape == d["fix16"].shape
    is16 = np.array([Image.open(p).mode.startswith("I;16") for p in d["paths"]]); assert is16.sum() == 18
    keep = torch.stack([cache[p].float() for p in d["paths"]]); assert torch.allclose(d["fix16"][~torch.as_tensor(is16)], keep[~torch.as_tensor(is16)]) and not torch.allclose(d["fix16"][torch.as_tensor(is16)], keep[torch.as_tensor(is16)])


@pytest.fixture(scope="module")
def ctx_a():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as scratch: yield core.make_ctx(11100, "A", scratch, core.feature_cache())


@needs_cache
def test_baseline_specification_reproduces_the_incumbent_learner_exactly(ctx_a):
    import nd_graph_learner as G
    import nd_synthetic as Y
    test = core.TestSet(ctx_a); labelled = core.random_labelled(ctx_a, 10, 0)
    d0 = G.LEARNERS["lap3_k10_st4"](ctx_a, labelled, 3)(test); d1 = Y.LEARNERS["sy_base_check"](ctx_a, labelled, 3)(test); assert np.abs(d0 - d1).max() == 0.0
    d2 = Y.LEARNERS["sy_mask"](ctx_a, labelled, 3)(test); assert np.abs(d0 - d2).max() > 1e-3                      # a real change of the anchors changes the fit


@needs_cache
def test_weighted_anchor_loss_equals_the_plain_loss_for_unit_weights(ctx_a):
    import nd_graph_learner as G
    import nd_synthetic as Y
    torch.manual_seed(0); cols = {"a": 0, "b": 2, "c": 3}; sc = {k: torch.randn(n, 5) for k, n in (("a", 5), ("b", 7), ("c", 4))}
    assert abs(float(G._anchor_loss(sc, cols, None)) - float(Y._anchor_loss_w(sc, cols, {k: torch.ones(len(v)) for k, v in sc.items()}))) < 1e-6


def test_calibration_references_exclude_every_labelled_pair_group_image():
    import syn_calibrate
    src = Path(syn_calibrate.__file__).read_text(); assert "labelled_imgs" in src and "p not in labelled_imgs" in src


@needs_cache
def test_sixteen_bit_ideal_images_are_the_18_saturated_ones_and_drop16_removes_exactly_them():
    import nd_synthetic as Y
    from PIL import Image
    ps = Y.is16_paths(); assert len(ps) == 18 and all(Image.open(p).mode.startswith("I;16") for p in ps) and sum("RT13" in p for p in ps) == 8 and sum("HTR" in p for p in ps) == 10


@needs_cache
def test_type_mean_alignment_sets_the_type_means_to_the_real_masked_ideal_means_and_gaussian_control_keeps_moments():
    import syn_bank, syn_ideal_fix
    from PIL import Image
    if not (syn_bank.OUT / "bank_v1_typemean.pt").exists() or not (syn_bank.OUT / "bank_v1_tmgauss.pt").exists(): pytest.skip("aligned banks not built")
    tm = torch.load(syn_bank.OUT / "bank_v1_typemean.pt", weights_only=False); gs = torch.load(syn_bank.OUT / "bank_v1_tmgauss.pt", weights_only=False); fx = syn_ideal_fix.load(); idx = {p: i for i, p in enumerate(fx["paths"])}
    for t, key in zip(S.TYPES, ("1x1", "c6x2", "RT13", "HTR")):
        ps = [p for p in fx["paths"] if f"STO_ideal_{key}/" in p and not Image.open(p).mode.startswith("I;16")]; target = fx["mask"][[idx[p] for p in ps]].float().mean(0)
        m = torch.tensor([x == t for x in tm["typ"]]); assert torch.allclose(tm["feat"][m].mean(0), target, atol=1e-3)
        mg = torch.tensor([x == t for x in gs["typ"]]); assert torch.allclose(gs["feat"][mg].mean(0), target, atol=0.15 * float(target.abs().mean() + 1e-6) + 0.05)   # sampled mean close to the target
        assert abs(float(gs["feat"][mg].var(0).sum() / tm["feat"][m].var(0).sum()) - 1.0) < 0.2


@needs_cache
def test_gate_uses_only_image_features_and_marks_a_nonzero_share_far(ctx_a):
    import nd_synthetic as Y
    test = core.TestSet(ctx_a); dist = Y.anchor_distance_fn(ctx_a); far = np.minimum(dist(test.img1), dist(test.img2)) > 0.03; assert 0.02 < far.mean() < 0.8
