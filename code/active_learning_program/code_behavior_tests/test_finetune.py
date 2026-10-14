"""Tests for the CPU fine-tuning study (nd_finetune.py, ft_shared.py): seed block, layer-4-on-cached-layer-3 reproduces the cached feature, augmentation, zero-step refinement equals the incumbent,
frozen-encoder control leaves layer 4 untouched, adaptation images contain no pair-group image or content twin."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

NEW = Path(__file__).resolve().parents[2] / "active_learning_studies" / "new_directions"
sys.path.insert(0, str(NEW)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nd_core as core  # noqa: E402

pytestmark = pytest.mark.skipif(not core.single_study.CACHE.exists() or not (NEW / "results" / "ft" / "l3_cache.pt").exists(), reason="feature / layer-3 caches not built (nd_finetune.py build)")


@pytest.fixture(scope="module")
def ctx_a():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as scratch: yield core.make_ctx(10300, "A", scratch, core.feature_cache())


def test_seed_blocks_are_new_and_disjoint():
    assert core.FT_SEEDS_ALL <= core.USED_SEEDS_BY_EARLIER_WORK and len(core.FT_SEEDS_ALL) == 90
    assert set(core.FT_DEV_SEEDS) == set(range(10300, 10320)) and set(core.FT_CONFIRM1) == set(range(10330, 10365)) and set(core.FT_CONFIRM2) == set(range(10365, 10400))
    assert not core.FT_SEEDS_ALL & (core.RC_SEEDS_ALL | core.CAL_SEEDS_ALL | core.TG_SEEDS_ALL | core.RT_SEEDS_ALL)


def test_layer4_on_cached_layer3_reproduces_the_cached_simclr_feature():
    import nd_finetune as FT
    cache = core.feature_cache(); paths = sorted(cache)[:20]; layer4 = FT.fresh_layer4()
    f = FT.encode_all(layer4, paths); ref = torch.stack([cache[p].float() for p in paths])
    assert float((f - ref).abs().max()) < 2e-3


def test_augmentation_shapes_identity_and_determinism():
    import nd_finetune as FT
    x = torch.rand(6, 256, 14, 14); g1, g2 = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
    y = FT.augment(x, g1); assert y.shape == x.shape and torch.allclose(y, FT.augment(x, g2))
    assert torch.allclose(FT.augment(x, torch.Generator().manual_seed(0), flip=False, shift=0, jitter=0.0), x)


def test_zero_step_refinement_equals_the_incumbent_and_frozen_control_keeps_layer4(ctx_a):
    import nd_finetune as FT
    import nd_graph_learner as G
    from dataclasses import replace
    labelled = core.random_labelled(ctx_a, 10, 0); test = core.TestSet(ctx_a)
    d0 = G.LEARNERS["lap3_k10_st4"](ctx_a, labelled, 5)(test); d1 = FT.refine_learner(replace(FT.BASE, steps=0))(ctx_a, labelled, 5)(test)
    assert np.allclose(d0, d1, atol=1e-6)
    layer4, head, info = FT.fit_refine(ctx_a, labelled, 5, replace(FT.BASE, enc_lr=0.0, steps=2))
    ref = FT.fresh_layer4()
    assert all(torch.equal(a, b) for a, b in zip(layer4.parameters(), ref.parameters())) and info["drift"] == 0.0
    layer4, head, info = FT.fit_refine(ctx_a, labelled, 5, replace(FT.BASE, enc_lr=1e-4, steps=2, sp=0.0))
    assert info["drift"] > 0.0


def test_parse_spec_names():
    import nd_finetune as FT
    assert FT.parse("ftp:enc_lr=0;aug=0;head_lr_mult=0.2;shuf=1") is not None
    with pytest.raises(AttributeError): FT.parse("ftp:nonsense=1")


def test_adaptation_images_exclude_every_pair_group_image_and_its_content_twins():
    import ft_shared as S
    import nd_graph_learner as G
    frames, ideal, bad = S.adaptation_images()
    with tempfile.TemporaryDirectory() as scratch: ctx = core.make_ctx(10300, "A", scratch, core.feature_cache())
    group_imgs = {str(x) for g in ctx.exp.groups.values() for r in g.itertuples() for x in (r.resolved_img1, r.resolved_img2)}
    assert not set(frames) & group_imgs and len(frames) > 500
    hashes = G.content_hashes(sorted(set(frames) | group_imgs)); assert not {hashes[p] for p in frames} & {hashes[p] for p in group_imgs}
    assert all("Trajectories" in p for p in frames)
