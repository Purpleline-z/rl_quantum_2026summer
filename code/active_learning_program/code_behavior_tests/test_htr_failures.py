"""Integrity tests of the HTR failure-analysis tooling: seed blocks, label tables, similarity features, HTR-only variants, paired-gain statistics."""
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

pytestmark = pytest.mark.skipif(not core.single_study.CACHE.exists(), reason="feature cache not built (build_feature_cache.py)")


def test_seed_blocks_are_fresh_and_disjoint():
    assert set(core.HF_DEV_SEEDS) == set(range(10400, 10420)) and set(core.HF_CONFIRM1) == set(range(10430, 10465)) and set(core.HF_CONFIRM2) == set(range(10465, 10500))
    assert not set(core.HF_DEV_SEEDS) & set(core.HF_CONFIRM1) and not set(core.HF_CONFIRM1) & set(core.HF_CONFIRM2) and not set(core.HF_DEV_SEEDS) & set(core.HF_CONFIRM2)
    assert not core.HF_SEEDS_ALL & (core.RT_SEEDS_ALL | core.G2_SEEDS_ALL | core.RC_SEEDS_ALL | core.TG_SEEDS_ALL | core.CAL_SEEDS_ALL | core.AE_SEEDS_ALL)


def test_static_label_table_matches_the_raw_csv():
    import hf_table as T
    j, img = T.build(); raw = pd.read_csv(core.single_study.DATA / "original data" / "Quantum Label Data - Pairwise_Comparisonv1.8.csv")
    raw = raw[raw.Reconstruction_Type == "HTR"]; assert len(j) == 521 and (j.type == "HTR").sum() == len(raw) and ((j.type == "HTR") & j.decisive).sum() == raw.Winner.isin(["1", "2"]).sum() == 45
    assert j.groupby(["pair", "type"]).size().max() == 1                                  # no (pair, type) judged twice
    assert np.all(np.isin(j.sign, [-1.0, 0.0, 1.0])) and np.all(j.loc[j.decisive, "sign"] != 0) and set(img.nn_class) <= set(T.ACTIVE_CLASSES)


def test_similarity_excludes_the_identical_image_and_blend_only_touches_htr():
    import hf_learners as H, nd_graph_learner as G
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as scratch:
        ctx = core.make_ctx(core.HF_DEV_SEEDS[0], "A", scratch, core.feature_cache()); anchors = H.anchor_paths(ctx, "HTR"); assert len(anchors) >= 20
        s_self = H.similarity(ctx, anchors[:3], "HTR", 1)
        xa = ctx.features.get(anchors).float() - G.graph_for(ctx, 10, False).mu; xa = xa / xa.norm(dim=1, keepdim=True); full = xa[:3] @ xa.T
        assert torch.allclose(full.diagonal()[:3] if full.shape[0] == full.shape[1] else full[torch.arange(3), torch.arange(3)], torch.ones(3), atol=1e-5)   # self-similarity is 1 ...
        assert torch.allclose(s_self, full.topk(2, dim=1).values[:, 1], atol=1e-5)                                                                               # ... and is excluded: the top-1 is the runner-up
        test = core.TestSet(ctx); base = lambda t: np.linspace(-1, 1, len(t.k)); fake = {"x": base}
        G.LEARNERS["_dummy"] = lambda c, l, s: base
        try:
            fn = H.blend_learner(10.0, base="_dummy")(ctx, list(ctx.initial), 0); d0, d1 = base(test), fn(test); changed = ~np.isclose(d0, d1)
            assert not changed[test.k != 4].any() and changed[test.k == 4].any()                           # only the HTR rows move
        finally: G.LEARNERS.pop("_dummy")


def test_strict_context_removes_near_copy_anchors():
    import hf_a1, rt_core as rt
    torch.set_num_threads(1); cache = core.feature_cache(); sim = rt.Sim(cache); make = core.make_ctx
    try:
        hf_a1.install_strict(0.01)
        with tempfile.TemporaryDirectory() as scratch:
            ctx = core.make_ctx(core.HF_DEV_SEEDS[0], "A", scratch, cache); held = sorted(rt.heldout_images(ctx))
            for tag in ("references", "utility_images", "test_images"):
                for cls, ps in getattr(ctx.exp, tag).items():
                    if ps: assert (1 - sim.max_sim([str(p) for p in ps], held) > 0.01).all()
    finally: core.make_ctx = make


def test_holm_and_bootstrap_helpers():
    import hf_gain as g
    assert np.allclose(g.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    mean, lo, hi, p, pf = g.boot_mean(np.full(20, 0.05)); assert mean == pytest.approx(0.05) and lo > 0 and p < 0.01 and pf < 0.01
    mean, lo, hi, p, pf = g.boot_mean(np.array([0.1, -0.1] * 10)); assert abs(mean) < 1e-12 and lo < 0 < hi and p > 0.5


def test_gain_script_signs():
    import hf_gain as g
    rows = []
    for seed in range(5):
        for learner, shift in (("base", 0.0), ("better", 2.0)):
            for budget in (10, 20):
                for j in range(12):
                    k = (4, 3, 0, 2)[j % 4]; sign = 1.0 if j % 2 else -1.0; d = sign * (1.0 + shift) if j != 0 else -sign * 0.5
                    rows.append({"seed": seed, "split": "A", "budget": budget, "draw": 0, "learner": learner, "j": j, "d": d, "sign": sign, "weight": 1.0, "k": k})
    d = pd.DataFrame(rows); d["margin"] = d.sign * d.d; d["correct"] = (d.margin > 0).astype(float); d["ll"] = np.logaddexp(0.0, -d.margin)
    r = g.compare(d, "base", ["better"]); assert r.ll_HTR.iloc[0] > 0 and abs(r.acc_RT13.iloc[0]) < 1e-12 and r.acc_HTR.iloc[0] == pytest.approx(0.0)
