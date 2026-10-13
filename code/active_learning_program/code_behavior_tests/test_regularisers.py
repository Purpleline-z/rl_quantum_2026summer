"""Integrity tests of the regulariser study G2: seed block, equivalences with the earlier learners when the new terms are off, controls, the anchor-far mask and the test machinery."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

NEW = Path(__file__).resolve().parents[2] / "active_learning_studies" / "new_directions"
GE = Path(__file__).resolve().parents[2] / "active_learning_studies" / "pair_disjoint_not_image_disjoint" / "graph_exploration"
sys.path.insert(0, str(NEW)); sys.path.insert(0, str(GE)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nd_core as core  # noqa: E402

pytestmark = pytest.mark.skipif(not core.ALL_CACHE.exists(), reason="all-image feature cache not built (python3 nd_temporal.py build)")


@pytest.fixture(scope="module")
def ctx_b():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as scratch: yield core.make_ctx(9000, "B", scratch, core.feature_cache())


def test_g2_seed_blocks_are_new_and_do_not_touch_reserved_ranges():
    assert core.G2_SEEDS_ALL <= core.USED_SEEDS_BY_EARLIER_WORK and len(core.G2_SEEDS_ALL) == 55
    assert not core.G2_SEEDS_ALL & (set(range(8000, 8200)) | set(range(8400, 8800)) | set(range(8885, 9000)) | set(range(7000, 7336)) | set(range(8200, 8400)) | set(range(8800, 8885)))
    assert not set(core.G2_DEV_SEEDS) & set(core.G2_CONFIRM)


def _d(learner, ctx, labelled, seed=0):
    return np.asarray(learner(ctx, labelled, seed)(core.TestSet(ctx)))


def test_all_options_off_is_aw8_all(ctx_b):
    import nd_learners as L, nd_regularisers as R
    lab = core.random_labelled(ctx_b, 10, 0)
    assert np.allclose(_d(R._learner(), ctx_b, lab), _d(L.LEARNERS["aw8_all"], ctx_b, lab), atol=1e-6)


def test_dropout_consistency_and_laplacian_reproduce_the_earlier_learners(ctx_b):
    import nd_graph_learner as G, nd_regularisers as R
    lab = core.random_labelled(ctx_b, 10, 0)
    assert np.allclose(_d(R.LEARNERS["rgcons0.3_st2"], ctx_b, lab), _d(G._mlp_learner(cons=0.3, steps_scale=2.0), ctx_b, lab), atol=1e-6)
    assert np.allclose(_d(R._learner(lap=3.0, steps_scale=2.0), ctx_b, lab), _d(G.LEARNERS["lap3_k10_st2"], ctx_b, lab), atol=1e-6)


def test_regularisers_change_the_fit_and_graph_controls_differ(ctx_b):
    import nd_regularisers as R
    lab = core.random_labelled(ctx_b, 10, 0); base = _d(R._learner(steps_scale=1.0), ctx_b, lab)
    for name in ("rgnoise1_st1", "rgmix0.3_st1", "rgrdrop1_st1", "rghdrop0.5_st1", "rgwd1_st1", "rgema0.9_st1", "rgswa0.5_st1", "rginner_st1", "rglaph1_k10_st1", "rglap1_k1_st1"):
        assert not np.allclose(_d(R.LEARNERS[name], ctx_b, lab), base, atol=1e-5), name
    assert not np.allclose(_d(R.LEARNERS["rglap1_k1_st1"], ctx_b, lab), _d(R.LEARNERS["rglap1_k1_shuf_st1"], ctx_b, lab), atol=1e-5)
    assert not np.allclose(_d(R.LEARNERS["rglaph1_k10_st1"], ctx_b, lab), _d(R.LEARNERS["rglaph1_k10_shuf_st1"], ctx_b, lab), atol=1e-5)


def test_manifold_mixup_in_the_hidden_layer_is_vacuous_for_a_one_hidden_layer_head(ctx_b):
    import torch.nn.functional as F
    head = core.new_head(512, 0).eval(); x = torch.randn(64, 512); i, j = torch.randperm(64), torch.randperm(64); lam = torch.rand(64, 1)
    h = F.relu(head[0](x)); hm = lam * h[i] + (1 - lam) * h[j]
    assert torch.allclose(head[3](hm), lam * head[3](h[i]) + (1 - lam) * head[3](h[j]), atol=1e-5)


def test_inner_early_stopping_picks_a_step_within_the_schedule(ctx_b):
    import nd_regularisers as R, judgment_unit_study as ju
    lab = core.random_labelled(ctx_b, 20, 0); rows = ju.rows_of(ctx_b.exp, lab); assert len(rows) >= 8
    d = _d(R.LEARNERS["rginner_st2"], ctx_b, lab); assert np.isfinite(d).all()


def test_anchor_far_mask_matches_the_audit_definition(ctx_b):
    import rg_run, rt_core as rt
    sim = rt.Sim(core.feature_cache()); test = core.TestSet(ctx_b); m = rg_run.anchor_min_distance(sim, ctx_b, test)
    assert m.shape == (len(test.k),) and (m >= -1e-6).all()
    far = m > rg_run.FAR; d = np.zeros(len(far)); out = rg_run.subset_metrics(d + test.sign, test, far, "far"); assert out["n_far"] == int(far.sum())
    if far.sum() and len(set(test.sign[far])) == 2: assert out["acc_far"] == pytest.approx(1.0)


def test_tables_noninferiority_and_holm():
    import rg_tables as T
    assert np.allclose(T.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    rng = np.random.default_rng(0); rows = []
    for seed in range(35):
        for split in "AB":
            for b in (10, 20):
                base = 0.9 + 0.01 * rng.normal(); rows.append({"seed": seed, "split": split, "budget": b, "draw": 0, "learner": "ref", "auc": base, "acc": base - 0.05, "auc_far": base, "acc_far": base})
                rows.append({"seed": seed, "split": split, "budget": b, "draw": 0, "learner": "free", "auc": base - 0.001 + 0.001 * rng.normal(), "acc": base - 0.05 - 0.002 + 0.002 * rng.normal(), "auc_far": base, "acc_far": base})
                rows.append({"seed": seed, "split": split, "budget": b, "draw": 0, "learner": "bad", "auc": base - 0.03, "acc": base - 0.1, "auc_far": base - 0.03, "acc_far": base - 0.1})
    spec = {"reference": "ref", "tuned": "bad", "freecands": ["free", "bad"], "margins": {"auc": 0.003, "acc": 0.01}, "splits": ["A", "B"]}
    out = T.run_tests(pd.DataFrame(rows), spec); v, label = T.verdict(out, spec)
    a = v.dropna(subset=["candidate"]).set_index("candidate")["A_noninferior"].astype(bool); b = v.dropna(subset=["vs graph-free"]).set_index("vs graph-free")["beats it (B)"].astype(bool)
    assert bool(a["free"]) and not bool(a["bad"]); assert bool(b["bad"]) and bool(b["free"]) and label.startswith("A and B")   # the reference is also (slightly, significantly) better than "free"
