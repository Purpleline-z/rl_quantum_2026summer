"""Score-propagation study: free scores, kernel prediction, hybrids, distillation hook, identity safety, seed block."""
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
def ctx_a():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as scratch: yield core.make_ctx(11200, "A", scratch, core.feature_cache())


def test_seed_blocks_are_new_and_disjoint():
    assert core.SP_SEEDS_ALL <= core.USED_SEEDS_BY_EARLIER_WORK and len(core.SP_SEEDS_ALL) == 90
    assert not set(core.SP_DEV_SEEDS) & (set(core.SP_CONFIRM1) | set(core.SP_CONFIRM2)) and not set(core.SP_CONFIRM1) & set(core.SP_CONFIRM2)
    assert not core.SP_SEEDS_ALL & (core.RC_SEEDS_ALL | core.TG_SEEDS_ALL | core.CAL_SEEDS_ALL | core.RT_SEEDS_ALL)


def test_parse_name_and_defaults():
    import nd_propagate as P
    p = P.parse_name("fs_lam30_aw8_rho0.01_kt5_tau0.05_sh1"); assert p["kind"] == "fs" and p["lam"] == 30 and p["rho"] == 0.01 and p["kt"] == 5 and p["sh"] == 1
    assert P.parse_name("hy_lamr10_rhor0.1_penS")["penS"] is True and P.parse_name("ls")["kind"] == "ls"


def test_test_images_are_never_nodes_and_extras_are_not_heldout(ctx_a):
    import nd_graph_learner as G
    import nd_propagate as P
    pd_ = P.prop_for(ctx_a, 10); test = core.TestSet(ctx_a); held = {str(x) for x in list(test.img1) + list(test.img2)}
    assert not held & set(pd_.index) and pd_.M == pd_.N + len(pd_.extra_paths) and pd_.M > pd_.N
    held_all = {str(x) for g in list(ctx_a.validation) + list(ctx_a.test) for r in ctx_a.exp.groups[g].itertuples() for x in (r.resolved_img1, r.resolved_img2)}
    held_h = set(G.content_hashes(sorted(held_all)).values()); lab = {str(ctx_a.info[j][k]) for j in list(ctx_a.initial) + list(ctx_a.pool) for k in ('img1', 'img2')}
    hx = G.content_hashes(sorted(pd_.extra_paths)); assert not [p for p in pd_.extra_paths if hx[p] in held_h and p not in lab] and not (held_all & set(pd_.extra_paths))
    ids = G.content_hashes(sorted(pd_.extra_paths)); heldh = set(G.content_hashes(sorted(held_all)).values())
    twins = [p for p in pd_.extra_paths if ids[p] in heldh]; assert all(p in {str(ctx_a.info[j][k]) for j in list(ctx_a.initial) + list(ctx_a.pool) for k in ("img1", "img2")} for p in twins)   # only labelled content twins may appear


def test_lbfgs_reaches_a_stationary_point_and_penalty_smooths(ctx_a):
    import nd_propagate as P
    pd_ = P.prop_for(ctx_a, 10); rows = P.rows_of(pd_, core.random_labelled(ctx_a, 20, 0))
    X = P.solve(pd_, rows, 8.0, 0.1, 30.0, 0.01).requires_grad_(True); loss = P.objective(X, None, pd_, rows, 8.0, 0.1, 30.0, 0.01); loss.backward()
    assert float(X.grad.abs().max()) < 1e-5
    rough = lambda Z: float(((Z[pd_.edges[:, 0]] - Z[pd_.edges[:, 1]]) ** 2).sum(1).mean())
    assert rough(P.solve(pd_, rows, 8.0, 0.1, 300.0, 0.01)) < rough(P.solve(pd_, rows, 8.0, 0.1, 3.0, 0.01))


def test_kernel_weights_and_nearest_neighbour_limit(ctx_a):
    import nd_propagate as P
    pd_ = P.prop_for(ctx_a, 10); test = core.TestSet(ctx_a); s1, _ = pd_.test_sims(test)
    w = P.kernel_weights(s1, 5, 0.05); assert torch.allclose(w.sum(1), torch.ones(len(w)), atol=1e-5) and ((w > 0).sum(1) <= 5).all()
    w1 = P.kernel_weights(s1, 1, 0.05); assert (w1.argmax(1) == s1.argmax(1)).all()


def test_shuffled_graph_keeps_edge_count_and_label_spreading_is_a_distribution(ctx_a):
    import nd_propagate as P
    a, b = P.prop_for(ctx_a, 10, False), P.prop_for(ctx_a, 10, True); assert a.edges.shape == b.edges.shape
    import nd_graph_learner as G
    q = G.graph_for(ctx_a, 10).q(); assert torch.allclose(q.sum(1), torch.ones(len(q)), atol=1e-4)


def test_cached_head_equals_registry_learner_and_hybrid_with_huge_ridge_is_the_head(ctx_a):
    import nd_graph_learner as G
    import nd_recipe as R
    import sp_run
    test = core.TestSet(ctx_a); lab = core.random_labelled(ctx_a, 20, 0); cell = sp_run.Cell(ctx_a, lab, 5, test); reg = {**G.LEARNERS, **R.LEARNERS}
    d0 = sp_run.predict(cell, "lap3_k10_st4", reg); assert np.allclose(d0, G.LEARNERS["lap3_k10_st4"](ctx_a, lab, 5)(test), atol=1e-6)
    d1 = sp_run.predict(cell, "hy_lamr10_rhor1000000", reg); assert np.allclose(d0, d1, atol=2e-2)
    d2 = sp_run.predict(cell, "ds_mu0.0_lam30", reg); assert np.allclose(d0, d2, atol=1e-6)


def test_predictions_do_not_depend_on_test_labels(ctx_a):
    import nd_propagate as P
    test = core.TestSet(ctx_a); lab = core.random_labelled(ctx_a, 20, 0); f = P.fit_variant(ctx_a, lab, "fs_lam30"); d = f(test); test.sign = -test.sign
    assert np.allclose(d, f(test))
