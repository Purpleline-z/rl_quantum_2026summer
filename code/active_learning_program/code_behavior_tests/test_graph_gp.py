"""Tests of the graph-kernel GP learner: deformed kernel algebra, Laplace solver, identity safety, symmetry, label independence, seed blocks."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

NEW = Path(__file__).resolve().parents[2] / "active_learning_studies" / "new_directions"
sys.path.insert(0, str(NEW)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nd_core as core  # noqa: E402
import nd_gpg as P  # noqa: E402

DT = torch.float64


def test_graph_gp_seed_blocks_are_new_and_registered():
    assert set(core.GP_DEV_SEEDS) == set(range(11000, 11020)) and set(core.GP_CONFIRM1) == set(range(11030, 11065)) and set(core.GP_CONFIRM2) == set(range(11065, 11100)) and core.GP_SEEDS_ALL <= core.USED_SEEDS_BY_EARLIER_WORK
    assert min(core.GP_SEEDS_ALL) >= 11000 and not core.GP_SEEDS_ALL & (core.RT_SEEDS_ALL | core.CAL_SEEDS_ALL | core.SH_SEEDS_ALL | core.GL_SEEDS_ALL)


def toy(n=40, seed=0):
    g = torch.Generator().manual_seed(seed); x = torch.randn(n, 5, generator=g, dtype=DT); x = x / x.norm(dim=1, keepdim=True); d2 = (2 - 2 * x @ x.T).clamp_min(0); K = torch.exp(-d2 / (2 * 0.7 ** 2)) + 1e-6 * torch.eye(n, dtype=DT)
    A = (d2 < d2.kthvalue(5, dim=1).values[:, None]).to(DT); A = ((A + A.T) > 0).to(DT); A.fill_diagonal_(0); L = torch.diag(A.sum(1)) - A
    return x, K, L


def test_deformed_kernel_equals_the_inverse_of_prior_precision_plus_laplacian():
    x, K, L = toy(); beta = 0.7; B = torch.linalg.solve(torch.eye(len(K), dtype=DT) + beta * L @ K, beta * L); Kt = K - K @ B @ K
    assert torch.allclose(Kt, torch.linalg.inv(torch.linalg.inv(K) + beta * L), atol=1e-6) and torch.allclose(B, B.T, atol=1e-10)
    ev = torch.linalg.eigvalsh(0.5 * (Kt + Kt.T)); assert ev.min() > -1e-9 and (torch.diagonal(Kt) <= torch.diagonal(K) + 1e-9).all()      # deformation only shrinks the prior variance


def test_laplace_newton_reaches_a_stationary_point_with_a_finite_evidence():
    x, K, L = toy(); m = 12; idx = torch.arange(m); A = 2.0 * K[:m, :m]
    i = torch.tensor([0, 2, 4, 6, 8]); j = torch.tensor([1, 3, 5, 7, 9]); code = torch.tensor([0, 0, 1, 2, 0]); w = torch.ones(5, dtype=DT)
    terms = P.Terms(i, j, code, w, [(torch.tensor([0, 1, 2]), torch.tensor([9, 10, 11]), 0.5)], torch.tensor([11]), 0.1, m)
    alpha, f, H, J, logev = P.solve_laplace(A, terms)
    val, g, _ = terms.value_grad_hess(f); assert torch.linalg.norm(alpha + g) < 1e-5 and np.isfinite(logev) and torch.linalg.eigvalsh(H).min() > -1e-9      # gradient of 1/2 a^T A a + data(A a) w.r.t. alpha = A (alpha + g) = 0
    # the Hessian used by Newton is the exact one: compare with autograd
    f0 = (A @ alpha).detach().clone().requires_grad_(True)
    def data(ff): return terms.value_grad_hess(ff, want_hess=False)[0]
    Hn = torch.autograd.functional.hessian(lambda ff: (lambda u: (terms.w * torch.where(terms.code == 2, torch.sqrt(u ** 2 + P.EPS_TIE), torch.nn.functional.softplus(torch.where(terms.code == 0, -u, u)))).sum())(terms.C @ ff) + sum(0.5 * torch.nn.functional.softplus(-(ff[P_][:, None] - ff[O_][None, :])).sum() for P_, O_, _w in terms.anchors) + 0.1 * torch.nn.functional.softplus(ff[terms.bad_idx] + 1.0).sum(), f0)
    assert torch.allclose(H, Hn, atol=1e-6)


@pytest.fixture(scope="module")
def ctx_a():
    if not (core.single_study.CACHE.exists() and core.ALL_CACHE.exists()): pytest.skip("feature caches not built")
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as scratch: yield core.make_ctx(11000, "A", scratch, core.feature_cache())


def test_nodes_exclude_held_out_images_and_predictions_are_antisymmetric_and_label_independent(ctx_a):
    geo = P.geometry_for(ctx_a); test = core.TestSet(ctx_a); held = {str(x) for g in list(ctx_a.validation) + list(ctx_a.test) for r in ctx_a.exp.groups[g].itertuples() for x in (r.resolved_img1, r.resolved_img2)}
    assert not held & set(geo.gd.paths)
    lab = core.random_labelled(ctx_a, 20, 0); model = P.fit_gpg(ctx_a, lab, 1, dict(select="fixed", amp={0: .5, 2: 1., 3: 3., 4: 3.}), geo=geo); a = model.predict(test)
    class Swapped:                                   # same judgments with the two images exchanged
        img1, img2, k = test.img2, test.img1, test.k
    b = model.predict(Swapped); assert np.allclose(a["mu"], -b["mu"], atol=1e-8) and np.allclose(a["var"], b["var"], atol=1e-8)
    test2 = core.TestSet(ctx_a); test2.sign = -test2.sign; test2.weight = test2.weight * 3; c = model.predict(test2); assert np.array_equal(a["mu"], c["mu"])    # prediction never reads test labels
    assert np.isfinite(a["mu"]).all() and (a["var"] >= 0).all() and not ({x for x in test.img1} & {ctx_a.info[j]["img1"] for j in lab})


def test_shuffled_graph_keeps_the_degree_sequence_and_oof_never_trains_on_its_own_group(ctx_a):
    g1, g2 = P.Geometry(ctx_a, 10, False), P.Geometry(ctx_a, 10, True); assert sorted(g1.deg.tolist()) == sorted(g2.deg.tolist()) and not torch.equal(g1.L, g2.L)
    lab = core.random_labelled(ctx_a, 40, 0); model = P.fit_gpg(ctx_a, lab, 3, dict(select="fixed", amp={0: .5, 2: 1., 3: 3., 4: 3.}), geo=g1); oof = P.gp_oof(model, 3, 5)
    assert np.isfinite(oof["d"][np.isin(oof["outcome"], (1, -1))]).all() and len(set(oof["fold"])) == 5
    rows = model.obs.rows; groups = rows.pair_id.to_numpy() if "pair_id" in rows else None
    if groups is not None:
        for f in range(5): assert not (set(groups[oof["fold"] == f]) & set(groups[oof["fold"] != f]))
